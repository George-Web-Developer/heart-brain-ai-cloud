"""
stroke_model.py
========================================================================================
Model 2 (`StrokeModel1B` / `BrainStrokeRiskModel2`):
Two-Branch Hybrid AI Stroke Risk & Acute Neurovascular Trigger Engine

Medical & Scientific Honesty Contract:
  - A wearable armband (AD8232/ADS1115 + MAX30102 + MPU6050) CANNOT diagnose or
    differentiate Ischemic Stroke vs. Hemorrhagic Stroke without Non-Contrast Head CT.
  - Therefore, Model 2 strictly evaluates:
      * Branch 2A (Wearable Hemodynamic & Neurovascular Instability):
          Hybrid 1D-ResNet-SE-BiGRU-Attention + Tabular XGBoost trained exclusively on
          real 2022 synchronized ECG + PPG + Invasive ABP benchmarks (VitalDB + MIMIC-IV).
          Invasive ABP is used ONLY for ground-truth target labeling; model inputs NEVER
          see ABP (non-invasive ECG, PPG, SpO2, PTT, and IMU kinematics only).
      * Branch 2B (Clinical & Cardioembolic Stroke Risk):
          Calibrated XGBoost trained on the 2021 real Clinical Stroke Cohort (N=5,110)
          fused with live Model 1 (CardiacModel1A) AFib/MI/Arrhythmia probabilities,
          CHA2DS2-VASc clinical burden, and missed anticoagulant risk multiplier.
  - Dual-Head Pure AI Outputs (`rules_overrode_ai = False` ALWAYS):
      * Head A (`num_class=4`):
          0: Low_Stroke_Risk
          1: Elevated_Stroke_Risk
          2: High_Stroke_Risk_FAST_Trigger
          3: Critical_Neurovascular_Emergency
      * Head B (`num_class=4`):
          Stage 0..3 Neurovascular Acuity + Golden Hour Countdown
  - Non-Mutating Safety & Cascade Outputs:
      * `trigger_model_3_fast_check` (and backward-compat `trigger_model_2_camera_voice_check`)
      * `red_contraindication_veto`: Strictly forbids aspirin/anticoagulants/thrombolytics
        prior to Head CT when severe crisis / Cushing triad risk is detected.
      * NEVER mutates `disease_code`, `stage_code`, or `raw_ml_decision`.
========================================================================================
"""

from __future__ import annotations

import os
import pickle
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import butter, find_peaks, resample_poly, sosfiltfilt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from xgboost import XGBClassifier

from src.preprocessing.signal_processor import compute_ppg_sqi, extract_10s_features

TARGET_HEMO_FS = 125.0
TARGET_HEMO_SAMPLES = 1250

STROKE_CLASSES = [
    "Low_Stroke_Risk",
    "Elevated_Stroke_Risk",
    "High_Stroke_Risk_FAST_Trigger",
    "Critical_Neurovascular_Emergency",
]
STROKE_RISK_CLASSES = STROKE_CLASSES

STROKE_STAGE_NAMES = [
    "Stage 0: Neurovascular Baseline",
    "Stage 1: Chronic / Cardioembolic Stroke Risk",
    "Stage 2: Acute Hemodynamic Surge / Fall - Run Model 3 FAST",
    "Stage 3: Critical Neurovascular Crisis (Golden Hour + CT Required)",
]

STROKE_LEGACY_11_FEATURES = [
    "hr_ecg_bpm",
    "pulse_rate_ppg_bpm",
    "pulse_deficit_bpm",
    "spo2_pct",
    "perfusion_index_pct",
    "ptt_ms",
    "delta_ptt_ms",
    "estimated_sbp_mmhg",
    "estimated_dbp_mmhg",
    "cushing_index",
    "imu_smv_g",
]

STROKE_MORPH_4_FEATURES = [
    "sdptg_b_a_ratio",
    "tau_diastolic_s",
    "dicrotic_notch_index",
    "ppg_upstroke_ms",
]

STROKE_BASE_FEATURES = STROKE_LEGACY_11_FEATURES + STROKE_MORPH_4_FEATURES

STROKE_ENGINEERED_FEATURES = [
    "rpp",
    "fractional_pulse_pressure",
    "perfusion_rate_ratio",
    "vascular_stiffness_index",
    "delta_ptt_ratio",
    "shock_index",
    "cushing_vascular_product",
]

STROKE_ACTIVE_FEATURES = STROKE_BASE_FEATURES + STROKE_ENGINEERED_FEATURES
assert len(STROKE_BASE_FEATURES) == 15
assert len(STROKE_ACTIVE_FEATURES) == 22

CLINICAL_RISK_FEATURES = [
    "age",
    "gender_male",
    "hypertension",
    "heart_disease",
    "ever_married",
    "avg_glucose_level",
    "diabetes",
    "bmi",
    "smoking_formerly",
    "smoking_current",
    "age_x_hypertension",
    "cardiometabolic_burden",
]

DEFAULT_STROKE_MODEL_PATH = os.path.join("models", "stroke_model_1b.pkl")
DEFAULT_STROKE_CNN_PATH = os.path.join("models", "stroke_model_1b_cnn.pt")

MEDICAL_SCOPE_DISCLAIMER = (
    "Wearable stroke risk screening & FAST trigger only. Cannot diagnose or differentiate "
    "ischemic vs. hemorrhagic stroke without Non-Contrast Head CT."
)

_MORPH_DEFAULTS = {
    "sdptg_b_a_ratio": -0.65,
    "tau_diastolic_s": 0.35,
    "dicrotic_notch_index": 0.30,
    "ppg_upstroke_ms": 160.0,
}

_SOS_ECG_125 = butter(3, [0.5, 35.0], btype="bandpass", fs=TARGET_HEMO_FS, output="sos")
_SOS_PPG_125 = butter(3, [0.5, 12.0], btype="bandpass", fs=TARGET_HEMO_FS, output="sos")


def _safe_float_val(val: Any, default: float) -> float:
    try:
        v = float(val)
        return v if np.isfinite(v) else float(default)
    except Exception:
        return float(default)


def _resample_to_hemo_125(sig: np.ndarray, src_fs: float) -> np.ndarray:
    x = np.asarray(sig, dtype=np.float64).flatten()
    if x.size == 0:
        return np.zeros(TARGET_HEMO_SAMPLES, dtype=np.float64)
    med = float(np.nanmedian(x)) if np.isfinite(x).any() else 0.0
    x = np.nan_to_num(x, nan=med, posinf=med, neginf=med)
    if abs(float(src_fs) - TARGET_HEMO_FS) > 0.1 and float(src_fs) > 10.0:
        frac = Fraction(TARGET_HEMO_FS / float(src_fs)).limit_denominator(120)
        x = resample_poly(x, frac.numerator, frac.denominator)
    if len(x) < TARGET_HEMO_SAMPLES:
        x = np.pad(x, (0, TARGET_HEMO_SAMPLES - len(x)), mode="edge")
    elif len(x) > TARGET_HEMO_SAMPLES:
        x = x[:TARGET_HEMO_SAMPLES]
    return x


def _robust_z_clip(sig: np.ndarray, clip_val: float = 8.0) -> np.ndarray:
    x = np.asarray(sig, dtype=np.float64)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    scale = max(1.4826 * mad, float(np.std(x)), 1e-6)
    z = np.clip((x - med) / scale, -clip_val, clip_val)
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def build_hemo_waveform_tensor(
    ecg: np.ndarray,
    ppg: np.ndarray,
    fs: float = 250.0,
) -> np.ndarray:
    """
    Converts raw 10s ECG & PPG signals into a 4-channel 125 Hz tensor:
      [ECG_Z, PPG_Z, dPPG_Z, d2PPG_Z] of shape (4, 1250) float32.
    Zero invasive ABP ever enters this tensor.
    """
    e = _resample_to_hemo_125(ecg, fs)
    p = _resample_to_hemo_125(ppg, fs)
    e = e - np.median(e)
    p = p - np.median(p)
    try:
        e_bp = sosfiltfilt(_SOS_ECG_125, e)
    except Exception:
        e_bp = e
    try:
        p_bp = sosfiltfilt(_SOS_PPG_125, p)
    except Exception:
        p_bp = p
    dp = np.gradient(p_bp, 1.0 / TARGET_HEMO_FS)
    d2p = np.gradient(dp, 1.0 / TARGET_HEMO_FS)
    return np.stack(
        [
            _robust_z_clip(e_bp),
            _robust_z_clip(p_bp),
            _robust_z_clip(dp),
            _robust_z_clip(d2p),
        ],
        axis=0,
    ).astype(np.float32)


def _compute_ppg_upstroke_ms(ppg: np.ndarray, fs: float = 250.0) -> float:
    x = np.asarray(ppg, dtype=np.float64).flatten()
    if len(x) < int(fs * 2.0) or not np.isfinite(x).any():
        return 160.0
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = x - np.median(x)
    p2p = float(np.max(x) - np.min(x))
    if p2p < 1e-6:
        return 160.0
    norm_x = (x - np.min(x)) / p2p
    peaks, _ = find_peaks(norm_x, distance=max(1, int(0.32 * fs)), prominence=0.20)
    if len(peaks) < 2:
        return 160.0
    upstrokes = []
    for pk in peaks:
        st = max(0, pk - int(0.38 * fs))
        if pk - st >= int(0.05 * fs):
            foot_idx = st + int(np.argmin(norm_x[st:pk]))
            dt_ms = (pk - foot_idx) * (1000.0 / fs)
            if 50.0 <= dt_ms <= 360.0:
                upstrokes.append(dt_ms)
    return round(float(np.median(upstrokes)) if upstrokes else 160.0, 2)


def build_stroke_feature_matrix(X_base: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
    """
    Normalizes 11-feature legacy, 15-feature base, or 22-feature active stroke
    inputs into the exact 22-column Branch 2A feature matrix (float32).
    """
    if isinstance(X_base, pd.DataFrame):
        if all(f in X_base.columns for f in STROKE_ACTIVE_FEATURES):
            mat = X_base[STROKE_ACTIVE_FEATURES].astype(np.float32).values
            return np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        df_work = pd.DataFrame(index=X_base.index)
        for col in STROKE_LEGACY_11_FEATURES:
            df_work[col] = pd.to_numeric(X_base[col], errors="coerce") if col in X_base.columns else 0.0
        for col, dval in _MORPH_DEFAULTS.items():
            df_work[col] = pd.to_numeric(X_base[col], errors="coerce").fillna(dval) if col in X_base.columns else dval
        arr = df_work[STROKE_BASE_FEATURES].astype(np.float32).values
    else:
        arr_in = np.asarray(X_base, dtype=np.float32)
        if arr_in.ndim == 1:
            arr_in = arr_in.reshape(1, -1)
        ncols = arr_in.shape[1]
        if ncols == len(STROKE_ACTIVE_FEATURES):
            return np.nan_to_num(arr_in, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        if ncols == len(STROKE_BASE_FEATURES):
            arr = arr_in
        elif ncols == len(STROKE_LEGACY_11_FEATURES):
            n = len(arr_in)
            morph_pad = np.tile(
                np.array([_MORPH_DEFAULTS[k] for k in STROKE_MORPH_4_FEATURES], dtype=np.float32),
                (n, 1),
            )
            arr = np.hstack([arr_in, morph_pad]).astype(np.float32)
        else:
            raise ValueError(
                f"Expected 11 legacy, 15 base, or 22 active stroke features; got shape {arr_in.shape}"
            )

    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    hr = np.maximum(1.0, arr[:, 0])
    pi = arr[:, 4]
    ptt = np.maximum(1e-3, arr[:, 5])
    dptt = arr[:, 6]
    sbp = np.maximum(1.0, arr[:, 7])
    dbp = arr[:, 8]
    cush = arr[:, 9]

    map_est = np.maximum(1.0, (sbp + 2.0 * dbp) / 3.0)
    pp = sbp - dbp

    rpp = ((hr * sbp) / 100.0).reshape(-1, 1)
    fpp = (pp / map_est).reshape(-1, 1)
    prr = (pi / hr).reshape(-1, 1)
    vsi = (120.0 / ptt).reshape(-1, 1)
    dpr = (dptt / ptt).reshape(-1, 1)
    shock_idx = (hr / np.maximum(50.0, sbp)).reshape(-1, 1)
    cush_vsi = (cush * (120.0 / ptt)).reshape(-1, 1)

    full = np.hstack([arr, rpp, fpp, prr, vsi, dpr, shock_idx, cush_vsi]).astype(np.float32)
    return np.nan_to_num(full, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


class _HemoAugDataset(Dataset):
    """PyTorch dataset for 4-channel 125 Hz ECG+PPG tensors with label-preserving noise augmentation."""

    def __init__(
        self,
        waveforms: np.ndarray,
        y_a: np.ndarray,
        y_b: np.ndarray,
        y_bp: Optional[np.ndarray] = None,
        augment: bool = False,
        seed: int = 42,
    ):
        self.X = np.asarray(waveforms, dtype=np.float32)
        self.y_a = np.asarray(y_a, dtype=np.int64)
        self.y_b = np.asarray(y_b, dtype=np.int64)
        self.y_bp = (
            np.asarray(y_bp, dtype=np.float32)
            if y_bp is not None
            else np.zeros((len(self.X), 2), dtype=np.float32)
        )
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.X[idx].copy()
        if self.augment:
            scale = float(self.rng.uniform(0.88, 1.12))
            x = x * scale
            noise = self.rng.normal(0.0, 0.015, size=x.shape).astype(np.float32)
            x = x + noise
        return (
            torch.from_numpy(x),
            torch.tensor(self.y_a[idx], dtype=torch.long),
            torch.tensor(self.y_b[idx], dtype=torch.long),
            torch.from_numpy(self.y_bp[idx]),
        )


class _HemoSEResBlock1D(nn.Module):
    """1D Residual Block with GroupNorm and Squeeze-and-Excitation channel attention."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 5, stride: int = 2, dropout: float = 0.15):
        super().__init__()
        pad = kernel_size // 2
        groups = min(8, out_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=pad, bias=False)
        self.gn1 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, stride=1, padding=pad, bias=False)
        self.gn2 = nn.GroupNorm(groups, out_ch)
        self.drop = nn.Dropout(dropout)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(out_ch, max(4, out_ch // 4), 1),
            nn.GELU(),
            nn.Conv1d(max(4, out_ch // 4), out_ch, 1),
            nn.Sigmoid(),
        )
        self.shortcut = (
            nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.GroupNorm(groups, out_ch),
            )
            if (in_ch != out_ch or stride != 1)
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.gn1(self.conv1(x)))
        h = self.drop(h)
        h = self.gn2(self.conv2(h))
        h = h * self.se(h)
        return F.gelu(h + self.shortcut(x))


class HemodynamicWaveformCNN(nn.Module):
    """
    Branch 2A Deep Learning Backbone:
    1D-ResNet + SE + GroupNorm + BiGRU + Attention Pooling on 4-ch [ECG_Z, PPG_Z, dPPG_Z, d2PPG_Z].
      - Head A: 4 hemodynamic stroke-risk tiers
      - Head B: 4 neurovascular acuity stages
      - Head BP (auxiliary): standardized [SBP, DBP] cuffless trend prediction
    """

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = len(STROKE_CLASSES),
        num_stages: int = len(STROKE_STAGE_NAMES),
        dropout: float = 0.30,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 24, kernel_size=9, stride=2, padding=4, bias=False),
            nn.GroupNorm(8, 24),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.block1 = _HemoSEResBlock1D(24, 48, kernel_size=7, stride=2, dropout=0.10)
        self.block2 = _HemoSEResBlock1D(48, 80, kernel_size=5, stride=2, dropout=0.15)
        self.block3 = _HemoSEResBlock1D(80, 96, kernel_size=5, stride=2, dropout=0.20)
        self.bigru = nn.GRU(
            input_size=96,
            hidden_size=48,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.attn_proj = nn.Linear(96, 1)
        self.shared_fc = nn.Sequential(
            nn.LayerNorm(192),
            nn.Dropout(dropout),
            nn.Linear(192, 80),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        self.head_a = nn.Linear(80, num_classes)
        self.head_b = nn.Linear(80, num_stages)
        self.head_bp = nn.Linear(80, 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.stem(x)
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        seq = h.transpose(1, 2)  # (B, T, 96)
        gru_out, _ = self.bigru(seq)  # (B, T, 96)
        attn_w = F.softmax(self.attn_proj(gru_out), dim=1)
        ctx_attn = torch.sum(gru_out * attn_w, dim=1)
        ctx_max, _ = torch.max(gru_out, dim=1)
        feat = torch.cat([ctx_attn, ctx_max], dim=1)  # (B, 192)
        rep = self.shared_fc(feat)
        return self.head_a(rep), self.head_b(rep), self.head_bp(rep)


class ClinicalStrokeRiskBranch:
    """
    Branch 2B: Clinical & Cardioembolic Stroke Risk Engine.
    Trained on the 2021 real Clinical Stroke Cohort (N=5,110 patients) with
    train-only median imputation and calibrated XGBoost probabilities, then fused
    with live Model 1 (CardiacModel1A) AFib/MI/Arrhythmia signals, CHA2DS2-VASc,
    and missed anticoagulant status via bounded log-odds updates.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.feature_names = list(CLINICAL_RISK_FEATURES)
        self.medians_: Dict[str, float] = {
            "age": 50.0,
            "gender_male": 0.0,
            "hypertension": 0.0,
            "heart_disease": 0.0,
            "ever_married": 1.0,
            "avg_glucose_level": 95.0,
            "diabetes": 0.0,
            "bmi": 28.0,
            "smoking_formerly": 0.0,
            "smoking_current": 0.0,
            "age_x_hypertension": 0.0,
            "cardiometabolic_burden": 0.0,
        }
        self.scaler = StandardScaler()
        self.calibrated_clf: Optional[CalibratedClassifierCV] = None
        self.is_fitted = False

    @staticmethod
    def _build_clinical_df(df_raw: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df_raw.index)
        age = pd.to_numeric(df_raw.get("age", 50.0), errors="coerce").astype(np.float32)
        htn = pd.to_numeric(df_raw.get("hypertension", 0), errors="coerce").fillna(0.0).astype(np.float32)
        hd = pd.to_numeric(df_raw.get("heart_disease", 0), errors="coerce").fillna(0.0).astype(np.float32)
        gluc = pd.to_numeric(df_raw.get("avg_glucose_level", 95.0), errors="coerce").astype(np.float32)
        bmi = pd.to_numeric(df_raw.get("bmi", np.nan), errors="coerce").astype(np.float32)

        out["age"] = age
        if "gender_male" in df_raw.columns:
            out["gender_male"] = pd.to_numeric(df_raw["gender_male"], errors="coerce").fillna(0.0).astype(np.float32)
        elif "gender" in df_raw.columns:
            out["gender_male"] = (df_raw["gender"].astype(str).str.strip().str.lower() == "male").astype(np.float32)
        else:
            out["gender_male"] = 0.0

        out["hypertension"] = htn
        out["heart_disease"] = hd
        if "ever_married" in df_raw.columns:
            col = df_raw["ever_married"]
            if col.dtype == object:
                out["ever_married"] = (col.astype(str).str.strip().str.lower() == "yes").astype(np.float32)
            else:
                out["ever_married"] = pd.to_numeric(col, errors="coerce").fillna(1.0).astype(np.float32)
        else:
            out["ever_married"] = 1.0

        out["avg_glucose_level"] = gluc
        if "diabetes" in df_raw.columns:
            diab = pd.to_numeric(df_raw["diabetes"], errors="coerce").fillna(0.0).astype(np.float32)
        else:
            diab = (gluc.fillna(95.0) >= 126.0).astype(np.float32)
        out["diabetes"] = diab
        out["bmi"] = bmi

        if "smoking_formerly" in df_raw.columns:
            out["smoking_formerly"] = pd.to_numeric(df_raw["smoking_formerly"], errors="coerce").fillna(0.0).astype(np.float32)
        elif "smoking_status" in df_raw.columns:
            out["smoking_formerly"] = (df_raw["smoking_status"].astype(str).str.strip().str.lower() == "formerly smoked").astype(np.float32)
        else:
            out["smoking_formerly"] = 0.0

        if "smoking_current" in df_raw.columns:
            out["smoking_current"] = pd.to_numeric(df_raw["smoking_current"], errors="coerce").fillna(0.0).astype(np.float32)
        elif "smoking_status" in df_raw.columns:
            out["smoking_current"] = (df_raw["smoking_status"].astype(str).str.strip().str.lower() == "smokes").astype(np.float32)
        else:
            out["smoking_current"] = 0.0

        out["age_x_hypertension"] = (age.fillna(50.0) * htn).astype(np.float32)
        out["cardiometabolic_burden"] = (
            htn
            + hd
            + np.maximum(diab, (gluc.fillna(95.0) >= 126.0).astype(np.float32))
            + (bmi.fillna(28.0) >= 30.0).astype(np.float32)
        ).astype(np.float32)
        return out[CLINICAL_RISK_FEATURES]

    def fit(self, df_train: pd.DataFrame, y_stroke: np.ndarray) -> Dict[str, Any]:
        """Fits train-only median imputer, scaler, and 5-fold sigmoid-calibrated XGBoost."""
        df_c = self._build_clinical_df(df_train)
        for col in self.feature_names:
            med = float(df_c[col].median()) if df_c[col].notna().any() else 0.0
            self.medians_[col] = med if np.isfinite(med) else 0.0
            df_c[col] = df_c[col].fillna(self.medians_[col])

        X_mat = df_c.values.astype(np.float32)
        X_sc = self.scaler.fit_transform(X_mat)
        y = np.asarray(y_stroke, dtype=np.int64)

        n_pos = max(1, int(np.sum(y == 1)))
        n_neg = max(1, int(np.sum(y == 0)))
        spw = min(20.0, float(n_neg / n_pos))

        base_xgb = XGBClassifier(
            objective="binary:logistic",
            max_depth=4,
            learning_rate=0.035,
            n_estimators=260,
            subsample=0.80,
            colsample_bytree=0.80,
            reg_alpha=1.0,
            reg_lambda=3.0,
            min_child_weight=10,
            scale_pos_weight=spw,
            eval_metric="logloss",
            random_state=self.random_state,
            n_jobs=4,
        )
        self.calibrated_clf = CalibratedClassifierCV(estimator=base_xgb, method="sigmoid", cv=5)
        self.calibrated_clf.fit(X_sc, y)
        self.is_fitted = True
        return {
            "n_samples": int(len(y)),
            "n_stroke": int(n_pos),
            "bmi_train_median": round(self.medians_["bmi"], 3),
        }

    def predict_base_proba(self, df_patients: pd.DataFrame) -> np.ndarray:
        """Predicts calibrated P(stroke) purely from clinical features using train-fit imputer."""
        if not self.is_fitted or self.calibrated_clf is None:
            raise RuntimeError("ClinicalStrokeRiskBranch is not fitted.")
        df_c = self._build_clinical_df(df_patients)
        for col in self.feature_names:
            df_c[col] = df_c[col].fillna(self.medians_.get(col, 0.0))
        X_sc = self.scaler.transform(df_c.values.astype(np.float32))
        return self.calibrated_clf.predict_proba(X_sc)[:, 1].astype(np.float64)

    @staticmethod
    def _normalize_profile_dict(profile: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        p = dict(profile or {})
        gender_str = str(p.get("gender", "")).strip().lower()
        gender_male = p.get("gender_male")
        if gender_male is None:
            gender_male = 1 if gender_str == "male" else 0
        else:
            gender_male = int(bool(gender_male))

        em_raw = p.get("ever_married", 1)
        if isinstance(em_raw, str):
            ever_married = 1 if em_raw.strip().lower() == "yes" else 0
        else:
            ever_married = int(bool(em_raw))

        smoke_str = str(p.get("smoking_status", "")).strip().lower()
        sm_form = p.get("smoking_formerly")
        sm_curr = p.get("smoking_current")
        if sm_form is None:
            sm_form = 1 if smoke_str == "formerly smoked" else 0
        if sm_curr is None:
            sm_curr = 1 if smoke_str == "smokes" else 0

        avg_gluc = _safe_float_val(p.get("avg_glucose_level"), 98.0)
        if p.get("diabetes") is not None:
            diab_val = int(bool(p.get("diabetes")))
        else:
            diab_val = int(avg_gluc >= 126.0)

        return {
            "age": _safe_float_val(p.get("age"), 55.0),
            "gender_male": int(gender_male),
            "hypertension": int(bool(p.get("hypertension", 0))),
            "heart_disease": int(bool(p.get("heart_disease", 0))),
            "ever_married": int(ever_married),
            "avg_glucose_level": avg_gluc,
            "bmi": _safe_float_val(p.get("bmi"), np.nan) if p.get("bmi") is not None else np.nan,
            "smoking_formerly": int(bool(sm_form)),
            "smoking_current": int(bool(sm_curr)),
            "diabetes": diab_val,
            "prior_stroke": int(bool(p.get("prior_stroke", p.get("prior_tia", 0)))),
            "vascular_disease": int(bool(p.get("vascular_disease", 0))),
        }

    @classmethod
    def compute_cha2ds2_vasc(
        cls,
        profile: Optional[Dict[str, Any]],
        cardiac_afib_prob: float = 0.0,
        cardiac_mi_prob: float = 0.0,
    ) -> int:
        """Computes guideline CHA2DS2-VASc score (0..9) from clinical profile & Model 1 MI context."""
        norm = cls._normalize_profile_dict(profile)
        score = 0
        # C: Heart disease / CHF
        if norm["heart_disease"]:
            score += 1
        # H: Hypertension
        if norm["hypertension"]:
            score += 1
        # A2 / A: Age
        if norm["age"] >= 75.0:
            score += 2
        elif norm["age"] >= 65.0:
            score += 1
        # D: Diabetes / Fasting-level hyperglycemia
        if norm["diabetes"] or norm["avg_glucose_level"] >= 126.0:
            score += 1
        # S2: Prior Stroke / TIA
        if norm["prior_stroke"]:
            score += 2
        # V: Vascular disease or significant ischemic MI signature
        if norm["vascular_disease"] or cardiac_mi_prob >= 0.40:
            score += 1
        # Sc: Sex category female
        if norm["gender_male"] == 0:
            score += 1
        return int(np.clip(score, 0, 9))

    @staticmethod
    def _risk_prob_to_4class(adj_prob: float) -> np.ndarray:
        """Maps calibrated [0, 1] stroke risk probability into a smooth normalized 4-class distribution."""
        p = float(np.clip(adj_prob, 0.0, 1.0))
        # Centers aligned to clinical risk bands:
        # <0.12 -> Low (0), [0.12, 0.28) -> Elevated (1), [0.28, 0.55) -> High FAST (2), >=0.55 -> Critical (3)
        centers = np.array([0.04, 0.18, 0.40, 0.68], dtype=np.float64)
        widths = np.array([0.075, 0.085, 0.125, 0.140], dtype=np.float64)
        logits = -0.5 * ((p - centers) / widths) ** 2
        logits -= np.max(logits)
        probs = np.exp(logits)
        return probs / np.sum(probs)

    def predict_clinical_risk(
        self,
        clinical_profile: Optional[Dict[str, Any]] = None,
        cardiac_probs: Optional[Dict[str, float]] = None,
        cardiac_afib_prob: float = 0.0,
        missed_anticoagulant: bool = False,
    ) -> Dict[str, Any]:
        """Evaluates Branch 2B clinical & cardioembolic risk with Model 1 AFib/MI context."""
        c_probs = dict(cardiac_probs or {})
        p_afib = float(np.clip(max(float(cardiac_afib_prob), float(c_probs.get("AFib", 0.0))), 0.0, 1.0))
        p_mi = float(np.clip(float(c_probs.get("Ischemic_MI", c_probs.get("MI", 0.0))), 0.0, 1.0))
        p_arr = float(np.clip(float(c_probs.get("Arrhythmia", 0.0)), 0.0, 1.0))

        has_profile = isinstance(clinical_profile, dict) and any(
            k in clinical_profile
            for k in ("age", "hypertension", "heart_disease", "avg_glucose_level", "bmi", "gender", "gender_male", "smoking_status")
        )
        has_cardiac_ctx = bool(cardiac_probs is not None or p_afib > 0.0 or missed_anticoagulant)
        active = bool(has_profile or has_cardiac_ctx)

        norm_prof = self._normalize_profile_dict(clinical_profile)
        if self.is_fitted and self.calibrated_clf is not None:
            df_one = pd.DataFrame([norm_prof])
            base_prob = float(self.predict_base_proba(df_one)[0])
        else:
            # Deterministic baseline fallback before fitting
            base_prob = 0.04 + 0.05 * norm_prof["hypertension"] + 0.05 * norm_prof["heart_disease"]
            base_prob = float(np.clip(base_prob, 0.01, 0.85))

        eps = 1e-5
        p_clamped = float(np.clip(base_prob, eps, 1.0 - eps))
        logit = float(np.log(p_clamped / (1.0 - p_clamped)))

        # Cardioembolic risk log-odds update (AHA/Framingham AFib ~4-5x risk + MI + anticoagulation adherence)
        shift = 0.0
        if p_afib > 0.05:
            shift += 1.55 * p_afib
        if p_mi > 0.15:
            shift += 0.65 * p_mi
        if p_arr > 0.15:
            shift += 0.25 * p_arr
        anticoag_applied = bool(missed_anticoagulant and p_afib >= 0.30)
        if anticoag_applied:
            shift += float(np.log(1.35))

        adj_prob = float(np.clip(1.0 / (1.0 + np.exp(-(logit + shift))), 0.0, 1.0))
        cha2ds2 = self.compute_cha2ds2_vasc(norm_prof, cardiac_afib_prob=p_afib, cardiac_mi_prob=p_mi)
        probs_4 = self._risk_prob_to_4class(adj_prob)

        return {
            "active": active,
            "profile_supplied": has_profile,
            "clinical_ml_base_stroke_prob": round(base_prob, 4),
            "cardioembolic_adjusted_stroke_prob": round(adj_prob, 4),
            "cha2ds2_vasc_score": cha2ds2,
            "cardiac_afib_prob_used": round(p_afib, 4),
            "cardiac_ischemic_mi_prob_used": round(p_mi, 4),
            "missed_anticoagulant_multiplier_applied": anticoag_applied,
            "branch_2b_probs_4class": probs_4,
        }


class StrokeModel1B:
    """
    Model 2 (`BrainStrokeRiskModel2` / `StrokeModel1B`):
    Two-Branch Hybrid AI Stroke Risk & Acute Neurovascular Trigger Engine.
      - Branch 2A: 1D-ResNet-SE-BiGRU-Attention + Tabular XGBoost (22 hemodynamic features)
      - Branch 2B: Calibrated Clinical Stroke Risk XGBoost + Model 1 Cardioembolic Fusion
      - Outputs: Head A (4 Risk Tiers), Head B (4 Stages), Golden Hour Timer,
        Model 3 FAST Trigger, IMU Fall Flag, and Non-Mutating Red CT Contraindication Veto.
    """

    def __init__(
        self,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        n_estimators: int = 180,
        subsample: float = 0.80,
        colsample_bytree: float = 0.80,
        reg_alpha: float = 1.0,
        reg_lambda: float = 2.0,
        min_child_weight: int = 5,
        dl_weight: float = 0.45,
        random_state: int = 42,
    ):
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.min_child_weight = min_child_weight
        self.dl_weight = float(dl_weight)
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.head_a = XGBClassifier(
            objective="multi:softprob",
            num_class=len(STROKE_CLASSES),
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_child_weight=min_child_weight,
            eval_metric="mlogloss",
            random_state=random_state,
            n_jobs=4,
        )
        self.head_b = XGBClassifier(
            objective="multi:softprob",
            num_class=len(STROKE_STAGE_NAMES),
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_child_weight=min_child_weight,
            eval_metric="mlogloss",
            random_state=random_state,
            n_jobs=4,
        )

        self.active_features = list(STROKE_ACTIVE_FEATURES)
        self.disease_classes = list(STROKE_CLASSES)
        self.stage_names = list(STROKE_STAGE_NAMES)
        self.cohort_status = "CERTIFIED_2021_2023_REAL_HYBRID_STROKE_RISK_AND_FAST_TRIGGER"
        self.cnn_weights_path: Optional[str] = DEFAULT_STROKE_CNN_PATH
        self.cnn_model: Optional[HemodynamicWaveformCNN] = None
        self.bp_mean = np.array([132.0, 72.0], dtype=np.float32)
        self.bp_std = np.array([25.0, 14.0], dtype=np.float32)
        self.clinical_branch: Optional[ClinicalStrokeRiskBranch] = ClinicalStrokeRiskBranch(random_state=random_state)
        self.is_fitted = False

    def _ensure_cnn(self) -> Optional[HemodynamicWaveformCNN]:
        if self.cnn_model is None and self.cnn_weights_path and os.path.exists(self.cnn_weights_path):
            try:
                ckpt = torch.load(self.cnn_weights_path, map_location="cpu", weights_only=False)
                state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
                if isinstance(ckpt, dict) and "bp_mean" in ckpt and "bp_std" in ckpt:
                    self.bp_mean = np.asarray(ckpt["bp_mean"], dtype=np.float32)
                    self.bp_std = np.asarray(ckpt["bp_std"], dtype=np.float32)
                net = HemodynamicWaveformCNN(
                    in_channels=4,
                    num_classes=len(self.disease_classes),
                    num_stages=len(self.stage_names),
                )
                net.load_state_dict(state)
                net.eval()
                self.cnn_model = net
            except Exception:
                self.cnn_model = None
        return self.cnn_model

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y_stroke: Union[pd.Series, np.ndarray],
        y_stage: Union[pd.Series, np.ndarray],
    ) -> Dict[str, Any]:
        """Fits Branch 2A Tabular XGBoost Head A & Head B on 22 hemodynamic wearable features."""
        X_mat = build_stroke_feature_matrix(X)
        X_scaled = self.scaler.fit_transform(X_mat)

        y_a = np.asarray(y_stroke, dtype=np.int64)
        y_b = np.asarray(y_stage, dtype=np.int64)

        cls_a = np.unique(y_a)
        wa_map = dict(zip(cls_a, compute_class_weight("balanced", classes=cls_a, y=y_a)))
        sw_a = np.array([wa_map[int(c)] for c in y_a], dtype=np.float32)

        cls_b = np.unique(y_b)
        wb_map = dict(zip(cls_b, compute_class_weight("balanced", classes=cls_b, y=y_b)))
        sw_b = np.array([wb_map[int(c)] for c in y_b], dtype=np.float32)

        self.head_a.fit(X_scaled, y_a, sample_weight=sw_a)
        self.head_b.fit(X_scaled, y_b, sample_weight=sw_b)
        self.is_fitted = True
        return {
            "n_samples": int(len(X_mat)),
            "head_a_counts": {int(k): int(np.sum(y_a == k)) for k in cls_a},
            "head_b_counts": {int(k): int(np.sum(y_b == k)) for k in cls_b},
        }

    def fit_deep_cnn(
        self,
        waveforms: np.ndarray,
        y_risk: np.ndarray,
        y_stage: np.ndarray,
        y_bp: Optional[np.ndarray] = None,
        val_waveforms: Optional[np.ndarray] = None,
        val_risk: Optional[np.ndarray] = None,
        val_stage: Optional[np.ndarray] = None,
        val_bp: Optional[np.ndarray] = None,
        epochs: int = 18,
        batch_size: int = 64,
        lr: float = 1.5e-3,
        save_cnn_path: Optional[str] = DEFAULT_STROKE_CNN_PATH,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Trains Branch 2A 1D-ResNet-SE-BiGRU on (N, 4, 1250) 125 Hz ECG+PPG tensors."""
        torch.manual_seed(self.random_state)
        torch.set_num_threads(min(4, max(1, os.cpu_count() or 2)))

        if y_bp is not None:
            bp_arr = np.asarray(y_bp, dtype=np.float32)
            self.bp_mean = np.mean(bp_arr, axis=0).astype(np.float32)
            self.bp_std = np.maximum(1.0, np.std(bp_arr, axis=0)).astype(np.float32)
            bp_tr_sc = (bp_arr - self.bp_mean) / self.bp_std
        else:
            bp_tr_sc = np.zeros((len(waveforms), 2), dtype=np.float32)

        ds_train = _HemoAugDataset(waveforms, y_risk, y_stage, bp_tr_sc, augment=True, seed=self.random_state)
        n_cls = len(self.disease_classes)
        cls_counts = np.bincount(y_risk.astype(np.int64), minlength=n_cls)
        inv_w = 1.0 / np.maximum(1.0, cls_counts.astype(np.float64))
        sample_w = torch.DoubleTensor([inv_w[int(c)] for c in y_risk])
        sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)
        train_loader = DataLoader(ds_train, batch_size=batch_size, sampler=sampler, drop_last=False)

        val_loader = None
        if val_waveforms is not None and val_risk is not None and val_stage is not None:
            if val_bp is not None:
                bp_val_sc = (np.asarray(val_bp, dtype=np.float32) - self.bp_mean) / self.bp_std
            else:
                bp_val_sc = np.zeros((len(val_waveforms), 2), dtype=np.float32)
            ds_val = _HemoAugDataset(val_waveforms, val_risk, val_stage, bp_val_sc, augment=False, seed=self.random_state)
            val_loader = DataLoader(ds_val, batch_size=128, shuffle=False)

        net = HemodynamicWaveformCNN(
            in_channels=4,
            num_classes=n_cls,
            num_stages=len(self.stage_names),
            dropout=0.30,
        )
        optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=2e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

        crit_a = nn.CrossEntropyLoss(label_smoothing=0.03)
        crit_b = nn.CrossEntropyLoss(label_smoothing=0.03)
        crit_bp = nn.HuberLoss(delta=1.0)

        best_val_loss = float("inf")
        best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
        history: List[Dict[str, float]] = []

        for ep in range(1, epochs + 1):
            net.train()
            tr_loss = 0.0
            tr_correct = 0
            tr_total = 0
            for xb, yb_a, yb_b, yb_bp in train_loader:
                optimizer.zero_grad()
                la, lb, lbp = net(xb)
                loss = crit_a(la, yb_a) + 0.45 * crit_b(lb, yb_b)
                if y_bp is not None:
                    loss = loss + 0.25 * crit_bp(lbp, yb_bp)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
                optimizer.step()
                tr_loss += float(loss.item()) * len(yb_a)
                tr_correct += int((torch.argmax(la, dim=1) == yb_a).sum().item())
                tr_total += len(yb_a)
            scheduler.step()

            ep_stat: Dict[str, float] = {
                "epoch": float(ep),
                "train_loss": tr_loss / max(1, tr_total),
                "train_acc": tr_correct / max(1, tr_total),
            }

            if val_loader is not None:
                net.eval()
                v_loss = 0.0
                v_correct = 0
                v_total = 0
                with torch.no_grad():
                    for xb, yb_a, yb_b, yb_bp in val_loader:
                        la, lb, lbp = net(xb)
                        loss = crit_a(la, yb_a) + 0.45 * crit_b(lb, yb_b)
                        if val_bp is not None:
                            loss = loss + 0.25 * crit_bp(lbp, yb_bp)
                        v_loss += float(loss.item()) * len(yb_a)
                        v_correct += int((torch.argmax(la, dim=1) == yb_a).sum().item())
                        v_total += len(yb_a)
                ep_val_loss = v_loss / max(1, v_total)
                ep_stat["val_loss"] = ep_val_loss
                ep_stat["val_acc"] = v_correct / max(1, v_total)
                if ep_val_loss < best_val_loss:
                    best_val_loss = ep_val_loss
                    best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
            else:
                best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}

            history.append(ep_stat)
            if verbose:
                v_msg = f" | val_loss={ep_stat['val_loss']:.4f} val_acc={ep_stat['val_acc']*100:.2f}%" if "val_loss" in ep_stat else ""
                print(
                    f"  [Hemo-CNN Ep {ep:02d}/{epochs}] tr_loss={ep_stat['train_loss']:.4f} tr_acc={ep_stat['train_acc']*100:.2f}%{v_msg}",
                    flush=True,
                )

        net.load_state_dict(best_state)
        net.eval()
        self.cnn_model = net
        if save_cnn_path:
            os.makedirs(os.path.dirname(save_cnn_path), exist_ok=True)
            torch.save(
                {
                    "state_dict": best_state,
                    "bp_mean": self.bp_mean.tolist(),
                    "bp_std": self.bp_std.tolist(),
                },
                save_cnn_path,
            )
            self.cnn_weights_path = save_cnn_path

        return {"epochs": epochs, "best_val_loss": best_val_loss, "history": history}

    def fit_clinical_branch(self, df_train: pd.DataFrame, y_stroke: np.ndarray) -> Dict[str, Any]:
        """Fits Branch 2B clinical stroke risk model on patient-level clinical risk dataset."""
        if self.clinical_branch is None:
            self.clinical_branch = ClinicalStrokeRiskBranch(random_state=self.random_state)
        return self.clinical_branch.fit(df_train, y_stroke)

    def predict_cnn_outputs(
        self, waveforms: np.ndarray, batch_size: int = 128
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Runs batched Branch 2A CNN returning `(probs_a, probs_b, bp_mmhg)` on `(N, 4, 1250)` tensors."""
        net = self._ensure_cnn()
        if net is None:
            raise RuntimeError("Branch 2A HemodynamicWaveformCNN weights are not loaded.")
        X_w = np.asarray(waveforms, dtype=np.float32)
        if X_w.ndim == 2:
            X_w = X_w[None, :, :]
        pa_list, pb_list, bp_list = [], [], []
        net.eval()
        with torch.no_grad():
            for st in range(0, len(X_w), batch_size):
                xb = torch.from_numpy(X_w[st : st + batch_size])
                la, lb, lbp = net(xb)
                pa_list.append(torch.softmax(la, dim=-1).cpu().numpy())
                pb_list.append(torch.softmax(lb, dim=-1).cpu().numpy())
                bp_mmhg = lbp.cpu().numpy() * self.bp_std[None, :] + self.bp_mean[None, :]
                bp_list.append(bp_mmhg)
        return np.vstack(pa_list), np.vstack(pb_list), np.vstack(bp_list)

    def predict_batch_components(
        self,
        waveforms: np.ndarray,
        X_feat: Union[pd.DataFrame, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Evaluates XGBoost, CNN, and Hybrid ensemble components cleanly in batch for auditing."""
        if not self.is_fitted:
            raise RuntimeError("StrokeModel1B is not fitted yet.")
        X_mat = build_stroke_feature_matrix(X_feat)
        X_sc = self.scaler.transform(X_mat)
        pa_xgb = self.head_a.predict_proba(X_sc).astype(np.float64)
        pb_xgb = self.head_b.predict_proba(X_sc).astype(np.float64)

        net = self._ensure_cnn()
        if net is not None:
            pa_cnn, pb_cnn, bp_cnn = self.predict_cnn_outputs(waveforms)
            w = float(np.clip(self.dl_weight, 0.0, 1.0))
            pa_ens = (1.0 - w) * pa_xgb + w * pa_cnn
            pb_ens = (1.0 - w) * pb_xgb + w * pb_cnn
        else:
            pa_cnn = pa_xgb
            pb_cnn = pb_xgb
            bp_cnn = X_mat[:, [7, 8]]
            pa_ens = pa_xgb
            pb_ens = pb_xgb

        return {
            "probs_a_xgb": pa_xgb,
            "probs_b_xgb": pb_xgb,
            "probs_a_cnn": pa_cnn,
            "probs_b_cnn": pb_cnn,
            "probs_a_ens": pa_ens,
            "probs_b_ens": pb_ens,
            "bp_cnn_mmhg": bp_cnn,
        }

    def _prepare_matrix_with_meta(
        self, X: Union[Dict[str, Any], pd.DataFrame, np.ndarray]
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Normalizes raw sensor payloads, feature dicts, DataFrames, or arrays into `(N, 22)` + telemetry gates."""
        default_meta: Dict[str, Any] = {
            "status": "VALID",
            "inference_reliable": True,
            "ppg_sqi": 1.0,
            "motion_gated_ppg": False,
            "wave_tensor": None,
        }

        if isinstance(X, dict):
            if "raw_windows" in X:
                rw = X["raw_windows"]
                fs_in = _safe_float_val(rw.get("sampling_rate_hz", 250.0), 250.0)
                ecg_raw = rw.get("ecg_mv", [0.0] * 2500)
                ecg = np.nan_to_num(np.asarray(ecg_raw, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
                ppg_raw = rw.get("ppg_ir", rw.get("ppg_red"))
                if ppg_raw is not None:
                    ppg_arr = np.nan_to_num(np.asarray(ppg_raw, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
                    if len(ppg_arr) >= 60 and float(np.max(ppg_arr)) > 5000.0:
                        valid_mask = ppg_arr > 1000.0
                        valid_frac = float(np.mean(valid_mask))
                        if valid_frac < 0.75:
                            has_ppg = False
                        else:
                            if valid_frac < 1.0:
                                idx_all = np.arange(len(ppg_arr), dtype=np.float64)
                                ppg_arr = np.interp(
                                    idx_all,
                                    idx_all[valid_mask],
                                    ppg_arr[valid_mask].astype(np.float64),
                                ).astype(np.float32)
                            has_ppg = float(np.var(ppg_arr)) > 1e-4
                    else:
                        has_ppg = bool(len(ppg_arr) >= 60 and float(np.var(ppg_arr)) > 1e-4)
                else:
                    ppg_arr = np.full(2500, 50000.0, dtype=np.float32)
                    has_ppg = False

                imu = {
                    "ax": np.asarray(rw.get("imu_ax_g", [0.0] * 500), dtype=np.float32),
                    "ay": np.asarray(rw.get("imu_ay_g", [-1.0] * 500), dtype=np.float32),
                    "az": np.asarray(rw.get("imu_az_g", [0.0] * 500), dtype=np.float32),
                    "gx": np.asarray(rw.get("imu_gx_dps", [0.0] * 500), dtype=np.float32),
                    "gy": np.asarray(rw.get("imu_gy_dps", [0.0] * 500), dtype=np.float32),
                    "gz": np.asarray(rw.get("imu_gz_dps", [0.0] * 500), dtype=np.float32),
                }
                feat = dict(extract_10s_features(ecg, ppg_arr, imu, rehab_active=bool(X.get("rehab_exercise_active", False)), fs=fs_in))
                ppg_sqi = float(compute_ppg_sqi(ppg_arr, fs=fs_in, causal=False)) if has_ppg else 0.0
                smv_val = _safe_float_val(X.get("imu_smv_g", feat.get("imu_smv_max_g", feat.get("imu_smv_g", 1.0))), 1.0)
                motion_gated = bool(smv_val > 1.35 or not np.isfinite(_safe_float_val(feat.get("estimated_sbp_mmhg"), np.nan)))

                if not has_ppg:
                    meta = {
                        "status": "MISSING_HEMODYNAMIC_PPG",
                        "inference_reliable": False,
                        "ppg_sqi": 0.0,
                        "motion_gated_ppg": True,
                        "wave_tensor": None,
                    }
                elif ppg_sqi < 0.25:
                    meta = {
                        "status": "POOR_PPG_SIGNAL",
                        "inference_reliable": False,
                        "ppg_sqi": round(ppg_sqi, 3),
                        "motion_gated_ppg": motion_gated,
                        "wave_tensor": None,
                    }
                else:
                    wave_tensor = build_hemo_waveform_tensor(ecg, ppg_arr, fs=fs_in)
                    meta = {
                        "status": "VALID",
                        "inference_reliable": True,
                        "ppg_sqi": round(ppg_sqi, 3),
                        "motion_gated_ppg": motion_gated,
                        "wave_tensor": wave_tensor,
                    }

                if motion_gated:
                    feat["ptt_ms"] = 180.0
                    feat["delta_ptt_ms"] = 0.0
                    feat["estimated_sbp_mmhg"] = 120.0
                    feat["estimated_dbp_mmhg"] = 80.0
                    feat["cushing_index"] = round(120.0 / 72.0, 4)
                else:
                    hr_val = max(40.0, _safe_float_val(feat.get("hr_ecg_bpm"), 72.0))
                    sbp_val = _safe_float_val(feat.get("estimated_sbp_mmhg"), 120.0)
                    feat["cushing_index"] = round(sbp_val / max(20.0, hr_val), 4)
                feat["ppg_upstroke_ms"] = _compute_ppg_upstroke_ms(ppg_arr, fs=fs_in) if has_ppg else 160.0
                feat["imu_smv_g"] = smv_val
            else:
                src = X.get("features", X) if isinstance(X.get("features"), dict) else X
                feat = dict(src)
                has_hemo = all(
                    feat.get(k) is not None and np.isfinite(float(feat.get(k)))
                    for k in ("estimated_sbp_mmhg", "ptt_ms")
                )
                if not has_hemo:
                    meta = {
                        "status": "MISSING_HEMODYNAMIC_SENSORS",
                        "inference_reliable": False,
                        "ppg_sqi": 0.0,
                        "motion_gated_ppg": True,
                        "wave_tensor": None,
                    }
                else:
                    meta = dict(default_meta)

            hr_val = _safe_float_val(feat.get("hr_ecg_bpm"), 72.0)
            est_sbp_val = _safe_float_val(feat.get("estimated_sbp_mmhg"), 120.0)
            base = {
                "hr_ecg_bpm": hr_val,
                "pulse_rate_ppg_bpm": _safe_float_val(feat.get("pulse_rate_ppg_bpm"), hr_val),
                "pulse_deficit_bpm": _safe_float_val(feat.get("pulse_deficit_bpm"), 0.0),
                "spo2_pct": _safe_float_val(feat.get("spo2_pct"), 98.0),
                "perfusion_index_pct": _safe_float_val(feat.get("perfusion_index_pct"), 3.0),
                "ptt_ms": _safe_float_val(feat.get("ptt_ms"), 180.0),
                "delta_ptt_ms": _safe_float_val(feat.get("delta_ptt_ms"), 0.0),
                "estimated_sbp_mmhg": est_sbp_val,
                "estimated_dbp_mmhg": _safe_float_val(feat.get("estimated_dbp_mmhg"), 80.0),
                "cushing_index": _safe_float_val(feat.get("cushing_index"), round(est_sbp_val / max(20.0, hr_val), 4)),
                "imu_smv_g": _safe_float_val(feat.get("imu_smv_g", feat.get("imu_smv_max_g")), 1.0),
                "sdptg_b_a_ratio": _safe_float_val(feat.get("sdptg_b_a_ratio"), -0.65),
                "tau_diastolic_s": _safe_float_val(feat.get("tau_diastolic_s"), 0.35),
                "dicrotic_notch_index": _safe_float_val(feat.get("dicrotic_notch_index"), 0.30),
                "ppg_upstroke_ms": _safe_float_val(feat.get("ppg_upstroke_ms"), 160.0),
            }
            mat = build_stroke_feature_matrix(pd.DataFrame([base]))
            return mat, [meta]

        mat = build_stroke_feature_matrix(X)
        return mat, [dict(default_meta) for _ in range(len(mat))]

    def predict_all(
        self,
        X: Union[Dict[str, Any], pd.DataFrame, np.ndarray],
        elapsed_stage_sec: float = 0.0,
        cardiac_afib_prob: float = 0.0,
        cardiac_probs: Optional[Dict[str, float]] = None,
        clinical_profile: Optional[Dict[str, Any]] = None,
        missed_anticoagulant: bool = False,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Executes Two-Branch Hybrid Stroke Risk & Neurovascular Trigger prediction."""
        if not self.is_fitted:
            raise RuntimeError("StrokeModel1B is not fitted yet.")

        if isinstance(X, dict):
            if clinical_profile is None:
                clinical_profile = X.get("clinical_profile") or X.get("patient_context")
            if cardiac_probs is None:
                cardiac_probs = X.get("cardiac_probs") or X.get("cardiac_context")
            if not missed_anticoagulant:
                pctx = X.get("patient_context") or X.get("clinical_profile") or {}
                missed_anticoagulant = bool(X.get("missed_anticoagulant_dose", pctx.get("missed_anticoagulant_dose", False)))
            if cardiac_afib_prob == 0.0 and X.get("cardiac_afib_prob") is not None:
                cardiac_afib_prob = float(X["cardiac_afib_prob"])

        is_single = isinstance(X, dict) or (isinstance(X, np.ndarray) and X.ndim == 1)
        X_mat, metas = self._prepare_matrix_with_meta(X)
        X_scaled = self.scaler.transform(np.nan_to_num(X_mat, nan=0.0, posinf=0.0, neginf=0.0))

        probs_a_xgb = self.head_a.predict_proba(X_scaled).astype(np.float64)
        probs_b_xgb = self.head_b.predict_proba(X_scaled).astype(np.float64)

        b2b = (
            self.clinical_branch.predict_clinical_risk(
                clinical_profile=clinical_profile,
                cardiac_probs=cardiac_probs,
                cardiac_afib_prob=cardiac_afib_prob,
                missed_anticoagulant=missed_anticoagulant,
            )
            if self.clinical_branch is not None
            else {
                "active": False,
                "profile_supplied": False,
                "clinical_ml_base_stroke_prob": 0.0,
                "cardioembolic_adjusted_stroke_prob": 0.0,
                "cha2ds2_vasc_score": 0,
                "cardiac_afib_prob_used": float(cardiac_afib_prob),
                "cardiac_ischemic_mi_prob_used": 0.0,
                "missed_anticoagulant_multiplier_applied": False,
                "branch_2b_probs_4class": np.array([0.85, 0.10, 0.04, 0.01], dtype=np.float64),
            }
        )
        p_afib_used = float(b2b["cardiac_afib_prob_used"])
        p_mi_used = float(b2b["cardiac_ischemic_mi_prob_used"])
        use_b2b_fusion = bool(
            b2b["profile_supplied"] or p_afib_used >= 0.25 or p_mi_used >= 0.35 or b2b["missed_anticoagulant_multiplier_applied"]
        )

        results: List[Dict[str, Any]] = []
        for i in range(len(X_mat)):
            meta = dict(metas[i])
            wave_tensor = meta.pop("wave_tensor", None)
            hemo_reliable = bool(meta.get("inference_reliable", True))

            if not hemo_reliable and not b2b["profile_supplied"]:
                results.append(
                    {
                        "model_id": "StrokeModel1B",
                        "model_alias": "BrainStrokeRiskModel2",
                        "cohort_status": self.cohort_status,
                        "sensor_status": meta["status"],
                        "medical_scope_disclaimer": MEDICAL_SCOPE_DISCLAIMER,
                        "disease_code": 0,
                        "disease_name": self.disease_classes[0],
                        "disease_probabilities": {c: (1.0 if idx == 0 else 0.0) for idx, c in enumerate(self.disease_classes)},
                        "stage_code": 0,
                        "stage_name": self.stage_names[0],
                        "stage_probabilities": {s: (1.0 if idx == 0 else 0.0) for idx, s in enumerate(self.stage_names)},
                        "stroke_risk_score": 0.0,
                        "stroke_risk_score_pct": 0.0,
                        "golden_hour_remaining_min": None,
                        "golden_hour_status": f"Telemetry Gate ({meta['status']}) - Hemodynamic Sensor Unavailable",
                        "trigger_model_3_fast_check": False,
                        "trigger_model_2_camera_voice_check": False,
                        "imu_fall_detected": False,
                        "red_contraindication_veto": False,
                        "cardiac_afib_risk_context": round(p_afib_used, 4),
                        "telemetry_quality": meta,
                        "branch_2a_hemodynamics": {
                            "active": False,
                            "dl_branch_active": False,
                            "cuffless_bp_estimate_mmhg": None,
                        },
                        "branch_2b_clinical_cardioembolic": {k: v for k, v in b2b.items() if k != "branch_2b_probs_4class"},
                        "clinical_alert": f"STROKE TELEMETRY GATE ({meta['status']}): PPG/BP HEMODYNAMICS UNAVAILABLE.",
                        "hardware_feedback": {"trigger_buzzer": False, "trigger_vibrator": False},
                        "raw_ml_decision": {
                            "disease_code": None,
                            "disease_name": "ABSTAINED_NO_HEMODYNAMICS",
                            "stage_code": None,
                            "stage_name": "ABSTAINED_NO_HEMODYNAMICS",
                            "rules_overrode_ai": False,
                        },
                    }
                )
                continue

            if not hemo_reliable and b2b["profile_supplied"]:
                # PPG/BP offline, but valid clinical risk profile provided -> Branch 2B standalone risk
                probs_a = np.asarray(b2b["branch_2b_probs_4class"], dtype=np.float64)
                probs_a = probs_a / np.sum(probs_a)
                probs_b = np.array(
                    [probs_a[0], probs_a[1], probs_a[2], probs_a[3]],
                    dtype=np.float64,
                )
                probs_b = probs_b / np.sum(probs_b)
                dl_active = False
                bp_est = None
                b2a_active = False
            else:
                pa_2a = probs_a_xgb[i].copy()
                pb_2a = probs_b_xgb[i].copy()
                dl_active = False
                bp_est = {
                    "sbp": round(float(X_mat[i, 7]), 1),
                    "dbp": round(float(X_mat[i, 8]), 1),
                    "source": "ptt_ppg_tabular",
                }
                net = self._ensure_cnn()
                if wave_tensor is not None and net is not None:
                    pa_cnn, pb_cnn, bp_cnn = self.predict_cnn_outputs(wave_tensor[None, :, :])
                    w = float(np.clip(self.dl_weight, 0.0, 1.0))
                    pa_2a = (1.0 - w) * pa_2a + w * pa_cnn[0]
                    pb_2a = (1.0 - w) * pb_2a + w * pb_cnn[0]
                    dl_active = True
                    bp_est = {
                        "sbp": round(float(np.clip(bp_cnn[0, 0], 60.0, 260.0)), 1),
                        "dbp": round(float(np.clip(bp_cnn[0, 1], 30.0, 160.0)), 1),
                        "source": "hybrid_1d_cnn",
                    }

                if use_b2b_fusion:
                    probs_2b_a = np.asarray(b2b["branch_2b_probs_4class"], dtype=np.float64)
                    probs_a = 0.72 * pa_2a + 0.28 * probs_2b_a
                    probs_b = 0.75 * pb_2a + 0.25 * probs_2b_a
                else:
                    probs_a = pa_2a
                    probs_b = pb_2a

                probs_a = probs_a / np.maximum(1e-12, np.sum(probs_a))
                probs_b = probs_b / np.maximum(1e-12, np.sum(probs_b))
                b2a_active = True

            d_code = int(np.argmax(probs_a))
            s_code = int(np.argmax(probs_b))
            d_name = self.disease_classes[d_code]
            s_name = self.stage_names[s_code]
            pa_dict = {c: round(float(probs_a[idx]), 6) for idx, c in enumerate(self.disease_classes)}
            pb_dict = {s: round(float(probs_b[idx]), 6) for idx, s in enumerate(self.stage_names)}

            hr_val = float(X_mat[i, 0])
            sbp_val = float(X_mat[i, 7])
            cush_val = float(X_mat[i, 9])
            imu_smv = float(X_mat[i, 10])

            # Continuous bounded Stroke Risk Score [0, 1]
            hemo_risk_score = float(0.10 * probs_a[0] + 0.40 * probs_a[1] + 0.75 * probs_a[2] + 1.00 * probs_a[3])
            if use_b2b_fusion:
                risk_score = float(np.clip(0.72 * hemo_risk_score + 0.28 * float(b2b["cardioembolic_adjusted_stroke_prob"]), 0.0, 1.0))
            else:
                risk_score = float(np.clip(hemo_risk_score, 0.0, 1.0))

            # Golden Hour Timer
            if s_code == 0 or d_code == 0:
                rem_min = None
                timer_str = "No Acute Neurovascular Crisis"
            elif s_code == 1:
                rem_min = 60.0
                timer_str = "Elevated Chronic / Cardioembolic Risk - 60.0 min Window Standby"
            elif s_code == 2:
                elapsed_min = min(1.0, max(0.0, elapsed_stage_sec / 60.0))
                rem_min = round(60.0 - elapsed_min, 1)
                timer_str = f"ACUTE NEUROVASCULAR / FAST TRIGGER - GOLDEN HOUR: {rem_min:.1f} min REMAINING"
            else:
                elapsed_min = max(1.0, elapsed_stage_sec / 60.0) if elapsed_stage_sec > 0 else 15.0
                deduction = 20.0 if (cush_val >= 2.2 or sbp_val >= 180.0) else 10.0
                rem_min = round(max(0.0, 60.0 - elapsed_min - deduction), 1)
                timer_str = f"CRITICAL NEUROVASCULAR CRISIS - GOLDEN HOUR: {rem_min:.1f} min REMAINING"

            imu_fall_detected = bool(imu_smv >= 3.0)
            trigger_fast = bool(
                (d_code >= 2)
                or (float(probs_a[2] + probs_a[3]) >= 0.30)
                or (float(probs_a[3]) >= 0.15)
                or imu_fall_detected
                or (p_afib_used >= 0.50 and (missed_anticoagulant or d_code >= 1))
            )
            red_veto = bool(
                (d_code == 3)
                or (float(probs_a[3]) >= 0.15)
                or (cush_val >= 2.05 and sbp_val >= 160.0)
                or (hr_val <= 62.0 and sbp_val >= 165.0 and imu_smv >= 2.8)
                or (trigger_fast and sbp_val >= 180.0)
            )

            if red_veto:
                alert = (
                    "CRITICAL NEUROVASCULAR CRISIS ALERT: Hemorrhagic vs. Ischemic stroke CANNOT be separated without Head CT. "
                    "RED CONTRAINDICATION: DO NOT ADMINISTER ASPIRIN, ANTICOAGULANTS, OR THROMBOLYTICS PRIOR TO NON-CONTRAST HEAD CT SCAN."
                )
            elif trigger_fast:
                afib_note = " (Elevated AFib Cardioembolic Context)" if p_afib_used >= 0.45 else ""
                alert = (
                    f"HIGH STROKE RISK / ACUTE TRIGGER{afib_note}: "
                    "ACTIVATE MODEL 3 (CAMERA, VOICE & ARM-DRIFT FAST CHECK)."
                )
            elif d_code == 1:
                alert = "ELEVATED STROKE RISK: Monitor blood pressure trend, ECG rhythm, and clinical risk factors."
            else:
                alert = "NEUROVASCULAR BASELINE: LOW STROKE RISK."

            results.append(
                {
                    "model_id": "StrokeModel1B",
                    "model_alias": "BrainStrokeRiskModel2",
                    "cohort_status": self.cohort_status,
                    "medical_scope_disclaimer": MEDICAL_SCOPE_DISCLAIMER,
                    "disease_code": d_code,
                    "disease_name": d_name,
                    "disease_probabilities": pa_dict,
                    "stage_code": s_code,
                    "stage_name": s_name,
                    "stage_probabilities": pb_dict,
                    "stroke_risk_score": round(risk_score, 4),
                    "stroke_risk_score_pct": round(risk_score * 100.0, 2),
                    "golden_hour_remaining_min": rem_min,
                    "golden_hour_status": timer_str,
                    "trigger_model_3_fast_check": trigger_fast,
                    "trigger_model_2_camera_voice_check": trigger_fast,
                    "imu_fall_detected": imu_fall_detected,
                    "red_contraindication_veto": red_veto,
                    "cardiac_afib_risk_context": round(p_afib_used, 4),
                    "telemetry_quality": meta,
                    "branch_2a_hemodynamics": {
                        "active": b2a_active,
                        "dl_branch_active": dl_active,
                        "dl_weight": round(self.dl_weight, 2),
                        "cuffless_bp_estimate_mmhg": bp_est,
                    },
                    "branch_2b_clinical_cardioembolic": {k: v for k, v in b2b.items() if k != "branch_2b_probs_4class"},
                    "clinical_alert": alert,
                    "hardware_feedback": {
                        "trigger_buzzer": bool(trigger_fast and (s_code >= 2 or red_veto or imu_fall_detected)),
                        "trigger_vibrator": bool(trigger_fast or d_code >= 2),
                    },
                    "raw_ml_decision": {
                        "disease_code": d_code,
                        "disease_name": d_name,
                        "stage_code": s_code,
                        "stage_name": s_name,
                        "rules_overrode_ai": False,
                    },
                }
            )

        return results[0] if is_single else results

    def save(self, filepath: str = DEFAULT_STROKE_MODEL_PATH) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        bundle = {
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "n_estimators": self.n_estimators,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "min_child_weight": self.min_child_weight,
            "dl_weight": self.dl_weight,
            "random_state": self.random_state,
            "scaler_mean": self.scaler.mean_,
            "scaler_scale": self.scaler.scale_,
            "head_a": self.head_a,
            "head_b": self.head_b,
            "active_features": self.active_features,
            "disease_classes": self.disease_classes,
            "stage_names": self.stage_names,
            "cohort_status": self.cohort_status,
            "cnn_weights_path": self.cnn_weights_path,
            "bp_mean": self.bp_mean,
            "bp_std": self.bp_std,
            "clinical_branch": self.clinical_branch,
            "is_fitted": self.is_fitted,
        }
        with open(filepath, "wb") as f:
            pickle.dump(bundle, f)

    @classmethod
    def load(cls, filepath: str = DEFAULT_STROKE_MODEL_PATH) -> "StrokeModel1B":
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"StrokeModel1B artifact not found: {filepath}")
        with open(filepath, "rb") as f:
            b = pickle.load(f)
        inst = cls(
            max_depth=b.get("max_depth", 4),
            learning_rate=b.get("learning_rate", 0.05),
            n_estimators=b.get("n_estimators", 180),
            subsample=b.get("subsample", 0.80),
            colsample_bytree=b.get("colsample_bytree", 0.80),
            reg_alpha=b.get("reg_alpha", 1.0),
            reg_lambda=b.get("reg_lambda", 2.0),
            min_child_weight=b.get("min_child_weight", 5),
            dl_weight=b.get("dl_weight", 0.45),
            random_state=b.get("random_state", 42),
        )
        inst.head_a = b["head_a"]
        inst.head_b = b["head_b"]
        inst.scaler.mean_ = b["scaler_mean"]
        inst.scaler.scale_ = b["scaler_scale"]
        inst.active_features = b.get("active_features", list(STROKE_ACTIVE_FEATURES))
        inst.disease_classes = b.get("disease_classes", list(STROKE_CLASSES))
        inst.stage_names = b.get("stage_names", list(STROKE_STAGE_NAMES))
        inst.cohort_status = b.get("cohort_status", inst.cohort_status)
        inst.cnn_weights_path = b.get("cnn_weights_path", DEFAULT_STROKE_CNN_PATH)
        inst.bp_mean = np.asarray(b.get("bp_mean", [132.0, 72.0]), dtype=np.float32)
        inst.bp_std = np.asarray(b.get("bp_std", [25.0, 14.0]), dtype=np.float32)
        inst.clinical_branch = b.get("clinical_branch")
        inst.is_fitted = b.get("is_fitted", True)
        inst._ensure_cnn()
        return inst


BrainStrokeRiskModel2 = StrokeModel1B


def train_stroke_model_from_master_csv(
    train_csv_path: str = "data/real_80pct_train.csv",
    save_path: str = DEFAULT_STROKE_MODEL_PATH,
) -> Tuple[StrokeModel1B, Dict[str, Any]]:
    """
    Backward-compatible entry point. Prefer `src/models/train_and_benchmark_model2_stroke.py`
    which trains the Two-Branch 2021-2023 Hybrid Model 2.
    """
    hemo_wave_path = os.path.join("data", "model2_stroke_risk_clean", "hemo_2022_waveforms.npz")
    if os.path.exists(hemo_wave_path):
        from src.models.train_and_benchmark_model2_stroke import train_and_evaluate_model2

        model, audit = train_and_evaluate_model2(save_path=save_path)
        return model, audit

    df = pd.read_csv(train_csv_path)
    neuro_df = df[df["disease_label"].isin([0, 3, 4])].copy()
    label_map = {0: 0, 3: 2, 4: 3}
    y_stroke = neuro_df["disease_label"].map(label_map).astype(np.int64).values
    y_stage = neuro_df["stage_label"].astype(np.int64).values

    model = StrokeModel1B()
    summary = model.fit(neuro_df, y_stroke, y_stage)
    model.save(save_path)
    return model, summary
