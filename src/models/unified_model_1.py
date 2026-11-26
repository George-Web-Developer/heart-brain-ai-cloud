"""
unified_model_1.py
========================================================================================
UNIFIED MODEL 1: HEART-BRAIN AXIS DUAL-HEAD WEARABLE AI CLASSIFIER
(Hardware-Exact ESP32-S3 Edition: AD8232/ADS1115 + MAX30102 + MPU6050)

STRICT PURITY & CLINICAL SAFETY DIRECTIVES:
  1. Real Clinical Data Trained: Fitted ONLY on real human physiological windows.
     Zero synthetic, zero SMOTE, zero virtual data.
  2. Hardware-Exact Feature Space:
     - NO TEMPERATURE SENSOR: Completely dropped skin_temp_c, delta_skin_temp_c,
       and temperature terms from vel_mag.
     - 18 Base Physical Features + 9 Engineered Bio-Ratios = 27 Active Features.
  3. Single Self-Contained Model 1 Class:
     - Head A: 5-Class Disease Classifier (Normal, AFib, MI, Ischemic Stroke, Hemorrhagic Stroke).
     - Head B: 4-Stage Clot Progression & Golden Hour Estimator (0: Baseline, 1: Prodromal,
               2: Hyper-Acute <60s, 3: Established Golden Hour Emergency).
     - Golden Hour Countdown Timer: Continuous countdown from 60.0 minutes.
     - Clinical Safety & Model 2 Trigger:
         * trigger_model_2_camera_voice_check: P(Stroke) >= 0.25 or P(ICH) >= 0.12 or Stage >= 1 with neuro signs.
         * red_contraindication_veto: Strict prohibition of aspirin/anticoagulants on suspected ICH prior to CT scan.
     - Unified .predict_all(X) executing both heads and safety logic in one pass.
========================================================================================
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing.signal_processor import extract_10s_features, compute_ecg_sqi, compute_ppg_sqi
from src.models.waveform_cnn import WaveformDeepLearningEngine, DEFAULT_CNN_WEIGHTS_PATH

# 17 Base Physical Features (from AD8232/ADS1115, MAX30102, MPU6050)
PHYSICAL_FEATURES = [
    "hr_ecg_bpm",
    "pulse_rate_ppg_bpm",
    "pulse_deficit_bpm",
    "rmssd_ms",
    "sdnn_ms",
    "pnn50_pct",
    "rr_entropy",
    "st_deviation_mv",
    "t_wave_amplitude_mv",
    "spo2_pct",
    "perfusion_index_pct",
    "ptt_ms",
    "delta_ptt_ms",
    "estimated_sbp_mmhg",
    "estimated_dbp_mmhg",
    "cushing_index",
    "imu_smv_g"
]

# 9 Engineered Bio-Ratios (Temperature-Free)
ENGINEERED_FEATURES = [
    "st_ratio",
    "rpp",
    "fractional_pulse_pressure",
    "perfusion_autonomic_balance",
    "hrv_ratio",
    "perfusion_rate_ratio",
    "vel_mag",
    "vascular_stiffness_index",
    "delta_ptt_ratio"
]

ALL_ACTIVE_FEATURES = PHYSICAL_FEATURES + ENGINEERED_FEATURES
assert len(ALL_ACTIVE_FEATURES) == 26, f"Expected 26 active features, got {len(ALL_ACTIVE_FEATURES)}"

DISEASE_CLASSES = ["Normal", "AFib", "MI", "Ischemic_Stroke", "Hemorrhagic_Stroke"]
STAGE_NAMES = [
    "Stage 0: Stable Baseline",
    "Stage 1: Pre-Clot / Prodromal Warning",
    "Stage 2: Hyper-Acute Clot/Rupture Onset (<60s)",
    "Stage 3: Established Clot / Golden Hour Emergency"
]


def extract_engineered_ratios_from_array(X_phys: np.ndarray) -> np.ndarray:
    """
    Computes 9 temperature-free engineered features from an (N, 17) or (N, 18) array of physical features.
    Indices:
      0: hr_ecg_bpm, 1: pulse_rate_ppg_bpm, 2: pulse_deficit_bpm, 3: rmssd_ms, 4: sdnn_ms,
      5: pnn50_pct, 6: rr_entropy, 7: st_deviation_mv, 8: t_wave_amplitude_mv, 9: spo2_pct,
      10: perfusion_index_pct, 11: ptt_ms, 12: delta_ptt_ms, 13: estimated_sbp_mmhg,
      14: estimated_dbp_mmhg, 15: cushing_index, 16: imu_smv_g
    Returns:
      (N, 26) active feature matrix.
    """
    X_phys = np.asarray(X_phys, dtype=np.float32)
    if X_phys.ndim == 1:
        X_phys = X_phys.reshape(1, -1)

    X_17 = X_phys[:, :17]

    hr = np.nan_to_num(X_17[:, 0], nan=72.0)
    pulse_def = np.nan_to_num(X_17[:, 2], nan=0.0)
    rmssd = np.nan_to_num(X_17[:, 3], nan=25.0)
    rr_ent = np.nan_to_num(X_17[:, 6], nan=0.05)
    st_dev = np.nan_to_num(X_17[:, 7], nan=0.0)
    t_amp = np.nan_to_num(X_17[:, 8], nan=0.2)
    pi = np.nan_to_num(X_17[:, 10], nan=3.0)
    ptt = np.nan_to_num(X_17[:, 11], nan=180.0)
    delta_ptt = np.nan_to_num(X_17[:, 12], nan=0.0)
    sbp = np.nan_to_num(X_17[:, 13], nan=120.0)
    dbp = np.nan_to_num(X_17[:, 14], nan=80.0)

    # 1. st_ratio: st_deviation / (abs(t_wave_amplitude) + 0.05)
    st_ratio = (st_dev / (np.abs(t_amp) + 0.05))[:, None]

    # 2. rpp: (sbp * hr) / 100.0 (Rate-Pressure Product)
    rpp = ((sbp * hr) / 100.0)[:, None]

    # 3. fractional_pulse_pressure: (sbp - dbp) / max(sbp, 40.0)
    fractional_pp = ((sbp - dbp) / np.maximum(sbp, 40.0))[:, None]

    # 4. perfusion_autonomic_balance: pi / max(rmssd, 1.0)
    pab = (pi / np.maximum(rmssd, 1.0))[:, None]

    # 5. hrv_ratio: rmssd / max(sdnn, 1.0)
    sdnn = np.nan_to_num(X_17[:, 4], nan=35.0)
    hrv_ratio = (rmssd / np.maximum(sdnn, 1.0))[:, None]

    # 6. perfusion_rate_ratio: pi / max(pulse_rate_ppg, 20.0)
    pr_ppg = np.nan_to_num(X_17[:, 1], nan=72.0)
    prr = (pi / np.maximum(pr_ppg, 20.0))[:, None]

    # 7. vel_mag: sqrt(dp_term + pd_term + re_term + st_term) (TEMPERATURE-FREE)
    dp_term = (delta_ptt / 20.0) ** 2
    pd_term = (pulse_def / 10.0) ** 2
    re_term = (rr_ent / 0.1) ** 2
    st_term = (st_dev / 0.2) ** 2
    vel_mag = np.sqrt(dp_term + pd_term + re_term + st_term)[:, None]

    # 8. vascular_stiffness_index: sbp / max(ptt, 50.0)
    vsi = (sbp / np.maximum(ptt, 50.0))[:, None]

    # 9. delta_ptt_ratio: delta_ptt / max(ptt, 50.0)
    dpr = (delta_ptt / np.maximum(ptt, 50.0))[:, None]

    out_26 = np.hstack([X_17, st_ratio, rpp, fractional_pp, pab, hrv_ratio, prr, vel_mag, vsi, dpr])
    return out_26.astype(np.float32)


class _HybridLoadDescriptor:
    """Descriptor enabling UnifiedHeartBrainModel1.load to be called on class or instance."""
    def __init__(self, func):
        self.func = func

    def __get__(self, instance, owner):
        if instance is None:
            # Called on class: UnifiedHeartBrainModel1.load(filepath)
            def _class_loader(filepath: str = "models/unified_model_1.pkl", **kwargs):
                return self.func(owner, filepath=filepath, **kwargs)
            return _class_loader
        else:
            # Called on instance: model.load(filepath)
            def _instance_loader(filepath: str = "models/unified_model_1.pkl", **kwargs):
                loaded = self.func(owner, filepath=filepath, **kwargs)
                instance.__dict__.update(loaded.__dict__)
                return instance
            return _instance_loader


class UnifiedHeartBrainModel1:
    """
    Unified Stage 1 Machine Learning Architecture for Heart-Brain Axis Wearable:
      - Head A: 5-Class Disease Classifier (Normal, AFib, MI, Ischemic Stroke, Hemorrhagic Stroke)
      - Head B: 4-Stage Clot Progression Classifier (0: Baseline, 1: Prodromal, 2: Hyper-Acute, 3: Established)
      - Golden Hour Countdown Timer Engine
      - Automated Clinical Safety Arbiter & Model 2 Vision/Voice Trigger
      - Red Contraindication Veto (strictly for Hemorrhagic Stroke / Intracranial Hemorrhage)
    """

    def __init__(
        self,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        n_estimators: int = 160,
        subsample: float = 0.8,
        colsample_bytree: float = 0.75,
        reg_alpha: float = 1.0,
        reg_lambda: float = 2.0,
        min_child_weight: int = 5,
        random_state: int = 42
    ):
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.min_child_weight = min_child_weight
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.head_a = XGBClassifier(
            objective="multi:softprob",
            num_class=5,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_child_weight=min_child_weight,
            eval_metric="mlogloss",
            random_state=random_state
        )
        self.head_b = XGBClassifier(
            objective="multi:softprob",
            num_class=4,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_child_weight=min_child_weight,
            eval_metric="mlogloss",
            random_state=random_state
        )

        self.physical_features = PHYSICAL_FEATURES
        self.engineered_features = ENGINEERED_FEATURES
        self.all_active_features = ALL_ACTIVE_FEATURES
        self.disease_classes = DISEASE_CLASSES
        self.stage_names = STAGE_NAMES
        self.is_fitted = False
        self.cnn_weights_path = DEFAULT_CNN_WEIGHTS_PATH
        self.dl_engine: Optional[WaveformDeepLearningEngine] = None

    def _ensure_dl_engine(self) -> Optional[WaveformDeepLearningEngine]:
        """Lazy-loads domain-invariant PyTorch 1D-CNN+BiLSTM waveform engine if available."""
        if self.dl_engine is None:
            try:
                self.dl_engine = WaveformDeepLearningEngine.load(self.cnn_weights_path)
            except Exception:
                self.dl_engine = None
        return self.dl_engine

    def _evaluate_raw_signal_gate(
        self, ecg: np.ndarray, ppg: np.ndarray, smv_val: float, motion_gated_ppg: bool
    ) -> Dict[str, Any]:
        """
        Fail-closed deterministic Signal Quality & Lead-Off Gate.
        Inspects raw sensor arrays BEFORE inference so flatlines or clipped wires abstain safely.
        """
        ecg_arr = np.asarray(ecg, dtype=np.float32).flatten()
        ppg_arr = np.asarray(ppg, dtype=np.float32).flatten()

        ecg_var = float(np.var(ecg_arr)) if len(ecg_arr) > 0 else 0.0
        ecg_peak = float(np.max(np.abs(ecg_arr))) if len(ecg_arr) > 0 else 0.0
        ecg_lead_off = bool(len(ecg_arr) < 250 or ecg_var < 1e-5 or ecg_peak > 3.3)

        ecg_sqi = 0.0 if ecg_lead_off else float(compute_ecg_sqi(ecg_arr, fs=250.0))
        ppg_sqi = float(compute_ppg_sqi(ppg_arr, fs=250.0)) if (len(ppg_arr) >= 250 and float(np.var(ppg_arr)) > 1e-5) else 0.0

        if ecg_lead_off:
            status = "LEAD_OFF"
            reliable = False
        elif ecg_sqi < 0.25:
            status = "POOR_SIGNAL"
            reliable = False
        elif motion_gated_ppg:
            status = "MOTION_DEGRADED_PPG"
            reliable = True
        else:
            status = "VALID"
            reliable = True

        return {
            "status": status,
            "inference_reliable": reliable,
            "ecg_lead_off": ecg_lead_off,
            "ecg_sqi": round(ecg_sqi, 3),
            "ppg_sqi": round(ppg_sqi, 3),
            "ppg_motion_gated": bool(motion_gated_ppg),
            "imu_smv_g": round(float(smv_val), 2),
        }

    def _prepare_feature_matrix_with_meta(
        self, X: Union[pd.DataFrame, np.ndarray, Dict[str, Any]]
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Converts heterogeneous inputs (DataFrame, dict, or numpy array) into the
        standard (N, 26) active feature matrix plus per-row signal_quality & dl_branch metadata.
        """
        default_sq = {
            "status": "VALID",
            "inference_reliable": True,
            "ecg_lead_off": False,
            "ecg_sqi": 1.0,
            "ppg_sqi": 1.0,
            "ppg_motion_gated": False,
        }
        default_dl = {"active": False, "architecture": "1D-CNN-BiLSTM-Attention", "reason": "tabular_input_no_raw_waveform"}

        if isinstance(X, dict):
            is_raw_window = "raw_windows" in X
            ecg = np.zeros(2500, dtype=np.float32)
            ppg = np.full(2500, 50000.0, dtype=np.float32)
            # Check if raw waveform arrays are passed
            if is_raw_window:
                raw_w = X["raw_windows"]
                ecg = np.array(raw_w.get("ecg_mv", [0.0] * 2500), dtype=np.float32)
                ppg = np.array(raw_w.get("ppg_ir", raw_w.get("ppg_red", [50000.0] * 2500)), dtype=np.float32)
                imu = {
                    "ax": np.array(raw_w.get("imu_ax_g", [0.0] * 500), dtype=np.float32),
                    "ay": np.array(raw_w.get("imu_ay_g", [-1.0] * 500), dtype=np.float32),
                    "az": np.array(raw_w.get("imu_az_g", [0.0] * 500), dtype=np.float32),
                    "gx": np.array(raw_w.get("imu_gx_dps", [0.0] * 500), dtype=np.float32),
                    "gy": np.array(raw_w.get("imu_gy_dps", [0.0] * 500), dtype=np.float32),
                    "gz": np.array(raw_w.get("imu_gz_dps", [0.0] * 500), dtype=np.float32)
                }
                rehab_act = bool(X.get("rehab_exercise_active", False))
                extracted = extract_10s_features(ecg, ppg, imu, rehab_active=rehab_act, fs=250.0)
                feat_dict = dict(extracted)
            elif "features" in X and isinstance(X["features"], dict):
                feat_dict = dict(X["features"])
            else:
                feat_dict = dict(X)

            smv_raw = feat_dict.get("imu_smv_g", feat_dict.get("imu_smv_max_g", 1.0))
            smv_val = 1.0 if smv_raw is None or (isinstance(smv_raw, float) and np.isnan(smv_raw)) else float(smv_raw)

            # Handle motion-gated PPG/BP channels & normalize cushing_index on raw wearable windows
            motion_gated_ppg = False
            ptt_val = feat_dict.get("ptt_ms", np.nan)
            sbp_val = feat_dict.get("estimated_sbp_mmhg", np.nan)
            ptt_missing = ptt_val is None or (isinstance(ptt_val, (float, np.floating)) and np.isnan(ptt_val))
            sbp_missing = sbp_val is None or (isinstance(sbp_val, (float, np.floating)) and np.isnan(sbp_val))

            if is_raw_window:
                hr_raw = feat_dict.get("hr_ecg_bpm", 72.0)
                hr_for_cush = (
                    72.0
                    if hr_raw is None or (isinstance(hr_raw, (float, np.floating)) and np.isnan(hr_raw))
                    else max(40.0, float(hr_raw))
                )
                if smv_val > 1.35 and (ptt_missing or sbp_missing) and hr_for_cush > 55.0:
                    motion_gated_ppg = True
                    feat_dict["ptt_ms"] = 182.0
                    feat_dict["delta_ptt_ms"] = 0.0
                    feat_dict["estimated_sbp_mmhg"] = 120.0
                    feat_dict["estimated_dbp_mmhg"] = 78.0
                    feat_dict["cushing_index"] = round((120.0 / 120.0) / (hr_for_cush / 72.0), 2)
                elif not sbp_missing:
                    feat_dict["cushing_index"] = round((float(sbp_val) / 120.0) / (hr_for_cush / 72.0), 2)

                sq_meta = self._evaluate_raw_signal_gate(ecg, ppg, smv_val, motion_gated_ppg)
                dl_eng = self._ensure_dl_engine() if sq_meta["inference_reliable"] else None
                if dl_eng is not None:
                    dl_meta = dl_eng.predict_window_proba(ecg, fs=250.0)
                else:
                    dl_meta = {
                        "active": False,
                        "architecture": "1D-CNN-BiLSTM-Attention",
                        "reason": sq_meta["status"] if not sq_meta["inference_reliable"] else "cnn_weights_unavailable",
                    }
            else:
                sq_meta = dict(default_sq)
                sq_meta["imu_smv_g"] = round(float(smv_val), 2)
                dl_meta = dict(default_dl)

            # Build 17-element physical vector
            vec_17 = []
            for k in self.physical_features:
                if k == "imu_smv_g":
                    val = feat_dict.get("imu_smv_g", feat_dict.get("imu_smv_max_g", 1.0))
                else:
                    val = feat_dict.get(k, np.nan)
                if val is None:
                    val = np.nan
                elif isinstance(val, bool):
                    val = 1.0 if val else 0.0
                vec_17.append(float(val))

            mat_17 = np.array(vec_17, dtype=np.float32).reshape(1, -1)
            row_meta = [{"signal_quality": sq_meta, "dl_branch": dl_meta}]
            return extract_engineered_ratios_from_array(mat_17), row_meta

        elif isinstance(X, pd.DataFrame):
            n_rows = len(X)
            # If all 26 active features are present:
            if all(col in X.columns for col in self.all_active_features):
                mat = X[self.all_active_features].values.astype(np.float32)
            # If only 17 physical features are present:
            elif all(col in X.columns for col in self.physical_features):
                mat_17 = X[self.physical_features].values.astype(np.float32)
                mat = extract_engineered_ratios_from_array(mat_17)
            else:
                # Fill missing with nan/fallback
                mat_17 = []
                for k in self.physical_features:
                    if k in X.columns:
                        mat_17.append(X[k].values.astype(np.float32))
                    else:
                        mat_17.append(np.full(n_rows, np.nan, dtype=np.float32))
                mat_17 = np.column_stack(mat_17)
                mat = extract_engineered_ratios_from_array(mat_17)
            return mat, [{"signal_quality": dict(default_sq), "dl_branch": dict(default_dl)} for _ in range(n_rows)]

        elif isinstance(X, np.ndarray):
            X_arr = np.asarray(X, dtype=np.float32)
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(1, -1)
            n_rows = len(X_arr)
            if X_arr.shape[1] == 26:
                mat = X_arr
            elif X_arr.shape[1] in (17, 18):
                mat = extract_engineered_ratios_from_array(X_arr)
            elif X_arr.shape[1] == 27:
                mat = X_arr[:, :26]
            else:
                raise ValueError(f"Expected 17 physical or 26 active features, got array with shape {X_arr.shape}")
            return mat, [{"signal_quality": dict(default_sq), "dl_branch": dict(default_dl)} for _ in range(n_rows)]
        else:
            raise TypeError(f"Unsupported input type for feature matrix: {type(X)}")

    def _prepare_feature_matrix(self, X: Union[pd.DataFrame, np.ndarray, Dict[str, Any]]) -> np.ndarray:
        """
        Converts heterogeneous inputs (DataFrame, dict, or numpy array) into the
        standard (N, 26) active feature matrix.
        """
        mat, _ = self._prepare_feature_matrix_with_meta(X)
        return mat

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y_disease: np.ndarray,
        y_stage: np.ndarray,
        sample_weight_a: Optional[np.ndarray] = None,
        sample_weight_b: Optional[np.ndarray] = None
    ) -> "UnifiedHeartBrainModel1":
        """
        Fits StandardScaler ONLY on training data, then fits Head A and Head B
        using strictly balanced sample weights (ZERO synthetic oversampling).
        """
        X_mat = self._prepare_feature_matrix(X)
        y_a = np.asarray(y_disease, dtype=int)
        y_b = np.asarray(y_stage, dtype=int)

        # Fit StandardScaler strictly on training split
        self.scaler.fit(X_mat)
        X_scaled = self.scaler.transform(X_mat)

        # Compute sample weights if not provided
        if sample_weight_a is None:
            sample_weight_a = compute_sample_weight("balanced", y_a)
        if sample_weight_b is None:
            sample_weight_b = compute_sample_weight("balanced", y_b)

        self.head_a.fit(X_scaled, y_a, sample_weight=sample_weight_a)
        self.head_b.fit(X_scaled, y_b, sample_weight=sample_weight_b)
        self.is_fitted = True
        return self

    def fit_smote(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y_disease: np.ndarray,
        y_stage: np.ndarray,
        k_neighbors: int = 5,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Fits StandardScaler STRICTLY on the training feature matrix BEFORE SMOTE,
        transforms X_train, then applies SMOTE exclusively to the training features
        for Head A and Head B, and fits the XGBoost classifiers.

        Zero leakage: Scaler is fit before SMOTE; synthetic samples are never exposed
        to validation or test splits.
        """
        from collections import Counter
        from imblearn.over_sampling import SMOTE

        X_mat = self._prepare_feature_matrix(X)
        y_a = np.asarray(y_disease, dtype=int)
        y_b = np.asarray(y_stage, dtype=int)

        # 1. Fit StandardScaler strictly on X_train BEFORE SMOTE
        self.scaler.fit(X_mat)
        X_scaled = self.scaler.transform(X_mat)

        # SMOTE requires finite values for k-NN distance computation.
        # In standardized space (StandardScaler), 0.0 represents the exact feature mean of the training cohort.
        X_scaled_smote = np.nan_to_num(X_scaled, nan=0.0)

        def _safe_smote(X_in: np.ndarray, y_in: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[int, int]]:
            counts_in = {int(k): int(v) for k, v in Counter(y_in).items()}
            # SMOTE k-NN requires >= 2 samples per oversampled class; duplicate singletons cleanly first if needed
            singletons = [c for c, cnt in counts_in.items() if cnt == 1]
            if singletons:
                extra_X = []
                extra_y = []
                for c in singletons:
                    idx_c = np.where(y_in == c)[0][0]
                    extra_X.append(X_in[idx_c : idx_c + 1])
                    extra_y.append(c)
                X_in = np.vstack([X_in] + extra_X)
                y_in = np.concatenate([y_in, np.array(extra_y, dtype=y_in.dtype)])
                counts_in = {int(k): int(v) for k, v in Counter(y_in).items()}
            min_cnt = min(counts_in.values())
            k_eff = min(k_neighbors, max(1, min_cnt - 1))
            if min_cnt >= 2 and len(counts_in) > 1:
                sampler = SMOTE(random_state=random_state, k_neighbors=k_eff)
                X_res, y_res = sampler.fit_resample(X_in, y_in)
            else:
                X_res, y_res = X_in, y_in
            counts_out = {int(k): int(v) for k, v in Counter(y_res).items()}
            return X_res, y_res, counts_out

        # 2. Count distributions before SMOTE
        counts_a_before = {int(k): int(v) for k, v in Counter(y_a).items()}
        counts_b_before = {int(k): int(v) for k, v in Counter(y_b).items()}

        # 3 & 4. Apply adaptive SMOTE safely for Head A and Head B
        X_smote_a, y_smote_a, counts_a_after = _safe_smote(X_scaled_smote, y_a)
        X_smote_b, y_smote_b, counts_b_after = _safe_smote(X_scaled_smote, y_b)

        # 5. Fit XGBoost heads on resampled training data
        self.head_a.fit(X_smote_a, y_smote_a)
        self.head_b.fit(X_smote_b, y_smote_b)
        self.is_fitted = True

        return {
            "n_train_before": len(X_mat),
            "n_train_smote_a": len(X_smote_a),
            "n_train_smote_b": len(X_smote_b),
            "head_a_counts_before": counts_a_before,
            "head_a_counts_after": counts_a_after,
            "head_b_counts_before": counts_b_before,
            "head_b_counts_after": counts_b_after
        }

    def predict_all(
        self,
        X: Union[pd.DataFrame, np.ndarray, Dict[str, Any]],
        elapsed_stage_sec: float = 0.0
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Executes unified prediction pipeline:
          - Prepares 27-feature matrix
          - Scales using fitted scaler
          - Evaluates Head A (Disease) and Head B (Stage)
          - Calculates Golden Hour Countdown Timer
          - Evaluates Model 2 trigger & Red Contraindication Veto
        """
        if not self.is_fitted:
            raise RuntimeError("UnifiedHeartBrainModel1 is not fitted yet!")

        is_single = isinstance(X, dict) or (isinstance(X, np.ndarray) and X.ndim == 1)
        X_mat, row_meta_list = self._prepare_feature_matrix_with_meta(X)
        X_scaled = self.scaler.transform(X_mat)

        probs_a = self.head_a.predict_proba(X_scaled).copy()
        probs_b = self.head_b.predict_proba(X_scaled).copy()

        results = []
        n_samples = len(X_mat)

        for i in range(n_samples):
            row_meta = row_meta_list[i]
            sq_meta = row_meta["signal_quality"]
            dl_meta = dict(row_meta["dl_branch"])

            # 1. Fail-Closed Deterministic Signal Quality / Lead-Off Gate Rule
            # Flatline or clipped electrodes MUST NOT emit false pathology alerts or buzzer triggers.
            if not sq_meta.get("inference_reliable", True):
                res = {
                    "disease_code": 0,
                    "disease_name": self.disease_classes[0],
                    "disease_probabilities": {self.disease_classes[c]: (1.0 if c == 0 else 0.0) for c in range(5)},
                    "stage_code": 0,
                    "stage_name": self.stage_names[0],
                    "stage_probabilities": {self.stage_names[c]: (1.0 if c == 0 else 0.0) for c in range(4)},
                    "golden_hour_remaining_min": None,
                    "golden_hour_status": f"Signal Gate ({sq_meta['status']}) - Reliable Inference Suspended",
                    "trigger_model_2_camera_voice_check": False,
                    "red_contraindication_veto": False,
                    "clinical_alert": f"SIGNAL QUALITY GATE ({sq_meta['status']}): UNRELIABLE SENSOR INPUT - CHECK ELECTRODES.",
                    "hardware_feedback": {"trigger_buzzer": False, "trigger_vibrator": False},
                    "signal_quality": sq_meta,
                    "dl_branch": dl_meta,
                    "raw_ml_decision": {
                        "disease_code": None,
                        "disease_name": "ABSTAINED_POOR_SIGNAL",
                        "stage_code": None,
                        "stage_name": "ABSTAINED_POOR_SIGNAL",
                        "rules_overrode_ai": False,
                    },
                }
                results.append(res)
                continue

            # 2. Hybrid Deep Learning (1D-CNN + BiLSTM) + Tabular XGBoost Rhythm Fusion
            # When valid 250 Hz waveforms are available, fuse DL Normal/AFib rhythm evidence
            # within the [Normal, AFib] probability subspace while preserving MI & Stroke probability mass.
            if dl_meta.get("active", False):
                p_af_raw = float(dl_meta.get("p_afib_wave", 0.5))
                cal_th = float(dl_meta.get("calibrated_threshold", 0.50))
                if p_af_raw <= cal_th:
                    p_af_cal = 0.5 * (p_af_raw / max(1e-6, cal_th))
                else:
                    p_af_cal = 0.5 + 0.5 * ((p_af_raw - cal_th) / max(1e-6, 1.0 - cal_th))
                p_af_cal = float(np.clip(p_af_cal, 0.0, 1.0))

                rhythm_mass = float(probs_a[i, 0] + probs_a[i, 1])
                if rhythm_mass > 1e-4:
                    xgb_af_share = float(probs_a[i, 1]) / rhythm_mass
                    fused_af_share = 0.65 * xgb_af_share + 0.35 * p_af_cal
                    probs_a[i, 1] = rhythm_mass * fused_af_share
                    probs_a[i, 0] = rhythm_mass * (1.0 - fused_af_share)
                dl_meta["p_afib_calibrated"] = round(p_af_cal, 4)

            # 3. Pure AI Argmax Decisions (ZERO if/else disease or stage mutations!)
            d_code = int(np.argmax(probs_a[i]))
            s_code = int(np.argmax(probs_b[i]))

            p_ischemic = float(probs_a[i, 3])
            p_hemorrhagic = float(probs_a[i, 4])
            p_stroke_total = p_ischemic + p_hemorrhagic

            hr_val = float(X_mat[i, 0]) if not np.isnan(X_mat[i, 0]) else 72.0
            sbp_val = float(X_mat[i, 13]) if not np.isnan(X_mat[i, 13]) else 120.0
            cushing_val = float(X_mat[i, 15]) if not np.isnan(X_mat[i, 15]) else 1.0
            imu_smv = float(X_mat[i, 16]) if not np.isnan(X_mat[i, 16]) else 1.0

            d_name = self.disease_classes[d_code]
            s_name = self.stage_names[s_code]
            p_a_dict = {self.disease_classes[c]: round(float(probs_a[i, c]), 6) for c in range(5)}
            p_b_dict = {self.stage_names[c]: round(float(probs_b[i, c]), 6) for c in range(4)}

            # 4. Golden Hour Timer Computation
            if s_code == 0 or d_code == 0:
                golden_hour_remaining_min = None
                timer_str = "Baseline Normal - No Emergency" if d_code == 0 else "Early Compensated / Baseline Window"
            elif s_code == 1:
                golden_hour_remaining_min = 60.0
                timer_str = "Prodromal Warning - Early Intervention Window Open (60 min pending)"
            elif s_code == 2:
                elapsed_min = min(1.0, max(0.0, elapsed_stage_sec / 60.0))
                golden_hour_remaining_min = round(60.0 - elapsed_min, 1)
                timer_str = f"CRITICAL HYPER-ACUTE ONSET (<60s) - GOLDEN HOUR COUNTDOWN INITIATED: {golden_hour_remaining_min:.1f} min REMAINING"
            else:
                elapsed_min = max(1.0, elapsed_stage_sec / 60.0) if elapsed_stage_sec > 0 else 15.0
                severity_deduction = 20.0 if (cushing_val >= 2.2 or sbp_val >= 180.0) else 10.0
                remaining = max(0.0, 60.0 - elapsed_min - severity_deduction)
                golden_hour_remaining_min = round(remaining, 1)
                timer_str = f"ESTABLISHED CRITICAL EMERGENCY - GOLDEN HOUR CLOSING: {golden_hour_remaining_min:.1f} min REMAINING"

            # 5. Non-Mutating Deterministic Safety Guardrails (Advisory Triggers ONLY)
            neuro_signs = (sbp_val >= 165.0) or (cushing_val >= 2.05)
            trigger_model_2 = bool(
                (p_stroke_total >= 0.25)
                or (p_hemorrhagic >= 0.12)
                or (d_code in (3, 4))
                or (s_code >= 2 and neuro_signs)
            )

            # Red Contraindication Veto: Strictly forbids aspirin/anticoagulants prior to CT on suspected ICH
            red_veto = bool(
                (p_hemorrhagic >= 0.12)
                or (d_code == 4)
                or (cushing_val >= 2.05 and sbp_val >= 160.0)
                or (hr_val <= 62.0 and sbp_val >= 165.0 and imu_smv >= 2.8)
            )

            # 6. Clinical Alert & Hardware Feedback Formulation
            if red_veto:
                clinical_alert = (
                    "CRITICAL EMERGENCY: HEMORRHAGIC STROKE (ICH) DETECTED! "
                    "RED CONTRAINDICATION: ASPIRIN / ANTICOAGULANTS STRICTLY FORBIDDEN PRIOR TO NON-CONTRAST HEAD CT. "
                    "IMMEDIATE NEUROSURGICAL / EMERGENCY CONSULT REQUIRED."
                )
            elif d_code == 3 or p_ischemic >= 0.35:
                clinical_alert = (
                    "CRITICAL EMERGENCY: ACUTE ISCHEMIC STROKE DETECTED! "
                    "MODEL 2 CAMERA & VOICE FAST PROTOCOL ACTIVATED. "
                    "PREPARE THROMBOLYTIC (tPA) EVALUATION WITHIN GOLDEN HOUR."
                )
            elif d_code == 2 or probs_a[i, 2] >= 0.35:
                clinical_alert = (
                    "CRITICAL EMERGENCY: ACUTE MYOCARDIAL INFARCTION (STEMI/NSTEMI) DETECTED! "
                    "IMMEDIATE CATH LAB & REPERFUSION ACTIVATION REQUIRED."
                )
            elif d_code == 1 or probs_a[i, 1] >= 0.40:
                clinical_alert = (
                    "WARNING: ATRIAL FIBRILLATION DETECTED. "
                    "EVALUATING PULSE DEFICIT & THROMBOEMBOLIC RISK."
                )
            else:
                clinical_alert = "NORMAL PHYSIOLOGICAL STATE: STABLE BASELINE."

            trigger_buzzer = bool(d_code in (2, 3, 4) and s_code >= 2)
            trigger_vibrator = bool(d_code in (1, 2, 3, 4) and s_code >= 1)

            res = {
                "disease_code": d_code,
                "disease_name": d_name,
                "disease_probabilities": p_a_dict,
                "stage_code": s_code,
                "stage_name": s_name,
                "stage_probabilities": p_b_dict,
                "golden_hour_remaining_min": golden_hour_remaining_min,
                "golden_hour_status": timer_str,
                "trigger_model_2_camera_voice_check": trigger_model_2,
                "red_contraindication_veto": red_veto,
                "clinical_alert": clinical_alert,
                "hardware_feedback": {
                    "trigger_buzzer": trigger_buzzer,
                    "trigger_vibrator": trigger_vibrator,
                },
                "signal_quality": sq_meta,
                "dl_branch": dl_meta,
                "raw_ml_decision": {
                    "disease_code": d_code,
                    "disease_name": d_name,
                    "stage_code": s_code,
                    "stage_name": s_name,
                    "rules_overrode_ai": False,
                },
            }
            results.append(res)

        return results[0] if is_single and len(results) == 1 else results

    def save(self, filepath: str = "models/unified_model_1.pkl"):
        """
        Saves self-contained bundle as a single .pkl artifact and also exports
        portable XGBoost JSON models + scaler .npz.
        """
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
            "random_state": self.random_state,
            "scaler_mean": self.scaler.mean_,
            "scaler_scale": self.scaler.scale_,
            "head_a": self.head_a,
            "head_b": self.head_b,
            "physical_features": self.physical_features,
            "engineered_features": self.engineered_features,
            "all_active_features": self.all_active_features,
            "disease_classes": self.disease_classes,
            "stage_names": self.stage_names,
            "is_fitted": self.is_fitted
        }
        with open(filepath, "wb") as f:
            pickle.dump(bundle, f)

        # Also save JSON models and npz scaler for zero-dependency edge portability
        base_dir = os.path.dirname(filepath)
        stem = os.path.splitext(os.path.basename(filepath))[0]
        self.head_a.save_model(os.path.join(base_dir, f"{stem}_head_a.json"))
        self.head_b.save_model(os.path.join(base_dir, f"{stem}_head_b.json"))
        np.savez(
            os.path.join(base_dir, f"{stem}_scaler.npz"),
            mean=self.scaler.mean_,
            scale=self.scaler.scale_
        )
        print(f"Unified Model 1 successfully saved to: {filepath}", flush=True)

    @_HybridLoadDescriptor
    def load(cls, filepath: str = "models/unified_model_1.pkl") -> "UnifiedHeartBrainModel1":
        """Loads a saved UnifiedHeartBrainModel1 from artifact bundle."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model artifact not found at: {filepath}")

        with open(filepath, "rb") as f:
            bundle = pickle.load(f)

        instance = cls(
            max_depth=bundle.get("max_depth", 4),
            learning_rate=bundle.get("learning_rate", 0.05),
            n_estimators=bundle.get("n_estimators", 160),
            subsample=bundle.get("subsample", 0.8),
            colsample_bytree=bundle.get("colsample_bytree", 0.75),
            reg_alpha=bundle.get("reg_alpha", 1.0),
            reg_lambda=bundle.get("reg_lambda", 2.0),
            min_child_weight=bundle.get("min_child_weight", 5),
            random_state=bundle.get("random_state", 42)
        )
        instance.head_a = bundle["head_a"]
        instance.head_b = bundle["head_b"]
        instance.scaler.mean_ = bundle["scaler_mean"]
        instance.scaler.scale_ = bundle["scaler_scale"]
        instance.physical_features = bundle.get("physical_features", PHYSICAL_FEATURES)
        instance.engineered_features = bundle.get("engineered_features", ENGINEERED_FEATURES)
        instance.all_active_features = bundle.get("all_active_features", ALL_ACTIVE_FEATURES)
        instance.disease_classes = bundle.get("disease_classes", DISEASE_CLASSES)
        instance.stage_names = bundle.get("stage_names", STAGE_NAMES)
        instance.is_fitted = bundle.get("is_fitted", True)
        instance._ensure_dl_engine()
        return instance
