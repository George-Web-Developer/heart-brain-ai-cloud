"""
cardiac_model.py
========================================================================================
Model 1A: Decoupled Cardiac Rhythm & Ischemic MI Engine
(Strictly 4-Class Cardiac Specialist: Normal [0], AFib [1], Arrhythmia [2], Ischemic_MI [3])

Architecture:
  1. Gain-Invariant & Sampling-Rate-Agnostic ECG Preprocessing (0.5 - 35.0 Hz SOS):
       - Matches AD8232 hardware 0.5 Hz HPF band while remaining below PTB-XL's 50 Hz Nyquist
         so there is zero train-serve skew and zero hospital amplifier-gain leakage.
  2. Deep Learning Branch (`CardiacWaveformCNN4Class` / `CardiacWaveformCNN3Class`):
       - 3-channel normalized waveform tensor [Lead I Z, Lead II Z, QRS Slope-Energy Env]
       - 1D ResNet + Squeeze-and-Excitation + GroupNorm + BiGRU + Attention Pooling
       - Joint multi-task outputs: Head A (4-class disease) & Head B (4-stage cardiac acuity)
       - Trained with random single-lead Lead-II dropout so 1-lead AD8232 hardware works.
  3. Tabular ML Branch (`XGBoost` Head A + Head B):
       - Operates strictly on 23 pure ECG morphological, HRV, and sub-30Hz spectral features
         normalized by QRS amplitude (`qrs_pp`) to eliminate device/hospital gain shortcuts.
  4. Non-Mutating Probability Ensemble, Signal Quality Gate & AD8232 Hardware Guard:
       - Deterministic SQI / Lead-Off gate abstains BEFORE inference on flatlines/clipping.
       - Non-mutating `ischemic_mi_hardware_guard` verifies SQI, IMU motion stability,
         and ST/T/Q morphology while explicitly stating single-lead scope and MAX30003/0.05 Hz path.
       - Zero `if/else` overrides of Head A or Head B argmax (`rules_overrode_ai = False`).
========================================================================================
"""

from __future__ import annotations

import os
import pickle
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import butter, find_peaks, resample_poly, sosfiltfilt
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from xgboost import XGBClassifier

from src.preprocessing.signal_processor import compute_ecg_sqi

TARGET_FS = 250.0
TARGET_SAMPLES = 2500

CARDIAC_CLASSES = ["Normal", "AFib", "Arrhythmia", "Ischemic_MI"]
CARDIAC_STAGE_NAMES = [
    "Stage 0: Stable Baseline",
    "Stage 1: Early / Compensated Cardiac Strain",
    "Stage 2: Acute Arrhythmia / Active Myocardial Injury",
    "Stage 3: Critical High-Risk Cardiac Crisis",
]

# 29 Pure, Temperature-Free, Gain-Normalized ECG Features (identical on 1-lead or 2-lead)
CARDIAC_ECG_FEATURES = [
    "hr_bpm",
    "beat_count_10s",
    "rmssd_ms",
    "sdnn_ms",
    "pnn50_pct",
    "rr_cv",
    "rr_iqr_norm",
    "rr_mad_norm",
    "rr_diff_med_norm",
    "p_wave_presence_ratio",
    "st_j40_rel_l1",
    "st_j80_rel_l1",
    "st_j80_rel_l2",
    "st_abs_max_rel",
    "st_slope_rel",
    "t_wave_rel_l1",
    "t_wave_rel_l2",
    "q_dip_rel_min",
    "qrs_width_ms",
    "spec_low_1_4_pct",
    "spec_mid_4_15_pct",
    "spec_high_15_30_pct",
    "ecg_sqi",
    "rr_min_med_ratio",
    "rr_max_med_ratio",
    "ectopic_beat_frac",
    "qrs_width_iqr_ms",
    "st_lead_diff_rel",
    "t_neg_beat_frac",
]

DEFAULT_CARDIAC_MODEL_PATH = os.path.join("models", "cardiac_model_1a.pkl")
DEFAULT_CARDIAC_CNN_PATH = os.path.join("models", "cardiac_model_1a_cnn.pt")

# Zero-phase SOS bandpass [0.5 - 35.0 Hz] preserves ST elevation/depression while sitting
# strictly below PTB-XL's 100 Hz Nyquist (50 Hz) to eliminate sampling-rate artifacts.
_SOS_BP_250 = butter(3, [0.5, 35.0], btype="bandpass", fs=TARGET_FS, output="sos")


def _safe_bandpass_250(sig: np.ndarray, fs: float = TARGET_FS) -> np.ndarray:
    """Resamples if needed to 250 Hz / 2500 samples and applies zero-phase 0.5-35 Hz bandpass."""
    x = np.asarray(sig, dtype=np.float64).flatten()
    if x.size == 0:
        return np.zeros(TARGET_SAMPLES, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if abs(float(fs) - TARGET_FS) > 1.0 and float(fs) > 10.0:
        num = int(round(TARGET_SAMPLES))
        den = max(1, len(x))
        x = resample_poly(x, num, den)
    if len(x) < TARGET_SAMPLES:
        x = np.pad(x, (0, TARGET_SAMPLES - len(x)), mode="edge")
    elif len(x) > TARGET_SAMPLES:
        x = x[:TARGET_SAMPLES]

    x = x - np.median(x)
    if np.std(x) < 1e-7:
        return np.zeros(TARGET_SAMPLES, dtype=np.float32)
    try:
        bp = sosfiltfilt(_SOS_BP_250, x)
        return np.nan_to_num(bp, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    except Exception:
        return x.astype(np.float32)


def _robust_zscore_clip(sig: np.ndarray, clip_val: float = 8.0) -> np.ndarray:
    """Gain-invariant per-window robust standardization."""
    med = float(np.median(sig))
    mad = float(np.median(np.abs(sig - med)))
    scale = max(1.4826 * mad, float(np.std(sig)) * 0.5, 1e-4)
    z = (sig - med) / scale
    return np.clip(z, -clip_val, clip_val).astype(np.float32)


def _detect_r_peaks_fast(bp_sig: np.ndarray, fs: float = TARGET_FS) -> np.ndarray:
    """Fast gain-invariant QRS detector using absolute slope energy & adaptive percentile thresholding."""
    n = len(bp_sig)
    if n < int(1.0 * fs) or float(np.std(bp_sig)) < 1e-6:
        return np.array([], dtype=np.int64)

    diff = np.diff(bp_sig, prepend=bp_sig[0])
    sq_energy = diff * diff
    win = max(3, int(round(0.08 * fs)))
    kernel = np.ones(win, dtype=np.float64) / float(win)
    mwi = np.convolve(sq_energy, kernel, mode="same")

    q98 = float(np.percentile(mwi, 98))
    med_mwi = float(np.median(mwi))
    hi_thr = max(q98 * 0.28, med_mwi * 4.0, float(np.mean(mwi)) * 0.90, 1e-9)
    min_dist = max(1, int(round(0.25 * fs)))  # Max ~240 BPM physiological limit
    coarse_peaks, _ = find_peaks(mwi, height=hi_thr, prominence=hi_thr * 0.45, distance=min_dist)
    if len(coarse_peaks) == 0:
        return np.array([], dtype=np.int64)

    # Refine each peak to true local R extremum inside clean filtered signal (+/- 48 ms)
    rad = int(round(0.048 * fs))
    refined: List[int] = []
    # Determine dominant QRS polarity from median peak values
    pol_samples = [bp_sig[p] for p in coarse_peaks]
    abs_sig = np.abs(bp_sig)
    for cp in coarse_peaks:
        lo = max(0, int(cp) - rad)
        hi = min(n, int(cp) + rad + 1)
        local_idx = lo + int(np.argmax(abs_sig[lo:hi]))
        if not refined or (local_idx - refined[-1]) >= min_dist:
            refined.append(local_idx)
        elif abs_sig[local_idx] > abs_sig[refined[-1]]:
            refined[-1] = local_idx
    _ = pol_samples
    return np.asarray(refined, dtype=np.int64)


def _extract_single_lead_morph(
    bp: np.ndarray, r_peaks: np.ndarray, fs: float = TARGET_FS
) -> Dict[str, float]:
    """Extracts gain-normalized P/Q/R/ST/T/QRS morphology from a single bandpass lead."""
    n = len(bp)
    defaults = {
        "p_ratio": 0.05,
        "st_j40": 0.0,
        "st_j80": 0.0,
        "st_slope": 0.0,
        "t_wave": 0.15,
        "q_dip": -0.05,
        "qrs_width_ms": 85.0,
        "qrs_width_iqr_ms": 5.0,
        "t_neg_frac": 0.0,
    }
    if len(r_peaks) < 2:
        return defaults

    qrs_amps: List[float] = []
    p_vals: List[float] = []
    st40_vals: List[float] = []
    st80_vals: List[float] = []
    st_slopes: List[float] = []
    t_vals: List[float] = []
    q_dips: List[float] = []
    qrs_widths: List[float] = []

    pre_min = int(round(0.200 * fs))
    post_max = int(round(0.380 * fs))

    for idx, r in enumerate(r_peaks):
        if r < pre_min or r + post_max >= n:
            continue

        # PQ-to-TP linear baseline removes slow repolarization wander
        pq_lo = max(0, r - int(round(0.080 * fs)))
        pq_hi = max(pq_lo + 1, r - int(round(0.045 * fs)))
        iso_before = float(np.median(bp[pq_lo:pq_hi]))
        t_before = 0.5 * (pq_lo + pq_hi)

        tp_lo = r + int(round(0.320 * fs))
        tp_hi = r + int(round(0.380 * fs))
        next_r = int(r_peaks[idx + 1]) if (idx + 1) < len(r_peaks) else n
        if tp_hi < n and (next_r - r) > int(round(0.50 * fs)):
            iso_after = float(np.median(bp[tp_lo:tp_hi]))
            t_after = 0.5 * (tp_lo + tp_hi)
        else:
            iso_after = iso_before
            t_after = t_before + 1.0

        def _iso_at(pos: int) -> float:
            alpha = (pos - t_before) / max(1.0, t_after - t_before)
            alpha = float(np.clip(alpha, 0.0, 1.0))
            return (1.0 - alpha) * iso_before + alpha * iso_after

        qrs_lo = r - int(round(0.045 * fs))
        qrs_hi = r + int(round(0.045 * fs))
        qrs_seg = bp[qrs_lo : qrs_hi + 1]
        qrs_pp = float(np.max(qrs_seg) - np.min(qrs_seg))
        if qrs_pp < 0.05:
            continue
        qrs_amps.append(qrs_pp)

        # Normalize relative to beat sign so inverted lead polarity handles R polarity cleanly
        r_polarity = 1.0 if (bp[r] - iso_before) >= 0 else -1.0

        # QRS FWHM width estimate (samples above 30% of R excursion)
        r_abs = abs(float(bp[r] - iso_before))
        above_30 = np.abs(qrs_seg - iso_before) >= 0.30 * max(1e-4, r_abs)
        qrs_widths.append(float(np.sum(above_30)) * (1000.0 / fs))

        # P-wave window [-190 ms .. -85 ms] RMS deflection normalized by QRS peak-to-peak
        p_lo = r - int(round(0.190 * fs))
        p_hi = r - int(round(0.085 * fs))
        p_seg = bp[p_lo:p_hi] - _iso_at(int(round(0.5 * (p_lo + p_hi))))
        p_vals.append(float(np.max(np.abs(p_seg))) / qrs_pp)

        # Pathological Q-wave dip [-42 ms .. -14 ms] before R (oriented with positive R)
        q_lo = r - int(round(0.042 * fs))
        q_hi = r - int(round(0.014 * fs))
        q_seg = r_polarity * (bp[q_lo : q_hi + 1] - _iso_at(q_lo))
        q_dips.append(float(np.min(q_seg)) / qrs_pp)

        # J+40ms and J+80ms ST shifts normalized by QRS amplitude
        j40 = r + int(round(0.040 * fs))
        j80 = r + int(round(0.080 * fs))
        st40 = r_polarity * (float(bp[j40]) - _iso_at(j40)) / qrs_pp
        st80 = r_polarity * (float(bp[j80]) - _iso_at(j80)) / qrs_pp
        st40_vals.append(st40)
        st80_vals.append(st80)
        st_slopes.append(st80 - st40)

        # T-wave signed peak [+110 ms .. +290 ms] post-R (negative -> T-wave inversion)
        tw_lo = r + int(round(0.110 * fs))
        tw_hi = min(n, r + int(round(0.290 * fs)))
        t_seg = r_polarity * (bp[tw_lo:tw_hi] - _iso_at(int(round(0.5 * (tw_lo + tw_hi)))))
        if len(t_seg) > 0:
            peak_idx = int(np.argmax(np.abs(t_seg)))
            t_vals.append(float(t_seg[peak_idx]) / qrs_pp)

    if not qrs_amps:
        return defaults

    if len(qrs_widths) >= 2:
        qw_q75, qw_q25 = np.percentile(qrs_widths, [75, 25])
        qrs_width_iqr = float(np.clip(qw_q75 - qw_q25, 0.0, 120.0))
    else:
        qrs_width_iqr = 5.0
    t_neg_frac = (
        float(np.clip(np.mean(np.asarray(t_vals, dtype=np.float64) < -0.08), 0.0, 1.0))
        if t_vals
        else 0.0
    )

    return {
        "p_ratio": float(np.clip(np.median(p_vals), 0.0, 0.6)),
        "st_j40": float(np.clip(np.median(st40_vals), -0.8, 0.8)),
        "st_j80": float(np.clip(np.median(st80_vals), -0.8, 0.8)),
        "st_slope": float(np.clip(np.median(st_slopes), -0.6, 0.6)),
        "t_wave": float(np.clip(np.median(t_vals), -0.9, 1.1)) if t_vals else 0.15,
        "q_dip": float(np.clip(np.median(q_dips), -0.9, 0.2)) if q_dips else -0.05,
        "qrs_width_ms": float(np.clip(np.median(qrs_widths), 35.0, 220.0)),
        "qrs_width_iqr_ms": qrs_width_iqr,
        "t_neg_frac": t_neg_frac,
    }


def extract_cardiac_ecg_features_from_waveform(
    lead1: np.ndarray,
    lead2: Optional[np.ndarray] = None,
    fs: float = TARGET_FS,
) -> Dict[str, float]:
    """
    Extracts the exact 29 gain-normalized, temperature-free pure-ECG biomarkers
    from a 10s ECG window (single-lead AD8232 or dual-lead clinical ECG).
    """
    bp1 = _safe_bandpass_250(lead1, fs=fs)
    if lead2 is not None and len(lead2) >= 250 and float(np.var(lead2)) > 1e-6:
        bp2 = _safe_bandpass_250(lead2, fs=fs)
    else:
        bp2 = bp1

    # 1. QRS / RR irregularity & ectopic beat extraction
    r_peaks = _detect_r_peaks_fast(bp1, fs=TARGET_FS)
    if len(r_peaks) < 3:
        r_peaks2 = _detect_r_peaks_fast(bp2, fs=TARGET_FS)
        if len(r_peaks2) > len(r_peaks):
            r_peaks = r_peaks2

    beat_count = float(len(r_peaks))
    if len(r_peaks) >= 3:
        rr_ms = np.diff(r_peaks) * (1000.0 / TARGET_FS)
        valid_rr = rr_ms[(rr_ms >= 250.0) & (rr_ms <= 2000.0)]
        if len(valid_rr) >= 2:
            med_rr = float(np.median(valid_rr))
            mean_rr = float(np.mean(valid_rr))
            hr_bpm = float(np.clip(60000.0 / max(240.0, med_rr), 25.0, 220.0))
            sdnn_ms = float(np.std(valid_rr))
            diffs = np.diff(valid_rr)
            rmssd_ms = float(np.sqrt(np.mean(diffs * diffs))) if len(diffs) > 0 else 20.0
            pnn50_pct = (
                float(np.sum(np.abs(diffs) > 50.0) / max(1, len(diffs)) * 100.0)
                if len(diffs) > 0
                else 0.0
            )
            rr_cv = float(sdnn_ms / max(240.0, mean_rr))
            q75, q25 = np.percentile(valid_rr, [75, 25])
            rr_iqr_norm = float((q75 - q25) / max(240.0, med_rr))
            rr_mad_norm = float(np.median(np.abs(valid_rr - med_rr)) / max(240.0, med_rr))
            rr_diff_med_norm = (
                float(np.median(np.abs(diffs)) / max(240.0, med_rr)) if len(diffs) > 0 else 0.03
            )
            rr_min_med_ratio = float(np.clip(np.min(valid_rr) / max(240.0, med_rr), 0.25, 1.0))
            rr_max_med_ratio = float(np.clip(np.max(valid_rr) / max(240.0, med_rr), 1.0, 3.0))
            ectopic_beat_frac = float(
                np.clip(np.mean((np.abs(valid_rr - med_rr) / max(240.0, med_rr)) > 0.20), 0.0, 1.0)
            )
        else:
            hr_bpm, rmssd_ms, sdnn_ms, pnn50_pct = 72.0, 22.0, 26.0, 5.0
            rr_cv, rr_iqr_norm, rr_mad_norm, rr_diff_med_norm = 0.04, 0.04, 0.02, 0.02
            rr_min_med_ratio, rr_max_med_ratio, ectopic_beat_frac = 0.95, 1.05, 0.0
    else:
        hr_bpm, rmssd_ms, sdnn_ms, pnn50_pct = 72.0, 22.0, 26.0, 5.0
        rr_cv, rr_iqr_norm, rr_mad_norm, rr_diff_med_norm = 0.04, 0.04, 0.02, 0.02
        rr_min_med_ratio, rr_max_med_ratio, ectopic_beat_frac = 0.95, 1.05, 0.0

    # 2. Gain-normalized P / Q / ST / T morphology across Lead I & Lead II
    m1 = _extract_single_lead_morph(bp1, r_peaks, fs=TARGET_FS)
    m2 = _extract_single_lead_morph(bp2, r_peaks, fs=TARGET_FS) if bp2 is not bp1 else m1

    # 3. Sub-30 Hz relative spectral energy ratios (strictly invariant to 100 vs 500 Hz sources)
    z1 = bp1 / max(1e-5, float(np.std(bp1)))
    fft_pow = np.abs(np.fft.rfft(z1)) ** 2
    freqs = np.fft.rfftfreq(len(z1), d=1.0 / TARGET_FS)
    total_pow = max(1e-9, float(np.sum(fft_pow[(freqs >= 0.5) & (freqs <= 30.0)])))
    spec_low = float(np.sum(fft_pow[(freqs >= 1.0) & (freqs < 4.0)]) / total_pow)
    spec_mid = float(np.sum(fft_pow[(freqs >= 4.0) & (freqs < 15.0)]) / total_pow)
    spec_high = float(np.sum(fft_pow[(freqs >= 15.0) & (freqs <= 30.0)]) / total_pow)

    sqi1 = float(compute_ecg_sqi(bp1, fs=TARGET_FS, causal=False))

    return {
        "hr_bpm": round(hr_bpm, 2),
        "beat_count_10s": round(beat_count, 1),
        "rmssd_ms": round(float(np.clip(rmssd_ms, 0.0, 600.0)), 2),
        "sdnn_ms": round(float(np.clip(sdnn_ms, 0.0, 500.0)), 2),
        "pnn50_pct": round(float(np.clip(pnn50_pct, 0.0, 100.0)), 2),
        "rr_cv": round(float(np.clip(rr_cv, 0.0, 1.0)), 5),
        "rr_iqr_norm": round(float(np.clip(rr_iqr_norm, 0.0, 1.0)), 5),
        "rr_mad_norm": round(float(np.clip(rr_mad_norm, 0.0, 1.0)), 5),
        "rr_diff_med_norm": round(float(np.clip(rr_diff_med_norm, 0.0, 1.0)), 5),
        "p_wave_presence_ratio": round(0.5 * (m1["p_ratio"] + m2["p_ratio"]), 5),
        "st_j40_rel_l1": round(m1["st_j40"], 5),
        "st_j80_rel_l1": round(m1["st_j80"], 5),
        "st_j80_rel_l2": round(m2["st_j80"], 5),
        "st_abs_max_rel": round(max(abs(m1["st_j80"]), abs(m2["st_j80"])), 5),
        "st_slope_rel": round(0.5 * (m1["st_slope"] + m2["st_slope"]), 5),
        "t_wave_rel_l1": round(m1["t_wave"], 5),
        "t_wave_rel_l2": round(m2["t_wave"], 5),
        "q_dip_rel_min": round(min(m1["q_dip"], m2["q_dip"]), 5),
        "qrs_width_ms": round(0.5 * (m1["qrs_width_ms"] + m2["qrs_width_ms"]), 2),
        "spec_low_1_4_pct": round(spec_low, 5),
        "spec_mid_4_15_pct": round(spec_mid, 5),
        "spec_high_15_30_pct": round(spec_high, 5),
        "ecg_sqi": round(sqi1, 4),
        "rr_min_med_ratio": round(rr_min_med_ratio, 5),
        "rr_max_med_ratio": round(rr_max_med_ratio, 5),
        "ectopic_beat_frac": round(ectopic_beat_frac, 5),
        "qrs_width_iqr_ms": round(0.5 * (m1["qrs_width_iqr_ms"] + m2["qrs_width_iqr_ms"]), 2),
        "st_lead_diff_rel": round(float(np.clip(m1["st_j80"] - m2["st_j80"], -1.2, 1.2)), 5),
        "t_neg_beat_frac": round(0.5 * (m1["t_neg_frac"] + m2["t_neg_frac"]), 5),
    }


def extract_cardiac_feature_batch(waveforms: np.ndarray, fs: float = TARGET_FS) -> pd.DataFrame:
    """
    Batch extractor for `(N, 2, 2500)` or `(N, 2500)` waveforms -> DataFrame with `CARDIAC_ECG_FEATURES`.
    """
    w = np.asarray(waveforms, dtype=np.float32)
    if w.ndim == 2:
        w = w[:, np.newaxis, :]
    rows: List[Dict[str, float]] = []
    has_lead2 = w.shape[1] >= 2
    for i in range(len(w)):
        l1 = w[i, 0]
        l2 = w[i, 1] if has_lead2 else None
        rows.append(extract_cardiac_ecg_features_from_waveform(l1, l2, fs=fs))
    return pd.DataFrame(rows, columns=CARDIAC_ECG_FEATURES)


def assign_cardiac_acuity_stage(
    disease_labels: np.ndarray, feat_df: pd.DataFrame
) -> np.ndarray:
    """
    Assigns uniform ESC/AHA physiological ECG severity stages (0..3) across 4 cardiac classes:
      - Stage 0: Stable Baseline (`Normal`)
      - Stage 1: Early / Compensated Cardiac Strain (controlled AFib / mild sinus arrhythmia / chronic scar)
      - Stage 2: Acute Arrhythmia / Active Myocardial Injury
      - Stage 3: Critical High-Risk Cardiac Crisis (marked STEMI |ST| >= 0.22, severe T-inversion + ST, HR >= 125 or <= 48)
    """
    dis = np.asarray(disease_labels, dtype=np.int64)
    st_max = feat_df["st_abs_max_rel"].astype(float).values
    hr = feat_df["hr_bpm"].astype(float).values
    t_min = np.minimum(feat_df["t_wave_rel_l1"].astype(float).values, feat_df["t_wave_rel_l2"].astype(float).values)
    q_min = feat_df["q_dip_rel_min"].astype(float).values
    rr_cv = feat_df["rr_cv"].astype(float).values

    stages = np.zeros(len(dis), dtype=np.int64)
    active = dis != 0

    stg3 = active & (
        (st_max >= 0.22)
        | (hr >= 125.0)
        | (hr <= 48.0)
        | ((st_max >= 0.12) & (t_min <= -0.18))
    )
    stg2 = active & (~stg3) & (
        (st_max >= 0.12)
        | (t_min <= -0.14)
        | (q_min <= -0.25)
        | ((dis == 1) & ((hr >= 100.0) | (rr_cv >= 0.20)))
        | ((dis == 2) & ((hr >= 110.0) | (hr <= 52.0) | (rr_cv >= 0.18)))
        | ((dis == 3) & (st_max >= 0.08))
    )
    stg1 = active & (~stg3) & (~stg2)

    stages[stg3] = 3
    stages[stg2] = 2
    stages[stg1] = 1
    return stages


def build_cardiac_cnn_channels(waves: np.ndarray, fs: float = TARGET_FS) -> np.ndarray:
    """
    Converts `(N, 2, L)` or `(N, L)` ECG waveforms into `(N, 3, 2500)` gain-agnostic CNN inputs:
      - Ch 0: 0.5-35 Hz Z-scored Lead I
      - Ch 1: 0.5-35 Hz Z-scored Lead II (or Lead I duplicate if single-lead hardware)
      - Ch 2: Smoothed QRS instantaneous slope-energy envelope (beat regularity invariance)
    """
    w = np.asarray(waves, dtype=np.float32)
    if w.ndim == 1:
        w = w[np.newaxis, np.newaxis, :]
    elif w.ndim == 2:
        w = w[:, np.newaxis, :]

    n = w.shape[0]
    has_l2 = w.shape[1] >= 2
    out = np.zeros((n, 3, TARGET_SAMPLES), dtype=np.float32)
    win = int(round(0.060 * TARGET_FS))
    kernel = np.ones(win, dtype=np.float32) / float(win)

    for i in range(n):
        bp1 = _safe_bandpass_250(w[i, 0], fs=fs)
        z1 = _robust_zscore_clip(bp1, clip_val=8.0)

        if has_l2 and float(np.var(w[i, 1])) > 1e-6:
            bp2 = _safe_bandpass_250(w[i, 1], fs=fs)
            z2 = _robust_zscore_clip(bp2, clip_val=8.0)
        else:
            z2 = z1.copy()

        d1 = np.diff(z1, prepend=z1[0])
        env = np.convolve(np.sqrt(d1 * d1 + 1e-6), kernel, mode="same")
        e_mean = float(np.mean(env))
        e_std = max(1e-4, float(np.std(env)))
        z_env = np.clip((env - e_mean) / e_std, -3.0, 8.0).astype(np.float32)

        out[i, 0] = z1
        out[i, 1] = z2
        out[i, 2] = z_env

    return out


class _CardiacAugDataset(Dataset):
    """Training/validation dataset with domain-invariant ECG & single-lead dropout augmentations."""

    def __init__(
        self,
        channels: np.ndarray,
        y_disease: np.ndarray,
        y_stage: np.ndarray,
        augment: bool = False,
        seed: int = 42,
    ):
        channels_f32 = (
            channels
            if (isinstance(channels, np.ndarray) and channels.dtype == np.float32 and channels.flags["C_CONTIGUOUS"])
            else np.ascontiguousarray(channels, dtype=np.float32)
        )
        self.channels = torch.from_numpy(channels_f32)
        self.y_disease = torch.from_numpy(y_disease.astype(np.int64))
        self.y_stage = torch.from_numpy(y_stage.astype(np.int64))
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.y_disease)

    def __getitem__(self, idx: int):
        x = self.channels[idx].clone()
        if self.augment:
            # Random temporal circular shift (up to +/- 1.5s) breaks boundary memorization
            shift = int(self.rng.integers(-375, 376))
            if shift != 0:
                x = torch.roll(x, shifts=shift, dims=-1)

            # Random gain/amplitude perturbation & subtle EMG micro-noise
            gain = float(self.rng.uniform(0.75, 1.30))
            x[:2] = x[:2] * gain
            noise_std = float(self.rng.uniform(0.0, 0.05))
            if noise_std > 0:
                x[:2] = x[:2] + torch.randn_like(x[:2]) * noise_std

            # Single-lead hardware simulation (40% of training items duplicate Ch0 -> Ch1)
            if self.rng.random() < 0.40:
                x[1] = x[0] + torch.randn_like(x[0]) * 0.02

            # Random low-frequency respiratory baseline wander (~0.25 Hz)
            if self.rng.random() < 0.50:
                t = torch.linspace(0.0, 10.0, TARGET_SAMPLES)
                phase = float(self.rng.uniform(0.0, 2.0 * np.pi))
                amp = float(self.rng.uniform(0.04, 0.14))
                wander = amp * torch.sin(2.0 * np.pi * 0.25 * t + phase)
                x[0] = x[0] + wander
                x[1] = x[1] + wander * 0.85

        return x, self.y_disease[idx], self.y_stage[idx]


class _SEResBlock1D(nn.Module):
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


class CardiacWaveformCNN3Class(nn.Module):
    """
    Domain-invariant PyTorch 1D ResNet + BiGRU + Attention model for Cardiac ECG:
      - Head A: 4 classes (`Normal`, `AFib`, `Arrhythmia`, `Ischemic_MI`)
      - Head B: 4 cardiac acuity stages (`Stage 0..3`)
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = len(CARDIAC_CLASSES),
        num_stages: int = 4,
        dropout: float = 0.35,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=11, stride=2, padding=5, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.block1 = _SEResBlock1D(32, 64, kernel_size=7, stride=2, dropout=0.10)
        self.block2 = _SEResBlock1D(64, 96, kernel_size=5, stride=2, dropout=0.15)
        self.block3 = _SEResBlock1D(96, 128, kernel_size=5, stride=2, dropout=0.20)
        self.bigru = nn.GRU(
            input_size=128,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.attn_proj = nn.Linear(128, 1)
        self.shared_fc = nn.Sequential(
            nn.LayerNorm(256),
            nn.Dropout(dropout),
            nn.Linear(256, 96),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        self.head_a = nn.Linear(96, num_classes)
        self.head_b = nn.Linear(96, num_stages)

    def forward_with_embedding(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.stem(x)
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        seq = h.transpose(1, 2)  # (B, T, 128)
        gru_out, _ = self.bigru(seq)  # (B, T, 128)
        attn_w = F.softmax(self.attn_proj(gru_out), dim=1)
        ctx_attn = torch.sum(gru_out * attn_w, dim=1)
        ctx_max, _ = torch.max(gru_out, dim=1)
        feat = torch.cat([ctx_attn, ctx_max], dim=1)  # (B, 256)
        rep = self.shared_fc(feat)
        return self.head_a(rep), self.head_b(rep), rep

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        la, lb, _ = self.forward_with_embedding(x)
        return la, lb


CardiacWaveformCNN4Class = CardiacWaveformCNN3Class


def _compute_hospital_class_weights(
    y: np.ndarray,
    sample_sources: Optional[np.ndarray] = None,
    exponent: float = 0.70,
    min_stratum: int = 25,
) -> np.ndarray:
    """
    Computes inverse-class-frequency weights, optionally tempered across hospital sources
    (`1 / count(class, source)^exponent`, normalized so every class receives equal total weight).
    """
    y_arr = np.asarray(y, dtype=np.int64)
    n = len(y_arr)
    classes = np.unique(y_arr)
    n_cls = max(1, len(classes))
    w = np.ones(n, dtype=np.float64)

    if sample_sources is None or len(sample_sources) != n:
        for c in classes:
            mask = y_arr == c
            cnt = int(mask.sum())
            if cnt > 0:
                w[mask] = float(n) / (float(n_cls) * float(cnt))
        return w.astype(np.float32)

    src_arr = np.asarray(sample_sources, dtype=str)
    strata_counts: Dict[Tuple[int, str], int] = {}
    for c_i, s_i in zip(y_arr.tolist(), src_arr.tolist()):
        key = (c_i, s_i)
        strata_counts[key] = strata_counts.get(key, 0) + 1

    for idx, (c_i, s_i) in enumerate(zip(y_arr.tolist(), src_arr.tolist())):
        cnt = max(min_stratum, strata_counts[(c_i, s_i)])
        w[idx] = 1.0 / (float(cnt) ** exponent)

    target_cls_sum = float(n) / float(n_cls)
    for c in classes:
        mask = y_arr == c
        s_sum = float(w[mask].sum())
        if s_sum > 0:
            w[mask] *= target_cls_sum / s_sum
    return w.astype(np.float32)


def evaluate_ad8232_ischemic_hardware_guard(
    feat_row: Dict[str, float],
    gate: Dict[str, Any],
    pa_dict: Dict[str, float],
    d_code: int,
    imu_smv_g: float = 1.0,
) -> Dict[str, Any]:
    """
    Non-mutating AD8232 hardware-aware guard for Single-Lead Ischemic_MI sensitivity.
    Verifies high SQI, stationary IMU posture, and corroborated ST/T/Q morphology
    while never overriding `disease_code` (`rules_overrode_ai = False`).
    """
    ecg_sqi = float(gate.get("ecg_sqi", feat_row.get("ecg_sqi", 0.0)))
    sqi_ok = bool(gate.get("inference_reliable", False) and ecg_sqi >= 0.45)
    motion_val = float(imu_smv_g) if np.isfinite(imu_smv_g) else 1.0
    motion_ok = bool(motion_val <= 1.25)

    st_abs = float(feat_row.get("st_abs_max_rel", 0.0))
    t_min = min(float(feat_row.get("t_wave_rel_l1", 0.15)), float(feat_row.get("t_wave_rel_l2", 0.15)))
    q_min = float(feat_row.get("q_dip_rel_min", -0.05))
    morph_ok = bool(st_abs >= 0.07 or t_min <= -0.10 or q_min <= -0.18)

    isc_prob = float(pa_dict.get("Ischemic_MI", 0.0))
    verified = bool((d_code == 3 or isc_prob >= 0.38) and sqi_ok and motion_ok and morph_ok)

    return {
        "matched_05hz_filter_active": True,
        "sqi_sufficient_for_st": sqi_ok,
        "motion_stable_for_st": motion_ok,
        "st_morphology_corroborated": morph_ok,
        "ad8232_st_alert_verified": verified,
        "single_lead_scope_warning": (
            "Single-lead (Lead I) captures anterolateral/high-lateral ischemic shifts; "
            "cannot exclude inferior/posterior MI without 12-lead ECG."
        ),
        "hardware_upgrade_recommendation": (
            "Current AD8232 hardware uses a 0.5 Hz high-pass filter; for clinical diagnostic "
            "ST-segment fidelity (0.05 Hz IEC standard), upgrade AFE to MAX30003 "
            "(18-bit SPI, CMRR > 115 dB, 0.05 Hz HPF)."
        ),
    }


class CardiacModel1A:
    """
    Model 1A: Specialized Hybrid 4-Class Cardiac Rhythm & Ischemic MI Classifier.
    Completely decoupled from Brain Stroke inference. Supports BOTH Head A
    (Normal / AFib / Arrhythmia / Ischemic_MI) and Head B (4-stage cardiac acuity)
    via Hybrid 1D-CNN-BiGRU + XGBoost without rule overrides.
    """

    def __init__(
        self,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        n_estimators: int = 180,
        subsample: float = 0.80,
        colsample_bytree: float = 0.75,
        reg_alpha: float = 1.2,
        reg_lambda: float = 2.5,
        min_child_weight: int = 6,
        dl_weight: float = 0.45,
        dl_weight_a: Optional[float] = None,
        dl_weight_b: Optional[float] = None,
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
        self.dl_weight_a = float(dl_weight if dl_weight_a is None else dl_weight_a)
        self.dl_weight_b = float(dl_weight if dl_weight_b is None else dl_weight_b)
        self.random_state = random_state

        xgb_jobs = min(8, max(1, os.cpu_count() or 4))
        self.scaler = StandardScaler()
        self.head_a = XGBClassifier(
            objective="multi:softprob",
            num_class=len(CARDIAC_CLASSES),
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
            n_jobs=xgb_jobs,
        )
        self.head_b = XGBClassifier(
            objective="multi:softprob",
            num_class=len(CARDIAC_STAGE_NAMES),
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
            n_jobs=xgb_jobs,
        )

        self.feature_names = list(CARDIAC_ECG_FEATURES)
        self.disease_classes = list(CARDIAC_CLASSES)
        self.stage_names = list(CARDIAC_STAGE_NAMES)
        self.cnn_weights_path: Optional[str] = DEFAULT_CARDIAC_CNN_PATH
        self.cnn_model: Optional[CardiacWaveformCNN3Class] = None
        self.use_meta_fusion: bool = False
        self.use_meta_fusion_a: bool = False
        self.use_meta_fusion_b: bool = False
        self.emb_pca: Optional[PCA] = None
        self.meta_scaler: Optional[StandardScaler] = None
        self.meta_head_a: Optional[XGBClassifier] = None
        self.meta_head_b: Optional[XGBClassifier] = None
        self.is_fitted = False

    def _ensure_cnn(self) -> Optional[CardiacWaveformCNN3Class]:
        if self.cnn_model is None and self.cnn_weights_path and os.path.exists(self.cnn_weights_path):
            try:
                ckpt = torch.load(self.cnn_weights_path, map_location="cpu", weights_only=True)
                state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
                net = CardiacWaveformCNN3Class(
                    in_channels=3,
                    num_classes=len(self.disease_classes),
                    num_stages=len(self.stage_names),
                )
                net.load_state_dict(state)
                net.eval()
                self.cnn_model = net
            except Exception:
                self.cnn_model = None
        return self.cnn_model

    def fit_tabular(
        self,
        X_feat: Union[pd.DataFrame, np.ndarray],
        y_disease: np.ndarray,
        y_stage: np.ndarray,
        sample_sources: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Fits both Head A (4-class cardiac disease) and Head B (4-stage cardiac acuity) on real ECG features."""
        X_df = (
            X_feat[self.feature_names]
            if isinstance(X_feat, pd.DataFrame)
            else pd.DataFrame(X_feat, columns=self.feature_names)
        )
        X_mat = X_df.astype(np.float32).values
        X_mat = np.nan_to_num(X_mat, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = self.scaler.fit_transform(X_mat)

        y_a = np.asarray(y_disease, dtype=np.int64)
        y_b = np.asarray(y_stage, dtype=np.int64)

        cls_a = np.unique(y_a)
        sw_a = _compute_hospital_class_weights(y_a, sample_sources=sample_sources)

        cls_b = np.unique(y_b)
        w_b_map = dict(zip(cls_b, compute_class_weight("balanced", classes=cls_b, y=y_b)))
        sw_b = np.array([w_b_map[int(c)] for c in y_b], dtype=np.float32)

        self.head_a.fit(X_scaled, y_a, sample_weight=sw_a)
        self.head_b.fit(X_scaled, y_b, sample_weight=sw_b)
        self.is_fitted = True
        return {
            "n_samples": int(len(X_mat)),
            "head_a_distribution": {int(k): int(np.sum(y_a == k)) for k in cls_a},
            "head_b_distribution": {int(k): int(np.sum(y_b == k)) for k in cls_b},
        }

    def fit_deep_cnn(
        self,
        waveforms: np.ndarray,
        y_disease: np.ndarray,
        y_stage: np.ndarray,
        val_waveforms: Optional[np.ndarray] = None,
        val_disease: Optional[np.ndarray] = None,
        val_stage: Optional[np.ndarray] = None,
        epochs: int = 16,
        batch_size: int = 64,
        lr: float = 1.5e-3,
        save_cnn_path: Optional[str] = DEFAULT_CARDIAC_CNN_PATH,
        verbose: bool = False,
        sample_sources: Optional[np.ndarray] = None,
        focal_gamma: float = 0.0,
    ) -> Dict[str, Any]:
        """Trains joint 1D-CNN-BiGRU-Attention on 3-channel normalized 250 Hz ECG waveforms."""
        torch.manual_seed(self.random_state)
        torch.set_num_threads(min(8, max(1, os.cpu_count() or 2)))

        ch_train = (
            np.ascontiguousarray(waveforms, dtype=np.float32)
            if (isinstance(waveforms, np.ndarray) and waveforms.ndim == 3 and waveforms.shape[1] == 3)
            else build_cardiac_cnn_channels(waveforms, fs=TARGET_FS)
        )
        ds_train = _CardiacAugDataset(ch_train, y_disease, y_stage, augment=True, seed=self.random_state)

        n_cls = len(self.disease_classes)
        if sample_sources is not None:
            w_arr = _compute_hospital_class_weights(y_disease, sample_sources=sample_sources)
            sample_w = torch.DoubleTensor(w_arr.astype(np.float64))
        else:
            cls_counts = np.bincount(y_disease.astype(np.int64), minlength=n_cls)
            inv_w = 1.0 / np.maximum(1.0, cls_counts.astype(np.float64))
            sample_w = torch.DoubleTensor([inv_w[int(c)] for c in y_disease])
        sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)

        train_loader = DataLoader(ds_train, batch_size=batch_size, sampler=sampler, drop_last=False)

        val_loader = None
        if val_waveforms is not None and val_disease is not None and val_stage is not None:
            ch_val = (
                np.ascontiguousarray(val_waveforms, dtype=np.float32)
                if (isinstance(val_waveforms, np.ndarray) and val_waveforms.ndim == 3 and val_waveforms.shape[1] == 3)
                else build_cardiac_cnn_channels(val_waveforms, fs=TARGET_FS)
            )
            ds_val = _CardiacAugDataset(ch_val, val_disease, val_stage, augment=False, seed=self.random_state)
            val_loader = DataLoader(ds_val, batch_size=128, shuffle=False)

        net = CardiacWaveformCNN3Class(
            in_channels=3,
            num_classes=n_cls,
            num_stages=len(self.stage_names),
            dropout=0.35,
        )
        optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=2e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

        crit_a = nn.CrossEntropyLoss(label_smoothing=0.04)
        crit_b = nn.CrossEntropyLoss(label_smoothing=0.04)

        best_val_loss = float("inf")
        best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
        history: List[Dict[str, float]] = []

        def _save_atomic(state: Dict[str, Any]) -> None:
            if not save_cnn_path:
                return
            os.makedirs(os.path.dirname(save_cnn_path) or ".", exist_ok=True)
            tmp_path = f"{save_cnn_path}.tmp"
            torch.save({"state_dict": state}, tmp_path)
            os.replace(tmp_path, save_cnn_path)

        for ep in range(1, epochs + 1):
            t_ep = time.time()
            net.train()
            tr_loss = 0.0
            tr_correct = 0
            tr_total = 0
            for xb, yb_a, yb_b in train_loader:
                optimizer.zero_grad()
                la, lb = net(xb)
                if focal_gamma > 0.0:
                    ce_a = F.cross_entropy(la, yb_a, label_smoothing=0.04, reduction="none")
                    pt_a = (
                        torch.softmax(la, dim=-1)
                        .gather(1, yb_a.unsqueeze(1))
                        .squeeze(1)
                        .clamp(1e-6, 1.0)
                    )
                    loss_a = (((1.0 - pt_a) ** focal_gamma) * ce_a).mean()
                else:
                    loss_a = crit_a(la, yb_a)
                loss = loss_a + 0.45 * crit_b(lb, yb_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
                optimizer.step()
                tr_loss += float(loss.item()) * len(yb_a)
                tr_correct += int((torch.argmax(la, dim=1) == yb_a).sum().item())
                tr_total += len(yb_a)
            scheduler.step()

            ep_stat = {
                "epoch": ep,
                "train_loss": tr_loss / max(1, tr_total),
                "train_acc": tr_correct / max(1, tr_total),
            }

            if val_loader is not None:
                net.eval()
                v_loss = 0.0
                v_correct = 0
                v_total = 0
                with torch.no_grad():
                    for xb, yb_a, yb_b in val_loader:
                        la, lb = net(xb)
                        loss = crit_a(la, yb_a) + 0.45 * crit_b(lb, yb_b)
                        v_loss += float(loss.item()) * len(yb_a)
                        v_correct += int((torch.argmax(la, dim=1) == yb_a).sum().item())
                        v_total += len(yb_a)
                ep_val_loss = v_loss / max(1, v_total)
                ep_stat["val_loss"] = ep_val_loss
                ep_stat["val_acc"] = v_correct / max(1, v_total)
                if ep_val_loss < best_val_loss:
                    best_val_loss = ep_val_loss
                    best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
                    _save_atomic(best_state)
            else:
                best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}

            dt = time.time() - t_ep
            ep_stat["elapsed_sec"] = round(dt, 1)
            history.append(ep_stat)
            if verbose:
                val_msg = (
                    f" | val_loss={ep_stat['val_loss']:.4f} val_acc={ep_stat['val_acc']*100:.2f}%"
                    if "val_loss" in ep_stat
                    else ""
                )
                print(
                    f"  [CNN Epoch {ep:02d}/{epochs}] train_loss={ep_stat['train_loss']:.4f} "
                    f"train_acc={ep_stat['train_acc']*100:.2f}%{val_msg} ({dt:.1f}s)",
                    flush=True,
                )

        net.load_state_dict(best_state)
        net.eval()
        self.cnn_model = net
        if save_cnn_path:
            _save_atomic(best_state)
            self.cnn_weights_path = save_cnn_path

        return {"epochs": epochs, "best_val_loss": best_val_loss, "history": history}

    def predict_cnn_probs_and_embeddings(
        self, waveforms: np.ndarray, fs: float = TARGET_FS, batch_size: int = 128
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Runs batched 1D-CNN-BiGRU inference returning `(probs_a, probs_b, emb_96)`."""
        net = self._ensure_cnn()
        if net is None:
            raise RuntimeError("Cardiac CNN weights are not loaded.")
        ch = (
            np.ascontiguousarray(waveforms, dtype=np.float32)
            if (isinstance(waveforms, np.ndarray) and waveforms.ndim == 3 and waveforms.shape[1] == 3)
            else build_cardiac_cnn_channels(waveforms, fs=fs)
        )
        n = len(ch)
        pa_list: List[np.ndarray] = []
        pb_list: List[np.ndarray] = []
        emb_list: List[np.ndarray] = []
        net.eval()
        with torch.no_grad():
            for start in range(0, n, batch_size):
                xb = torch.from_numpy(ch[start : start + batch_size])
                la, lb, rep = net.forward_with_embedding(xb)
                pa_list.append(torch.softmax(la, dim=-1).cpu().numpy())
                pb_list.append(torch.softmax(lb, dim=-1).cpu().numpy())
                emb_list.append(rep.cpu().numpy())
        return np.vstack(pa_list), np.vstack(pb_list), np.vstack(emb_list)

    def predict_cnn_probs(
        self, waveforms: np.ndarray, fs: float = TARGET_FS, batch_size: int = 128
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Runs batched 1D-CNN-BiGRU inference returning `(probs_head_a, probs_head_b)`."""
        pa, pb, _ = self.predict_cnn_probs_and_embeddings(waveforms, fs=fs, batch_size=batch_size)
        return pa, pb

    def _build_meta_xgb(self, num_class: int) -> XGBClassifier:
        xgb_jobs = min(8, max(1, os.cpu_count() or 4))
        return XGBClassifier(
            objective="multi:softprob",
            num_class=num_class,
            max_depth=3,
            learning_rate=0.05,
            n_estimators=120,
            subsample=0.80,
            colsample_bytree=0.70,
            reg_alpha=2.0,
            reg_lambda=5.0,
            min_child_weight=8,
            eval_metric="mlogloss",
            random_state=self.random_state,
            n_jobs=xgb_jobs,
        )

    def fit_meta_fusion(
        self,
        val_waveforms: np.ndarray,
        X_val_feat: Union[pd.DataFrame, np.ndarray],
        y_val_a: np.ndarray,
        y_val_b: np.ndarray,
        val_groups: np.ndarray,
        fs: float = TARGET_FS,
    ) -> Dict[str, Any]:
        """
        Evaluates out-of-fold 96-dim CNN embedding + tabular stacking via 5-fold StratifiedGroupKFold
        on the train-only validation fold, enabling meta-fusion per head only if CV Macro-F1 improves.
        """
        if not self.is_fitted:
            raise RuntimeError("Fit tabular branch before fit_meta_fusion.")
        X_df = (
            X_val_feat[self.feature_names]
            if isinstance(X_val_feat, pd.DataFrame)
            else pd.DataFrame(X_val_feat, columns=self.feature_names)
        )
        X_mat = np.nan_to_num(X_df.astype(np.float32).values, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = self.scaler.transform(X_mat)
        pa_xgb = self.head_a.predict_proba(X_scaled).astype(np.float64)
        pb_xgb = self.head_b.predict_proba(X_scaled).astype(np.float64)
        pa_cnn, pb_cnn, emb_96 = self.predict_cnn_probs_and_embeddings(val_waveforms, fs=fs)

        ya = np.asarray(y_val_a, dtype=np.int64)
        yb = np.asarray(y_val_b, dtype=np.int64)
        groups = np.asarray(val_groups)

        w_a = float(np.clip(self.dl_weight_a, 0.0, 1.0))
        w_b = float(np.clip(self.dl_weight_b, 0.0, 1.0))
        blend_a = (1.0 - w_a) * pa_xgb + w_a * pa_cnn.astype(np.float64)
        blend_b = (1.0 - w_b) * pb_xgb + w_b * pb_cnn.astype(np.float64)
        cv_blend_f1_a = float(f1_score(ya, np.argmax(blend_a, axis=1), average="macro"))
        cv_blend_f1_b = float(f1_score(yb, np.argmax(blend_b, axis=1), average="macro"))

        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        oof_meta_a = np.zeros_like(pa_cnn, dtype=np.float64)
        oof_meta_b = np.zeros_like(pb_cnn, dtype=np.float64)
        n_comp = min(24, emb_96.shape[0], emb_96.shape[1])

        for tr_f, va_f in sgkf.split(X_scaled, ya, groups=groups):
            pca_f = PCA(n_components=n_comp, random_state=self.random_state).fit(emb_96[tr_f])
            X_tr_m = np.hstack([
                pca_f.transform(emb_96[tr_f]),
                pa_cnn[tr_f],
                pb_cnn[tr_f],
                pa_xgb[tr_f],
                pb_xgb[tr_f],
                X_scaled[tr_f],
            ])
            X_va_m = np.hstack([
                pca_f.transform(emb_96[va_f]),
                pa_cnn[va_f],
                pb_cnn[va_f],
                pa_xgb[va_f],
                pb_xgb[va_f],
                X_scaled[va_f],
            ])

            cls_af = np.unique(ya[tr_f])
            map_af = dict(zip(cls_af, compute_class_weight("balanced", classes=cls_af, y=ya[tr_f])))
            sw_af = np.array([map_af[int(c)] for c in ya[tr_f]], dtype=np.float32)
            ma = self._build_meta_xgb(len(self.disease_classes))
            ma.fit(X_tr_m, ya[tr_f], sample_weight=sw_af)
            oof_meta_a[va_f] = ma.predict_proba(X_va_m)

            cls_bf = np.unique(yb[tr_f])
            map_bf = dict(zip(cls_bf, compute_class_weight("balanced", classes=cls_bf, y=yb[tr_f])))
            sw_bf = np.array([map_bf[int(c)] for c in yb[tr_f]], dtype=np.float32)
            mb = self._build_meta_xgb(len(self.stage_names))
            mb.fit(X_tr_m, yb[tr_f], sample_weight=sw_bf)
            oof_meta_b[va_f] = mb.predict_proba(X_va_m)

        cv_meta_f1_a = float(f1_score(ya, np.argmax(oof_meta_a, axis=1), average="macro"))
        cv_meta_f1_b = float(f1_score(yb, np.argmax(oof_meta_b, axis=1), average="macro"))

        self.use_meta_fusion_a = bool(cv_meta_f1_a > cv_blend_f1_a)
        self.use_meta_fusion_b = bool(cv_meta_f1_b > cv_blend_f1_b)
        self.use_meta_fusion = bool(self.use_meta_fusion_a or self.use_meta_fusion_b)

        if self.use_meta_fusion:
            self.emb_pca = PCA(n_components=n_comp, random_state=self.random_state).fit(emb_96)
            X_full_m = np.hstack([
                self.emb_pca.transform(emb_96),
                pa_cnn,
                pb_cnn,
                pa_xgb,
                pb_xgb,
                X_scaled,
            ])
            if self.use_meta_fusion_a:
                cls_af = np.unique(ya)
                map_af = dict(zip(cls_af, compute_class_weight("balanced", classes=cls_af, y=ya)))
                sw_af = np.array([map_af[int(c)] for c in ya], dtype=np.float32)
                self.meta_head_a = self._build_meta_xgb(len(self.disease_classes))
                self.meta_head_a.fit(X_full_m, ya, sample_weight=sw_af)
            else:
                self.meta_head_a = None
            if self.use_meta_fusion_b:
                cls_bf = np.unique(yb)
                map_bf = dict(zip(cls_bf, compute_class_weight("balanced", classes=cls_bf, y=yb)))
                sw_bf = np.array([map_bf[int(c)] for c in yb], dtype=np.float32)
                self.meta_head_b = self._build_meta_xgb(len(self.stage_names))
                self.meta_head_b.fit(X_full_m, yb, sample_weight=sw_bf)
            else:
                self.meta_head_b = None
        else:
            self.emb_pca = None
            self.meta_head_a = None
            self.meta_head_b = None

        return {
            "cv_blend_macro_f1_a": round(cv_blend_f1_a, 4),
            "cv_meta_macro_f1_a": round(cv_meta_f1_a, 4),
            "use_meta_fusion_a": self.use_meta_fusion_a,
            "cv_blend_macro_f1_b": round(cv_blend_f1_b, 4),
            "cv_meta_macro_f1_b": round(cv_meta_f1_b, 4),
            "use_meta_fusion_b": self.use_meta_fusion_b,
        }

    def _evaluate_ecg_gate(
        self,
        lead1: np.ndarray,
        hardware_metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Fail-closed Lead-Off & SQI gate prior to running any inference."""
        if isinstance(hardware_metrics, dict):
            lead_ok_flag = hardware_metrics.get("ecg_lead_ok")
            lead_ok_pct = hardware_metrics.get("ecg_lead_ok_pct")
            if lead_ok_flag is False or (lead_ok_pct is not None and float(lead_ok_pct) < 75.0):
                return {
                    "status": "LEAD_OFF",
                    "inference_reliable": False,
                    "ecg_lead_off": True,
                    "ecg_sqi": 0.0,
                    "reason": "hardware_lead_off",
                }

        arr = np.asarray(lead1, dtype=np.float32).flatten()
        var = float(np.var(arr)) if arr.size > 0 else 0.0
        peak = float(np.max(np.abs(arr))) if arr.size > 0 else 0.0
        if arr.size < 250 or var < 1e-5 or peak > 6.0 or not np.isfinite(arr).all():
            return {"status": "LEAD_OFF", "inference_reliable": False, "ecg_lead_off": True, "ecg_sqi": 0.0}

        zero_mask = (np.abs(arr) < 1e-5).astype(np.int8)
        if np.any(zero_mask):
            padded = np.concatenate(([0], zero_mask, [0]))
            diffs = np.diff(padded)
            starts = np.where(diffs == 1)[0]
            ends = np.where(diffs == -1)[0]
            max_zero_run = int(np.max(ends - starts)) if len(starts) > 0 else 0
            if max_zero_run >= int(0.60 * TARGET_FS):
                return {
                    "status": "LEAD_OFF",
                    "inference_reliable": False,
                    "ecg_lead_off": True,
                    "ecg_sqi": 0.0,
                    "reason": "contiguous_zero_dropout",
                }

        bp = _safe_bandpass_250(arr, fs=TARGET_FS)
        p2p = float(np.percentile(bp, 99) - np.percentile(bp, 1))
        if p2p < 0.04:
            return {"status": "LEAD_OFF", "inference_reliable": False, "ecg_lead_off": True, "ecg_sqi": 0.0}

        seg_stds = [float(np.std(seg)) for seg in np.array_split(bp, 5)]
        if sum(s < 0.004 for s in seg_stds) >= 2:
            return {
                "status": "LEAD_OFF",
                "inference_reliable": False,
                "ecg_lead_off": True,
                "ecg_sqi": 0.0,
                "reason": "intermittent_lead_dropout",
            }

        sqi = float(compute_ecg_sqi(bp, fs=TARGET_FS, causal=False))
        med_bp = float(np.median(bp))
        mad = 1.4826 * float(np.median(np.abs(bp - med_bp))) + 1e-6
        crest = float(np.percentile(np.abs(bp), 99.5)) / mad
        r_peaks = _detect_r_peaks_fast(bp, fs=TARGET_FS)
        n_beats = len(r_peaks)

        if sqi < 0.30 or crest < 3.2 or n_beats < 4 or n_beats > 38:
            return {
                "status": "POOR_SIGNAL",
                "inference_reliable": False,
                "ecg_lead_off": False,
                "ecg_sqi": round(sqi, 3),
                "beat_count": int(n_beats),
                "crest_factor": round(crest, 2),
            }
        return {
            "status": "VALID",
            "inference_reliable": True,
            "ecg_lead_off": False,
            "ecg_sqi": round(sqi, 3),
            "beat_count": int(n_beats),
            "crest_factor": round(crest, 2),
        }

    def _combine_dl_and_xgb_branches(
        self,
        waveforms: np.ndarray,
        X_scaled: np.ndarray,
        pa_xgb: np.ndarray,
        pb_xgb: np.ndarray,
        fs: float = TARGET_FS,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Blends CNN + XGBoost per head (meta-fusion when enabled, else decoupled dl_weight_a/b)."""
        need_emb = bool((self.use_meta_fusion_a or self.use_meta_fusion_b) and self.emb_pca is not None)
        if need_emb:
            pa_cnn, pb_cnn, emb_96 = self.predict_cnn_probs_and_embeddings(waveforms, fs=fs)
            X_meta = np.hstack([
                self.emb_pca.transform(emb_96),
                pa_cnn,
                pb_cnn,
                pa_xgb,
                pb_xgb,
                X_scaled,
            ])
        else:
            pa_cnn, pb_cnn = self.predict_cnn_probs(waveforms, fs=fs)
            X_meta = None

        w_a = float(np.clip(self.dl_weight_a, 0.0, 1.0))
        w_b = float(np.clip(self.dl_weight_b, 0.0, 1.0))
        if self.use_meta_fusion_a and self.meta_head_a is not None and X_meta is not None:
            pa_ens = self.meta_head_a.predict_proba(X_meta).astype(np.float64)
        else:
            pa_ens = (1.0 - w_a) * pa_xgb + w_a * pa_cnn.astype(np.float64)

        if self.use_meta_fusion_b and self.meta_head_b is not None and X_meta is not None:
            pb_ens = self.meta_head_b.predict_proba(X_meta).astype(np.float64)
        else:
            pb_ens = (1.0 - w_b) * pb_xgb + w_b * pb_cnn.astype(np.float64)

        dl_meta = {
            "active": True,
            "architecture": "1D-ResNet-BiGRU-Attention",
            "dl_weight": w_a,
            "dl_weight_a": w_a,
            "dl_weight_b": w_b,
            "meta_fusion_a": bool(self.use_meta_fusion_a and self.meta_head_a is not None),
            "meta_fusion_b": bool(self.use_meta_fusion_b and self.meta_head_b is not None),
        }
        return pa_cnn, pb_cnn, pa_ens, pb_ens, dl_meta

    def predict_batch_components(
        self,
        waveforms: np.ndarray,
        X_feat: Optional[pd.DataFrame] = None,
        fs: float = TARGET_FS,
    ) -> Dict[str, np.ndarray]:
        """
        Evaluates all three branches cleanly in batch for rigorous benchmark auditing:
          - `probs_a_xgb`, `probs_b_xgb`
          - `probs_a_cnn`, `probs_b_cnn` (if CNN available)
          - `probs_a_ens`, `probs_b_ens`
        """
        if not self.is_fitted:
            raise RuntimeError("CardiacModel1A is not fitted yet.")
        if X_feat is None:
            X_feat = extract_cardiac_feature_batch(waveforms, fs=fs)
        X_mat = X_feat[self.feature_names].astype(np.float32).values
        X_mat = np.nan_to_num(X_mat, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = self.scaler.transform(X_mat)

        pa_xgb = self.head_a.predict_proba(X_scaled).astype(np.float64)
        pb_xgb = self.head_b.predict_proba(X_scaled).astype(np.float64)

        net = self._ensure_cnn()
        if net is not None:
            pa_cnn, pb_cnn, pa_ens, pb_ens, _ = self._combine_dl_and_xgb_branches(
                waveforms, X_scaled, pa_xgb, pb_xgb, fs=fs
            )
        else:
            pa_cnn = np.empty((0, len(self.disease_classes)))
            pb_cnn = np.empty((0, len(self.stage_names)))
            pa_ens = pa_xgb.copy()
            pb_ens = pb_xgb.copy()

        return {
            "probs_a_xgb": pa_xgb,
            "probs_b_xgb": pb_xgb,
            "probs_a_cnn": pa_cnn,
            "probs_b_cnn": pb_cnn,
            "probs_a_ens": pa_ens,
            "probs_b_ens": pb_ens,
        }

    def predict_all(
        self,
        X: Union[Dict[str, Any], pd.DataFrame, np.ndarray],
        fs: float = TARGET_FS,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Unified inference entrypoint accepting:
          - Raw sensor payload dict `{"raw_windows": {"ecg_mv": [...], "ecg_lead2_mv": [...]}}`
          - Pre-extracted feature DataFrame / dict matching `CARDIAC_ECG_FEATURES`
          - Batch numpy waveform array `(N, 2, 2500)` or `(N, 2500)`
        """
        if not self.is_fitted:
            raise RuntimeError("CardiacModel1A is not fitted yet.")

        if isinstance(X, dict):
            is_single = True
            raw_imu = X.get(
                "imu_smv_g",
                X.get("features", {}).get("imu_smv_g", 1.0) if isinstance(X.get("features"), dict) else 1.0,
            )
            if "raw_windows" in X:
                rw = X["raw_windows"]
                l1 = np.asarray(rw.get("ecg_mv", [0.0] * TARGET_SAMPLES), dtype=np.float32)
                l2_raw = rw.get("ecg_lead2_mv")
                l2 = np.asarray(l2_raw, dtype=np.float32) if l2_raw is not None else l1
                waves = np.stack([_safe_bandpass_250(l1, fs), _safe_bandpass_250(l2, fs)], axis=0)[np.newaxis, ...]
                feat_dict = extract_cardiac_ecg_features_from_waveform(l1, l2_raw, fs=fs)
                X_feat = pd.DataFrame([feat_dict], columns=self.feature_names)
                gates = [self._evaluate_ecg_gate(l1, hardware_metrics=X.get("hardware_metrics"))]
                dl_available = True
                if "imu_smv_g" not in X and all(k in rw for k in ("imu_ax_g", "imu_ay_g", "imu_az_g")):
                    ax = np.asarray(rw["imu_ax_g"], dtype=np.float32)
                    ay = np.asarray(rw["imu_ay_g"], dtype=np.float32)
                    az = np.asarray(rw["imu_az_g"], dtype=np.float32)
                    if ax.size > 0 and ax.size == ay.size == az.size:
                        raw_imu = float(np.max(np.sqrt(ax * ax + ay * ay + az * az)))
            else:
                src = X.get("features", X) if isinstance(X.get("features"), dict) else X
                row = {c: float(src.get(c, 0.0)) for c in self.feature_names}
                X_feat = pd.DataFrame([row], columns=self.feature_names)
                waves = None
                gates = [{"status": "VALID", "inference_reliable": True, "ecg_lead_off": False, "ecg_sqi": float(row.get("ecg_sqi", 1.0))}]
                dl_available = False
            imu_vals = [float(raw_imu) if raw_imu is not None and np.isfinite(float(raw_imu)) else 1.0]
        elif isinstance(X, np.ndarray):
            w = np.asarray(X, dtype=np.float32)
            is_single = w.ndim == 1
            if w.ndim == 1:
                w = w[np.newaxis, np.newaxis, :]
            elif w.ndim == 2:
                w = w[:, np.newaxis, :]
            waves = w
            X_feat = extract_cardiac_feature_batch(waves, fs=fs)
            gates = [self._evaluate_ecg_gate(waves[i, 0]) for i in range(len(waves))]
            dl_available = True
            imu_vals = [1.0] * len(waves)
        elif isinstance(X, pd.DataFrame):
            is_single = False
            X_feat = X.reindex(columns=self.feature_names, fill_value=0.0)
            waves = None
            gates = [{"status": "VALID", "inference_reliable": True, "ecg_lead_off": False, "ecg_sqi": 1.0}] * len(X_feat)
            dl_available = False
            if "imu_smv_g" in X.columns:
                imu_vals = [float(v) if np.isfinite(float(v)) else 1.0 for v in X["imu_smv_g"].values]
            else:
                imu_vals = [1.0] * len(X_feat)
        else:
            raise TypeError(f"Unsupported input type for CardiacModel1A.predict_all: {type(X)}")

        X_mat = np.nan_to_num(X_feat[self.feature_names].astype(np.float32).values, nan=0.0)
        X_scaled = self.scaler.transform(X_mat)
        pa_xgb = self.head_a.predict_proba(X_scaled).astype(np.float64)
        pb_xgb = self.head_b.predict_proba(X_scaled).astype(np.float64)

        net = self._ensure_cnn()
        if dl_available and waves is not None and net is not None:
            _, _, probs_a, probs_b, dl_meta = self._combine_dl_and_xgb_branches(
                waves, X_scaled, pa_xgb, pb_xgb, fs=fs
            )
        else:
            probs_a = pa_xgb
            probs_b = pb_xgb
            dl_meta = {"active": False, "architecture": "1D-ResNet-BiGRU-Attention", "reason": "tabular_input_no_raw_waveform"}

        results: List[Dict[str, Any]] = []
        for i in range(len(X_mat)):
            gate = gates[i]
            feat_row = {c: float(X_feat.iloc[i][c]) for c in self.feature_names}
            imu_val = imu_vals[i]
            if not gate.get("inference_reliable", True):
                abstained_probs = {c: 0.0 for c in self.disease_classes}
                abstained_stage_probs = {s: 0.0 for s in self.stage_names}
                status_label = "SENSOR_DISCONNECTED" if gate["status"] == "LEAD_OFF" else "POOR_SIGNAL"
                hw_guard = evaluate_ad8232_ischemic_hardware_guard(
                    feat_row=feat_row,
                    gate=gate,
                    pa_dict=abstained_probs,
                    d_code=0,
                    imu_smv_g=imu_val,
                )
                results.append(
                    {
                        "model_id": "CardiacModel1A",
                        "disease_code": -1,
                        "disease_name": status_label,
                        "disease_probabilities": abstained_probs,
                        "stage_code": -1,
                        "stage_name": f"Sensor Error ({gate['status']}): Check Electrodes",
                        "stage_probabilities": abstained_stage_probs,
                        "signal_quality": gate,
                        "ischemic_mi_hardware_guard": hw_guard,
                        "dl_branch": {"active": False, "reason": f"abstained_{gate['status']}"},
                        "clinical_alert": (
                            f"CARDIAC SENSOR GATE ({gate['status']}): ELECTRODES DISCONNECTED OR POOR SIGNAL — NO PREDICTION MADE."
                        ),
                        "hardware_feedback": {"trigger_buzzer": False, "trigger_vibrator": False},
                        "raw_ml_decision": {
                            "disease_code": None,
                            "disease_name": "ABSTAINED_POOR_SIGNAL",
                            "stage_code": None,
                            "stage_name": "ABSTAINED_POOR_SIGNAL",
                            "rules_overrode_ai": False,
                        },
                    }
                )
                continue

            d_code = int(np.argmax(probs_a[i]))
            s_code = int(np.argmax(probs_b[i]))
            d_name = self.disease_classes[d_code]
            s_name = self.stage_names[s_code]
            pa_dict = {c: round(float(probs_a[i, idx]), 6) for idx, c in enumerate(self.disease_classes)}
            pb_dict = {s: round(float(probs_b[i, idx]), 6) for idx, s in enumerate(self.stage_names)}

            hw_guard = evaluate_ad8232_ischemic_hardware_guard(
                feat_row=feat_row,
                gate=gate,
                pa_dict=pa_dict,
                d_code=d_code,
                imu_smv_g=imu_val,
            )

            if d_code == 3 or pa_dict.get("Ischemic_MI", 0.0) >= 0.38:
                if hw_guard["ad8232_st_alert_verified"]:
                    alert = (
                        "CARDIAC EMERGENCY: ISCHEMIC MI / ST-T PATTERN DETECTED "
                        "(STATIONARY SQI VERIFIED). CONFIRM WITH 12-LEAD ECG."
                    )
                else:
                    alert = (
                        "CARDIAC CAUTION: POSSIBLE ISCHEMIC MI PATTERN FLAGGED UNDER MOTION OR LOW SQI "
                        "(AD8232 0.5 HZ LIMIT). RETEST STILL & CONFIRM WITH 12-LEAD ECG."
                    )
            elif d_code == 2 or pa_dict.get("Arrhythmia", 0.0) >= 0.40:
                alert = "CARDIAC WARNING: NON-AF CARDIAC ARRHYTHMIA DETECTED. MONITOR HEART RATE & RHYTHM."
            elif d_code == 1 or pa_dict.get("AFib", 0.0) >= 0.40:
                alert = "CARDIAC WARNING: ATRIAL FIBRILLATION (AFib) DETECTED. EVALUATE THROMBOEMBOLIC STROKE RISK."
            else:
                alert = "NORMAL SINUS RHYTHM: STABLE CARDIAC BASELINE."

            results.append(
                {
                    "model_id": "CardiacModel1A",
                    "disease_code": d_code,
                    "disease_name": d_name,
                    "disease_probabilities": pa_dict,
                    "stage_code": s_code,
                    "stage_name": s_name,
                    "stage_probabilities": pb_dict,
                    "signal_quality": gate,
                    "ischemic_mi_hardware_guard": hw_guard,
                    "dl_branch": dict(dl_meta),
                    "clinical_alert": alert,
                    "hardware_feedback": {
                        "trigger_buzzer": bool(
                            (d_code == 3 and s_code >= 2 and hw_guard["ad8232_st_alert_verified"])
                            or (d_code in (1, 2) and s_code >= 3)
                        ),
                        "trigger_vibrator": bool(d_code in (1, 2, 3) and s_code >= 1),
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

    def save(self, filepath: str = DEFAULT_CARDIAC_MODEL_PATH):
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
            "dl_weight_a": self.dl_weight_a,
            "dl_weight_b": self.dl_weight_b,
            "random_state": self.random_state,
            "scaler_mean": self.scaler.mean_,
            "scaler_scale": self.scaler.scale_,
            "head_a": self.head_a,
            "head_b": self.head_b,
            "feature_names": self.feature_names,
            "disease_classes": self.disease_classes,
            "stage_names": self.stage_names,
            "cnn_weights_path": self.cnn_weights_path,
            "use_meta_fusion": self.use_meta_fusion,
            "use_meta_fusion_a": self.use_meta_fusion_a,
            "use_meta_fusion_b": self.use_meta_fusion_b,
            "emb_pca": self.emb_pca,
            "meta_scaler": self.meta_scaler,
            "meta_head_a": self.meta_head_a,
            "meta_head_b": self.meta_head_b,
            "is_fitted": self.is_fitted,
        }
        with open(filepath, "wb") as f:
            pickle.dump(bundle, f)

    @classmethod
    def load(cls, filepath: str = DEFAULT_CARDIAC_MODEL_PATH) -> "CardiacModel1A":
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"CardiacModel1A artifact not found: {filepath}")
        with open(filepath, "rb") as f:
            b = pickle.load(f)
        dl_w = b.get("dl_weight", 0.45)
        inst = cls(
            max_depth=b.get("max_depth", 4),
            learning_rate=b.get("learning_rate", 0.05),
            n_estimators=b.get("n_estimators", 180),
            subsample=b.get("subsample", 0.80),
            colsample_bytree=b.get("colsample_bytree", 0.75),
            reg_alpha=b.get("reg_alpha", 1.2),
            reg_lambda=b.get("reg_lambda", 2.5),
            min_child_weight=b.get("min_child_weight", 6),
            dl_weight=dl_w,
            dl_weight_a=b.get("dl_weight_a", dl_w),
            dl_weight_b=b.get("dl_weight_b", dl_w),
            random_state=b.get("random_state", 42),
        )
        inst.head_a = b["head_a"]
        inst.head_b = b["head_b"]
        inst.scaler.mean_ = b["scaler_mean"]
        inst.scaler.scale_ = b["scaler_scale"]
        inst.feature_names = b.get("feature_names", list(CARDIAC_ECG_FEATURES))
        inst.disease_classes = b.get("disease_classes", list(CARDIAC_CLASSES))
        inst.stage_names = b.get("stage_names", list(CARDIAC_STAGE_NAMES))
        inst.cnn_weights_path = b.get("cnn_weights_path", DEFAULT_CARDIAC_CNN_PATH)
        inst.use_meta_fusion = bool(b.get("use_meta_fusion", False))
        inst.use_meta_fusion_a = bool(b.get("use_meta_fusion_a", False))
        inst.use_meta_fusion_b = bool(b.get("use_meta_fusion_b", False))
        inst.emb_pca = b.get("emb_pca", None)
        inst.meta_scaler = b.get("meta_scaler", None)
        inst.meta_head_a = b.get("meta_head_a", None)
        inst.meta_head_b = b.get("meta_head_b", None)
        inst.is_fitted = b.get("is_fitted", True)
        inst._ensure_cnn()
        return inst
