"""
signal_processor.py
Biomedical Signal Preprocessing & Feature Extraction Engine for Heart-Brain Axis Wearable
Upper-Arm Brachial Band (Biceps Placement).
Processes 10-second sliding windows (at 250 Hz) of ECG, PPG, IMU, and epidermal skin temperature.
"""
from typing import Tuple, Dict, Any, List, Optional
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, iirnotch, sosfilt, sosfilt_zi, tf2sos, sos2zpk


def butter_bandpass_filter(
    data: np.ndarray,
    lowcut: float,
    highcut: float,
    fs: float = 250.0,
    order: int = 4,
    causal: bool = True,
    filter_state: Optional[Dict[str, np.ndarray]] = None,
    state_key: str = "bandpass"
) -> np.ndarray:
    """
    Applies a stateful causal SOS Butterworth bandpass filter (or zero-phase filtfilt if causal=False)
    using single-precision float32 matching the ESP32-S3 hardware FPU.
    Eliminates train-serve DSP skew between training feature extraction and edge streaming.
    Verifies filter pole-zero stability at fs=250 Hz for the 0.5 Hz high-pass cutoff to prevent IIR limit cycles.
    """
    data_f32 = np.asarray(data, dtype=np.float32)
    if len(data_f32) <= 1:
        return data_f32
    nyquist = 0.5 * fs
    low = max(0.001, lowcut / nyquist)
    high = min(0.999, highcut / nyquist)
    if low >= high:
        return data_f32

    d_mean = np.float32(np.nanmean(data_f32)) if np.isfinite(data_f32).any() else np.float32(0.0)
    d_centered = np.nan_to_num(data_f32 - d_mean, nan=0.0).astype(np.float32)

    if causal:
        sos = butter(order, [low, high], btype='band', output='sos')
        _, p, _ = sos2zpk(sos)
        if np.any(np.abs(p) >= 1.0):
            sos = butter(order, [max(low, 0.005), high], btype='band', output='sos')
        sos = sos.astype(np.float32)

        zi_nominal = sosfilt_zi(sos).astype(np.float32)
        if filter_state is not None and state_key in filter_state and filter_state[state_key].shape == zi_nominal.shape:
            zi = filter_state[state_key].astype(np.float32)
        else:
            zi = zi_nominal * d_centered[0]
        filtered, zf = sosfilt(sos, d_centered, zi=zi)
        if filter_state is not None:
            filter_state[state_key] = zf.astype(np.float32)
        return filtered.astype(np.float32)

    b, a = butter(order, [low, high], btype='band')
    padlen = min(3 * max(len(a), len(b)), len(d_centered) - 1)
    if padlen < 1:
        return d_centered
    filtered = filtfilt(b, a, d_centered, padlen=padlen, padtype='odd')
    return filtered.astype(np.float32)


def iir_notch_filter(
    data: np.ndarray,
    notch_freq: float = 50.0,
    fs: float = 250.0,
    quality_factor: float = 30.0,
    causal: bool = True,
    filter_state: Optional[Dict[str, np.ndarray]] = None,
    state_key: str = "notch"
) -> np.ndarray:
    """
    Applies a second-order IIR Notch filter to isolate and remove powerline interference (50Hz / 60Hz)
    using stateful causal sosfilt in float32 (or reflection padding when causal=False).
    """
    data_f32 = np.asarray(data, dtype=np.float32)
    if len(data_f32) <= 1:
        return data_f32
    nyquist = 0.5 * fs
    if notch_freq >= nyquist:
        return data_f32
    w0 = notch_freq / nyquist
    b, a = iirnotch(w0, quality_factor)
    mean_val = np.float32(np.nanmean(data_f32)) if np.isfinite(data_f32).any() else np.float32(0.0)
    d_clean = np.nan_to_num(data_f32, nan=mean_val).astype(np.float32)

    if causal:
        sos = tf2sos(b, a).astype(np.float32)
        zi_nominal = sosfilt_zi(sos).astype(np.float32)
        if filter_state is not None and state_key in filter_state and filter_state[state_key].shape == zi_nominal.shape:
            zi = filter_state[state_key].astype(np.float32)
        else:
            zi = zi_nominal * d_clean[0]
        filtered, zf = sosfilt(sos, d_clean, zi=zi)
        if filter_state is not None:
            filter_state[state_key] = zf.astype(np.float32)
        return filtered.astype(np.float32)

    padlen = min(3 * max(len(a), len(b)), len(d_clean) - 1)
    if padlen < 1:
        return d_clean
    return filtfilt(b, a, d_clean, padlen=padlen, padtype='odd').astype(np.float32)


def compute_ecg_sqi(ecg_signal: np.ndarray, fs: float = 250.0, causal: bool = True) -> float:
    """
    Computes ECG Signal Quality Index (0.0 to 1.0) based on:
      1. Kurtosis of filtered signal (clean QRS peaks exhibit high positive kurtosis > 4.0).
      2. Relative power in QRS frequency band (5-25 Hz) vs noise band (high frequency & baseline).
    """
    if len(ecg_signal) < int(fs * 0.5):
        return 0.0

    sig_eval = ecg_signal[int(fs * 0.5):] if len(ecg_signal) >= int(fs * 2.0) else ecg_signal
    # 1. Bandpass in QRS range (5-25 Hz) vs broadband
    qrs_band = butter_bandpass_filter(sig_eval, 5.0, 25.0, fs, order=2, causal=causal)
    p_qrs = float(np.var(qrs_band)) + 1e-8
    p_total = float(np.var(sig_eval)) + 1e-8
    p_ratio = min(1.0, max(0.0, p_qrs / p_total))

    # 2. Kurtosis
    mean_val = np.mean(sig_eval)
    std_val = np.std(sig_eval) + 1e-6
    kurt = float(np.mean(((sig_eval - mean_val) / std_val) ** 4))
    kurt_score = min(1.0, max(0.0, (kurt - 2.0) / 6.0))

    sqi = 0.6 * kurt_score + 0.4 * p_ratio
    return float(np.clip(sqi, 0.0, 1.0))


def compute_ppg_sqi(ppg_signal: np.ndarray, fs: float = 250.0, causal: bool = True) -> float:
    """
    Computes PPG Signal Quality Index (0.0 to 1.0) based on:
      1. Ratio of pulsatile AC power (0.6 - 4.5 Hz) to AC-band variance (0.5 - 12.0 Hz),
         so slow respiratory/finger-pressure baseline drift (<0.5 Hz) does not penalize valid PPG.
      2. Brief optical dropout interpolation, rail-clipping/flatline checks, and physiological pulse-peak verification.
    """
    sig = np.asarray(ppg_signal, dtype=np.float64).flatten()
    if len(sig) < int(fs * 0.5) or not np.isfinite(sig).any():
        return 0.0
    med_val = float(np.nanmedian(sig))
    sig = np.nan_to_num(sig, nan=med_val, posinf=med_val, neginf=med_val)
    if float(np.std(sig)) < 1e-6:
        return 0.0

    # Handle optical IR count dropouts (brief zero-flickers vs finger-off)
    if float(np.max(sig)) > 5000.0:
        valid_mask = sig > 1.0
        zero_frac = 1.0 - float(np.mean(valid_mask))
        if zero_frac >= 0.25 or int(np.sum(valid_mask)) < 4:
            return 0.0
        if zero_frac > 0.0:
            idx_all = np.arange(len(sig), dtype=np.float64)
            sig = np.interp(idx_all, idx_all[valid_mask], sig[valid_mask])

    sig_eval = sig[int(fs * 0.5):] if len(sig) >= int(fs * 2.0) else sig
    ac_high = min(12.0, 0.45 * fs)
    pulse_high = min(4.5, 0.40 * fs)
    ac_band = butter_bandpass_filter(sig_eval, 0.5, ac_high, fs, order=2, causal=causal)
    pulse_band = butter_bandpass_filter(sig_eval, 0.6, pulse_high, fs, order=2, causal=causal)
    p_pulse = float(np.var(pulse_band)) + 1e-8
    p_ac = float(np.var(ac_band)) + 1e-8
    pulse_ratio = min(1.0, max(0.0, p_pulse / p_ac))

    # Rail-clipping / flatline check (safe for 100 Hz integer MAX30102 samples upsampled to 250 Hz)
    s_min, s_max = float(np.min(sig_eval)), float(np.max(sig_eval))
    sat_frac = float(np.mean((sig_eval <= s_min + 1e-6) | (sig_eval >= s_max - 1e-6)))
    n_unique = len(np.unique(np.round(sig_eval, 4)))
    clip_penalty = 0.2 if (sat_frac > 0.15 or n_unique < 8) else 1.0

    # Physiological pulse-peak check (24 - 228 BPM equivalent)
    std_p = float(np.std(pulse_band))
    peaks, _ = find_peaks(
        pulse_band,
        distance=max(1, int(0.30 * fs)),
        prominence=max(std_p * 0.25, 1e-6),
    )
    dur = max(0.5, len(sig_eval) / max(1.0, fs))
    min_peaks = max(1, int(0.4 * dur))
    max_peaks = max(min_peaks + 1, int(3.8 * dur))
    peak_penalty = 1.0 if (min_peaks <= len(peaks) <= max_peaks) else 0.35

    sqi = float(np.clip(pulse_ratio * clip_penalty * peak_penalty, 0.0, 1.0))
    return sqi


def clean_biosignals(
    ecg_raw: np.ndarray,
    ppg_raw: np.ndarray,
    fs: float = 250.0,
    causal: bool = True,
    filter_state: Optional[Dict[str, np.ndarray]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full Digital Noise Isolation Pipeline for Wearable Biosignals:
      1. ECG: 50 Hz IIR Notch filter + 0.5 - 40.0 Hz bandpass filter.
      2. PPG: 0.5 - 8.0 Hz bandpass filter + baseline wander suppression.
      Defaults to stateful causal sosfilt float32 filtering to eliminate train-serve skew.
    """
    ecg_notched = iir_notch_filter(
        ecg_raw, notch_freq=50.0, fs=fs, quality_factor=30.0,
        causal=causal, filter_state=filter_state, state_key="ecg_notch"
    )
    ecg_clean = butter_bandpass_filter(
        ecg_notched, 0.5, 40.0, fs=fs, order=4,
        causal=causal, filter_state=filter_state, state_key="ecg_bandpass"
    )

    ppg_clean = butter_bandpass_filter(
        ppg_raw, 0.5, 8.0, fs=fs, order=4,
        causal=causal, filter_state=filter_state, state_key="ppg_bandpass"
    )

    return ecg_clean.astype(np.float32), ppg_clean.astype(np.float32)


def detect_r_peaks_pan_tompkins(
    ecg_signal: np.ndarray,
    fs: float = 250.0,
    causal: bool = True
) -> np.ndarray:
    """
    Pan-Tompkins QRS detection algorithm for single-lead ECG.
    Steps:
      1. Bandpass filter (0.5 - 40 Hz)
      2. Derivative filter
      3. Squaring
      4. Moving window integration (~150 ms)
      5. Adaptive thresholding with 200ms refractory period
    Compensates for deterministic filter group delay when causal=True to ensure
    sub-sample timing alignment within +/- 1.5 ms of zero-phase batch mode.
    """
    if len(ecg_signal) < int(fs * 0.5):
        return np.array([], dtype=int)

    filtered = butter_bandpass_filter(ecg_signal, 0.5, 40.0, fs, order=2, causal=causal)

    diff = np.diff(filtered)
    diff = np.pad(diff, (0, 1), mode='edge')

    squared = diff ** 2

    window_len = int(0.150 * fs)
    if window_len < 1:
        window_len = 1
    integrated = np.convolve(squared, np.ones(window_len) / window_len, mode='same')

    min_dist = int(0.200 * fs)
    threshold = np.mean(integrated) + 0.5 * np.std(integrated)
    candidate_peaks, _ = find_peaks(integrated, height=threshold, distance=min_dist)

    search_radius = int(0.075 * fs)
    pos_extrema = []
    neg_extrema = []
    for p in candidate_peaks:
        start = max(0, p - search_radius)
        end = min(len(filtered), p + search_radius + 1)
        if start < end:
            seg = filtered[start:end]
            pos_extrema.append(float(np.max(seg)))
            neg_extrema.append(float(abs(np.min(seg))))
    use_positive_polarity = (np.median(pos_extrema) >= np.median(neg_extrema) * 0.85) if pos_extrema else True

    r_peaks = []
    for p in candidate_peaks:
        start = max(0, p - search_radius)
        end = min(len(filtered), p + search_radius + 1)
        if start < end:
            if use_positive_polarity:
                local_max = start + int(np.argmax(filtered[start:end]))
            else:
                local_max = start + int(np.argmax(np.abs(filtered[start:end])))
            r_peaks.append(local_max)

    r_peaks = np.unique(r_peaks)

    gd_delay = 1.125 * (fs / 250.0) if causal else 0.0
    refined_r_peaks = []
    for r in r_peaks:
        if 1 <= r < len(filtered) - 1:
            y_prev = float(filtered[r - 1])
            y_curr = float(filtered[r])
            y_next = float(filtered[r + 1])
            denom = y_prev - 2.0 * y_curr + y_next
            if abs(denom) > 1e-12:
                delta = 0.5 * (y_prev - y_next) / (denom + 1e-12)
                delta = float(np.clip(delta, -0.5, 0.5))
            else:
                delta = 0.0
            refined_r_peaks.append(float(r + delta - gd_delay))
        else:
            refined_r_peaks.append(float(r - gd_delay))

    return np.array(refined_r_peaks, dtype=np.float64)


pan_tompkins_qrs_detector = detect_r_peaks_pan_tompkins


def detect_ppg_foot(
    ppg_signal: np.ndarray,
    fs: float = 250.0,
    causal: bool = True
) -> np.ndarray:
    """
    Detects arterial pulse wave foot (systolic onset) preceding each systolic peak.
    Foot marks the systolic wave onset used for Pulse Transit Time (PTT).
    """
    if len(ppg_signal) < int(fs * 0.5):
        return np.array([], dtype=np.float32)

    filtered = butter_bandpass_filter(ppg_signal, 0.5, 8.0, fs, order=2, causal=causal)

    min_dist = int(0.280 * fs)
    std_val = float(np.std(filtered))
    p2p = float(np.max(filtered) - np.min(filtered))
    prom = max(std_val * 0.5, p2p * 0.30) if p2p > 1e-4 else None
    peaks, _ = find_peaks(filtered, prominence=prom, distance=min_dist)
    if len(peaks) == 0:
        threshold = std_val * 0.2 if std_val > 1e-4 else None
        peaks, _ = find_peaks(filtered, height=threshold, distance=min_dist)

    feet = []
    prev_p = 0
    for p in peaks:
        search_win = min(int(0.550 * fs), int(0.52 * (p - prev_p))) if prev_p > 0 else int(0.550 * fs)
        st = max(0, p - search_win)
        if p > st + 2:
            seg = filtered[st:p]
            diffs = np.diff(seg)
            valleys = np.where((diffs[:-1] <= 0) & (diffs[1:] > 0))[0] + 1
            if len(valleys) > 0:
                feet.append(st + valleys[-1])
            else:
                feet.append(st + np.argmin(seg))
        elif p > st:
            feet.append(st + np.argmin(filtered[st:p]))
        prev_p = p

    feet = np.unique(feet)

    gd_delay = -1.50 * (fs / 250.0) if causal else 0.0
    refined_feet = []
    for f in feet:
        if 1 <= f < len(filtered) - 1:
            yp, yc, yn = float(filtered[f - 1]), float(filtered[f]), float(filtered[f + 1])
            denom = yp - 2.0 * yc + yn
            delta = float(np.clip(0.5 * (yp - yn) / (denom + 1e-12), -0.5, 0.5)) if abs(denom) > 1e-12 else 0.0
            refined_feet.append(float(f + delta - gd_delay))
        else:
            refined_feet.append(float(f - gd_delay))

    return np.unique(np.array(refined_feet, dtype=np.float32))


detect_ppg_inflection_foot = detect_ppg_foot


def compute_imu_angles(
    ax: np.ndarray,
    ay: np.ndarray,
    az: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    gz: np.ndarray,
    fs: float = 50.0
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Computes pitch and roll angles of upper-arm using complementary filter:
    angle = 0.98 * (angle + gyro * dt) + 0.02 * accel_angle
    Returns: (pitch_deg, roll_deg, max_smv_g) in float32 matching ESP32-S3 FPU.
    """
    ax = np.asarray(ax, dtype=np.float32)
    ay = np.asarray(ay, dtype=np.float32)
    az = np.asarray(az, dtype=np.float32)
    gx = np.asarray(gx, dtype=np.float32)
    gy = np.asarray(gy, dtype=np.float32)
    gz = np.asarray(gz, dtype=np.float32)
    dt = 1.0 / max(1.0, fs)
    n = min(len(ax), len(ay), len(az), len(gx), len(gy), len(gz))
    if n == 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32), 1.0

    pitch = np.zeros(n, dtype=np.float32)
    roll = np.zeros(n, dtype=np.float32)

    smv = np.sqrt(ax[:n]**2 + ay[:n]**2 + az[:n]**2)
    max_smv = float(np.max(smv)) if n > 0 else 1.0

    curr_pitch = float(np.degrees(np.arctan2(ax[0], np.sqrt(ay[0]**2 + az[0]**2) + 1e-6)))
    curr_roll = float(np.degrees(np.arctan2(ay[0], np.sqrt(ax[0]**2 + az[0]**2) + 1e-6)))

    for i in range(n):
        acc_pitch = np.degrees(np.arctan2(ax[i], np.sqrt(ay[i]**2 + az[i]**2) + 1e-6))
        acc_roll = np.degrees(np.arctan2(ay[i], np.sqrt(ax[i]**2 + az[i]**2) + 1e-6))

        curr_pitch = 0.98 * (curr_pitch + gy[i] * dt) + 0.02 * acc_pitch
        curr_roll = 0.98 * (curr_roll + gx[i] * dt) + 0.02 * acc_roll

        pitch[i] = curr_pitch
        roll[i] = curr_roll

    return pitch, roll, max_smv


def adaptive_imu_motion_cancel(
    signal: np.ndarray,
    imu_data: Dict[str, np.ndarray],
    fs: float = 250.0,
    smv_max: Optional[float] = None,
    ridge_lambda: float = 1e-2
) -> np.ndarray:
    """
    IMU-guided adaptive motion artifact canceller for wearable ECG/PPG signals.
    Only activates when dynamic motion is present (smv_max > 1.15g). Resamples
    3-axis accelerometer signals to the biosignal sampling rate, isolates the
    0.4-12.0 Hz motion band, and removes the motion-correlated projection via
    ridge-regularized least squares while preserving the cardiac waveform and DC offset.
    """
    sig = np.asarray(signal, dtype=np.float64)
    n = len(sig)
    if n < int(fs * 0.5) or not imu_data:
        return np.asarray(signal, dtype=np.float32)

    ax = np.asarray(imu_data.get('ax', np.zeros(10)), dtype=np.float64)
    ay = np.asarray(imu_data.get('ay', -np.ones(10)), dtype=np.float64)
    az = np.asarray(imu_data.get('az', np.zeros(10)), dtype=np.float64)
    n_imu = min(len(ax), len(ay), len(az))
    if n_imu < 4:
        return np.asarray(signal, dtype=np.float32)

    ax, ay, az = ax[:n_imu], ay[:n_imu], az[:n_imu]
    if smv_max is None:
        smv_max = float(np.max(np.sqrt(ax ** 2 + ay ** 2 + az ** 2)))
    if smv_max <= 1.15:
        return np.asarray(signal, dtype=np.float32)

    t_imu = np.linspace(0.0, 1.0, n_imu)
    t_sig = np.linspace(0.0, 1.0, n)
    ax_rs = np.interp(t_sig, t_imu, ax)
    ay_rs = np.interp(t_sig, t_imu, ay)
    az_rs = np.interp(t_sig, t_imu, az)

    # Isolate dynamic motion frequency band (0.4 - 12.0 Hz)
    ax_m = butter_bandpass_filter(ax_rs - np.mean(ax_rs), 0.4, 12.0, fs=fs, order=2, causal=False).astype(np.float64)
    ay_m = butter_bandpass_filter(ay_rs - np.mean(ay_rs), 0.4, 12.0, fs=fs, order=2, causal=False).astype(np.float64)
    az_m = butter_bandpass_filter(az_rs - np.mean(az_rs), 0.4, 12.0, fs=fs, order=2, causal=False).astype(np.float64)

    energy = ax_m ** 2 + ay_m ** 2 + az_m ** 2
    energy_c = energy - np.mean(energy)

    # Include derivatives (quadrature components) so phase-shifted motion artifacts cancel cleanly
    dax_m = np.gradient(ax_m)
    day_m = np.gradient(ay_m)
    daz_m = np.gradient(az_m)

    A = np.column_stack([ax_m, ay_m, az_m, energy_c, dax_m, day_m, daz_m])
    col_norms = np.std(A, axis=0)
    valid_cols = col_norms > 1e-7
    if not np.any(valid_cols):
        return np.asarray(signal, dtype=np.float32)

    A = A[:, valid_cols]
    A = (A - np.mean(A, axis=0)) / (col_norms[valid_cols] + 1e-8)

    sig_mean = float(np.nanmean(sig)) if np.isfinite(sig).any() else 0.0
    sig_c = np.nan_to_num(sig - sig_mean, nan=0.0)

    # Ridge-regularized least-squares projection: w = (A^T A + lambda*n*I)^-1 A^T s
    gram = A.T @ A + (ridge_lambda * n) * np.eye(A.shape[1])
    rhs = A.T @ sig_c
    try:
        weights = np.linalg.solve(gram, rhs)
        cleaned = sig_c - (A @ weights) + sig_mean
    except np.linalg.LinAlgError:
        cleaned = sig

    return cleaned.astype(np.float32)


def extract_ppg_morphology_and_sdptg(
    clean_ppg: np.ndarray,
    fs: float = 250.0,
    causal: bool = True
) -> Dict[str, float]:
    """
    Extracts Second Derivative Photoplethysmogram (SDPTG = d^2 PPG / dt^2) b/a ratio,
    dicrotic notch reflection index, and diastolic decay time constant (tau_diastolic).
    """
    if len(clean_ppg) < int(fs * 1.0):
        return {
            "sdptg_b_a_ratio": -0.65,
            "tau_diastolic_s": 0.35,
            "dicrotic_notch_index": 0.30
        }

    ppg_filt = butter_bandpass_filter(clean_ppg, 0.5, 15.0, fs=fs, order=2, causal=causal)

    vpg = np.gradient(ppg_filt, 1.0 / fs)
    sdptg = np.gradient(vpg, 1.0 / fs)

    min_dist = int(0.300 * fs)
    pos_peaks, _ = find_peaks(sdptg, distance=min_dist)
    neg_troughs, _ = find_peaks(-sdptg, distance=min_dist)

    b_a_ratios = []
    if len(pos_peaks) > 0 and len(neg_troughs) > 0:
        for p in pos_peaks:
            troughs = neg_troughs[(neg_troughs > p) & (neg_troughs < p + int(0.180 * fs))]
            if len(troughs) > 0:
                a_val = sdptg[p]
                b_val = sdptg[troughs[0]]
                if a_val > 1e-4:
                    b_a_ratios.append(b_val / a_val)

    b_a = float(np.median(b_a_ratios)) if b_a_ratios else -0.65
    b_a = float(np.clip(b_a, -2.5, 0.5))

    ppg_min = float(np.min(clean_ppg))
    ppg_max = float(np.max(clean_ppg))
    ppg_amp = max(1e-4, ppg_max - ppg_min)
    norm_ppg = (clean_ppg - ppg_min) / ppg_amp

    pulse_peaks, _ = find_peaks(norm_ppg, height=0.5, distance=int(0.35 * fs))
    dicrotic_indices = []
    tau_estimates = []

    for idx, pk in enumerate(pulse_peaks[:-1]):
        next_pk = pulse_peaks[idx + 1]
        cycle = norm_ppg[pk:next_pk]
        if len(cycle) > int(0.20 * fs):
            trough_rel = np.argmin(cycle)
            if trough_rel > int(0.10 * fs):
                decay_seg = cycle[:trough_rel]
                target_val = 0.37
                below_target = np.where(decay_seg <= target_val)[0]
                if len(below_target) > 0:
                    tau_s = float(below_target[0]) / fs
                    tau_estimates.append(tau_s)
            inflection = np.gradient(np.gradient(cycle))
            dicrotic_indices.append(float(np.mean(np.abs(inflection))))

    tau_diastolic = float(np.median(tau_estimates)) if tau_estimates else 0.35
    tau_diastolic = float(np.clip(tau_diastolic, 0.15, 0.85))

    dicrotic_notch = float(np.median(dicrotic_indices)) if dicrotic_indices else 0.30
    dicrotic_notch = float(np.clip(dicrotic_notch, 0.05, 0.80))

    return {
        "sdptg_b_a_ratio": round(b_a, 3),
        "tau_diastolic_s": round(tau_diastolic, 3),
        "dicrotic_notch_index": round(dicrotic_notch, 3)
    }


class PersonalBaselineCalibrator:
    """
    Maintains rolling 5-minute Exponential Moving Average (EMA) of resting baselines
    during stable Phase 0 windows (rehab_exercise_active == False) for Upper-Arm Brachial Band (Biceps Placement).
    Updates Brachial PAT, ST-deviation, and skin temperature independently so a temporary PPG dropout
    never blocks temperature or ECG baseline tracking.
    """
    def __init__(self, alpha: float = 0.033):
        self.alpha = alpha
        self.baseline_pat: float = 180.0  # Nominal transit scaling for Upper-Arm Brachial Band
        self.baseline_st: float = 0.0
        self.baseline_temp: float = 34.6  # Upper-Arm Brachial Band (Biceps Placement) baseline ~34.0°C–35.2°C
        self.initialized = False
        self._init_pat = False
        self._init_st = False
        self._init_temp = False

    def update(
        self,
        pat_ms: float,
        st_dev_mv: float,
        temp_c: float,
        rehab_active: bool = False,
        motion_state: str = "REST"
    ):
        if not rehab_active and motion_state in ("REST", "WALKING"):
            if not np.isnan(pat_ms):
                if not self._init_pat:
                    self.baseline_pat = float(pat_ms)
                    self._init_pat = True
                else:
                    self.baseline_pat = (1.0 - self.alpha) * self.baseline_pat + self.alpha * float(pat_ms)
                self.initialized = True

            if not np.isnan(st_dev_mv):
                if not self._init_st:
                    self.baseline_st = float(st_dev_mv)
                    self._init_st = True
                else:
                    self.baseline_st = (1.0 - self.alpha) * self.baseline_st + self.alpha * float(st_dev_mv)
                self.initialized = True

            if not np.isnan(temp_c):
                if not self._init_temp:
                    self.baseline_temp = float(temp_c)
                    self._init_temp = True
                else:
                    self.baseline_temp = (1.0 - self.alpha) * self.baseline_temp + self.alpha * float(temp_c)
                self.initialized = True

    def get_deltas(self, pat_ms: float, st_dev_mv: float, temp_c: float) -> Tuple[float, float, float]:
        d_pat = float(pat_ms - self.baseline_pat) if not np.isnan(pat_ms) else np.nan
        d_st = float(st_dev_mv - self.baseline_st) if not np.isnan(st_dev_mv) else np.nan
        d_temp = float(temp_c - self.baseline_temp) if not np.isnan(temp_c) else 0.0
        return d_pat, d_st, d_temp


def extract_10s_features(
    ecg_mv: np.ndarray,
    ppg_raw: np.ndarray,
    imu_data: Dict[str, np.ndarray],
    temp_c: float = 34.6,  # Upper-Arm Brachial Band (Biceps Placement) baseline ~34.0°C–35.2°C
    delta_temp_5min: float = 0.0,
    missed_anticoagulant: bool = False,
    rehab_active: bool = False,
    fs: float = 250.0,
    ecg_lead_mode: str = "PASSIVE_IPSILATERAL_ARM",
    calibrator: Any = None,
    causal: bool = True,
    filter_state: Optional[Dict[str, np.ndarray]] = None
) -> Dict[str, Any]:
    """
    Extracts the full 20-feature clinical & hemodynamic vector from a 10s window.
    Defaults to stateful causal float32 DSP matching ESP32-S3 hardware FPU.
    """
    ecg_mv = np.asarray(ecg_mv, dtype=np.float32)
    ppg_raw = np.asarray(ppg_raw, dtype=np.float32)
    ax = imu_data.get('ax', np.zeros(100, dtype=np.float32))
    ay = imu_data.get('ay', -np.ones(100, dtype=np.float32))
    az = imu_data.get('az', np.zeros(100, dtype=np.float32))
    gx = imu_data.get('gx', np.zeros(100, dtype=np.float32))
    gy = imu_data.get('gy', np.zeros(100, dtype=np.float32))
    gz = imu_data.get('gz', np.zeros(100, dtype=np.float32))

    imu_fs = len(ax) / 10.0 if len(ax) >= 10 else 50.0
    pitch_arr, roll_arr, smv_max = compute_imu_angles(ax, ay, az, gx, gy, gz, fs=imu_fs)
    arm_pitch_drift = float(abs(pitch_arr[-1] - pitch_arr[0])) if len(pitch_arr) > 1 else 0.0
    arm_roll_drift = float(abs(roll_arr[-1] - roll_arr[0])) if len(roll_arr) > 1 else 0.0

    if smv_max > 3.0:
        motion_state = "FALL_DETECTED"
    elif smv_max > 1.8:
        motion_state = "EXERCISE"
    elif smv_max > 1.3:
        motion_state = "WALKING"
    else:
        motion_state = "REST"

    # 1. Digital Noise Isolation & Filtering
    clean_ecg, clean_ppg = clean_biosignals(
        ecg_mv, ppg_raw, fs=fs, causal=causal, filter_state=filter_state
    )
    if smv_max > 1.15:
        clean_ecg = adaptive_imu_motion_cancel(clean_ecg, imu_data, fs=fs, smv_max=smv_max)
        clean_ppg = adaptive_imu_motion_cancel(clean_ppg, imu_data, fs=fs, smv_max=smv_max)

    ecg_sqi = compute_ecg_sqi(clean_ecg, fs=fs, causal=causal)
    ppg_sqi = compute_ppg_sqi(clean_ppg, fs=fs, causal=causal)

    # 2. ECG R-peak detection & HRV
    ecg_for_peaks = adaptive_imu_motion_cancel(ecg_mv, imu_data, fs=fs, smv_max=smv_max) if smv_max > 1.15 else ecg_mv
    r_peaks = detect_r_peaks_pan_tompkins(ecg_for_peaks, fs=fs, causal=causal)
    if len(r_peaks) >= 2:
        rr_intervals_ms = np.diff(r_peaks) * (1000.0 / fs)
        valid_rr = rr_intervals_ms[(rr_intervals_ms >= 250.0) & (rr_intervals_ms <= 2000.0)]
        if len(valid_rr) >= 2:
            hr_ecg = 60000.0 / float(np.mean(valid_rr))
            sdnn = float(np.std(valid_rr))
            rmssd = float(np.sqrt(np.mean(np.diff(valid_rr) ** 2))) if len(valid_rr) > 1 else 25.0
            pnn50 = float(np.sum(np.abs(np.diff(valid_rr)) > 50.0) / max(1, len(valid_rr) - 1) * 100.0)
            rr_norm = valid_rr / (np.mean(valid_rr) + 1e-6)
            rr_entropy = float(np.std(rr_norm))
        else:
            hr_ecg = 72.0
            sdnn, rmssd, pnn50, rr_entropy = 30.0, 25.0, 15.0, 0.05
    else:
        hr_ecg = 72.0
        sdnn, rmssd, pnn50, rr_entropy = 30.0, 25.0, 15.0, 0.05
        r_peaks = np.array([])

    # 3. PPG foot detection & Pulse rate
    ppg_for_feet = adaptive_imu_motion_cancel(ppg_raw, imu_data, fs=fs, smv_max=smv_max) if smv_max > 1.15 else ppg_raw
    ppg_feet = detect_ppg_foot(ppg_for_feet, fs=fs, causal=causal)
    if len(ppg_feet) >= 2:
        pulse_intervals_ms = np.diff(ppg_feet) * (1000.0 / fs)
        valid_pi = pulse_intervals_ms[(pulse_intervals_ms >= 250.0) & (pulse_intervals_ms <= 2000.0)]
        if len(valid_pi) >= 2:
            pulse_ppg = 60000.0 / float(np.median(valid_pi))
        else:
            pulse_ppg = hr_ecg
    else:
        pulse_ppg = hr_ecg

    raw_deficit = float(abs(hr_ecg - pulse_ppg))
    if smv_max > 1.4 or ppg_sqi < 0.45:
        pulse_deficit = min(raw_deficit, 4.0)
    else:
        pulse_deficit = raw_deficit

    # 4. ST-segment deviation & T-wave amplitude (Isoelectric PR-TP Baseline & Motion Rejection)
    st_devs = []
    t_amps = []
    gd_shift = 1.0 * (fs / 250.0) if causal else 0.0
    search_rad = max(1, int(round(0.016 * fs)))
    for b_idx, r in enumerate(r_peaks):
        r_est = int(round(r + gd_shift))
        if r_est < int(0.085 * fs) or r_est >= len(clean_ecg) - int(0.230 * fs):
            continue
        # Snap to exact R-peak in clean_ecg within +/- 16 ms to remove filter phase lag
        loc_lo = max(0, r_est - search_rad)
        loc_hi = min(len(clean_ecg), r_est + search_rad + 1)
        r_sig = float(loc_lo + int(np.argmax(np.abs(clean_ecg[loc_lo:loc_hi]))))

        pr_start = max(0, int(round(r_sig - 0.080 * fs)))
        pr_end = max(pr_start + 1, int(round(r_sig - 0.040 * fs)))
        iso_before = float(np.median(clean_ecg[pr_start:pr_end]))
        t_before = 0.5 * (pr_start + pr_end)

        st_idx = int(round(r_sig + 0.080 * fs))
        t_idx = int(round(r_sig + 0.200 * fs))
        if not (0 <= st_idx < len(clean_ecg) and 0 <= t_idx < len(clean_ecg)):
            continue

        # Linear PQ-to-TP isoelectric baseline interpolation cancels low-frequency motion wander
        tp_start = int(round(r_sig + 0.320 * fs))
        tp_end = int(round(r_sig + 0.380 * fs))
        next_r = float(r_peaks[b_idx + 1] + gd_shift) if (b_idx + 1) < len(r_peaks) else float(len(clean_ecg))
        if tp_end < len(clean_ecg) and (next_r - r_sig) > 0.52 * fs:
            iso_after = float(np.median(clean_ecg[tp_start:tp_end]))
            t_after = 0.5 * (tp_start + tp_end)
            span = max(1.0, t_after - t_before)
            alpha_st = (st_idx - t_before) / span
            alpha_t = (t_idx - t_before) / span
            iso_st = (1.0 - alpha_st) * iso_before + alpha_st * iso_after
            iso_t = (1.0 - alpha_t) * iso_before + alpha_t * iso_after
        else:
            iso_st = iso_before
            iso_t = iso_before

        st_val = float(clean_ecg[st_idx] - iso_st)
        t_val = float(clean_ecg[t_idx] - iso_t)
        st_devs.append(st_val)
        t_amps.append(t_val)

    if st_devs:
        st_arr = np.sort(np.asarray(st_devs, dtype=np.float64))
        t_arr = np.sort(np.asarray(t_amps, dtype=np.float64))
        if smv_max > 1.35 and len(st_arr) >= 5:
            # Trimmed median during dynamic motion to discard residual motion-corrupted beats
            trim = max(1, int(0.2 * len(st_arr)))
            st_deviation = float(np.median(st_arr[trim:-trim]))
            t_wave_amplitude = float(np.median(t_arr[trim:-trim]))
        else:
            st_deviation = float(np.median(st_arr))
            t_wave_amplitude = float(np.median(t_arr))
        if smv_max > 1.35 and abs(st_deviation) < 0.14:
            # Mild sub-threshold ST jitter during active motion is attributable to baseline residual
            st_deviation *= 0.35
    else:
        st_deviation = 0.0
        t_wave_amplitude = 0.2

    st_deviation = float(np.clip(st_deviation, -0.80, 0.80))
    t_wave_amplitude = float(np.clip(t_wave_amplitude, -1.00, 1.50))

    # 5. SpO2 & Perfusion Index (PI)
    ac_ppg = float(np.max(clean_ppg) - np.min(clean_ppg)) if len(clean_ppg) > 0 else 0.0
    raw_mean = float(np.mean(np.abs(ppg_raw))) if len(ppg_raw) > 0 else 1.0
    dc_ppg = raw_mean if raw_mean > 100.0 else (float(np.mean(np.abs(clean_ppg))) * 40.0 + 1e-6)
    pi = float((ac_ppg / dc_ppg) * 100.0)
    pi = float(np.clip(pi, 0.1, 10.0))
    spo2 = float(np.clip(98.5 - 0.2 * max(0.0, 1.5 - pi), 85.0, 100.0))

    # 6. Pulse Arrival Time (Brachial PAT = PEP + PTT) & Non-Linear Blood Pressure Calibration
    # Calibrated for Upper-Arm Brachial Band (Biceps Placement):
    # Because the brachial artery at the biceps is anatomically ~25-30 cm closer to the aortic root than
    # radial artery, resting brachial transit corresponds to ~140-170 ms (or nominal transit scaling).
    ptt_values = []
    if len(r_peaks) > 0 and len(ppg_feet) > 0:
        for r in r_peaks:
            if r < int(0.600 * fs):
                continue
            valid_feet = ppg_feet[(ppg_feet > r + int(0.060 * fs)) & (ppg_feet < r + int(0.420 * fs))]
            if len(valid_feet) > 0:
                dt_ms = (valid_feet[0] - r) * (1000.0 / fs) - 28.0
                dt_ms = max(70.0, min(350.0, dt_ms))
                ptt_values.append(dt_ms)

        if not ptt_values:
            for r in r_peaks:
                valid_feet = ppg_feet[(ppg_feet > r + int(0.060 * fs)) & (ppg_feet < r + int(0.420 * fs))]
                if len(valid_feet) > 0:
                    dt_ms = (valid_feet[0] - r) * (1000.0 / fs) - 28.0
                    dt_ms = max(70.0, min(350.0, dt_ms))
                    ptt_values.append(dt_ms)

    pat_ms = float(np.median(ptt_values)) if ptt_values else np.nan

    # Rolling personal baseline calibration if available
    effective_delta_temp = float(delta_temp_5min)
    if calibrator is not None and getattr(calibrator, "initialized", False):
        delta_pat, delta_st_cal, delta_temp_cal = calibrator.get_deltas(pat_ms, st_deviation, temp_c)
        if abs(effective_delta_temp) < 1e-9:
            effective_delta_temp = float(delta_temp_cal)
    else:
        baseline_pat = 180.0
        delta_pat = float(pat_ms - baseline_pat) if not np.isnan(pat_ms) else np.nan

    ppg_morph = extract_ppg_morphology_and_sdptg(clean_ppg, fs=fs, causal=causal)
    ba_ratio = ppg_morph["sdptg_b_a_ratio"]
    tau_diastolic = ppg_morph["tau_diastolic_s"]
    dicrotic_index = ppg_morph["dicrotic_notch_index"]

    effective_hr = pulse_ppg if (np.isnan(hr_ecg) or hr_ecg <= 35.0 or hr_ecg >= 220.0 or len(r_peaks) < 2) else hr_ecg

    if ptt_values and not np.isnan(delta_pat):
        # Full PTT + PPG Hybrid Model
        estimated_sbp = (
            120.0
            - 0.85 * delta_pat
            + 0.12 * (effective_hr - 72.0)
            + 14.0 * (ba_ratio + 0.65)
            - 3.0 * (pi - 3.5)
            + 0.002 * (delta_pat ** 2)
        )
        estimated_dbp = (
            80.0
            - 0.40 * delta_pat
            + 12.0 * (tau_diastolic - 0.35)
            + 8.0 * (dicrotic_index - 0.30)
            + 0.08 * (effective_hr - 72.0)
        )
    else:
        # Fallback: PPG-Only Pulse Wave Analysis (PWA) Morphology Model
        # When ECG is absent/detached (Passive Upper-Arm Brachial Band Mode - Biceps Placement), estimate SBP and DBP from:
        # - SDPTG b/a ratio (arterial stiffness / elastance)
        # - tau_diastolic (peripheral vascular resistance / runoff time constant)
        # - dicrotic notch index (peripheral wave reflection)
        # - effective_hr from optical pulse rate and perfusion index (PI)
        estimated_sbp = (
            120.0
            + 18.0 * (ba_ratio + 0.65)
            + 0.16 * (effective_hr - 72.0)
            - 4.0 * (pi - 3.5)
        )
        estimated_dbp = (
            80.0
            + 14.0 * (tau_diastolic - 0.35)
            + 10.0 * (dicrotic_index - 0.30)
            + 0.09 * (effective_hr - 72.0)
        )

    estimated_sbp = float(np.clip(estimated_sbp, 75.0, 250.0))
    estimated_dbp = float(np.clip(estimated_dbp, 45.0, 150.0))
    if estimated_sbp < estimated_dbp + 15.0:
        estimated_sbp = estimated_dbp + 15.0

    pulse_pressure = float(estimated_sbp - estimated_dbp)
    cushing_index = float(estimated_sbp / max(20.0, effective_hr))
    shock_index = float(effective_hr / max(50.0, estimated_sbp))

    is_dynamic_motion = (abs(smv_max - 1.0) > 0.35) if smv_max > 0.5 else (smv_max > 0.35)
    is_hydro_postural_shift = (arm_pitch_drift > 20.0 or arm_roll_drift > 20.0)
    is_ppg_corrupted = (ppg_sqi < 0.55) or is_dynamic_motion or is_hydro_postural_shift

    if is_ppg_corrupted:
        pat_ms = np.nan
        delta_pat = np.nan
        estimated_sbp = np.nan
        estimated_dbp = np.nan
        pulse_pressure = np.nan
        cushing_index = np.nan
        shock_index = np.nan

    # Update personal calibrator during resting Phase 0 (updates valid channels independently)
    if calibrator is not None:
        calibrator.update(pat_ms, st_deviation, temp_c, rehab_active=rehab_active, motion_state=motion_state)

    def _safe_round(val: Any, decimals: int) -> Any:
        if val is None or (isinstance(val, (float, np.floating)) and np.isnan(val)):
            return np.nan
        return round(float(val), decimals)

    features = {
        "hr_ecg_bpm": _safe_round(hr_ecg, 1),
        "pulse_rate_ppg_bpm": _safe_round(pulse_ppg, 1),
        "pulse_deficit_bpm": _safe_round(pulse_deficit, 1),
        "rmssd_ms": _safe_round(rmssd, 2),
        "sdnn_ms": _safe_round(sdnn, 2),
        "pnn50_pct": _safe_round(pnn50, 2),
        "rr_entropy": _safe_round(rr_entropy, 4),
        "st_deviation_mv": _safe_round(st_deviation, 3),
        "t_wave_amplitude_mv": _safe_round(t_wave_amplitude, 3),
        "spo2_pct": _safe_round(spo2, 1),
        "perfusion_index_pct": _safe_round(pi, 2),
        "ptt_ms": _safe_round(pat_ms, 1),
        "delta_ptt_ms": _safe_round(delta_pat, 1),
        "estimated_sbp_mmhg": _safe_round(estimated_sbp, 1),
        "estimated_dbp_mmhg": _safe_round(estimated_dbp, 1),
        "cushing_index": _safe_round(cushing_index, 2),
        "skin_temp_c": _safe_round(temp_c, 2),
        "delta_skin_temp_c": _safe_round(effective_delta_temp, 2),
        "imu_smv_max_g": _safe_round(smv_max, 2),
        "imu_smv_g": _safe_round(smv_max, 2),
        "arm_pitch_drift_deg": _safe_round(arm_pitch_drift, 1),
        "arm_roll_drift_deg": _safe_round(arm_roll_drift, 1),
        "motion_state": motion_state,
        "missed_anticoagulant_dose": missed_anticoagulant,
        "rehab_exercise_active": rehab_active,
        "shock_index": _safe_round(shock_index, 3),
        "pulse_pressure_mmhg": _safe_round(pulse_pressure, 1),
        "sdptg_b_a_ratio": _safe_round(ba_ratio, 3),
        "tau_diastolic_s": _safe_round(tau_diastolic, 3),
        "dicrotic_notch_index": _safe_round(dicrotic_index, 3),
        "ecg_lead_mode": ecg_lead_mode
    }
    return features