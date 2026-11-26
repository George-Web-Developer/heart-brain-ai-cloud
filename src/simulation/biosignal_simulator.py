"""
biosignal_simulator.py
Physics-Based Multi-State Biosignal Simulator for Heart-Brain Axis Wearable.
Generates realistic 250 Hz multi-channel sensor streams across 5 clinical states:
  0: Normal Baseline
  1: Atrial Fibrillation (AFib)
  2: Myocardial Infarction (MI / Heart Attack)
  3: Acute Ischemic Stroke
  4: Acute Hemorrhagic Stroke (Intracerebral Hemorrhage - Cushing's Triad)
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
from src.preprocessing.signal_processor import extract_10s_features


CLASS_NAMES = [
    "Normal",
    "AFib",
    "MI",
    "Ischemic_Stroke",
    "Hemorrhagic_Stroke"
]


def _synthesize_ecg_beat(
    fs: float,
    st_dev: float = 0.0,
    t_inv: float = 0.0,
    has_p: bool = True
) -> np.ndarray:
    """Generates a single ECG beat waveform with configurable P, QRS, ST, and T components."""
    # Approximate beat duration ~ 0.8 sec (75 bpm base)
    duration = 0.8
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    beat = np.zeros_like(t)

    # P-wave: Gaussian peak around t = 0.15s
    if has_p:
        beat += 0.15 * np.exp(-((t - 0.15) ** 2) / (2 * (0.025 ** 2)))

    # Q-wave: negative dip at t = 0.28s
    beat -= 0.10 * np.exp(-((t - 0.28) ** 2) / (2 * (0.010 ** 2)))

    # R-peak: sharp positive spike at t = 0.30s
    beat += 1.20 * np.exp(-((t - 0.30) ** 2) / (2 * (0.012 ** 2)))

    # S-wave: negative dip at t = 0.33s
    beat -= 0.25 * np.exp(-((t - 0.33) ** 2) / (2 * (0.012 ** 2)))

    # ST segment shift (elevation or depression) between t = 0.36s and 0.45s
    st_mask = (t >= 0.35) & (t <= 0.46)
    beat[st_mask] += st_dev

    # T-wave: normal positive or inverted around t = 0.50s
    t_amp = 0.25 + t_inv
    beat += t_amp * np.exp(-((t - 0.50) ** 2) / (2 * (0.045 ** 2)))

    return beat


def _synthesize_ppg_pulse(fs: float, pi_scale: float = 1.0) -> np.ndarray:
    """Generates a single arterial PPG pulse wave."""
    duration = 0.8
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    # Systolic primary peak around t = 0.18s
    pulse = 1000.0 * pi_scale * np.exp(-((t - 0.18) ** 2) / (2 * (0.06 ** 2)))
    # Dicrotic notch / diastolic secondary wave around t = 0.35s
    pulse += 400.0 * pi_scale * np.exp(-((t - 0.35) ** 2) / (2 * (0.08 ** 2)))
    return pulse


def generate_synthetic_window(
    clinical_class: int,
    duration_sec: float = 10.0,
    fs: float = 250.0,
    exercise: Optional[bool] = None
) -> Dict[str, Any]:
    """
    Generates a synchronized 10-second multi-sensor window matching a clinical state
    for the Upper-Arm Brachial Band (Biceps Placement):
      0: Normal Baseline (HR ~72, Brachial PAT ~170-190ms, stable arm, temp ~34.0-35.2C)
      1: AFib (no P waves, chaotic RR intervals, pulse deficit)
      2: MI (ST elevation or subtle NSTEMI, inverted T, drop in skin temp and PI)
      3: Ischemic Stroke (AFib history, surge in BP / PTT drop, gradual arm drift)
      4: Hemorrhagic Stroke (Fall impact SMV > 3g, Cushing reflex: severe hypertension + bradycardia or hyper-acute tachycardia)
    """
    n_samples = int(duration_sec * fs)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)

    # Default parameters calibrated for Upper-Arm Brachial Band (Biceps Placement)
    st_dev = 0.0
    t_inv = 0.0
    has_p = True
    temp_c = 34.6 + np.random.normal(0, 0.10)  # Biceps baseline ~34.0°C–35.2°C
    delta_temp = 0.0
    pi_scale = 1.0
    ptt_offset_ms = 180.0
    missed_anticoagulant = False
    rehab_active = False

    # Kinematics (50 Hz, 500 samples)
    n_imu = int(duration_sec * 50.0)
    ax = np.random.normal(0.01, 0.02, n_imu)
    ay = np.random.normal(-0.98, 0.02, n_imu)
    az = np.random.normal(0.15, 0.02, n_imu)
    gx = np.random.normal(0.0, 0.1, n_imu)
    gy = np.random.normal(0.0, 0.1, n_imu)
    gz = np.random.normal(0.0, 0.1, n_imu)

    sub_p = np.random.rand()
    if exercise is False and clinical_class == 0 and 0.50 <= sub_p < 0.70:
        sub_p = 0.25  # Force resting phenotype
    elif exercise is True and clinical_class == 0:
        sub_p = 0.60  # Force exercise phenotype

    if clinical_class == 0:
        # Normal Baseline Sub-Phenotypes (Upper-Arm Brachial Band):
        if sub_p < 0.50:
            # (a) Resting Healthy (50%)
            base_hr = np.random.uniform(62.0, 78.0)
            temp_c = 34.6 + np.random.normal(0, 0.12)
            delta_temp = np.random.normal(0, 0.04)
            ptt_offset_ms = np.random.uniform(170.0, 195.0)
            rr_std = 0.03
        elif sub_p < 0.70:
            # (b) Vigorous Exercise / Sinus Tachycardia (20%)
            base_hr = np.random.uniform(115.0, 155.0)
            temp_c = 35.0 + np.random.normal(0, 0.15)
            delta_temp = 0.15
            ptt_offset_ms = np.random.uniform(150.0, 175.0)
            rehab_active = True
            rr_std = 0.02
        elif sub_p < 0.85:
            # (c) Cold Room Exposure (15%) - Cold Upper-Arm Brachial Band (Biceps Placement), but NOT an MI!
            base_hr = np.random.uniform(66.0, 82.0)
            temp_c = np.random.uniform(33.2, 33.9)
            delta_temp = -0.35
            pi_scale = 0.6
            ptt_offset_ms = np.random.uniform(170.0, 190.0)
            rr_std = 0.03
        else:
            # (d) Healthy Elderly with Stiff Arteries / Low Baseline PAT (15%)
            base_hr = np.random.uniform(58.0, 74.0)
            temp_c = 34.5 + np.random.normal(0, 0.12)
            delta_temp = 0.0
            ptt_offset_ms = np.random.uniform(142.0, 160.0)
            rr_std = 0.035

    elif clinical_class == 1:
        # AFib Sub-Phenotypes:
        has_p = False
        temp_c = 34.6 + np.random.normal(0, 0.12)
        if sub_p < 0.65:
            # (a) Rapid Ventricular Response (RVR) AFib (65%)
            base_hr = np.random.uniform(110.0, 160.0)
            rr_std = 0.22
            pi_scale = 0.75
            ptt_offset_ms = np.random.uniform(160.0, 190.0)
            missed_anticoagulant = np.random.choice([True, False], p=[0.45, 0.55])
        else:
            # (b) Rate-Controlled AFib on Beta-Blockers (35%)
            base_hr = np.random.uniform(65.0, 88.0)
            rr_std = 0.16
            pi_scale = 0.85
            ptt_offset_ms = np.random.uniform(175.0, 205.0)
            missed_anticoagulant = np.random.choice([True, False], p=[0.35, 0.65])

    elif clinical_class == 2:
        # Acute MI Sub-Phenotypes:
        if sub_p < 0.20:
            # (a) Lead-I Blind Spot / NSTEMI (~20% of Class 2 MI)
            # Inferior/posterior MI or NSTEMI: Lead-I ST is nearly isoelectric (-0.04 to +0.07 mV)
            # Autonomic cues: low RMSSD, cold diaphoresis, suppressed perfusion index
            base_hr = np.random.uniform(78.0, 102.0)
            st_dev = np.random.uniform(-0.04, 0.07)
            t_inv = np.random.uniform(-0.15, 0.12)
            temp_c = np.random.uniform(33.8, 34.5)
            delta_temp = np.random.uniform(-0.75, -0.30)  # Cold diaphoresis
            pi_scale = np.random.uniform(0.40, 0.65)      # Peripheral hypoperfusion
            ptt_offset_ms = np.random.uniform(165.0, 190.0)
            rr_std = 0.018
        elif sub_p < 0.70:
            # (b) Uncomplicated Anterior/Lateral STEMI with NORMAL skin temperature (50%)
            base_hr = np.random.uniform(78.0, 105.0)
            st_dev = np.random.uniform(0.20, 0.38)
            t_inv = -0.40
            temp_c = np.random.uniform(34.2, 34.8)  # Normothermic biceps
            delta_temp = np.random.normal(0, 0.05)
            pi_scale = 0.70
            ptt_offset_ms = np.random.uniform(170.0, 195.0)
            rr_std = 0.015
        elif sub_p < 0.90:
            # (c) Inferior Wall MI with Bezold-Jarisch Vagal Bradycardia (20%)
            base_hr = np.random.uniform(46.0, 60.0)  # Vagal bradycardia
            st_dev = np.random.uniform(0.18, 0.32)
            t_inv = -0.30
            temp_c = np.random.uniform(34.0, 34.6)
            delta_temp = -0.15
            pi_scale = 0.55
            ptt_offset_ms = np.random.uniform(205.0, 235.0)  # Hypotension (long PAT)
            rr_std = 0.02
        else:
            # (d) Cardiogenic Shock MI with Cold Diaphoresis (10%)
            base_hr = np.random.uniform(98.0, 125.0)
            st_dev = np.random.uniform(0.28, 0.45)
            t_inv = -0.50
            temp_c = np.random.uniform(32.8, 33.5)  # Diaphoresis cold drop
            delta_temp = np.random.uniform(-0.65, -1.10)
            pi_scale = 0.25  # Severe hypoperfusion
            ptt_offset_ms = np.random.uniform(185.0, 215.0)
            rr_std = 0.012

    elif clinical_class == 3:
        # Ischemic Stroke Sub-Phenotypes:
        has_p = False
        drift = np.linspace(0, 0.45, n_imu)
        ax += drift * 0.5
        ay += drift * 0.3
        if sub_p < 0.25:
            # (a) Normotensive Lacunar Stroke (25% of Class 3)
            # Overlaps baseline vitals (HR 68-85, SBP 130-150 mmHg) with Class 0 mild hypertension/stress
            base_hr = np.random.uniform(68.0, 85.0)
            rr_std = 0.06
            ptt_offset_ms = np.random.uniform(165.0, 185.0)  # Normotensive / mild elevation
            temp_c = 34.5 + np.random.normal(0, 0.1)
            missed_anticoagulant = False
        elif sub_p < 0.80:
            # (b) Standard Thromboembolic Stroke with Reactive Hypertension (55%)
            base_hr = np.random.uniform(75.0, 102.0)
            rr_std = 0.12
            ptt_offset_ms = np.random.uniform(128.0, 145.0)  # Moderate SBP surge (155-185 mmHg)
            temp_c = 34.5 + np.random.normal(0, 0.1)
            missed_anticoagulant = True
        else:
            # (c) Ischemic Stroke on Beta-Blockers (20%) - Bradycardic stroke mimic!
            base_hr = np.random.uniform(52.0, 61.0)
            rr_std = 0.08
            ptt_offset_ms = np.random.uniform(138.0, 155.0)  # Moderate SBP surge, normal pulse pressure
            temp_c = 34.4 + np.random.normal(0, 0.1)
            missed_anticoagulant = True

    elif clinical_class == 4:
        # Hemorrhagic Stroke Sub-Phenotypes:
        fall_idx = int(n_imu * 0.3)
        fall_window = slice(fall_idx, min(n_imu, fall_idx + 15))
        ax[fall_window] += np.random.uniform(2.5, 3.5)
        ay[fall_window] += np.random.uniform(1.8, 2.5)
        az[fall_window] += np.random.uniform(1.5, 2.2)

        if sub_p < 0.40:
            # (a) Early Hyper-Acute Tachycardic ICH (40%)
            # Intracranial bleed with severe pain/catecholamine surge: TACHYCARDIA + SEVERE HYPERTENSION
            base_hr = np.random.uniform(92.0, 120.0)
            t_inv = -0.50  # Deep cerebral T-waves
            ptt_offset_ms = np.random.uniform(95.0, 118.0)  # Extreme SBP surge > 195 mmHg
            temp_c = 34.6 + np.random.normal(0, 0.12)
            rr_std = 0.025
        elif sub_p < 0.85:
            # (b) Classic Late Cushing's Triad ICH (45%)
            # Brainstem herniation with reflex bradycardia + extreme hypertension + widened PP
            base_hr = np.random.uniform(42.0, 58.0)
            t_inv = -0.60
            ptt_offset_ms = np.random.uniform(90.0, 115.0)
            temp_c = 34.6 + np.random.normal(0, 0.12)
            rr_std = 0.02
        else:
            # (c) ICH with Neurogenic Stunned Myocardium (15%)
            # Acute ICH triggering transient myocardial injury: ST-elevation + extreme ICP surge!
            base_hr = np.random.uniform(85.0, 110.0)
            st_dev = np.random.uniform(0.16, 0.26)  # Transient ST-elevation mimicking MI!
            t_inv = -0.45
            ptt_offset_ms = np.random.uniform(92.0, 115.0)  # Extreme hypertension
            temp_c = 34.6 + np.random.normal(0, 0.12)
            rr_std = 0.025
    else:
        raise ValueError(f"Unknown clinical class: {clinical_class}")

    # Build continuous ECG and PPG by tiling beats with inter-beat intervals
    ecg = np.zeros(n_samples)
    ppg = np.zeros(n_samples) + 52000.0

    beat_template = _synthesize_ecg_beat(fs, st_dev=st_dev, t_inv=t_inv, has_p=has_p)
    ppg_pulse_template = _synthesize_ppg_pulse(fs, pi_scale=pi_scale)

    curr_idx = int(0.1 * fs)
    mean_interval = fs * (60.0 / base_hr)
    # Model PAT = PEP + PTT where Pre-Ejection Period (PEP) adds +/-12ms physiological variability
    pep_ms = float(np.random.normal(35.0, 12.0))
    pat_offset_ms = max(75.0, ptt_offset_ms + pep_ms - 35.0)
    ptt_samples = int((pat_offset_ms / 1000.0) * fs)

    while curr_idx < n_samples:
        beat_len = min(len(beat_template), n_samples - curr_idx)
        ecg[curr_idx:curr_idx + beat_len] += beat_template[:beat_len]

        # PPG pulse delayed by PTT from ventricular contraction (ECG R-peak at +0.30s)
        r_peak_offset = int(0.30 * fs)
        ppg_idx = curr_idx + r_peak_offset + ptt_samples
        if ppg_idx < n_samples:
            ppg_len = min(len(ppg_pulse_template), n_samples - ppg_idx)
            # In AFib, randomly drop 1 out of 5 peripheral pulses (Pulse Deficit)
            if not (clinical_class == 1 and np.random.rand() < 0.20):
                ppg[ppg_idx:ppg_idx + ppg_len] += ppg_pulse_template[:ppg_len]

        # Next beat interval
        jitter = np.random.normal(0, rr_std * mean_interval)
        interval = max(int(0.35 * fs), int(mean_interval + jitter))
        curr_idx += interval

    # Add realistic baseline noise and 50Hz electrical hum
    noise = np.random.normal(0, 0.02, n_samples) + 0.01 * np.sin(2 * np.pi * 50 * t)
    ecg += noise

    # Extract 20-feature clinical vector
    imu_dict = {'ax': ax, 'ay': ay, 'az': az, 'gx': gx, 'gy': gy, 'gz': gz}
    features = extract_10s_features(
        ecg_mv=ecg,
        ppg_raw=ppg,
        imu_data=imu_dict,
        temp_c=float(temp_c),
        delta_temp_5min=float(delta_temp),
        missed_anticoagulant=bool(missed_anticoagulant),
        rehab_active=bool(rehab_active),
        fs=fs
    )

    def _to_native(o):
        if isinstance(o, dict):
            return {k: _to_native(v) for k, v in o.items()}
        elif isinstance(o, (list, tuple)):
            return [_to_native(v) for v in o]
        elif isinstance(o, (np.bool_, bool)):
            return bool(o)
        elif isinstance(o, (np.integer, int)):
            return int(o)
        elif isinstance(o, (np.floating, float)):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        return o

    payload = {
        "device_id": "ARMBAND_ESP32S3_01",
        "timestamp_ms": 1760615780000,
        "ground_truth_label": int(clinical_class),
        "ground_truth_name": CLASS_NAMES[clinical_class],
        "patient_context": {
            "missed_anticoagulant_dose": bool(missed_anticoagulant),
            "rehab_exercise_active": bool(rehab_active)
        },
        "raw_windows": {
            "sampling_rate_hz": int(fs),
            "ecg_mv": ecg.tolist(),
            "ppg_ir": ppg.tolist(),
            "imu_ax_g": ax.tolist(),
            "imu_ay_g": ay.tolist(),
            "imu_az_g": az.tolist(),
            "imu_gx_dps": gx.tolist(),
            "imu_gy_dps": gy.tolist(),
            "imu_gz_dps": gz.tolist()
        },
        "extracted_features": features
    }
    return _to_native(payload)


def generate_batch_dataset(
    n_samples_per_class: int = 100,
    fs: float = 250.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a complete dataset tensor ready for PyTorch Stage 1 model training:
      - waveforms: (N, 2, 2500) -> [ECG, PPG]
      - tabular: (N, 20) -> 20 extracted features
      - labels: (N,) -> integers 0..4
    """
    waveforms_list = []
    tabular_list = []
    labels_list = []

    feature_keys = [
        "hr_ecg_bpm", "pulse_rate_ppg_bpm", "pulse_deficit_bpm", "rmssd_ms",
        "sdnn_ms", "pnn50_pct", "rr_entropy", "st_deviation_mv", "t_wave_amplitude_mv",
        "spo2_pct", "perfusion_index_pct", "ptt_ms", "delta_ptt_ms",
        "estimated_sbp_mmhg", "estimated_dbp_mmhg", "cushing_index",
        "skin_temp_c", "delta_skin_temp_c", "missed_anticoagulant_dose",
        "rehab_exercise_active"
    ]

    for cls in range(5):
        for _ in range(n_samples_per_class):
            window = generate_synthetic_window(cls, duration_sec=10.0, fs=fs)
            ecg = np.array(window["raw_windows"]["ecg_mv"])
            ppg = np.array(window["raw_windows"]["ppg_ir"])
            # Normalize PPG to standard scale for CNN
            ppg_norm = (ppg - np.mean(ppg)) / (np.std(ppg) + 1e-6)

            wave_2ch = np.stack([ecg, ppg_norm], axis=0)  # Shape: (2, 2500)
            waveforms_list.append(wave_2ch)

            feats = window["extracted_features"]
            feat_vec = [float(feats.get(k, 0.0)) for k in feature_keys]
            tabular_list.append(feat_vec)

            labels_list.append(cls)

    waves = np.array(waveforms_list, dtype=np.float32)
    tabular = np.array(tabular_list, dtype=np.float32)
    labels = np.array(labels_list, dtype=np.int64)

    return waves, tabular, labels
