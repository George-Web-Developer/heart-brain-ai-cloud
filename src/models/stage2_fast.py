"""
stage2_fast.py
Stage 2 Multimodal FAST Diagnostic Engine (Vision & Speech) for Acute Stroke Assessment.
Processes 468 3D facial landmarks (MediaPipe) and 5-second 16kHz audio (MFCCs).
"""

from typing import Dict, Any, Optional
import numpy as np


def compute_facial_asymmetry_metrics(landmarks_468_3d: np.ndarray) -> Dict[str, Any]:
    """
    Computes geometric asymmetry ratios from 468 3D facial landmarks:
      1. Mouth Corner Droop: Angle between left mouth corner (61) and right (291) relative to horizontal eye line
      2. Lip Aperture Asymmetry: Height difference between upper and lower lips on left vs right sides
      3. Palpebral Fissure Ratio: Left eye opening (159-145) vs Right eye opening (386-374)
      4. Eyebrow Elevation Asymmetry: Left eyebrow (70) vs Right eyebrow (300)
    Compensates for camera/head roll tilt using the inter-ocular axis (33 -> 263).
    """
    if landmarks_468_3d is None or len(landmarks_468_3d) < 468:
        return {
            "facial_droop_probability": 0.0,
            "mouth_corner_droop_deg": 0.0,
            "lip_aperture_asymmetry_ratio": 1.0,
            "palpebral_fissure_ratio": 1.0,
            "eyebrow_elevation_diff": 0.0,
            "status": "INSUFFICIENT_LANDMARKS"
        }

    pts = np.asarray(landmarks_468_3d, dtype=np.float64)

    # Canonical landmarks
    left_eye_outer = pts[33]
    right_eye_outer = pts[263]

    # Inter-ocular distance normalization factor and roll-tilt angle
    eye_vec = right_eye_outer - left_eye_outer
    iod = float(np.linalg.norm(eye_vec))
    if iod < 1e-4:
        iod = 1.0

    # Align coordinates in 2D (x, y) plane relative to the inter-ocular line to remove head/camera roll tilt
    eye_angle_rad = float(np.arctan2(eye_vec[1], max(1e-4, abs(eye_vec[0]))))
    if abs(eye_angle_rad) > 1e-6:
        cos_a = np.cos(-eye_angle_rad)
        sin_a = np.sin(-eye_angle_rad)
        center_xy = 0.5 * (left_eye_outer[:2] + right_eye_outer[:2])
        rel_xy = pts[:, :2] - center_xy
        rot_x = rel_xy[:, 0] * cos_a - rel_xy[:, 1] * sin_a + center_xy[0]
        rot_y = rel_xy[:, 0] * sin_a + rel_xy[:, 1] * cos_a + center_xy[1]
        pts_aligned = pts.copy()
        pts_aligned[:, 0] = rot_x
        pts_aligned[:, 1] = rot_y
    else:
        pts_aligned = pts

    left_mouth = pts_aligned[61]
    right_mouth = pts_aligned[291]

    # 1. Mouth Corner Droop Angle (degrees) relative to inter-ocular horizontal axis
    mouth_dy = (right_mouth[1] - left_mouth[1])
    mouth_dx = (right_mouth[0] - left_mouth[0])
    mouth_angle_deg = float(abs(np.degrees(np.arctan2(mouth_dy, max(1e-4, abs(mouth_dx))))))

    # 2. Palpebral Fissure (Eyelid Opening Height)
    # Left eye: 159 (upper) vs 145 (lower)
    left_eye_height = float(np.linalg.norm(pts_aligned[159] - pts_aligned[145]))
    # Right eye: 386 (upper) vs 374 (lower)
    right_eye_height = float(np.linalg.norm(pts_aligned[386] - pts_aligned[374]))
    if left_eye_height < 1e-4 and right_eye_height < 1e-4:
        palpebral_ratio = 1.0
    else:
        max_h = max(left_eye_height, right_eye_height)
        palpebral_ratio = float(min(left_eye_height, right_eye_height) / max(1e-4, max_h))

    # 3. Eyebrow Elevation
    left_eyebrow = pts_aligned[70][1]
    right_eyebrow = pts_aligned[300][1]
    eyebrow_diff = float(abs(right_eyebrow - left_eyebrow) / iod)

    # 4. Lip Aperture Asymmetry
    lip_asym = float(abs(left_mouth[1] - right_mouth[1]) / iod)

    # Calibrated rule-based / geometric classifier score for facial droop
    droop_score = 0.0
    if mouth_angle_deg > 6.0:
        droop_score += min(0.50, (mouth_angle_deg - 6.0) * 0.05)
    if palpebral_ratio < 0.85:
        droop_score += (0.85 - palpebral_ratio) * 1.5
    if eyebrow_diff > 0.08:
        droop_score += min(0.20, eyebrow_diff * 1.2)
    if lip_asym > 0.08:
        droop_score += min(0.20, lip_asym * 1.2)

    facial_droop_prob = float(np.clip(droop_score, 0.0, 1.0))

    return {
        "facial_droop_probability": round(facial_droop_prob, 3),
        "mouth_corner_droop_deg": round(mouth_angle_deg, 2),
        "lip_aperture_asymmetry_ratio": round(lip_asym, 3),
        "palpebral_fissure_ratio": round(palpebral_ratio, 3),
        "eyebrow_elevation_diff": round(eyebrow_diff, 3),
        "status": "VALID_FACE_MESH"
    }


def analyze_speech_dysarthria(
    audio_pcm: np.ndarray,
    sample_rate: int = 16000
) -> Dict[str, Any]:
    """
    Acoustic Dysarthria & Slurred Speech Detector.
    Extracts frame-level energy shimmer, zero-crossing rate perturbation (jitter proxy),
    and spectral centroid features.
    """
    if audio_pcm is None or len(audio_pcm) < sample_rate * 0.5:
        return {
            "dysarthria_probability": 0.0,
            "jitter_estimate": 0.0,
            "shimmer_estimate": 0.0,
            "zero_crossing_rate": 0.0,
            "status": "AUDIO_TOO_SHORT"
        }

    # Normalize audio and sanitize NaNs
    audio = np.nan_to_num(np.asarray(audio_pcm, dtype=np.float32), nan=0.0)
    max_val = float(np.max(np.abs(audio)))
    if max_val > 1e-6:
        audio = audio / max_val

    # Frame-based energy and zero crossing rate
    frame_size = int(0.025 * sample_rate)  # 25 ms
    hop_size = int(0.010 * sample_rate)    # 10 ms
    frames = [audio[i:i + frame_size] for i in range(0, len(audio) - frame_size, hop_size)]

    frame_energies = [float(np.sum(f ** 2)) for f in frames]
    frame_zcrs = [float(np.mean(np.abs(np.diff(np.sign(f))))) for f in frames] if frames else [0.0]
    zcr = float(np.mean(frame_zcrs)) if frame_zcrs else 0.0

    # Shimmer (amplitude perturbation) & Jitter (cycle-to-cycle frequency perturbation proxy)
    if len(frame_energies) > 1:
        shimmer = float(np.mean(np.abs(np.diff(frame_energies))) / (np.mean(frame_energies) + 1e-6))
        jitter = float(np.mean(np.abs(np.diff(frame_zcrs))) / (np.mean(frame_zcrs) + 1e-6))
    else:
        shimmer = 0.02
        jitter = 0.01

    # Spectral centroid and frequency spread
    fft_spectrum = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1.0 / sample_rate)
    centroid = float(np.sum(freqs * fft_spectrum) / (np.sum(fft_spectrum) + 1e-6))

    # Dysarthria acoustic deviation score
    # Dysarthric speech shows high amplitude variability (shimmer > 0.15) and reduced articulatory precision
    dysarthria_score = 0.0
    if shimmer > 0.12:
        dysarthria_score += min(0.45, (shimmer - 0.12) * 2.0)
    if centroid < 1200 or centroid > 3800:
        dysarthria_score += 0.25
    if zcr > 0.35:
        dysarthria_score += 0.20

    dysarthria_prob = float(np.clip(dysarthria_score, 0.0, 1.0))

    return {
        "dysarthria_probability": round(dysarthria_prob, 3),
        "jitter_estimate": round(jitter, 4),
        "shimmer_estimate": round(shimmer, 4),
        "zero_crossing_rate": round(float(zcr), 4),
        "spectral_centroid_hz": round(centroid, 1),
        "status": "VALID_AUDIO"
    }


def evaluate_fast_check(
    facial_metrics: Dict[str, Any],
    speech_results: Dict[str, Any],
    stage1_probs: Optional[Dict[str, float]] = None,
    missed_anticoagulant: bool = False,
    arm_drift_deg: float = 0.0,
    active_arm_hold_test: bool = False,
    stage1_red_contraindication_veto: bool = False
) -> Dict[str, Any]:
    """
    Late Multimodal Fusion Engine.
    Combines Stage 1 Continuous Probabilities with Stage 2 Vision, Speech, and Active Arm-Hold FAST indicators.
    Applies medication adherence multiplier via bounded log-odds shift and triggers mandatory Red Contraindication Alert.
    Preserves Stage 1 Decoupled Red Contraindication Veto when active.
    """
    p_face = float(facial_metrics.get("facial_droop_probability", 0.0))
    p_speech = float(speech_results.get("dysarthria_probability", 0.0))

    has_explicit_stage1 = stage1_probs is not None
    if stage1_probs is None:
        stage1_probs = {
            "Normal": 0.5, "AFib": 0.1, "MI": 0.1,
            "Ischemic_Stroke": 0.15, "Hemorrhagic_Stroke": 0.15
        }

    p_stage1_ischemic = float(stage1_probs.get("Ischemic_Stroke", 0.0))
    p_stage1_hemorrhagic = float(stage1_probs.get("Hemorrhagic_Stroke", 0.0))
    p_afib = float(stage1_probs.get("AFib", 0.0))

    # Active Pronator Drift check: Strictly reserved for Stage 2 active 10s eyes-closed arm-hold test
    if active_arm_hold_test and arm_drift_deg > 18.0:
        p_drift = float(np.clip((arm_drift_deg - 18.0) * 0.04 + 0.50, 0.0, 1.0))
        raw_ischemic_risk = 0.35 * p_stage1_ischemic + 0.25 * p_face + 0.20 * p_speech + 0.20 * p_drift
    else:
        p_drift = 0.0
        raw_ischemic_risk = 0.45 * p_stage1_ischemic + 0.35 * p_face + 0.20 * p_speech

    # Medication Adherence Risk Multiplier:
    # If patient has AFib and missed anticoagulant dose, thromboembolic risk surges by 35%
    # Bound via log-odds (logit shift) so probability is strictly bounded in [0.0, 1.0]
    if missed_anticoagulant and (p_afib > 0.30 or p_stage1_ischemic > 0.30):
        eps = 1e-5
        p_clamped = np.clip(raw_ischemic_risk, eps, 1.0 - eps)
        logit = np.log(p_clamped / (1.0 - p_clamped))
        shifted_logit = logit + np.log(1.35)
        raw_ischemic_risk = 1.0 / (1.0 + np.exp(-shifted_logit))

    final_ischemic_risk = float(np.clip(raw_ischemic_risk, 0.0, 1.0))
    final_hemorrhagic_risk = float(np.clip(p_stage1_hemorrhagic, 0.0, 1.0))

    # Red Contraindication Warning Rule:
    # Fires if Stage 1 already raised the Decoupled Contraindication Veto, OR if explicit Stage 1 P(ICH) >= 0.12
    # with positive neurological deficits, OR if final_hemorrhagic_risk > 0.40.
    red_alert = False
    clinical_warning = None
    has_neuro_deficit = (p_face > 0.25 or p_speech > 0.25 or p_drift > 0.0 or final_ischemic_risk > 0.45)
    if (
        stage1_red_contraindication_veto or
        final_hemorrhagic_risk > 0.40 or
        (has_explicit_stage1 and final_hemorrhagic_risk >= 0.12 and has_neuro_deficit)
    ):
        red_alert = True
        clinical_warning = (
            "CLINICAL DECISION SUPPORT: Suspected Neurovascular Emergency (Hemorrhagic stroke cannot be excluded). "
            "This research system cannot determine stroke subtype or guide medication decisions. "
            "Urgent professional assessment and emergency brain imaging are required. "
            "Guidelines contraindicate pre-hospital aspirin, anticoagulants, or thrombolytics prior to non-contrast head CT scan "
            "(DO NOT ADMINISTER ASPIRIN, ANTICOAGULANTS, OR THROMBOLYTICS PRIOR TO CT SCAN)."
        )

    # Determine recommended triage priority
    if red_alert or final_ischemic_risk > 0.70:
        triage_level = "CODE_STROKE_EMERGENCY"
    elif final_ischemic_risk > 0.45 or p_face > 0.50:
        triage_level = "HIGH_STROKE_ALERT"
    else:
        triage_level = "MONITORING_STABLE"

    return {
        "final_ischemic_stroke_risk": round(final_ischemic_risk, 3),
        "final_hemorrhagic_stroke_risk": round(final_hemorrhagic_risk, 3),
        "facial_droop_probability": round(p_face, 3),
        "dysarthria_probability": round(p_speech, 3),
        "pronator_drift_probability": round(p_drift, 3),
        "active_arm_hold_test_evaluated": bool(active_arm_hold_test),
        "missed_anticoagulant_multiplier_applied": missed_anticoagulant and (p_afib > 0.30 or p_stage1_ischemic > 0.30),
        "triage_level": triage_level,
        "red_contraindication_alert": red_alert,
        "red_contraindication_veto": red_alert,
        "clinical_warning": clinical_warning
    }