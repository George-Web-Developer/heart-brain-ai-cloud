"""
modality_eligibility_gate.py
========================================================================================
Modality Eligibility Gate for Dual-Head Disease Classifier (Head A).

Ensures fail-closed verification of physical sensor availability BEFORE:
  1. Feature engineering (prevents vel_mag from zero-substituting missing channels).
  2. StandardScaler transformation.
  3. XGBoost model inference (predict / predict_proba).
  4. Clinical Safety Arbiter and downstream operational logic.

Clinical and Algorithmic Rationale:
  The frozen Dual-Head XGBoost model was trained on multimodal wearable sensor data
  with random channel dropout (5-18%), but was NEVER trained or validated on complete
  block dropout of non-ECG modalities (e.g., Chapman-Shaoxing ECG-only external testing).
  When 13 non-ECG features are missing (NaN), XGBoost default split branches systematically
  route samples into high log-odds for Stroke (Class 4), causing 1,014 false positives.
  This gate blocks unsupported inference configurations and returns an explicit
  ABSTENTION status rather than generating invalid clinical predictions.
========================================================================================
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np


# Canonical 20 Raw Features Contract
CANONICAL_20_FEATURES = [
    "hr_ecg_bpm",                 # 0: ECG modality
    "pulse_rate_ppg_bpm",         # 1: PPG modality (optical photoplethysmography)
    "pulse_deficit_bpm",          # 2: Cross-modality ECG - PPG
    "rmssd_ms",                   # 3: ECG modality (HRV)
    "sdnn_ms",                    # 4: ECG modality (HRV)
    "pnn50_pct",                  # 5: ECG modality (HRV)
    "rr_entropy",                 # 6: ECG modality (HRV / RR coefficient of variation)
    "st_deviation_mv",            # 7: ECG modality (morphology)
    "t_wave_amplitude_mv",        # 8: ECG modality (morphology)
    "spo2_pct",                   # 9: PPG / Optical oximetry modality
    "perfusion_index_pct",        # 10: PPG modality (AC/DC pulsatile ratio)
    "ptt_ms",                     # 11: Cross-modality ECG + PPG (pulse transit / arrival time)
    "delta_ptt_ms",               # 12: Cross-modality ECG + PPG rate of change
    "estimated_sbp_mmhg",         # 13: Blood pressure / Hemodynamics modality
    "estimated_dbp_mmhg",         # 14: Blood pressure / Hemodynamics modality
    "cushing_index",              # 15: Cross-modality BP + Heart Rate (SBP / HR)
    "skin_temp_c",                # 16: Cutaneous thermometry / Temperature modality
    "delta_skin_temp_c",          # 17: Cutaneous thermometry rate of change
    "missed_anticoagulant_dose",  # 18: Contextual clinical history (NOT a physical sensor)
    "rehab_exercise_active"       # 19: Contextual activity status (NOT a physical continuous sensor)
]

# Physical Modality Index Mapping
MODALITY_FEATURE_INDICES = {
    "ECG": [0, 3, 4, 5, 6, 7, 8],
    "PPG": [1, 9, 10],
    "HEMODYNAMICS_BP_PTT": [2, 11, 12, 13, 14, 15],
    "TEMPERATURE": [16, 17],
    "CONTEXTUAL": [18, 19]
}

# Gate Status Codes
STATUS_ELIGIBLE = "ELIGIBLE"
STATUS_INSUFFICIENT_MODALITY = "INSUFFICIENT_MODALITY"
STATUS_INVALID_INPUT = "INVALID_INPUT_DATA"

# Detected Availability Patterns
PATTERN_SUPPORTED_FULL_MULTIMODAL = "SUPPORTED_FULL_MULTIMODAL"
PATTERN_SUPPORTED_PASSIVE_ARMBAND = "SUPPORTED_PASSIVE_ARMBAND"
PATTERN_SUPPORTED_MINOR_DROPOUT = "SUPPORTED_MINOR_CHANNEL_DROPOUT"

PATTERN_ECG_ONLY_BLOCK_DROPOUT = "ECG_ONLY_BLOCK_DROPOUT"
PATTERN_ALL_MODALITIES_MISSING = "ALL_MODALITIES_MISSING"
PATTERN_MISSING_PPG_MODALITY = "MISSING_PPG_MODALITY"
PATTERN_MISSING_HEMODYNAMIC_INPUTS = "MISSING_HEMODYNAMIC_INPUTS"
PATTERN_MISSING_REQUIRED_TEMPERATURE = "MISSING_REQUIRED_TEMPERATURE"
PATTERN_MISSING_HEMODYNAMIC_AND_PPG = "MISSING_HEMODYNAMIC_AND_PPG"
PATTERN_NO_CARDIAC_RHYTHM_MODALITY = "NO_CARDIAC_RHYTHM_MODALITY"
PATTERN_REQUIRED_FEATURE_MISSING = "REQUIRED_FEATURE_MISSING"
PATTERN_EXCESSIVE_FEATURE_DROPOUT = "EXCESSIVE_FEATURE_DROPOUT"
PATTERN_NON_FINITE_VALUES = "NON_FINITE_VALUES"
PATTERN_DIMENSION_MISMATCH = "DIMENSION_MISMATCH"
PATTERN_MALFORMED_METADATA = "MALFORMED_METADATA"
PATTERN_UNSUPPORTED_PARTIAL = "UNSUPPORTED_PARTIAL_CONFIGURATION"


@dataclass
class ModalityGateResult:
    """Structured result returned by the Modality Eligibility Gate."""
    is_eligible: bool
    inference_status: str
    missing_modalities: List[str]
    detected_pattern: str
    technical_reason: str
    model_inference_skipped: bool
    feature_availability: Dict[str, bool] = field(default_factory=dict)
    metadata_used: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_eligible": self.is_eligible,
            "inference_status": self.inference_status,
            "missing_modalities": self.missing_modalities,
            "detected_pattern": self.detected_pattern,
            "technical_reason": self.technical_reason,
            "model_inference_skipped": self.model_inference_skipped,
            "metadata_used": self.metadata_used
        }


class ModalityEligibilityGate:
    """
    Fail-closed Modality Eligibility Gate for the Dual-Head Disease Classifier.
    Evaluates raw 20-canonical feature vectors before feature engineering.
    """

    # Maximum allowable isolated feature dropout fraction (15%, up to 3 non-critical features)
    MAX_ALLOWABLE_DROPOUT_COUNT = 3

    @classmethod
    def evaluate_vector(
        cls,
        x_raw: Union[np.ndarray, List[float]],
        sensor_metadata: Optional[Dict[str, Any]] = None
    ) -> ModalityGateResult:
        """
        Evaluates a single 20-feature raw input vector against evidence-based eligibility criteria.

        Args:
            x_raw: 1D array-like of shape (20,) representing CANONICAL_20_FEATURES.
            sensor_metadata: Optional dict containing explicit hardware provenance
                             (e.g., {"ecg_connected": True, "ppg_connected": False}).

        Returns:
            ModalityGateResult with boolean eligibility and diagnostic metadata.
        """
        # 1. Dimension and Type Check
        try:
            x = np.asarray(x_raw, dtype=np.float64)
        except Exception as e:
            return ModalityGateResult(
                is_eligible=False,
                inference_status=STATUS_INVALID_INPUT,
                missing_modalities=[],
                detected_pattern=PATTERN_DIMENSION_MISMATCH,
                technical_reason=f"Input could not be parsed as numerical array: {str(e)}",
                model_inference_skipped=True
            )

        if x.ndim != 1 or x.shape[0] != 20:
            return ModalityGateResult(
                is_eligible=False,
                inference_status=STATUS_INVALID_INPUT,
                missing_modalities=[],
                detected_pattern=PATTERN_DIMENSION_MISMATCH,
                technical_reason=f"Input vector has shape {x.shape}, expected exactly (20,).",
                model_inference_skipped=True
            )

        # 2. Check for Non-Finite Values (Infinities)
        if np.isinf(x).any():
            inf_indices = [int(i) for i in np.where(np.isinf(x))[0]]
            inf_names = [CANONICAL_20_FEATURES[i] for i in inf_indices]
            return ModalityGateResult(
                is_eligible=False,
                inference_status=STATUS_INVALID_INPUT,
                missing_modalities=[],
                detected_pattern=PATTERN_NON_FINITE_VALUES,
                technical_reason=f"Input contains non-finite infinities at indices {inf_indices} ({inf_names}).",
                model_inference_skipped=True
            )

        # 3. Validate Sensor Metadata Structure (if provided)
        metadata_used = False
        meta_connected: Dict[str, bool] = {}
        required_features_list: List[str] = []
        required_modalities_list: List[str] = []

        if sensor_metadata is not None:
            if not isinstance(sensor_metadata, dict):
                return ModalityGateResult(
                    is_eligible=False,
                    inference_status=STATUS_INVALID_INPUT,
                    missing_modalities=[],
                    detected_pattern=PATTERN_MALFORMED_METADATA,
                    technical_reason="sensor_metadata was provided but is not a valid dictionary.",
                    model_inference_skipped=True
                )
            metadata_used = True
            # Validate metadata types
            for k, v in sensor_metadata.items():
                if k.lower() in ["required_features", "required_modalities"]:
                    if not isinstance(v, (list, tuple, set)):
                        return ModalityGateResult(
                            is_eligible=False,
                            inference_status=STATUS_INVALID_INPUT,
                            missing_modalities=[],
                            detected_pattern=PATTERN_MALFORMED_METADATA,
                            technical_reason=f"Metadata field '{k}' must be a list of strings.",
                            model_inference_skipped=True
                        )
                elif not isinstance(v, (bool, int, np.bool_, str)):
                    return ModalityGateResult(
                        is_eligible=False,
                        inference_status=STATUS_INVALID_INPUT,
                        missing_modalities=[],
                        detected_pattern=PATTERN_MALFORMED_METADATA,
                        technical_reason=f"Metadata key '{k}' has unexpected non-boolean/non-scalar value type: {type(v)}.",
                        model_inference_skipped=True
                    )

            # Standardize connection keys
            meta_lower = {k.lower(): v for k, v in sensor_metadata.items()}
            if "ecg_connected" in meta_lower:
                meta_connected["ECG"] = bool(meta_lower["ecg_connected"])
            elif "ecg" in meta_lower and isinstance(meta_lower["ecg"], (bool, int)):
                meta_connected["ECG"] = bool(meta_lower["ecg"])

            if "ppg_connected" in meta_lower:
                meta_connected["PPG"] = bool(meta_lower["ppg_connected"])
            elif "ppg" in meta_lower and isinstance(meta_lower["ppg"], (bool, int)):
                meta_connected["PPG"] = bool(meta_lower["ppg"])

            if "bp_connected" in meta_lower:
                meta_connected["HEMODYNAMICS_BP_PTT"] = bool(meta_lower["bp_connected"])
            elif "ptt_connected" in meta_lower:
                meta_connected["HEMODYNAMICS_BP_PTT"] = bool(meta_lower["ptt_connected"])

            if "temp_connected" in meta_lower:
                meta_connected["TEMPERATURE"] = bool(meta_lower["temp_connected"])
            elif "temperature_connected" in meta_lower:
                meta_connected["TEMPERATURE"] = bool(meta_lower["temperature_connected"])

            if "required_features" in meta_lower:
                required_features_list = [str(f) for f in meta_lower["required_features"]]
            if "required_modalities" in meta_lower:
                required_modalities_list = [str(m).upper() for m in meta_lower["required_modalities"]]

        # 4. Determine Feature Availability without relying on vel_mag or zero substitutions
        # A feature is present if it is finite and not NaN.
        # Genuine measured zeros (e.g. pnn50_pct == 0.0 or st_deviation_mv == 0.0) are NOT missing.
        feat_avail = {CANONICAL_20_FEATURES[i]: not np.isnan(x[i]) for i in range(20)}

        # Check explicit required_features from metadata
        for req_feat in required_features_list:
            if req_feat in feat_avail and not feat_avail[req_feat]:
                return ModalityGateResult(
                    is_eligible=False,
                    inference_status=STATUS_INSUFFICIENT_MODALITY,
                    missing_modalities=[],
                    detected_pattern=PATTERN_REQUIRED_FEATURE_MISSING,
                    technical_reason=f"Required feature '{req_feat}' specified in sensor metadata is NaN.",
                    model_inference_skipped=True,
                    feature_availability=feat_avail,
                    metadata_used=metadata_used
                )

        # Modality presence:
        # A modality is available if:
        #   (a) Its hardware connection flag in metadata is not explicitly False, AND
        #   (b) It has valid, non-NaN measurements for its primary indicators.
        modality_avail: Dict[str, bool] = {}

        for mod_name, indices in MODALITY_FEATURE_INDICES.items():
            if mod_name == "CONTEXTUAL":
                # Contextual features are patient context, not physical sensors
                modality_avail[mod_name] = True
                continue

            # Check explicit metadata override
            if metadata_used and mod_name in meta_connected and not meta_connected[mod_name]:
                modality_avail[mod_name] = False
                continue

            # Check raw measurement presence
            mod_vals = x[indices]
            all_nan = bool(np.isnan(mod_vals).all())

            if mod_name == "ECG":
                # For ECG: hr_ecg_bpm (index 0) or at least 2 morphological features must be present
                has_ecg_hr = not np.isnan(x[0])
                has_morph = bool((~np.isnan(x[[3, 4, 5, 6, 7, 8]])).sum() >= 2)
                modality_avail[mod_name] = not all_nan and (has_ecg_hr or has_morph)
            elif mod_name == "PPG":
                # For PPG: pulse_rate_ppg_bpm (index 1) or optical parameters (SpO2, PI) must be present
                has_ppg_pr = not np.isnan(x[1])
                has_opt = not np.isnan(x[9]) or not np.isnan(x[10])
                modality_avail[mod_name] = not all_nan and (has_ppg_pr or has_opt)
            elif mod_name == "HEMODYNAMICS_BP_PTT":
                # For Hemodynamics: blood pressure (SBP/DBP) or PTT must be present
                has_bp = not np.isnan(x[13]) or not np.isnan(x[14])
                has_ptt = not np.isnan(x[11])
                modality_avail[mod_name] = not all_nan and (has_bp or has_ptt)
            elif mod_name == "TEMPERATURE":
                # Cutaneous temperature (index 16)
                modality_avail[mod_name] = not np.isnan(x[16])

        # Identify missing modalities
        missing_mods = [m for m, avail in modality_avail.items() if not avail and m != "CONTEXTUAL"]

        # Check explicit required_modalities from metadata
        for req_mod in required_modalities_list:
            if req_mod in modality_avail and not modality_avail[req_mod]:
                return ModalityGateResult(
                    is_eligible=False,
                    inference_status=STATUS_INSUFFICIENT_MODALITY,
                    missing_modalities=missing_mods,
                    detected_pattern=PATTERN_UNSUPPORTED_PARTIAL,
                    technical_reason=f"Required modality '{req_mod}' specified in sensor metadata is missing.",
                    model_inference_skipped=True,
                    feature_availability=feat_avail,
                    metadata_used=metadata_used
                )

        # 5. Evaluate Against Rigorous Evidence-Supported Training Policies

        # Policy Case 1: Complete Disconnection (All Physical Modalities Missing)
        if len(missing_mods) >= 4:
            return ModalityGateResult(
                is_eligible=False,
                inference_status=STATUS_INSUFFICIENT_MODALITY,
                missing_modalities=missing_mods,
                detected_pattern=PATTERN_ALL_MODALITIES_MISSING,
                technical_reason="All physical physiological sensor modalities are missing (all-NaN vector). Model inference cannot proceed.",
                model_inference_skipped=True,
                feature_availability=feat_avail,
                metadata_used=metadata_used
            )

        # Policy Case 2: Chapman-Shaoxing Pattern (ECG-Only Block Dropout)
        # ECG is present, but PPG, Hemodynamics, and Temperature are 100% missing.
        # This regime was never trained and produces 1,014 false hemorrhagic stroke predictions.
        if modality_avail.get("ECG") and not modality_avail.get("PPG") and not modality_avail.get("HEMODYNAMICS_BP_PTT"):
            return ModalityGateResult(
                is_eligible=False,
                inference_status=STATUS_INSUFFICIENT_MODALITY,
                missing_modalities=missing_mods,
                detected_pattern=PATTERN_ECG_ONLY_BLOCK_DROPOUT,
                technical_reason=(
                    "Head A multimodal disease classifier cannot execute in an ECG-only configuration. "
                    "Required non-ECG hemodynamics (PPG, PTT, SBP/DBP, Cushing Index) are completely missing, "
                    "which forces unvalidated tree default routing into acute stroke classes."
                ),
                model_inference_skipped=True,
                feature_availability=feat_avail,
                metadata_used=metadata_used
            )

        # Policy Case 3: Missing Both Cardiac Rhythm Modalities (No ECG and No PPG)
        if not modality_avail.get("ECG") and not modality_avail.get("PPG"):
            return ModalityGateResult(
                is_eligible=False,
                inference_status=STATUS_INSUFFICIENT_MODALITY,
                missing_modalities=missing_mods,
                detected_pattern=PATTERN_NO_CARDIAC_RHYTHM_MODALITY,
                technical_reason="Neither ECG nor optical PPG is available. Cannot establish baseline cardiac rate or rhythm.",
                model_inference_skipped=True,
                feature_availability=feat_avail,
                metadata_used=metadata_used
            )

        # Policy Case 4: Missing Critical Hemodynamic Grounding for Stroke Classification
        # If PPG is missing AND BP/PTT is missing, stroke discrimination is mathematically ungrounded
        if not modality_avail.get("PPG") and not modality_avail.get("HEMODYNAMICS_BP_PTT"):
            return ModalityGateResult(
                is_eligible=False,
                inference_status=STATUS_INSUFFICIENT_MODALITY,
                missing_modalities=missing_mods,
                detected_pattern=PATTERN_MISSING_HEMODYNAMIC_AND_PPG,
                technical_reason="Both PPG and cuffless blood pressure/PTT are missing. Stroke differential diagnosis is unsupported.",
                model_inference_skipped=True,
                feature_availability=feat_avail,
                metadata_used=metadata_used
            )

        # Policy Case 5: Missing Entire Optical PPG Modality
        # Optical photoplethysmography is mandatory for pulse deficit, SpO2, and microvascular assessment.
        if not modality_avail.get("PPG"):
            return ModalityGateResult(
                is_eligible=False,
                inference_status=STATUS_INSUFFICIENT_MODALITY,
                missing_modalities=missing_mods,
                detected_pattern=PATTERN_MISSING_PPG_MODALITY,
                technical_reason="The entire optical PPG modality is missing or disconnected. Pulse deficit and SpO2 cannot be computed.",
                model_inference_skipped=True,
                feature_availability=feat_avail,
                metadata_used=metadata_used
            )

        # Policy Case 6: Missing Required BP/PTT Inputs
        # If Hemodynamics / BP / PTT is completely missing or disconnected
        if not modality_avail.get("HEMODYNAMICS_BP_PTT"):
            return ModalityGateResult(
                is_eligible=False,
                inference_status=STATUS_INSUFFICIENT_MODALITY,
                missing_modalities=missing_mods,
                detected_pattern=PATTERN_MISSING_HEMODYNAMIC_INPUTS,
                technical_reason="Required blood pressure and pulse transit time (PTT) inputs are missing. Hemodynamic evaluation cannot proceed.",
                model_inference_skipped=True,
                feature_availability=feat_avail,
                metadata_used=metadata_used
            )

        # Policy Case 7: Missing Required Temperature
        # If temperature is disconnected via metadata or temp is explicitly required
        if not modality_avail.get("TEMPERATURE") and metadata_used and ("temp_required" in meta_lower and meta_lower["temp_required"]):
            return ModalityGateResult(
                is_eligible=False,
                inference_status=STATUS_INSUFFICIENT_MODALITY,
                missing_modalities=missing_mods,
                detected_pattern=PATTERN_MISSING_REQUIRED_TEMPERATURE,
                technical_reason="Cutaneous temperature modality is required but missing or disconnected.",
                model_inference_skipped=True,
                feature_availability=feat_avail,
                metadata_used=metadata_used
            )

        # Policy Case 8: Supported Passive Armband Mode (Regime A from training)
        # ECG is missing (Lead-I contact open), but PPG + Blood Pressure + Temperature are present
        if not modality_avail.get("ECG") and modality_avail.get("PPG") and modality_avail.get("HEMODYNAMICS_BP_PTT"):
            nan_count = int(np.isnan(x).sum())
            # In Regime A, ECG features (7 features) are NaN, but other channels are present
            return ModalityGateResult(
                is_eligible=True,
                inference_status=STATUS_ELIGIBLE,
                missing_modalities=["ECG"],
                detected_pattern=PATTERN_SUPPORTED_PASSIVE_ARMBAND,
                technical_reason="Passive upper-arm brachial band mode detected (PPG and hemodynamics present). Supported by training Regime A.",
                model_inference_skipped=False,
                feature_availability=feat_avail,
                metadata_used=metadata_used
            )

        # Policy Case 9: Multimodal Configurations (ECG + PPG + Hemodynamics present)
        if modality_avail.get("ECG") and modality_avail.get("PPG") and modality_avail.get("HEMODYNAMICS_BP_PTT"):
            nan_count = int(np.isnan(x).sum())
            if nan_count == 0:
                return ModalityGateResult(
                    is_eligible=True,
                    inference_status=STATUS_ELIGIBLE,
                    missing_modalities=[],
                    detected_pattern=PATTERN_SUPPORTED_FULL_MULTIMODAL,
                    technical_reason="Full multimodal wearable configuration verified with all physical sensors active.",
                    model_inference_skipped=False,
                    feature_availability=feat_avail,
                    metadata_used=metadata_used
                )
            elif nan_count <= cls.MAX_ALLOWABLE_DROPOUT_COUNT:
                # Within validated 5-18% training dropout bound (<= 3 features out of 20)
                return ModalityGateResult(
                    is_eligible=True,
                    inference_status=STATUS_ELIGIBLE,
                    missing_modalities=missing_mods,
                    detected_pattern=PATTERN_SUPPORTED_MINOR_DROPOUT,
                    technical_reason=f"Multimodal configuration with {nan_count} isolated missing feature(s), within validated 5-18% training dropout tolerance.",
                    model_inference_skipped=False,
                    feature_availability=feat_avail,
                    metadata_used=metadata_used
                )
            else:
                # Exceeds validated dropout threshold (> 3 missing features across channels)
                return ModalityGateResult(
                    is_eligible=False,
                    inference_status=STATUS_INSUFFICIENT_MODALITY,
                    missing_modalities=missing_mods,
                    detected_pattern=PATTERN_EXCESSIVE_FEATURE_DROPOUT,
                    technical_reason=f"Input vector has {nan_count} missing features, exceeding the validated 15% isolated dropout limit.",
                    model_inference_skipped=True,
                    feature_availability=feat_avail,
                    metadata_used=metadata_used
                )

        # Default Fail-Closed for any other configuration
        return ModalityGateResult(
            is_eligible=False,
            inference_status=STATUS_INSUFFICIENT_MODALITY,
            missing_modalities=missing_mods,
            detected_pattern=PATTERN_UNSUPPORTED_PARTIAL,
            technical_reason="The detected sensor modality configuration is not supported by training or validation evidence.",
            model_inference_skipped=True,
            feature_availability=feat_avail,
            metadata_used=metadata_used
        )

    @classmethod
    def evaluate_batch(
        cls,
        X_batch: Union[np.ndarray, List[List[float]]],
        sensor_metadata: Optional[Dict[str, Any]] = None
    ) -> List[ModalityGateResult]:
        """Evaluates a batch of 20-feature vectors."""
        X_arr = np.asarray(X_batch, dtype=np.float64)
        if X_arr.ndim == 1:
            return [cls.evaluate_vector(X_arr, sensor_metadata)]
        return [cls.evaluate_vector(row, sensor_metadata) for row in X_arr]


# Functional convenience helper
def check_modality_eligibility(
    x_raw: Union[np.ndarray, List[float]],
    sensor_metadata: Optional[Dict[str, Any]] = None
) -> ModalityGateResult:
    """Convenience functional interface for ModalityEligibilityGate."""
    return ModalityEligibilityGate.evaluate_vector(x_raw, sensor_metadata)
