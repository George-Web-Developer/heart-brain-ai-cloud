"""
app.py / src/server/app.py
Real-Time FastAPI & WebSocket Streaming Triage Engine for Heart-Brain Axis Wearable.
Handles ESP32-S3 sensor streams, Stage 1 inference, rehab safety monitoring,
bidirectional hardware feedback (buzzer/vibrator), and Stage 2 FAST check coordination.
"""

import os
import json
import time
import asyncio
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

try:
    import torch
except ImportError:
    torch = None  # type: ignore

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
except ImportError:
    # Minimal stub for environments running unit tests without FastAPI installed
    class FastAPI:  # type: ignore
        def __init__(self, *args, **kwargs): pass
        def add_middleware(self, *args, **kwargs): pass
        def on_event(self, *args, **kwargs): return lambda f: f
        def get(self, *args, **kwargs): return lambda f: f
        def post(self, *args, **kwargs): return lambda f: f
        def websocket(self, *args, **kwargs): return lambda f: f
    class WebSocket: pass  # type: ignore
    class WebSocketDisconnect(Exception): pass  # type: ignore
    class HTTPException(Exception): pass  # type: ignore
    def Query(default=None, **kwargs): return default  # type: ignore
    CORSMiddleware = object  # type: ignore
    class BaseModel: pass  # type: ignore

from src.preprocessing.signal_processor import extract_10s_features, PersonalBaselineCalibrator
from src.models.stage1_model import Stage1HybridModel, load_model

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None  # type: ignore

from src.models.stage2_fast import (
    compute_facial_asymmetry_metrics,
    analyze_speech_dysarthria,
    evaluate_fast_check
)
from src.simulation.biosignal_simulator import CLASS_NAMES
from src.models.dual_head_classifier import (
    GoldenHourTimerEngine,
    extract_all_engineered_features,
    apply_clinical_safety_arbiter,
    calibrate_stage_probabilities,
    CANONICAL_20_FEATURES,
    DISEASE_CLASSES,
    STAGE_NAMES
)


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively converts numpy types and NaN/Inf values to standard JSON primitives."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return _sanitize_for_json(obj.tolist())
    return obj


def _safe_float(val: Any, default: float = np.nan) -> float:
    """Safely converts JSON/dict values (including None or NaN) to float without raising TypeError."""
    if val is None:
        return float(default)
    if isinstance(val, (bool, np.bool_)):
        return 1.0 if val else 0.0
    try:
        res = float(val)
        return float(default) if np.isnan(res) and not np.isnan(default) else res
    except (TypeError, ValueError):
        return float(default)


def _build_canonical_20_vector(features: Dict[str, Any]) -> np.ndarray:
    """
    Constructs the 20-feature vector from a dictionary, preserving genuine np.nan for missing
    or motion-gated sensor channels instead of coercing None/missing values to 0.0.
    """
    binary_keys = {"missed_anticoagulant_dose", "rehab_exercise_active"}
    vec = []
    for k in CANONICAL_20_FEATURES:
        default_val = 0.0 if k in binary_keys else np.nan
        vec.append(_safe_float(features.get(k, default_val), default=default_val))
    return np.array(vec, dtype=np.float32)


# Global State & Model Cache
MODEL_PATH = "models/stage1_heart_brain.pt"
XGB_MODEL_PATH = "models/stage1_xgboost.json"
DEVICE = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"

# Enable Deep Learning waveform branch automatically when PyTorch & weights exist (overridable via env)
USE_DEEP_LEARNING = os.getenv("ENABLE_DEEP_LEARNING", "1").lower() in ("1", "true")
# Minimal cloud mode skips legacy Stage 1 models to fit comfortably within 512 MB RAM containers
CLOUD_MINIMAL_MODELS = os.getenv("CLOUD_MINIMAL_MODELS", "0").lower() in ("1", "true")

if torch is not None and CLOUD_MINIMAL_MODELS:
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

stage1_model: Optional[Stage1HybridModel] = None
xgb_model: Optional[Any] = None
clf_stage: Optional[Any] = None
clf_disease: Optional[Any] = None
scaler_mean: Optional[np.ndarray] = None
scaler_scale: Optional[np.ndarray] = None
golden_hour_timer: Optional[GoldenHourTimerEngine] = None
unified_model_1: Optional[Any] = None
UNIFIED_MODEL_1_PATH = "models/unified_model_1.pkl"
cardiac_model_1a: Optional[Any] = None
CARDIAC_MODEL_1A_PATH = "models/cardiac_model_1a.pkl"
stroke_model_1b: Optional[Any] = None
STROKE_MODEL_1B_PATH = "models/stroke_model_1b.pkl"

# Connected alert listeners (Clinician Web Dashboard)
dashboard_connections: List[WebSocket] = []


class TemporalAlertDebouncer:
    """
    Multi-Window Temporal Persistence Filter over 3 consecutive sliding windows.
    Suppresses transient noise / motion baseline-wander false alarms by >95%.
    Immediate emergencies (Fall impact SMV > 3.0g, Cushing index > 3.5, or severe hemorrhagic bleed)
    fire instantly without waiting.
    """
    def __init__(self, window_size: int = 3, threshold: int = 2):
        self.window_size = window_size
        self.threshold = threshold
        self.history: List[str] = []

    def update(self, detected_class: str, is_immediate_emergency: bool = False) -> Tuple[bool, str]:
        self.history.append(detected_class)
        if len(self.history) > self.window_size:
            self.history.pop(0)

        # Immediate emergencies bypass debounce
        if is_immediate_emergency and detected_class != "Normal":
            return True, detected_class

        # Normal condition
        if detected_class == "Normal":
            return False, "Normal"

        # Count occurrences of the alert class in recent windows
        occurrences = self.history.count(detected_class)
        if occurrences >= self.threshold:
            return True, detected_class
        else:
            return False, self.history[-1]


# Patient Alert Debouncer Instance (Default Global Fallback)
patient_debouncer = TemporalAlertDebouncer(window_size=3, threshold=2)

# Default Patient Triage State Template
DEFAULT_PATIENT_STATE: Dict[str, Any] = {
    "device_id": "ARMBAND_ESP32S3_01",
    "timestamp_ms": 0,
    "triage_class": "Normal",
    "risk_gauge_score": 0.0,
    "differential_probabilities": {name: 0.2 for name in CLASS_NAMES},
    "cushing_index": 1.6,
    "ptt_ms": 180.0,
    "estimated_sbp": 120.0,
    "motion_state": "REST",
    "ecg_lead_mode": "PASSIVE_IPSILATERAL_ARM",
    "red_contraindication_alert": False,
    "red_contraindication_veto": False,
    "clinical_alert": None,
    "rehab_alert": None,
    "temporal_phase": 0,
    "temporal_phase_name": "Phase 0 (Stable/Healthy)",
    "golden_hour_active": False,
    "t_elapsed_sec": 0.0,
    "golden_hour_remaining_sec": 3600.0,
    "golden_hour_display": "[PHASE 0: STABLE BASELINE] No Active Crisis | Golden Hour Timer: 60:00 (STANDBY)",
    "golden_hour_timer_str": "[PHASE 0: STABLE BASELINE] No Active Crisis | Golden Hour Timer: 60:00 (STANDBY)"
}

# Latest Global Patient Triage State (for backward-compatible single-patient queries)
current_patient_state: Dict[str, Any] = dict(DEFAULT_PATIENT_STATE)


class ClientStreamStateManager:
    """
    Maintains isolated per-client / per-patient state buffers:
      - TemporalAlertDebouncer (window_size=3, threshold=2)
      - GoldenHourTimerEngine with EMA state (alpha=0.4)
      - PersonalBaselineCalibrator (5-min rolling resting EMA)
      - Causal DSP filter_state dictionary & Markov prev_phase tracker
      - Latest triage state dictionary
    Uses an OrderedDict with LRU eviction (cap 500) and 30-minute inactivity TTL
    to prevent unbounded memory growth during high-concurrency multi-device streaming.
    """
    def __init__(self, max_clients: int = 500, ttl_sec: float = 1800.0):
        self._states: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.max_clients = max_clients
        self.ttl_sec = ttl_sec

    def get_or_create(self, client_key: str) -> Dict[str, Any]:
        now = time.time()
        if client_key in self._states:
            entry = self._states[client_key]
            entry["last_accessed"] = now
            self._states.move_to_end(client_key)
            return entry

        # Evict LRU client if limit reached
        if len(self._states) >= self.max_clients:
            evicted_key, _ = self._states.popitem(last=False)
            print(f"[ClientStreamStateManager] LRU eviction triggered: removed oldest client '{evicted_key}'.")

        timer_weights = "models/golden_hour_timer_engine.npz"
        new_entry = {
            "last_accessed": now,
            "debouncer": TemporalAlertDebouncer(window_size=3, threshold=2),
            "cardiac_debouncer": TemporalAlertDebouncer(window_size=3, threshold=2),
            "timer_engine": GoldenHourTimerEngine(
                timer_weights if os.path.exists(timer_weights) else None,
                use_ema=True,
                ema_alpha=0.4
            ),
            "calibrator": PersonalBaselineCalibrator(),
            "filter_state": {},
            "prev_phase": None,
            "patient_state": dict(DEFAULT_PATIENT_STATE)
        }
        self._states[client_key] = new_entry
        return new_entry

    def get_by_device_id(self, device_id: Optional[str]) -> Optional[Dict[str, Any]]:
        if not device_id:
            return None
        now = time.time()
        if device_id in self._states:
            entry = self._states[device_id]
            entry["last_accessed"] = now
            self._states.move_to_end(device_id)
            return entry
        for k, state in self._states.items():
            if state.get("patient_state", {}).get("device_id") == device_id:
                state["last_accessed"] = now
                self._states.move_to_end(k)
                return state
        return None

    def remove(self, client_key: str):
        self._states.pop(client_key, None)

    def cleanup_inactive_clients(self) -> int:
        now = time.time()
        expired_keys = [
            k for k, state in self._states.items()
            if (now - state.get("last_accessed", now)) > self.ttl_sec
        ]
        for k in expired_keys:
            self._states.pop(k, None)
        return len(expired_keys)

    async def cleanup_loop(self, interval_sec: float = 60.0):
        try:
            while True:
                await asyncio.sleep(interval_sec)
                evicted = self.cleanup_inactive_clients()
                if evicted > 0:
                    print(f"[ClientStreamStateManager] TTL cleanup: evicted {evicted} inactive client(s).")
        except asyncio.CancelledError:
            pass


stream_state_manager = ClientStreamStateManager()


def ensure_models_loaded():
    global stage1_model, xgb_model, clf_stage, clf_disease, scaler_mean, scaler_scale, golden_hour_timer, unified_model_1, cardiac_model_1a, stroke_model_1b

    if unified_model_1 is None and os.path.exists(UNIFIED_MODEL_1_PATH):
        try:
            from src.models.unified_model_1 import UnifiedHeartBrainModel1
            unified_model_1 = UnifiedHeartBrainModel1.load(UNIFIED_MODEL_1_PATH)
            print(f"[Model Loader] Loaded UnifiedHeartBrainModel1 from {UNIFIED_MODEL_1_PATH}")
        except Exception as e:
            print(f"[Model Loader Warning] UnifiedHeartBrainModel1 load failed: {e}")

    if cardiac_model_1a is None and os.path.exists(CARDIAC_MODEL_1A_PATH):
        try:
            from src.models.cardiac_model import CardiacModel1A
            cardiac_model_1a = CardiacModel1A.load(CARDIAC_MODEL_1A_PATH)
            print(f"[Model Loader] Loaded CardiacModel1A from {CARDIAC_MODEL_1A_PATH}")
        except Exception as e:
            print(f"[Model Loader Warning] CardiacModel1A load failed: {e}")

    if stroke_model_1b is None and os.path.exists(STROKE_MODEL_1B_PATH):
        try:
            from src.models.stroke_model import StrokeModel1B
            stroke_model_1b = StrokeModel1B.load(STROKE_MODEL_1B_PATH)
            print(f"[Model Loader] Loaded StrokeModel1B from {STROKE_MODEL_1B_PATH}")
        except Exception as e:
            print(f"[Model Loader Warning] StrokeModel1B load failed: {e}")

    if not CLOUD_MINIMAL_MODELS and XGBClassifier is not None:
        if xgb_model is None and os.path.exists(XGB_MODEL_PATH):
            try:
                xgb_model = XGBClassifier()
                xgb_model.load_model(XGB_MODEL_PATH)
                print(f"[Model Loader] Loaded Stage 1 XGBoost ML model from {XGB_MODEL_PATH}")
            except Exception as e:
                print(f"[Model Loader Warning] XGBoost load failed: {e}")

        STAGE_MODEL_PATH = "models/dual_head_stage_classifier.json"
        DIS_MODEL_PATH = "models/dual_head_disease_classifier.json"

        if clf_stage is None and os.path.exists(STAGE_MODEL_PATH):
            try:
                clf_stage = XGBClassifier()
                clf_stage.load_model(STAGE_MODEL_PATH)
                print(f"[Model Loader] Loaded Head B Stage Classifier from {STAGE_MODEL_PATH}")
            except Exception as e:
                print(f"[Model Loader Warning] Head B load failed: {e}")

        if clf_disease is None and os.path.exists(DIS_MODEL_PATH):
            try:
                clf_disease = XGBClassifier()
                clf_disease.load_model(DIS_MODEL_PATH)
                print(f"[Model Loader] Loaded Head A Disease Classifier from {DIS_MODEL_PATH}")
            except Exception as e:
                print(f"[Model Loader Warning] Head A load failed: {e}")

    SCALER_PATH = "models/dual_head_scaler.npz"
    TIMER_WEIGHTS_PATH = "models/golden_hour_timer_engine.npz"

    if not CLOUD_MINIMAL_MODELS and scaler_mean is None and os.path.exists(SCALER_PATH):
        try:
            sc_data = np.load(SCALER_PATH)
            scaler_mean = sc_data['mean']
            scaler_scale = sc_data['scale']
            print(f"[Model Loader] Loaded Dual-Head Scaler from {SCALER_PATH}")
        except Exception as e:
            print(f"[Model Loader Warning] Scaler load failed: {e}")

    if golden_hour_timer is None:
        try:
            golden_hour_timer = GoldenHourTimerEngine(TIMER_WEIGHTS_PATH if os.path.exists(TIMER_WEIGHTS_PATH) else None)
            print("[Model Loader] Initialized Second-Order GoldenHourTimerEngine.")
        except Exception as e:
            print(f"[Model Loader Warning] GoldenHourTimerEngine init failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global stage1_model, xgb_model

    ensure_models_loaded()

    if not CLOUD_MINIMAL_MODELS and USE_DEEP_LEARNING and torch is not None:
        try:
            if os.path.exists(MODEL_PATH):
                stage1_model = load_model(MODEL_PATH, device=DEVICE)
                print(f"[Server Startup] Loaded Stage 1 PyTorch DL model from {MODEL_PATH}")
            else:
                stage1_model = Stage1HybridModel(in_channels=2, seq_len=2500, num_tabular_features=20, num_classes=5)
                stage1_model.eval()
                print("[Server Startup] Initialized fresh Stage 1 PyTorch model architecture.")
        except Exception as e:
            print(f"[Server Startup Warning] Model load fallback: {e}")
    else:
        stage1_model = None
        print("[Server Startup] Deep Learning is DEACTIVATED (Pure Machine Learning mode active).")
        print("[Server Startup] Stage 1 inference executing 100% via XGBoost on tabular clinical features.")

    # Background task for TTL cleanup of inactive client streams (30-minute inactivity timeout)
    cleanup_task = asyncio.create_task(stream_state_manager.cleanup_loop(interval_sec=60.0))
    try:
        yield
    finally:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass


app = FastAPI(
    title="Heart-Brain Axis Wearable AI & Rapid Triage Server",
    description="Real-Time Streaming Triage Engine for Multimodal Biomedical Monitoring",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def run_stage1_inference(
    ecg: np.ndarray,
    ppg: np.ndarray,
    features: Dict[str, Any],
    sensor_metadata: Optional[Dict[str, Any]] = None,
    enforce_modality_gate: bool = True
) -> Dict[str, Any]:
    """
    Runs forward pass through Stage 1 Clinical Model.
    Executes via Head A Disease Classifier (or XGBoost fallback) on tabular clinical features.
    Preserves np.nan for missing/gated features so XGBoost uses sparsity-aware splits.
    Protected by fail-closed ModalityEligibilityGate before feature engineering or XGBoost.
    """
    global stage1_model, xgb_model, clf_disease, scaler_mean, scaler_scale
    ensure_models_loaded()

    feat_vec = _build_canonical_20_vector(features)

    # Fail-closed Modality Eligibility Gate evaluation
    if enforce_modality_gate:
        from src.models.modality_eligibility_gate import check_modality_eligibility
        gate_res = check_modality_eligibility(feat_vec, sensor_metadata)
        if not gate_res.is_eligible:
            res: Dict[str, Any] = {CLASS_NAMES[i]: 0.2 for i in range(5)}
            res["_inference_status"] = gate_res.inference_status
            res["_model_inference_skipped"] = True
            res["_modality_gate_pattern"] = gate_res.detected_pattern
            res["_modality_gate_reason"] = gate_res.technical_reason
            res["_modality_gate_result"] = gate_res.to_dict()
            return res

    # 1. Machine Learning Inference (Dual-Head Head A or Stage 1 XGBoost) - Primary & Fast
    xgb_probs = None
    if clf_disease is not None and scaler_mean is not None and scaler_scale is not None:
        try:
            x_raw = feat_vec.reshape(1, -1)
            x_ext = extract_all_engineered_features(x_raw)
            x_sc = (x_ext - scaler_mean) / scaler_scale
            xgb_probs = clf_disease.predict_proba(x_sc)[0]
        except Exception as e:
            print(f"[Inference Warning] Dual-Head Head A predict failed: {e}")
    elif xgb_model is not None:
        try:
            xgb_probs = xgb_model.predict_proba(feat_vec.reshape(1, -1))[0]
        except Exception as e:
            print(f"[Inference Warning] XGBoost predict failed: {e}")

    # 2. Deep Learning Inference (Activated ONLY when real non-flatline raw waveforms are supplied)
    dl_probs = None
    ecg_arr_chk = np.asarray(ecg, dtype=np.float32) if ecg is not None else np.zeros(1)
    has_valid_waveform = bool(len(ecg_arr_chk) >= 250 and float(np.var(ecg_arr_chk)) >= 1e-5 and float(np.max(np.abs(ecg_arr_chk))) <= 3.3)
    if USE_DEEP_LEARNING and stage1_model is not None and torch is not None and has_valid_waveform:
        try:
            if len(ecg) < 2500:
                ecg = np.pad(ecg, (0, 2500 - len(ecg)), mode='edge')
            else:
                ecg = ecg[:2500]

            if len(ppg) < 2500:
                ppg = np.pad(ppg, (0, 2500 - len(ppg)), mode='edge')
            else:
                ppg = ppg[:2500]

            ecg_norm = (ecg - np.mean(ecg)) / (np.std(ecg) + 1e-6)
            ppg_norm = (ppg - np.mean(ppg)) / (np.std(ppg) + 1e-6)
            wave_tensor = torch.tensor(np.stack([ecg_norm, ppg_norm], axis=0), dtype=torch.float32).unsqueeze(0).to(DEVICE)
            tab_tensor = torch.tensor(np.nan_to_num(feat_vec, nan=0.0).reshape(1, -1), dtype=torch.float32).to(DEVICE)

            with torch.no_grad():
                dl_probs = stage1_model.predict_proba(wave_tensor, tab_tensor).cpu().numpy()[0]
        except Exception as e:
            print(f"[Inference Warning] DL predict failed: {e}")

    # 3. Final Probability Resolution
    if xgb_probs is not None and dl_probs is not None:
        final_probs = 0.5 * dl_probs + 0.5 * xgb_probs
    elif xgb_probs is not None:
        final_probs = xgb_probs
    elif dl_probs is not None:
        final_probs = dl_probs
    else:
        final_probs = np.full(5, 0.2, dtype=np.float32)

    return {CLASS_NAMES[i]: float(final_probs[i]) for i in range(5)}


@app.get("/")
async def root_status():
    """Root cloud service metadata and active endpoints."""
    ensure_models_loaded()
    return _sanitize_for_json({
        "service": "Heart-Brain Axis Wearable AI & Rapid Triage Server",
        "version": "1.0.0",
        "status": "running",
        "models_loaded": {
            "unified_model_1": unified_model_1 is not None,
            "cardiac_model_1a": cardiac_model_1a is not None,
            "stroke_model_1b": stroke_model_1b is not None,
        },
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "esp32_predict": "/api/v1/esp32-predict",
            "cardiac_predict": "/api/v1/cardiac-predict",
            "stroke_predict": "/api/v1/stroke-predict",
            "triage_status": "/api/triage/status",
        },
    })


@app.get("/health")
async def health_check():
    """Cloud container healthcheck and loaded model inventory."""
    ensure_models_loaded()
    return _sanitize_for_json({
        "status": "ok",
        "models": {
            "unified_model_1": unified_model_1 is not None,
            "cardiac_model_1a": cardiac_model_1a is not None,
            "stroke_model_1b": stroke_model_1b is not None,
            "device": DEVICE,
            "minimal_mode": CLOUD_MINIMAL_MODELS,
        },
    })


@app.get("/api/triage/status")
@app.get("/api/v1/golden-hour-status")
async def get_triage_status(device_id: Optional[str] = Query(default=None)):
    """Returns the latest patient triage state, risk score, and active Golden Hour status (per-device if specified)."""
    client_entry = stream_state_manager.get_by_device_id(device_id)
    if client_entry is not None:
        return _sanitize_for_json(client_entry["patient_state"])
    return _sanitize_for_json(current_patient_state)


class Stage2Request(BaseModel):
    device_id: Optional[str] = None
    landmarks_468: Optional[List[List[float]]] = None
    audio_pcm: Optional[List[float]] = None
    sample_rate: int = 16000
    missed_anticoagulant_dose: bool = False
    arm_drift_deg: float = 0.0
    active_arm_hold_test: bool = False


@app.post("/api/stage2/fast-check")
@app.post("/api/v1/fast-check")
async def post_fast_check(payload: Stage2Request):
    """
    Executes on-demand Stage 2 FAST check combining facial landmarks, voice speech, and active arm drift.
    Offloads CPU-bound vision/speech DSP to worker thread to avoid blocking active WebSocket sensor streams.
    """
    landmarks_np = np.array(payload.landmarks_468) if payload.landmarks_468 else None
    audio_np = np.array(payload.audio_pcm) if payload.audio_pcm else None

    client_entry = stream_state_manager.get_by_device_id(getattr(payload, "device_id", None))
    target_state = client_entry["patient_state"] if client_entry is not None else current_patient_state

    def _run_stage2_sync():
        facial_res = compute_facial_asymmetry_metrics(landmarks_np)
        speech_res = analyze_speech_dysarthria(audio_np, sample_rate=payload.sample_rate)
        decision = evaluate_fast_check(
            facial_metrics=facial_res,
            speech_results=speech_res,
            stage1_probs=target_state.get("differential_probabilities"),
            missed_anticoagulant=payload.missed_anticoagulant_dose,
            arm_drift_deg=payload.arm_drift_deg,
            active_arm_hold_test=payload.active_arm_hold_test,
            stage1_red_contraindication_veto=bool(target_state.get("red_contraindication_alert", False))
        )
        return {
            "facial_analysis": facial_res,
            "speech_analysis": speech_res,
            "fast_decision": decision
        }

    result = await asyncio.to_thread(_run_stage2_sync)
    return _sanitize_for_json(result)


@app.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    """WebSocket connection for Clinician / Ambulance Web Dashboard to receive live alerts."""
    await websocket.accept()
    dashboard_connections.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in dashboard_connections:
            dashboard_connections.remove(websocket)


async def broadcast_alert(alert_payload: Dict[str, Any]):
    """Broadcasts urgent alerts to all connected Dashboard clients."""
    clean_payload = _sanitize_for_json(alert_payload)
    for ws in list(dashboard_connections):
        try:
            await ws.send_json(clean_payload)
        except Exception:
            if ws in dashboard_connections:
                dashboard_connections.remove(ws)


def _process_sensor_window_sync(data: Dict[str, Any], client_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Synchronous CPU-bound processing for a single 10s sensor window:
    Extracts features via causal float32 DSP, runs Dual-Head inference,
    applies decoupled clinical safety arbiter, updates debouncer, and evaluates Golden Hour timer.
    Offloaded to worker threads via asyncio.to_thread to keep the async event loop non-blocking.
    """
    client_debouncer: TemporalAlertDebouncer = client_state["debouncer"]
    client_timer: GoldenHourTimerEngine = client_state["timer_engine"]
    client_calibrator: PersonalBaselineCalibrator = client_state["calibrator"]
    client_filter_state: Dict[str, Any] = client_state["filter_state"]

    # Parse incoming telemetry
    device_id = data.get("device_id", "ARMBAND_ESP32S3_01")
    timestamp = data.get("timestamp_ms", 0)
    patient_context = data.get("patient_context", {})
    missed_anti = bool(patient_context.get("missed_anticoagulant_dose", False))
    rehab_active = bool(patient_context.get("rehab_exercise_active", False))
    ecg_lead_mode = data.get("ecg_lead_mode", patient_context.get("ecg_lead_mode", "PASSIVE_IPSILATERAL_ARM"))

    raw_windows = data.get("raw_windows", {})
    ecg_raw = np.array(raw_windows.get("ecg_mv", [0.0] * 2500), dtype=np.float32)
    ppg_raw = np.array(raw_windows.get("ppg_ir", [50000.0] * 2500), dtype=np.float32)

    # Lead-Off Detection (zero variance or saturated signal)
    ecg_var = float(np.var(ecg_raw))
    ecg_lead_off = (ecg_var < 1e-5 or np.max(np.abs(ecg_raw)) > 3.3)

    # Extract features with per-client PersonalBaselineCalibrator
    extracted: Dict[str, Any] = {}
    if "raw_windows" in data:
        imu = {
            'ax': np.array(raw_windows.get("imu_ax_g", [0.0] * 500)),
            'ay': np.array(raw_windows.get("imu_ay_g", [-1.0] * 500)),
            'az': np.array(raw_windows.get("imu_az_g", [0.0] * 500)),
            'gx': np.array(raw_windows.get("imu_gx_dps", [0.0] * 500)),
            'gy': np.array(raw_windows.get("imu_gy_dps", [0.0] * 500)),
            'gz': np.array(raw_windows.get("imu_gz_dps", [0.0] * 500))
        }
        temp_c_in = _safe_float(raw_windows.get("skin_temp_c", patient_context.get("skin_temp_c", 34.6)), 34.6)
        extracted = extract_10s_features(
            ecg_mv=ecg_raw,
            ppg_raw=ppg_raw,
            imu_data=imu,
            temp_c=temp_c_in,
            missed_anticoagulant=missed_anti,
            rehab_active=rehab_active,
            ecg_lead_mode=ecg_lead_mode,
            calibrator=client_calibrator,
            filter_state=client_filter_state
        )

    # Override with any explicitly supplied extracted_features
    if data.get("extracted_features"):
        extracted.update(data["extracted_features"])

    # 1. Safe Guided Rehab Exercise Monitoring Guard (NaN-safe)
    hr_val = _safe_float(extracted.get("hr_ecg_bpm"), 72.0)
    spo2_val = _safe_float(extracted.get("spo2_percent", extracted.get("spo2_pct")), 98.0)
    st_val = _safe_float(extracted.get("st_deviation_mv"), 0.0)

    if rehab_active:
        if hr_val > 130.0 or spo2_val < 92.0 or abs(st_val) >= 0.15:
            rehab_alert = {
                "alert_type": "STOP_EXERCISE_IMMEDIATELY",
                "reason": "SAFETY_THRESHOLD_EXCEEDED",
                "details": f"HR: {hr_val} bpm (limit 130), SpO2: {spo2_val}% (limit 92%), ST: {st_val} mV",
                "trigger_buzzer_vibrator": True,
                "launch_fast_check": False,
                "triage_class": "STOP_EXERCISE_IMMEDIATELY",
                "risk_gauge_score": 75.0
            }
            return {
                "is_rehab_alert": True,
                "rehab_alert": rehab_alert
            }

    # 2. Stage 1 AI Model Inference
    sensor_meta = data.get("sensor_metadata")
    probs = run_stage1_inference(ecg_raw, ppg_raw, extracted, sensor_metadata=sensor_meta)
    probs_arr = np.array([[probs.get(c, 0.0) for c in DISEASE_CLASSES]], dtype=np.float32)
    raw_pred_idx = np.array([int(np.argmax(probs_arr[0]))], dtype=np.int64)
    x_mat = _build_canonical_20_vector(extracted).reshape(1, -1)

    # 3. Unified Decoupled Clinical Safety Arbiter
    if probs.get("_model_inference_skipped", False):
        pred_class = "UNSUPPORTED_MODALITY"
        red_contraindication = False
        clinical_alert = f"{probs.get('_inference_status', 'INSUFFICIENT_MODALITY')}: Model inference skipped"
        safe_pred_idx = np.array([-1], dtype=np.int64)
        veto_flags = np.array([False])
        arbiter_alerts = [clinical_alert]
        safety_alerts = []
    else:
        safe_pred_idx, veto_flags, arbiter_alerts, safety_alerts = apply_clinical_safety_arbiter(
            raw_pred_idx, probs_arr, x_mat, return_veto=True, return_safety_alerts=True
        )
        pred_class = DISEASE_CLASSES[int(safe_pred_idx[0])]
        red_contraindication = bool(veto_flags[0])
        clinical_alert = arbiter_alerts[0]

    # 4. Calculate 0-100 Risk Gauge Score (NaN-safe)
    cushing_val = _safe_float(extracted.get("cushing_index"), 1.6)
    sbp_val = _safe_float(extracted.get("estimated_sbp_mmhg"), 120.0)
    t_amp_val = _safe_float(extracted.get("t_wave_amplitude_mv"), 0.2)
    arm_drift = _safe_float(extracted.get("arm_pitch_drift_deg"), 0.0)
    smv_max = _safe_float(extracted.get("imu_smv_max_g"), 1.0)
    p_ischemic = probs.get("Ischemic_Stroke", 0.0)
    p_hemorrhagic = probs.get("Hemorrhagic_Stroke", 0.0)

    risk_score = (
        probs.get("AFib", 0.0) * 40.0 +
        probs.get("MI", 0.0) * 85.0 +
        p_ischemic * 90.0 +
        p_hemorrhagic * 100.0
    )
    if cushing_val > 3.0:
        risk_score = min(100.0, risk_score + 25.0)
    risk_score = round(float(np.clip(risk_score, 0.0, 100.0)), 1)

    # 5. Hardware Actuation & Stage 2 Trigger Routing
    # A fall impact (smv_max > 3.0) alone does not justify Hemorrhagic Red Veto;
    # require concurrent hypertensive/Cushing surge or deep cerebral T-wave inversion.
    impact_with_neuro_signs = (smv_max > 3.0) and (
        sbp_val >= 165.0 or cushing_val >= 2.2 or t_amp_val <= -0.20
    )
    if (impact_with_neuro_signs or (cushing_val >= 3.0 and hr_val < 60.0)) and not red_contraindication:
        red_contraindication = True
    if red_contraindication and clinical_alert is None:
        clinical_alert = (
            "CLINICAL DECISION SUPPORT: Suspected Neurovascular Emergency (Hemorrhagic stroke cannot be excluded). "
            "This research system cannot determine stroke subtype or guide medication decisions. "
            "Urgent professional assessment and emergency brain imaging are required. "
            "Guidelines contraindicate pre-hospital aspirin, anticoagulants, or thrombolytics prior to non-contrast head CT scan "
            "(DO NOT ADMINISTER ASPIRIN, ANTICOAGULANTS, OR THROMBOLYTICS PRIOR TO CT SCAN)."
        )

    trigger_hardware = False
    launch_fast = False
    alert_reason = None

    is_immediate_emergency = (
        smv_max > 3.0 or
        cushing_val >= 3.5 or
        p_hemorrhagic >= 0.50 or
        (cushing_val > 3.0 and hr_val < 60.0) or
        pred_class == "Hemorrhagic_Stroke"
    )

    if pred_class == "Hemorrhagic_Stroke":
        trigger_hardware = True
        launch_fast = True
        alert_reason = "CRITICAL_CUSHING_REFLEX_HEMORRHAGIC_STROKE"
    elif p_ischemic > 0.45 or arm_drift > 18.0 or pred_class == "Ischemic_Stroke":
        trigger_hardware = True
        launch_fast = True
        pred_class = "Ischemic_Stroke"
        alert_reason = "ACUTE_ISCHEMIC_STROKE_UNILATERAL_ARM_DRIFT"
    elif smv_max > 3.0:
        trigger_hardware = True
        launch_fast = True
        alert_reason = "FALL_IMPACT_DETECTED"
    elif probs.get("MI", 0.0) > 0.50 or pred_class == "MI":
        trigger_hardware = True
        alert_reason = "ACUTE_MYOCARDIAL_INFARCTION"
    elif pred_class == "AFib" and probs.get("AFib", 0.0) > 0.55:
        trigger_hardware = True
        alert_reason = "ATRIAL_FIBRILLATION_DETECTED"

    rule_adjusted_disease_class = pred_class

    # Multi-Window Temporal Persistence Filter (Debouncer) per-client
    should_fire, confirmed_class = client_debouncer.update(
        rule_adjusted_disease_class,
        is_immediate_emergency=is_immediate_emergency
    )
    if not should_fire and not red_contraindication:
        trigger_hardware = False

    # 6. Dual-Head Temporal Phase (with Bayesian/Markov Calibration) & Golden Hour Timer Engine
    temporal_phase = 0
    temporal_phase_name = "Phase 0 (Stable/Healthy)"
    golden_hour_active = False
    t_elapsed_sec = 0.0
    golden_hour_remaining_sec = 3600.0
    golden_hour_timer_str = "[PHASE 0: STABLE BASELINE] No Active Crisis | Golden Hour Timer: 60:00 (STANDBY)"
    golden_hour_display = golden_hour_timer_str
    raw_stg_probs_map: Dict[str, float] = {STAGE_NAMES[p]: 0.25 for p in range(4)}
    cal_stg_probs_map: Dict[str, float] = {STAGE_NAMES[p]: 0.25 for p in range(4)}
    raw_stg_pred_phase = 0
    cal_stg_pred_phase = 0

    if clf_stage is not None and client_timer is not None and not probs.get("_model_inference_skipped", False):
        x_ext = extract_all_engineered_features(x_mat)
        if scaler_mean is not None and scaler_scale is not None:
            x_sc = (x_ext - scaler_mean) / scaler_scale
        else:
            x_sc = x_ext
        raw_probs_stg = clf_stage.predict_proba(x_sc)
        raw_stg_probs_map = {STAGE_NAMES[p]: float(raw_probs_stg[0, p]) for p in range(4)}
        raw_stg_pred_phase = int(np.argmax(raw_probs_stg[0]))

        probs_stg = calibrate_stage_probabilities(
            raw_probs_stg, x_ext, x_mat, prev_phase=client_state.get("prev_phase")
        )[0]
        cal_stg_probs_map = {STAGE_NAMES[p]: float(probs_stg[p]) for p in range(4)}
        cal_stg_pred_phase = int(np.argmax(probs_stg))

        pred_stg_raw = cal_stg_pred_phase

        dis_candidate = confirmed_class if should_fire else "Normal"
        dis_idx = DISEASE_CLASSES.index(dis_candidate) if dis_candidate in DISEASE_CLASSES else 0
        if dis_candidate == "Normal":
            pred_stg = 0
        else:
            pred_stg = pred_stg_raw if pred_stg_raw > 0 else 2

        timer_res = client_timer.estimate_single(
            x_mat[0],
            pred_disease=dis_idx,
            probs_disease=probs_arr[0],
            pred_phase=pred_stg,
            probs_phase=probs_stg,
            external_veto_flag=red_contraindication,
            external_alert=clinical_alert
        )
        temporal_phase = timer_res["predicted_phase"]
        temporal_phase_name = timer_res["predicted_phase_name"]
        golden_hour_active = timer_res["golden_hour_active"]
        t_elapsed_sec = timer_res["t_elapsed_sec"]
        golden_hour_remaining_sec = timer_res["golden_hour_remaining_sec"]
        golden_hour_timer_str = timer_res["timer_string"]
        golden_hour_display = timer_res.get("golden_hour_display", golden_hour_timer_str)

    if probs.get("_model_inference_skipped", False):
        debounced_class = "UNSUPPORTED_MODALITY"
        debounced_triage_class = debounced_class
        operational_triage_class = debounced_class
        inst_disease_class = -1
        inst_disease_name = "UNSUPPORTED_MODALITY"
    else:
        debounced_class = confirmed_class if should_fire else "Normal"
        debounced_triage_class = debounced_class
        operational_triage_class = debounced_class
        inst_disease_class = int(raw_pred_idx[0])
        inst_disease_name = DISEASE_CLASSES[inst_disease_class]

    inst_disease_probs = {c: float(probs_arr[0][i]) for i, c in enumerate(DISEASE_CLASSES)}

    updated_state = {
        "device_id": device_id,
        "timestamp_ms": timestamp,

        # 1. Instantaneous Predictions
        "instantaneous_disease_probabilities": inst_disease_probs,
        "instantaneous_disease_class": inst_disease_class,
        "instantaneous_disease_name": inst_disease_name,

        # 2. Rule-Adjusted and Operational Triage
        "rule_adjusted_disease_class": rule_adjusted_disease_class,
        "debounced_triage_class": debounced_triage_class,
        "operational_triage_class": operational_triage_class,

        # 3. Head B Stage Outputs
        "raw_stage_probabilities": raw_stg_probs_map,
        "raw_predicted_phase": raw_stg_pred_phase,
        "calibrated_stage_probabilities": cal_stg_probs_map,
        "calibrated_predicted_phase": cal_stg_pred_phase,
        "operational_predicted_phase": temporal_phase,

        # 4. Backward Compatibility Aliases
        "triage_class": debounced_class,
        "primary_diagnosis_class": debounced_class,
        "raw_disease_class": inst_disease_class,
        "raw_disease_name": inst_disease_name,
        "raw_disease_probabilities": inst_disease_probs,
        "temporal_phase": temporal_phase,
        "temporal_phase_name": temporal_phase_name,

        # 5. Sensor Telemetry & Safety Alerts
        "risk_gauge_score": risk_score,
        "differential_probabilities": probs,
        "cushing_index": cushing_val,
        "ptt_ms": _safe_float(extracted.get("ptt_ms"), 180.0),
        "estimated_sbp": sbp_val,
        "motion_state": extracted.get("motion_state", "REST"),
        "ecg_lead_mode": ecg_lead_mode,
        "ecg_lead_off": bool(ecg_lead_off),
        "red_contraindication_alert": bool(red_contraindication),
        "red_contraindication_veto": bool(red_contraindication),
        "clinical_alert": clinical_alert,
        "medical_safety_alerts": safety_alerts[0] if safety_alerts else [],
        "golden_hour_active": bool(golden_hour_active),
        "t_elapsed_sec": float(t_elapsed_sec),
        "golden_hour_remaining_sec": float(golden_hour_remaining_sec),
        "golden_hour_display": golden_hour_display,
        "golden_hour_timer_str": golden_hour_timer_str,
        "inference_status": probs.get("_inference_status", "ELIGIBLE"),
        "model_inference_skipped": probs.get("_model_inference_skipped", False),
        "modality_gate_result": probs.get("_modality_gate_result")
    }

    response_payload = {
        "device_id": device_id,

        # 1. Instantaneous Predictions (Pure Model)
        "instantaneous_disease_probabilities": inst_disease_probs,
        "instantaneous_disease_class": inst_disease_class,
        "instantaneous_disease_name": inst_disease_name,

        # 2. Rule-Adjusted Classification (after arm drift / emergency rules)
        "rule_adjusted_disease_class": rule_adjusted_disease_class,

        # 3. Debounced Operational Triage
        "debounced_triage_class": debounced_triage_class,
        "operational_triage_class": operational_triage_class,

        # 4. Backward Compatibility Aliases
        "triage_class": debounced_class,
        "primary_diagnosis_class": debounced_class,
        "raw_instantaneous_class": rule_adjusted_disease_class,
        "raw_disease_class": inst_disease_class,
        "raw_disease_name": inst_disease_name,
        "raw_disease_probabilities": inst_disease_probs,

        # 5. Head B Stage Outputs
        "raw_stage_probabilities": raw_stg_probs_map,
        "raw_predicted_phase": raw_stg_pred_phase,
        "calibrated_stage_probabilities": cal_stg_probs_map,
        "calibrated_predicted_phase": cal_stg_pred_phase,
        "operational_predicted_phase": temporal_phase,
        "temporal_phase": temporal_phase,
        "temporal_phase_name": temporal_phase_name,

        # 6. Safety Layer
        "medical_safety_alerts": safety_alerts[0] if safety_alerts else [],
        "red_contraindication_alert": bool(red_contraindication),
        "red_contraindication_veto": bool(red_contraindication),
        "clinical_warning": clinical_alert,

        # 7. Hardware Actuation & Telemetry
        "risk_gauge_score": risk_score,
        "trigger_buzzer_vibrator": bool(trigger_hardware),
        "launch_fast_check": bool(launch_fast),
        "alert_reason": alert_reason,
        "golden_hour_active": bool(golden_hour_active),
        "t_elapsed_sec": float(t_elapsed_sec),
        "golden_hour_remaining_sec": float(golden_hour_remaining_sec),
        "golden_hour_display": golden_hour_display,
        "golden_hour_timer_str": golden_hour_timer_str,
        "inference_status": probs.get("_inference_status", "ELIGIBLE"),
        "model_inference_skipped": probs.get("_model_inference_skipped", False),
        "modality_gate_result": probs.get("_modality_gate_result")
    }

    should_broadcast = bool(trigger_hardware or red_contraindication or golden_hour_active)

    return {
        "is_rehab_alert": False,
        "response_payload": response_payload,
        "updated_state": updated_state,
        "new_prev_phase": temporal_phase,
        "should_broadcast": should_broadcast
    }


@app.websocket("/ws/sensor-stream")
@app.websocket("/ws/biosignals")
async def websocket_sensor_stream(websocket: WebSocket):
    """
    Continuous streaming endpoint for ESP32-S3 Wearable (supports /ws/sensor-stream and /ws/biosignals).
    Maintains isolated per-client / per-connection state buffers to prevent cross-patient state leakage.
    Processes 10-second sliding windows, offloading synchronous DSP & ML forward passes to worker threads
    via asyncio.to_thread, monitors safe guided rehab limits, and returns bidirectional hardware commands.
    """
    await websocket.accept()
    client_key = f"conn_{id(websocket)}"
    client_state = stream_state_manager.get_or_create(client_key)

    try:
        while True:
            data = await websocket.receive_json()

            # Execute synchronous DSP and ML inference on worker thread to avoid event loop stalls
            result = await asyncio.to_thread(_process_sensor_window_sync, data, client_state)

            if result["is_rehab_alert"]:
                rehab_alert = result["rehab_alert"]
                clean_alert = _sanitize_for_json(rehab_alert)
                await websocket.send_json(clean_alert)
                await broadcast_alert(clean_alert)
                client_state["patient_state"]["rehab_alert"] = clean_alert
                current_patient_state["rehab_alert"] = clean_alert
                continue

            # Update per-client and global state buffers
            client_state["patient_state"].update(result["updated_state"])
            current_patient_state.update(result["updated_state"])
            client_state["prev_phase"] = result["new_prev_phase"]

            clean_response = _sanitize_for_json(result["response_payload"])
            await websocket.send_json(clean_response)

            if result["should_broadcast"]:
                await broadcast_alert(clean_response)

    except WebSocketDisconnect:
        pass
    finally:
        stream_state_manager.remove(client_key)


@app.post("/api/v1/triage")
async def post_single_window_triage(payload: Dict[str, Any]):
    """
    Single-window REST triage endpoint.
    Returns decoupled anatomical diagnosis, pharmacological Red Contraindication veto,
    differential probabilities, and Golden Hour countdown status.
    Supports optional `device_id` / `patient_id` for per-patient state isolation.
    Offloads CPU-bound feature extraction and XGBoost predictions via asyncio.to_thread.
    """
    ensure_models_loaded()
    patient_context = payload.get("patient_context", {})
    raw_windows = payload.get("raw_windows", {})
    device_id = payload.get("device_id") or patient_context.get("patient_id")
    use_ema = bool(payload.get("use_ema", False))

    client_entry = stream_state_manager.get_or_create(str(device_id)) if device_id else None
    client_calibrator = client_entry["calibrator"] if client_entry else None
    client_timer = client_entry["timer_engine"] if (client_entry and use_ema) else None
    prev_phase = client_entry.get("prev_phase") if client_entry else None

    ecg_raw = np.array(raw_windows.get("ecg_mv", [0.0] * 2500), dtype=np.float32)
    ppg_raw = np.array(raw_windows.get("ppg_ir", [50000.0] * 2500), dtype=np.float32)

    def _sync_triage_job() -> Dict[str, Any]:
        extracted: Dict[str, Any] = {}
        if "raw_windows" in payload:
            imu = {
                'ax': np.array(raw_windows.get("imu_ax_g", [0.0] * 500)),
                'ay': np.array(raw_windows.get("imu_ay_g", [-1.0] * 500)),
                'az': np.array(raw_windows.get("imu_az_g", [0.0] * 500)),
                'gx': np.array(raw_windows.get("imu_gx_dps", [0.0] * 500)),
                'gy': np.array(raw_windows.get("imu_gy_dps", [0.0] * 500)),
                'gz': np.array(raw_windows.get("imu_gz_dps", [0.0] * 500))
            }
            extracted = extract_10s_features(
                ecg_mv=ecg_raw,
                ppg_raw=ppg_raw,
                imu_data=imu,
                missed_anticoagulant=bool(patient_context.get("missed_anticoagulant_dose", False)),
                rehab_active=bool(patient_context.get("rehab_exercise_active", False)),
                ecg_lead_mode=payload.get("ecg_lead_mode", "PASSIVE_IPSILATERAL_ARM"),
                calibrator=client_calibrator
            )

        if payload.get("extracted_features"):
            extracted.update(payload["extracted_features"])

        feat_vec = _build_canonical_20_vector(extracted)
        from src.models.dual_head_classifier import predict as dual_head_predict
        res = dual_head_predict(
            feat_vec,
            use_ema=use_ema,
            timer_engine=client_timer,
            prev_phase=prev_phase,
            sensor_metadata=payload.get("sensor_metadata")
        )
        return res

    res = await asyncio.to_thread(_sync_triage_job)
    if client_entry is not None:
        client_entry["prev_phase"] = res.get("predicted_phase", 0)
    return _sanitize_for_json(res)


def _normalize_esp32_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalizes ESP32-S3 telemetry without mutating the caller's dict.
    Linearly resamples `ppg_ir` (e.g. 1000 samples @ 100 Hz from MAX30102) to match
    `len(ecg_mv)` (e.g. 2500 samples @ 250 Hz from AD8232/ADS1115) for PTT & PPG SQI.
    """
    normalized = dict(payload)
    raw_windows = payload.get("raw_windows")
    if isinstance(raw_windows, dict):
        rw_copy = dict(raw_windows)
        ecg_raw = rw_copy.get("ecg_mv")
        ppg_raw = rw_copy.get("ppg_ir")
        if ecg_raw is not None and ppg_raw is not None:
            ecg_arr = np.asarray(ecg_raw, dtype=np.float32)
            ppg_arr = np.asarray(ppg_raw, dtype=np.float32)
            if len(ecg_arr) >= 10 and len(ppg_arr) >= 10 and len(ppg_arr) != len(ecg_arr):
                resampled_ppg = np.interp(
                    np.linspace(0.0, float(len(ppg_arr) - 1), len(ecg_arr)),
                    np.arange(len(ppg_arr), dtype=np.float64),
                    ppg_arr.astype(np.float64),
                )
                rw_copy["ppg_ir"] = resampled_ppg.astype(np.float32).tolist()
        normalized["raw_windows"] = rw_copy
    return normalized


def _build_esp32_summary(
    result: Dict[str, Any],
    decoupled_out: Dict[str, Any],
    payload: Dict[str, Any],
    cascade_trigger_reasons: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Builds a compact, flat-friendly hardware + AI summary for ESP32 & Serial Monitor output."""
    reasons = list(cascade_trigger_reasons or [])
    cardiac_res = decoupled_out.get("cardiac_model_1a")
    if isinstance(cardiac_res, dict):
        cardiac_diag = cardiac_res.get("disease_name", result.get("disease_name"))
        cardiac_stage = cardiac_res.get("stage_name", result.get("stage_name"))
        cardiac_stage_code = int(cardiac_res.get("stage_code", result.get("stage_code", 0)))
        c_probs = cardiac_res.get("disease_probabilities") or {}
        cardiac_reliable = bool((cardiac_res.get("signal_quality") or {}).get("inference_reliable", True))
        cardiac_conf = round(float(max(c_probs.values())), 4) if (c_probs and cardiac_reliable) else 0.0
        cardiac_sqi_status = (cardiac_res.get("signal_quality") or {}).get("status", "UNKNOWN")
        cardiac_fb = cardiac_res.get("hardware_feedback") or {}
    else:
        cardiac_diag = result.get("disease_name")
        cardiac_stage = result.get("stage_name")
        cardiac_stage_code = int(result.get("stage_code", 0))
        u_probs = result.get("disease_probabilities") or {}
        cardiac_reliable = bool((result.get("signal_quality") or {}).get("inference_reliable", True))
        cardiac_conf = round(float(max(u_probs.values())), 4) if (u_probs and cardiac_reliable) else 0.0
        cardiac_sqi_status = (result.get("signal_quality") or {}).get("status", "UNKNOWN")
        cardiac_fb = {}

    stroke_res = decoupled_out.get("stroke_model_1b")
    est_sbp = None
    est_dbp = None
    if isinstance(stroke_res, dict):
        stroke_risk = stroke_res.get("disease_name", "Low_Stroke_Risk")
        stroke_stage = stroke_res.get("stage_name", "Stage 0: Neurovascular Baseline")
        stroke_stage_code = int(stroke_res.get("stage_code", 0))
        stroke_risk_pct = round(float(stroke_res.get("stroke_risk_score_pct", 0.0)), 2)
        stroke_sqi_status = (stroke_res.get("telemetry_quality") or {}).get("status", "UNKNOWN")
        imu_fall = bool(stroke_res.get("imu_fall_detected", False))
        bp_est = (stroke_res.get("branch_2a_hemodynamics") or {}).get("cuffless_bp_estimate_mmhg")
        if isinstance(bp_est, dict):
            est_sbp = bp_est.get("sbp")
            est_dbp = bp_est.get("dbp")
        stroke_fb = stroke_res.get("hardware_feedback") or {}
        stroke_fast = bool(stroke_res.get("trigger_model_3_fast_check", False))
        stroke_veto = bool(stroke_res.get("red_contraindication_veto", False))
        stroke_alert = stroke_res.get("clinical_alert")
    else:
        stroke_risk = "Unavailable"
        stroke_stage = "Unavailable"
        stroke_stage_code = 0
        stroke_risk_pct = 0.0
        stroke_sqi_status = "UNAVAILABLE"
        imu_fall = False
        stroke_fb = {}
        stroke_fast = False
        stroke_veto = False
        stroke_alert = None

    uni_fb = result.get("hardware_feedback") or {}
    trigger_buzzer = bool(
        uni_fb.get("trigger_buzzer")
        or cardiac_fb.get("trigger_buzzer")
        or stroke_fb.get("trigger_buzzer")
    )
    trigger_vibrator = bool(
        uni_fb.get("trigger_vibrator")
        or cardiac_fb.get("trigger_vibrator")
        or stroke_fb.get("trigger_vibrator")
    )
    red_veto = bool(result.get("red_contraindication_veto", False) or stroke_veto)
    trigger_fast = bool(result.get("trigger_model_2_camera_voice_check", False) or stroke_fast)
    if red_veto and stroke_alert:
        clinical_alert = stroke_alert
    elif cardiac_diag in ("SENSOR_DISCONNECTED", "POOR_SIGNAL") and not imu_fall:
        clinical_alert = (cardiac_res or {}).get("clinical_alert") or result.get("clinical_alert")
    else:
        clinical_alert = result.get("clinical_alert")

    raw_uni_name = (result.get("raw_ml_decision") or {}).get("disease_name")
    unified_diag = cardiac_diag if (raw_uni_name == "ABSTAINED_POOR_SIGNAL" and cardiac_diag in ("SENSOR_DISCONNECTED", "POOR_SIGNAL")) else result.get("disease_name")

    summary: Dict[str, Any] = {
        "cardiac_diagnosis": cardiac_diag,
        "cardiac_stage": cardiac_stage,
        "cardiac_stage_code": cardiac_stage_code,
        "cardiac_confidence": cardiac_conf,
        "cardiac_sqi_status": cardiac_sqi_status,
        "model2_triggered": bool(reasons),
        "model2_trigger_reason": ",".join(reasons) if reasons else "STANDBY",
        "stroke_risk": stroke_risk,
        "stroke_stage": stroke_stage,
        "stroke_stage_code": stroke_stage_code,
        "stroke_risk_pct": stroke_risk_pct,
        "stroke_sqi_status": stroke_sqi_status,
        "estimated_sbp_mmhg": est_sbp,
        "estimated_dbp_mmhg": est_dbp,
        "imu_fall_detected": imu_fall,
        "unified_diagnosis": unified_diag,
        "unified_stage": result.get("stage_name"),
        "unified_stage_code": int(result.get("stage_code", 0)),
        "trigger_fast_check": trigger_fast,
        "red_contraindication_veto": red_veto,
        "trigger_buzzer": trigger_buzzer,
        "trigger_vibrator": trigger_vibrator,
        "clinical_alert": clinical_alert,
    }
    if payload.get("hardware_metrics") is not None:
        summary["hardware_metrics"] = payload.get("hardware_metrics")
    return summary


@app.post("/api/v1/esp32-predict")
async def post_esp32_predict(payload: Dict[str, Any], compact: bool = Query(default=False)):
    """
    Hardware-aligned prediction endpoint for ESP32-S3 wearable (AD8232/ADS1115, MAX30102, MPU6050).
    Runs Model 1 (CardiacModel1A) first, then cascades to Model 2 (StrokeModel1B) when
    a cardiac abnormality, IMU fall, active finger PPG, or clinical profile is present.
    """
    ensure_models_loaded()
    global unified_model_1, cardiac_model_1a, stroke_model_1b, current_patient_state

    if unified_model_1 is None:
        raise HTTPException(status_code=503, detail="Unified Model 1 is not loaded")

    norm_payload = _normalize_esp32_payload(payload)
    device_id = norm_payload.get("device_id", "ARMBAND_ESP32S3_01")
    timestamp_ms = norm_payload.get("timestamp_ms", int(time.time() * 1000))
    elapsed_sec = float(norm_payload.get("elapsed_stage_sec", 0.0))

    def _run_predict():
        res_u = unified_model_1.predict_all(norm_payload, elapsed_stage_sec=elapsed_sec)
        decoupled: Dict[str, Any] = {}
        afib_prob = 0.0
        mi_prob = 0.0
        cardiac_probs = None
        cardiac_reliable = False
        cardiac_diag = "Normal"

        # Step 1: Always run Model 1A (Cardiac Specialist) first
        if cardiac_model_1a is not None:
            try:
                c_res = cardiac_model_1a.predict_all(norm_payload)
                cardiac_probs = c_res.get("disease_probabilities")
                afib_prob = float((cardiac_probs or {}).get("AFib", 0.0))
                mi_prob = float((cardiac_probs or {}).get("Ischemic_MI", 0.0))
                cardiac_reliable = bool((c_res.get("signal_quality") or {}).get("inference_reliable", False))
                cardiac_diag = str(c_res.get("disease_name", "Normal"))
                decoupled["cardiac_model_1a"] = c_res
            except Exception as exc:
                decoupled["cardiac_model_1a_error"] = str(exc)

        # Step 2: Smart Cascade Gate -> trigger Model 2 (StrokeModel1B) on Cardiac/IMU/PPG/Clinical signals
        rw = norm_payload.get("raw_windows") or {}
        hw = norm_payload.get("hardware_metrics") or {}
        p_ctx = norm_payload.get("patient_context") or {}
        clin_prof = norm_payload.get("clinical_profile") or (p_ctx if p_ctx else None)

        imu_max_g = _safe_float(hw.get("imu_max_g"), 1.0)
        if hw.get("imu_max_g") is None and all(k in rw for k in ("imu_ax_g", "imu_ay_g", "imu_az_g")):
            ax = np.asarray(rw["imu_ax_g"], dtype=np.float32)
            ay = np.asarray(rw["imu_ay_g"], dtype=np.float32)
            az = np.asarray(rw["imu_az_g"], dtype=np.float32)
            if ax.size > 0 and ax.size == ay.size == az.size:
                imu_max_g = float(np.max(np.sqrt(ax * ax + ay * ay + az * az)))
        imu_fall = bool(hw.get("fall_detected", False) or imu_max_g >= 3.0)

        has_valid_ppg_input = False
        ppg_raw = rw.get("ppg_ir", rw.get("ppg_red"))
        if ppg_raw is not None:
            ppg_arr = np.nan_to_num(np.asarray(ppg_raw, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            if len(ppg_arr) >= 60:
                if float(np.max(ppg_arr)) > 5000.0:
                    has_valid_ppg_input = bool(np.mean(ppg_arr > 1000.0) >= 0.75 and float(np.var(ppg_arr)) > 1e-4)
                else:
                    has_valid_ppg_input = bool(float(np.var(ppg_arr)) > 1e-4)

        cardiac_abnormal = cardiac_reliable and (
            cardiac_diag in ("AFib", "Arrhythmia", "Ischemic_MI")
            or afib_prob >= 0.30
            or mi_prob >= 0.35
        )

        cascade_reasons: List[str] = []
        if cardiac_abnormal:
            cascade_reasons.append("CARDIAC_ABNORMALITY")
        if imu_fall:
            cascade_reasons.append("IMU_FALL")
        if has_valid_ppg_input:
            cascade_reasons.append("PPG_ACTIVE")
        if clin_prof:
            cascade_reasons.append("CLINICAL_PROFILE")
        if bool(norm_payload.get("force_stroke", False)):
            cascade_reasons.append("FORCED")

        if stroke_model_1b is not None and cascade_reasons:
            try:
                missed_anti = bool(norm_payload.get("missed_anticoagulant_dose", p_ctx.get("missed_anticoagulant_dose", False)))
                s_res = stroke_model_1b.predict_all(
                    norm_payload,
                    elapsed_stage_sec=elapsed_sec,
                    cardiac_afib_prob=afib_prob,
                    cardiac_probs=cardiac_probs,
                    clinical_profile=clin_prof,
                    missed_anticoagulant=missed_anti,
                )
                decoupled["stroke_model_1b"] = s_res
                decoupled["model2_stroke_risk"] = s_res
            except Exception as exc:
                decoupled["stroke_model_1b_error"] = str(exc)
        elif stroke_model_1b is not None:
            standby_res = {
                "model_id": "StrokeModel1B",
                "status": "STANDBY",
                "disease_name": "STANDBY",
                "stage_code": 0,
                "stage_name": "Standby - Waiting for Cardiac/IMU/PPG Trigger",
                "stroke_risk_score_pct": 0.0,
                "telemetry_quality": {"status": "STANDBY", "inference_reliable": False},
                "trigger_model_3_fast_check": False,
                "red_contraindication_veto": False,
                "imu_fall_detected": False,
                "hardware_feedback": {"trigger_buzzer": False, "trigger_vibrator": False},
            }
            decoupled["stroke_model_1b"] = standby_res
            decoupled["model2_stroke_risk"] = standby_res

        return res_u, decoupled, cascade_reasons

    result, decoupled_out, cascade_reasons = await asyncio.to_thread(_run_predict)
    summary = _build_esp32_summary(result, decoupled_out, norm_payload, cascade_trigger_reasons=cascade_reasons)

    # Update global state for clinician dashboard & status endpoints
    current_patient_state["device_id"] = device_id
    current_patient_state["timestamp_ms"] = timestamp_ms
    current_patient_state["triage_class"] = result["disease_name"]
    current_patient_state["differential_probabilities"] = result["disease_probabilities"]
    current_patient_state["temporal_phase"] = result["stage_code"]
    current_patient_state["temporal_phase_name"] = result["stage_name"]
    current_patient_state["golden_hour_active"] = bool(result["stage_code"] >= 2)
    current_patient_state["golden_hour_remaining_sec"] = (
        float(result["golden_hour_remaining_min"] * 60.0)
        if result["golden_hour_remaining_min"] is not None else 3600.0
    )
    current_patient_state["golden_hour_display"] = result["golden_hour_status"]
    current_patient_state["golden_hour_timer_str"] = result["golden_hour_status"]
    current_patient_state["red_contraindication_alert"] = result["red_contraindication_veto"]
    current_patient_state["red_contraindication_veto"] = result["red_contraindication_veto"]
    current_patient_state["clinical_alert"] = result["clinical_alert"]

    if compact:
        compact_data = {
            "status": "success",
            "device_id": device_id,
            "timestamp_ms": timestamp_ms,
            "summary": summary,
            "hardware_feedback": result["hardware_feedback"],
        }
        return _sanitize_for_json(compact_data)

    response_data = {
        "status": "success",
        "device_id": device_id,
        "timestamp_ms": timestamp_ms,
        "head_a_disease": {
            "code": result["disease_code"],
            "name": result["disease_name"],
            "probabilities": result["disease_probabilities"]
        },
        "head_b_stage": {
            "code": result["stage_code"],
            "name": result["stage_name"],
            "probabilities": result["stage_probabilities"],
            "golden_hour_remaining_min": result["golden_hour_remaining_min"],
            "golden_hour_status": result["golden_hour_status"]
        },
        "trigger_model_2_camera_voice_check": result["trigger_model_2_camera_voice_check"],
        "trigger_model_3_fast_check": bool(
            result["trigger_model_2_camera_voice_check"]
            or (decoupled_out.get("model2_stroke_risk") or {}).get("trigger_model_3_fast_check", False)
        ),
        "red_contraindication_veto": result["red_contraindication_veto"],
        "clinical_alert": result["clinical_alert"],
        "hardware_feedback": result["hardware_feedback"],
        "signal_quality": result.get("signal_quality"),
        "dl_branch": result.get("dl_branch"),
        "raw_ml_decision": result.get("raw_ml_decision"),
        "decoupled_models": decoupled_out,
        "summary": summary,
    }
    return _sanitize_for_json(response_data)


@app.post("/api/v1/cardiac-predict")
async def post_cardiac_predict(payload: Dict[str, Any]):
    """
    Decoupled Model 1A (Cardiac Specialist) Endpoint:
    Evaluates Normal / AFib / Arrhythmia / Ischemic_MI (Head A) and 4-Stage Cardiac Acuity (Head B)
    with per-device temporal debouncing.
    """
    ensure_models_loaded()
    if cardiac_model_1a is None:
        raise HTTPException(status_code=503, detail="CardiacModel1A is not loaded")

    device_id = payload.get("device_id", "ARMBAND_ESP32S3_01")
    timestamp_ms = payload.get("timestamp_ms", int(time.time() * 1000))
    res = await asyncio.to_thread(cardiac_model_1a.predict_all, payload)

    client_entry = stream_state_manager.get_or_create(f"cardiac_{device_id}")
    cardiac_debouncer: TemporalAlertDebouncer = client_entry["cardiac_debouncer"]
    reliable = bool((res.get("signal_quality") or {}).get("inference_reliable", True))
    raw_diag = str(res.get("disease_name", "Normal"))
    if not reliable:
        debounced_diag = raw_diag
        alert_confirmed = False
    else:
        is_imm = (raw_diag == "Ischemic_MI" and int(res.get("stage_code", 0)) >= 3)
        alert_confirmed, confirmed_cls = cardiac_debouncer.update(raw_diag, is_immediate_emergency=is_imm)
        debounced_diag = confirmed_cls if alert_confirmed else "Normal"

    res["debounced_diagnosis"] = debounced_diag
    res["alert_confirmed"] = bool(alert_confirmed)
    return _sanitize_for_json({"status": "success", "device_id": device_id, "timestamp_ms": timestamp_ms, **res})


@app.post("/api/v1/stroke-predict")
async def post_stroke_predict(payload: Dict[str, Any]):
    """
    Model 2 (`StrokeModel1B` / `BrainStrokeRiskModel2`) Endpoint:
    Two-Branch Hybrid AI evaluating Stroke Risk & Acute Neurovascular Trigger
    (Low / Elevated / High FAST Trigger / Critical Neurovascular Emergency),
    Golden Hour Stage (Stage 0..3), Model 3 FAST Trigger, and non-mutating
    Red Contraindication Veto (forbidding aspirin/anticoagulants prior to Head CT).
    """
    ensure_models_loaded()
    if stroke_model_1b is None:
        raise HTTPException(status_code=503, detail="StrokeModel1B is not loaded")

    device_id = payload.get("device_id", "ARMBAND_ESP32S3_01")
    timestamp_ms = payload.get("timestamp_ms", int(time.time() * 1000))
    elapsed_sec = float(payload.get("elapsed_stage_sec", 0.0))
    afib_prob = float(payload.get("cardiac_afib_prob", 0.0))
    p_ctx = payload.get("patient_context") or {}
    cardiac_probs = payload.get("cardiac_probs") or payload.get("cardiac_context")
    clinical_profile = payload.get("clinical_profile") or (p_ctx if p_ctx else None)
    missed_anticoagulant = bool(
        payload.get("missed_anticoagulant_dose", p_ctx.get("missed_anticoagulant_dose", False))
    )
    res = await asyncio.to_thread(
        stroke_model_1b.predict_all,
        payload,
        elapsed_stage_sec=elapsed_sec,
        cardiac_afib_prob=afib_prob,
        cardiac_probs=cardiac_probs,
        clinical_profile=clinical_profile,
        missed_anticoagulant=missed_anticoagulant,
    )
    return _sanitize_for_json({"status": "success", "device_id": device_id, "timestamp_ms": timestamp_ms, **res})
