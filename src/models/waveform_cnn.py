"""
waveform_cnn.py
========================================================================================
Domain-Invariant PyTorch 1D-CNN + BiLSTM + Temporal Attention Waveform Branch
for Unified Heart-Brain Model 1.

Solves cross-hospital domain shift and motion-noise sensitivity:
  1. Applies uniform 1.0 - 35.0 Hz zero-phase SOS bandpass filtering across devices.
  2. Extracts two device-agnostic temporal channels per 10s window (2, 2500):
       - Ch0: Per-window Z-scored bandpass ECG (gain & offset invariant).
       - Ch1: Smoothed QRS slope-energy envelope (lead-orientation & rhythm invariant).
  3. Uses GroupNorm instead of BatchNorm so inference normalization does not drift
     when moving from 12-lead hospital ECGs (PTB-XL) to wearable/Holter hardware.
  4. Trains with polarity flips, hardware gain randomization, Holter-band EMG bursts,
     baseline motion wander, and random temporal phase shifts.
========================================================================================
"""

import os
import sys
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import butter, resample, sosfiltfilt
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

if not torch.cuda.is_available() and torch.get_num_threads() > 4:
    torch.set_num_threads(4)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

TARGET_FS = 250.0
TARGET_SAMPLES = 2500
DEFAULT_CNN_WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "models", "unified_model_1_cnn.pt")

_SOS_BANDPASS = butter(4, [1.0, 35.0], btype="bandpass", fs=TARGET_FS, output="sos")


def _resample_and_normalize_length(waves: np.ndarray, fs: float = TARGET_FS, target_len: int = TARGET_SAMPLES) -> np.ndarray:
    """Resamples 2D ECG batch (N, T) to 250 Hz and pads/crops cleanly to target_len."""
    arr = np.asarray(waves, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    if not np.isclose(fs, TARGET_FS) and arr.shape[1] > 16:
        new_len = max(16, int(round(arr.shape[1] * (TARGET_FS / float(fs)))))
        arr = resample(arr, new_len, axis=1)

    n_samples = arr.shape[1]
    if n_samples < target_len:
        pad_width = target_len - n_samples
        arr = np.pad(arr, ((0, 0), (0, pad_width)), mode="edge")
    elif n_samples > target_len:
        arr = arr[:, :target_len]
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _envelope_from_normalized(z_waves: np.ndarray, kernel_size: int = 15) -> np.ndarray:
    """Computes normalized smoothed QRS slope-energy envelope for a (N, T) batch."""
    diff = np.abs(np.diff(z_waves, axis=1, prepend=z_waves[:, :1]))
    pad_l = kernel_size // 2
    pad_r = kernel_size - pad_l
    cs = np.cumsum(np.pad(diff, ((0, 0), (pad_l, pad_r)), mode="edge"), axis=1)
    env = (cs[:, kernel_size:] - cs[:, :-kernel_size]) / float(kernel_size)
    std_e = env.std(axis=1, keepdims=True)
    env = np.where(std_e > 1e-6, (env - env.mean(axis=1, keepdims=True)) / (std_e + 1e-6), 0.0)
    return np.clip(env, -6.0, 6.0).astype(np.float32)


def build_ecg_cnn_channels(waves: np.ndarray, fs: float = TARGET_FS) -> np.ndarray:
    """
    Converts raw 10s single-lead ECG waveforms into invariant (N, 2, 2500) float32 tensors:
      Channel 0: 1-35 Hz bandpass + per-window Z-scored ECG.
      Channel 1: Smoothed QRS slope-energy envelope.
    """
    arr = _resample_and_normalize_length(waves, fs=fs, target_len=TARGET_SAMPLES)
    # Identify flatline windows before sosfiltfilt edge-ringing
    stds_raw = np.std(arr, axis=1, keepdims=True)
    valid = (stds_raw[:, 0] > 1e-6)

    bp = np.zeros_like(arr, dtype=np.float32)
    if np.any(valid):
        bp_valid = sosfiltfilt(_SOS_BANDPASS, arr[valid], axis=1).astype(np.float32)
        sd_v = bp_valid.std(axis=1, keepdims=True)
        bp_valid = np.where(
            sd_v > 1e-6,
            np.clip((bp_valid - bp_valid.mean(axis=1, keepdims=True)) / (sd_v + 1e-6), -8.0, 8.0),
            0.0,
        )
        bp[valid] = bp_valid.astype(np.float32)

    env = _envelope_from_normalized(bp)
    return np.stack([bp, env], axis=1).astype(np.float32)


class HolterDomainAugDataset(Dataset):
    """Real waveform dataset applying wearable motion/EMG/polarity/gain augmentation during training."""

    def __init__(self, clean_bandpass_waves: np.ndarray, labels: np.ndarray):
        self.clean = np.asarray(clean_bandpass_waves, dtype=np.float32)
        self.y = np.asarray(labels, dtype=np.int64)
        self.t = np.linspace(0.0, 10.0, self.clean.shape[1], endpoint=False, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        w = self.clean[idx].copy()
        if np.random.rand() < 0.5:
            w = -w
        w = w * np.random.uniform(0.5, 2.0)

        # Simulated motion/respiration drift + wearable EMG muscle interference
        if np.random.rand() < 0.8:
            amp = np.random.uniform(0.05, 0.35) * (w.std() + 1e-6)
            freq = np.random.uniform(0.8, 2.5)
            phase = np.random.uniform(0.0, 2.0 * np.pi)
            w = w + amp * np.sin(2.0 * np.pi * freq * self.t + phase)

        if np.random.rand() < 0.85:
            emg = np.convolve(np.random.randn(len(w)).astype(np.float32), np.ones(3, dtype=np.float32) / 3.0, mode="same")
            w = w + emg * np.random.uniform(0.04, 0.25) * (w.std() + 1e-6)

        w = np.roll(w, int(np.random.randint(-125, 125)))
        sd = w.std()
        w = np.clip((w - w.mean()) / (sd + 1e-6), -8.0, 8.0) if sd > 1e-6 else np.zeros_like(w)
        env = _envelope_from_normalized(w[None, :])[0]
        x2 = np.stack([w, env], axis=0).astype(np.float32)
        return torch.tensor(x2, dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)


class InvariantECGCNNBiLSTM(nn.Module):
    """
    GroupNorm 1D-CNN + Bidirectional LSTM + Attention Pooling on (2, 2500) ECG waveforms.
    Downsamples 2500 samples -> 79 temporal steps (~125 ms resolution) so BiLSTM captures
    beat-to-beat R-R regularity alongside local P/QRS morphology.
    """

    def __init__(self, in_channels: int = 2, num_classes: int = 2, dropout: float = 0.35):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 24, kernel_size=11, stride=2, padding=5),
            nn.GroupNorm(4, 24),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(24, 48, kernel_size=7, stride=2, padding=3),
            nn.GroupNorm(4, 48),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(48, 64, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )
        self.lstm = nn.LSTM(64, 48, num_layers=1, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(96, 1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(96, 32),
            nn.GELU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x).permute(0, 2, 1)  # (B, 79, 64)
        out, _ = self.lstm(h)              # (B, 79, 96)
        weights = F.softmax(self.attn(out), dim=1)
        emb = torch.sum(out * weights, dim=1)
        return self.head(emb)


class WaveformDeepLearningEngine:
    """Thread-safe inference wrapper around InvariantECGCNNBiLSTM for raw ESP32/hospital ECG."""

    def __init__(self, model: InvariantECGCNNBiLSTM, calibrated_threshold: float = 0.50, metadata: Optional[Dict[str, Any]] = None):
        self.model = model
        self.model.eval()
        self.calibrated_threshold = float(calibrated_threshold)
        self.metadata = metadata or {}

    @classmethod
    def load(cls, weights_path: str = DEFAULT_CNN_WEIGHTS_PATH) -> Optional["WaveformDeepLearningEngine"]:
        if not os.path.exists(weights_path):
            return None
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=True)
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        meta = ckpt.get("metadata", {}) if isinstance(ckpt, dict) else {}
        thresh = float(ckpt.get("calibrated_threshold", 0.50)) if isinstance(ckpt, dict) else 0.50

        model = InvariantECGCNNBiLSTM(in_channels=2, num_classes=2)
        model.load_state_dict(state)
        model.eval()
        return cls(model=model, calibrated_threshold=thresh, metadata=meta)

    def predict_window_proba(self, ecg_mv: np.ndarray, fs: float = TARGET_FS) -> Dict[str, Any]:
        """Runs 1D-CNN+BiLSTM inference on one 10s ECG window and returns rhythm probabilities."""
        arr = np.asarray(ecg_mv, dtype=np.float32).flatten()
        if len(arr) < 250 or np.var(arr) < 1e-5:
            return {
                "active": False,
                "architecture": "1D-CNN-BiLSTM-Attention",
                "reason": "insufficient_or_flatline_ecg",
                "p_normal_sinus_wave": 0.5,
                "p_afib_wave": 0.5,
            }

        x_np = build_ecg_cnn_channels(arr, fs=fs)
        x_tensor = torch.from_numpy(x_np)
        with torch.no_grad():
            logits = self.model(x_tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        p_norm = float(probs[0])
        p_afib = float(probs[1])
        return {
            "active": True,
            "architecture": "1D-CNN-BiLSTM-Attention",
            "p_normal_sinus_wave": round(p_norm, 4),
            "p_afib_wave": round(p_afib, 4),
            "calibrated_threshold": round(self.calibrated_threshold, 3),
            "wave_afib_detected": bool(p_afib >= self.calibrated_threshold),
        }


def train_and_save_waveform_cnn(
    data_dir: str = os.path.join(PROJECT_ROOT, "data", "ecg_only"),
    weights_path: str = DEFAULT_CNN_WEIGHTS_PATH,
    epochs: int = 35,
    batch_size: int = 32,
    lr: float = 1.2e-3,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Trains InvariantECGCNNBiLSTM exclusively on real 250 Hz PTB-XL training waveforms,
    calibrates threshold on PTB-XL validation, and evaluates zero-shot on PTB-XL Internal Test
    and completely unseen external Holter MIT-AFDB.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    tr = np.load(os.path.join(data_dir, "ecg_train_dataset.npz"))
    va = np.load(os.path.join(data_dir, "ecg_val_dataset.npz"))
    it = np.load(os.path.join(data_dir, "ecg_test_internal_dataset.npz"))
    ex = np.load(os.path.join(data_dir, "ecg_test_external_afdb.npz"))

    tr_clean = sosfiltfilt(_SOS_BANDPASS, tr["X_waveforms"].astype(np.float64), axis=1).astype(np.float32)
    x_va = torch.from_numpy(build_ecg_cnn_channels(va["X_waveforms"]))
    x_it = torch.from_numpy(build_ecg_cnn_channels(it["X_waveforms"]))
    x_ex = torch.from_numpy(build_ecg_cnn_channels(ex["X_waveforms"]))
    y_va_t = torch.tensor(va["y"], dtype=torch.long)

    dl = DataLoader(HolterDomainAugDataset(tr_clean, tr["y"]), batch_size=batch_size, shuffle=True)
    model = InvariantECGCNNBiLSTM(in_channels=2, num_classes=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-3)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_loss = float("inf")
    best_state = None
    for _ in range(epochs):
        model.train()
        for xb, yb in dl:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(x_va), y_va_t).item())
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        p_va = torch.softmax(model(x_va), dim=1)[:, 1].numpy()
        p_it = torch.softmax(model(x_it), dim=1)[:, 1].numpy()
        p_ex = torch.softmax(model(x_ex), dim=1)[:, 1].numpy()

    # Select conservative threshold strictly on Val (highest threshold achieving optimal Val BalAcc)
    grid = np.linspace(0.45, 0.85, 41)
    best_bal = max(balanced_accuracy_score(va["y"], p_va >= t) for t in grid)
    candidates = [t for t in grid if balanced_accuracy_score(va["y"], p_va >= t) >= best_bal - 1e-4]
    cal_thresh = float(max(candidates))

    metrics = {
        "calibrated_threshold": round(cal_thresh, 3),
        "val_auc": round(float(roc_auc_score(va["y"], p_va)), 4),
        "val_acc": round(float(accuracy_score(va["y"], p_va >= cal_thresh)), 4),
        "val_f1": round(float(f1_score(va["y"], p_va >= cal_thresh)), 4),
        "internal_test_auc": round(float(roc_auc_score(it["y"], p_it)), 4),
        "internal_test_acc": round(float(accuracy_score(it["y"], p_it >= cal_thresh)), 4),
        "internal_test_f1": round(float(f1_score(it["y"], p_it >= cal_thresh)), 4),
        "external_afdb_auc": round(float(roc_auc_score(ex["y"], p_ex)), 4),
        "external_afdb_acc": round(float(accuracy_score(ex["y"], p_ex >= cal_thresh)), 4),
        "external_afdb_bal_acc": round(float(balanced_accuracy_score(ex["y"], p_ex >= cal_thresh)), 4),
    }

    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    torch.save(
        {
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "calibrated_threshold": cal_thresh,
            "metadata": metrics,
        },
        weights_path,
    )
    print(f"Saved Domain-Invariant 1D-CNN+BiLSTM weights to: {weights_path}")
    print(f"  PTB-XL Val:        AUC={metrics['val_auc']:.4f} | Acc={metrics['val_acc']*100:.2f}% | F1={metrics['val_f1']:.4f} (th={cal_thresh:.2f})")
    print(f"  PTB-XL Int Test:   AUC={metrics['internal_test_auc']:.4f} | Acc={metrics['internal_test_acc']*100:.2f}% | F1={metrics['internal_test_f1']:.4f}")
    print(f"  UNSEEN MIT-AFDB:   AUC={metrics['external_afdb_auc']:.4f} | Acc={metrics['external_afdb_acc']*100:.2f}% | BalAcc={metrics['external_afdb_bal_acc']*100:.2f}%")
    return metrics


if __name__ == "__main__":
    train_and_save_waveform_cnn()
