"""
stage1_model.py
PyTorch End-to-End Hybrid Deep Learning & Tabular Architecture for Stage 1 Continuous Monitoring.
Fuses raw waveforms (1D-CNN + SE-Attention + BiLSTM) with tabular clinical/kinematic features (MLP).
"""

import os
from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
if not torch.cuda.is_available() and torch.get_num_threads() > 4:
    torch.set_num_threads(4)
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


class SqueezeExcitation1D(nn.Module):
    """1D Channel Attention (Squeeze-and-Excitation) Block."""
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        reduced = max(4, channels // reduction)
        self.fc1 = nn.Linear(channels, reduced, bias=False)
        self.fc2 = nn.Linear(reduced, channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, channels, length)
        b, c, _ = x.shape
        # Squeeze: Global Average Pooling across length
        squeezed = x.mean(dim=2)  # (batch, channels)
        # Excitation
        excited = F.relu(self.fc1(squeezed))
        excited = torch.sigmoid(self.fc2(excited)).unsqueeze(2)  # (batch, channels, 1)
        return x * excited


class Stage1HybridModel(nn.Module):
    """
    Hybrid Dual-Branch Neural Architecture:
      - Branch 1A (Raw Waveforms): 3 Conv1D+SE blocks + 2-layer BiLSTM + Attention Pooling -> 128-dim
      - Branch 1B (Tabular Features): 20 features -> 2-layer Dense MLP -> 64-dim
      - Fusion Head: 192-dim concatenated embedding -> Dense layers -> 5 clinical logits
    """
    def __init__(
        self,
        in_channels: int = 2,
        seq_len: int = 2500,
        num_tabular_features: int = 20,
        num_classes: int = 5
    ):
        super().__init__()
        self.in_channels = in_channels
        self.seq_len = seq_len
        self.num_classes = num_classes

        # --- Branch 1A: 1D-CNN Waveform Feature Extractor ---
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=15, stride=1, padding=7)
        self.bn1 = nn.BatchNorm1d(32)
        self.se1 = SqueezeExcitation1D(32)
        self.pool1 = nn.MaxPool1d(2)  # length -> 1250

        self.conv2 = nn.Conv1d(32, 64, kernel_size=11, stride=1, padding=5)
        self.bn2 = nn.BatchNorm1d(64)
        self.se2 = SqueezeExcitation1D(64)
        self.pool2 = nn.MaxPool1d(2)  # length -> 625

        self.conv3 = nn.Conv1d(64, 128, kernel_size=7, stride=1, padding=3)
        self.bn3 = nn.BatchNorm1d(128)
        self.se3 = SqueezeExcitation1D(128)
        self.pool3 = nn.MaxPool1d(2)  # length -> 312

        # 2-Layer Bidirectional LSTM
        self.bilstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )

        # Temporal Attention Pooling (BiLSTM output is 64*2 = 128 features)
        self.temporal_attn = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # --- Branch 1B: Tabular Features MLP ---
        self.tab_bn = nn.BatchNorm1d(num_tabular_features)
        self.tab_mlp = nn.Sequential(
            nn.Linear(num_tabular_features, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.GELU()
        )

        # --- Late Fusion & Classification Head ---
        # 128 (waveforms) + 64 (tabular) = 192
        self.fusion = nn.Sequential(
            nn.Linear(128 + 64, 96),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(96, num_classes)
        )

    def forward(self, waveforms: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        # 1. Waveform Branch
        # x: (batch, 2, 2500)
        x = self.pool1(self.se1(F.gelu(self.bn1(self.conv1(waveforms)))))
        x = self.pool2(self.se2(F.gelu(self.bn2(self.conv2(x)))))
        x = self.pool3(self.se3(F.gelu(self.bn3(self.conv3(x)))))

        # Permute to (batch, time_steps, channels) for LSTM
        x = x.permute(0, 2, 1)  # (batch, 312, 128)
        lstm_out, _ = self.bilstm(x)  # (batch, 312, 128)

        # Attention pooling over time
        attn_weights = F.softmax(self.temporal_attn(lstm_out), dim=1)  # (batch, 312, 1)
        wave_emb = torch.sum(lstm_out * attn_weights, dim=1)  # (batch, 128)

        # 2. Tabular Branch
        tab_norm = self.tab_bn(tabular)
        tab_emb = self.tab_mlp(tab_norm)  # (batch, 64)

        # 3. Concatenation & Classification Head
        fused = torch.cat([wave_emb, tab_emb], dim=1)  # (batch, 192)
        logits = self.fusion(fused)  # (batch, 5)
        return logits

    def predict_proba(self, waveforms: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        """Returns calibrated class probabilities via Softmax."""
        logits = self.forward(waveforms, tabular)
        return F.softmax(logits, dim=-1)


class FocalLoss(nn.Module):
    """Multi-class Focal Loss for addressing severe clinical class imbalances."""
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 15,
    lr: float = 1e-3,
    class_weights: Optional[torch.Tensor] = None,
    device: str = "cpu"
) -> Dict[str, Any]:
    """Trains the Stage 1 Hybrid model with AdamW and returns training metrics history."""
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    # Balanced class weights for unbiased multi-class learning
    if class_weights is None:
        class_weights = torch.ones(5, device=device)
    else:
        class_weights = class_weights.to(device)
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        for waves, tab, targets in train_loader:
            waves, tab, targets = waves.to(device), tab.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(waves, tab)
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item() * len(targets)

        scheduler.step()
        avg_train_loss = total_train_loss / len(train_loader.dataset)
        history["train_loss"].append(avg_train_loss)

        # Validation
        model.eval()
        total_val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for waves, tab, targets in val_loader:
                waves, tab, targets = waves.to(device), tab.to(device), targets.to(device)
                logits = model(waves, tab)
                loss = criterion(logits, targets)
                total_val_loss += loss.item() * len(targets)
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == targets).sum().item()

        avg_val_loss = total_val_loss / len(val_loader.dataset)
        val_acc = correct / len(val_loader.dataset)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)
        print(f"  Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc*100:.1f}%", flush=True)

    return history


def save_model(model: nn.Module, path: str):
    """Saves model weights."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(path: str, device: str = "cpu") -> Stage1HybridModel:
    """Loads and initializes Stage 1 model from checkpoint."""
    model = Stage1HybridModel()
    if os.path.exists(path):
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


class Stage1MLClassifier:
    """
    Stage 1 Machine Learning Classifier (XGBoost).
    Engineered for high clinical accuracy and robust generalization in low-data / small-dataset regimes.
    Operates directly on the 20 extracted tabular bio-signal, hemodynamic, and kinematic features.
    """
    def __init__(self, model_path: Optional[str] = None):
        from xgboost import XGBClassifier
        self.model = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.08,
            eval_metric='mlogloss',
            random_state=42
        )
        self.is_fitted = False
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fits the XGBoost classifier on tabular clinical features."""
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts discrete clinical class (0..4)."""
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicts calibrated class probability distribution across 5 classes."""
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if not self.is_fitted:
            n_samples = len(X)
            return np.full((n_samples, 5), 0.2, dtype=np.float32)
        return self.model.predict_proba(X)

    def save_model(self, path: str):
        """Saves JSON model checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save_model(path)

    def load_model(self, path: str):
        """Loads JSON model checkpoint."""
        if os.path.exists(path):
            self.model.load_model(path)
            self.is_fitted = True
        else:
            raise FileNotFoundError(f"Model file not found at: {path}")
