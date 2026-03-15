"""
Autoencoder Neural Network — fraud detection via reconstruction error.

How it works
------------
1. Train on LEGITIMATE transactions only
2. The network learns to compress and reconstruct normal patterns
3. At inference: fraud transactions have HIGH reconstruction error
   because the network has never learned to reconstruct them
4. Reconstruction error threshold → binary fraud prediction

Architecture
------------
Encoder: input → 64 → 32 → 16 → bottleneck(8)
Decoder: 8 → 16 → 32 → 64 → output
"""
import numpy as np
import joblib
from pathlib import Path
from loguru import logger

from src.models.base import BaseModel
from src.config import DATA_PROCESSED_DIR, RANDOM_STATE

# Graceful import — PyTorch is optional (falls back if not installed)
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not installed — AutoEncoder will be unavailable")


class _AutoencoderNet(nn.Module if TORCH_AVAILABLE else object):
    """The actual PyTorch network."""

    def __init__(self, input_dim: int):
        if not TORCH_AVAILABLE:
            return
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
        )

        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class AutoEncoderFraudModel(BaseModel):

    name = "autoencoder"

    def __init__(
        self,
        epochs: int      = 30,
        batch_size: int  = 256,
        learning_rate: float = 1e-3,
        threshold_percentile: float = 95,  # top 5% reconstruction errors = fraud
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("Install PyTorch: pip install torch")

        self.epochs               = epochs
        self.batch_size           = batch_size
        self.learning_rate        = learning_rate
        self.threshold_percentile = threshold_percentile
        self.net: _AutoencoderNet = None
        self._threshold: float    = None
        self._score_min: float    = None
        self._score_max: float    = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(RANDOM_STATE)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        logger.info(f"Training {self.name} on {self.device} …")
        X_legit = X_train[y_train == 0].astype(np.float32)
        logger.info(f"  Legitimate samples: {len(X_legit):,}")

        input_dim  = X_legit.shape[1]
        self.net   = _AutoencoderNet(input_dim).to(self.device)
        optimizer  = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        criterion  = nn.MSELoss()
        scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, factor=0.5, verbose=False
        )

        dataset    = TensorDataset(torch.tensor(X_legit))
        loader     = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.net.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(self.device)
                recon = self.net(batch)
                loss  = criterion(recon, batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(loader)
            scheduler.step(avg_loss)

            if (epoch + 1) % 5 == 0:
                logger.info(f"  Epoch {epoch+1:3d}/{self.epochs}  loss={avg_loss:.6f}")

        # Set threshold from reconstruction error on training data
        errors = self._reconstruction_errors(X_train.astype(np.float32))
        self._threshold  = np.percentile(errors, self.threshold_percentile)
        self._score_min  = errors.min()
        self._score_max  = errors.max()
        logger.info(f"  Threshold (p{self.threshold_percentile}): {self._threshold:.6f}")
        logger.success(f"{self.name} training complete ✓")

    def predict(self, X: np.ndarray) -> np.ndarray:
        errors = self._reconstruction_errors(X.astype(np.float32))
        return (errors >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        errors = self._reconstruction_errors(X.astype(np.float32))
        score_range = self._score_max - self._score_min + 1e-8
        return np.clip((errors - self._score_min) / score_range, 0, 1)

    def _reconstruction_errors(self, X: np.ndarray) -> np.ndarray:
        self.net.eval()
        errors = []
        with torch.no_grad():
            loader = DataLoader(
                TensorDataset(torch.tensor(X)),
                batch_size=512,
                shuffle=False,
            )
            for (batch,) in loader:
                batch = batch.to(self.device)
                recon = self.net(batch)
                mse   = ((recon - batch) ** 2).mean(dim=1)
                errors.append(mse.cpu().numpy())
        return np.concatenate(errors)

    def save(self, path=None) -> None:
        path = path or DATA_PROCESSED_DIR / "autoencoder_model.pkl"
        joblib.dump(self, path)
        logger.info(f"Model saved → {path}")

    @classmethod
    def load(cls, path=None) -> "AutoEncoderFraudModel":
        path = path or DATA_PROCESSED_DIR / "autoencoder_model.pkl"
        return joblib.load(path)

    def get_params(self) -> dict:
        return {
            "epochs":               self.epochs,
            "batch_size":           self.batch_size,
            "learning_rate":        self.learning_rate,
            "threshold_percentile": self.threshold_percentile,
        }
