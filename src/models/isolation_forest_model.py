"""
Isolation Forest — unsupervised anomaly detection.

Key design decisions
--------------------
- Trained on LEGITIMATE transactions only — learns "normal" behaviour
- contamination = fraud rate in training data (~0.17%)
- anomaly score converted to fraud probability via sigmoid scaling
- This is the only UNSUPERVISED model — powerful for detecting novel fraud patterns
  that supervised models haven't seen before
"""
import numpy as np
import joblib
from loguru import logger
from sklearn.ensemble import IsolationForest

from src.models.base import BaseModel
from src.config import DATA_PROCESSED_DIR, RANDOM_STATE


class IsolationForestFraudModel(BaseModel):

    name = "isolation_forest"

    def __init__(
        self,
        n_estimators: int  = 200,
        contamination: float = 0.001727,  # actual fraud rate in dataset
        max_samples: str   = "auto",
        max_features: float = 1.0,
    ):
        self.params = dict(
            n_estimators  = n_estimators,
            contamination = contamination,
            max_samples   = max_samples,
            max_features  = max_features,
            random_state  = RANDOM_STATE,
            n_jobs        = -1,
        )
        self.model: IsolationForest = None
        self._score_min: float = None
        self._score_max: float = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train on LEGITIMATE transactions only.
        The model learns what 'normal' looks like —
        anomalies (fraud) will have high reconstruction error.
        """
        logger.info(f"Training {self.name} on legitimate transactions only …")
        X_legit = X_train[y_train == 0]
        logger.info(f"  Legitimate samples for training: {len(X_legit):,}")

        self.model = IsolationForest(**self.params)
        self.model.fit(X_legit)

        # Calibrate score range for probability conversion
        raw_scores = self.model.score_samples(X_train)
        self._score_min = raw_scores.min()
        self._score_max = raw_scores.max()

        logger.success(f"{self.name} training complete ✓")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return 1 (fraud) where Isolation Forest predicts -1 (anomaly)."""
        raw = self.model.predict(X)
        return (raw == -1).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Convert anomaly score to [0, 1] fraud probability.
        Isolation Forest scores are negative — more negative = more anomalous.
        We invert and normalize so high score = high fraud probability.
        """
        raw_scores = self.model.score_samples(X)

        # Invert: lower (more anomalous) score → higher fraud probability
        inverted = -raw_scores

        # Normalize to [0, 1]
        score_range = (-self._score_min) - (-self._score_max)
        if score_range == 0:
            return np.zeros(len(X))

        normalized = (inverted - (-self._score_max)) / score_range
        return np.clip(normalized, 0, 1)

    def save(self, path=None) -> None:
        path = path or DATA_PROCESSED_DIR / "isolation_forest_model.pkl"
        joblib.dump(self, path)
        logger.info(f"Model saved → {path}")

    @classmethod
    def load(cls, path=None) -> "IsolationForestFraudModel":
        path = path or DATA_PROCESSED_DIR / "isolation_forest_model.pkl"
        return joblib.load(path)

    def get_params(self) -> dict:
        return self.params
