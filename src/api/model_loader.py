"""
Model loader — loads all artifacts once at API startup.
Uses a singleton pattern so models are loaded into memory only once.
"""
from pathlib import Path
from typing import Optional, List
import numpy as np
import joblib
from loguru import logger

from src.config import DATA_PROCESSED_DIR, FRAUD_THRESHOLD


class ModelRegistry:
    """
    Singleton that holds all loaded artifacts.
    Loaded once at FastAPI startup, reused for every request.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def initialize(self) -> None:
        if self._initialized:
            return

        logger.info("Loading model artifacts …")

        # Load best model (try ensemble first, fall back to xgboost)
        self.model = self._load_model()
        self.scaler = self._load_scaler()
        self.explainer = self._load_explainer()
        self.feature_names = self._load_feature_names()
        self.threshold = FRAUD_THRESHOLD
        self._initialized = True

        logger.success(f"Model registry initialized ✓")
        logger.info(f"  Model: {self.model.name}")
        logger.info(f"  Features: {len(self.feature_names)}")
        logger.info(f"  Threshold: {self.threshold}")

    def _load_model(self):
        # Try ensemble first, then xgboost
        for name in ["ensemble_model", "xgboost_model", "random_forest_model"]:
            path = DATA_PROCESSED_DIR / f"{name}.pkl"
            if path.exists():
                model = joblib.load(path)
                logger.info(f"Loaded model: {name}")
                return model
        raise FileNotFoundError(
            "No trained model found. Run: python src/models/trainer.py"
        )

    def _load_scaler(self):
        path = DATA_PROCESSED_DIR / "scaler.pkl"
        if path.exists():
            logger.info("Loaded scaler ✓")
            return joblib.load(path)
        logger.warning("Scaler not found — using raw features")
        return None

    def _load_explainer(self):
        path = DATA_PROCESSED_DIR / "shap_explainer.pkl"
        if path.exists():
            logger.info("Loaded SHAP explainer ✓")
            return joblib.load(path)
        logger.warning("SHAP explainer not found — /explain will be unavailable")
        return None

    def _load_feature_names(self) -> List[str]:
        path = DATA_PROCESSED_DIR / "feature_names.pkl"
        if path.exists():
            names = joblib.load(path)
            return names
        # Fallback to default feature order
        from src.config import PCA_FEATURES, RAW_FEATURES
        return PCA_FEATURES + RAW_FEATURES

    def preprocess_input(self, X: np.ndarray) -> np.ndarray:
        """Apply scaler to raw feature array."""
        if self.scaler is None:
            return X
        from src.config import PCA_FEATURES, RAW_FEATURES
        n_pca = len(PCA_FEATURES)
        n_raw = len(RAW_FEATURES)
        # Only scale the raw features (Time, Amount) — PCA already scaled
        X_scaled = X.copy()
        X_scaled[:, n_pca:n_pca + n_raw] = self.scaler.transform(
            X[:, n_pca:n_pca + n_raw]
        )
        return X_scaled

    @property
    def is_ready(self) -> bool:
        return self._initialized


# Global singleton instance
registry = ModelRegistry()
