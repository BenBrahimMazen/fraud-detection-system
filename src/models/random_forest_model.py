"""
Random Forest fraud detector.

Key design decisions
--------------------
- class_weight='balanced_subsample' — each tree gets a balanced bootstrap sample
- max_features='sqrt' — standard for classification, reduces correlation between trees
- min_samples_leaf=2 — prevents overfitting to single-sample leaves
- n_jobs=-1 — full parallelism
"""
import numpy as np
import joblib
from loguru import logger
from sklearn.ensemble import RandomForestClassifier

from src.models.base import BaseModel
from src.config import DATA_PROCESSED_DIR, RANDOM_STATE


class RandomForestFraudModel(BaseModel):

    name = "random_forest"

    def __init__(
        self,
        n_estimators: int   = 300,
        max_depth: int      = None,
        max_features: str   = "sqrt",
        min_samples_leaf: int = 2,
        class_weight: str   = "balanced_subsample",
    ):
        self.params = dict(
            n_estimators      = n_estimators,
            max_depth         = max_depth,
            max_features      = max_features,
            min_samples_leaf  = min_samples_leaf,
            class_weight      = class_weight,
            random_state      = RANDOM_STATE,
            n_jobs            = -1,
        )
        self.model: RandomForestClassifier = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        logger.info(f"Training {self.name} …")
        self.model = RandomForestClassifier(**self.params)
        self.model.fit(X_train, y_train)
        logger.success(f"{self.name} training complete ✓")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    @property
    def feature_importances_(self) -> np.ndarray:
        return self.model.feature_importances_

    def save(self, path=None) -> None:
        path = path or DATA_PROCESSED_DIR / "random_forest_model.pkl"
        joblib.dump(self, path)
        logger.info(f"Model saved → {path}")

    @classmethod
    def load(cls, path=None) -> "RandomForestFraudModel":
        path = path or DATA_PROCESSED_DIR / "random_forest_model.pkl"
        return joblib.load(path)

    def get_params(self) -> dict:
        return self.params
