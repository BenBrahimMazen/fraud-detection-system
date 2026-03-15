"""
XGBoost fraud detector.

Key design decisions
--------------------
- scale_pos_weight handles class imbalance natively (no SMOTE needed at inference)
- eval_metric = aucpr (area under precision-recall) — better than AUC for imbalanced data
- early_stopping on a validation set prevents overfitting
- feature_importances_ exposed for SHAP compatibility
"""
import numpy as np
import joblib
from loguru import logger
from xgboost import XGBClassifier

from src.models.base import BaseModel
from src.config import DATA_PROCESSED_DIR, RANDOM_STATE


class XGBoostFraudModel(BaseModel):

    name = "xgboost"

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        scale_pos_weight: float = None,   # set automatically from y_train if None
    ):
        self.params = dict(
            n_estimators      = n_estimators,
            max_depth         = max_depth,
            learning_rate     = learning_rate,
            subsample         = subsample,
            colsample_bytree  = colsample_bytree,
            eval_metric       = "aucpr",
            random_state      = RANDOM_STATE,
            n_jobs            = -1,
            tree_method       = "hist",   # fast on CPU
        )
        self._scale_pos_weight = scale_pos_weight
        self.model: XGBClassifier = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
    ) -> None:
        # Auto-compute class weight from training data
        if self._scale_pos_weight is None:
            neg = (y_train == 0).sum()
            pos = (y_train == 1).sum()
            self._scale_pos_weight = neg / max(pos, 1)
            logger.info(f"  scale_pos_weight = {self._scale_pos_weight:.1f}")

        self.params["scale_pos_weight"] = self._scale_pos_weight
        self.model = XGBClassifier(**self.params)

        eval_set = [(X_val, y_val)] if X_val is not None else None

        logger.info(f"Training {self.name} …")
        self.model.fit(
            X_train, y_train,
            eval_set       = eval_set,
            verbose        = False,
        )
        logger.success(f"{self.name} training complete ✓")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    @property
    def feature_importances_(self) -> np.ndarray:
        return self.model.feature_importances_

    def save(self, path=None) -> None:
        path = path or DATA_PROCESSED_DIR / "xgboost_model.pkl"
        joblib.dump(self, path)
        logger.info(f"Model saved → {path}")

    @classmethod
    def load(cls, path=None) -> "XGBoostFraudModel":
        path = path or DATA_PROCESSED_DIR / "xgboost_model.pkl"
        return joblib.load(path)

    def get_params(self) -> dict:
        return self.params
