"""
Stacking Ensemble — combines all base models with a meta-learner.

Architecture
------------
Level 0 (base models):  XGBoost, RandomForest, IsolationForest, AutoEncoder
Level 1 (meta-learner): LogisticRegression trained on out-of-fold predictions

This is the most powerful approach:
- Each base model sees the full feature space
- The meta-learner learns HOW to combine their outputs optimally
- Out-of-fold predictions prevent leakage into the meta-learner
"""
from typing import List, Optional
import numpy as np
import joblib
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from src.models.base import BaseModel
from src.config import DATA_PROCESSED_DIR, RANDOM_STATE, CV_FOLDS


class StackingEnsemble(BaseModel):

    name = "stacking_ensemble"

    def __init__(self, base_models: List[BaseModel]):
        self.base_models  = base_models
        self.meta_learner = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            max_iter=1000,
        )
        self._is_fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        logger.info(f"Training stacking ensemble with {len(self.base_models)} base models …")

        # ── Step 1: Train all base models on full training set ────
        for model in self.base_models:
            model.fit(X_train, y_train)

        # ── Step 2: Generate out-of-fold predictions for meta-learner ──
        oof_preds = self._generate_oof_predictions(X_train, y_train)

        # ── Step 3: Train meta-learner on OOF predictions ────────
        logger.info("Training meta-learner on out-of-fold predictions …")
        self.meta_learner.fit(oof_preds, y_train)
        self._is_fitted = True
        logger.success("Stacking ensemble training complete ✓")

    def _generate_oof_predictions(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """
        Generate out-of-fold predictions.
        Each fold: train base models on train split, predict on val split.
        This prevents the meta-learner from seeing training labels directly.
        """
        n_models = len(self.base_models)
        oof = np.zeros((len(X), n_models))
        skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            logger.info(f"  OOF fold {fold + 1}/{CV_FOLDS} …")
            X_fold_train, X_fold_val = X[train_idx], X[val_idx]
            y_fold_train             = y[train_idx]

            for i, model in enumerate(self.base_models):
                # Clone-fit on fold
                model.fit(X_fold_train, y_fold_train)
                oof[val_idx, i] = model.predict_proba(X_fold_val)

        return oof

    def predict(self, X: np.ndarray) -> np.ndarray:
        meta_input = self._get_meta_input(X)
        return self.meta_learner.predict(meta_input)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        meta_input = self._get_meta_input(X)
        return self.meta_learner.predict_proba(meta_input)[:, 1]

    def _get_meta_input(self, X: np.ndarray) -> np.ndarray:
        """Stack base model probabilities as input to meta-learner."""
        return np.column_stack([
            m.predict_proba(X) for m in self.base_models
        ])

    def get_model_weights(self) -> dict:
        """Return meta-learner coefficients — how much each model contributes."""
        coeffs = self.meta_learner.coef_[0]
        return {
            m.name: round(float(c), 4)
            for m, c in zip(self.base_models, coeffs)
        }

    def save(self, path=None) -> None:
        path = path or DATA_PROCESSED_DIR / "ensemble_model.pkl"
        joblib.dump(self, path)
        logger.info(f"Ensemble saved → {path}")

    @classmethod
    def load(cls, path=None) -> "StackingEnsemble":
        path = path or DATA_PROCESSED_DIR / "ensemble_model.pkl"
        return joblib.load(path)

    def get_params(self) -> dict:
        return {"base_models": [m.name for m in self.base_models]}
