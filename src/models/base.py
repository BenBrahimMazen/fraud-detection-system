"""
Base model interface and shared evaluation utilities.
All models inherit from BaseModel and get logging + metrics for free.
"""
from abc import ABC, abstractmethod
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)


class BaseModel(ABC):
    """Abstract base — every model must implement fit, predict, predict_proba."""

    name: str = "base"

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability of fraud (class 1) for each sample."""
        pass

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """
        Compute all metrics and return as a flat dict (MLflow-friendly).
        Works for both probabilistic and non-probabilistic models.
        """
        try:
            proba = self.predict_proba(X_test)
            preds = (proba >= threshold).astype(int)
            roc_auc = roc_auc_score(y_test, proba)
            avg_prec = average_precision_score(y_test, proba)
        except NotImplementedError:
            preds = self.predict(X_test)
            proba = preds.astype(float)
            roc_auc = roc_auc_score(y_test, proba)
            avg_prec = average_precision_score(y_test, proba)

        metrics = {
            "roc_auc":          round(roc_auc, 6),
            "avg_precision":    round(avg_prec, 6),
            "f1":               round(f1_score(y_test, preds, zero_division=0), 6),
            "precision":        round(precision_score(y_test, preds, zero_division=0), 6),
            "recall":           round(recall_score(y_test, preds, zero_division=0), 6),
            "fraud_detected":   int(preds[y_test == 1].sum()),
            "fraud_missed":     int((y_test == 1).sum() - preds[y_test == 1].sum()),
            "false_positives":  int(preds[y_test == 0].sum()),
        }

        self._log_metrics(metrics, y_test)
        return metrics

    def _log_metrics(self, metrics: Dict, y_test: np.ndarray) -> None:
        total_fraud = int((y_test == 1).sum())
        logger.info(f"  ── {self.name} results ──────────────────")
        logger.info(f"  ROC-AUC        : {metrics['roc_auc']:.4f}")
        logger.info(f"  Avg Precision  : {metrics['avg_precision']:.4f}")
        logger.info(f"  F1 Score       : {metrics['f1']:.4f}")
        logger.info(f"  Precision      : {metrics['precision']:.4f}")
        logger.info(f"  Recall         : {metrics['recall']:.4f}")
        logger.info(f"  Fraud caught   : {metrics['fraud_detected']} / {total_fraud}")
        logger.info(f"  False alarms   : {metrics['false_positives']}")


def compute_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    cm = confusion_matrix(y_true, y_pred)
    return {
        "true_negatives":  int(cm[0, 0]),
        "false_positives": int(cm[0, 1]),
        "false_negatives": int(cm[1, 0]),
        "true_positives":  int(cm[1, 1]),
    }
