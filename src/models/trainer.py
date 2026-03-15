"""
Training orchestrator — trains all models, logs to MLflow.

Usage
-----
    python src/models/trainer.py
    python src/models/trainer.py --models xgboost random_forest
    python src/models/trainer.py --skip-ensemble
"""
import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import mlflow
import mlflow.sklearn
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    DATA_PROCESSED_DIR,
)
from src.data.preprocessor import load_splits
from src.models.base import BaseModel, compute_confusion
from src.models.xgboost_model import XGBoostFraudModel
from src.models.random_forest_model import RandomForestFraudModel
from src.models.isolation_forest_model import IsolationForestFraudModel
from src.models.ensemble import StackingEnsemble


def get_model_registry() -> Dict[str, BaseModel]:
    """Factory — returns all available models."""
    return {
        "xgboost":          XGBoostFraudModel(),
        "random_forest":    RandomForestFraudModel(),
        "isolation_forest": IsolationForestFraudModel(),
    }


def train_and_log(
    model: BaseModel,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str] = None,
) -> Dict:
    """
    Train a single model, evaluate, and log everything to MLflow.
    Returns the metrics dict.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name=model.name):

        # ── Train ──────────────────────────────────────────────
        if hasattr(model, 'fit'):
            model.fit(X_train, y_train)

        # ── Evaluate ───────────────────────────────────────────
        metrics = model.evaluate(X_test, y_test)
        cm      = compute_confusion(y_test, model.predict(X_test))
        metrics.update(cm)

        # ── Log to MLflow ──────────────────────────────────────
        mlflow.log_params(model.get_params())
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})

        # Log feature importances if available
        if hasattr(model, "feature_importances_") and feature_names:
            importance_dict = {
                f"importance_{n}": float(v)
                for n, v in zip(feature_names, model.feature_importances_)
            }
            mlflow.log_metrics(importance_dict)

        # Save model artifact
        model.save()
        mlflow.log_artifact(str(DATA_PROCESSED_DIR / f"{model.name}_model.pkl"))

        logger.success(
            f"{model.name} → ROC-AUC={metrics['roc_auc']:.4f} "
            f"F1={metrics['f1']:.4f} "
            f"Recall={metrics['recall']:.4f}"
        )

    return metrics


def train_all(
    models_to_train: List[str] = None,
    skip_ensemble: bool = False,
) -> Dict[str, Dict]:
    """
    Train all (or selected) models and return all metrics.
    """
    logger.info("=" * 60)
    logger.info("  PHASE 3: MODEL TRAINING")
    logger.info("=" * 60)

    # Load preprocessed splits
    X_train, X_test, y_train, y_test = load_splits()
    logger.info(f"Train: {X_train.shape}  Test: {X_test.shape}")

    # Load feature names if available
    feature_names_path = DATA_PROCESSED_DIR / "feature_names.pkl"
    feature_names = None
    if feature_names_path.exists():
        import joblib
        feature_names = joblib.load(feature_names_path)

    registry  = get_model_registry()
    to_train  = models_to_train or list(registry.keys())
    all_metrics = {}

    # ── Train base models ─────────────────────────────────────
    trained_models = []
    for name in to_train:
        if name not in registry:
            logger.warning(f"Unknown model: {name} — skipping")
            continue

        model = registry[name]
        metrics = train_and_log(model, X_train, X_test, y_train, y_test, feature_names)
        all_metrics[name] = metrics
        trained_models.append(model)

    # ── Train stacking ensemble ───────────────────────────────
    if not skip_ensemble and len(trained_models) >= 2:
        logger.info("Training stacking ensemble …")
        ensemble = StackingEnsemble(base_models=trained_models)
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

        with mlflow.start_run(run_name="stacking_ensemble"):
            ensemble.fit(X_train, y_train)
            metrics  = ensemble.evaluate(X_test, y_test)
            cm       = compute_confusion(y_test, ensemble.predict(X_test))
            metrics.update(cm)
            mlflow.log_params({"base_models": str([m.name for m in trained_models])})
            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
            ensemble.save()
            mlflow.log_artifact(str(DATA_PROCESSED_DIR / "ensemble_model.pkl"))

        all_metrics["ensemble"] = metrics
        weights = ensemble.get_model_weights()
        logger.info(f"Ensemble weights: {weights}")

    # ── Summary table ─────────────────────────────────────────
    _print_summary(all_metrics)
    return all_metrics


def _print_summary(all_metrics: Dict[str, Dict]) -> None:
    logger.info("=" * 60)
    logger.info("  RESULTS SUMMARY")
    logger.info(f"  {'Model':<22} {'ROC-AUC':>8} {'F1':>8} {'Recall':>8} {'Precision':>10}")
    logger.info("  " + "-" * 58)
    for name, m in all_metrics.items():
        logger.info(
            f"  {name:<22} "
            f"{m['roc_auc']:>8.4f} "
            f"{m['f1']:>8.4f} "
            f"{m['recall']:>8.4f} "
            f"{m['precision']:>10.4f}"
        )
    logger.info("=" * 60)

    # Find best model by ROC-AUC
    best = max(all_metrics, key=lambda k: all_metrics[k]["roc_auc"])
    logger.success(f"Best model: {best} (ROC-AUC={all_metrics[best]['roc_auc']:.4f})")
    logger.success("Ready for Phase 4: SHAP Explainability 🚀")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train fraud detection models")
    parser.add_argument(
        "--models", nargs="+",
        choices=["xgboost", "random_forest", "isolation_forest"],
        help="Which models to train (default: all)",
    )
    parser.add_argument("--skip-ensemble", action="store_true")
    args = parser.parse_args()

    train_all(
        models_to_train=args.models,
        skip_ensemble=args.skip_ensemble,
    )
