"""
SHAP Explainability — Phase 4

Provides
--------
- Global feature importance (bar + beeswarm plots)
- Per-transaction waterfall plot ("why was THIS flagged?")
- Force plot for interactive explanation
- generate_explanation() — called by the API for every prediction
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for servers
import matplotlib.pyplot as plt
from loguru import logger

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("shap not installed — run: pip install shap")

from src.config import DATA_PROCESSED_DIR, PCA_FEATURES, RAW_FEATURES

PLOTS_DIR    = DATA_PROCESSED_DIR / "shap_plots"
EXPLAINER_PATH = DATA_PROCESSED_DIR / "shap_explainer.pkl"


# ─── Public API ──────────────────────────────────────────────────────────────

def build_explainer(model, X_background: np.ndarray, feature_names: List[str]) -> "shap.Explainer":
    """
    Build and save a SHAP TreeExplainer for the given model.

    Parameters
    ----------
    model          : Fitted XGBoost or RandomForest model (must have .model attr)
    X_background   : Sample of training data used as background (200-500 rows)
    feature_names  : List of feature names matching X columns
    """
    if not SHAP_AVAILABLE:
        raise ImportError("Install shap: pip install shap")

    logger.info("Building SHAP TreeExplainer …")

    # TreeExplainer is exact (not approximate) for tree-based models
    explainer = shap.TreeExplainer(
        model.model,
        data=shap.sample(X_background, 200),
        feature_names=feature_names,
    )

    joblib.dump(explainer, EXPLAINER_PATH)
    logger.success(f"Explainer saved → {EXPLAINER_PATH}")
    return explainer


def load_explainer() -> "shap.Explainer":
    if not EXPLAINER_PATH.exists():
        raise FileNotFoundError("Run build_explainer() first.")
    return joblib.load(EXPLAINER_PATH)


def compute_shap_values(
    explainer: "shap.Explainer",
    X: np.ndarray,
) -> np.ndarray:
    """Compute SHAP values for a batch of samples."""
    logger.info(f"Computing SHAP values for {len(X)} samples …")
    shap_values = explainer.shap_values(X)
    # For binary classifiers shap_values may be a list [class0, class1]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    return shap_values


def plot_global_importance(
    shap_values: np.ndarray,
    feature_names: List[str],
    top_n: int = 20,
    save: bool = True,
) -> plt.Figure:
    """
    Bar chart of mean absolute SHAP values — global feature importance.
    This is the plot that answers: 'Which features drive fraud predictions most?'
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    mean_abs = np.abs(shap_values).mean(axis=0)
    indices  = np.argsort(mean_abs)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#E8593C" if i < 5 else "#4A90D9" for i in range(top_n)]
    ax.barh(
        [feature_names[i] for i in indices[::-1]],
        mean_abs[indices[::-1]],
        color=colors[::-1],
    )
    ax.set_xlabel("Mean |SHAP value|", fontsize=12)
    ax.set_title("Global Feature Importance (SHAP)", fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if save:
        path = PLOTS_DIR / "global_importance.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"Global importance plot saved → {path}")

    return fig


def plot_waterfall(
    explainer: "shap.Explainer",
    X_single: np.ndarray,
    feature_names: List[str],
    transaction_id: int = 0,
    save: bool = True,
) -> plt.Figure:
    """
    Waterfall plot for a single transaction.
    Shows exactly which features pushed the prediction toward or away from fraud.
    Red bars = pushed toward fraud, Blue bars = pushed away from fraud.
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    shap_vals = explainer.shap_values(X_single.reshape(1, -1))
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    shap_vals = shap_vals[0]

    # Sort by absolute magnitude
    indices   = np.argsort(np.abs(shap_vals))[::-1][:15]
    top_vals  = shap_vals[indices]
    top_names = [feature_names[i] for i in indices]

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#E8593C" if v > 0 else "#4A90D9" for v in top_vals[::-1]]
    ax.barh(top_names[::-1], top_vals[::-1], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP value (impact on fraud probability)", fontsize=11)
    ax.set_title(
        f"Transaction #{transaction_id} — Why was this flagged?",
        fontsize=13,
        fontweight="bold",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add value labels
    for i, (val, name) in enumerate(zip(top_vals[::-1], top_names[::-1])):
        ax.text(
            val + (0.001 if val >= 0 else -0.001),
            i,
            f"{val:+.3f}",
            va="center",
            ha="left" if val >= 0 else "right",
            fontsize=9,
        )

    plt.tight_layout()

    if save:
        path = PLOTS_DIR / f"waterfall_tx{transaction_id}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"Waterfall plot saved → {path}")

    return fig


def plot_beeswarm(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: List[str],
    save: bool = True,
) -> plt.Figure:
    """
    Beeswarm summary plot — shows distribution of SHAP values per feature.
    Color = feature value (red=high, blue=low).
    This is the most comprehensive single view of model behaviour.
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 9))
    shap.summary_plot(
        shap_values,
        X,
        feature_names=feature_names,
        show=False,
        max_display=20,
        plot_size=None,
    )
    fig = plt.gcf()
    plt.title("SHAP Beeswarm — Feature Impact Distribution", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save:
        path = PLOTS_DIR / "beeswarm.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"Beeswarm plot saved → {path}")

    return fig


def generate_explanation(
    explainer: "shap.Explainer",
    X_single: np.ndarray,
    feature_names: List[str],
    top_n: int = 5,
) -> Dict:
    """
    Generate a structured explanation for ONE transaction.
    Called by the FastAPI /explain endpoint.

    Returns
    -------
    {
        "top_fraud_drivers":  [{"feature": ..., "shap_value": ..., "direction": ...}],
        "top_fraud_reducers": [{"feature": ..., "shap_value": ..., "direction": ...}],
        "base_value":         float,
        "prediction_value":   float,
    }
    """
    shap_vals = explainer.shap_values(X_single.reshape(1, -1))
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    shap_vals = shap_vals[0]

    base_value = float(explainer.expected_value)
    if isinstance(base_value, list):
        base_value = base_value[1]

    # Split into fraud drivers (positive) and reducers (negative)
    positive_idx = np.argsort(shap_vals)[::-1][:top_n]
    negative_idx = np.argsort(shap_vals)[:top_n]

    def make_factor(idx):
        return {
            "feature":    feature_names[idx],
            "shap_value": round(float(shap_vals[idx]), 4),
            "direction":  "increases_fraud_risk" if shap_vals[idx] > 0 else "decreases_fraud_risk",
            "feature_value": round(float(X_single[idx]), 4),
        }

    return {
        "top_fraud_drivers":  [make_factor(i) for i in positive_idx],
        "top_fraud_reducers": [make_factor(i) for i in negative_idx],
        "base_value":         round(base_value, 4),
        "prediction_value":   round(float(base_value + shap_vals.sum()), 4),
    }


# ─── Runner ──────────────────────────────────────────────────────────────────

def run_explainability(model_name: str = "xgboost") -> None:
    """
    Full explainability pipeline:
    1. Load model + data
    2. Build explainer
    3. Compute SHAP values on test set
    4. Generate and save all plots
    """
    logger.info("=" * 60)
    logger.info("  PHASE 4: SHAP EXPLAINABILITY")
    logger.info("=" * 60)

    # Load model
    if model_name == "xgboost":
        from src.models.xgboost_model import XGBoostFraudModel
        model = XGBoostFraudModel.load()
    elif model_name == "random_forest":
        from src.models.random_forest_model import RandomForestFraudModel
        model = RandomForestFraudModel.load()
    else:
        raise ValueError(f"SHAP TreeExplainer supports: xgboost, random_forest")

    # Load data
    from src.data.preprocessor import load_splits
    X_train, X_test, y_train, y_test = load_splits()

    # Load feature names
    feature_names_path = DATA_PROCESSED_DIR / "feature_names.pkl"
    if feature_names_path.exists():
        feature_names = joblib.load(feature_names_path)
        # Trim to match actual feature count
        feature_names = feature_names[:X_train.shape[1]]
    else:
        feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

    # Build explainer
    explainer = build_explainer(model, X_train, feature_names)

    # Compute SHAP on a sample of test set (full test set is slow)
    sample_size = min(500, len(X_test))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_test), size=sample_size, replace=False)
    X_sample = X_test[idx]

    logger.info(f"Computing SHAP values on {sample_size} test samples …")
    shap_values = compute_shap_values(explainer, X_sample)

    # Generate all plots
    plot_global_importance(shap_values, feature_names)
    plot_beeswarm(shap_values, X_sample, feature_names)

    # Waterfall for first fraud transaction found
    # Convert y_test to numpy array first to avoid pandas index issues
    y_test_arr = np.array(y_test)
    fraud_idx = np.where(y_test_arr[idx] == 1)[0]
    if len(fraud_idx) > 0:
        tx_idx = fraud_idx[0]
        plot_waterfall(explainer, X_sample[tx_idx], feature_names, transaction_id=tx_idx)
        logger.info("Waterfall plot generated for first fraud transaction")

    # Example explanation dict
    example_explanation = generate_explanation(explainer, X_sample[0], feature_names)
    logger.info(f"Example explanation: {example_explanation['top_fraud_drivers'][:2]}")

    logger.success("=" * 60)
    logger.success(f"  Plots saved → {PLOTS_DIR}")
    logger.success("  Ready for Phase 5: FastAPI 🚀")
    logger.success("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="xgboost", choices=["xgboost", "random_forest"])
    args = parser.parse_args()
    run_explainability(args.model)
