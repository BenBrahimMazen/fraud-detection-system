"""
Preprocessing pipeline.

Steps
-----
1. Scale Amount and Time (StandardScaler)
2. Stratified train / test split
3. Handle class imbalance with SMOTE (training set only)
4. Persist fitted scalers for inference
"""
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import joblib
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from src.config import (
    DATA_PROCESSED_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    PCA_FEATURES,
    RAW_FEATURES,
    TARGET,
)

SCALER_PATH = DATA_PROCESSED_DIR / "scaler.pkl"


# ─── Public API ──────────────────────────────────────────────────────────────

def preprocess(
    df: pd.DataFrame,
    apply_smote: bool = True,
    save_artifacts: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    df           : Raw dataframe from loader
    apply_smote  : Whether to oversample the minority class on train set
    save_artifacts : Persist scaler to disk

    Returns
    -------
    X_train, X_test, y_train, y_test  (all numpy arrays)
    """
    logger.info("Starting preprocessing pipeline …")

    X, y = _split_features_target(df)
    X_scaled = _scale_features(X, fit=True, save=save_artifacts)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,          # preserve class ratio in both splits
    )
    logger.info(f"Train size: {len(X_train):,}  |  Test size: {len(X_test):,}")

    if apply_smote:
        X_train, y_train = _apply_smote(X_train, y_train)

    _save_splits(X_train, X_test, y_train, y_test)

    logger.success("Preprocessing complete ✓")
    return X_train, X_test, y_train, y_test


def scale_for_inference(X: pd.DataFrame) -> np.ndarray:
    """
    Apply the *fitted* scaler to new inference data.
    Must call preprocess() (with save_artifacts=True) at least once first.
    """
    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Scaler not found at {SCALER_PATH}. Run preprocessing first."
        )
    scaler: StandardScaler = joblib.load(SCALER_PATH)
    return scaler.transform(X[RAW_FEATURES].values.reshape(-1, len(RAW_FEATURES)))


# ─── Private helpers ─────────────────────────────────────────────────────────

def _split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    return df.drop(columns=[TARGET]), df[TARGET]


def _scale_features(
    X: pd.DataFrame,
    fit: bool = True,
    save: bool = True,
) -> np.ndarray:
    """
    Scale Amount and Time; PCA features (V1–V28) are already scaled by Kaggle.
    Returns a numpy array with all features.
    """
    scaler = StandardScaler()

    if fit:
        X = X.copy()
        X[RAW_FEATURES] = scaler.fit_transform(X[RAW_FEATURES])
        logger.info("StandardScaler fitted on Amount + Time")
        if save:
            DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
            joblib.dump(scaler, SCALER_PATH)
            logger.info(f"Scaler saved → {SCALER_PATH}")
    else:
        loaded_scaler = joblib.load(SCALER_PATH)
        X = X.copy()
        X[RAW_FEATURES] = loaded_scaler.transform(X[RAW_FEATURES])

    return X.values   # return numpy for sklearn compatibility


def _apply_smote(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    SMOTE oversampling — applied ONLY on the training set to prevent leakage.

    Strategy: 'minority' — only creates synthetic fraud samples.
    k_neighbors=5 is the SMOTE default; safe for this dataset size.
    """
    logger.info("Applying SMOTE to training set …")
    fraud_before = y_train.sum()
    legit_before = len(y_train) - fraud_before

    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    fraud_after = y_res.sum()
    logger.info(
        f"Before SMOTE → legit: {legit_before:,}  fraud: {fraud_before:,}\n"
        f"After  SMOTE → legit: {len(y_res) - fraud_after:,}  fraud: {fraud_after:,}"
    )
    return X_res, y_res


def _save_splits(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """Persist splits so other modules (feature eng, models) can reload them."""
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump((X_train, X_test, y_train, y_test),
                DATA_PROCESSED_DIR / "splits.pkl")
    logger.info(f"Splits saved → {DATA_PROCESSED_DIR / 'splits.pkl'}")


def load_splits() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reload persisted train/test splits."""
    path = DATA_PROCESSED_DIR / "splits.pkl"
    if not path.exists():
        raise FileNotFoundError("Run preprocess() first to generate splits.")
    return joblib.load(path)
