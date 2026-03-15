"""
Feature Engineering — Phase 2

New features created
--------------------
Time-based
    hour_of_day        : 0–23 extracted from Time column
    is_night           : 1 if hour between 0–6
    is_weekend         : 1 if day falls on Sat/Sun (approximated from Time)
    time_since_last    : seconds since previous transaction (global)

Velocity (rolling windows — how fast is spending happening?)
    tx_count_1h        : number of transactions in last 1 hour
    tx_count_6h        : number of transactions in last 6 hours
    tx_count_24h       : number of transactions in last 24 hours

Amount statistics (rolling windows)
    amt_mean_1h        : rolling mean of Amount in last 1 hour
    amt_std_1h         : rolling std  of Amount in last 1 hour
    amt_mean_24h       : rolling mean of Amount in last 24 hours
    amt_std_24h        : rolling std  of Amount in last 24 hours

Amount deviation (how unusual is THIS transaction?)
    amt_zscore_1h      : z-score of Amount vs last 1h window
    amt_zscore_24h     : z-score of Amount vs last 24h window

PCA-based interaction features
    v1_v2_interaction  : V1 × V2
    v3_v4_interaction  : V3 × V4
    pca_magnitude      : L2 norm of all V1–V28 features (overall anomaly signal)
    pca_top5_mean      : mean of the 5 most fraud-correlated PCA components

Amount transformations
    amount_log         : log1p(Amount) — reduces right skew
    amount_squared     : Amount² — amplifies large outlier amounts
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import joblib
from loguru import logger

from src.config import (
    DATA_PROCESSED_DIR,
    DATA_INTERIM_DIR,
    PCA_FEATURES,
    TARGET,
)

FEATURE_NAMES_PATH = DATA_PROCESSED_DIR / "feature_names.pkl"

# PCA components most correlated with fraud (from EDA on Kaggle dataset)
TOP_FRAUD_PCA = ["V17", "V14", "V12", "V10", "V16"]


# ─── Public API ──────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """
    Run the full feature engineering pipeline on the raw dataframe.

    Parameters
    ----------
    df   : Raw dataframe from loader (must contain Time, Amount, V1-V28, Class)
    save : Persist the feature-engineered dataframe and feature name list

    Returns
    -------
    pd.DataFrame with all original + new features
    """
    logger.info("Starting feature engineering …")
    df = df.copy()

    df = _add_time_features(df)
    df = _add_velocity_features(df)
    df = _add_amount_stats(df)
    df = _add_amount_deviation(df)
    df = _add_pca_interactions(df)
    df = _add_amount_transforms(df)

    # Fill any NaNs created by rolling windows (first rows have no history)
    df = df.fillna(0)

    new_feature_cols = _get_new_feature_names()
    logger.info(f"New features added: {len(new_feature_cols)}")
    logger.info(f"Total features: {len(df.columns) - 1}")  # -1 for Target

    if save:
        _save_engineered(df, new_feature_cols)

    logger.success("Feature engineering complete ✓")
    return df


def load_engineered() -> pd.DataFrame:
    """Reload the saved feature-engineered dataframe."""
    path = DATA_INTERIM_DIR / "features.pkl"
    if not path.exists():
        raise FileNotFoundError("Run engineer_features() first.")
    return joblib.load(path)


def get_feature_names() -> list:
    """Return the list of ALL feature names (excluding target)."""
    if not FEATURE_NAMES_PATH.exists():
        raise FileNotFoundError("Run engineer_features() first.")
    return joblib.load(FEATURE_NAMES_PATH)


# ─── Feature groups ──────────────────────────────────────────────────────────

def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract hour, night flag, weekend flag, and time-since-last."""
    logger.info("  Adding time features …")

    # Time is in seconds from start of data collection (2 days)
    # Convert to hours for readability
    df["hour_of_day"] = (df["Time"] / 3600 % 24).astype(int)
    df["is_night"]    = (df["hour_of_day"].between(0, 6)).astype(int)

    # Approximate day of week from Time (dataset spans ~2 days)
    df["day_of_week"]  = (df["Time"] // 86400 % 7).astype(int)
    df["is_weekend"]   = (df["day_of_week"].isin([5, 6])).astype(int)

    # Time since last transaction (global, not per-card — we have no card IDs)
    df["time_since_last"] = df["Time"].diff().fillna(0)
    df["time_since_last"] = df["time_since_last"].clip(lower=0)

    return df


def _add_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count transactions within rolling time windows.

    Since we have no card IDs, we use global rolling counts as a proxy
    for burst activity — a common fraud pattern.
    """
    logger.info("  Adding velocity features …")

    # Sort by time for rolling operations
    df = df.sort_values("Time").reset_index(drop=True)

    time_seconds = df["Time"].values

    def count_in_window(idx: int, window_seconds: float) -> int:
        t = time_seconds[idx]
        return np.sum((time_seconds[:idx+1] >= t - window_seconds) &
                      (time_seconds[:idx+1] <= t))

    # Vectorized approach using searchsorted (much faster than apply)
    def rolling_count(window_sec: float) -> np.ndarray:
        left  = np.searchsorted(time_seconds, time_seconds - window_sec, side="left")
        right = np.arange(1, len(time_seconds) + 1)
        return (right - left).astype(float)

    df["tx_count_1h"]  = rolling_count(3600)
    df["tx_count_6h"]  = rolling_count(21600)
    df["tx_count_24h"] = rolling_count(86400)

    return df


def _add_amount_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling mean and std of Amount within time windows.
    Uses pandas rolling with a time-based index.
    """
    logger.info("  Adding amount rolling statistics …")

    # Set Time as index for time-based rolling
    df_time = df.set_index(
        pd.to_datetime(df["Time"], unit="s")
    )["Amount"]

    df["amt_mean_1h"]  = df_time.rolling("1h",  min_periods=1).mean().values
    df["amt_std_1h"]   = df_time.rolling("1h",  min_periods=1).std().fillna(0).values
    df["amt_mean_24h"] = df_time.rolling("24h", min_periods=1).mean().values
    df["amt_std_24h"]  = df_time.rolling("24h", min_periods=1).std().fillna(0).values

    return df


def _add_amount_deviation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Z-score of the current Amount vs recent rolling window.
    High z-score = unusually large transaction = fraud signal.
    """
    logger.info("  Adding amount deviation (z-scores) …")

    eps = 1e-8  # avoid division by zero

    df["amt_zscore_1h"] = (
        (df["Amount"] - df["amt_mean_1h"]) /
        (df["amt_std_1h"] + eps)
    )
    df["amt_zscore_24h"] = (
        (df["Amount"] - df["amt_mean_24h"]) /
        (df["amt_std_24h"] + eps)
    )

    # Clip extreme z-scores (don't let outliers dominate)
    df["amt_zscore_1h"]  = df["amt_zscore_1h"].clip(-10, 10)
    df["amt_zscore_24h"] = df["amt_zscore_24h"].clip(-10, 10)

    return df


def _add_pca_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Interaction terms between PCA components.
    PCA features are already meaningful signals — their products
    can capture non-linear fraud patterns.
    """
    logger.info("  Adding PCA interaction features …")

    df["v1_v2_interaction"] = df["V1"] * df["V2"]
    df["v3_v4_interaction"] = df["V3"] * df["V4"]

    # L2 norm of all PCA features — overall "distance from normal"
    pca_matrix = df[PCA_FEATURES].values
    df["pca_magnitude"] = np.linalg.norm(pca_matrix, axis=1)

    # Mean of top 5 fraud-correlated PCA components
    df["pca_top5_mean"] = df[TOP_FRAUD_PCA].mean(axis=1)

    return df


def _add_amount_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mathematical transformations of Amount to help tree models
    and neural networks handle the skewed distribution.
    """
    logger.info("  Adding amount transformations …")

    df["amount_log"]     = np.log1p(df["Amount"])
    df["amount_squared"] = df["Amount"] ** 2

    return df


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _get_new_feature_names() -> list:
    return [
        "hour_of_day", "is_night", "day_of_week", "is_weekend", "time_since_last",
        "tx_count_1h", "tx_count_6h", "tx_count_24h",
        "amt_mean_1h", "amt_std_1h", "amt_mean_24h", "amt_std_24h",
        "amt_zscore_1h", "amt_zscore_24h",
        "v1_v2_interaction", "v3_v4_interaction", "pca_magnitude", "pca_top5_mean",
        "amount_log", "amount_squared",
    ]


def _save_engineered(df: pd.DataFrame, new_cols: list) -> None:
    DATA_INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(df, DATA_INTERIM_DIR / "features.pkl")

    all_features = [c for c in df.columns if c != TARGET]
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(all_features, FEATURE_NAMES_PATH)

    logger.info(f"Engineered dataset saved → {DATA_INTERIM_DIR / 'features.pkl'}")
    logger.info(f"Feature names saved     → {FEATURE_NAMES_PATH}")
