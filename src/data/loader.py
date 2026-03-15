"""
Data loading module.

Responsibilities
----------------
- Load raw CSV from Kaggle
- Validate schema & types
- Report basic dataset statistics
- Return clean DataFrame ready for preprocessing
"""
from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np
from loguru import logger

from src.config import DATA_RAW_PATH, PCA_FEATURES, RAW_FEATURES, TARGET


# ─── Expected schema ─────────────────────────────────────────────────────────
REQUIRED_COLUMNS = PCA_FEATURES + RAW_FEATURES + [TARGET]


def load_raw_data(path: Path = DATA_RAW_PATH) -> pd.DataFrame:
    """
    Load the raw Kaggle credit card fraud CSV.

    Parameters
    ----------
    path : Path
        Location of creditcard.csv

    Returns
    -------
    pd.DataFrame
        Validated raw dataframe.

    Raises
    ------
    FileNotFoundError  – if CSV is missing
    ValueError         – if required columns are absent
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Download it from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud "
            "and place creditcard.csv in data/raw/"
        )

    logger.info(f"Loading dataset from {path} …")
    df = pd.read_csv(path)
    logger.info(f"Raw shape: {df.shape}")

    _validate_schema(df)
    _log_statistics(df)

    return df


def _validate_schema(df: pd.DataFrame) -> None:
    """Raise ValueError if any required column is missing."""
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    logger.success("Schema validation passed ✓")


def _log_statistics(df: pd.DataFrame) -> None:
    """Log key dataset statistics — class balance, nulls, dtypes."""
    n_total  = len(df)
    n_fraud  = df[TARGET].sum()
    n_legit  = n_total - n_fraud
    fraud_pct = n_fraud / n_total * 100

    logger.info("─" * 50)
    logger.info(f"  Total transactions : {n_total:,}")
    logger.info(f"  Legitimate         : {n_legit:,}  ({100 - fraud_pct:.4f}%)")
    logger.info(f"  Fraudulent         : {n_fraud:,}  ({fraud_pct:.4f}%)")
    logger.info(f"  Imbalance ratio    : 1 : {n_legit // n_fraud}")
    logger.info("─" * 50)

    null_count = df.isnull().sum().sum()
    if null_count:
        logger.warning(f"  Null values found  : {null_count}")
    else:
        logger.info("  Null values        : none ✓")

    logger.info(f"  Amount range       : ${df['Amount'].min():.2f} – ${df['Amount'].max():.2f}")
    logger.info(f"  Time range         : {df['Time'].min():.0f}s – {df['Time'].max():.0f}s")


def get_class_distribution(df: pd.DataFrame) -> dict:
    """Return a dict with class counts and percentages."""
    counts = df[TARGET].value_counts().to_dict()
    total  = len(df)
    return {
        "legitimate": {
            "count": counts.get(0, 0),
            "pct":   round(counts.get(0, 0) / total * 100, 4),
        },
        "fraud": {
            "count": counts.get(1, 0),
            "pct":   round(counts.get(1, 0) / total * 100, 4),
        },
        "imbalance_ratio": counts.get(0, 0) // max(counts.get(1, 1), 1),
    }


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Convenience split: returns (X, y)."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y
