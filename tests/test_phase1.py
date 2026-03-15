"""
Tests for Phase 1: data loading and preprocessing.

Run with:  pytest tests/test_phase1.py -v
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PCA_FEATURES, RAW_FEATURES, TARGET
from src.data.loader import (
    _validate_schema,
    _log_statistics,
    get_class_distribution,
    split_features_target,
)
from src.data.preprocessor import _apply_smote, _scale_features


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Tiny synthetic dataframe that mimics the Kaggle schema."""
    n = 1000
    rng = np.random.default_rng(42)

    data = {f: rng.normal(size=n) for f in PCA_FEATURES}
    data["Time"]   = rng.uniform(0, 172800, size=n)
    data["Amount"] = rng.exponential(scale=100, size=n)
    # 1% fraud
    data["Class"]  = (rng.random(n) < 0.01).astype(int)

    return pd.DataFrame(data)


# ─── Loader tests ─────────────────────────────────────────────────────────────

class TestLoader:
    def test_validate_schema_passes(self, sample_df):
        _validate_schema(sample_df)   # should not raise

    def test_validate_schema_raises_on_missing_col(self, sample_df):
        df_bad = sample_df.drop(columns=["V1"])
        with pytest.raises(ValueError, match="missing columns"):
            _validate_schema(df_bad)

    def test_class_distribution_keys(self, sample_df):
        dist = get_class_distribution(sample_df)
        assert "legitimate" in dist
        assert "fraud" in dist
        assert "imbalance_ratio" in dist

    def test_class_distribution_sum(self, sample_df):
        dist = get_class_distribution(sample_df)
        total = dist["legitimate"]["count"] + dist["fraud"]["count"]
        assert total == len(sample_df)

    def test_split_features_target_shapes(self, sample_df):
        X, y = split_features_target(sample_df)
        assert X.shape[1] == len(PCA_FEATURES) + len(RAW_FEATURES)
        assert len(y) == len(sample_df)
        assert TARGET not in X.columns


# ─── Preprocessor tests ───────────────────────────────────────────────────────

class TestPreprocessor:
    def test_scale_features_output_shape(self, sample_df, tmp_path, monkeypatch):
        import src.data.preprocessor as prep_mod
        monkeypatch.setattr(prep_mod, "SCALER_PATH", tmp_path / "scaler.pkl")
        monkeypatch.setattr(prep_mod, "DATA_PROCESSED_DIR", tmp_path)

        X = sample_df.drop(columns=[TARGET])
        result = _scale_features(X, fit=True, save=True)
        assert result.shape == X.shape

    def test_smote_balances_classes(self, sample_df):
        from sklearn.preprocessing import StandardScaler
        X = sample_df.drop(columns=[TARGET]).values
        y = sample_df[TARGET].values

        X_res, y_res = _apply_smote(X, y)

        fraud_count = y_res.sum()
        legit_count = len(y_res) - fraud_count
        # After SMOTE the minority class should be significantly larger
        assert fraud_count > sample_df[TARGET].sum()
        # Total should be larger than original
        assert len(y_res) > len(y)

    def test_smote_preserves_majority(self, sample_df):
        X = sample_df.drop(columns=[TARGET]).values
        y = sample_df[TARGET].values
        legit_before = (y == 0).sum()

        _, y_res = _apply_smote(X, y)
        legit_after = (y_res == 0).sum()

        # SMOTE should NOT touch the majority class
        assert legit_after == legit_before
