"""
Tests for Phase 2: Feature Engineering

Run with:  pytest tests/test_phase2.py -v
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PCA_FEATURES, TARGET
from src.features.engineer import (
    engineer_features,
    _add_time_features,
    _add_velocity_features,
    _add_amount_stats,
    _add_amount_deviation,
    _add_pca_interactions,
    _add_amount_transforms,
    _get_new_feature_names,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df() -> pd.DataFrame:
    n = 500
    rng = np.random.default_rng(42)
    data = {f: rng.normal(size=n) for f in PCA_FEATURES}
    data["Time"]   = np.sort(rng.uniform(0, 172800, size=n))
    data["Amount"] = rng.exponential(scale=100, size=n)
    data["Class"]  = (rng.random(n) < 0.01).astype(int)
    return pd.DataFrame(data)


# ─── Time features ────────────────────────────────────────────────────────────

class TestTimeFeatures:
    def test_hour_of_day_range(self, sample_df):
        df = _add_time_features(sample_df.copy())
        assert df["hour_of_day"].between(0, 23).all()

    def test_is_night_binary(self, sample_df):
        df = _add_time_features(sample_df.copy())
        assert set(df["is_night"].unique()).issubset({0, 1})

    def test_is_weekend_binary(self, sample_df):
        df = _add_time_features(sample_df.copy())
        assert set(df["is_weekend"].unique()).issubset({0, 1})

    def test_time_since_last_non_negative(self, sample_df):
        df = _add_time_features(sample_df.copy())
        assert (df["time_since_last"] >= 0).all()


# ─── Velocity features ────────────────────────────────────────────────────────

class TestVelocityFeatures:
    def test_tx_count_columns_exist(self, sample_df):
        df = _add_velocity_features(sample_df.copy())
        assert "tx_count_1h"  in df.columns
        assert "tx_count_6h"  in df.columns
        assert "tx_count_24h" in df.columns

    def test_tx_count_positive(self, sample_df):
        df = _add_velocity_features(sample_df.copy())
        assert (df["tx_count_1h"]  >= 1).all()
        assert (df["tx_count_6h"]  >= 1).all()
        assert (df["tx_count_24h"] >= 1).all()

    def test_tx_count_ordering(self, sample_df):
        df = _add_velocity_features(sample_df.copy())
        # 24h window should always be >= 6h >= 1h
        assert (df["tx_count_24h"] >= df["tx_count_6h"]).all()
        assert (df["tx_count_6h"]  >= df["tx_count_1h"]).all()


# ─── Amount stats ─────────────────────────────────────────────────────────────

class TestAmountStats:
    def test_rolling_mean_positive(self, sample_df):
        df = _add_amount_stats(sample_df.copy())
        assert (df["amt_mean_1h"]  > 0).all()
        assert (df["amt_mean_24h"] > 0).all()

    def test_rolling_std_non_negative(self, sample_df):
        df = _add_amount_stats(sample_df.copy())
        assert (df["amt_std_1h"]  >= 0).all()
        assert (df["amt_std_24h"] >= 0).all()


# ─── Amount deviation ─────────────────────────────────────────────────────────

class TestAmountDeviation:
    def test_zscore_clipped(self, sample_df):
        df = _add_amount_stats(sample_df.copy())
        df = _add_amount_deviation(df)
        assert df["amt_zscore_1h"].between(-10, 10).all()
        assert df["amt_zscore_24h"].between(-10, 10).all()

    def test_no_nan_in_zscores(self, sample_df):
        df = _add_amount_stats(sample_df.copy())
        df = _add_amount_deviation(df)
        assert df["amt_zscore_1h"].isna().sum()  == 0
        assert df["amt_zscore_24h"].isna().sum() == 0


# ─── PCA interactions ─────────────────────────────────────────────────────────

class TestPCAInteractions:
    def test_interaction_columns_exist(self, sample_df):
        df = _add_pca_interactions(sample_df.copy())
        assert "v1_v2_interaction" in df.columns
        assert "v3_v4_interaction" in df.columns
        assert "pca_magnitude"     in df.columns
        assert "pca_top5_mean"     in df.columns

    def test_pca_magnitude_non_negative(self, sample_df):
        df = _add_pca_interactions(sample_df.copy())
        assert (df["pca_magnitude"] >= 0).all()

    def test_v1_v2_interaction_correct(self, sample_df):
        df = _add_pca_interactions(sample_df.copy())
        expected = (sample_df["V1"] * sample_df["V2"]).values
        np.testing.assert_array_almost_equal(
            df["v1_v2_interaction"].values, expected
        )


# ─── Amount transforms ────────────────────────────────────────────────────────

class TestAmountTransforms:
    def test_log_transform_non_negative(self, sample_df):
        df = _add_amount_transforms(sample_df.copy())
        assert (df["amount_log"] >= 0).all()

    def test_log_transform_correct(self, sample_df):
        df = _add_amount_transforms(sample_df.copy())
        expected = np.log1p(sample_df["Amount"].values)
        np.testing.assert_array_almost_equal(
            df["amount_log"].values, expected
        )

    def test_squared_non_negative(self, sample_df):
        df = _add_amount_transforms(sample_df.copy())
        assert (df["amount_squared"] >= 0).all()


# ─── Full pipeline ────────────────────────────────────────────────────────────

class TestFullPipeline:
    def test_no_nulls_after_engineering(self, sample_df):
        df = engineer_features(sample_df.copy(), save=False)
        assert df.isnull().sum().sum() == 0

    def test_all_new_features_present(self, sample_df):
        df = engineer_features(sample_df.copy(), save=False)
        for feat in _get_new_feature_names():
            assert feat in df.columns, f"Missing feature: {feat}"

    def test_target_preserved(self, sample_df):
        df = engineer_features(sample_df.copy(), save=False)
        assert TARGET in df.columns
        assert df[TARGET].equals(
            sample_df.sort_values("Time").reset_index(drop=True)[TARGET]
        )

    def test_row_count_preserved(self, sample_df):
        df = engineer_features(sample_df.copy(), save=False)
        assert len(df) == len(sample_df)
