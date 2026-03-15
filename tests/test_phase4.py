"""
Tests for Phase 4: SHAP Explainability + Risk Scoring
Run with: pytest tests/test_phase4.py -v
"""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.scoring.risk_scorer import (
    score_transaction,
    score_batch,
    get_risk_distribution,
    to_dict,
    RiskScore,
    THRESHOLDS,
)


# ─── Risk Scorer tests ────────────────────────────────────────────────────────

class TestRiskScorer:

    def test_score_range(self):
        for p in [0.0, 0.1, 0.5, 0.85, 1.0]:
            r = score_transaction(p)
            assert 0 <= r.score <= 100

    def test_low_probability_is_low_tier(self):
        r = score_transaction(0.05)
        assert r.tier == "LOW"

    def test_high_probability_is_critical(self):
        r = score_transaction(0.95)
        assert r.tier == "CRITICAL"

    def test_medium_tier(self):
        r = score_transaction(0.45)
        assert r.tier == "MEDIUM"

    def test_high_tier(self):
        r = score_transaction(0.72)
        assert r.tier == "HIGH"

    def test_score_increases_with_probability(self):
        scores = [score_transaction(p).score for p in [0.1, 0.3, 0.6, 0.9]]
        assert scores == sorted(scores)

    def test_probability_clipped(self):
        r1 = score_transaction(-0.5)
        r2 = score_transaction(1.5)
        assert r1.probability == 0.0
        assert r2.probability == 1.0

    def test_to_dict_keys(self):
        r = score_transaction(0.5)
        d = to_dict(r)
        assert all(k in d for k in ["probability", "score", "tier", "action", "confidence", "color"])

    def test_batch_scoring_length(self):
        probs  = np.array([0.1, 0.4, 0.7, 0.95])
        scores = score_batch(probs)
        assert len(scores) == 4

    def test_risk_distribution_keys(self):
        probs = np.random.uniform(0, 1, 100)
        dist  = get_risk_distribution(probs)
        assert set(dist.keys()) == {"LOW", "MEDIUM", "HIGH", "CRITICAL"}

    def test_risk_distribution_sums_to_100(self):
        probs = np.random.uniform(0, 1, 100)
        dist  = get_risk_distribution(probs)
        total = sum(d["count"] for d in dist.values())
        assert total == 100

    def test_action_not_empty(self):
        for p in [0.1, 0.45, 0.72, 0.95]:
            r = score_transaction(p)
            assert len(r.action) > 0

    def test_color_is_hex(self):
        for p in [0.1, 0.45, 0.72, 0.95]:
            r = score_transaction(p)
            assert r.color.startswith("#")
            assert len(r.color) == 7

    def test_all_tiers_reachable(self):
        tiers = {score_transaction(p).tier for p in [0.1, 0.45, 0.72, 0.95]}
        assert tiers == {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
