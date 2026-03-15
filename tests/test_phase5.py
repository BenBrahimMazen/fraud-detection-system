"""
Tests for Phase 5: FastAPI
Run with: pytest tests/test_phase5.py -v
"""
import sys
from pathlib import Path
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api.schemas import TransactionInput, BatchTransactionInput
from src.scoring.risk_scorer import score_transaction, to_dict


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def make_transaction(**overrides) -> dict:
    base = {f"V{i}": 0.0 for i in range(1, 29)}
    base.update({"Time": 1000.0, "Amount": 50.0})
    base.update(overrides)
    return base


# ─── Schema validation tests ──────────────────────────────────────────────────

class TestSchemas:
    def test_valid_transaction(self):
        tx = TransactionInput(**make_transaction())
        assert tx.Amount == 50.0
        assert tx.Time == 1000.0

    def test_negative_amount_rejected(self):
        with pytest.raises(Exception):
            TransactionInput(**make_transaction(Amount=-10.0))

    def test_batch_min_one(self):
        with pytest.raises(Exception):
            BatchTransactionInput(transactions=[])

    def test_batch_valid(self):
        txs = [TransactionInput(**make_transaction()) for _ in range(3)]
        batch = BatchTransactionInput(transactions=txs)
        assert len(batch.transactions) == 3

    def test_risk_response_fields(self):
        risk = score_transaction(0.9)
        d    = to_dict(risk)
        assert d["tier"]   == "CRITICAL"
        assert d["score"]  >= 85
        assert d["action"] != ""


# ─── API client tests (no real model needed) ──────────────────────────────────

class TestAPIEndpoints:

    @pytest.fixture
    def client(self):
        """Test client with mocked model registry."""
        from fastapi.testclient import TestClient
        from src.api.main import app

        mock_model = MagicMock()
        mock_model.name = "xgboost"
        mock_model.predict_proba = MagicMock(return_value=np.array([0.05]))

        mock_registry = MagicMock()
        mock_registry.is_ready = True
        mock_registry.model = mock_model
        mock_registry.threshold = 0.5
        mock_registry.explainer = None
        mock_registry.feature_names = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]
        mock_registry.preprocess_input = lambda x: x

        with patch("src.api.main.registry", mock_registry):
            with TestClient(app) as c:
                yield c

    def test_health_endpoint(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] in ("ok", "degraded")

    def test_predict_low_risk(self, client):
        r = client.post("/predict", json=make_transaction())
        assert r.status_code == 200
        data = r.json()
        assert "is_fraud"  in data
        assert "risk"      in data
        assert "risk_tier" in data["risk"] or "tier" in data["risk"]

    def test_predict_returns_processing_time(self, client):
        r = client.post("/predict", json=make_transaction())
        assert r.status_code == 200
        assert r.json()["processing_ms"] >= 0

    def test_batch_predict(self, client):
        from unittest.mock import MagicMock
        import src.api.main as main_module
        main_module.registry.model.predict_proba = MagicMock(
            return_value=np.array([0.05, 0.9, 0.3])
        )
        payload = {"transactions": [make_transaction() for _ in range(3)]}
        r = client.post("/batch", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 3

    def test_explain_without_explainer_returns_503(self, client):
        r = client.post("/explain", json=make_transaction())
        assert r.status_code == 503

    def test_model_info(self, client):
        r = client.get("/model/info")
        assert r.status_code == 200
        assert "model_name" in r.json()
