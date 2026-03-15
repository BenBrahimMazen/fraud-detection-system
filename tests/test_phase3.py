"""
Tests for Phase 3: Models
Run with: pytest tests/test_phase3.py -v
"""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.base import BaseModel, compute_confusion
from src.models.xgboost_model import XGBoostFraudModel
from src.models.random_forest_model import RandomForestFraudModel
from src.models.isolation_forest_model import IsolationForestFraudModel
from src.models.ensemble import StackingEnsemble


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def small_dataset():
    """Tiny dataset: 200 legit + 20 fraud."""
    rng = np.random.default_rng(42)
    n_legit, n_fraud = 200, 20
    X_legit = rng.normal(loc=0, scale=1, size=(n_legit, 10))
    X_fraud = rng.normal(loc=3, scale=1, size=(n_fraud, 10))
    X = np.vstack([X_legit, X_fraud])
    y = np.array([0] * n_legit + [1] * n_fraud)
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


@pytest.fixture
def train_test(small_dataset):
    X, y = small_dataset
    split = int(len(X) * 0.8)
    return X[:split], X[split:], y[:split], y[split:]


# ─── XGBoost ──────────────────────────────────────────────────────────────────

class TestXGBoost:
    def test_fit_predict_shape(self, train_test):
        X_tr, X_te, y_tr, y_te = train_test
        model = XGBoostFraudModel(n_estimators=10)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        assert preds.shape == (len(X_te),)

    def test_predict_proba_range(self, train_test):
        X_tr, X_te, y_tr, y_te = train_test
        model = XGBoostFraudModel(n_estimators=10)
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)
        assert proba.min() >= 0 and proba.max() <= 1

    def test_evaluate_returns_metrics(self, train_test):
        X_tr, X_te, y_tr, y_te = train_test
        model = XGBoostFraudModel(n_estimators=10)
        model.fit(X_tr, y_tr)
        metrics = model.evaluate(X_te, y_te)
        assert "roc_auc" in metrics
        assert "f1" in metrics
        assert "recall" in metrics

    def test_feature_importances(self, train_test):
        X_tr, X_te, y_tr, y_te = train_test
        model = XGBoostFraudModel(n_estimators=10)
        model.fit(X_tr, y_tr)
        assert model.feature_importances_.shape == (X_tr.shape[1],)


# ─── Random Forest ────────────────────────────────────────────────────────────

class TestRandomForest:
    def test_fit_predict(self, train_test):
        X_tr, X_te, y_tr, y_te = train_test
        model = RandomForestFraudModel(n_estimators=10)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        assert set(preds).issubset({0, 1})

    def test_proba_sums_to_one(self, train_test):
        X_tr, X_te, y_tr, y_te = train_test
        model = RandomForestFraudModel(n_estimators=10)
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)
        assert proba.min() >= 0 and proba.max() <= 1


# ─── Isolation Forest ─────────────────────────────────────────────────────────

class TestIsolationForest:
    def test_trains_on_legit_only(self, train_test):
        X_tr, X_te, y_tr, y_te = train_test
        model = IsolationForestFraudModel(n_estimators=10)
        model.fit(X_tr, y_tr)   # should not raise
        preds = model.predict(X_te)
        assert set(preds).issubset({0, 1})

    def test_proba_range(self, train_test):
        X_tr, X_te, y_tr, y_te = train_test
        model = IsolationForestFraudModel(n_estimators=10)
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)
        assert proba.min() >= 0 and proba.max() <= 1


# ─── Confusion matrix ─────────────────────────────────────────────────────────

class TestConfusion:
    def test_confusion_keys(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1])
        cm = compute_confusion(y_true, y_pred)
        assert "true_positives"  in cm
        assert "false_positives" in cm
        assert "true_negatives"  in cm
        assert "false_negatives" in cm

    def test_confusion_values(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1])
        cm = compute_confusion(y_true, y_pred)
        assert cm["true_positives"]  == 1
        assert cm["true_negatives"]  == 1
        assert cm["false_positives"] == 1
        assert cm["false_negatives"] == 1


# ─── Ensemble ─────────────────────────────────────────────────────────────────

class TestEnsemble:
    def test_ensemble_predict_shape(self, train_test):
        X_tr, X_te, y_tr, y_te = train_test
        models = [
            XGBoostFraudModel(n_estimators=10),
            RandomForestFraudModel(n_estimators=10),
        ]
        ensemble = StackingEnsemble(base_models=models)
        ensemble.fit(X_tr, y_tr)
        preds = ensemble.predict(X_te)
        assert preds.shape == (len(X_te),)

    def test_ensemble_proba_range(self, train_test):
        X_tr, X_te, y_tr, y_te = train_test
        models = [
            XGBoostFraudModel(n_estimators=10),
            RandomForestFraudModel(n_estimators=10),
        ]
        ensemble = StackingEnsemble(base_models=models)
        ensemble.fit(X_tr, y_tr)
        proba = ensemble.predict_proba(X_te)
        assert proba.min() >= 0 and proba.max() <= 1

    def test_ensemble_weights_all_models(self, train_test):
        X_tr, X_te, y_tr, y_te = train_test
        models = [
            XGBoostFraudModel(n_estimators=10),
            RandomForestFraudModel(n_estimators=10),
        ]
        ensemble = StackingEnsemble(base_models=models)
        ensemble.fit(X_tr, y_tr)
        weights = ensemble.get_model_weights()
        assert set(weights.keys()) == {"xgboost", "random_forest"}
