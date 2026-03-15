"""
Fraud Detection API — FastAPI application.

Endpoints
---------
GET  /health          — liveness check
GET  /model/info      — model metadata
POST /predict         — score a single transaction
POST /batch           — score up to 1000 transactions
POST /explain         — predict + SHAP explanation
GET  /docs            — auto-generated Swagger UI (built-in)

Usage
-----
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
"""
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.api.schemas import (
    TransactionInput,
    BatchTransactionInput,
    PredictionResponse,
    BatchPredictionResponse,
    ExplanationResponse,
    HealthResponse,
    ModelInfoResponse,
    RiskScoreResponse,
    ShapFactor,
)
from src.api.model_loader import registry
from src.scoring.risk_scorer import score_transaction, to_dict
from src.config import FRAUD_THRESHOLD


# ─── Lifespan (startup / shutdown) ───────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models at startup, clean up at shutdown."""
    logger.info("Starting Fraud Detection API …")
    try:
        registry.initialize()
        logger.success("API ready ✓")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        logger.warning("API starting without models — /predict will fail")
    yield
    logger.info("API shutting down …")


# ─── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Fraud Detection API",
    description="""
## Production Fraud Detection System

Real-time credit card fraud detection powered by an XGBoost + Random Forest
stacking ensemble trained on 284,807 transactions.

### Features
- **Real-time scoring** — sub-10ms predictions
- **Risk tiers** — LOW / MEDIUM / HIGH / CRITICAL with recommended actions
- **SHAP explanations** — know exactly *why* a transaction was flagged
- **Batch processing** — score up to 1,000 transactions per request
    """,
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def transaction_to_array(tx: TransactionInput) -> np.ndarray:
    """Convert a TransactionInput to a numpy row in the correct feature order."""
    from src.config import PCA_FEATURES, RAW_FEATURES
    values = [getattr(tx, f) for f in PCA_FEATURES + RAW_FEATURES]
    return np.array(values, dtype=np.float32)


def make_risk_response(probability: float) -> RiskScoreResponse:
    risk = score_transaction(probability)
    return RiskScoreResponse(**to_dict(risk))


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Liveness check — returns model status."""
    return HealthResponse(
        status="ok" if registry.is_ready else "degraded",
        model_loaded=registry.is_ready,
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["System"])
async def model_info():
    """Return metadata about the loaded model."""
    if not registry.is_ready:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return ModelInfoResponse(
        model_name=registry.model.name,
        roc_auc=None,   # would load from MLflow in production
        features=len(registry.feature_names),
        threshold=registry.threshold,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(
    transaction: TransactionInput,
    transaction_id: Optional[str] = Query(default=None),
    threshold: Optional[float] = Query(default=None, ge=0.0, le=1.0),
):
    """
    Score a single transaction.

    Returns a risk score (0-100), tier (LOW/MEDIUM/HIGH/CRITICAL),
    and recommended action.
    """
    if not registry.is_ready:
        raise HTTPException(status_code=503, detail="Model not loaded")

    t0  = time.perf_counter()
    tid = transaction_id or str(uuid.uuid4())[:8]
    th  = threshold or registry.threshold

    try:
        X = transaction_to_array(transaction).reshape(1, -1)
        X = registry.preprocess_input(X)

        probability = float(registry.model.predict_proba(X)[0])
        is_fraud    = probability >= th
        risk        = make_risk_response(probability)
        elapsed_ms  = (time.perf_counter() - t0) * 1000

        logger.info(
            f"[{tid}] prob={probability:.4f} tier={risk.tier} "
            f"fraud={is_fraud} ({elapsed_ms:.1f}ms)"
        )

        return PredictionResponse(
            transaction_id = tid,
            is_fraud       = is_fraud,
            risk           = risk,
            model_used     = registry.model.name,
            processing_ms  = round(elapsed_ms, 2),
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def batch_predict(payload: BatchTransactionInput):
    """
    Score a batch of up to 1,000 transactions at once.

    More efficient than calling /predict in a loop.
    """
    if not registry.is_ready:
        raise HTTPException(status_code=503, detail="Model not loaded")

    t0 = time.perf_counter()

    try:
        X = np.vstack([
            transaction_to_array(tx).reshape(1, -1)
            for tx in payload.transactions
        ])
        X = registry.preprocess_input(X)

        probabilities = registry.model.predict_proba(X)
        th = registry.threshold

        predictions = []
        for i, (tx, prob) in enumerate(zip(payload.transactions, probabilities)):
            is_fraud = float(prob) >= th
            risk     = make_risk_response(float(prob))
            predictions.append(PredictionResponse(
                transaction_id = str(i),
                is_fraud       = is_fraud,
                risk           = risk,
                model_used     = registry.model.name,
                processing_ms  = 0.0,
            ))

        fraud_count = sum(1 for p in predictions if p.is_fraud)
        elapsed_ms  = (time.perf_counter() - t0) * 1000

        logger.info(
            f"Batch: {len(predictions)} transactions, "
            f"{fraud_count} fraud ({elapsed_ms:.1f}ms)"
        )

        return BatchPredictionResponse(
            total         = len(predictions),
            fraud_count   = fraud_count,
            fraud_rate    = round(fraud_count / len(predictions), 4),
            predictions   = predictions,
            processing_ms = round(elapsed_ms, 2),
        )

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain", response_model=ExplanationResponse, tags=["Explainability"])
async def explain(
    transaction: TransactionInput,
    transaction_id: Optional[str] = Query(default=None),
    top_n: int = Query(default=5, ge=1, le=15),
):
    """
    Predict AND explain why the model made its decision.

    Returns SHAP values showing which features drove the prediction
    toward or away from fraud.
    """
    if not registry.is_ready:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if registry.explainer is None:
        raise HTTPException(
            status_code=503,
            detail="SHAP explainer not available. Run: python -m src.explainability.shap_explainer"
        )

    t0  = time.perf_counter()
    tid = transaction_id or str(uuid.uuid4())[:8]

    try:
        X = transaction_to_array(transaction).reshape(1, -1)
        X = registry.preprocess_input(X)

        probability = float(registry.model.predict_proba(X)[0])
        is_fraud    = probability >= registry.threshold
        risk        = make_risk_response(probability)

        # Generate SHAP explanation
        from src.explainability.shap_explainer import generate_explanation
        explanation = generate_explanation(
            registry.explainer,
            X[0],
            registry.feature_names[:X.shape[1]],
            top_n=top_n,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return ExplanationResponse(
            transaction_id     = tid,
            is_fraud           = is_fraud,
            risk               = risk,
            top_fraud_drivers  = [ShapFactor(**f) for f in explanation["top_fraud_drivers"]],
            top_fraud_reducers = [ShapFactor(**f) for f in explanation["top_fraud_reducers"]],
            base_value         = explanation["base_value"],
            prediction_value   = explanation["prediction_value"],
            model_used         = registry.model.name,
            processing_ms      = round(elapsed_ms, 2),
        )

    except Exception as e:
        logger.error(f"Explanation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
