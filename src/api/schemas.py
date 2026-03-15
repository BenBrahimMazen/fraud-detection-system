"""
Pydantic schemas — input validation and response models for the API.
Pydantic ensures every request is validated before hitting the model.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator


# ─── Request schemas ─────────────────────────────────────────────────────────

class TransactionInput(BaseModel):
    """Single transaction for /predict and /explain endpoints."""

    # PCA features (V1-V28)
    V1:  float; V2:  float; V3:  float; V4:  float
    V5:  float; V6:  float; V7:  float; V8:  float
    V9:  float; V10: float; V11: float; V12: float
    V13: float; V14: float; V15: float; V16: float
    V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float
    V25: float; V26: float; V27: float; V28: float

    # Raw features
    Time:   float = Field(..., ge=0, description="Seconds since first transaction")
    Amount: float = Field(..., ge=0, description="Transaction amount in USD")

    @field_validator("Amount")
    @classmethod
    def amount_must_be_positive(cls, v):
        if v < 0:
            raise ValueError("Amount must be non-negative")
        return v

    model_config = {"json_schema_extra": {
        "example": {
            "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38,
            "V5": -0.34, "V6": 0.46, "V7": 0.24, "V8": 0.10,
            "V9": 0.36, "V10": 0.09, "V11": -0.55, "V12": -0.62,
            "V13": -0.99, "V14": -0.31, "V15": 1.47, "V16": -0.47,
            "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25,
            "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07,
            "V25": 0.13, "V26": -0.19, "V27": 0.13, "V28": -0.02,
            "Time": 406.0, "Amount": 149.62
        }
    }}


class BatchTransactionInput(BaseModel):
    """Batch of transactions for /batch endpoint."""
    transactions: List[TransactionInput] = Field(
        ..., min_length=1, max_length=1000,
        description="List of transactions (max 1000)"
    )


# ─── Response schemas ─────────────────────────────────────────────────────────

class RiskScoreResponse(BaseModel):
    """Risk assessment for a single transaction."""
    probability:  float
    score:        int
    tier:         str
    action:       str
    confidence:   str
    color:        str


class PredictionResponse(BaseModel):
    """Response from /predict endpoint."""
    transaction_id:  Optional[str]
    is_fraud:        bool
    risk:            RiskScoreResponse
    model_used:      str
    processing_ms:   float


class BatchPredictionResponse(BaseModel):
    """Response from /batch endpoint."""
    total:           int
    fraud_count:     int
    fraud_rate:      float
    predictions:     List[PredictionResponse]
    processing_ms:   float


class ShapFactor(BaseModel):
    feature:        str
    shap_value:     float
    direction:      str
    feature_value:  float


class ExplanationResponse(BaseModel):
    """Response from /explain endpoint."""
    transaction_id:      Optional[str]
    is_fraud:            bool
    risk:                RiskScoreResponse
    top_fraud_drivers:   List[ShapFactor]
    top_fraud_reducers:  List[ShapFactor]
    base_value:          float
    prediction_value:    float
    model_used:          str
    processing_ms:       float


class HealthResponse(BaseModel):
    status:       str
    model_loaded: bool
    version:      str = "1.0.0"


class ModelInfoResponse(BaseModel):
    model_name:    str
    roc_auc:       Optional[float]
    features:      int
    threshold:     float
