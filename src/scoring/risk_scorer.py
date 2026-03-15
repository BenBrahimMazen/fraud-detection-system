"""
Risk Scoring Engine — Phase 4

Converts raw model probability (0–1) into:
- A human-readable risk score (0–100)
- A risk tier: LOW / MEDIUM / HIGH / CRITICAL
- A recommended action
- Confidence level

This is what banks actually display to fraud analysts.
"""
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
from loguru import logger


@dataclass
class RiskScore:
    """Structured risk assessment for a single transaction."""
    probability:   float        # raw model output 0–1
    score:         int          # 0–100 human-readable
    tier:          str          # LOW / MEDIUM / HIGH / CRITICAL
    action:        str          # recommended action
    confidence:    str          # LOW / MEDIUM / HIGH
    color:         str          # hex color for dashboard


# ─── Tier thresholds (tunable) ───────────────────────────────────────────────
THRESHOLDS = {
    "LOW":      (0.00, 0.30),
    "MEDIUM":   (0.30, 0.60),
    "HIGH":     (0.60, 0.85),
    "CRITICAL": (0.85, 1.00),
}

TIER_CONFIG = {
    "LOW": {
        "action":     "Allow transaction",
        "color":      "#27AE60",
        "confidence": "HIGH",
    },
    "MEDIUM": {
        "action":     "Flag for review",
        "color":      "#F39C12",
        "confidence": "MEDIUM",
    },
    "HIGH": {
        "action":     "Request additional verification",
        "color":      "#E67E22",
        "confidence": "MEDIUM",
    },
    "CRITICAL": {
        "action":     "Block transaction immediately",
        "color":      "#C0392B",
        "confidence": "HIGH",
    },
}


def score_transaction(probability: float) -> RiskScore:
    """
    Convert a fraud probability into a full RiskScore.

    Parameters
    ----------
    probability : float in [0, 1] — model's fraud probability output

    Returns
    -------
    RiskScore dataclass
    """
    probability = float(np.clip(probability, 0.0, 1.0))

    # Convert to 0–100 score with slight nonlinearity
    # (makes high-risk scores more spread out for analyst visibility)
    score = int(round(_calibrate_score(probability) * 100))
    score = max(0, min(100, score))

    tier   = _get_tier(probability)
    config = TIER_CONFIG[tier]

    return RiskScore(
        probability = round(probability, 4),
        score       = score,
        tier        = tier,
        action      = config["action"],
        confidence  = config["confidence"],
        color       = config["color"],
    )


def score_batch(probabilities: np.ndarray) -> list:
    """Score a batch of transactions. Returns list of RiskScore."""
    return [score_transaction(p) for p in probabilities]


def get_risk_distribution(probabilities: np.ndarray) -> Dict:
    """
    Summarize risk tier distribution across a batch.
    Useful for dashboard analytics.
    """
    scores = score_batch(probabilities)
    tiers  = [s.tier for s in scores]
    total  = len(tiers)

    return {
        tier: {
            "count": tiers.count(tier),
            "pct":   round(tiers.count(tier) / total * 100, 2),
        }
        for tier in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    }


def to_dict(risk: RiskScore) -> Dict:
    """Serialize RiskScore to a plain dict (JSON-safe)."""
    return {
        "probability": risk.probability,
        "score":       risk.score,
        "tier":        risk.tier,
        "action":      risk.action,
        "confidence":  risk.confidence,
        "color":       risk.color,
    }


# ─── Private helpers ─────────────────────────────────────────────────────────

def _get_tier(probability: float) -> str:
    for tier, (low, high) in THRESHOLDS.items():
        if low <= probability < high:
            return tier
    return "CRITICAL"


def _calibrate_score(p: float) -> float:
    """
    Slight sigmoid-like calibration so mid-range scores are more
    spread out and extreme scores are more definitive.
    Makes the 0–100 scale more intuitive for analysts.
    """
    # Apply mild power curve
    if p < 0.5:
        return 0.5 * (2 * p) ** 0.85
    else:
        return 1 - 0.5 * (2 * (1 - p)) ** 0.85
