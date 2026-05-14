"""Pydantic request and response schemas for the Market Pulse Predictor API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, List, Optional

from pydantic import BaseModel, Field, field_validator


# ── Request schemas ───────────────────────────────────────────────────────── #

class PredictRequest(BaseModel):
    """Request body for POST /predict."""

    ticker: str = Field(..., description="Stock ticker symbol (e.g. AAPL)")
    model: str = Field(
        default="bilstm_attention",
        description="Model to use: rnn | lstm | gru | bilstm_attention",
    )

    @field_validator("ticker")
    @classmethod
    def uppercase_ticker(cls, v: str) -> str:
        return v.strip().upper()

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        allowed = {"rnn", "lstm", "gru", "bilstm_attention"}
        v = v.strip().lower()
        if v not in allowed:
            raise ValueError(f"model must be one of {allowed}")
        return v


class RetrainRequest(BaseModel):
    """Request body for POST /retrain."""

    tickers: Optional[List[str]] = Field(
        default=None,
        description="Tickers to retrain. If None, all configured tickers are used.",
    )
    models: Optional[List[str]] = Field(
        default=None,
        description="Model types to retrain. If None, all 4 models are used.",
    )
    target: str = Field(
        default="direction_3class",
        description="Target variable: direction | direction_3class | volatility_spike | next_day_return",
    )


# ── Response sub-schemas ──────────────────────────────────────────────────── #

class PredictionDetail(BaseModel):
    """Prediction details nested inside PredictResponse."""

    direction: str = Field(..., description="Predicted direction: up | down | neutral")
    confidence: float = Field(..., description="Softmax confidence of top class [0, 1]")
    predicted_return: Optional[float] = Field(
        default=None, description="Predicted next-day return (regression head, if available)"
    )
    volatility_spike_risk: Optional[float] = Field(
        default=None, description="Probability of a volatility spike (if available)"
    )


class SentimentSummary(BaseModel):
    """Latest sentiment summary nested inside PredictResponse."""

    mean_score: float = Field(..., description="Mean ensemble compound score [-1, 1]")
    article_count: int = Field(..., description="Number of articles used")
    dominant_sentiment: str = Field(..., description="positive | negative | neutral")


class PredictResponse(BaseModel):
    """Response body for POST /predict."""

    ticker: str
    model_used: str
    prediction: PredictionDetail
    latest_sentiment: SentimentSummary
    timestamp: str


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str
    models_loaded: list[str]
    timestamp: str


class ModelInfo(BaseModel):
    """Per-model metadata for GET /models."""

    model_name: str
    ticker: str
    checkpoint_exists: bool
    test_accuracy: Optional[float] = None
    test_f1_macro: Optional[float] = None
    test_rmse: Optional[float] = None
    mlflow_run_id: Optional[str] = None


class ModelsResponse(BaseModel):
    """Response body for GET /models."""

    models: list[ModelInfo]
    total: int


class SentimentResponse(BaseModel):
    """Response body for GET /sentiment/{ticker}."""

    ticker: str
    mean_compound: float
    std_compound: float
    positive_ratio: float
    negative_ratio: float
    neutral_ratio: float
    article_count: int
    dominant_sentiment: str
    last_updated: Optional[str]


class RetrainResponse(BaseModel):
    """Response body for POST /retrain."""

    status: str
    message: str
    job_id: str

class PredictionResponse(BaseModel):
    confidence: float
    prediction_interval: float
    model_version: str
    timestamp: datetime


class PredictionResponse(BaseModel):
    confidence: float
    prediction_interval: float
    model_version: str
    timestamp: datetime

