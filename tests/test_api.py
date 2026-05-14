"""Tests for the FastAPI application endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ── App fixture ────────────────────────────────────────────────────────────── #


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Return a TestClient with the lifespan initialise() bypassed."""
    with patch("src.api.routes.initialise"):
        from src.api.main import app

        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


# ── Health check ───────────────────────────────────────────────────────────── #


class TestHealthEndpoint:
    def test_health_returns_200(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_has_status_key(self, client: TestClient) -> None:
        data = client.get("/health").json()
        assert "status" in data
        assert data["status"] == "healthy"

    def test_health_response_has_timestamp(self, client: TestClient) -> None:
        data = client.get("/health").json()
        assert "timestamp" in data


# ── Predict endpoint ───────────────────────────────────────────────────────── #


class TestPredictEndpoint:
    def test_predict_returns_non_500(self, client: TestClient) -> None:
        """Without trained models, predict returns 422 or 503, not 500."""
        payload = {"ticker": "AAPL", "model": "lstm"}
        response = client.post("/predict", json=payload)
        assert response.status_code in {200, 422, 503}

    def test_predict_invalid_ticker_returns_422(self, client: TestClient) -> None:
        payload = {"ticker": "FAKE", "model": "lstm"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_invalid_model_returns_422(self, client: TestClient) -> None:
        payload = {"ticker": "AAPL", "model": "transformer"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    @pytest.mark.parametrize("model_name", ["rnn", "lstm", "gru", "bilstm_attention"])
    def test_predict_all_model_names_accepted(
        self, client: TestClient, model_name: str
    ) -> None:
        """All 4 valid model names should pass schema validation (may return 503)."""
        payload = {"ticker": "MSFT", "model": model_name}
        response = client.post("/predict", json=payload)
        # 503 = no checkpoint; 200 = successful; 422 = validation error (unexpected)
        assert response.status_code in {200, 503}


# ── Models listing endpoint ────────────────────────────────────────────────── #


class TestModelsEndpoint:
    def test_models_returns_200(self, client: TestClient) -> None:
        response = client.get("/models")
        assert response.status_code == 200

    def test_models_response_has_models_key(self, client: TestClient) -> None:
        data = client.get("/models").json()
        assert "models" in data
        assert isinstance(data["models"], list)


# ── Sentiment endpoint ─────────────────────────────────────────────────────── #


class TestSentimentEndpoint:
    def test_sentiment_valid_ticker_does_not_crash(self, client: TestClient) -> None:
        """Without data files, sentiment returns 500 but should not crash the server."""
        response = client.get("/sentiment/AAPL")
        assert response.status_code in {200, 500}

    def test_sentiment_unknown_ticker_returns_error(self, client: TestClient) -> None:
        response = client.get("/sentiment/FAKE999")
        assert response.status_code in {422, 500}


# ── Retrain endpoint ───────────────────────────────────────────────────────── #


class TestRetrainEndpoint:
    def test_retrain_returns_accepted(self, client: TestClient) -> None:
        response = client.post("/retrain", json={})
        assert response.status_code == 200

    def test_retrain_response_has_status(self, client: TestClient) -> None:
        data = client.post("/retrain", json={}).json()
        assert "status" in data
        assert data["status"] == "accepted"

    def test_retrain_with_specific_tickers(self, client: TestClient) -> None:
        payload = {"tickers": ["AAPL", "MSFT"], "models": ["lstm"]}
        response = client.post("/retrain", json=payload)
        assert response.status_code == 200


# ── Schema validation ──────────────────────────────────────────────────────── #


class TestSchemas:
    def test_predict_request_valid(self) -> None:
        from src.api.schemas import PredictRequest

        req = PredictRequest(ticker="AAPL", model="lstm")
        assert req.ticker == "AAPL"
        assert req.model == "lstm"

    def test_predict_request_invalid_model(self) -> None:
        from pydantic import ValidationError
        from src.api.schemas import PredictRequest

        with pytest.raises(ValidationError):
            PredictRequest(ticker="AAPL", model="unknown_model")

    def test_predict_request_ticker_uppercased(self) -> None:
        from src.api.schemas import PredictRequest

        req = PredictRequest(ticker="aapl", model="lstm")
        assert req.ticker == "AAPL"

    def test_retrain_request_optional_fields(self) -> None:
        from src.api.schemas import RetrainRequest

        req = RetrainRequest()
        assert req.tickers is None
        assert req.models is None

    def test_retrain_request_with_tickers(self) -> None:
        from src.api.schemas import RetrainRequest

        req = RetrainRequest(tickers=["AAPL", "MSFT"], models=["lstm"])
        assert req.tickers == ["AAPL", "MSFT"]
        assert req.models == ["lstm"]
