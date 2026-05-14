"""FastAPI route handlers for all API endpoints."""

from __future__ import annotations

import json
import os
import pickle
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from fastapi import APIRouter, BackgroundTasks, HTTPException

from src.api.schemas import (
    HealthResponse,
    ModelInfo,
    ModelsResponse,
    PredictRequest,
    PredictResponse,
    PredictionDetail,
    RetrainRequest,
    RetrainResponse,
    SentimentResponse,
    SentimentSummary,
)
from src.config import FEATURES_DIR, MODELS_DIR, PROCESSED_DIR, TICKERS, get_params
from src.models.bilstm_attention import BiLSTMAttentionModel
from src.models.gru_model import GRUModel
from src.models.lstm_model import LSTMModel
from src.models.rnn_model import RNNModel
from src.utils.logger import get_logger

log = get_logger(__name__)

router = APIRouter()

_MODEL_CLASSES = {
    "rnn": RNNModel,
    "lstm": LSTMModel,
    "gru": GRUModel,
    "bilstm_attention": BiLSTMAttentionModel,
}

_DIRECTION_MAP = {0: "down", 1: "up"}   # binary direction (output_dim=2)
_DIRECTION_BINARY = {0: "down", 1: "up"}

_loaded_models: dict[str, Any] = {}   # key: "{model_name}_{ticker}"
_scaler: Any = None
_feature_names: list[str] = []


# ── Internal helpers ──────────────────────────────────────────────────────── #

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_scaler_and_features() -> None:
    """Load the global scaler and feature name list (once at startup).

    Falls back to downloading from HuggingFace Hub if local files are missing
    and the HF_MODEL_REPO environment variable is set.
    """
    global _scaler, _feature_names

    scaler_path = FEATURES_DIR / "scaler.pkl"
    names_path = FEATURES_DIR / "feature_names.json"

    hf_repo = os.getenv("HF_MODEL_REPO", "").strip()

    # Scaler
    if not scaler_path.exists() and hf_repo:
        try:
            from huggingface_hub import hf_hub_download
            log.info("Downloading scaler.pkl from HF Hub: %s", hf_repo)
            scaler_path = Path(hf_hub_download(repo_id=hf_repo, filename="features/scaler.pkl"))
        except Exception as exc:
            log.warning("HF Hub scaler download failed: %s", exc)

    if scaler_path.exists():
        with open(scaler_path, "rb") as fh:
            _scaler = pickle.load(fh)
        log.info("Scaler loaded from %s", scaler_path)
    else:
        log.warning("Scaler not found at %s — predictions will fail", scaler_path)

    # Feature names
    if not names_path.exists() and hf_repo:
        try:
            from huggingface_hub import hf_hub_download
            log.info("Downloading feature_names.json from HF Hub: %s", hf_repo)
            names_path = Path(hf_hub_download(repo_id=hf_repo, filename="features/feature_names.json"))
        except Exception as exc:
            log.warning("HF Hub feature_names download failed: %s", exc)

    if names_path.exists():
        with open(names_path) as fh:
            _feature_names = json.load(fh)
        log.info("Feature names loaded: %d features", len(_feature_names))
    else:
        log.warning("feature_names.json not found — predictions will fail")


def _load_model(model_name: str, ticker: str) -> Any | None:
    """Load a model checkpoint lazily and cache it.

    Args:
        model_name: One of rnn | lstm | gru | bilstm_attention.
        ticker: Ticker symbol.

    Returns:
        Loaded PyTorch model in eval mode, or None if checkpoint missing.
    """
    key = f"{model_name}_{ticker}"
    if key in _loaded_models:
        return _loaded_models[key]

    ckpt_path = MODELS_DIR / f"{model_name}_{ticker}.pt"
    if not ckpt_path.exists():
        # Fall back to HuggingFace Hub when running on a fresh deployment
        # (e.g. HF Spaces) where the checkpoints aren't bundled in the image.
        hf_repo = os.getenv("HF_MODEL_REPO", "").strip()
        if hf_repo:
            try:
                from huggingface_hub import hf_hub_download
                log.info("Local checkpoint missing — downloading %s/%s from %s",
                         model_name, ticker, hf_repo)
                ckpt_path = Path(hf_hub_download(
                    repo_id=hf_repo,
                    filename=f"checkpoints/{model_name}_{ticker}.pt",
                ))
            except Exception as exc:
                log.warning("HF Hub download failed for %s/%s: %s",
                            model_name, ticker, exc)
                return None
        else:
            log.warning("Checkpoint not found: %s", ckpt_path)
            return None

    params = get_params()
    output_dim = 2   # binary direction (down=0, up=1)
    ModelClass = _MODEL_CLASSES.get(model_name)
    if ModelClass is None:
        return None

    model = ModelClass(
        input_dim=len(_feature_names),
        hidden_dim=int(params["training"]["hidden_dim"]),
        num_layers=int(params["training"]["num_layers"]),
        output_dim=output_dim,
        dropout=float(params["training"]["dropout"]),
    )
    try:
        model.load_state_dict(
            torch.load(ckpt_path, map_location=torch.device("cpu"))
        )
        model.eval()
        _loaded_models[key] = model
        log.info("Loaded model: %s", key)
    except Exception as exc:
        log.error("Failed to load %s: %s", ckpt_path, exc)
        return None

    return model


def _build_inference_window(ticker: str) -> np.ndarray | None:
    """Construct the most recent 30-day feature window for a ticker.

    Reads technical_features.csv and daily_sentiment_features.csv, merges them,
    filters to the last `lookback` rows for `ticker`, scales, and returns a
    (1, lookback, F) NumPy array ready for model inference.

    Args:
        ticker: Ticker symbol.

    Returns:
        Float32 array of shape (1, lookback, F), or None if data unavailable.
    """
    if _scaler is None or not _feature_names:
        log.error("Scaler / feature names not loaded — cannot build inference window")
        return None

    params = get_params()
    lookback = int(params["features"]["lookback_days"])

    tech_path = PROCESSED_DIR / "technical_features.csv"
    sent_path = PROCESSED_DIR / "daily_sentiment_features.csv"

    if not tech_path.exists():
        log.error("Technical features not found: %s", tech_path)
        return None

    try:
        tech_df = pd.read_csv(tech_path, parse_dates=["date"])
        tech_df = tech_df[tech_df["ticker"] == ticker].sort_values("date")

        if sent_path.exists():
            sent_df = pd.read_csv(sent_path, parse_dates=["date"])
            sent_df = sent_df[sent_df["ticker"] == ticker]
            merged = tech_df.merge(sent_df, on=["date", "ticker"], how="left")
        else:
            merged = tech_df

        available_features = [c for c in _feature_names if c in merged.columns]
        if len(available_features) < len(_feature_names):
            log.warning(
                "Missing %d features — padding with zeros",
                len(_feature_names) - len(available_features),
            )

        merged = merged.tail(lookback)
        if len(merged) < lookback:
            log.warning(
                "%s: only %d rows available (need %d)", ticker, len(merged), lookback
            )

        # Build full matrix, zero-filling missing feature columns
        matrix = np.zeros((len(merged), len(_feature_names)), dtype=np.float32)
        for i, feat in enumerate(_feature_names):
            if feat in merged.columns:
                matrix[:, i] = merged[feat].fillna(0.0).values.astype(np.float32)

        scaled = _scaler.transform(matrix).astype(np.float32)

        # Pad to lookback at the front if not enough history
        if scaled.shape[0] < lookback:
            pad = np.zeros((lookback - scaled.shape[0], scaled.shape[1]), dtype=np.float32)
            scaled = np.vstack([pad, scaled])

        return scaled[np.newaxis, ...]  # (1, lookback, F)

    except Exception as exc:
        log.error("Error building inference window for %s: %s", ticker, exc)
        return None


def _get_latest_sentiment(ticker: str) -> dict[str, Any]:
    """Read the most recent daily sentiment row for a ticker.

    Args:
        ticker: Ticker symbol.

    Returns:
        Dict with mean_score, article_count, dominant_sentiment.
    """
    sent_path = PROCESSED_DIR / "daily_sentiment_features.csv"
    defaults = {"mean_score": 0.0, "article_count": 0, "dominant_sentiment": "neutral"}

    if not sent_path.exists():
        return defaults

    try:
        df = pd.read_csv(sent_path)
        sub = df[df["ticker"] == ticker].tail(1)
        if sub.empty:
            return defaults

        row = sub.iloc[0]
        mean_score = float(row.get("daily_sentiment_mean", 0.0))
        article_count = int(row.get("daily_article_count", 0))
        pos = float(row.get("daily_positive_ratio", 0.0))
        neg = float(row.get("daily_negative_ratio", 0.0))

        if pos > neg and pos > (1 - pos - neg):
            dominant = "positive"
        elif neg > pos:
            dominant = "negative"
        else:
            dominant = "neutral"

        return {
            "mean_score": round(mean_score, 4),
            "article_count": article_count,
            "dominant_sentiment": dominant,
        }
    except Exception as exc:
        log.warning("Could not read sentiment for %s: %s", ticker, exc)
        return defaults


def _pick_best_model(ticker: str) -> str:
    """Select the best available model for a ticker from results/model_comparison.csv.

    Falls back to 'bilstm_attention' if the results file is absent.

    Args:
        ticker: Ticker symbol.

    Returns:
        Model name string.
    """
    results_path = Path("results") / "model_comparison.csv"
    if not results_path.exists():
        return "bilstm_attention"

    try:
        df = pd.read_csv(results_path)
        sub = df[df["ticker"] == ticker]
        if sub.empty or "test_accuracy" not in sub.columns:
            return "bilstm_attention"
        best_row = sub.loc[sub["test_accuracy"].idxmax()]
        return str(best_row["model_name"])
    except Exception:
        return "bilstm_attention"


def _list_all_checkpoints() -> list[str]:
    """Return names of all available model checkpoints."""
    if not MODELS_DIR.exists():
        return []
    return [p.stem for p in MODELS_DIR.glob("*.pt")]


# ── Route handlers ────────────────────────────────────────────────────────── #

@router.get("/health", response_model=HealthResponse, tags=["status"])
async def health() -> HealthResponse:
    """Return API health status and list of loaded model checkpoints."""
    return HealthResponse(
        status="healthy",
        models_loaded=_list_all_checkpoints(),
        timestamp=_now_iso(),
    )


@router.post("/predict", response_model=PredictResponse, tags=["inference"])
async def predict(request: PredictRequest) -> PredictResponse:
    """Run inference for a ticker using the specified (or best) model.

    Fetches the latest feature window from saved CSVs, scales it, and
    runs a forward pass through the requested model checkpoint.
    """
    ticker = request.ticker
    model_name = request.model

    if ticker not in TICKERS:
        raise HTTPException(
            status_code=422,
            detail=f"Ticker '{ticker}' not supported. Choose from {TICKERS}.",
        )

    # Resolve "best" model automatically if not explicitly requested
    effective_model = model_name if model_name else _pick_best_model(ticker)

    model = _load_model(effective_model, ticker)
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"No trained checkpoint found for {effective_model}/{ticker}. "
                "Run scripts/run_training.py first."
            ),
        )

    window = _build_inference_window(ticker)
    if window is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Feature data unavailable for {ticker}. "
                "Run the ingestion and feature engineering pipelines first."
            ),
        )

    # Inference
    with torch.no_grad():
        x = torch.tensor(window, dtype=torch.float32)
        logits = model(x)                        # (1, output_dim)
        probs = torch.softmax(logits, dim=-1)[0].numpy()

    top_class = int(np.argmax(probs))
    confidence = float(probs[top_class])

    # Map class index to direction string for binary classification
    direction = _DIRECTION_MAP.get(top_class, "down")

    sentiment = _get_latest_sentiment(ticker)

    return PredictResponse(
        ticker=ticker,
        model_used=effective_model,
        prediction=PredictionDetail(
            direction=direction,
            confidence=round(confidence, 4),
            predicted_return=None,      # populated when regression model runs
            volatility_spike_risk=None,
        ),
        latest_sentiment=SentimentSummary(
            mean_score=sentiment["mean_score"],
            article_count=sentiment["article_count"],
            dominant_sentiment=sentiment["dominant_sentiment"],
        ),
        timestamp=_now_iso(),
    )


@router.get("/models", response_model=ModelsResponse, tags=["models"])
async def list_models() -> ModelsResponse:
    """List all available model checkpoints with their performance metrics."""
    params = get_params()
    all_tickers = params["ingestion"]["tickers"]
    all_models = ["rnn", "lstm", "gru", "bilstm_attention"]

    results_path = Path("results") / "model_comparison.csv"
    results_df = pd.DataFrame()
    if results_path.exists():
        try:
            results_df = pd.read_csv(results_path)
        except Exception:
            pass

    model_infos: list[ModelInfo] = []

    # When deployed to HF Spaces the checkpoints aren't bundled locally — the
    # API lazy-downloads them from HF Hub on first /predict. Report them as
    # available so the dashboard doesn't show a misleading empty checkbox.
    hf_repo_configured = bool(os.getenv("HF_MODEL_REPO", "").strip())

    for m in all_models:
        for t in all_tickers:
            ckpt = MODELS_DIR / f"{m}_{t}.pt"
            info = ModelInfo(
                model_name=m,
                ticker=t,
                checkpoint_exists=ckpt.exists() or hf_repo_configured,
            )

            if not results_df.empty:
                row = results_df[
                    (results_df["model_name"] == m) & (results_df["ticker"] == t)
                ]
                if not row.empty:
                    r = row.iloc[0]
                    info.test_accuracy = float(r["test_accuracy"]) if "test_accuracy" in r else None
                    info.test_f1_macro = float(r["test_f1_macro"]) if "test_f1_macro" in r else None
                    info.test_rmse = float(r["test_rmse"]) if "test_rmse" in r else None
                    info.mlflow_run_id = str(r["mlflow_run_id"]) if "mlflow_run_id" in r else None

            model_infos.append(info)

    return ModelsResponse(models=model_infos, total=len(model_infos))


@router.get("/sentiment/{ticker}", response_model=SentimentResponse, tags=["sentiment"])
async def get_sentiment(ticker: str) -> SentimentResponse:
    """Return the most recent daily sentiment aggregation for a ticker."""
    ticker = ticker.strip().upper()
    if ticker not in TICKERS:
        raise HTTPException(
            status_code=422,
            detail=f"Ticker '{ticker}' not supported. Choose from {TICKERS}.",
        )

    sent_path = PROCESSED_DIR / "daily_sentiment_features.csv"
    if not sent_path.exists():
        raise HTTPException(
            status_code=503,
            detail="Sentiment data not available. Run the sentiment pipeline first.",
        )

    try:
        df = pd.read_csv(sent_path)
        sub = df[df["ticker"] == ticker].sort_values("date")
        if sub.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No sentiment data found for ticker '{ticker}'.",
            )

        row = sub.iloc[-1]
        mean_c = float(row.get("daily_sentiment_mean", 0.0))
        std_c = float(row.get("daily_sentiment_std", 0.0))
        pos_r = float(row.get("daily_positive_ratio", 0.0))
        neg_r = float(row.get("daily_negative_ratio", 0.0))
        neu_r = float(row.get("daily_neutral_ratio", 0.0))
        n_art = int(row.get("daily_article_count", 0))
        last_dt = str(row.get("date", ""))

        dominant = "positive" if pos_r >= neg_r and pos_r >= neu_r else (
            "negative" if neg_r >= neu_r else "neutral"
        )

        return SentimentResponse(
            ticker=ticker,
            mean_compound=round(mean_c, 4),
            std_compound=round(std_c, 4),
            positive_ratio=round(pos_r, 4),
            negative_ratio=round(neg_r, 4),
            neutral_ratio=round(neu_r, 4),
            article_count=n_art,
            dominant_sentiment=dominant,
            last_updated=last_dt,
        )
    except HTTPException:
        raise
    except Exception as exc:
        log.error("Error reading sentiment for %s: %s", ticker, exc)
        raise HTTPException(status_code=500, detail="Internal error reading sentiment data.")


def _retrain_background(tickers: list[str] | None, models: list[str] | None, target: str) -> None:
    """Background task that runs the full training pipeline."""
    try:
        from scripts.run_training import run as run_training
        log.info("Background retrain started — tickers=%s models=%s target=%s",
                 tickers, models, target)
        run_training(tickers=tickers, model_names=models, target=target)
        log.info("Background retrain completed.")
    except Exception as exc:
        log.error("Background retrain failed: %s", exc)


@router.post("/retrain", response_model=RetrainResponse, tags=["training"])
async def retrain(request: RetrainRequest, background_tasks: BackgroundTasks) -> RetrainResponse:
    """Trigger model retraining as a background task.

    Returns immediately with a job ID; training runs asynchronously.
    """
    job_id = str(uuid.uuid4())
    background_tasks.add_task(
        _retrain_background,
        tickers=request.tickers,
        models=request.models,
        target=request.target,
    )
    log.info("Retrain job %s queued.", job_id)
    return RetrainResponse(
        status="accepted",
        message=(
            f"Retraining started for tickers={request.tickers or 'all'}, "
            f"models={request.models or 'all'}, target={request.target}."
        ),
        job_id=job_id,
    )


# ── Startup initialisation (called from main.py lifespan) ────────────────── #

def initialise() -> None:
    """Load scaler and feature names; eagerly load any existing checkpoints."""
    _load_scaler_and_features()
    # Eagerly cache all available checkpoints into memory
    for ckpt in MODELS_DIR.glob("*.pt") if MODELS_DIR.exists() else []:
        parts = ckpt.stem.rsplit("_", 1)
        if len(parts) == 2:
            m_name, tick = parts
            if m_name in _MODEL_CLASSES:
                _load_model(m_name, tick)
