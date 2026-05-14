"""Train all 4 models (RNN, LSTM, GRU, BiLSTM-Attention) on each of 6 tickers.

Produces 24 MLflow runs minimum (4 models × 6 tickers). Loads pre-built
feature CSVs and the global scaler, filters per ticker, builds sliding-window
sequences on the fly, then trains and logs each model.

Usage:
    python scripts/run_training.py
    python scripts/run_training.py --tickers AAPL MSFT
    python scripts/run_training.py --models rnn lstm
    python scripts/run_training.py --target direction
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import FEATURES_DIR, PROCESSED_DIR, get_params
from src.feature_engineering.dataset_builder import _NON_FEATURE_COLS, _sliding_windows
from src.models.bilstm_attention import BiLSTMAttentionModel
from src.models.gru_model import GRUModel
from src.models.lstm_model import LSTMModel
from src.models.rnn_model import RNNModel
from src.models.trainer import Trainer
from src.utils.logger import get_logger

log = get_logger(__name__)

_MODEL_REGISTRY = {
    "rnn": RNNModel,
    "lstm": LSTMModel,
    "gru": GRUModel,
    "bilstm_attention": BiLSTMAttentionModel,
}

_TECHNICAL_FILE = PROCESSED_DIR / "technical_features.csv"
_SENTIMENT_FILE = PROCESSED_DIR / "daily_sentiment_features.csv"
_RESULTS_DIR = Path("results")


def _load_global_artifacts() -> tuple[list[str], Any]:
    """Load feature column names and the fitted scaler from data/features/.

    Returns:
        Tuple of (feature_names, scaler).

    Raises:
        FileNotFoundError: If artefacts are missing (run dataset_builder first).
    """
    names_path = FEATURES_DIR / "feature_names.json"
    scaler_path = FEATURES_DIR / "scaler.pkl"

    if not names_path.exists() or not scaler_path.exists():
        raise FileNotFoundError(
            "Feature artefacts not found. Run scripts/run_ingestion.py, "
            "the sentiment pipeline, then python -m src.feature_engineering.dataset_builder first."
        )

    with open(names_path) as fh:
        feature_names: list[str] = json.load(fh)

    with open(scaler_path, "rb") as fh:
        scaler = pickle.load(fh)

    return feature_names, scaler


def _build_ticker_sequences(
    merged_df: pd.DataFrame,
    ticker: str,
    feature_names: list[str],
    scaler: Any,
    target: str,
    lookback: int,
    train_ratio: float,
    val_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build train/val/test sequences for a single ticker.

    Args:
        merged_df: Combined technical + sentiment DataFrame (all tickers).
        ticker: Ticker symbol to filter.
        feature_names: Ordered list of feature column names.
        scaler: Fitted StandardScaler (from global dataset build).
        target: Target column name.
        lookback: Sliding-window size.
        train_ratio: Fraction for training split.
        val_ratio: Fraction for validation split.

    Returns:
        Tuple (X_train, y_train, X_val, y_val, X_test, y_test).
    """
    sub = merged_df[merged_df["ticker"] == ticker].sort_values("date").reset_index(drop=True)

    if len(sub) < lookback + 10:
        raise ValueError(f"Not enough data for {ticker}: {len(sub)} rows")

    sub[feature_names] = sub[feature_names].fillna(0.0)
    scaled = scaler.transform(sub[feature_names].values).astype(np.float32)
    targets = sub[target].values

    n = len(scaled)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    X_tr, y_tr = _sliding_windows(scaled[:train_end], targets[:train_end], lookback)
    X_val, y_val = _sliding_windows(scaled[train_end:val_end], targets[train_end:val_end], lookback)
    X_te, y_te = _sliding_windows(scaled[val_end:], targets[val_end:], lookback)

    return X_tr, y_tr, X_val, y_val, X_te, y_te


def run(
    tickers: list[str] | None = None,
    model_names: list[str] | None = None,
    target: str = "direction_3class",
) -> pd.DataFrame:
    """Train all model/ticker combinations and return a comparison DataFrame.

    Args:
        tickers: Tickers to train on. Defaults to params.yaml list.
        model_names: Model keys to train. Defaults to all 4.
        target: Target variable to predict.

    Returns:
        DataFrame with one row per (model, ticker) containing test metrics.
    """
    params = get_params()
    if tickers is None:
        tickers = params["ingestion"]["tickers"]
    if model_names is None:
        model_names = params["training"]["models"]

    lookback = int(params["features"]["lookback_days"])
    train_ratio = float(params["features"]["train_ratio"])
    val_ratio = float(params["features"]["val_ratio"])
    hidden_dim = int(params["training"]["hidden_dim"])
    num_layers = int(params["training"]["num_layers"])
    dropout = float(params["training"]["dropout"])

    # Determine output_dim from target
    output_dim = 1 if target == "next_day_return" else (
        3 if target == "direction_3class" else 2
    )

    log.info("═" * 60)
    log.info("TRAINING PIPELINE")
    log.info("Models: %s | Tickers: %s | Target: %s", model_names, tickers, target)
    log.info("═" * 60)

    # Load merged feature data
    log.info("Loading feature data …")
    if not _TECHNICAL_FILE.exists():
        raise FileNotFoundError(f"Technical features not found: {_TECHNICAL_FILE}")

    tech_df = pd.read_csv(_TECHNICAL_FILE, parse_dates=["date"])
    tech_df["date"] = pd.to_datetime(tech_df["date"]).dt.normalize()

    if _SENTIMENT_FILE.exists():
        sent_df = pd.read_csv(_SENTIMENT_FILE, parse_dates=["date"])
        sent_df["date"] = pd.to_datetime(sent_df["date"]).dt.normalize()
        merged = tech_df.merge(sent_df, on=["date", "ticker"], how="left")
    else:
        log.warning("Sentiment features not found — training without sentiment.")
        merged = tech_df.copy()

    # Build targets inline (reuse the logic from dataset_builder)
    from src.feature_engineering.dataset_builder import _build_targets
    merged = (
        merged.groupby("ticker", group_keys=False)
        .apply(_build_targets)
        .reset_index(drop=True)
    )
    merged = (
        merged.groupby("ticker", group_keys=False)
        .apply(lambda g: g.iloc[:-1])
        .reset_index(drop=True)
    )
    merged = merged.dropna(subset=[target]).reset_index(drop=True)

    feature_names, scaler = _load_global_artifacts()
    # Keep only cols that exist in this merged frame
    feature_names = [c for c in feature_names if c in merged.columns]
    input_dim = len(feature_names)

    all_results: list[dict[str, Any]] = []
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    total_runs = len(model_names) * len(tickers)
    run_count = 0

    for ticker in tickers:
        log.info("─" * 50)
        log.info("Ticker: %s", ticker)

        try:
            X_tr, y_tr, X_val, y_val, X_te, y_te = _build_ticker_sequences(
                merged, ticker, feature_names, scaler,
                target, lookback, train_ratio, val_ratio,
            )
        except ValueError as exc:
            log.error("Skipping %s: %s", ticker, exc)
            continue

        log.info(
            "%s sequences — train=%d val=%d test=%d",
            ticker, len(X_tr), len(X_val), len(X_te),
        )

        for model_key in model_names:
            run_count += 1
            log.info("[%d/%d] %s + %s", run_count, total_runs, model_key.upper(), ticker)

            ModelClass = _MODEL_REGISTRY[model_key]
            model = ModelClass(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                output_dim=output_dim,
                dropout=dropout,
            )

            trainer = Trainer(
                model=model,
                target=target,
                ticker=ticker,
                results_dir=_RESULTS_DIR,
            )

            try:
                result = trainer.fit(
                    X_tr, y_tr, X_val, y_val, X_te, y_te,
                    experiment_name="market-pulse-predictor",
                )
                all_results.append(result)
                log.info(
                    "  ✓ %s/%s | val_loss=%.4f best_epoch=%d",
                    model_key, ticker,
                    result["best_val_loss"],
                    result["best_epoch"],
                )
            except Exception as exc:
                log.error("Training failed for %s/%s: %s", model_key, ticker, exc)

    if not all_results:
        log.warning("No results collected — check errors above.")
        return pd.DataFrame()

    comparison_df = pd.DataFrame(all_results)
    comparison_path = _RESULTS_DIR / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    log.info("Model comparison saved to %s", comparison_path)

    # Best model per ticker (by test_accuracy for classification, test_rmse for regression)
    metric = "test_rmse" if target == "next_day_return" else "test_accuracy"
    ascending = target == "next_day_return"

    if metric in comparison_df.columns:
        best_per_ticker = (
            comparison_df.sort_values(metric, ascending=ascending)
            .groupby("ticker")
            .first()
            .reset_index()[["ticker", "model_name", metric]]
        )
        log.info("Best model per ticker:\n%s", best_per_ticker.to_string(index=False))

    # Summary table
    log.info("═" * 60)
    log.info("TRAINING COMPLETE — %d MLflow runs", len(all_results))
    log.info("Results: %s", comparison_path)

    return comparison_df


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train all models on all tickers")
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="Subset of tickers (default: all from params.yaml)")
    parser.add_argument(
        "--models", nargs="+", default=None,
        choices=list(_MODEL_REGISTRY.keys()),
        help="Models to train (default: all 4)",
    )
    parser.add_argument(
        "--target", default="direction_3class",
        choices=["direction", "direction_3class", "volatility_spike", "next_day_return"],
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    results = run(
        tickers=args.tickers,
        model_names=args.models,
        target=args.target,
    )
    if not results.empty:
        print("\n" + results.to_string(index=False))
