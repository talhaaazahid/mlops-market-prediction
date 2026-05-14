"""End-to-end pipeline execution script.

Runs the full pipeline in order:
  1. Data ingestion  (price + news + reddit + newsdata)
  2. Sentiment analysis
  3. Feature engineering (technical + sentiment aggregation + dataset build)
  4. (Optional) Model training

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --skip-training
    python scripts/run_pipeline.py --no-reddit --no-newsdata
    python scripts/run_pipeline.py --tickers AAPL MSFT --target direction
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ensure_dirs, get_params
from src.utils.logger import get_logger

log = get_logger(__name__)


def run(
    skip_training: bool = False,
    skip_reddit: bool = False,
    skip_newsdata: bool = False,
    tickers: list[str] | None = None,
    model_names: list[str] | None = None,
    target: str = "direction_3class",
) -> None:
    """Execute the full ML pipeline from ingestion to training.

    Args:
        skip_training: Skip model training when True (useful for data refresh).
        skip_reddit: Skip Reddit ingestion.
        skip_newsdata: Skip NewsData.io ingestion.
        tickers: Override ticker list.
        model_names: Override model list for training.
        target: Target variable for training.
    """
    ensure_dirs()
    t0 = time.time()

    log.info("╔" + "═" * 58 + "╗")
    log.info("║  MARKET PULSE PREDICTOR — FULL PIPELINE                 ║")
    log.info("╚" + "═" * 58 + "╝")

    # ── Step 1: Data Ingestion ────────────────────────────────────────────── #
    log.info("\n[1/4] Data Ingestion")
    from scripts.run_ingestion import run as run_ingestion

    ingestion_results = run_ingestion(
        skip_reddit=skip_reddit,
        skip_newsdata=skip_newsdata,
    )
    log.info(
        "      ✓ price rows=%d  text rows=%d",
        len(ingestion_results["price"]),
        len(ingestion_results["all_text"]),
    )

    # ── Step 2: Sentiment Analysis ────────────────────────────────────────── #
    log.info("\n[2/4] Sentiment Analysis")
    from src.sentiment.ensemble import run_sentiment_pipeline

    labeled_df = run_sentiment_pipeline()
    log.info("      ✓ labeled rows=%d", len(labeled_df))

    # ── Step 3: Feature Engineering ───────────────────────────────────────── #
    log.info("\n[3/4] Feature Engineering")

    from src.feature_engineering.technical_indicators import compute_all_indicators
    from src.feature_engineering.sentiment_aggregator import aggregate_daily_sentiment
    from src.feature_engineering.dataset_builder import build_dataset

    tech_df = compute_all_indicators(save=True)
    log.info("      ✓ technical features: %d rows", len(tech_df))

    sent_agg_df = aggregate_daily_sentiment(save=True)
    log.info("      ✓ sentiment features: %d rows", len(sent_agg_df))

    dataset = build_dataset(
        technical_df=tech_df,
        sentiment_df=sent_agg_df,
        target=target,
        save=True,
    )
    log.info(
        "      ✓ X_train=%s  X_val=%s  X_test=%s",
        dataset["X_train"].shape,
        dataset["X_val"].shape,
        dataset["X_test"].shape,
    )

    # ── Step 4: Model Training ────────────────────────────────────────────── #
    if skip_training:
        log.info("\n[4/4] Training — SKIPPED (--skip-training flag)")
    else:
        log.info("\n[4/4] Model Training")
        from scripts.run_training import run as run_training

        results_df = run_training(
            tickers=tickers,
            model_names=model_names,
            target=target,
        )
        if not results_df.empty:
            log.info("      ✓ completed %d training runs", len(results_df))
            # Print summary table
            cols = ["model_name", "ticker"]
            metric_cols = [c for c in results_df.columns if c.startswith("test_") and
                           isinstance(results_df[cols[0]].iloc[0], str)]
            if metric_cols:
                print("\n" + results_df[cols + metric_cols[:3]].to_string(index=False))

    elapsed = time.time() - t0
    log.info(
        "\n╔" + "═" * 58 + "╗"
        "\n║  Pipeline complete in %.1f s%s║"
        "\n╚" + "═" * 58 + "╝",
        elapsed,
        " " * max(0, 36 - len(f"{elapsed:.1f}")),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full Market Pulse pipeline")
    parser.add_argument("--skip-training", action="store_true",
                        help="Run ingestion + features only; skip model training")
    parser.add_argument("--no-reddit", dest="skip_reddit", action="store_true",
                        help="Skip Reddit scraping")
    parser.add_argument("--no-newsdata", dest="skip_newsdata", action="store_true",
                        help="Skip NewsData.io fetch")
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="Override ticker list for training")
    parser.add_argument(
        "--models", nargs="+", default=None,
        choices=["rnn", "lstm", "gru", "bilstm_attention"],
        help="Override model list for training",
    )
    parser.add_argument(
        "--target", default="direction_3class",
        choices=["direction", "direction_3class", "volatility_spike", "next_day_return"],
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        skip_training=args.skip_training,
        skip_reddit=args.skip_reddit,
        skip_newsdata=args.skip_newsdata,
        tickers=args.tickers,
        model_names=args.models,
        target=args.target,
    )
