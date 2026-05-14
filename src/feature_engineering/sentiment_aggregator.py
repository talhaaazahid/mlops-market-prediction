"""Aggregate per-article sentiment scores into daily time-series features per ticker.

Reads data/processed/sentiment_labeled.csv, groups by (ticker, date), computes
10 daily sentiment features, forward-fills missing trading days, and writes
data/processed/daily_sentiment_features.csv.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, TICKERS, get_params
from src.utils.logger import get_logger

log = get_logger(__name__)

_SENTIMENT_LABELED_FILE = PROCESSED_DIR / "sentiment_labeled.csv"
_DAILY_SENTIMENT_FILE = PROCESSED_DIR / "daily_sentiment_features.csv"

_REDDIT_SOURCE_PREFIX = "reddit_"
_NEWS_SOURCES = {"reuters_business", "reuters_markets", "marketwatch_top",
                 "marketwatch_pulse", "cnbc", "google_news", "seeking_alpha",
                 "investing_com", "newsdata"}


def _explode_ticker_mentions(df: pd.DataFrame) -> pd.DataFrame:
    """Expand rows so each row represents one (article, ticker) pair.

    Articles with no ticker mentions are assigned to all configured tickers
    (broad-market sentiment).

    Args:
        df: sentiment_labeled.csv DataFrame with a `ticker_mentions` JSON column.

    Returns:
        Expanded DataFrame with a new `ticker` column.
    """
    params = get_params()
    all_tickers = params["ingestion"]["tickers"]

    rows: list[dict] = []
    for _, row in df.iterrows():
        try:
            mentions = json.loads(row.get("ticker_mentions") or "[]")
            mentions = [t for t in mentions if t in all_tickers]
        except (json.JSONDecodeError, TypeError):
            mentions = []

        targets = mentions if mentions else all_tickers
        for ticker in targets:
            rows.append({**row.to_dict(), "ticker": ticker})

    return pd.DataFrame(rows)


def _extract_engagement(row: pd.Series) -> float:
    """Extract engagement weight from metadata_json for weighting sentiment.

    Reddit posts use upvote score; news articles use weight=1.

    Args:
        row: A DataFrame row with `source` and `metadata_json` columns.

    Returns:
        Float engagement weight (minimum 1.0).
    """
    try:
        meta = json.loads(row.get("metadata_json") or "{}")
        if str(row.get("source", "")).startswith(_REDDIT_SOURCE_PREFIX):
            return max(float(meta.get("score", 1)), 1.0)
    except (json.JSONDecodeError, TypeError):
        pass
    return 1.0


def aggregate_daily_sentiment(
    labeled_df: pd.DataFrame | None = None,
    input_path: Path | str = _SENTIMENT_LABELED_FILE,
    output_path: Path | str = _DAILY_SENTIMENT_FILE,
    save: bool = True,
) -> pd.DataFrame:
    """Aggregate article-level sentiment into daily ticker-level features.

    Args:
        labeled_df: Pre-loaded DataFrame. If None, reads from `input_path`.
        input_path: Path to sentiment_labeled.csv.
        output_path: Destination CSV for daily features.
        save: Write output to disk when True.

    Returns:
        DataFrame with columns:
        [date, ticker, daily_sentiment_mean, daily_sentiment_std,
         daily_positive_ratio, daily_negative_ratio, daily_neutral_ratio,
         daily_article_count, daily_reddit_sentiment, daily_news_sentiment,
         sentiment_momentum, weighted_sentiment].
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if labeled_df is None:
        if not input_path.exists():
            raise FileNotFoundError(
                f"Sentiment file not found: {input_path}. "
                "Run the sentiment pipeline first."
            )
        log.info("Reading %s …", input_path)
        labeled_df = pd.read_csv(input_path)

    if labeled_df.empty:
        log.warning("Sentiment DataFrame is empty — returning empty features.")
        return pd.DataFrame()

    # Normalise dates
    labeled_df["published_date"] = pd.to_datetime(
        labeled_df["published_date"], errors="coerce", utc=True
    ).dt.tz_localize(None)
    labeled_df = labeled_df.dropna(subset=["published_date"])
    labeled_df["date"] = labeled_df["published_date"].dt.normalize()

    # Expand to (article, ticker) rows
    log.info("Exploding ticker mentions …")
    expanded = _explode_ticker_mentions(labeled_df)

    if expanded.empty:
        log.warning("No ticker mentions found after expansion.")
        return pd.DataFrame()

    # Engagement weights for weighted_sentiment
    log.info("Computing engagement weights …")
    expanded["engagement"] = expanded.apply(_extract_engagement, axis=1)

    # Source classification
    expanded["is_reddit"] = expanded["source"].str.startswith(_REDDIT_SOURCE_PREFIX)
    expanded["is_news"] = ~expanded["is_reddit"]

    # ── Per-(ticker, date) aggregation ───────────────────────────────────── #
    log.info("Aggregating by (ticker, date) …")

    def _agg(grp: pd.DataFrame) -> pd.Series:
        compound = grp["ensemble_compound"]
        label = grp["ensemble_label"]
        n = len(grp)

        reddit_mask = grp["is_reddit"]
        news_mask = grp["is_news"]

        reddit_mean = (
            grp.loc[reddit_mask, "ensemble_compound"].mean()
            if reddit_mask.any()
            else np.nan
        )
        news_mean = (
            grp.loc[news_mask, "ensemble_compound"].mean()
            if news_mask.any()
            else np.nan
        )

        weights = grp["engagement"]
        w_sum = weights.sum()
        weighted_sent = (
            (compound * weights).sum() / w_sum if w_sum > 0 else 0.0
        )

        return pd.Series(
            {
                "daily_sentiment_mean": compound.mean(),
                "daily_sentiment_std": compound.std(ddof=0),
                "daily_positive_ratio": (label == "positive").sum() / n,
                "daily_negative_ratio": (label == "negative").sum() / n,
                "daily_neutral_ratio": (label == "neutral").sum() / n,
                "daily_article_count": n,
                "daily_reddit_sentiment": reddit_mean,
                "daily_news_sentiment": news_mean,
                "weighted_sentiment": weighted_sent,
            }
        )

    daily = (
        expanded.groupby(["ticker", "date"])
        .apply(_agg)
        .reset_index()
    )

    # ── Sentiment momentum: today's mean − yesterday's mean ─────────────── #
    daily = daily.sort_values(["ticker", "date"])
    daily["sentiment_momentum"] = daily.groupby("ticker")["daily_sentiment_mean"].diff()

    # ── Fill missing trading days ────────────────────────────────────────── #
    # Build a full date range and forward-fill missing days per ticker
    params = get_params()
    tickers = params["ingestion"]["tickers"]
    min_date = daily["date"].min()
    max_date = daily["date"].max()
    full_index = pd.date_range(min_date, max_date, freq="D")

    filled_parts: list[pd.DataFrame] = []
    for ticker in tickers:
        sub = daily[daily["ticker"] == ticker].set_index("date")
        sub = sub.reindex(full_index)
        sub["ticker"] = ticker
        # Forward-fill gaps, then zero-fill any remaining NaN at the start
        sub = sub.ffill().fillna(0.0)
        sub.index.name = "date"
        filled_parts.append(sub.reset_index())

    result = pd.concat(filled_parts, ignore_index=True)
    result = result.sort_values(["ticker", "date"]).reset_index(drop=True)

    if save:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)
        log.info("Saved %d daily sentiment rows to %s", len(result), output_path)

    log.info(
        "Sentiment aggregation complete: %d rows, %d tickers, date range %s → %s",
        len(result),
        result["ticker"].nunique(),
        result["date"].min(),
        result["date"].max(),
    )
    return result


if __name__ == "__main__":
    df = aggregate_daily_sentiment(save=True)
    print(df.head(10).to_string())
    print(f"\nShape: {df.shape}")
