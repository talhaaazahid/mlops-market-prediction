"""Unified data ingestion runner.

Orchestrates all four ingestion modules (Yahoo Finance, RSS news, Reddit,
NewsData.io), merges text data into a single unified CSV, logs pipeline
statistics, and is idempotent (running twice does not create duplicates).

Usage:
    python scripts/run_ingestion.py
    python scripts/run_ingestion.py --no-reddit   # skip if no credentials
    python scripts/run_ingestion.py --dry-run     # fetch but don't save
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Allow running from repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (
    ALL_TEXT_FILE,
    NEWSDATA_FILE,
    NEWS_FILE,
    PRICE_DATA_FILE,
    REDDIT_FILE,
    ensure_dirs,
    get_params,
)
from src.data_ingestion import news_rss, newsdata_api, reddit_scraper, yahoo_finance
from src.utils.helpers import append_csv, deduplicate_dataframe
from src.utils.logger import get_logger

log = get_logger(__name__)


# ── Unified text schema ──────────────────────────────────────────────────── #
# columns: [id, source, text, ticker_mentions, published_date, metadata_json]

def _news_to_unified(df: pd.DataFrame) -> pd.DataFrame:
    """Convert news/RSS DataFrame to unified schema."""
    if df.empty:
        return pd.DataFrame(columns=["id", "source", "text", "ticker_mentions",
                                     "published_date", "metadata_json"])
    out = pd.DataFrame()
    out["id"] = df["id"]
    out["source"] = df["source"]
    out["text"] = (df.get("title", "").fillna("") + " " +
                   df.get("summary", "").fillna("")).str.strip()
    out["ticker_mentions"] = df.get("ticker_mentions", "[]")
    out["published_date"] = df.get("published_date")
    meta_cols = ["link"]
    out["metadata_json"] = df.apply(
        lambda row: json.dumps({"link": row.get("link", "")}), axis=1
    )
    return out


def _reddit_to_unified(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Reddit DataFrame to unified schema."""
    if df.empty:
        return pd.DataFrame(columns=["id", "source", "text", "ticker_mentions",
                                     "published_date", "metadata_json"])
    out = pd.DataFrame()
    out["id"] = df["post_id"]
    out["source"] = "reddit_" + df["subreddit"].str.lower()
    out["text"] = df.get("full_text", "").fillna("")
    out["ticker_mentions"] = df.get("ticker_mentions", "[]")
    out["published_date"] = df.get("created_utc")
    out["metadata_json"] = df.apply(
        lambda row: json.dumps({
            "subreddit": row.get("subreddit", ""),
            "score": int(row.get("score", 0)),
            "num_comments": int(row.get("num_comments", 0)),
            "permalink": row.get("permalink", ""),
        }),
        axis=1,
    )
    return out


def _log_stats(all_text: pd.DataFrame, price: pd.DataFrame) -> None:
    """Log ingestion statistics."""
    log.info("─" * 60)
    log.info("INGESTION STATISTICS")
    log.info("─" * 60)
    log.info("Total text records : %d", len(all_text))
    if not all_text.empty:
        for src, grp in all_text.groupby("source"):
            log.info("  %-35s %d articles", src, len(grp))
        dates = pd.to_datetime(all_text["published_date"], errors="coerce")
        valid = dates.dropna()
        if not valid.empty:
            log.info("Date range         : %s → %s", valid.min().date(), valid.max().date())

        # Ticker distribution
        from collections import Counter
        ticker_counts: Counter = Counter()
        for mentions_json in all_text["ticker_mentions"].dropna():
            try:
                mentions = json.loads(mentions_json)
                ticker_counts.update(mentions)
            except (json.JSONDecodeError, TypeError):
                pass
        if ticker_counts:
            log.info("Ticker mentions    : %s", dict(ticker_counts.most_common(10)))

    log.info("Price rows         : %d", len(price))
    log.info("─" * 60)


def run(
    skip_reddit: bool = False,
    skip_newsdata: bool = False,
    dry_run: bool = False,
) -> dict[str, pd.DataFrame]:
    """Full ingestion pipeline.

    Args:
        skip_reddit: Do not attempt Reddit scraping.
        skip_newsdata: Do not attempt NewsData.io fetch.
        dry_run: Fetch everything but skip all CSV writes.

    Returns:
        Dict with keys: "price", "news", "reddit", "newsdata", "all_text".
    """
    ensure_dirs()
    params = get_params()
    save = not dry_run

    log.info("═" * 60)
    log.info("MARKET PULSE DATA INGESTION")
    log.info("═" * 60)

    # 1. Price data
    log.info("Step 1/4 — Yahoo Finance price data")
    daily_df, _ = yahoo_finance.run(save=save)

    # 2. RSS news
    log.info("Step 2/4 — RSS news feeds")
    news_df = news_rss.fetch_latest_news(save=save)

    # 3. Reddit
    reddit_df = pd.DataFrame()
    if not skip_reddit:
        log.info("Step 3/4 — Reddit posts")
        reddit_df = reddit_scraper.fetch_latest_reddit(save=save)
    else:
        log.info("Step 3/4 — Reddit skipped")

    # 4. NewsData.io
    newsdata_df = pd.DataFrame()
    if not skip_newsdata:
        log.info("Step 4/4 — NewsData.io articles")
        newsdata_df = newsdata_api.fetch_latest_newsdata(save=save)
    else:
        log.info("Step 4/4 — NewsData.io skipped")

    # Unify all text data
    log.info("Unifying text data …")
    parts = [
        _news_to_unified(news_df),
        _reddit_to_unified(reddit_df),
        _news_to_unified(newsdata_df) if not newsdata_df.empty else pd.DataFrame(),
    ]
    parts = [p for p in parts if not p.empty]

    if parts:
        all_text = pd.concat(parts, ignore_index=True)
        all_text = deduplicate_dataframe(all_text, key_col="id")
    else:
        all_text = pd.DataFrame(
            columns=["id", "source", "text", "ticker_mentions", "published_date", "metadata_json"]
        )

    if save and not all_text.empty:
        ALL_TEXT_FILE.parent.mkdir(parents=True, exist_ok=True)
        # Idempotent: load existing and re-dedup
        if ALL_TEXT_FILE.exists():
            try:
                existing = pd.read_csv(ALL_TEXT_FILE)
                merged = pd.concat([existing, all_text], ignore_index=True)
                merged = deduplicate_dataframe(merged, key_col="id")
            except pd.errors.EmptyDataError:
                merged = all_text
        else:
            merged = all_text
        merged.to_csv(ALL_TEXT_FILE, index=False)
        log.info("Wrote %d rows to %s", len(merged), ALL_TEXT_FILE)

    _log_stats(all_text, daily_df)
    log.info("═" * 60)
    log.info("Ingestion complete.")

    return {
        "price": daily_df,
        "news": news_df,
        "reddit": reddit_df,
        "newsdata": newsdata_df,
        "all_text": all_text,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Market Pulse data ingestion pipeline")
    parser.add_argument("--no-reddit", dest="skip_reddit", action="store_true",
                        help="Skip Reddit scraping (use if no credentials)")
    parser.add_argument("--no-newsdata", dest="skip_newsdata", action="store_true",
                        help="Skip NewsData.io fetch (use if no API key)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Fetch but do not write any CSVs")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    results = run(
        skip_reddit=args.skip_reddit,
        skip_newsdata=args.skip_newsdata,
        dry_run=args.dry_run,
    )
    print(f"\nDone. all_text rows: {len(results['all_text'])}, "
          f"price rows: {len(results['price'])}")
