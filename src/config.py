"""Centralized configuration: paths, constants, and parameter loading.

All other modules import from here rather than hard-coding paths or magic numbers.
Hyperparameters live in `params.yaml` and are loaded lazily through `get_params()`.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# --------------------------------------------------------------------------- #
# Project paths
# --------------------------------------------------------------------------- #

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
FEATURES_DIR: Path = DATA_DIR / "features"

MODELS_DIR: Path = PROJECT_ROOT / "models"
SAVED_MODELS_DIR: Path = MODELS_DIR / "saved"

RESULTS_DIR: Path = PROJECT_ROOT / "results"
LOGS_DIR: Path = PROJECT_ROOT / "logs"

PARAMS_FILE: Path = PROJECT_ROOT / "params.yaml"

# Output file paths used across the pipeline
PRICE_DATA_FILE: Path = RAW_DIR / "price_data.csv"
PRICE_INTRADAY_FILE: Path = RAW_DIR / "price_data_intraday.csv"
NEWS_FILE: Path = RAW_DIR / "news_articles.csv"
REDDIT_FILE: Path = RAW_DIR / "reddit_posts.csv"
NEWSDATA_FILE: Path = RAW_DIR / "newsdata_articles.csv"
ALL_TEXT_FILE: Path = RAW_DIR / "all_text_data.csv"

SENTIMENT_LABELED_FILE: Path = PROCESSED_DIR / "sentiment_labeled.csv"
DAILY_SENTIMENT_FILE: Path = PROCESSED_DIR / "daily_sentiment_features.csv"
TECHNICAL_FEATURES_FILE: Path = PROCESSED_DIR / "technical_features.csv"

SCALER_FILE: Path = FEATURES_DIR / "scaler.pkl"
FEATURE_NAMES_FILE: Path = FEATURES_DIR / "feature_names.json"
DATASET_METADATA_FILE: Path = FEATURES_DIR / "dataset_metadata.json"

MODEL_COMPARISON_FILE: Path = RESULTS_DIR / "model_comparison.csv"

# HuggingFace model name for FinBERT — used by src/sentiment/finbert_analyzer.py
FINBERT_MODEL: str = "ProsusAI/finbert"

# --------------------------------------------------------------------------- #
# RSS feeds (per spec section 1B)
# --------------------------------------------------------------------------- #

RSS_FEEDS: dict[str, str] = {
    "reuters_business": "https://www.reutersagency.com/feed/?best-topics=business-finance",
    "reuters_markets": "https://www.reutersagency.com/feed/?best-topics=markets",
    "marketwatch_top": "https://feeds.marketwatch.com/marketwatch/topstories",
    "marketwatch_pulse": "https://feeds.marketwatch.com/marketwatch/marketpulse",
    "cnbc_top": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
    "google_news_finance": "https://news.google.com/rss/search?q=stock+market+finance&hl=en-US&gl=US&ceid=US:en",
    "seeking_alpha": "https://seekingalpha.com/market_currents.xml",
    "investing_com": "https://www.investing.com/rss/news.rss",
}

# --------------------------------------------------------------------------- #
# Reddit subreddits with per-sub post limits (spec section 1C)
# --------------------------------------------------------------------------- #

REDDIT_SUBREDDITS: dict[str, int] = {
    "wallstreetbets": 50,
    "stocks": 50,
    "investing": 30,
    "StockMarket": 30,
    "finance": 20,
}

# --------------------------------------------------------------------------- #
# Ticker → company-name aliases for entity-mention extraction.
# Matched case-insensitively against text.
# --------------------------------------------------------------------------- #

TICKER_ALIASES: dict[str, list[str]] = {
    "AAPL": ["apple", "iphone", "tim cook"],
    "MSFT": ["microsoft", "satya nadella", "azure"],
    "GOOGL": ["google", "alphabet", "sundar pichai", "youtube"],
    "AMZN": ["amazon", "andy jassy", "aws"],
    "TSLA": ["tesla", "elon musk"],
    "META": ["meta platforms", "facebook", "instagram", "mark zuckerberg", "whatsapp"],
}

TICKERS: list[str] = list(TICKER_ALIASES.keys())

# Sources we treat as "news" vs "reddit" for source-stratified sentiment features.
NEWS_SOURCES: set[str] = {*RSS_FEEDS.keys(), "newsdata"}
REDDIT_SOURCES: set[str] = {f"reddit_{sub.lower()}" for sub in REDDIT_SUBREDDITS}

# --------------------------------------------------------------------------- #
# Environment / params
# --------------------------------------------------------------------------- #

# Load .env if present; missing file is fine for dev.
load_dotenv(PROJECT_ROOT / ".env", override=False)


@lru_cache(maxsize=1)
def get_params() -> dict[str, Any]:
    """Load and cache `params.yaml`.

    Returns:
        Parsed parameter tree (top-level keys: ingestion, sentiment, features,
        training).

    Raises:
        FileNotFoundError: if params.yaml is missing.
        yaml.YAMLError: if params.yaml is malformed.
    """
    if not PARAMS_FILE.exists():
        raise FileNotFoundError(f"params.yaml not found at {PARAMS_FILE}")
    with PARAMS_FILE.open("r", encoding="utf-8") as fh:
        params: dict[str, Any] = yaml.safe_load(fh)
    return params


def get_env(key: str, default: str | None = None, *, required: bool = False) -> str | None:
    """Read an environment variable.

    Args:
        key: Env var name.
        default: Fallback value if not set.
        required: If True, raise when the var is missing or empty.

    Returns:
        The env var value, or `default`.

    Raises:
        RuntimeError: if `required=True` and the var is missing.
    """
    value = os.environ.get(key, default)
    if required and not value:
        raise RuntimeError(f"Required environment variable {key!r} is not set")
    return value


def ensure_dirs() -> None:
    """Create all standard output directories. Idempotent."""
    for d in (
        RAW_DIR,
        PROCESSED_DIR,
        FEATURES_DIR,
        SAVED_MODELS_DIR,
        RESULTS_DIR,
        LOGS_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # Standalone sanity check
    ensure_dirs()
    p = get_params()
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Tickers: {TICKERS}")
    print(f"RSS feeds: {len(RSS_FEEDS)}")
    print(f"Subreddits: {list(REDDIT_SUBREDDITS)}")
    print(f"Top-level params keys: {list(p.keys())}")

# A/B Testing Configuration
AB_TEST_ENABLED = True
AB_TEST_SPLIT_RATIO = 0.5
AB_TEST_DURATION_DAYS = 7


# A/B Testing Configuration
AB_TEST_ENABLED = True
AB_TEST_SPLIT_RATIO = 0.5
AB_TEST_DURATION_DAYS = 7


# A/B Testing Configuration
AB_TEST_ENABLED = True
AB_TEST_SPLIT_RATIO = 0.5
AB_TEST_DURATION_DAYS = 7


# A/B Testing Configuration
AB_TEST_ENABLED = True
AB_TEST_SPLIT_RATIO = 0.5
AB_TEST_DURATION_DAYS = 7

