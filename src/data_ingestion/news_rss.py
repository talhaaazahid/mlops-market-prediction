"""Financial news ingestion from RSS feeds using feedparser.

Fetches 8 finance RSS sources, extracts structured article data, cleans HTML,
extracts ticker mentions, deduplicates, and appends to data/raw/news_articles.csv.
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import feedparser
from bs4 import BeautifulSoup

from src.config import NEWS_FILE, RSS_FEEDS, get_params
from src.utils.helpers import (
    append_csv,
    deduplicate_dataframe,
    extract_ticker_mentions,
    hash_text,
    to_utc,
)
from src.utils.logger import get_logger

import pandas as pd

log = get_logger(__name__)

# Regex to strip leftover HTML entities not caught by BS4
_HTML_ENTITY_RE = re.compile(r"&[a-zA-Z]+;|&#\d+;")


def _clean_html(text: str | None) -> str:
    """Strip HTML tags and normalise whitespace.

    Args:
        text: Raw HTML or plain text.

    Returns:
        Plain text with normalised whitespace.
    """
    if not text:
        return ""
    cleaned = BeautifulSoup(text, "lxml").get_text(separator=" ")
    cleaned = _HTML_ENTITY_RE.sub(" ", cleaned)
    return " ".join(cleaned.split())


def _parse_date(entry: Any) -> str | None:
    """Extract a UTC ISO timestamp from a feedparser entry.

    Tries `published_parsed`, then `updated_parsed`, then raw `published`.

    Args:
        entry: A `feedparser.FeedParserDict` entry.

    Returns:
        ISO-8601 UTC string or None.
    """
    # feedparser provides `*_parsed` as a `time.struct_time` in UTC
    for field in ("published_parsed", "updated_parsed"):
        t = getattr(entry, field, None)
        if t:
            try:
                dt = datetime(*t[:6], tzinfo=timezone.utc)
                return dt.isoformat()
            except (ValueError, TypeError):
                pass

    # Fall back to raw string fields
    for field in ("published", "updated"):
        raw = getattr(entry, field, None)
        if raw:
            try:
                dt = parsedate_to_datetime(raw).astimezone(timezone.utc)
                return dt.isoformat()
            except Exception:
                pass
    return None


def fetch_feed(name: str, url: str, delay: float = 1.0) -> list[dict[str, Any]]:
    """Download and parse a single RSS feed.

    Args:
        name: Human-readable feed name (used as `source` field).
        url: RSS feed URL.
        delay: Seconds to sleep after fetching (rate-limiting).

    Returns:
        List of article dicts with keys:
        [id, source, title, summary, published_date, link, ticker_mentions, metadata_json].
    """
    articles: list[dict[str, Any]] = []

    try:
        log.info("Fetching RSS feed: %s", name)
        feed = feedparser.parse(url)

        if feed.get("bozo") and not feed.entries:
            log.warning("%s: feed parse error — %s", name, feed.get("bozo_exception"))
            return articles

        log.info("%s: %d entries found", name, len(feed.entries))

        for entry in feed.entries:
            title = _clean_html(getattr(entry, "title", ""))
            summary = _clean_html(
                getattr(entry, "summary", None)
                or getattr(entry, "description", None)
                or ""
            )
            link = getattr(entry, "link", "")
            published_date = _parse_date(entry)

            full_text = f"{title} {summary}"
            mentions = extract_ticker_mentions(full_text)

            article_id = hash_text(title)

            metadata = {
                "feed_title": getattr(feed.feed, "title", ""),
                "tags": [t.get("term", "") for t in getattr(entry, "tags", [])],
            }

            articles.append(
                {
                    "id": article_id,
                    "source": name,
                    "title": title,
                    "summary": summary,
                    "published_date": published_date,
                    "link": link,
                    "ticker_mentions": json.dumps(mentions),
                    "metadata_json": json.dumps(metadata),
                }
            )
    except Exception as exc:
        log.error("Error fetching %s (%s): %s", name, url, exc)
    finally:
        time.sleep(delay)

    return articles


def fetch_all_feeds(
    feeds: dict[str, str] | None = None,
    delay: float | None = None,
) -> pd.DataFrame:
    """Fetch all configured RSS feeds.

    Args:
        feeds: Mapping of {name: url}. Defaults to `config.RSS_FEEDS`.
        delay: Seconds between feeds. Defaults to `params.ingestion.rss_request_delay_seconds`.

    Returns:
        Deduplicated DataFrame of all articles.
    """
    params = get_params()
    if feeds is None:
        feeds = RSS_FEEDS
    if delay is None:
        delay = float(params["ingestion"].get("rss_request_delay_seconds", 1.0))

    all_articles: list[dict[str, Any]] = []
    for name, url in feeds.items():
        articles = fetch_feed(name, url, delay=delay)
        all_articles.extend(articles)

    if not all_articles:
        return pd.DataFrame(
            columns=["id", "source", "title", "summary", "published_date",
                     "link", "ticker_mentions", "metadata_json"]
        )

    df = pd.DataFrame(all_articles)
    df = deduplicate_dataframe(df, key_col="id")
    log.info("RSS: total %d articles after dedup", len(df))
    return df


def fetch_latest_news(save: bool = True) -> pd.DataFrame:
    """Convenience wrapper for pipeline automation.

    Fetches all RSS feeds, appends new articles to data/raw/news_articles.csv,
    and returns the fetched DataFrame.

    Args:
        save: Persist to disk when True.

    Returns:
        DataFrame of newly fetched articles.
    """
    df = fetch_all_feeds()
    if save:
        append_csv(df, NEWS_FILE, dedup_key="id")
    return df


if __name__ == "__main__":
    articles = fetch_latest_news(save=True)
    print(f"Fetched {len(articles)} articles")
    if not articles.empty:
        print(articles[["source", "title", "published_date"]].head(10).to_string())
