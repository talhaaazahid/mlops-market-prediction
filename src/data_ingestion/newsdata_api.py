"""NewsData.io free-tier API ingestion (supplementary news source).

Endpoint: https://newsdata.io/api/1/latest?apikey={KEY}&category=business&language=en
Rate limit: 200 credits/day on free tier. Falls back gracefully if key is absent.

Required env var: NEWSDATA_API_KEY
"""

from __future__ import annotations

import json
from typing import Any

import requests

import pandas as pd

from src.config import NEWSDATA_FILE, get_env, get_params
from src.utils.helpers import (
    append_csv,
    deduplicate_dataframe,
    extract_ticker_mentions,
    hash_text,
    to_utc,
)
from src.utils.logger import get_logger

log = get_logger(__name__)

_NEWSDATA_ENDPOINT = "https://newsdata.io/api/1/latest"
_TIMEOUT_SECONDS = 30


def _fetch_page(api_key: str, page_token: str | None = None) -> dict[str, Any]:
    """Fetch one page from the NewsData.io /latest endpoint.

    Args:
        api_key: NewsData.io API key.
        page_token: Pagination token from a previous response (`nextPage`).

    Returns:
        Parsed JSON response dict.

    Raises:
        requests.HTTPError: On non-2xx status codes.
        requests.RequestException: On network-level failures.
    """
    params: dict[str, str] = {
        "apikey": api_key,
        "category": "business",
        "language": "en",
    }
    if page_token:
        params["page"] = page_token

    response = requests.get(_NEWSDATA_ENDPOINT, params=params, timeout=_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def _parse_article(article: dict[str, Any]) -> dict[str, Any]:
    """Normalise one NewsData.io article into the unified schema.

    Args:
        article: Raw article dict from NewsData.io response.

    Returns:
        Dict with keys: [id, source, title, summary, published_date, link,
        ticker_mentions, metadata_json].
    """
    title = (article.get("title") or "").strip()
    description = (article.get("description") or "").strip()
    content = (article.get("content") or "").strip()

    summary = description or content[:500]
    link = (article.get("link") or "").strip()

    pub_raw = article.get("pubDate") or article.get("publishedAt")
    published_date = None
    if pub_raw:
        ts = to_utc(pub_raw)
        published_date = ts.isoformat() if ts else None

    full_text = f"{title} {summary}"
    ticker_mentions = extract_ticker_mentions(full_text)
    article_id = hash_text(title)

    metadata = {
        "source_id": article.get("source_id"),
        "source_url": article.get("source_url"),
        "creator": article.get("creator"),
        "keywords": article.get("keywords") or [],
        "categories": article.get("category") or [],
        "country": article.get("country") or [],
    }

    return {
        "id": article_id,
        "source": "newsdata",
        "title": title,
        "summary": summary,
        "published_date": published_date,
        "link": link,
        "ticker_mentions": json.dumps(ticker_mentions),
        "metadata_json": json.dumps(metadata),
    }


def fetch_newsdata(
    api_key: str | None = None,
    max_pages: int = 3,
) -> pd.DataFrame:
    """Pull business news from NewsData.io.

    Args:
        api_key: API key. Falls back to `NEWSDATA_API_KEY` env var.
        max_pages: Pagination depth. Each page ≈ 10 articles and costs 1 credit.

    Returns:
        Deduplicated DataFrame of articles, or empty DataFrame if key is missing
        or all requests fail.
    """
    if api_key is None:
        api_key = get_env("NEWSDATA_API_KEY")

    if not api_key:
        log.warning("NEWSDATA_API_KEY not set — skipping NewsData.io ingestion")
        return pd.DataFrame(
            columns=["id", "source", "title", "summary", "published_date",
                     "link", "ticker_mentions", "metadata_json"]
        )

    parsed: list[dict[str, Any]] = []
    page_token: str | None = None

    for page_num in range(max_pages):
        try:
            log.info("NewsData.io: fetching page %d", page_num + 1)
            data = _fetch_page(api_key, page_token)

            if data.get("status") != "success":
                log.error("NewsData.io API error: %s", data.get("message"))
                break

            results = data.get("results") or []
            for article in results:
                try:
                    parsed.append(_parse_article(article))
                except Exception as exc:
                    log.debug("Skipping malformed article: %s", exc)

            page_token = data.get("nextPage")
            if not page_token:
                break

        except requests.HTTPError as exc:
            log.error("NewsData.io HTTP %s: %s", exc.response.status_code, exc)
            break
        except requests.RequestException as exc:
            log.error("NewsData.io network error: %s", exc)
            break

    if not parsed:
        return pd.DataFrame(
            columns=["id", "source", "title", "summary", "published_date",
                     "link", "ticker_mentions", "metadata_json"]
        )

    df = pd.DataFrame(parsed)
    df = deduplicate_dataframe(df, key_col="id")
    log.info("NewsData.io: %d articles fetched", len(df))
    return df


def fetch_latest_newsdata(save: bool = True) -> pd.DataFrame:
    """Convenience wrapper for pipeline automation.

    Args:
        save: Persist results to disk when True.

    Returns:
        DataFrame of fetched articles.
    """
    df = fetch_newsdata()
    if save and not df.empty:
        append_csv(df, NEWSDATA_FILE, dedup_key="id")
    return df


if __name__ == "__main__":
    articles = fetch_latest_newsdata(save=True)
    print(f"Fetched {len(articles)} NewsData.io articles")
    if not articles.empty:
        print(articles[["source", "title", "published_date"]].head(5).to_string())
