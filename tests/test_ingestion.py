"""Tests for data ingestion modules."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


# ── Helpers ────────────────────────────────────────────────────────────────── #


def _make_ticker_df(ticker: str = "AAPL", rows: int = 5) -> pd.DataFrame:
    import numpy as np

    dates = pd.date_range("2024-01-01", periods=rows, freq="B")
    return pd.DataFrame(
        {
            "Open": np.random.uniform(150, 200, rows),
            "High": np.random.uniform(200, 210, rows),
            "Low": np.random.uniform(140, 150, rows),
            "Close": np.random.uniform(150, 200, rows),
            "Volume": np.random.randint(1_000_000, 5_000_000, rows),
            "Ticker": ticker,
        },
        index=dates,
    )


# ── Yahoo Finance ──────────────────────────────────────────────────────────── #


class TestYahooFinance:
    def test_fetch_price_data_returns_dataframe(self) -> None:
        mock_df = _make_ticker_df("AAPL")

        with patch("yfinance.download", return_value=mock_df):
            from src.data_ingestion.yahoo_finance import fetch_price_data

            result = fetch_price_data(["AAPL"], period="5d")

        assert isinstance(result, pd.DataFrame)
        assert not result.empty

    def test_fetch_price_data_empty_tickers(self) -> None:
        from src.data_ingestion.yahoo_finance import fetch_price_data

        result = fetch_price_data([], period="5d")
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_fetch_price_data_handles_yfinance_failure(self) -> None:
        with patch("yfinance.download", side_effect=Exception("network error")):
            from src.data_ingestion.yahoo_finance import fetch_price_data

            result = fetch_price_data(["AAPL"], period="5d")

        assert isinstance(result, pd.DataFrame)
        assert result.empty


# ── RSS News ───────────────────────────────────────────────────────────────── #


class TestNewsRSS:
    def _mock_feed(self) -> MagicMock:
        entry = MagicMock()
        entry.title = "Apple stock surges on record earnings"
        entry.link = "https://example.com/article/1"
        entry.get.return_value = "2024-01-15T10:00:00Z"
        entry.summary = "Apple Inc reported record Q1 2024 earnings."

        feed = MagicMock()
        feed.entries = [entry]
        feed.bozo = False
        feed.feed.title = "Reuters Finance"
        return feed

    def test_fetch_feed_returns_list(self) -> None:
        with patch("feedparser.parse", return_value=self._mock_feed()):
            from src.data_ingestion.news_rss import fetch_feed

            # Correct signature: fetch_feed(name, url)
            result = fetch_feed("Reuters", "https://dummy.rss/feed")

        assert isinstance(result, list)
        assert len(result) >= 1
        assert "title" in result[0]

    def test_fetch_feed_assigns_stable_id_hash(self) -> None:
        """Identical titles must produce identical IDs, enabling downstream dedup."""
        mock_feed = self._mock_feed()
        mock_feed.entries = mock_feed.entries * 3  # same entry 3x

        with patch("feedparser.parse", return_value=mock_feed):
            from src.data_ingestion.news_rss import fetch_feed

            result = fetch_feed("Reuters", "https://dummy.rss/feed")

        # fetch_feed doesn't dedup itself, but gives each article a hash-based id
        ids = [r["id"] for r in result]
        # All 3 identical entries should have the same id (stable hash of title)
        assert len(set(ids)) == 1

    def test_fetch_feed_network_error_returns_empty(self) -> None:
        with patch("feedparser.parse", side_effect=Exception("timeout")):
            from src.data_ingestion.news_rss import fetch_feed

            result = fetch_feed("Test", "https://bad.url/feed")

        assert result == []


# ── Reddit Scraper ─────────────────────────────────────────────────────────── #


class TestRedditScraper:
    def _mock_submission(self, title: str = "AAPL to the moon") -> MagicMock:
        submission = MagicMock()
        submission.id = "abc123"
        submission.title = title
        submission.selftext = "Long discussion about Apple stock."
        submission.score = 150
        submission.url = "https://reddit.com/r/investing/abc123"
        submission.created_utc = 1705320000.0

        comment = MagicMock()
        comment.body = "Great analysis!"
        comment.score = 20
        submission.comments.list.return_value = [comment]
        return submission

    def test_fetch_subreddit_returns_list(self) -> None:
        mock_reddit = MagicMock()
        mock_sub = MagicMock()
        mock_sub.hot.return_value = [self._mock_submission()]
        mock_sub.new.return_value = []
        mock_reddit.subreddit.return_value = mock_sub

        from src.data_ingestion.reddit_scraper import fetch_subreddit

        # Correct signature: fetch_subreddit(reddit, subreddit_name)
        result = fetch_subreddit(mock_reddit, "investing")

        assert isinstance(result, list)

    def test_fetch_subreddit_filters_low_score(self) -> None:
        submission = self._mock_submission()
        submission.score = 2  # below threshold

        mock_reddit = MagicMock()
        mock_sub = MagicMock()
        mock_sub.hot.return_value = [submission]
        mock_sub.new.return_value = []
        mock_reddit.subreddit.return_value = mock_sub

        from src.data_ingestion.reddit_scraper import fetch_subreddit

        result = fetch_subreddit(mock_reddit, "investing")

        assert all(post.get("score", 0) >= 5 for post in result)


# ── helpers ────────────────────────────────────────────────────────────────── #


class TestHelpers:
    def test_hash_text_deterministic(self) -> None:
        from src.utils.helpers import hash_text

        assert hash_text("hello world") == hash_text("hello world")

    def test_hash_text_different_inputs(self) -> None:
        from src.utils.helpers import hash_text

        assert hash_text("apple") != hash_text("microsoft")

    def test_extract_ticker_mentions(self) -> None:
        from src.utils.helpers import extract_ticker_mentions

        text = "I am bullish on AAPL and also watching TSLA."
        tickers = extract_ticker_mentions(text)
        assert "AAPL" in tickers
        assert "TSLA" in tickers

    def test_deduplicate_dataframe(self) -> None:
        from src.utils.helpers import deduplicate_dataframe

        df = pd.DataFrame({"id": ["a", "b", "a", "c"], "val": [1, 2, 3, 4]})
        # Correct keyword: key_col (not subset)
        result = deduplicate_dataframe(df, key_col="id")
        assert len(result) == 3
        assert result["id"].nunique() == 3
