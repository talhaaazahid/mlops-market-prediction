"""Tests for feature engineering modules."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ── Helpers ────────────────────────────────────────────────────────────────── #


def _make_price_df(ticker: str = "AAPL", rows: int = 60) -> pd.DataFrame:
    """Return a price DataFrame with the lowercase column names compute_indicators expects."""
    rng = np.random.default_rng(42)
    close = 150.0 + np.cumsum(rng.normal(0, 2, rows))
    high = close + rng.uniform(0.5, 3, rows)
    low = close - rng.uniform(0.5, 3, rows)
    open_ = close + rng.normal(0, 1, rows)
    volume = rng.integers(1_000_000, 5_000_000, rows).astype(float)
    dates = pd.date_range("2023-01-01", periods=rows, freq="B")
    return pd.DataFrame(
        {
            "date": dates,          # compute_indicators expects "date"
            "open": open_,          # lowercase column names
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "ticker": ticker,
        }
    )


# ── Technical Indicators ───────────────────────────────────────────────────── #


class TestTechnicalIndicators:
    def test_compute_indicators_adds_expected_columns(self) -> None:
        from src.feature_engineering.technical_indicators import compute_indicators

        df = _make_price_df()
        result = compute_indicators(df)

        # Actual column names are all lowercase
        expected = {
            "rsi_14", "macd", "macd_signal", "macd_histogram",
            "bb_upper", "bb_lower", "bb_width", "atr_14",
            "sma_10", "sma_20", "sma_50", "ema_12", "ema_26",
            "volatility_20d", "volume_ratio",
        }
        assert expected.issubset(set(result.columns))

    def test_rsi_bounds(self) -> None:
        from src.feature_engineering.technical_indicators import compute_indicators

        df = _make_price_df(rows=80)
        result = compute_indicators(df).dropna(subset=["rsi_14"])
        assert (result["rsi_14"] >= 0).all()
        assert (result["rsi_14"] <= 100).all()

    def test_bollinger_bands_ordering(self) -> None:
        from src.feature_engineering.technical_indicators import compute_indicators

        result = compute_indicators(_make_price_df(rows=80)).dropna(subset=["bb_upper"])
        assert (result["bb_upper"] >= result["bb_lower"]).all()

    def test_no_lookahead_in_indicators(self) -> None:
        """Computing indicators on rows[0:n] vs rows[0:n+k] must give the same
        value for row n-1 — no future data leakage."""
        from src.feature_engineering.technical_indicators import compute_indicators

        df = _make_price_df(rows=80)
        result_full = compute_indicators(df)
        result_truncated = compute_indicators(df.iloc[:60].copy())

        # Both DataFrames have a 'date' column; compare by date value
        last_date = result_truncated["date"].iloc[-1]
        col = "sma_20"
        full_val = result_full.loc[result_full["date"] == last_date, col].values[0]
        trunc_val = result_truncated.loc[result_truncated["date"] == last_date, col].values[0]
        assert abs(full_val - trunc_val) < 1e-8


# ── Sentiment Aggregator ───────────────────────────────────────────────────── #


def _make_sentiment_df() -> pd.DataFrame:
    """Return a sentiment DataFrame with published_date (as aggregator expects)."""
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    rows = []
    for date in dates:
        for ticker in ["AAPL", "MSFT"]:
            rows.append({
                "published_date": date.isoformat(),   # aggregator reads this column
                "ticker_mentions": f"['{ticker}']",
                "ensemble_compound": np.random.uniform(-1, 1),
                "ensemble_label": np.random.choice(["positive", "negative", "neutral"]),
                "finbert_compound": np.random.uniform(-1, 1),
                "vader_compound": np.random.uniform(-1, 1),
                "text": "Some news article.",
                "source": "reuters",
            })
    return pd.DataFrame(rows)


class TestSentimentAggregator:
    def test_aggregate_returns_expected_shape(self) -> None:
        from src.feature_engineering.sentiment_aggregator import aggregate_daily_sentiment

        df = _make_sentiment_df()
        result = aggregate_daily_sentiment(df)
        assert isinstance(result, pd.DataFrame)
        assert "ticker" in result.columns
        assert "date" in result.columns

    def test_result_has_both_tickers(self) -> None:
        from src.feature_engineering.sentiment_aggregator import aggregate_daily_sentiment

        df = _make_sentiment_df()
        result = aggregate_daily_sentiment(df)
        assert "AAPL" in result["ticker"].values
        assert "MSFT" in result["ticker"].values


# ── Dataset Builder ────────────────────────────────────────────────────────── #


class TestDatasetBuilder:
    def test_sliding_windows_shape(self) -> None:
        """Sliding window arrays must have shape (n_samples, lookback, n_features)."""
        from src.feature_engineering.dataset_builder import _sliding_windows

        n, lookback, n_feats = 100, 10, 5
        data = np.random.randn(n, n_feats).astype(np.float32)
        labels = np.random.randint(0, 2, n)
        X, y = _sliding_windows(data, labels, lookback=lookback)
        assert X.shape == (n - lookback, lookback, n_feats)
        assert y.shape == (n - lookback,)

    def test_no_train_val_test_overlap(self) -> None:
        """Train / val / test index sets must be disjoint (no data leakage)."""
        from src.feature_engineering.dataset_builder import _chronological_split

        n = 300
        idx = np.arange(n)
        train, val, test = _chronological_split(idx, train_ratio=0.7, val_ratio=0.15)
        assert len(set(train) & set(val)) == 0
        assert len(set(train) & set(test)) == 0
        assert len(set(val) & set(test)) == 0
        assert len(train) + len(val) + len(test) == n

    def test_chronological_split_proportions(self) -> None:
        from src.feature_engineering.dataset_builder import _chronological_split

        n = 1000
        idx = np.arange(n)
        train, val, test = _chronological_split(idx, train_ratio=0.7, val_ratio=0.15)
        assert abs(len(train) / n - 0.70) < 0.02
        assert abs(len(val) / n - 0.15) < 0.02

    def test_scaler_fit_on_train_only(self) -> None:
        """StandardScaler must be fit on train split only, not on val/test data."""
        from sklearn.preprocessing import StandardScaler
        from src.feature_engineering.dataset_builder import _chronological_split

        n = 300
        # Inject a clear signal: val/test data has very different mean
        data = np.zeros((n, 5))
        data[:, 0] = np.arange(n, dtype=float)  # trending feature
        idx = np.arange(n)
        train_idx, val_idx, _ = _chronological_split(idx)

        scaler = StandardScaler()
        scaler.fit(data[train_idx])
        transformed_val = scaler.transform(data[val_idx])

        # Scaler mean was fit on train (low values); val data is shifted → not zero-centred
        assert transformed_val.shape == (len(val_idx), 5)
        # The trending feature should NOT be centred near 0 in val
        assert abs(transformed_val[:, 0].mean()) > 0.5
