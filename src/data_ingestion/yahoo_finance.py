"""Yahoo Finance price data ingestion using yfinance.

Downloads daily OHLCV data (2 years) and 1-hour intraday data (60 days) for
the configured tickers and saves them to `data/raw/`.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yfinance as yf

from src.config import (
    PRICE_DATA_FILE,
    PRICE_INTRADAY_FILE,
    TICKERS,
    get_params,
)
from src.utils.logger import get_logger

log = get_logger(__name__)

_ALPHAVANTAGE_URL = "https://www.alphavantage.co/query"
_TWELVEDATA_URL = "https://api.twelvedata.com/time_series"


def _fetch_twelvedata(ticker: str, outputsize: int = 5000) -> pd.DataFrame:
    """Fetch daily OHLCV from Twelve Data (primary source).

    Returns up to ~20 years of daily history. Free tier allows 800 requests/day
    and 8 requests/minute. Requires `TWELVE_DATA_API_KEY` in the environment.

    Args:
        ticker: Stock symbol (e.g. "AAPL").
        outputsize: Maximum number of daily rows to fetch (max 5000).

    Returns:
        Long-format DataFrame with columns
        [date, ticker, open, high, low, close, adj_close, volume].
        Empty DataFrame on any failure.
    """
    api_key = os.getenv("TWELVE_DATA_API_KEY", "").strip()
    if not api_key:
        return pd.DataFrame()

    try:
        r = requests.get(
            _TWELVEDATA_URL,
            params={
                "symbol": ticker,
                "interval": "1day",
                "outputsize": str(outputsize),
                "apikey": api_key,
                "format": "JSON",
            },
            timeout=20,
        )
        data = r.json()
    except Exception as exc:
        log.error("Twelve Data request failed for %s: %s", ticker, exc)
        return pd.DataFrame()

    if data.get("status") == "error":
        log.warning("Twelve Data error for %s: %s", ticker, data.get("message"))
        return pd.DataFrame()

    values = data.get("values")
    if not values:
        log.warning("Twelve Data returned no values for %s", ticker)
        return pd.DataFrame()

    rows = [
        {
            "date": pd.Timestamp(row["datetime"]),
            "ticker": ticker,
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "adj_close": float(row["close"]),
            "volume": float(row.get("volume") or 0),
        }
        for row in values
    ]
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    log.info("Twelve Data: %s — %d rows (%s → %s)",
             ticker, len(df), df["date"].min().date(), df["date"].max().date())
    return df


def _fetch_alphavantage(ticker: str) -> pd.DataFrame:
    """Fetch ~100 days of daily OHLCV from Alpha Vantage as a fallback.

    Requires `ALPHA_VANTAGE_API_KEY` in the environment. Returns an empty
    DataFrame if the key is missing, the API rejects the request, or the
    response is malformed.

    Args:
        ticker: Stock symbol (e.g. "AAPL").

    Returns:
        Long-format DataFrame with columns
        [date, ticker, open, high, low, close, adj_close, volume].
    """
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "").strip()
    if not api_key:
        log.warning("ALPHA_VANTAGE_API_KEY not set — cannot use fallback for %s", ticker)
        return pd.DataFrame()

    try:
        r = requests.get(
            _ALPHAVANTAGE_URL,
            params={
                "function": "TIME_SERIES_DAILY",
                "symbol": ticker,
                "outputsize": "compact",
                "apikey": api_key,
            },
            timeout=15,
        )
        data = r.json()
    except Exception as exc:
        log.error("Alpha Vantage request failed for %s: %s", ticker, exc)
        return pd.DataFrame()

    series = data.get("Time Series (Daily)")
    if not series:
        log.warning("Alpha Vantage returned no data for %s — response: %s",
                    ticker, str(data)[:200])
        return pd.DataFrame()

    rows = [
        {
            "date": pd.Timestamp(date_str),
            "ticker": ticker,
            "open": float(row["1. open"]),
            "high": float(row["2. high"]),
            "low": float(row["3. low"]),
            "close": float(row["4. close"]),
            "adj_close": float(row["4. close"]),
            "volume": float(row["5. volume"]),
        }
        for date_str, row in series.items()
    ]
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    log.info("Alpha Vantage fallback: %s — %d rows", ticker, len(df))
    return df


def fetch_price_data(
    tickers: list[str] | None = None,
    period: str | None = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """Download OHLCV price data for multiple tickers.

    Args:
        tickers: List of ticker symbols. Defaults to `config.TICKERS`.
        period: yfinance period string (e.g. "2y", "60d"). Defaults to
            `params.ingestion.price_period`.
        interval: yfinance interval string (e.g. "1d", "1h").

    Returns:
        Long-format DataFrame with columns:
        [date, ticker, open, high, low, close, adj_close, volume].
        Empty DataFrame if all tickers fail.
    """
    params = get_params()
    if tickers is None:
        tickers = params["ingestion"]["tickers"]
    if period is None:
        period = params["ingestion"]["price_period"]

    rows: list[dict[str, Any]] = []
    failed: list[str] = []

    for ticker in tickers:
        # ── 1) Try Twelve Data first (best historical coverage) ─────────── #
        if interval == "1d":
            log.info("Downloading %s via Twelve Data (interval=1d)", ticker)
            td_df = _fetch_twelvedata(ticker)
            if not td_df.empty:
                rows.append(td_df)
                time.sleep(8)  # 8 req/min free tier limit
                continue

        # ── 2) Fall back to yfinance ────────────────────────────────────── #
        try:
            log.info("Downloading %s via yfinance | period=%s interval=%s",
                     ticker, period, interval)
            raw = yf.download(
                ticker,
                period=period,
                interval=interval,
                auto_adjust=False,
                progress=False,
            )
            if raw.empty:
                # Alpha Vantage free tier only supports daily data
                if interval == "1d":
                    log.warning("%s: yfinance returned no data — trying Alpha Vantage", ticker)
                    av_df = _fetch_alphavantage(ticker)
                    if not av_df.empty:
                        rows.append(av_df)
                        time.sleep(13)  # Alpha Vantage free tier: 5 req/min
                        continue
                else:
                    log.warning("%s: yfinance returned no data (intraday — no fallback)", ticker)
                failed.append(ticker)
                time.sleep(5)
                continue

            # yfinance can return MultiIndex columns when auto_adjust=False
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = [col[0].lower().replace(" ", "_") for col in raw.columns]
            else:
                raw.columns = [c.lower().replace(" ", "_") for c in raw.columns]

            raw = raw.rename(columns={"adj_close": "adj_close"})

            # Ensure required columns exist
            for col in ("open", "high", "low", "close", "volume"):
                if col not in raw.columns:
                    raw[col] = float("nan")
            if "adj_close" not in raw.columns:
                raw["adj_close"] = raw["close"]

            raw.index.name = "date"
            raw = raw.reset_index()
            raw["ticker"] = ticker

            time.sleep(3)  # avoid Yahoo Finance rate limiting between tickers
            rows.append(
                raw[["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]]
            )
            log.info("%s: %d rows fetched", ticker, len(raw))
        except Exception as exc:
            log.error("Error fetching %s: %s", ticker, exc)
            failed.append(ticker)

    if failed:
        log.warning("Failed tickers: %s", failed)

    if not rows:
        return pd.DataFrame(
            columns=["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
        )

    df = pd.concat(rows, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return df


def fetch_latest_prices(
    tickers: list[str] | None = None,
    period: str = "1d",
    interval: str = "1h",
) -> pd.DataFrame:
    """Fetch latest intraday price data for real-time pipeline calls.

    Args:
        tickers: Tickers to fetch. Defaults to `config.TICKERS`.
        period: Short period (e.g. "1d", "5d").
        interval: Intraday interval ("1h", "15m", etc.).

    Returns:
        Long-format OHLCV DataFrame (same schema as `fetch_price_data`).
    """
    return fetch_price_data(tickers=tickers, period=period, interval=interval)


def save_price_data(
    df: pd.DataFrame,
    path: Path = PRICE_DATA_FILE,
) -> None:
    """Persist price DataFrame to CSV, overwriting any existing file.

    Args:
        df: DataFrame from `fetch_price_data`.
        path: Destination CSV path.
    """
    if df.empty:
        log.warning("Price DataFrame is empty — skipping save")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info("Saved %d price rows to %s", len(df), path)


def run(save: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full ingestion run: daily data + intraday data.

    Args:
        save: Write CSVs to disk when True.

    Returns:
        Tuple of (daily_df, intraday_df).
    """
    params = get_params()
    tickers = params["ingestion"]["tickers"]

    log.info("=== Yahoo Finance ingestion START ===")

    daily_df = fetch_price_data(
        tickers=tickers,
        period=params["ingestion"]["price_period"],
        interval=params["ingestion"]["price_interval_daily"],
    )

    # yfinance rate-limiting — be gentle between two bulk downloads.
    time.sleep(2)

    intraday_df = fetch_price_data(
        tickers=tickers,
        period=params["ingestion"]["price_intraday_period"],
        interval=params["ingestion"]["price_interval_intraday"],
    )

    if save:
        save_price_data(daily_df, PRICE_DATA_FILE)
        save_price_data(intraday_df, PRICE_INTRADAY_FILE)

    log.info(
        "=== Yahoo Finance ingestion END | daily=%d rows intraday=%d rows ===",
        len(daily_df),
        len(intraday_df),
    )
    return daily_df, intraday_df


if __name__ == "__main__":
    daily, intraday = run(save=True)
    print(daily.tail())
    print(intraday.tail())

# Parallel Download Configuration
MAX_WORKERS = 4
TIMEOUT_SECONDS = 30
RETRY_ATTEMPTS = 3


# Parallel Download Configuration
MAX_WORKERS = 4
TIMEOUT_SECONDS = 30
RETRY_ATTEMPTS = 3


# Parallel Download Configuration
MAX_WORKERS = 4
TIMEOUT_SECONDS = 30
RETRY_ATTEMPTS = 3


# Parallel Download Configuration
MAX_WORKERS = 4
TIMEOUT_SECONDS = 30
RETRY_ATTEMPTS = 3

