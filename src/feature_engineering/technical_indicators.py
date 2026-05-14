"""Technical indicator computation using pandas and numpy only (no TA-Lib).

Reads data/raw/price_data.csv and produces data/processed/technical_features.csv
with price-based returns, moving averages, volatility, momentum, Bollinger Bands,
and volume features — one row per (ticker, date).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PRICE_DATA_FILE, PROCESSED_DIR
from src.utils.logger import get_logger

log = get_logger(__name__)

_TECHNICAL_FILE = PROCESSED_DIR / "technical_features.csv"


# ── Indicator implementations ─────────────────────────────────────────────── #

def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder-smoothed RSI."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)  # neutral when no prior data


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range (Wilder smoothing)."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _bollinger(
    series: pd.Series, window: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands: (upper, lower, width)."""
    mid = series.rolling(window=window, min_periods=1).mean()
    std = series.rolling(window=window, min_periods=1).std(ddof=0).fillna(0.0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    width = (upper - lower) / mid.replace(0, np.nan)
    return upper, lower, width.fillna(0.0)


# ── Per-ticker feature computation ───────────────────────────────────────── #

def compute_indicators(price_df: pd.DataFrame) -> pd.DataFrame:
    """Compute all technical indicators for a single-ticker price DataFrame.

    Args:
        price_df: DataFrame with columns [date, open, high, low, close,
            adj_close, volume], sorted ascending by date.

    Returns:
        DataFrame with all indicator columns appended.
    """
    df = price_df.copy().sort_values("date").reset_index(drop=True)

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)

    # ── Returns ──────────────────────────────────────────────────────────── #
    df["returns_1d"] = close.pct_change()
    df["returns_5d"] = close.pct_change(periods=5)
    df["log_returns"] = np.log(close / close.shift(1))

    # ── Moving averages ──────────────────────────────────────────────────── #
    for w in (5, 10, 20, 50):
        df[f"sma_{w}"] = _sma(close, w)

    df["ema_12"] = _ema(close, span=12)
    df["ema_26"] = _ema(close, span=26)

    # ── Volatility ───────────────────────────────────────────────────────── #
    df["volatility_10d"] = df["returns_1d"].rolling(window=10, min_periods=1).std(ddof=0)
    df["volatility_20d"] = df["returns_1d"].rolling(window=20, min_periods=1).std(ddof=0)
    df["atr_14"] = _atr(high, low, close, period=14)

    # ── Momentum ─────────────────────────────────────────────────────────── #
    df["rsi_14"] = _rsi(close, period=14)

    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = _ema(df["macd"], span=9)
    df["macd_histogram"] = df["macd"] - df["macd_signal"]

    # ── Bollinger Bands ──────────────────────────────────────────────────── #
    df["bb_upper"], df["bb_lower"], df["bb_width"] = _bollinger(close)

    # ── Volume ───────────────────────────────────────────────────────────── #
    df["volume_sma_20"] = _sma(volume, 20)
    df["volume_ratio"] = volume / df["volume_sma_20"].replace(0, np.nan)
    df["volume_ratio"] = df["volume_ratio"].fillna(1.0)

    return df


def compute_all_indicators(
    price_df: pd.DataFrame | None = None,
    input_path: Path | str = PRICE_DATA_FILE,
    output_path: Path | str = _TECHNICAL_FILE,
    save: bool = True,
) -> pd.DataFrame:
    """Compute technical indicators for all tickers and save results.

    Args:
        price_df: Pre-loaded price DataFrame. If None, reads from `input_path`.
        input_path: Path to price_data.csv.
        output_path: Destination CSV.
        save: Write to disk when True.

    Returns:
        Long-format DataFrame with all indicator columns, one row per
        (ticker, date).
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if price_df is None:
        if not input_path.exists():
            raise FileNotFoundError(
                f"Price data not found: {input_path}. "
                "Run the ingestion pipeline first."
            )
        log.info("Reading %s …", input_path)
        price_df = pd.read_csv(input_path, parse_dates=["date"])

    if price_df.empty:
        log.warning("Price DataFrame is empty — returning empty indicators.")
        return pd.DataFrame()

    price_df["date"] = pd.to_datetime(price_df["date"]).dt.normalize()

    ticker_dfs: list[pd.DataFrame] = []
    for ticker, grp in price_df.groupby("ticker"):
        log.info("Computing indicators for %s (%d rows) …", ticker, len(grp))
        try:
            enriched = compute_indicators(grp)
            ticker_dfs.append(enriched)
        except Exception as exc:
            log.error("Failed to compute indicators for %s: %s", ticker, exc)

    if not ticker_dfs:
        return pd.DataFrame()

    result = pd.concat(ticker_dfs, ignore_index=True)
    result = result.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Drop first row per ticker (NaN returns from pct_change)
    result = result.groupby("ticker", group_keys=False).apply(
        lambda g: g.iloc[1:] if len(g) > 1 else g
    ).reset_index(drop=True)

    if save:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)
        log.info("Saved %d technical feature rows to %s", len(result), output_path)

    log.info(
        "Technical indicators complete: %d rows, %d tickers",
        len(result),
        result["ticker"].nunique(),
    )
    return result


if __name__ == "__main__":
    df = compute_all_indicators(save=True)
    print(df[["ticker", "date", "close", "rsi_14", "macd", "bb_width", "volume_ratio"]].head(20))
    print(f"\nShape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

# New Indicators
# - Ichimoku Cloud
# - Volume Profile
# - Market Profile


# New Indicators
# - Ichimoku Cloud
# - Volume Profile
# - Market Profile


# New Indicators
# - Ichimoku Cloud
# - Volume Profile
# - Market Profile


# New Indicators
# - Ichimoku Cloud
# - Volume Profile
# - Market Profile

