"""Shared helpers: ticker extraction, deduplication, datetime utilities."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.config import TICKER_ALIASES, TICKERS
from src.utils.logger import get_logger

log = get_logger(__name__)

# `$AAPL`-style cashtag.
_CASHTAG_RE: re.Pattern[str] = re.compile(r"\$([A-Z]{1,5})\b")
# Bare ticker (e.g., "AAPL is up") — restricted to known tickers to avoid noise.
_BARE_TICKER_RE: re.Pattern[str] = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in TICKERS) + r")\b"
)
# Pre-built case-insensitive alias regex per ticker.
_ALIAS_REGEXES: dict[str, re.Pattern[str]] = {
    ticker: re.compile(
        r"\b(" + "|".join(re.escape(a) for a in aliases) + r")\b",
        flags=re.IGNORECASE,
    )
    for ticker, aliases in TICKER_ALIASES.items()
    if aliases
}


def extract_ticker_mentions(text: str) -> list[str]:
    """Find tickers mentioned in `text`.

    Detection strategies (combined, deduped):
      1. Cashtag: `$AAPL`
      2. Bare uppercase ticker token from the known-ticker set
      3. Company-name aliases (case-insensitive)

    Args:
        text: Free-form text (article title+body, post+comments, etc.).

    Returns:
        Sorted unique list of ticker symbols. Empty list if `text` is empty/None.
    """
    if not text:
        return []

    found: set[str] = set()

    for symbol in _CASHTAG_RE.findall(text):
        if symbol in TICKERS:
            found.add(symbol)

    for symbol in _BARE_TICKER_RE.findall(text):
        found.add(symbol)

    for ticker, regex in _ALIAS_REGEXES.items():
        if regex.search(text):
            found.add(ticker)

    return sorted(found)


def hash_text(text: str) -> str:
    """Stable SHA-256 hex digest of `text` (lowercased, whitespace-collapsed).

    Used for deduplication of articles and Reddit posts.
    """
    if text is None:
        text = ""
    normalized = " ".join(text.lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def to_utc(dt: datetime | pd.Timestamp | str | int | float | None) -> pd.Timestamp | None:
    """Normalize any datetime-like input to a UTC `pandas.Timestamp`.

    Handles: pandas Timestamps, naive datetimes, ISO strings, Unix epoch seconds.

    Returns:
        UTC-localized Timestamp, or None if input is null/unparseable.
    """
    if dt is None or (isinstance(dt, float) and pd.isna(dt)):
        return None

    try:
        if isinstance(dt, (int, float)):
            ts = pd.Timestamp(dt, unit="s", tz="UTC")
        else:
            ts = pd.Timestamp(dt)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")
    except (ValueError, TypeError) as exc:
        log.debug("Could not parse datetime %r: %s", dt, exc)
        return None
    return ts


def deduplicate_dataframe(
    df: pd.DataFrame,
    *,
    key_col: str,
    keep: str = "first",
) -> pd.DataFrame:
    """Drop duplicate rows based on `key_col`.

    Args:
        df: Source dataframe.
        key_col: Column whose uniqueness defines a duplicate.
        keep: Which copy to keep ("first" | "last").

    Returns:
        Deduplicated dataframe with the original index reset.
    """
    if df.empty:
        return df
    before = len(df)
    out = df.drop_duplicates(subset=[key_col], keep=keep).reset_index(drop=True)
    after = len(out)
    if before != after:
        log.info("Deduplicated %d → %d rows on key=%s", before, after, key_col)
    return out


def append_csv(df: pd.DataFrame, path: Path, *, dedup_key: str | None = None) -> None:
    """Append `df` to a CSV at `path`, creating the file if needed.

    If `dedup_key` is given, the merged result is deduplicated on that column
    before writing. Parent directory is created automatically.

    Args:
        df: Rows to append. No-op if empty.
        path: Destination CSV.
        dedup_key: Optional column name for post-merge deduplication.
    """
    if df is None or df.empty:
        log.info("append_csv: nothing to write to %s", path.name)
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        try:
            existing = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            existing = pd.DataFrame()
        combined = pd.concat([existing, df], ignore_index=True)
    else:
        combined = df.copy()

    if dedup_key and dedup_key in combined.columns:
        combined = deduplicate_dataframe(combined, key_col=dedup_key)

    combined.to_csv(path, index=False)
    log.info("Wrote %d rows to %s", len(combined), path)


def utc_now() -> pd.Timestamp:
    """Current time as a UTC `pandas.Timestamp`."""
    return pd.Timestamp(datetime.now(tz=timezone.utc))


def chunked(iterable: Iterable, size: int) -> Iterable[list]:
    """Yield successive `size`-length chunks from `iterable`."""
    if size <= 0:
        raise ValueError("size must be positive")
    bucket: list = []
    for item in iterable:
        bucket.append(item)
        if len(bucket) == size:
            yield bucket
            bucket = []
    if bucket:
        yield bucket


if __name__ == "__main__":
    sample = "Apple ($AAPL) and Tesla beat estimates; MSFT and Microsoft also up. Random TICKER ignored."
    print("Mentions:", extract_ticker_mentions(sample))
    print("Hash:    ", hash_text(sample)[:16], "...")
    print("UTC now: ", utc_now())

def validate_data_quality(df):
    Validate data quality metrics.
    null_ratio = df.isnull().sum() / len(df)
    return null_ratio < 0.05

