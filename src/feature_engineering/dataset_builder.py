"""Final dataset construction: merge features, build targets, create sequences.

Merges technical_features.csv + daily_sentiment_features.csv on (date, ticker),
creates 4 target variables, applies a 30-day sliding window, normalises with
StandardScaler (fit on train only), splits 70/15/15 chronologically, and saves
NumPy arrays + metadata to data/features/.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import FEATURES_DIR, PROCESSED_DIR, get_params
from src.utils.logger import get_logger

log = get_logger(__name__)

_TECHNICAL_FILE = PROCESSED_DIR / "technical_features.csv"
_SENTIMENT_FILE = PROCESSED_DIR / "daily_sentiment_features.csv"

# Columns that must NOT be used as model input features
_NON_FEATURE_COLS = {
    "date", "ticker",
    "open", "high", "low", "close", "adj_close", "volume",
    "direction", "direction_3class", "volatility_spike", "next_day_return",
}


def _build_targets(
    df: pd.DataFrame,
    horizon: int = 5,
    threshold: float = 0.01,
) -> pd.DataFrame:
    """Append 4 target columns based on forward `horizon`-day returns.

    Multi-day forward returns are far more learnable than 1-day returns, which
    are dominated by market noise. The label for the window ending on day t is
    what happens between day t+1 and day t+horizon.

    Args:
        df: Per-ticker DataFrame sorted ascending by date, with a `close`
            column.
        horizon: Number of forward days to compute the cumulative return over.
            Defaults to 5 (one trading week).
        threshold: 3-class symmetric threshold for the cumulative return.
            Defaults to 1%, which is roughly the 1-sigma noise floor of
            5-day returns on liquid US equities.

    Returns:
        DataFrame with new columns:
        direction, direction_3class, volatility_spike, next_day_return.
        The last `horizon` rows per ticker will have NaN targets (no future
        data) and must be dropped by the caller.
    """
    df = df.copy()
    close = df["close"].astype(float)
    future_close = close.shift(-horizon)

    # forward-`horizon`-day cumulative return (column name kept for backward compat)
    df["next_day_return"] = (future_close - close) / close

    # Binary direction: 1 = up, 0 = down/flat (over the horizon)
    df["direction"] = (df["next_day_return"] > 0).astype(int)

    # 3-class: ±threshold defines Neutral
    df["direction_3class"] = 0
    df.loc[df["next_day_return"] > threshold, "direction_3class"] = 1
    df.loc[df["next_day_return"] < -threshold, "direction_3class"] = -1

    # Volatility spike: |horizon-day return| > 2 × 20-day rolling std
    vol_20d = df.get("volatility_20d", pd.Series(dtype=float))
    if vol_20d.empty or vol_20d.isna().all():
        returns = close.pct_change()
        vol_20d = returns.rolling(window=20, min_periods=1).std(ddof=0)
    df["volatility_spike"] = (df["next_day_return"].abs() > 2 * vol_20d).astype(int)

    return df


def _chronological_split(
    idx: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split index array chronologically into train / val / test portions.

    Args:
        idx: 1-D array of integer indices.
        train_ratio: Fraction of data for training (default 0.70).
        val_ratio: Fraction of data for validation (default 0.15).

    Returns:
        Tuple of (train_idx, val_idx, test_idx) arrays.
    """
    n = len(idx)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return idx[:train_end], idx[train_end:val_end], idx[val_end:]


def _sliding_windows(
    feature_matrix: np.ndarray,
    target_vector: np.ndarray,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create (X, y) pairs using a sliding window.

    Args:
        feature_matrix: Shape (T, F) — time-steps × features.
        target_vector: Shape (T,) — one label per time-step.
        lookback: Number of past days in each window.

    Returns:
        Tuple of (X, y):
        X shape: (T - lookback, lookback, F)
        y shape: (T - lookback,)
    """
    X, y = [], []
    for i in range(lookback, len(feature_matrix)):
        X.append(feature_matrix[i - lookback : i])
        y.append(target_vector[i])
    return np.array(X, dtype=np.float32), np.array(y)


def build_dataset(
    technical_df: pd.DataFrame | None = None,
    sentiment_df: pd.DataFrame | None = None,
    technical_path: Path | str = _TECHNICAL_FILE,
    sentiment_path: Path | str = _SENTIMENT_FILE,
    output_dir: Path | str = FEATURES_DIR,
    target: str = "direction_3class",
    save: bool = True,
) -> dict[str, Any]:
    """Merge features, create targets, build sliding windows, split, normalise.

    Args:
        technical_df: Pre-loaded technical features. If None, reads from file.
        sentiment_df: Pre-loaded sentiment features. If None, reads from file.
        technical_path: Path to technical_features.csv.
        sentiment_path: Path to daily_sentiment_features.csv.
        output_dir: Directory for NumPy arrays and metadata.
        target: Which target column to use for y arrays
            ("direction", "direction_3class", "volatility_spike",
             "next_day_return").
        save: Write all artefacts to disk when True.

    Returns:
        Dict with keys: X_train, X_val, X_test, y_train, y_val, y_test,
        feature_names, scaler, metadata.
    """
    params = get_params()
    lookback: int = int(params["features"]["lookback_days"])
    train_ratio: float = float(params["features"]["train_ratio"])
    val_ratio: float = float(params["features"]["val_ratio"])
    horizon: int = int(params["features"].get("target_horizon_days", 5))
    threshold: float = float(params["features"].get("return_threshold_3class", 0.01))

    output_dir = Path(output_dir)

    # ── Load data ─────────────────────────────────────────────────────────── #
    if technical_df is None:
        technical_path = Path(technical_path)
        if not technical_path.exists():
            raise FileNotFoundError(
                f"Technical features not found: {technical_path}. "
                "Run technical_indicators.py first."
            )
        log.info("Reading technical features from %s …", technical_path)
        technical_df = pd.read_csv(technical_path, parse_dates=["date"])

    if sentiment_df is None:
        sentiment_path = Path(sentiment_path)
        if not sentiment_path.exists():
            log.warning(
                "Sentiment features not found at %s — using zeros.", sentiment_path
            )
            sentiment_df = pd.DataFrame()
        else:
            log.info("Reading sentiment features from %s …", sentiment_path)
            sentiment_df = pd.read_csv(sentiment_path, parse_dates=["date"])

    # ── Merge ─────────────────────────────────────────────────────────────── #
    technical_df["date"] = pd.to_datetime(technical_df["date"]).dt.normalize()
    merged: pd.DataFrame

    if not sentiment_df.empty:
        sentiment_df["date"] = pd.to_datetime(sentiment_df["date"]).dt.normalize()
        merged = technical_df.merge(sentiment_df, on=["date", "ticker"], how="left")
    else:
        merged = technical_df.copy()

    merged = merged.sort_values(["ticker", "date"]).reset_index(drop=True)

    # ── Build targets ─────────────────────────────────────────────────────── #
    log.info("Building target variables (horizon=%d days, threshold=%.3f) …",
             horizon, threshold)
    merged = (
        merged.groupby("ticker", group_keys=False)
        .apply(lambda g: _build_targets(g, horizon=horizon, threshold=threshold))
        .reset_index(drop=True)
    )

    # Drop last `horizon` rows per ticker (no future data for the target)
    merged = (
        merged.groupby("ticker", group_keys=False)
        .apply(lambda g: g.iloc[:-horizon])
        .reset_index(drop=True)
    )

    # Drop rows where the chosen target is NaN
    merged = merged.dropna(subset=[target]).reset_index(drop=True)

    # ── Identify feature columns ──────────────────────────────────────────── #
    feature_cols = [
        c for c in merged.columns
        if c not in _NON_FEATURE_COLS and merged[c].dtype in (np.float64, np.float32, float, int, np.int64)
    ]
    log.info("Feature columns (%d): %s", len(feature_cols), feature_cols[:10], )

    # Fill NaN in features with 0 (e.g. first sentiment rows)
    merged[feature_cols] = merged[feature_cols].fillna(0.0)

    # ── Chronological 70/15/15 split across all tickers combined ─────────── #
    # Sort globally by date for split boundaries
    merged = merged.sort_values(["date", "ticker"]).reset_index(drop=True)
    n = len(merged)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = merged.iloc[:train_end]
    val_df = merged.iloc[train_end:val_end]
    test_df = merged.iloc[val_end:]

    log.info(
        "Split sizes — train: %d, val: %d, test: %d",
        len(train_df), len(val_df), len(test_df),
    )

    # ── Fit scaler on training data only ─────────────────────────────────── #
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols].values)

    def _scale(df: pd.DataFrame) -> np.ndarray:
        return scaler.transform(df[feature_cols].values).astype(np.float32)

    # ── Build sliding-window sequences per ticker, then concatenate ───────── #
    def _make_sequences(
        df: pd.DataFrame,
        tgt_col: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        X_parts, y_parts = [], []
        for _, grp in df.groupby("ticker"):
            grp = grp.sort_values("date")
            scaled = scaler.transform(grp[feature_cols].values).astype(np.float32)
            tgt = grp[tgt_col].values
            X_t, y_t = _sliding_windows(scaled, tgt, lookback)
            if len(X_t):
                X_parts.append(X_t)
                y_parts.append(y_t)
        if not X_parts:
            F = len(feature_cols)
            return np.empty((0, lookback, F), dtype=np.float32), np.empty((0,))
        return np.concatenate(X_parts), np.concatenate(y_parts)

    log.info("Building sliding-window sequences (lookback=%d) …", lookback)
    X_train, y_train = _make_sequences(train_df, target)
    X_val, y_val = _make_sequences(val_df, target)
    X_test, y_test = _make_sequences(test_df, target)

    log.info("Sequence shapes — X_train=%s, X_val=%s, X_test=%s",
             X_train.shape, X_val.shape, X_test.shape)

    # ── Metadata ─────────────────────────────────────────────────────────── #
    metadata: dict[str, Any] = {
        "target": target,
        "lookback_days": lookback,
        "feature_count": len(feature_cols),
        "train_samples": int(len(X_train)),
        "val_samples": int(len(X_val)),
        "test_samples": int(len(X_test)),
        "train_date_start": str(train_df["date"].min().date()),
        "train_date_end": str(train_df["date"].max().date()),
        "val_date_start": str(val_df["date"].min().date()),
        "val_date_end": str(val_df["date"].max().date()),
        "test_date_start": str(test_df["date"].min().date()),
        "test_date_end": str(test_df["date"].max().date()),
        "tickers": list(merged["ticker"].unique()),
        "class_distribution": (
            {str(int(k)): int(v)
             for k, v in zip(*np.unique(y_train, return_counts=True))}
            if len(y_train) > 0 else {}
        ),
    }

    # ── Save artefacts ───────────────────────────────────────────────────── #
    if save:
        output_dir.mkdir(parents=True, exist_ok=True)

        np.save(output_dir / "X_train.npy", X_train)
        np.save(output_dir / "y_train.npy", y_train)
        np.save(output_dir / "X_val.npy", X_val)
        np.save(output_dir / "y_val.npy", y_val)
        np.save(output_dir / "X_test.npy", X_test)
        np.save(output_dir / "y_test.npy", y_test)

        with open(output_dir / "feature_names.json", "w") as fh:
            json.dump(feature_cols, fh, indent=2)

        with open(output_dir / "scaler.pkl", "wb") as fh:
            pickle.dump(scaler, fh)

        with open(output_dir / "dataset_metadata.json", "w") as fh:
            json.dump(metadata, fh, indent=2)

        log.info("Saved all dataset artefacts to %s", output_dir)

    log.info("Dataset build complete. %s", metadata)

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "feature_names": feature_cols,
        "scaler": scaler,
        "metadata": metadata,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build ML dataset from features")
    parser.add_argument(
        "--target",
        default="direction_3class",
        choices=["direction", "direction_3class", "volatility_spike", "next_day_return"],
        help="Target variable to predict",
    )
    parser.add_argument("--no-save", action="store_true", help="Skip writing files")
    args = parser.parse_args()

    result = build_dataset(target=args.target, save=not args.no_save)
    print(f"\nX_train: {result['X_train'].shape}")
    print(f"X_val:   {result['X_val'].shape}")
    print(f"X_test:  {result['X_test'].shape}")
    print(f"Features ({len(result['feature_names'])}): {result['feature_names'][:8]} …")
    print(f"\nMetadata:\n{json.dumps(result['metadata'], indent=2)}")
