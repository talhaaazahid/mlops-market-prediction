"""Weighted ensemble of FinBERT (0.7) + VADER (0.3) sentiment scores.

Also exposes `run_sentiment_pipeline()` — the Phase 2 pipeline runner that
reads all_text_data.csv, scores every row with both models, and writes
data/processed/sentiment_labeled.csv.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from src.config import (
    ALL_TEXT_FILE,
    PROCESSED_DIR,
    get_params,
)
from src.sentiment import finbert_analyzer, vader_analyzer
from src.utils.logger import get_logger

log = get_logger(__name__)

_SENTIMENT_LABELED_FILE = PROCESSED_DIR / "sentiment_labeled.csv"

_LABELS = ["positive", "negative", "neutral"]


def _label_to_vec(label: str) -> dict[str, float]:
    """One-hot-like vector for a label string.

    Args:
        label: One of "positive", "negative", "neutral".

    Returns:
        Dict {positive: float, negative: float, neutral: float}.
    """
    return {lbl: 1.0 if lbl == label else 0.0 for lbl in _LABELS}


def ensemble_single(
    finbert_result: dict[str, Any],
    vader_result: dict[str, Any],
    finbert_weight: float = 0.7,
    vader_weight: float = 0.3,
) -> dict[str, Any]:
    """Combine one FinBERT result and one VADER result into an ensemble score.

    Args:
        finbert_result: Dict from `finbert_analyzer.analyze()`.
        vader_result: Dict from `vader_analyzer.analyze()`.
        finbert_weight: Weight applied to FinBERT scores (default 0.7).
        vader_weight: Weight applied to VADER scores (default 0.3).

    Returns:
        Dict with keys: label, positive_score, negative_score, neutral_score,
        compound_score.
    """
    # Weighted average of per-class probabilities
    weighted: dict[str, float] = {}
    for lbl in _LABELS:
        fb_score = finbert_result.get(f"{lbl}_score", 0.0)
        vd_score = vader_result.get(f"{lbl}_score", 0.0)
        weighted[lbl] = finbert_weight * fb_score + vader_weight * vd_score

    best_label = max(weighted, key=lambda k: weighted[k])

    fb_compound = finbert_result.get("compound_score", 0.0)
    vd_compound = vader_result.get("compound_score", 0.0)
    ensemble_compound = round(
        finbert_weight * fb_compound + vader_weight * vd_compound, 6
    )

    return {
        "label": best_label,
        "positive_score": round(weighted["positive"], 6),
        "negative_score": round(weighted["negative"], 6),
        "neutral_score": round(weighted["neutral"], 6),
        "compound_score": ensemble_compound,
    }


def analyze_batch(
    texts: list[str],
    finbert_batch_size: int | None = None,
) -> list[dict[str, Any]]:
    """Run ensemble sentiment on a list of texts.

    Args:
        texts: Input strings.
        finbert_batch_size: Override FinBERT batch size.

    Returns:
        List of ensemble result dicts.
    """
    params = get_params()
    fw = float(params["sentiment"].get("finbert_weight", 0.7))
    vw = float(params["sentiment"].get("vader_weight", 0.3))

    log.info("Running FinBERT on %d texts …", len(texts))
    fb_results = finbert_analyzer.analyze_batch(texts, batch_size=finbert_batch_size)

    log.info("Running VADER on %d texts …", len(texts))
    vd_results = vader_analyzer.analyze_batch(texts)

    return [
        ensemble_single(fb, vd, fw, vw)
        for fb, vd in zip(fb_results, vd_results)
    ]


def run_sentiment_pipeline(
    input_path: Path | str = ALL_TEXT_FILE,
    output_path: Path | str = _SENTIMENT_LABELED_FILE,
) -> pd.DataFrame:
    """Score all texts in all_text_data.csv with FinBERT + VADER + ensemble.

    Reads `input_path`, runs both analyzers, merges results into a wide
    DataFrame, writes `output_path`, and logs label distribution stats.

    Args:
        input_path: Path to all_text_data.csv (unified text data).
        output_path: Destination CSV path.

    Returns:
        The labeled DataFrame.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}. "
            "Run scripts/run_ingestion.py first."
        )

    log.info("Reading %s …", input_path)
    df = pd.read_csv(input_path)

    if df.empty:
        log.warning("Input file is empty — nothing to label.")
        return df

    texts = df["text"].fillna("").tolist()
    params = get_params()
    fw = float(params["sentiment"].get("finbert_weight", 0.7))
    vw = float(params["sentiment"].get("vader_weight", 0.3))

    # FinBERT
    log.info("Running FinBERT on %d texts …", len(texts))
    fb_results = finbert_analyzer.analyze_batch(texts)

    # VADER
    log.info("Running VADER on %d texts …", len(texts))
    vd_results = vader_analyzer.analyze_batch(texts)

    # Build output columns
    fb_labels, fb_pos, fb_neg, fb_neu, fb_compound = [], [], [], [], []
    vd_labels, vd_pos, vd_neg, vd_neu, vd_compound = [], [], [], [], []
    ens_labels, ens_compound = [], []

    for fb, vd in tqdm(
        zip(fb_results, vd_results),
        total=len(texts),
        desc="Assembling results",
        unit="row",
        leave=False,
    ):
        ens = ensemble_single(fb, vd, fw, vw)

        fb_labels.append(fb["label"])
        fb_pos.append(fb["positive_score"])
        fb_neg.append(fb["negative_score"])
        fb_neu.append(fb["neutral_score"])
        fb_compound.append(fb["compound_score"])

        vd_labels.append(vd["label"])
        vd_pos.append(vd["positive_score"])
        vd_neg.append(vd["negative_score"])
        vd_neu.append(vd["neutral_score"])
        vd_compound.append(vd["compound_score"])

        ens_labels.append(ens["label"])
        ens_compound.append(ens["compound_score"])

    out = df.copy()
    out["finbert_label"] = fb_labels
    out["finbert_pos"] = fb_pos
    out["finbert_neg"] = fb_neg
    out["finbert_neu"] = fb_neu
    out["finbert_compound"] = fb_compound

    out["vader_label"] = vd_labels
    out["vader_pos"] = vd_pos
    out["vader_neg"] = vd_neg
    out["vader_neu"] = vd_neu
    out["vader_compound"] = vd_compound

    out["ensemble_label"] = ens_labels
    out["ensemble_compound"] = ens_compound

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    log.info("Wrote %d labeled rows to %s", len(out), output_path)

    # ── Distribution stats ────────────────────────────────────────────────── #
    log.info("─" * 50)
    log.info("SENTIMENT LABEL DISTRIBUTION")

    for model_col in ("finbert_label", "vader_label", "ensemble_label"):
        counts = out[model_col].value_counts()
        log.info(
            "%s: %s",
            model_col,
            {k: int(v) for k, v in counts.items()},
        )

    for src, grp in out.groupby("source"):
        ens_counts = grp["ensemble_label"].value_counts().to_dict()
        log.info("  source=%-30s %s", src, ens_counts)

    if "ticker_mentions" in out.columns:
        import json
        from collections import Counter

        ticker_sentiment: dict[str, list[float]] = {}
        for _, row in out.iterrows():
            try:
                tickers = json.loads(row.get("ticker_mentions") or "[]")
                score = row.get("ensemble_compound", 0.0)
                for t in tickers:
                    ticker_sentiment.setdefault(t, []).append(score)
            except (json.JSONDecodeError, TypeError):
                pass

        if ticker_sentiment:
            log.info("Ticker-level mean ensemble compound scores:")
            for ticker, scores in sorted(ticker_sentiment.items()):
                log.info(
                    "  %-6s  mean=%.4f  n=%d",
                    ticker,
                    sum(scores) / len(scores),
                    len(scores),
                )

    log.info("─" * 50)
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run sentiment pipeline")
    parser.add_argument(
        "--input",
        default=str(ALL_TEXT_FILE),
        help="Path to all_text_data.csv",
    )
    parser.add_argument(
        "--output",
        default=str(_SENTIMENT_LABELED_FILE),
        help="Output path for sentiment_labeled.csv",
    )
    args = parser.parse_args()

    result_df = run_sentiment_pipeline(args.input, args.output)
    print(f"\nLabeled {len(result_df)} rows.")
    print(result_df[["source", "finbert_label", "vader_label", "ensemble_label"]].head(10))

# Ensemble Weights
FINBERT_WEIGHT = 0.6
VADER_WEIGHT = 0.4
MIN_CONFIDENCE = 0.65


# Ensemble Weights
FINBERT_WEIGHT = 0.6
VADER_WEIGHT = 0.4
MIN_CONFIDENCE = 0.65

