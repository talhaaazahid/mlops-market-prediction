"""VADER sentiment analyzer — baseline model using NLTK's SentimentIntensityAnalyzer.

Thresholds (compound score):
  compound > +0.05  →  positive
  compound < -0.05  →  negative
  otherwise         →  neutral
"""

from __future__ import annotations

from typing import Any

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from src.utils.logger import get_logger

log = get_logger(__name__)

_NEUTRAL_RESULT: dict[str, Any] = {
    "label": "neutral",
    "compound_score": 0.0,
    "positive_score": 0.0,
    "negative_score": 0.0,
    "neutral_score": 1.0,
}

# Module-level singleton to avoid repeated init overhead
_sia: SentimentIntensityAnalyzer | None = None


def _get_sia() -> SentimentIntensityAnalyzer:
    """Return the VADER SentimentIntensityAnalyzer, downloading lexicon once.

    Returns:
        Initialized SentimentIntensityAnalyzer instance.
    """
    global _sia
    if _sia is None:
        import nltk

        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            log.info("Downloading VADER lexicon …")
            nltk.download("vader_lexicon", quiet=True)
        _sia = SentimentIntensityAnalyzer()
        log.info("VADER SentimentIntensityAnalyzer initialized")
    return _sia


def _compound_to_label(compound: float) -> str:
    """Map a VADER compound score to a sentiment label.

    Args:
        compound: Float in [-1, 1].

    Returns:
        One of "positive", "negative", "neutral".
    """
    if compound > 0.05:
        return "positive"
    if compound < -0.05:
        return "negative"
    return "neutral"


def analyze(text: str) -> dict[str, Any]:
    """Score a single text string with VADER.

    Args:
        text: Input string. Empty or whitespace-only returns a neutral result.

    Returns:
        Dict with keys: label, compound_score, positive_score, negative_score,
        neutral_score.
    """
    if not text or not text.strip():
        return _NEUTRAL_RESULT.copy()

    sia = _get_sia()
    scores = sia.polarity_scores(text.strip())

    compound = round(scores["compound"], 6)
    return {
        "label": _compound_to_label(compound),
        "compound_score": compound,
        "positive_score": round(scores["pos"], 6),
        "negative_score": round(scores["neg"], 6),
        "neutral_score": round(scores["neu"], 6),
    }


def analyze_batch(texts: list[str]) -> list[dict[str, Any]]:
    """Score a list of texts with VADER.

    Args:
        texts: List of raw text strings.

    Returns:
        List of result dicts in the same order as `texts`.
    """
    return [analyze(t) for t in texts]


if __name__ == "__main__":
    samples = [
        "Apple reports record quarterly earnings, beating analyst expectations.",
        "Tesla stock crashes after CEO announces massive layoffs.",
        "Markets remain flat ahead of Federal Reserve decision.",
        "",
    ]
    for sample, result in zip(samples, analyze_batch(samples)):
        print(
            f"[{result['label']:8s}] compound={result['compound_score']:+.3f} "
            f"| {sample[:60] or '(empty)'}"
        )
