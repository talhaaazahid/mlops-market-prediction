"""FinBERT sentiment analyzer using ProsusAI/finbert from HuggingFace.

Loads the model once (singleton pattern), processes texts in batches of 32
with tqdm progress bars, handles GPU/CPU fallback, caches results by text hash,
and truncates inputs to 512 tokens.
"""

from __future__ import annotations

import hashlib
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import FINBERT_MODEL, get_params
from src.utils.logger import get_logger

log = get_logger(__name__)

# Singleton holders
_tokenizer: Any = None
_model: Any = None
_device: torch.device | None = None

# In-process cache: text_hash → result dict
_cache: dict[str, dict[str, Any]] = {}

_LABEL_MAP = {0: "positive", 1: "negative", 2: "neutral"}
_NEUTRAL_RESULT: dict[str, Any] = {
    "label": "neutral",
    "positive_score": 0.0,
    "negative_score": 0.0,
    "neutral_score": 1.0,
    "compound_score": 0.0,
}


def _get_model() -> tuple[Any, Any, torch.device]:
    """Return (tokenizer, model, device), loading once on first call.

    Returns:
        Tuple of (AutoTokenizer, AutoModelForSequenceClassification, device).
    """
    global _tokenizer, _model, _device

    if _tokenizer is None:
        log.info("Loading FinBERT model: %s", FINBERT_MODEL)
        _tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
        _model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
        _model.eval()

        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info("FinBERT running on device: %s", _device)
        _model.to(_device)

    return _tokenizer, _model, _device  # type: ignore[return-value]


def _text_hash(text: str) -> str:
    """Return a short MD5 hex digest for cache keying.

    Args:
        text: Input string.

    Returns:
        32-character hex string.
    """
    return hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()


def _scores_to_result(scores: torch.Tensor) -> dict[str, Any]:
    """Convert a softmax probability tensor (shape [3]) to a result dict.

    FinBERT output order: [positive, negative, neutral] (index 0, 1, 2).

    Args:
        scores: 1-D float tensor with 3 softmax probabilities.

    Returns:
        Dict with label, positive_score, negative_score, neutral_score,
        compound_score.
    """
    pos = float(scores[0])
    neg = float(scores[1])
    neu = float(scores[2])
    label_idx = int(scores.argmax().item())
    return {
        "label": _LABEL_MAP[label_idx],
        "positive_score": round(pos, 6),
        "negative_score": round(neg, 6),
        "neutral_score": round(neu, 6),
        "compound_score": round(pos - neg, 6),
    }


def analyze_batch(texts: list[str], batch_size: int | None = None) -> list[dict[str, Any]]:
    """Run FinBERT sentiment on a list of texts.

    Args:
        texts: Raw text strings. Null / empty strings return a neutral result.
        batch_size: Override default batch size from params.yaml.

    Returns:
        List of result dicts in the same order as `texts`.
    """
    params = get_params()
    if batch_size is None:
        batch_size = int(params["sentiment"].get("batch_size", 32))

    tokenizer, model, device = _get_model()
    results: list[dict[str, Any]] = []

    indices_to_run: list[int] = []
    hashes: list[str] = []

    # Split into cached vs. needs inference
    for i, text in enumerate(texts):
        if not text or not text.strip():
            results.append(_NEUTRAL_RESULT.copy())
            hashes.append("")
            continue
        h = _text_hash(text.strip())
        hashes.append(h)
        if h in _cache:
            results.append(_cache[h])
        else:
            results.append({})  # placeholder
            indices_to_run.append(i)

    if not indices_to_run:
        return results

    # Batch inference for uncached texts
    texts_to_run = [texts[i].strip() for i in indices_to_run]

    with torch.no_grad():
        for batch_start in tqdm(
            range(0, len(texts_to_run), batch_size),
            desc="FinBERT inference",
            unit="batch",
            leave=False,
        ):
            batch_texts = texts_to_run[batch_start : batch_start + batch_size]
            try:
                encoding = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True,
                )
                encoding = {k: v.to(device) for k, v in encoding.items()}
                logits = model(**encoding).logits
                probs = torch.softmax(logits, dim=-1).cpu()

                for j, score_row in enumerate(probs):
                    global_idx = indices_to_run[batch_start + j]
                    res = _scores_to_result(score_row)
                    h = hashes[global_idx]
                    _cache[h] = res
                    results[global_idx] = res

            except Exception as exc:
                log.error("FinBERT batch error at offset %d: %s", batch_start, exc)
                for j in range(len(batch_texts)):
                    global_idx = indices_to_run[batch_start + j]
                    if not results[global_idx]:
                        results[global_idx] = _NEUTRAL_RESULT.copy()

    return results


def analyze(text: str) -> dict[str, Any]:
    """Analyze a single text string.

    Args:
        text: Input text (may be empty or null-like).

    Returns:
        Sentiment result dict.
    """
    return analyze_batch([text])[0]


if __name__ == "__main__":
    samples = [
        "Apple reports record quarterly earnings, beating analyst expectations.",
        "Tesla stock crashes after CEO announces massive layoffs.",
        "Markets remain flat ahead of Federal Reserve decision.",
        "",  # empty — should return neutral
    ]
    for sample, result in zip(samples, analyze_batch(samples)):
        print(f"[{result['label']:8s}] compound={result['compound_score']:+.3f} | {sample[:60]}")
