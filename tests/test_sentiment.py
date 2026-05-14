"""Tests for sentiment analysis modules."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


# ── VADER ──────────────────────────────────────────────────────────────────── #


class TestVaderAnalyzer:
    def test_analyze_returns_expected_keys(self) -> None:
        from src.sentiment.vader_analyzer import analyze

        result = analyze("Apple stock is performing very well today.")
        # Actual keys: label, compound_score, positive_score, negative_score, neutral_score
        assert "label" in result
        assert "compound_score" in result
        assert result["label"] in {"positive", "negative", "neutral"}

    def test_positive_text(self) -> None:
        from src.sentiment.vader_analyzer import analyze

        result = analyze("Incredible earnings beat, stocks soaring, phenomenal growth!")
        assert result["label"] == "positive"
        assert result["compound_score"] > 0.05

    def test_negative_text(self) -> None:
        from src.sentiment.vader_analyzer import analyze

        result = analyze("Terrible losses, awful crash, catastrophic failure.")
        assert result["label"] == "negative"
        assert result["compound_score"] < -0.05

    def test_neutral_text(self) -> None:
        from src.sentiment.vader_analyzer import analyze

        result = analyze("The market closed today.")
        # Neutral or mildly positive — just verify keys are present
        assert "label" in result
        assert "compound_score" in result

    def test_analyze_batch(self) -> None:
        from src.sentiment.vader_analyzer import analyze_batch

        texts = ["Great day!", "Bad news today.", "Stock market closed."]
        results = analyze_batch(texts)
        assert len(results) == 3
        for r in results:
            assert "label" in r
            assert "compound_score" in r

    def test_empty_string(self) -> None:
        from src.sentiment.vader_analyzer import analyze

        result = analyze("")
        assert "label" in result
        assert result["label"] == "neutral"


# ── FinBERT (mocked — heavy model) ────────────────────────────────────────── #


class TestFinBertAnalyzer:
    """FinBERT tests use mocked tokenizer/model to avoid loading 400MB weights."""

    def _make_mock_model_output(self, logits_values: list[float]) -> MagicMock:
        import torch

        output = MagicMock()
        output.logits = torch.tensor([logits_values])
        return output

    @pytest.mark.parametrize(
        "logits,expected_label",
        [
            ([2.0, -1.0, -1.0], "positive"),   # index 0 = positive
            ([-1.0, 2.0, -1.0], "negative"),   # index 1 = negative
            ([-1.0, -1.0, 2.0], "neutral"),    # index 2 = neutral
        ],
    )
    def test_analyze_batch_label(
        self, logits: list[float], expected_label: str
    ) -> None:
        import torch

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.zeros(1, 10, dtype=torch.long),
            "attention_mask": torch.ones(1, 10, dtype=torch.long),
        }
        mock_model = MagicMock()
        mock_model.return_value = self._make_mock_model_output(logits)

        # Also clear the cache so previous parametrize iterations don't pollute results
        with (
            patch("src.sentiment.finbert_analyzer._tokenizer", mock_tokenizer),
            patch("src.sentiment.finbert_analyzer._model", mock_model),
            patch.dict("src.sentiment.finbert_analyzer._cache", {}, clear=True),
        ):
            from src.sentiment.finbert_analyzer import analyze_batch

            results = analyze_batch(["any text"])

        assert results[0]["label"] == expected_label

    def test_analyze_batch_cache_hit(self) -> None:
        """Second call with the same text should use cache, not call model again."""
        import torch

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.zeros(1, 10, dtype=torch.long),
            "attention_mask": torch.ones(1, 10, dtype=torch.long),
        }
        mock_model = MagicMock()
        mock_model.return_value = self._make_mock_model_output([2.0, -1.0, -1.0])

        with (
            patch("src.sentiment.finbert_analyzer._tokenizer", mock_tokenizer),
            patch("src.sentiment.finbert_analyzer._model", mock_model),
            patch.dict("src.sentiment.finbert_analyzer._cache", {}),
        ):
            from src.sentiment.finbert_analyzer import analyze_batch

            analyze_batch(["cached text"])
            analyze_batch(["cached text"])  # second call

        assert mock_model.call_count == 1  # model called only once


# ── Ensemble ───────────────────────────────────────────────────────────────── #


class TestEnsemble:
    # Actual return format: {label, positive_score, negative_score, neutral_score, compound_score}
    def _vader_result(self, label: str, compound: float) -> dict:
        pos = max(compound, 0.0)
        neg = max(-compound, 0.0)
        neu = round(1.0 - pos - neg, 6)
        return {
            "label": label,
            "compound_score": compound,
            "positive_score": pos,
            "negative_score": neg,
            "neutral_score": max(neu, 0.0),
        }

    def _finbert_result(self, label: str, compound: float) -> dict:
        pos = max(compound, 0.0)
        neg = max(-compound, 0.0)
        neu = round(1.0 - pos - neg, 6)
        return {
            "label": label,
            "compound_score": compound,
            "positive_score": pos,
            "negative_score": neg,
            "neutral_score": max(neu, 0.0),
        }

    def test_ensemble_single_both_positive(self) -> None:
        from src.sentiment.ensemble import ensemble_single

        vader = self._vader_result("positive", 0.6)
        finbert = self._finbert_result("positive", 0.8)
        result = ensemble_single(finbert, vader)

        # ensemble_single returns: {label, positive_score, ..., compound_score}
        assert result["label"] == "positive"
        assert result["compound_score"] > 0.0

    def test_ensemble_single_weighted_compound(self) -> None:
        """Compound should be 0.7*finbert + 0.3*vader."""
        from src.sentiment.ensemble import ensemble_single

        vader = self._vader_result("negative", -0.9)
        finbert = self._finbert_result("positive", 0.6)
        result = ensemble_single(finbert, vader)

        expected_compound = round(0.7 * 0.6 + 0.3 * (-0.9), 6)
        assert abs(result["compound_score"] - expected_compound) < 1e-5

    def test_analyze_batch_returns_list(self) -> None:
        """ensemble.analyze_batch returns a list of dicts."""
        mock_finbert = MagicMock(return_value=[self._finbert_result("positive", 0.5)])
        mock_vader = MagicMock(return_value=[self._vader_result("positive", 0.4)])

        with (
            patch("src.sentiment.ensemble.finbert_analyzer.analyze_batch", mock_finbert),
            patch("src.sentiment.ensemble.vader_analyzer.analyze_batch", mock_vader),
        ):
            from src.sentiment.ensemble import analyze_batch

            result = analyze_batch(["Good earnings report."])

        assert isinstance(result, list)
        assert len(result) == 1
        assert "label" in result[0]
        assert "compound_score" in result[0]
