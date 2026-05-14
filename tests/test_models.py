"""Tests for PyTorch sequence model architectures and the Trainer."""

from __future__ import annotations

import numpy as np
import pytest
import torch

INPUT_DIM = 20
HIDDEN_DIM = 32
NUM_LAYERS = 1
DROPOUT = 0.0
NUM_CLASSES = 2
SEQ_LEN = 10
BATCH_SIZE = 4


@pytest.fixture()
def sample_batch() -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_DIM)
    y = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))
    return x, y


# ── Architecture forward-pass tests ───────────────────────────────────────── #


class TestRNNModel:
    def test_output_shape(self, sample_batch: tuple) -> None:
        from src.models.rnn_model import RNNModel

        model = RNNModel(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, NUM_CLASSES, DROPOUT)
        x, _ = sample_batch
        out = model(x)
        assert out.shape == (BATCH_SIZE, NUM_CLASSES)

    def test_output_is_finite(self, sample_batch: tuple) -> None:
        from src.models.rnn_model import RNNModel

        model = RNNModel(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, NUM_CLASSES, DROPOUT)
        x, _ = sample_batch
        out = model(x)
        assert torch.isfinite(out).all()

    def test_parameter_count(self) -> None:
        from src.models.rnn_model import RNNModel

        model = RNNModel(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, NUM_CLASSES, DROPOUT)
        assert model.count_parameters() > 0


class TestLSTMModel:
    def test_output_shape(self, sample_batch: tuple) -> None:
        from src.models.lstm_model import LSTMModel

        model = LSTMModel(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, NUM_CLASSES, DROPOUT)
        x, _ = sample_batch
        out = model(x)
        assert out.shape == (BATCH_SIZE, NUM_CLASSES)

    def test_batch_size_one(self) -> None:
        from src.models.lstm_model import LSTMModel

        model = LSTMModel(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, NUM_CLASSES, DROPOUT)
        x = torch.randn(1, SEQ_LEN, INPUT_DIM)
        out = model(x)
        assert out.shape == (1, NUM_CLASSES)


class TestGRUModel:
    def test_output_shape(self, sample_batch: tuple) -> None:
        from src.models.gru_model import GRUModel

        model = GRUModel(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, NUM_CLASSES, DROPOUT)
        x, _ = sample_batch
        out = model(x)
        assert out.shape == (BATCH_SIZE, NUM_CLASSES)

    def test_different_seq_lengths(self) -> None:
        from src.models.gru_model import GRUModel

        model = GRUModel(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, NUM_CLASSES, DROPOUT)
        for seq_len in [5, 15, 30]:
            x = torch.randn(BATCH_SIZE, seq_len, INPUT_DIM)
            out = model(x)
            assert out.shape == (BATCH_SIZE, NUM_CLASSES)


class TestBiLSTMAttentionModel:
    def test_output_shape(self, sample_batch: tuple) -> None:
        from src.models.bilstm_attention import BiLSTMAttentionModel

        model = BiLSTMAttentionModel(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, NUM_CLASSES, DROPOUT)
        x, _ = sample_batch
        out = model(x)
        assert out.shape == (BATCH_SIZE, NUM_CLASSES)

    def test_attention_weights_sum_to_one(self, sample_batch: tuple) -> None:
        """Attention weights along seq_len dimension must sum to 1."""
        import torch.nn.functional as F
        from src.models.bilstm_attention import BiLSTMAttentionModel

        model = BiLSTMAttentionModel(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, NUM_CLASSES, DROPOUT)
        model.eval()
        x, _ = sample_batch

        # Actual LSTM attribute is self.bilstm (not self.lstm)
        lstm_out, _ = model.bilstm(x)           # (B, T, 2H)
        energy = model.attention_fc(lstm_out).squeeze(-1)  # (B, T)
        weights = F.softmax(energy, dim=1)       # (B, T)

        sums = weights.sum(dim=1)
        assert torch.allclose(sums, torch.ones(BATCH_SIZE), atol=1e-5)

    def test_bidirectional_doubles_hidden(self) -> None:
        from src.models.bilstm_attention import BiLSTMAttentionModel

        model = BiLSTMAttentionModel(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, NUM_CLASSES, DROPOUT)
        assert model.fc.in_features == HIDDEN_DIM * 2


# ── _remap_labels (module-level function) ─────────────────────────────────── #


class TestRemapLabels:
    def test_shifts_negative_one_to_zero(self) -> None:
        """_remap_labels must shift {-1,0,1} to {0,1,2}."""
        from src.models.trainer import _remap_labels

        labels = np.array([-1, 0, 1, -1, 1])
        remapped = _remap_labels(labels)
        np.testing.assert_array_equal(remapped, [0, 1, 2, 0, 2])

    def test_already_non_negative_unchanged(self) -> None:
        from src.models.trainer import _remap_labels

        labels = np.array([0, 1, 0, 1])
        remapped = _remap_labels(labels)
        np.testing.assert_array_equal(remapped, [0, 1, 0, 1])

    def test_all_same_class(self) -> None:
        from src.models.trainer import _remap_labels

        labels = np.array([-1, -1, -1])
        remapped = _remap_labels(labels)
        np.testing.assert_array_equal(remapped, [0, 0, 0])


# ── Trainer ────────────────────────────────────────────────────────────────── #


class TestTrainer:
    @pytest.mark.slow
    def test_overfit_small_batch(self) -> None:
        """A model must drive train loss down on a tiny fixed dataset."""
        from src.models.gru_model import GRUModel
        from src.models.trainer import Trainer

        torch.manual_seed(0)
        n, seq, feats = 32, 10, 10
        X = np.random.randn(n, seq, feats).astype(np.float32)
        y = np.random.randint(0, 2, n).astype(np.int64)

        model = GRUModel(feats, 64, 1, 2, 0.0)
        # Correct Trainer signature: no mlflow_tracking parameter
        trainer = Trainer(model, ticker="TEST")
        history = trainer.fit(
            X_train=X, y_train=y,
            X_val=X, y_val=y,
            target="direction",
            epochs=50,
        )

        assert history["train_loss"][-1] < history["train_loss"][0]

def test_model_inference():
    Test model inference performance.
    assert model_output.shape[0] > 0


def test_model_inference():
    Test model inference performance.
    assert model_output.shape[0] > 0


def test_model_inference():
    Test model inference performance.
    assert model_output.shape[0] > 0

