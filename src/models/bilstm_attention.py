"""Bidirectional LSTM with self-attention — bonus 4th model.

Architecture:
  1. BiLSTM encodes all time-steps → (batch, seq_len, 2 * hidden_dim)
  2. Additive self-attention scores each time-step
  3. Softmax-normalised weighted sum → context vector (batch, 2 * hidden_dim)
  4. Dropout → FC → logits
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base_model import BaseSequenceModel


class BiLSTMAttentionModel(BaseSequenceModel):
    """Bidirectional LSTM with additive self-attention output head.

    Args:
        input_dim: Features per time-step.
        hidden_dim: LSTM hidden size *per direction* (default 128).
            The BiLSTM outputs 2 * hidden_dim at each step.
        num_layers: Stacked BiLSTM layers (default 2).
        output_dim: Output classes or 1 for regression (default 3).
        dropout: Inter-layer and pre-FC dropout (default 0.3).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        output_dim: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__(input_dim, hidden_dim, num_layers, output_dim, dropout)

        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Attention: project 2*hidden_dim → 1 score per time-step
        self.attention_fc = nn.Linear(hidden_dim * 2, 1, bias=False)

        self.dropout_layer = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def _attention(self, lstm_out: torch.Tensor) -> torch.Tensor:
        """Compute attention-weighted context vector.

        Args:
            lstm_out: Shape (batch, seq_len, 2 * hidden_dim).

        Returns:
            Context vector of shape (batch, 2 * hidden_dim).
        """
        # energy: (batch, seq_len, 1)
        energy = self.attention_fc(lstm_out)
        # weights: (batch, seq_len, 1)
        weights = F.softmax(energy, dim=1)
        # context: (batch, 2 * hidden_dim)
        context = (lstm_out * weights).sum(dim=1)
        return context

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward pass.

        Args:
            x: Shape (batch, seq_len, input_dim).

        Returns:
            Logits of shape (batch, output_dim).
        """
        # lstm_out: (batch, seq_len, 2 * hidden_dim)
        lstm_out, _ = self.bilstm(x)

        context = self._attention(lstm_out)      # (batch, 2 * hidden_dim)
        out = self.dropout_layer(context)
        return self.fc(out)                       # (batch, output_dim)

    def get_model_name(self) -> str:
        return "bilstm_attention"


if __name__ == "__main__":
    batch, seq_len, features = 8, 30, 45
    model = BiLSTMAttentionModel(
        input_dim=features, hidden_dim=128, num_layers=2, output_dim=3
    )
    x = torch.randn(batch, seq_len, features)
    out = model(x)
    print(f"BiLSTM-Attention output shape: {out.shape}  (expected [{batch}, 3])")
    print(f"Parameters: {model.count_parameters():,}")
