"""Vanilla RNN model for market movement prediction."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.base_model import BaseSequenceModel


class RNNModel(BaseSequenceModel):
    """Two-layer vanilla RNN with dropout and a fully-connected output head.

    Args:
        input_dim: Features per time-step.
        hidden_dim: RNN hidden size (default 128).
        num_layers: Stacked RNN layers (default 2).
        output_dim: Output classes or 1 for regression (default 3).
        dropout: Inter-layer dropout (default 0.3).
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

        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            nonlinearity="tanh",
        )
        self.dropout_layer = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward pass.

        Args:
            x: Shape (batch, seq_len, input_dim).

        Returns:
            Logits of shape (batch, output_dim).
        """
        # out: (batch, seq_len, hidden_dim)
        # h_n: (num_layers, batch, hidden_dim)
        _, h_n = self.rnn(x)

        # Take the last layer's hidden state
        last_hidden = h_n[-1]                   # (batch, hidden_dim)
        out = self.dropout_layer(last_hidden)
        return self.fc(out)                      # (batch, output_dim)

    def get_model_name(self) -> str:
        return "rnn"


if __name__ == "__main__":
    batch, seq_len, features = 8, 30, 45
    model = RNNModel(input_dim=features, hidden_dim=128, num_layers=2, output_dim=3)
    x = torch.randn(batch, seq_len, features)
    out = model(x)
    print(f"RNN output shape: {out.shape}  (expected [{batch}, 3])")
    print(f"Parameters: {model.count_parameters():,}")
