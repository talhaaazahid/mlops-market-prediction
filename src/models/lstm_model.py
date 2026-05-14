"""LSTM model for market movement prediction."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.base_model import BaseSequenceModel


class LSTMModel(BaseSequenceModel):
    """Two-layer LSTM with dropout and a fully-connected output head.

    Args:
        input_dim: Features per time-step.
        hidden_dim: LSTM hidden size (default 128).
        num_layers: Stacked LSTM layers (default 2).
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

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
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
        # h_n: (num_layers, batch, hidden_dim)
        # c_n: (num_layers, batch, hidden_dim)  — cell state carried through
        _, (h_n, _c_n) = self.lstm(x)

        last_hidden = h_n[-1]                    # (batch, hidden_dim)
        out = self.dropout_layer(last_hidden)
        return self.fc(out)                       # (batch, output_dim)

    def get_model_name(self) -> str:
        return "lstm"


if __name__ == "__main__":
    batch, seq_len, features = 8, 30, 45
    model = LSTMModel(input_dim=features, hidden_dim=128, num_layers=2, output_dim=3)
    x = torch.randn(batch, seq_len, features)
    out = model(x)
    print(f"LSTM output shape: {out.shape}  (expected [{batch}, 3])")
    print(f"Parameters: {model.count_parameters():,}")
