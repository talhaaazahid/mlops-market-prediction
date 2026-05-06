"""Abstract base class shared by all sequence models."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseSequenceModel(ABC, nn.Module):
    """Abstract base for RNN / LSTM / GRU / BiLSTM-Attention models.

    All concrete models must accept the same constructor signature and
    implement `forward` + `get_model_name`.

    Args:
        input_dim: Number of features per time-step.
        hidden_dim: Hidden state dimensionality (default 128).
        num_layers: Number of stacked recurrent layers (default 2).
        output_dim: Number of output classes or 1 for regression.
        dropout: Dropout probability applied between layers (default 0.3).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        output_dim: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.dropout = dropout

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim).

        Returns:
            Logits tensor of shape (batch, output_dim).
        """

    @abstractmethod
    def get_model_name(self) -> str:
        """Return a short identifier string (e.g. 'rnn', 'lstm')."""

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# Ensemble Voting Strategy
VOTING_STRATEGY = 'weighted_majority'
VOTING_WEIGHTS = {'lstm': 0.35, 'gru': 0.25, 'bilstm': 0.4}

