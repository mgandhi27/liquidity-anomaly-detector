"""
model.py
========

PyTorch recurrent architecture for forecasting liquidity shocks from
sequences of engineered limit-order-book features.

The network is deliberately loss-agnostic: it emits a single raw linear
score (logit) per sequence. Downstream, `engine.ModelTrainer` pairs this
with `BCEWithLogitsLoss` for the binary "liquidity shock in the next W
periods" formulation, or `MSELoss` for a continuous liquidity-drop-score
regression -- keeping the network reusable across both target
definitions without re-architecting it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class LSTMConfig:
    """Configuration for `LiquidityAnomalyLSTM`."""

    input_size: int
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    sequence_length: int = 50
    output_size: int = 1
    bidirectional: bool = False


class LiquidityAnomalyLSTM(nn.Module):
    """
    Stacked LSTM classifier/regressor over Level-2 order-book sequences.

    Input tensor shape: (batch, sequence_length, input_size).
    Output tensor shape: (batch, output_size) -- raw logits/scores, NOT
    passed through a sigmoid; see `predict_proba` for the probability
    form used at inference time for the classification task.
    """

    def __init__(self, config: LSTMConfig) -> None:
        super().__init__()
        self._validate_config(config)
        self.config = config

        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            bidirectional=config.bidirectional,
        )
        num_directions = 2 if config.bidirectional else 1
        self.head_dropout = nn.Dropout(p=config.dropout)
        self.fc = nn.Linear(config.hidden_size * num_directions, config.output_size)

        self._init_weights()

    @staticmethod
    def _validate_config(config: LSTMConfig) -> None:
        if config.input_size <= 0:
            raise ValueError("input_size must be a positive integer.")
        if config.hidden_size <= 0:
            raise ValueError("hidden_size must be a positive integer.")
        if config.num_layers <= 0:
            raise ValueError("num_layers must be a positive integer.")
        if not (0.0 <= config.dropout < 1.0):
            raise ValueError("dropout must be in [0, 1).")
        if config.sequence_length <= 0:
            raise ValueError("sequence_length must be a positive integer.")
        if config.output_size <= 0:
            raise ValueError("output_size must be a positive integer.")

    def _init_weights(self) -> None:
        """
        Xavier init for input-to-hidden weights, orthogonal for
        hidden-to-hidden, zero biases with the forget-gate bias set to 1
        (Jozefowicz et al., 2015) to encourage long-range gradient flow
        early in training.
        """
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0.0)
                gate_size = param.size(0) // 4
                param.data[gate_size : 2 * gate_size].fill_(1.0)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Parameters
        ----------
        x:
            Float tensor of shape (batch, seq_len, input_size).
        lengths:
            Optional 1D tensor of true (unpadded) sequence lengths per
            batch element, enabling correct handling of variable-length
            windows via packed sequences. If omitted, all `seq_len`
            steps are assumed valid.

        Returns
        -------
        torch.Tensor
            Raw scores of shape (batch, output_size).
        """
        if x.dim() != 3:
            raise ValueError(f"Expected input of shape (batch, seq_len, features); got {tuple(x.shape)}.")
        if x.size(-1) != self.config.input_size:
            raise ValueError(
                f"Feature dimension mismatch: model expects {self.config.input_size}, got {x.size(-1)}."
            )

        if lengths is not None:
            packed_input = nn.utils.rnn.pack_padded_sequence(
                x, lengths.detach().cpu(), batch_first=True, enforce_sorted=False
            )
            _, (h_n, _) = self.lstm(packed_input)
        else:
            _, (h_n, _) = self.lstm(x)

        if self.config.bidirectional:
            final_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            final_hidden = h_n[-1]

        return self.fc(self.head_dropout(final_hidden))

    def predict_proba(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sigmoid-activated probability of a liquidity shock (classification use case)."""
        self.eval()
        with torch.no_grad():
            return torch.sigmoid(self.forward(x, lengths))
