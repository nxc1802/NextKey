"""BiLSTM dual-head character tagger for Vietnamese compact-text restoration.

Architecture:
    Embedding -> Dropout -> Bidirectional LSTM -> Dropout -> [Diacritic Head, Boundary Head]
"""

from __future__ import annotations

import torch
import torch.nn as nn

from nextkey.models.base import BaseCharTagger, register_model


@register_model("bilstm")
class BiLSTMCharTagger(BaseCharTagger):
    """Bidirectional LSTM backbone with dual classification heads."""

    def __init__(
        self,
        vocab_size: int,
        num_target_classes: int,
        embed_dim: int = 64,
        hidden_dim: int = 112,
        num_layers: int = 1,
        dropout: float = 0.1,
        num_boundary_classes: int = 1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.input_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_dropout = nn.Dropout(dropout)

        # Dual heads
        self.diacritic_head = nn.Linear(hidden_dim * 2, num_target_classes)
        self.boundary_head = nn.Linear(hidden_dim * 2, num_boundary_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        x = self.embedding(input_ids)
        x = self.input_dropout(x)

        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False,
            )
            lstm_out, _ = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=True, total_length=input_ids.shape[1],
            )
        else:
            lstm_out, _ = self.lstm(x)

        lstm_out = self.output_dropout(lstm_out)

        diacritic_logits = self.diacritic_head(lstm_out)
        boundary_logits = self.boundary_head(lstm_out).squeeze(-1)

        return {
            "diacritic_logits": diacritic_logits,
            "boundary_logits": boundary_logits,
        }
