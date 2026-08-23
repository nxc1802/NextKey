"""Tri-Head BiGRU Multi-Task Character Tagger for Vietnamese Restoration.

Architecture:
    Input (Noisy Chars) → Embedding → Dropout → Shared BiGRU Backbone → Dropout
                                                      │
                       ┌──────────────────────────────┼──────────────────────────────┐
                       ▼                              ▼                              ▼
              [Correction Head]                [Diacritic Head]               [Boundary Head]
          (Base Character Recovery)         (Accented Char Prediction)     (Word Boundary Flag)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from nextkey.models.base import BaseCharTagger, register_model


@register_model("tri_bigru")
class TriHeadBiGRUCharTagger(BaseCharTagger):
    """Bidirectional GRU backbone with 3 distinct task heads."""

    def __init__(
        self,
        vocab_size: int,
        num_target_classes: int,
        num_corr_classes: int | None = None,
        embed_dim: int = 48,
        hidden_dim: int = 96,
        num_layers: int = 1,
        dropout: float = 0.1,
        num_boundary_classes: int = 1,
    ):
        super().__init__()
        num_corr = num_corr_classes if num_corr_classes is not None else vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.input_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_dropout = nn.Dropout(dropout)

        # 3 Task-Specific Heads
        self.correction_head = nn.Linear(hidden_dim * 2, num_corr)
        self.diacritic_head = nn.Linear(hidden_dim * 2, num_target_classes)
        self.boundary_head = nn.Linear(hidden_dim * 2, num_boundary_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass across the shared backbone and 3 heads.

        Args:
            input_ids: [B, T] LongTensor of character IDs.
            lengths: [B] LongTensor of sequence lengths (for packing).

        Returns:
            dict containing:
                "correction_logits": [B, T, num_corr_classes]
                "diacritic_logits":  [B, T, num_target_classes]
                "boundary_logits":   [B, T]
        """
        x = self.embedding(input_ids)          # [B, T, E]
        x = self.input_dropout(x)

        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False,
            )
            gru_out, _ = self.gru(packed)
            gru_out, _ = nn.utils.rnn.pad_packed_sequence(
                gru_out, batch_first=True, total_length=input_ids.shape[1],
            )
        else:
            gru_out, _ = self.gru(x)           # [B, T, 2H]

        gru_out = self.output_dropout(gru_out)

        correction_logits = self.correction_head(gru_out)   # [B, T, C_corr]
        diacritic_logits = self.diacritic_head(gru_out)     # [B, T, C_diac]
        boundary_logits = self.boundary_head(gru_out).squeeze(-1)  # [B, T]

        return {
            "correction_logits": correction_logits,
            "diacritic_logits": diacritic_logits,
            "boundary_logits": boundary_logits,
        }
