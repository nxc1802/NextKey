"""Hybrid CNN + BiGRU dual-head character tagger for Vietnamese compact-text restoration.

Architecture:
    Embedding -> 1D Convolutions (local n-gram) -> BiGRU (long-range context) -> Dual Heads
"""

from __future__ import annotations

import torch
import torch.nn as nn

from nextkey.models.base import BaseCharTagger, register_model


@register_model("cnn_bigru")
class CNNBiGRUCharTagger(BaseCharTagger):
    """Hybrid CNN + BiGRU backbone with dual classification heads."""

    def __init__(
        self,
        vocab_size: int,
        num_target_classes: int,
        embed_dim: int = 64,
        conv_channels: int = 64,
        conv_kernel_size: int = 3,
        num_conv_layers: int = 2,
        hidden_dim: int = 96,
        num_layers: int = 1,
        dropout: float = 0.1,
        num_boundary_classes: int = 1,
        **kwargs,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.input_dropout = nn.Dropout(dropout)

        convs: list[nn.Module] = []
        in_c = embed_dim
        for _ in range(num_conv_layers):
            convs.extend([
                nn.Conv1d(in_c, conv_channels, kernel_size=conv_kernel_size, padding=conv_kernel_size // 2),
                nn.BatchNorm1d(conv_channels),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_c = conv_channels

        self.conv_stack = nn.Sequential(*convs)

        self.gru = nn.GRU(
            input_size=conv_channels,
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
        x = self.embedding(input_ids)          # [B, T, E]
        x = self.input_dropout(x)
        x = x.transpose(1, 2)                  # [B, E, T]

        conv_out = self.conv_stack(x)          # [B, ConvC, T]
        conv_out = conv_out.transpose(1, 2)    # [B, T, ConvC]

        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                conv_out, lengths.cpu(), batch_first=True, enforce_sorted=False,
            )
            gru_out, _ = self.gru(packed)
            gru_out, _ = nn.utils.rnn.pad_packed_sequence(
                gru_out, batch_first=True, total_length=input_ids.shape[1],
            )
        else:
            gru_out, _ = self.gru(conv_out)

        gru_out = self.output_dropout(gru_out)

        diacritic_logits = self.diacritic_head(gru_out)
        boundary_logits = self.boundary_head(gru_out).squeeze(-1)

        return {
            "diacritic_logits": diacritic_logits,
            "boundary_logits": boundary_logits,
        }
