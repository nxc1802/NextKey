"""Tiny Transformer Encoder dual-head character tagger for Vietnamese compact-text restoration.

Architecture:
    Embedding + Sinusoidal Positional Encoding -> Multi-Head Self-Attention Transformer Layers -> Dual Heads
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn

from nextkey.models.base import BaseCharTagger, register_model


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, E]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


@register_model("tiny_transformer")
class TinyTransformerCharTagger(BaseCharTagger):
    """Tiny Transformer Encoder backbone with dual classification heads."""

    def __init__(
        self,
        vocab_size: int,
        num_target_classes: int,
        embed_dim: int = 64,
        num_heads: int = 4,
        dim_feedforward: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        num_boundary_classes: int = 1,
        **kwargs,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=max_seq_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(embed_dim)

        # Dual heads
        self.diacritic_head = nn.Linear(embed_dim, num_target_classes)
        self.boundary_head = nn.Linear(embed_dim, num_boundary_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        # Padding mask: True where value is 0 (PAD)
        src_key_padding_mask = (input_ids == 0)

        x = self.embedding(input_ids)          # [B, T, E]
        x = self.pos_encoder(x)

        encoded = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        encoded = self.layer_norm(encoded)

        diacritic_logits = self.diacritic_head(encoded)
        boundary_logits = self.boundary_head(encoded).squeeze(-1)

        return {
            "diacritic_logits": diacritic_logits,
            "boundary_logits": boundary_logits,
        }
