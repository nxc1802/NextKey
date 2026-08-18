"""Dilated 1D TCN dual-head character tagger for Vietnamese compact-text restoration.

Architecture:
    Embedding -> Dilated Residual Conv1D Blocks -> [Diacritic Head, Boundary Head]
"""

from __future__ import annotations

import torch
import torch.nn as nn

from nextkey.models.base import BaseCharTagger, register_model


class ResidualConvBlock(nn.Module):
    """Dilated Conv1D block with residual connection and LayerNorm."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation,
        )
        self.relu = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation,
        )
        self.dropout2 = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        residual = self.downsample(x)
        out = self.dropout1(self.relu(self.conv1(x)))
        out = self.dropout2(self.conv2(out))
        out = out + residual
        # Permute for LayerNorm over channel dim: [B, T, C] -> LayerNorm -> [B, C, T]
        out = self.norm(out.transpose(1, 2)).transpose(1, 2)
        return self.relu(out)


@register_model("cnn_tcn")
class DilatedTCNCharTagger(BaseCharTagger):
    """Dilated Temporal Convolutional Network with dual classification heads."""

    def __init__(
        self,
        vocab_size: int,
        num_target_classes: int,
        embed_dim: int = 64,
        num_channels: list[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.1,
        num_boundary_classes: int = 1,
        **kwargs,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.input_dropout = nn.Dropout(dropout)

        channels = num_channels or [128, 128, 128, 128]
        layers: list[nn.Module] = []
        in_c = embed_dim
        for i, out_c in enumerate(channels):
            dilation = 2 ** i
            layers.append(ResidualConvBlock(in_c, out_c, kernel_size=kernel_size, dilation=dilation, dropout=dropout))
            in_c = out_c

        self.tcn = nn.Sequential(*layers)
        final_dim = channels[-1]

        # Dual heads
        self.diacritic_head = nn.Linear(final_dim, num_target_classes)
        self.boundary_head = nn.Linear(final_dim, num_boundary_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        x = self.embedding(input_ids)          # [B, T, E]
        x = self.input_dropout(x)
        x = x.transpose(1, 2)                  # [B, E, T]

        features = self.tcn(x)                 # [B, C, T]
        features = features.transpose(1, 2)    # [B, T, C]

        diacritic_logits = self.diacritic_head(features)
        boundary_logits = self.boundary_head(features).squeeze(-1)

        return {
            "diacritic_logits": diacritic_logits,
            "boundary_logits": boundary_logits,
        }
