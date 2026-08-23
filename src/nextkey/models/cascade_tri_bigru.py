"""Hierarchical Cascade Tri-Head Multi-Task BiGRU for Vietnamese Compact Restoration.

Architecture:
    Input Chars → Embedding (64) → Local Conv1D (K=3) → Shared BiGRU Backbone (H=128)
                                                              │
                       ┌──────────────────────────────────────┼──────────────────────────────────────┐
                       ▼                                      ▼                                      │
               [Correction Head]                      [Boundary Head]                                │
           (Base Character Recovery)               (Word Boundary Flag)                              │
                       │                                      │                                      │
                       ▼                                      ▼                                      │
               Corr Feature (32d)                    Bnd Feature (8d)                                │
                       └──────────────────────┬──────────────────────────────────────────────────────┘
                                              ▼
                                     [Cross-Head Fusion]
                                    H_fused = [H, Corr, Bnd] (296d)
                                              │
                                              ▼
                                       [Diacritic Head]
                                  (Accented Char Prediction)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nextkey.models.base import BaseCharTagger, register_model


@register_model("cascade_tri_bigru")
class CascadeTriBiGRUCharTagger(BaseCharTagger):
    """Hierarchical Cascaded Tri-Head BiGRU with Local Context Conv and Cross-Head Feature Flow."""

    def __init__(
        self,
        vocab_size: int,
        num_target_classes: int,
        num_corr_classes: int | None = None,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1,
        corr_proj_dim: int = 32,
        bnd_proj_dim: int = 8,
    ):
        super().__init__()
        num_corr = num_corr_classes if num_corr_classes is not None else vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.input_dropout = nn.Dropout(dropout)

        # 1. Local N-Gram Feature Extractor (Depthwise Separable Conv1D K=3)
        self.local_conv = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=1),
        )

        # 2. Shared BiGRU Backbone
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_dropout = nn.Dropout(dropout)

        backbone_dim = hidden_dim * 2  # 256

        # 3. Stage 1 Heads: Correction Head & Boundary Head
        self.correction_head = nn.Linear(backbone_dim, num_corr)
        self.boundary_head = nn.Linear(backbone_dim, 1)

        # 4. Cross-Head Feature Projections
        self.corr_feature_proj = nn.Sequential(
            nn.Linear(num_corr, corr_proj_dim),
            nn.GELU(),
        )
        self.bnd_feature_proj = nn.Sequential(
            nn.Linear(1, bnd_proj_dim),
            nn.GELU(),
        )

        # 5. Stage 2 Head: Diacritic Head with Fused Representation
        fused_dim = backbone_dim + corr_proj_dim + bnd_proj_dim
        self.diacritic_head = nn.Sequential(
            nn.Linear(fused_dim, backbone_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(backbone_dim, num_target_classes),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass with hierarchical task conditioning.

        Args:
            input_ids: [B, T] LongTensor of character IDs.
            lengths: [B] LongTensor of sequence lengths.

        Returns:
            dict with correction_logits [B, T, C_corr], diacritic_logits [B, T, C_diac], boundary_logits [B, T].
        """
        # Embed and apply local n-gram conv
        x = self.embedding(input_ids)          # [B, T, E]
        x_conv = self.local_conv(x.transpose(1, 2)).transpose(1, 2)  # [B, T, E]
        x = self.input_dropout(x + x_conv)     # Residual connection

        # Shared BiGRU Backbone
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

        h = self.output_dropout(gru_out)       # [B, T, 2H]

        # Stage 1: Predict Correction and Boundary
        correction_logits = self.correction_head(h)     # [B, T, C_corr]
        boundary_logits = self.boundary_head(h)         # [B, T, 1]

        # Stage 2: Cross-Head Conditioning
        corr_soft = F.softmax(correction_logits, dim=-1)
        bnd_sig = torch.sigmoid(boundary_logits)

        corr_feat = self.corr_feature_proj(corr_soft)   # [B, T, corr_proj_dim]
        bnd_feat = self.bnd_feature_proj(bnd_sig)       # [B, T, bnd_proj_dim]

        # Fuse representations for Diacritic Head
        h_fused = torch.cat([h, corr_feat, bnd_feat], dim=-1)  # [B, T, fused_dim]
        diacritic_logits = self.diacritic_head(h_fused)        # [B, T, C_diac]

        return {
            "correction_logits": correction_logits,
            "diacritic_logits": diacritic_logits,
            "boundary_logits": boundary_logits.squeeze(-1),
        }
