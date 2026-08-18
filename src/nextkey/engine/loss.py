"""Loss functions for the CharTagger dual-head architecture.

Provides:
    - DualHeadLoss: standard CE (diacritic) + BCE (boundary) training loss
    - DistillationLoss: Knowledge Distillation with soft + hard label mixing
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DualHeadLoss(nn.Module):
    """Combined loss for diacritic classification + boundary detection.

    Loss = CE(diacritic_logits, targets) + λ * BCE(boundary_logits, boundaries)

    Boundary targets padded with -100 are ignored.
    """

    def __init__(
        self,
        pad_target_id: int = 0,
        lambda_boundary: float = 1.0,
    ):
        super().__init__()
        self.char_loss = nn.CrossEntropyLoss(ignore_index=pad_target_id)
        self.boundary_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.lambda_boundary = lambda_boundary

    def forward(
        self,
        diacritic_logits: torch.Tensor,
        boundary_logits: torch.Tensor,
        targets: torch.Tensor,
        boundaries: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute combined loss.

        Args:
            diacritic_logits: [B, T, C] — character classification logits
            boundary_logits: [B, T] — boundary detection logits (pre-sigmoid)
            targets: [B, T] — ground-truth target character IDs
            boundaries: [B, T] — ground-truth boundary flags (-100 = padding)

        Returns:
            dict with "loss", "loss_char", "loss_boundary" tensors
        """
        # Diacritic head loss
        loss_char = self.char_loss(
            diacritic_logits.reshape(-1, diacritic_logits.size(-1)),
            targets.reshape(-1),
        )

        # Boundary head loss (only on valid positions)
        valid_mask = boundaries != -100
        if valid_mask.any():
            loss_boundary = self.boundary_loss(
                boundary_logits[valid_mask],
                boundaries[valid_mask].float(),
            ).mean()
        else:
            loss_boundary = torch.tensor(0.0, device=diacritic_logits.device)

        total = loss_char + self.lambda_boundary * loss_boundary

        return {
            "loss": total,
            "loss_char": loss_char.detach(),
            "loss_boundary": loss_boundary.detach(),
        }


class DistillationLoss(nn.Module):
    """Knowledge Distillation loss for CharTagger.

    L_total = (1 - α) * L_CE(hard) + α * T² * L_KL(soft)

    Applied independently to the diacritic head. Boundary head uses standard BCE.
    When α=0 or teacher logits are None, degrades to standard DualHeadLoss.
    """

    def __init__(
        self,
        pad_target_id: int = 0,
        lambda_boundary: float = 1.0,
        alpha: float = 0.5,
        temperature: float = 2.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.char_loss = nn.CrossEntropyLoss(ignore_index=pad_target_id)
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.boundary_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.lambda_boundary = lambda_boundary

    def forward(
        self,
        diacritic_logits: torch.Tensor,
        boundary_logits: torch.Tensor,
        targets: torch.Tensor,
        boundaries: torch.Tensor,
        teacher_diacritic_logits: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        # Hard label CE loss
        loss_ce = self.char_loss(
            diacritic_logits.reshape(-1, diacritic_logits.size(-1)),
            targets.reshape(-1),
        )

        # KD soft label loss
        if teacher_diacritic_logits is not None and self.alpha > 0:
            T = self.temperature
            p_student = F.log_softmax(diacritic_logits / T, dim=-1)
            p_teacher = F.softmax(teacher_diacritic_logits / T, dim=-1)
            # Mask out padding
            valid = targets.reshape(-1) != 0
            loss_kd = self.kl_loss(
                p_student.reshape(-1, p_student.size(-1))[valid],
                p_teacher.reshape(-1, p_teacher.size(-1))[valid],
            ) * (T ** 2)
            loss_diacritic = (1 - self.alpha) * loss_ce + self.alpha * loss_kd
        else:
            loss_kd = torch.tensor(0.0, device=diacritic_logits.device)
            loss_diacritic = loss_ce

        # Boundary loss
        valid_mask = boundaries != -100
        if valid_mask.any():
            loss_boundary = self.boundary_loss(
                boundary_logits[valid_mask],
                boundaries[valid_mask].float(),
            ).mean()
        else:
            loss_boundary = torch.tensor(0.0, device=diacritic_logits.device)

        total = loss_diacritic + self.lambda_boundary * loss_boundary

        return {
            "loss": total,
            "loss_char": loss_ce.detach(),
            "loss_kd": loss_kd.detach(),
            "loss_boundary": loss_boundary.detach(),
        }
