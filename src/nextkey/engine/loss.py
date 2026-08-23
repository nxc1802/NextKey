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


class TriHeadLoss(nn.Module):
    """Combined loss for Tri-Head architecture:

    L = λ_diac * CE(diacritic_logits, diac_targets)
      + λ_corr * CE(correction_logits, corr_targets)
      + λ_bnd  * BCE(boundary_logits, boundaries)
    """

    def __init__(
        self,
        pad_target_id: int = 0,
        pad_corr_id: int = 0,
        lambda_diacritic: float = 1.0,
        lambda_correction: float = 1.0,
        lambda_boundary: float = 1.0,
    ):
        super().__init__()
        self.diac_loss = nn.CrossEntropyLoss(ignore_index=pad_target_id)
        self.corr_loss = nn.CrossEntropyLoss(ignore_index=pad_corr_id)
        self.boundary_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.lambda_diacritic = lambda_diacritic
        self.lambda_correction = lambda_correction
        self.lambda_boundary = lambda_boundary

    def forward(
        self,
        correction_logits: torch.Tensor,
        diacritic_logits: torch.Tensor,
        boundary_logits: torch.Tensor,
        corr_targets: torch.Tensor,
        diac_targets: torch.Tensor,
        boundaries: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        loss_corr = self.corr_loss(
            correction_logits.reshape(-1, correction_logits.size(-1)),
            corr_targets.reshape(-1),
        )
        loss_diac = self.diac_loss(
            diacritic_logits.reshape(-1, diacritic_logits.size(-1)),
            diac_targets.reshape(-1),
        )

        valid_mask = boundaries != -100
        if valid_mask.any():
            loss_boundary = self.boundary_loss(
                boundary_logits[valid_mask],
                boundaries[valid_mask].float(),
            ).mean()
        else:
            loss_boundary = torch.tensor(0.0, device=diacritic_logits.device)

        total = (
            self.lambda_diacritic * loss_diac
            + self.lambda_correction * loss_corr
            + self.lambda_boundary * loss_boundary
        )

        return {
            "loss": total,
            "loss_corr": loss_corr.detach(),
            "loss_diac": loss_diac.detach(),
            "loss_boundary": loss_boundary.detach(),
        }


class DistillationLoss(nn.Module):
    """Knowledge Distillation loss for CharTagger with Dynamic Temperature & Outlier Regularization.

    L_total = (1 - α) * L_CE(hard) + α * T² * L_KL(soft) + λ_bnd * L_BCE + λ_outlier * L_outlier

    Applied independently to the diacritic head. Boundary head uses standard BCE.
    When α=0 or teacher logits are None, degrades to standard DualHeadLoss.
    """

    def __init__(
        self,
        pad_target_id: int = 0,
        lambda_boundary: float = 1.0,
        alpha: float = 0.5,
        temperature: float = 2.0,
        outlier_penalty: float = 0.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.outlier_penalty = outlier_penalty
        self.char_loss = nn.CrossEntropyLoss(ignore_index=pad_target_id)
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.boundary_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.lambda_boundary = lambda_boundary

    def set_temperature(self, temp: float) -> None:
        """Update distillation temperature dynamically (e.g. for cosine annealing)."""
        self.temperature = max(float(temp), 0.1)

    def get_temperature(self) -> float:
        return self.temperature

    def compute_outlier_penalty(self, model: nn.Module, threshold_sigma: float = 3.0) -> torch.Tensor:
        """Penalize outlier weights beyond threshold_sigma std devs to preserve INT8 dynamic range."""
        if self.outlier_penalty <= 0:
            return torch.tensor(0.0, device=next(model.parameters()).device)

        penalty = torch.tensor(0.0, device=next(model.parameters()).device)
        for name, p in model.named_parameters():
            if "weight" in name and p.requires_grad and p.numel() > 10:
                std = p.std().detach().clamp_min(1e-5)
                mean = p.mean().detach()
                cutoff = threshold_sigma * std
                outliers = torch.relu((p - mean).abs() - cutoff)
                penalty = penalty + outliers.pow(2).sum()
        return self.outlier_penalty * penalty

    def forward(
        self,
        diacritic_logits: torch.Tensor,
        boundary_logits: torch.Tensor,
        targets: torch.Tensor,
        boundaries: torch.Tensor,
        teacher_diacritic_logits: Optional[torch.Tensor] = None,
        model_for_regularization: Optional[nn.Module] = None,
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

        # Outlier weight penalty
        if model_for_regularization is not None and self.outlier_penalty > 0:
            reg_loss = self.compute_outlier_penalty(model_for_regularization)
            total = total + reg_loss
        else:
            reg_loss = torch.tensor(0.0, device=diacritic_logits.device)

        return {
            "loss": total,
            "loss_char": loss_ce.detach(),
            "loss_kd": loss_kd.detach(),
            "loss_boundary": loss_boundary.detach(),
            "loss_reg": reg_loss.detach(),
        }
