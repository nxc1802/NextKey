from __future__ import annotations

import pytest
import torch

from nextkey.engine.loss import DualHeadLoss, DistillationLoss


def test_dual_head_loss():
    loss_fn = DualHeadLoss(pad_target_id=0, lambda_boundary=1.0)
    batch_size, seq_len, num_classes = 2, 8, 20

    diacritic_logits = torch.randn(batch_size, seq_len, num_classes)
    boundary_logits = torch.randn(batch_size, seq_len)
    targets = torch.randint(0, num_classes, (batch_size, seq_len))
    boundaries = torch.randint(0, 2, (batch_size, seq_len))

    loss_dict = loss_fn(diacritic_logits, boundary_logits, targets, boundaries)
    assert "loss" in loss_dict
    assert "loss_char" in loss_dict
    assert "loss_boundary" in loss_dict
    assert loss_dict["loss"].item() > 0


def test_distillation_loss():
    loss_fn = DistillationLoss(pad_target_id=0, lambda_boundary=1.0, alpha=0.5, temperature=2.0)
    batch_size, seq_len, num_classes = 2, 8, 20

    student_d_logits = torch.randn(batch_size, seq_len, num_classes)
    student_b_logits = torch.randn(batch_size, seq_len)
    teacher_d_logits = torch.randn(batch_size, seq_len, num_classes)
    targets = torch.randint(0, num_classes, (batch_size, seq_len))
    boundaries = torch.randint(0, 2, (batch_size, seq_len))

    loss_dict = loss_fn(student_d_logits, student_b_logits, targets, boundaries, teacher_d_logits)
    assert "loss" in loss_dict
    assert "loss_kd" in loss_dict
    assert loss_dict["loss"].item() > 0
