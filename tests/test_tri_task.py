"""Unit tests for Tri-Task Restoration modules (Corruption, Tri-Head BiGRU, Loss, Metrics)."""

import torch
import pytest

from nextkey.data.corruption import SyntheticCorruptor, CorruptedSample
from nextkey.data.tokenizer import CharVocab
from nextkey.engine.loss import TriHeadLoss
from nextkey.models.tri_bigru import TriHeadBiGRUCharTagger
from nextkey.utils.metrics import TriMetricTotals


def test_synthetic_corruptor():
    corruptor = SyntheticCorruptor(typo_prob=0.3, swap_prob=0.1, seed=42)
    variants = corruptor.generate_variants("Tôi đang học bài tại Hà Nội.", num_variants=3)

    assert len(variants) == 3
    # Variant 0 must be clean compact
    assert variants[0].source == "toidanghocbaitaihanoi"
    assert variants[0].base_target == "toidanghocbaitaihanoi"
    assert len(variants[0].boundary_target) == len(variants[0].source)
    assert variants[0].boundary_target[3] == 1  # space before 'đ' in "đang"

    # Variant 1 & 2 must preserve length alignment
    for v in variants:
        assert len(v.source) == len(v.base_target) == len(v.diacritic_target) == len(v.boundary_target)


def test_tri_head_bigru_forward():
    vocab_size = 40
    num_target_classes = 80
    num_corr_classes = 40
    batch_size = 4
    seq_len = 16

    model = TriHeadBiGRUCharTagger(
        vocab_size=vocab_size,
        num_target_classes=num_target_classes,
        num_corr_classes=num_corr_classes,
        embed_dim=32,
        hidden_dim=48,
        num_layers=1,
    )

    input_ids = torch.randint(1, vocab_size, (batch_size, seq_len))
    lengths = torch.tensor([16, 14, 12, 10])

    outputs = model(input_ids, lengths=lengths)

    assert "correction_logits" in outputs
    assert "diacritic_logits" in outputs
    assert "boundary_logits" in outputs

    assert outputs["correction_logits"].shape == (batch_size, seq_len, num_corr_classes)
    assert outputs["diacritic_logits"].shape == (batch_size, seq_len, num_target_classes)
    assert outputs["boundary_logits"].shape == (batch_size, seq_len)


def test_cascade_tri_bigru_forward():
    from nextkey.models.cascade_tri_bigru import CascadeTriBiGRUCharTagger

    vocab_size = 40
    num_target_classes = 80
    num_corr_classes = 40
    batch_size = 4
    seq_len = 16

    model = CascadeTriBiGRUCharTagger(
        vocab_size=vocab_size,
        num_target_classes=num_target_classes,
        num_corr_classes=num_corr_classes,
        embed_dim=32,
        hidden_dim=48,
        num_layers=1,
    )

    input_ids = torch.randint(1, vocab_size, (batch_size, seq_len))
    lengths = torch.tensor([16, 14, 12, 10])

    outputs = model(input_ids, lengths=lengths)

    assert "correction_logits" in outputs
    assert "diacritic_logits" in outputs
    assert "boundary_logits" in outputs

    assert outputs["correction_logits"].shape == (batch_size, seq_len, num_corr_classes)
    assert outputs["diacritic_logits"].shape == (batch_size, seq_len, num_target_classes)
    assert outputs["boundary_logits"].shape == (batch_size, seq_len)


def test_tri_head_loss():
    loss_fn = TriHeadLoss(pad_target_id=0, pad_corr_id=0)

    B, T, C_corr, C_diac = 2, 8, 30, 60
    corr_logits = torch.randn(B, T, C_corr, requires_grad=True)
    diac_logits = torch.randn(B, T, C_diac, requires_grad=True)
    bnd_logits = torch.randn(B, T, requires_grad=True)

    corr_tgt = torch.randint(1, C_corr, (B, T))
    diac_tgt = torch.randint(1, C_diac, (B, T))
    bnd_tgt = torch.randint(0, 2, (B, T))

    losses = loss_fn(
        correction_logits=corr_logits,
        diacritic_logits=diac_logits,
        boundary_logits=bnd_logits,
        corr_targets=corr_tgt,
        diac_targets=diac_tgt,
        boundaries=bnd_tgt,
    )

    assert "loss" in losses
    assert "loss_corr" in losses
    assert "loss_diac" in losses
    assert "loss_boundary" in losses
    assert losses["loss"].item() > 0

    losses["loss"].backward()
    assert corr_logits.grad is not None
    assert diac_logits.grad is not None
    assert bnd_logits.grad is not None


def test_tri_metric_totals():
    metrics = TriMetricTotals()
    metrics.update_tri(
        source="tojdanghoc",
        pred_base="toidanghoc",
        gold_base="toidanghoc",
        pred_diac="tôiđanghọc",
        gold_diac="tôiđanghọc",
        pred_boundaries=[0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
        gold_boundaries=[0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
        final_prediction="tôi đang học",
        gold_sentence="tôi đang học",
    )

    d = metrics.as_dict()
    assert d["exact_match"] == 1.0
    assert d["corpus_cer"] == 0.0
    assert d["correction_accuracy"] == 1.0
    assert d["typo_recovery_rate"] == 1.0
    assert d["boundary_f1"] == 1.0
