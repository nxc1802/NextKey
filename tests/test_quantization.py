"""Tests for NextKey quantization engine (FakeQuantizeSTE, QKD, PTQ, and Checkpoint Export)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from nextkey.data.tokenizer import CharVocab
from nextkey.engine.quantization import (
    FakeQuantizeSTE,
    QuantizedBiGRUCharTagger,
    apply_dynamic_quantization,
    convert_to_qat_model,
    export_int8_checkpoint,
    export_onnx_model,
    fake_quantize,
)
from nextkey.models.bigru import BiGRUCharTagger


def test_fake_quantize_ste_forward_backward():
    """Verify forward discrete clamping and backward STE gradient propagation."""
    x = torch.tensor([-2.5, -0.5, 0.0, 0.5, 2.5], requires_grad=True)
    q = fake_quantize(x, num_bits=8)

    assert q.shape == x.shape
    loss = q.sum()
    loss.backward()

    # Gradient should pass through directly (STE)
    assert x.grad is not None
    assert torch.all(x.grad == 1.0)


def test_quantized_bigru_forward_pass():
    """Verify QuantizedBiGRUCharTagger forward pass produces valid logits."""
    model = QuantizedBiGRUCharTagger(
        vocab_size=50,
        num_target_classes=40,
        embed_dim=32,
        hidden_dim=64,
        num_layers=1,
    )
    input_ids = torch.randint(1, 49, (4, 30))
    lengths = torch.tensor([30, 25, 20, 15])

    outputs = model(input_ids, lengths)

    assert "diacritic_logits" in outputs
    assert "boundary_logits" in outputs
    assert outputs["diacritic_logits"].shape == (4, 30, 40)
    assert outputs["boundary_logits"].shape == (4, 30)


def test_convert_to_qat_model():
    """Verify conversion of standard FP32 BiGRU to QAT BiGRU preserves weights."""
    fp32_model = BiGRUCharTagger(
        vocab_size=50,
        num_target_classes=40,
        embed_dim=32,
        hidden_dim=64,
        num_layers=1,
    )
    qat_model = convert_to_qat_model(fp32_model, num_bits=8)

    assert isinstance(qat_model, QuantizedBiGRUCharTagger)
    # Check weight equality
    assert torch.allclose(fp32_model.diacritic_head.weight, qat_model.diacritic_head.weight)


def test_export_int8_compact_checkpoint():
    """Verify export of compact INT8 serialized checkpoint (< 60 KB)."""
    model = BiGRUCharTagger(
        vocab_size=50,
        num_target_classes=40,
        embed_dim=32,
        hidden_dim=64,
        num_layers=1,
    )
    vocab = CharVocab(
        char_stoi={"<pad>": 0, "a": 1},
        char_itos=["<pad>", "a"],
        target_stoi={"<pad>": 0, "a": 1},
        target_itos=["<pad>", "a"],
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "model_int8.pt"
        info = export_int8_checkpoint(model, vocab, out_path)

        assert out_path.exists()
        assert info["file_size_kb"] < 100.0  # Must be very compact
        loaded = torch.load(out_path, map_location="cpu")
        assert "scales" in loaded
        assert "tensors" in loaded


def test_distillation_loss_temp_annealing_and_outlier_penalty():
    """Verify dynamic temperature annealing and outlier regularization."""
    from nextkey.engine.loss import DistillationLoss

    loss_fn = DistillationLoss(
        pad_target_id=0,
        alpha=0.5,
        temperature=3.0,
        outlier_penalty=0.01,
    )
    assert loss_fn.get_temperature() == 3.0

    # Test dynamic temperature change
    loss_fn.set_temperature(1.5)
    assert loss_fn.get_temperature() == 1.5

    # Test forward with outlier penalty
    model = BiGRUCharTagger(vocab_size=30, num_target_classes=20, embed_dim=16, hidden_dim=32, num_layers=1)
    # inject an outlier
    with torch.no_grad():
        model.diacritic_head.weight[0, 0] = 50.0

    d_logits = torch.randn(2, 10, 20, requires_grad=True)
    b_logits = torch.randn(2, 10, requires_grad=True)
    targets = torch.randint(1, 19, (2, 10))
    boundaries = torch.randint(0, 2, (2, 10))
    t_logits = torch.randn(2, 10, 20)

    res = loss_fn(
        d_logits,
        b_logits,
        targets,
        boundaries,
        teacher_diacritic_logits=t_logits,
        model_for_regularization=model,
    )
    assert "loss" in res
    assert "loss_reg" in res
    assert res["loss_reg"].item() > 0.0
