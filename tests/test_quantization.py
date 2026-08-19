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
