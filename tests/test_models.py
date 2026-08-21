from __future__ import annotations

import pytest
import torch

from nextkey.models import create_model, list_models, BaseCharTagger
from nextkey.models.bigru import BiGRUCharTagger
from nextkey.models.bilstm import BiLSTMCharTagger
from nextkey.models.cnn_tcn import DilatedTCNCharTagger
from nextkey.models.cnn_bigru import CNNBiGRUCharTagger
from nextkey.models.tiny_transformer import TinyTransformerCharTagger


def test_model_registry():
    models = list_models()
    for name in ["bigru", "bilstm", "cnn_tcn", "cnn_bigru", "tiny_transformer"]:
        assert name in models


@pytest.mark.parametrize("model_name", ["bigru", "bilstm", "cnn_tcn", "cnn_bigru", "tiny_transformer"])
def test_model_forward_passes(model_name: str):
    vocab_size = 30
    num_targets = 45
    embed_dim = 32
    hidden_dim = 64
    batch_size = 4
    seq_len = 16

    model = create_model(
        model_name,
        vocab_size=vocab_size,
        num_target_classes=num_targets,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=1,
    )
    assert isinstance(model, BaseCharTagger)
    assert model.count_parameters() > 0

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    lengths = torch.tensor([16, 12, 10, 8])

    outputs = model(input_ids, lengths)
    assert "diacritic_logits" in outputs
    assert "boundary_logits" in outputs

    assert outputs["diacritic_logits"].shape == (batch_size, seq_len, num_targets)
    assert outputs["boundary_logits"].shape == (batch_size, seq_len)


@pytest.mark.parametrize(
    "config_name,expected_max_params",
    [
        ("width_xxxs", 25_000),
        ("width_xxs", 40_000),
        ("width_xs", 60_000),
    ],
)
def test_phase2_size_configs(config_name: str, expected_max_params: int):
    from nextkey.utils.config_parser import load_merged_config

    cfg = load_merged_config("configs/base.yaml", f"configs/phase2_size/{config_name}.yaml")
    model_cfg = dict(cfg["model"])
    model_name = model_cfg.pop("name")

    model = create_model(
        model_name,
        vocab_size=105,
        num_target_classes=105,
        **model_cfg,
    )
    assert model.count_parameters() < expected_max_params
    assert model.count_parameters() > 0
