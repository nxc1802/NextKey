"""Abstract base class and registry for CharTagger backbone models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

_MODEL_REGISTRY: dict[str, type["BaseCharTagger"]] = {}


def register_model(name: str):
    """Decorator to register a model class by name."""
    def decorator(cls):
        _MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def create_model(name: str, **kwargs) -> "BaseCharTagger":
    """Instantiate a registered model by name."""
    if name not in _MODEL_REGISTRY:
        available = ", ".join(sorted(_MODEL_REGISTRY))
        raise ValueError(f"Unknown model '{name}'. Available: {available}")
    return _MODEL_REGISTRY[name](**kwargs)


def list_models() -> list[str]:
    return sorted(_MODEL_REGISTRY)


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class BaseCharTagger(nn.Module, ABC):
    """Abstract dual-head character tagger.

    All backbone implementations must return a dict with:
        - "diacritic_logits": [B, T, num_target_classes]
        - "boundary_logits":  [B, T] (pre-sigmoid)
    """

    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        ...

    def count_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self) -> dict[str, Any]:
        """Model metadata for logging."""
        return {
            "class": self.__class__.__name__,
            "parameters": self.count_parameters(),
            "parameters_human": f"{self.count_parameters() / 1000:.1f}K",
        }
