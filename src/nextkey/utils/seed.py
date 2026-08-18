"""Reproducibility utilities."""

from __future__ import annotations

import random


def seed_everything(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch random generators for reproducibility."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Deterministic algorithms may slow training but improve reproducibility
        torch.use_deterministic_algorithms(False)
    except ImportError:
        pass
