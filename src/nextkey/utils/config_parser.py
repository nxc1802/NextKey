"""Configuration loading and merging utilities for NextKey research pipeline."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Optional

import yaml


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge *override* into *base*, returning a new dict."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a single YAML config file."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")
    return loaded


def load_merged_config(
    base_config_path: str | Path,
    phase_config_path: str | Path,
    mode: str = "smoke",
    cli_device: Optional[str] = None,
) -> dict[str, Any]:
    """Load and merge base config + phase config with mode overrides.

    Priority (highest to lowest):
        1. CLI arguments (cli_device)
        2. Phase config values
        3. Mode-specific values from base config
        4. Base config defaults

    Returns a flat config dict with resolved training parameters.
    """
    base = load_config(base_config_path)
    phase = load_config(phase_config_path)

    # Merge phase overrides into base
    cfg = deep_merge(base, phase)

    # Apply mode-specific training overrides
    mode_cfg = cfg.get("modes", {}).get(mode, {})
    if mode_cfg:
        training = cfg.setdefault("training", {})
        for key, value in mode_cfg.items():
            # Mode values fill in gaps; phase config takes priority
            if key not in training or key not in phase.get("training", {}):
                training[key] = value

    # Resolve runtime device
    runtime = cfg.setdefault("runtime", {})
    if cli_device:
        runtime["device"] = cli_device
    elif "device" not in runtime:
        # Use mode default or fall back to auto-detect
        runtime["device"] = mode_cfg.get("device", "cpu")

    cfg["_mode"] = mode
    return cfg
