"""Command entry points for the NextKey project scaffold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised when PyYAML is absent.
    yaml = None


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None", "~"}:
        return None
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def parse_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the small YAML subset used by scaffold configs.

    This keeps Task 0 runnable before optional dependencies are installed. It is
    intentionally limited to nested mappings and scalar lists.
    """
    lines: list[tuple[int, str]] = []
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        lines.append((indent, raw_line.strip()))

    def parse_block(index: int, indent: int) -> tuple[Any, int]:
        if index < len(lines) and lines[index][1].startswith("- "):
            values: list[Any] = []
            while index < len(lines):
                current_indent, current = lines[index]
                if current_indent != indent or not current.startswith("- "):
                    break
                values.append(parse_scalar(current[2:]))
                index += 1
            return values, index

        values: dict[str, Any] = {}
        while index < len(lines):
            current_indent, current = lines[index]
            if current_indent < indent:
                break
            if current_indent > indent:
                raise ValueError(f"Unexpected indentation near: {current}")
            if ":" not in current:
                raise ValueError(f"Expected key/value pair near: {current}")

            key, raw_value = current.split(":", 1)
            key = key.strip()
            raw_value = raw_value.strip()
            index += 1

            if raw_value:
                values[key] = parse_scalar(raw_value)
                continue

            if index >= len(lines) or lines[index][0] <= current_indent:
                values[key] = {}
                continue

            child, index = parse_block(index, lines[index][0])
            values[key] = child
        return values, index

    parsed, final_index = parse_block(0, 0)
    if final_index != len(lines) or not isinstance(parsed, dict):
        raise ValueError("Config must be a YAML mapping.")
    return parsed


def load_config(path: str | Path | None) -> dict[str, Any]:
    """Load a YAML config file, returning an empty config when no path is given."""
    if path is None:
        return {}

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        if yaml is not None:
            loaded = yaml.safe_load(handle) or {}
        else:
            loaded = parse_simple_yaml(handle.read())

    if not isinstance(loaded, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return loaded


def emit_status(command: str, config: dict[str, Any], extra: dict[str, Any] | None = None) -> None:
    payload: dict[str, Any] = {
        "command": command,
        "status": "scaffold-ready",
        "config_name": config.get("name", "unnamed"),
        "message": "Entrypoint is wired; implementation will be added in later tasks.",
    }
    if extra:
        payload.update(extra)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def config_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, help="Path to a YAML config file.")
    return parser


def build_dataset_main() -> None:
    parser = config_parser("Build or validate NextKey datasets.")
    args = parser.parse_args()
    config = load_config(args.config)
    output = config.get("output", {})
    emit_status(
        "build_dataset",
        config,
        {
            "planned_outputs": output,
            "next_task": "Implement clean-source loading and synthetic generation.",
        },
    )


def train_main() -> None:
    parser = config_parser("Train a NextKey model candidate.")
    args = parser.parse_args()
    config = load_config(args.config)
    model = config.get("model", {})
    emit_status(
        "train",
        config,
        {
            "model_role": model.get("role", "unset"),
            "model_candidate": model.get("candidate", "unset"),
            "next_task": "Implement baseline and model-selection training loops.",
        },
    )


def evaluate_main() -> None:
    parser = config_parser("Evaluate a NextKey model candidate.")
    args = parser.parse_args()
    config = load_config(args.config)
    emit_status(
        "evaluate",
        config,
        {
            "metrics": config.get("metrics", []),
            "next_task": "Implement metric functions and report generation.",
        },
    )


def export_model_main() -> None:
    parser = argparse.ArgumentParser(description="Export a NextKey model for deployment.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint path to export.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/exported"),
        help="Directory for exported model artifacts.",
    )
    args = parser.parse_args()
    emit_status(
        "export_model",
        {"name": "export-model"},
        {
            "checkpoint": str(args.checkpoint),
            "output_dir": str(args.output_dir),
            "next_task": "Implement ONNX/LiteRT export once a student model exists.",
        },
    )


def demo_main() -> None:
    parser = argparse.ArgumentParser(description="Run the NextKey demo scaffold.")
    parser.add_argument(
        "--mode",
        default="placeholder",
        choices=["placeholder", "api", "ui"],
        help="Demo runtime mode.",
    )
    args = parser.parse_args()
    emit_status(
        "demo",
        {"name": "demo-scaffold"},
        {
            "mode": args.mode,
            "next_task": "Implement /restore and feedback endpoints in Task 6.",
        },
    )
