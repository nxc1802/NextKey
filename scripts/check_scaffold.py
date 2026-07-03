#!/usr/bin/env python3
"""Run dependency-light scaffold checks."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_json(*args: str) -> dict[str, object]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    return json.loads(result.stdout)


def assert_equal(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def main() -> None:
    checks = [
        (
            ("scripts/build_dataset.py", "--config", "configs/data/synthetic_v1.yaml"),
            "build_dataset",
        ),
        (
            ("scripts/train.py", "--config", "configs/model/teacher_candidate.yaml"),
            "train",
        ),
        (
            ("scripts/evaluate.py", "--config", "configs/eval/dev_human.yaml"),
            "evaluate",
        ),
        (
            ("scripts/export_model.py", "--checkpoint", "models/checkpoints/student"),
            "export_model",
        ),
        (("scripts/run_demo.py",), "demo"),
    ]

    for args, expected_command in checks:
        payload = run_json(*args)
        assert_equal(payload.get("command"), expected_command, f"{expected_command} command")
        assert_equal(payload.get("status"), "scaffold-ready", f"{expected_command} status")

    print("scaffold checks passed")


if __name__ == "__main__":
    main()
