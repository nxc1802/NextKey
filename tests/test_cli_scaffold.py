from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_script(*args: str) -> dict[str, object]:
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


def test_build_dataset_entrypoint_reads_config() -> None:
    payload = run_script("scripts/build_dataset.py", "--config", "configs/data/synthetic_v1.yaml")

    assert payload["command"] == "build_dataset"
    assert payload["status"] == "scaffold-ready"
    assert payload["config_name"] == "synthetic-v1"


def test_train_entrypoint_preserves_open_model_gate() -> None:
    payload = run_script("scripts/train.py", "--config", "configs/model/teacher_candidate.yaml")

    assert payload["command"] == "train"
    assert payload["model_role"] == "teacher"
    assert payload["model_candidate"] == "unset"


def test_evaluate_entrypoint_reports_metrics() -> None:
    payload = run_script("scripts/evaluate.py", "--config", "configs/eval/dev_human.yaml")

    assert payload["command"] == "evaluate"
    assert "cer" in payload["metrics"]


def test_demo_entrypoint_runs_placeholder_mode() -> None:
    payload = run_script("scripts/run_demo.py")

    assert payload["command"] == "demo"
    assert payload["mode"] == "placeholder"
