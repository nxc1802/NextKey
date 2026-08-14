from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from nextkey.data.mvp_dataset import export_jdwr_v1_splits
from nextkey.evaluation.mvp_metrics import MetricTotals
from nextkey.models.mvp_chartagger import align_pair


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


def test_jdwr_grouped_split_prevents_duplicate_and_external_leakage(tmp_path: Path) -> None:
    (tmp_path / "alpha_dataset.csv").write_text(
        "Input_X,Target_Y\ntoidanghoc,Tôi đang học.\ntoidanghoc,Tôi đang học!\nchao,Chào.\n",
        encoding="utf-8",
    )
    (tmp_path / "sports_dataset.csv").write_text(
        "Input_X,Target_Y\ntoidanghoc,Tôi đang học.\ndabong,Đá bóng.\n",
        encoding="utf-8",
    )
    manifest = export_jdwr_v1_splits(
        tmp_path, "*_dataset.csv", "Input_X", "Target_Y", tmp_path / "jdwr", ["alpha"], "sports",
        0.8, 0.1, 0.1,
    )
    assert all(value == 0 for value in manifest["group_overlap_checks"].values())
    assert manifest["skipped"]["external_group_overlap"] == 2
    assert manifest["counts"]["external"]["sports"] == 2


def test_jdwr_alignment_and_boundary_metrics() -> None:
    example = align_pair("toidanghoc", "Tôi đang học")
    assert example is not None
    assert example.char_target == "tôiđanghọc"
    assert example.boundary_target == [0, 0, 0, 1, 0, 0, 0, 1, 0, 0]
    totals = MetricTotals()
    totals.update("tôi đang học", "tôi đang học")
    assert totals.as_dict()["boundary_f1"] == 1.0
