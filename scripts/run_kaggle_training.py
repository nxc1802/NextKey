#!/usr/bin/env python3
"""Run the full JDWR v1 Kaggle workflow with zero required arguments."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


CONFIG_PATH = Path("configs/model/mvp_chartagger_full_kaggle.yaml")
ARTIFACT_PATHS = (
    Path("models/checkpoints/mvp-chartagger-full-kaggle.pt"),
    Path("models/checkpoints/mvp-chartagger-full-kaggle-vocab.json"),
    Path("models/checkpoints/training_history.json"),
    Path("experiments/reports/mvp-chartagger-full-kaggle.json"),
    Path("experiments/reports/mvp-chartagger-full-kaggle.md"),
    Path("experiments/runs/mvp-chartagger-full-kaggle"),
)


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def find_processed_dir(root: Path) -> Path:
    local_processed = root / "data" / "processed"
    if (local_processed / "jdwr_v1" / "manifest.json").is_file():
        return local_processed

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        manifests = sorted(kaggle_input.glob("**/processed/jdwr_v1/manifest.json"))
        if manifests:
            return manifests[0].parents[1]

    raise FileNotFoundError(
        "Could not find processed/jdwr_v1/manifest.json locally or under /kaggle/input. "
        "Attach a Kaggle Dataset containing the processed directory."
    )


def prepare_data(root: Path) -> None:
    source = find_processed_dir(root)
    target = root / "data" / "processed"
    if source.resolve() != target.resolve():
        shutil.copytree(source, target, dirs_exist_ok=True)
    print(f"Using processed data: {target}")


def require_kaggle_gpu() -> None:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required; enable the Kaggle GPU image.") from error
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; enable a Kaggle GPU accelerator.")
    print(f"PyTorch {torch.__version__}; GPU: {torch.cuda.get_device_name(0)}")


def run(root: Path, script: str) -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root / "src")
    subprocess.run([sys.executable, "-u", script, "--config", str(CONFIG_PATH)], cwd=root,
                   env=environment, check=True)


def write_zip(root: Path) -> Path:
    missing = [path for path in ARTIFACT_PATHS if not (root / path).exists()]
    if missing:
        raise FileNotFoundError(f"Expected training artifacts are missing: {missing}")
    output = Path("/kaggle/working/nextkey-kaggle-results.zip")
    if not output.parent.is_dir():
        output = root / "nextkey-kaggle-results.zip"
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in ARTIFACT_PATHS:
            absolute = root / path
            if absolute.is_file():
                archive.write(absolute, path)
            else:
                for child in absolute.rglob("*"):
                    if child.is_file():
                        archive.write(child, child.relative_to(root))
    return output


def main() -> None:
    root = repository_root()
    prepare_data(root)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "."], cwd=root, check=True)
    require_kaggle_gpu()
    run(root, "scripts/train_mvp_chartagger.py")
    run(root, "scripts/evaluate_mvp_chartagger.py")
    print(f"Wrote {write_zip(root)}")


if __name__ == "__main__":
    main()
