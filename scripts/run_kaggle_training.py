#!/usr/bin/env python3
"""NextKey Kaggle Runner CLI.

Automatically prepares data from attached Kaggle input datasets, installs local
dependencies, verifies CUDA GPU accelerators, executes the requested research phase(s),
and bundles all generated model artifacts and evaluation reports into a zip archive.

Usage examples:
    # 1. Default: Phase 1 BiGRU backbone training on full dataset (Research mode)
    python scripts/run_kaggle_training.py

    # 2. Fast sanity check on Kaggle GPU (Smoke mode)
    python scripts/run_kaggle_training.py --mode smoke

    # 3. Phase 2 Size & Topology search
    python scripts/run_kaggle_training.py --phase 2 --config configs/phase2_size/width_s.yaml

    # 4. Phase 3 Edge optimization
    python scripts/run_kaggle_training.py --phase 3 --config configs/phase3_edge/distill.yaml

    # 5. Run all phases in sequence
    python scripts/run_kaggle_training.py --phase all
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def find_processed_dir(root: Path) -> Path:
    """Locate jdwr_v1 processed data directory locally or in /kaggle/input."""
    local_processed = root / "data" / "processed"
    if (local_processed / "jdwr_v1" / "manifest.json").is_file():
        return local_processed / "jdwr_v1"

    # Search Kaggle input datasets
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        # Look for manifest.json under jdwr_v1
        manifests = sorted(kaggle_input.glob("**/jdwr_v1/manifest.json"))
        if manifests:
            return manifests[0].parent
        # Look for train jsonl files
        train_files = sorted(kaggle_input.glob("**/jdwr_v1/train/*.jsonl"))
        if train_files:
            return train_files[0].parents[1]

    raise FileNotFoundError(
        "Could not find jdwr_v1 dataset locally or under /kaggle/input.\n"
        "Please attach a Kaggle Dataset containing 'data/processed/jdwr_v1' or 'jdwr_v1'."
    )


def prepare_data(root: Path) -> None:
    """Link or copy discovered processed data into data/processed/jdwr_v1."""
    source = find_processed_dir(root)
    target = root / "data" / "processed" / "jdwr_v1"

    if source.resolve() == target.resolve():
        print(f"✓ Data already in place: {target}")
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if target.is_symlink() or target.is_file():
            target.unlink()
        else:
            shutil.rmtree(target)

    try:
        # Prefer fast symlink
        target.symlink_to(source, target_is_directory=True)
        print(f"✓ Symlinked data from {source} -> {target}")
    except OSError:
        # Fallback to copy if symlinks are not permitted
        shutil.copytree(source, target, dirs_exist_ok=True)
        print(f"✓ Copied data from {source} -> {target}")


def verify_device(device_req: str) -> str:
    """Verify accelerator availability."""
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required; please ensure dependencies are installed.") from error

    if device_req == "cuda":
        if not torch.cuda.is_available():
            print("⚠️ WARNING: CUDA requested but torch.cuda.is_available() is False.")
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                print("   Falling back to MPS accelerator.")
                return "mps"
            print("   Falling back to CPU.")
            return "cpu"
        print(f"✓ CUDA Accelerator: {torch.cuda.get_device_name(0)} (PyTorch {torch.__version__})")
        return "cuda"

    if device_req == "mps":
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            print(f"✓ Apple MPS Accelerator (PyTorch {torch.__version__})")
            return "mps"
        print("⚠️ MPS unavailable, falling back to CPU.")
        return "cpu"

    return device_req


def run_command(cmd: list[str], root: Path) -> None:
    """Run a subprocess command with PYTHONPATH set."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root / "src")
    print(f"\n[EXEC] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=root, env=env, check=True)


def package_artifacts(root: Path, output_zip: Path, artifact_dirs: list[Path]) -> Path:
    """Zip all generated artifacts, checkpoints, and reports into a compact archive."""
    print(f"\n[INFO] Packaging artifacts into: {output_zip} ...")
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as archive:
        for adir in artifact_dirs:
            if not adir.exists():
                continue
            if adir.is_file():
                archive.write(adir, adir.relative_to(root))
                count += 1
            else:
                for child in adir.rglob("*"):
                    if child.is_file():
                        # Exclude huge raw string prediction dumps to keep zip fast and compact (< 10 MB)
                        if child.suffix == ".jsonl":
                            continue
                        archive.write(child, child.relative_to(root))
                        count += 1

    zip_size_mb = output_zip.stat().st_size / (1024 * 1024)
    print(f"✓ Packaged {count} artifact files into: {output_zip} ({zip_size_mb:.2f} MB)")
    return output_zip


def parse_args():
    parser = argparse.ArgumentParser(description="NextKey Kaggle Runner CLI")
    parser.add_argument(
        "--phase",
        choices=["1", "2", "3", "4", "all"],
        default="4",
        help="Research phase to execute (1: Backbone, 2: Size, 3: Edge, 4: Tri-Task 3-in-1, all: sequence). Default: 4",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all models/variants for the selected phase (e.g. all 5 backbones for Phase 1, or all 10 sizes for Phase 2).",
    )
    parser.add_argument(
        "--sweep",
        choices=["ultra_small", "width", "depth", "topo", "all"],
        default=None,
        help="Specific sweep to execute for Phase 2 (ultra_small, width, depth, topo, all).",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help="List of specific size config YAMLs to execute in Phase 2.",
    )
    parser.add_argument(
        "--strategy",
        choices=["traditional", "qkd", "all"],
        default=None,
        help="Distillation strategy for Phase 3 (traditional, qkd, all). Default: all",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to phase config YAML. If omitted, uses default config for the selected phase.",
    )
    parser.add_argument(
        "--mode",
        choices=["smoke", "research", "kaggle"],
        default="kaggle",
        help="Execution mode (kaggle: 2x GPU parallel + full dataset; research: 1x GPU full dataset; smoke: 1K samples quick check). Default: kaggle",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Target compute device (cuda, cpu, mps). Default: cuda",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts",
        help="Directory to store training & evaluation artifacts. Default: artifacts",
    )
    parser.add_argument(
        "--zip-output",
        type=str,
        default=None,
        help="Path for output zip file. Default: /kaggle/working/nextkey-results.zip (or ./nextkey-results.zip)",
    )
    parser.add_argument(
        "--no-zip",
        action="store_true",
        help="Disable automatic zip packaging of artifacts.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = repository_root()

    print("=" * 70)
    print("  NextKey Kaggle Training & Benchmark Runner")
    print(f"  Phase: {args.phase} | Mode: {args.mode.upper()} | Target Device: {args.device}")
    print("=" * 70)

    # 1. Prepare data
    prepare_data(root)

    # 2. Check and verify accelerator
    actual_device = verify_device(args.device)

    # 3. Determine phases and commands to run
    artifact_paths: list[Path] = []

    if args.phase in ("1", "all"):
        phase_output_dir = Path(args.output_dir) / "phase1"
        artifact_paths.append(root / phase_output_dir)
        cmd = [
            sys.executable,
            "scripts/run_phase1_backbone.py",
            "--mode",
            args.mode,
            "--device",
            actual_device,
            "--output-dir",
            str(phase_output_dir),
        ]
        if args.all:
            cmd.append("--all")
        elif args.config and args.phase == "1":
            cmd.extend(["--config", args.config])
        else:
            cmd.extend(["--config", "configs/phase1_backbone/bigru.yaml"])
        run_command(cmd, root)

    if args.phase in ("2", "all"):
        phase_output_dir = Path(args.output_dir) / "phase2"
        artifact_paths.append(root / phase_output_dir)
        cmd = [
            sys.executable,
            "scripts/run_phase2_size.py",
            "--mode",
            args.mode,
            "--device",
            actual_device,
            "--output-dir",
            str(phase_output_dir),
        ]
        if args.all:
            cmd.append("--all")
        elif args.configs and args.phase == "2":
            cmd.extend(["--configs"] + args.configs)
        elif args.sweep:
            cmd.extend(["--sweep", args.sweep])
        elif args.config and args.phase == "2":
            cmd.extend(["--config", args.config])
        else:
            cmd.extend(["--sweep", "ultra_small"])
        run_command(cmd, root)

    if args.phase in ("3", "all"):
        phase_output_dir = Path(args.output_dir) / "phase3"
        artifact_paths.append(root / phase_output_dir)
        strategy = args.strategy or "all"
        cmd = [
            sys.executable,
            "scripts/run_phase3_edge.py",
            "--strategy",
            strategy,
            "--mode",
            args.mode,
            "--device",
            actual_device,
            "--output-dir",
            str(phase_output_dir),
        ]
        run_command(cmd, root)

    if args.phase in ("4", "all"):
        phase_output_dir = Path(args.output_dir) / "phase4_tritask"
        artifact_paths.append(root / phase_output_dir)
        cmd = [
            sys.executable,
            "scripts/run_phase4_tritask.py",
            "--all",
            "--mode",
            args.mode,
            "--output-dir",
            str(phase_output_dir),
        ]
        if args.device:
            cmd.extend(["--device", actual_device])
        run_command(cmd, root)

    # 5. Package artifacts
    if not args.no_zip:
        if args.zip_output:
            zip_dest = Path(args.zip_output)
        elif Path("/kaggle/working").is_dir():
            zip_dest = Path("/kaggle/working/nextkey-results.zip")
        else:
            zip_dest = root / "nextkey-results.zip"

        package_artifacts(root, zip_dest, artifact_paths)

    print("\n" + "=" * 70)
    print("  ✓ All requested tasks completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
