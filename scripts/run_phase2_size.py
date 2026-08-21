#!/usr/bin/env python3
"""NextKey Phase 2: Size & Topology Ablation CLI.

Train and evaluate size variants (Width, Depth, Topology) of the chosen backbone.
Supports single-GPU, multi-GPU parallel (Kaggle Dual GPU T4x2), MPS, and CPU.

Usage:
    # 1. Single size variant
    python scripts/run_phase2_size.py --config configs/phase2_size/width_xxs.yaml --mode smoke --device mps

    # 2. Dual-GPU Parallel on Kaggle T4x2 (Ultra-Small sweep: XXS on cuda:0, XXXS on cuda:1)
    python scripts/run_phase2_size.py --sweep ultra_small --mode kaggle

    # 3. Custom parallel pair on 2 GPUs
    python scripts/run_phase2_size.py --configs configs/phase2_size/width_xxs.yaml configs/phase2_size/width_xxxs.yaml --parallel --mode research

    # 4. Width sweep (XXXS, XXS, XS, S, M, L)
    python scripts/run_phase2_size.py --sweep width --mode smoke --device mps

    # 5. All size & topology variants in sequence + Comparative Report
    python scripts/run_phase2_size.py --all --mode smoke --device mps
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nextkey.utils.config_parser import load_merged_config
from nextkey.utils.seed import seed_everything
from nextkey.data.dataset import load_examples, AlignedExample
from nextkey.data.tokenizer import build_vocab_from_examples, CharVocab
from nextkey.models import create_model
from nextkey.engine.trainer import ModelTrainer, resolve_device
from nextkey.engine.evaluator import ModelEvaluator


SWEEPS = {
    "ultra_small": [
        "configs/phase2_size/width_xxs.yaml",
        "configs/phase2_size/width_xxxs.yaml",
    ],
    "width": [
        "configs/phase2_size/width_xxxs.yaml",
        "configs/phase2_size/width_xxs.yaml",
        "configs/phase2_size/width_xs.yaml",
        "configs/phase2_size/width_s.yaml",
        "configs/phase2_size/width_m.yaml",
        "configs/phase2_size/width_l.yaml",
    ],
    "depth": [
        "configs/phase2_size/depth_1.yaml",
        "configs/phase2_size/depth_2.yaml",
        "configs/phase2_size/depth_3.yaml",
    ],
    "topo": [
        "configs/phase2_size/topo_a_wide_shallow.yaml",
        "configs/phase2_size/topo_b_mid_mid.yaml",
        "configs/phase2_size/topo_c_narrow_deep.yaml",
    ],
}


def train_single_variant(
    config_path: str | Path,
    base_config: str | Path,
    mode: str,
    device_req: str | None,
    output_base_dir: Path,
    train_examples: list[AlignedExample],
    val_examples: list[AlignedExample],
    vocab: CharVocab,
) -> dict[str, Any]:
    """Train and evaluate one size variant."""
    cfg = load_merged_config(base_config, config_path, mode=mode, cli_device=device_req)
    seed_everything(cfg["seed"])

    device = resolve_device(cfg["runtime"]["device"])
    training = cfg["training"]
    model_cfg = dict(cfg["model"])
    model_name = model_cfg.pop("name")

    tag = Path(config_path).stem

    print(f"\n{'='*65}")
    print(f"  [Phase 2] Training Size Variant: {tag.upper()}")
    print(f"  Mode: {mode.upper()} | Device: {device}")
    print(f"{'='*65}\n")

    model = create_model(
        model_name,
        vocab_size=vocab.input_vocab_size,
        num_target_classes=vocab.target_vocab_size,
        **model_cfg,
    )
    print(f"  Model spec: {model.summary()}")

    output_dir = output_base_dir / tag
    trainer = ModelTrainer(model, vocab, cfg, output_dir, device)
    summary = trainer.fit(train_examples, val_examples)

    # Evaluate
    print(f"\n{'─'*65}")
    print(f"  Evaluating size variant [{tag}]...")
    evaluator = ModelEvaluator(model, vocab, device, int(training["batch_size"]))
    eval_report = evaluator.full_evaluation(cfg, output_dir / "evaluation")

    for split_name, metrics in eval_report.get("metrics", {}).items():
        print(f"  [{split_name}] CER={metrics['corpus_cer']:.4f} "
              f"BF1={metrics['boundary_f1']:.4f} "
              f"EM={metrics['exact_match']:.4f}")

    if "domain_gap" in eval_report:
        gap = eval_report["domain_gap"]
        print(f"  Domain gap: CER +{gap['cer_gap']:.4f}, BF1 -{gap['bf1_gap']:.4f}")

    summary["variant_tag"] = tag
    summary["model_type"] = model_name
    summary["evaluation"] = eval_report

    summary_path = output_dir / "phase2_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def generate_size_report(results: list[dict[str, Any]], output_dir: Path) -> None:
    """Generate comparative summary JSON & Markdown table for all size variants."""
    report_path = output_dir / "size_ablation_results.json"
    md_path = output_dir / "size_ablation_results.md"

    report_data = {
        "title": "NextKey Phase 2: Size & Topology Ablation Comparison",
        "variants": results,
    }
    report_path.write_text(json.dumps(report_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# NextKey Phase 2 — Size & Topology Ablation Results",
        "",
        "| Variant | Params | Checkpoint Size | Val CER ↓ | Val BF1 ↑ | In-Domain CER ↓ | External CER ↓ | Domain Gap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for res in sorted(results, key=lambda x: x.get("best_val_corpus_cer", float("inf"))):
        tag = res.get("variant_tag", "unknown")
        params = f"{res.get('parameters', 0) / 1000:.1f}K"
        ckpt_kb = f"{res.get('parameters', 0) * 4 / 1024:.1f} KB"
        val_cer = f"{res.get('best_val_corpus_cer', 0.0):.4f}"

        eval_m = res.get("evaluation", {}).get("metrics", {})
        in_cer = f"{eval_m.get('in_domain', {}).get('corpus_cer', 0.0):.4f}"
        ext_cer = f"{eval_m.get('external', {}).get('corpus_cer', 0.0):.4f}" if "external" in eval_m else "N/A"
        val_bf1 = f"{eval_m.get('in_domain', {}).get('boundary_f1', 0.0):.4f}"

        gap = res.get("evaluation", {}).get("domain_gap", {}).get("cer_gap", "N/A")
        gap_str = f"+{gap:.4f}" if isinstance(gap, (int, float)) else str(gap)

        lines.append(f"| `{tag}` | {params} | {ckpt_kb} | **{val_cer}** | {val_bf1} | {in_cer} | {ext_cer} | {gap_str} |")

    lines.append("")
    md_content = "\n".join(lines)
    md_path.write_text(md_content, encoding="utf-8")

    print("\n" + "=" * 65)
    print("  ✓ Phase 2 Size Ablation Comparison Summary:")
    print("=" * 65)
    print(md_content)
    print(f"✓ Report saved: {md_path}")


def run_parallel_dual_gpu_phase2(
    configs: list[str],
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    """Launch 2 training processes in parallel across cuda:0 and cuda:1."""
    if len(configs) < 2:
        print("  [Parallel] Need at least 2 configs for dual-GPU execution. Falling back.")
        return

    cfg1, cfg2 = configs[0], configs[1]
    tag1 = Path(cfg1).stem
    tag2 = Path(cfg2).stem

    print("\n" + "=" * 75)
    print("  [Kaggle Dual-GPU Parallel Phase 2 Training]")
    print(f"  GPU 0 (cuda:0) -> Variant: {tag1.upper()} ({cfg1})")
    print(f"  GPU 1 (cuda:1) -> Variant: {tag2.upper()} ({cfg2})")
    print(f"  Mode: {args.mode.upper()} | Output: {output_dir}")
    print("=" * 75 + "\n")

    t_start = time.time()
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"

    cmd1 = [
        sys.executable, __file__,
        "--config", cfg1,
        "--base-config", args.base_config,
        "--mode", args.mode,
        "--device", "cuda:0",
        "--output-dir", str(output_dir),
    ]
    cmd2 = [
        sys.executable, __file__,
        "--config", cfg2,
        "--base-config", args.base_config,
        "--mode", args.mode,
        "--device", "cuda:1",
        "--output-dir", str(output_dir),
    ]

    p1 = subprocess.Popen(cmd1, env=env)
    p2 = subprocess.Popen(cmd2, env=env)

    exit1 = p1.wait()
    exit2 = p2.wait()

    if exit1 != 0 or exit2 != 0:
        raise RuntimeError(f"Dual GPU training failed: GPU 0 exit={exit1}, GPU 1 exit={exit2}")

    elapsed = round(time.time() - t_start, 1)
    print(f"\n✓ Dual GPU training finished in {elapsed}s (~2x speedup on Kaggle T4x2)!")

    # Aggregate summaries from both output directories
    results: list[dict[str, Any]] = []
    for cfg in (cfg1, cfg2):
        tag = Path(cfg).stem
        sum_path = output_dir / tag / "phase2_summary.json"
        if sum_path.exists():
            results.append(json.loads(sum_path.read_text(encoding="utf-8")))

    if results:
        generate_size_report(results, output_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="NextKey Phase 2: Size & Topology Ablation CLI")
    parser.add_argument("--config", type=str, default=None, help="Path to size config yaml")
    parser.add_argument("--configs", nargs="+", default=None, help="List of specific size config yamls to run")
    parser.add_argument(
        "--sweep",
        choices=["ultra_small", "width", "depth", "topo", "all"],
        default=None,
        help="Run a predefined sweep (ultra_small = XXS + XXXS)",
    )
    parser.add_argument("--all", action="store_true", help="Run all size and topology variants in sequence")
    parser.add_argument("--parallel", action="store_true", help="Run 2 variants in parallel on cuda:0 and cuda:1")
    parser.add_argument("--base-config", type=str, default="configs/base.yaml")
    parser.add_argument("--mode", choices=["smoke", "research", "kaggle"], default="smoke")
    parser.add_argument("--device", type=str, default=None, help="Force device (cpu, cuda, cuda:0, cuda:1, mps)")
    parser.add_argument("--output-dir", type=str, default="artifacts/phase2")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine list of configs to run
    if args.configs:
        configs_to_run = args.configs
    elif args.all or args.sweep == "all":
        configs_to_run = SWEEPS["width"] + SWEEPS["depth"] + SWEEPS["topo"]
    elif args.sweep in SWEEPS:
        configs_to_run = SWEEPS[args.sweep]
    elif args.config:
        configs_to_run = [args.config]
    else:
        # Default to ultra_small sweep
        configs_to_run = SWEEPS["ultra_small"]

    # Check for Dual GPU parallel execution on Kaggle / Multi-GPU
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    should_run_parallel = (
        len(configs_to_run) == 2
        and (args.parallel or args.mode == "kaggle" or (args.device in (None, "cuda") and num_gpus >= 2))
        and num_gpus >= 2
    )

    if should_run_parallel:
        run_parallel_dual_gpu_phase2(configs_to_run, args, output_dir)
        return

    # Single-process sequential run
    first_cfg = load_merged_config(args.base_config, configs_to_run[0], mode=args.mode, cli_device=args.device)
    data_cfg = first_cfg["data"]
    training_cfg = first_cfg["training"]

    print(f"Loading datasets for Phase 2 (Mode: {args.mode.upper()})...")
    t0 = time.time()
    max_train = int(training_cfg["max_train_samples"]) if training_cfg.get("max_train_samples") is not None else None
    max_eval = int(training_cfg["max_eval_samples"]) if training_cfg.get("max_eval_samples") is not None else None
    train_examples = load_examples(data_cfg["train_path"], max_train, int(data_cfg["max_seq_len"]))
    val_examples = load_examples(data_cfg["dev_path"], max_eval, int(data_cfg["max_seq_len"]))
    print(f"Data loaded in {time.time() - t0:.1f}s: train={len(train_examples):,d}, val={len(val_examples):,d}")

    if not train_examples:
        raise RuntimeError("No training examples found!")

    vocab = build_vocab_from_examples(train_examples)

    results: list[dict[str, Any]] = []
    for cfg_path in configs_to_run:
        res = train_single_variant(
            config_path=cfg_path,
            base_config=args.base_config,
            mode=args.mode,
            device_req=args.device,
            output_base_dir=output_dir,
            train_examples=train_examples,
            val_examples=val_examples,
            vocab=vocab,
        )
        results.append(res)

    if len(results) > 1:
        generate_size_report(results, output_dir)


if __name__ == "__main__":
    main()
