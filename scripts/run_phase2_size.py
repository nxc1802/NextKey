#!/usr/bin/env python3
"""NextKey Phase 2: Size & Topology Ablation CLI.

Train and evaluate size variants (Width, Depth, Topology) of the chosen backbone.

Usage:
    # 1. Single size variant
    python scripts/run_phase2_size.py --config configs/phase2_size/width_s.yaml --mode smoke --device mps

    # 2. Width sweep (XS, S, M, L)
    python scripts/run_phase2_size.py --sweep width --mode smoke --device mps

    # 3. Depth sweep (D1, D2, D3)
    python scripts/run_phase2_size.py --sweep depth --mode smoke --device mps

    # 4. Topology sweep (Wide-Shallow, Mid-Mid, Narrow-Deep)
    python scripts/run_phase2_size.py --sweep topo --mode smoke --device mps

    # 5. All size & topology variants in sequence + Comparative Report
    python scripts/run_phase2_size.py --all --mode smoke --device mps
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nextkey.utils.config_parser import load_merged_config
from nextkey.utils.seed import seed_everything
from nextkey.data.dataset import load_examples, AlignedExample
from nextkey.data.tokenizer import build_vocab_from_examples, CharVocab
from nextkey.models import create_model
from nextkey.engine.trainer import ModelTrainer, resolve_device
from nextkey.engine.evaluator import ModelEvaluator


SWEEPS = {
    "width": [
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
        "| Variant | Params | Val CER ↓ | Val BF1 ↑ | In-Domain CER ↓ | External CER ↓ | Domain Gap |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for res in sorted(results, key=lambda x: x.get("best_val_corpus_cer", float("inf"))):
        tag = res.get("variant_tag", "unknown")
        params = f"{res.get('parameters', 0) / 1000:.1f}K"
        val_cer = f"{res.get('best_val_corpus_cer', 0.0):.4f}"

        eval_m = res.get("evaluation", {}).get("metrics", {})
        in_cer = f"{eval_m.get('in_domain', {}).get('corpus_cer', 0.0):.4f}"
        ext_cer = f"{eval_m.get('external', {}).get('corpus_cer', 0.0):.4f}" if "external" in eval_m else "N/A"
        val_bf1 = f"{eval_m.get('in_domain', {}).get('boundary_f1', 0.0):.4f}"

        gap = res.get("evaluation", {}).get("domain_gap", {}).get("cer_gap", "N/A")
        gap_str = f"+{gap:.4f}" if isinstance(gap, (int, float)) else str(gap)

        lines.append(f"| `{tag}` | {params} | **{val_cer}** | {val_bf1} | {in_cer} | {ext_cer} | {gap_str} |")

    lines.append("")
    md_content = "\n".join(lines)
    md_path.write_text(md_content, encoding="utf-8")

    print("\n" + "=" * 65)
    print("  ✓ Phase 2 Size Ablation Comparison Summary:")
    print("=" * 65)
    print(md_content)
    print(f"✓ Report saved: {md_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="NextKey Phase 2: Size & Topology Ablation CLI")
    parser.add_argument("--config", type=str, default=None, help="Path to size config yaml")
    parser.add_argument("--sweep", choices=["width", "depth", "topo", "all"], default=None, help="Run a predefined sweep")
    parser.add_argument("--all", action="store_true", help="Run all size and topology variants in sequence")
    parser.add_argument("--base-config", type=str, default="configs/base.yaml")
    parser.add_argument("--mode", choices=["smoke", "research"], default="smoke")
    parser.add_argument("--device", type=str, default=None, help="Force device (cpu, cuda, mps)")
    parser.add_argument("--output-dir", type=str, default="artifacts/phase2")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine list of configs to run
    if args.all or args.sweep == "all":
        configs_to_run = SWEEPS["width"] + SWEEPS["depth"] + SWEEPS["topo"]
    elif args.sweep in SWEEPS:
        configs_to_run = SWEEPS[args.sweep]
    elif args.config:
        configs_to_run = [args.config]
    else:
        # Default to width sweep
        configs_to_run = SWEEPS["width"]

    # Pre-load shared dataset and vocab once
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
