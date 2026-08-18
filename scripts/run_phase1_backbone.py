#!/usr/bin/env python3
"""NextKey Phase 1: Backbone Selection CLI.

Train and evaluate individual backbones or run the full 5-backbone sweep.

Usage:
    # 1. Single model (e.g. BiGRU)
    python scripts/run_phase1_backbone.py --config configs/phase1_backbone/bigru.yaml --mode smoke --device mps

    # 2. All 5 Phase 1 backbones in sequence + Pareto comparison report
    python scripts/run_phase1_backbone.py --all --mode smoke --device mps
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


DEFAULT_BACKBONES = [
    "configs/phase1_backbone/bigru.yaml",
    "configs/phase1_backbone/bilstm.yaml",
    "configs/phase1_backbone/cnn_tcn.yaml",
    "configs/phase1_backbone/cnn_bigru.yaml",
    "configs/phase1_backbone/tiny_transformer.yaml",
]


def train_single_backbone(
    config_path: str | Path,
    base_config: str | Path,
    mode: str,
    device_req: str | None,
    output_base_dir: Path,
    train_examples: list[AlignedExample],
    val_examples: list[AlignedExample],
    vocab: CharVocab,
) -> dict[str, Any]:
    """Train and evaluate one backbone model candidate."""
    cfg = load_merged_config(base_config, config_path, mode=mode, cli_device=device_req)
    seed_everything(cfg["seed"])

    device = resolve_device(cfg["runtime"]["device"])
    training = cfg["training"]
    model_cfg = dict(cfg["model"])
    model_name = model_cfg.pop("name")

    tag = Path(config_path).stem

    print(f"\n{'='*65}")
    print(f"  [Phase 1] Training Candidate: {tag.upper()} ({model_name})")
    print(f"  Mode: {mode.upper()} | Device: {device}")
    print(f"{'='*65}\n")

    # Instantiate model with its specific config kwargs
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
    print(f"  Evaluating candidate [{tag}]...")
    evaluator = ModelEvaluator(model, vocab, device, int(training["batch_size"]))
    eval_report = evaluator.full_evaluation(cfg, output_dir / "evaluation")

    for split_name, metrics in eval_report.get("metrics", {}).items():
        print(f"  [{split_name}] CER={metrics['corpus_cer']:.4f} "
              f"BF1={metrics['boundary_f1']:.4f} "
              f"EM={metrics['exact_match']:.4f}")

    if "domain_gap" in eval_report:
        gap = eval_report["domain_gap"]
        print(f"  Domain gap: CER +{gap['cer_gap']:.4f}, BF1 -{gap['bf1_gap']:.4f}")

    summary["candidate_tag"] = tag
    summary["model_type"] = model_name
    summary["evaluation"] = eval_report

    summary_path = output_dir / "phase1_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def generate_pareto_report(results: list[dict[str, Any]], output_dir: Path) -> None:
    """Generate comparative summary JSON & Markdown table for all backbones."""
    report_path = output_dir / "pareto_backbone_report.json"
    md_path = output_dir / "pareto_backbone_report.md"

    report_data = {
        "title": "NextKey Phase 1: Backbone Selection Pareto Comparison",
        "candidates": results,
    }
    report_path.write_text(json.dumps(report_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Render Markdown table
    lines = [
        "# NextKey Phase 1 — Backbone Selection Comparison",
        "",
        "| Candidate | Model Class | Params | Val CER ↓ | Val BF1 ↑ | In-Domain CER ↓ | External CER ↓ | Domain Gap |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for res in sorted(results, key=lambda x: x.get("best_val_corpus_cer", float("inf"))):
        tag = res.get("candidate_tag", "unknown")
        m_type = res.get("model_type", "")
        params = f"{res.get('parameters', 0) / 1000:.1f}K"
        val_cer = f"{res.get('best_val_corpus_cer', 0.0):.4f}"

        eval_m = res.get("evaluation", {}).get("metrics", {})
        in_cer = f"{eval_m.get('in_domain', {}).get('corpus_cer', 0.0):.4f}"
        ext_cer = f"{eval_m.get('external', {}).get('corpus_cer', 0.0):.4f}" if "external" in eval_m else "N/A"
        val_bf1 = f"{eval_m.get('in_domain', {}).get('boundary_f1', 0.0):.4f}"

        gap = res.get("evaluation", {}).get("domain_gap", {}).get("cer_gap", "N/A")
        gap_str = f"+{gap:.4f}" if isinstance(gap, (int, float)) else str(gap)

        lines.append(f"| `{tag}` | {m_type} | {params} | **{val_cer}** | {val_bf1} | {in_cer} | {ext_cer} | {gap_str} |")

    lines.append("")
    md_content = "\n".join(lines)
    md_path.write_text(md_content, encoding="utf-8")

    print("\n" + "=" * 65)
    print("  ✓ Phase 1 Pareto Comparison Summary:")
    print("=" * 65)
    print(md_content)
    print(f"✓ Report saved: {md_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="NextKey Phase 1: Backbone Selection CLI")
    parser.add_argument("--config", type=str, default=None, help="Path to backbone config yaml")
    parser.add_argument("--all", action="store_true", help="Run all 5 candidate backbones in sequence")
    parser.add_argument("--base-config", type=str, default="configs/base.yaml")
    parser.add_argument("--mode", choices=["smoke", "research"], default="smoke")
    parser.add_argument("--device", type=str, default=None, help="Force device (cpu, cuda, mps)")
    parser.add_argument("--output-dir", type=str, default="artifacts/phase1")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine list of configs to run
    if args.all or not args.config:
        configs_to_run = DEFAULT_BACKBONES
    else:
        configs_to_run = [args.config]

    # Pre-load shared dataset and vocab once
    first_cfg = load_merged_config(args.base_config, configs_to_run[0], mode=args.mode, cli_device=args.device)
    data_cfg = first_cfg["data"]
    training_cfg = first_cfg["training"]

    print(f"Loading datasets for Phase 1 (Mode: {args.mode.upper()})...")
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
        res = train_single_backbone(
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
        generate_pareto_report(results, output_dir)


if __name__ == "__main__":
    main()
