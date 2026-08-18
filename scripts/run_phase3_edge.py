#!/usr/bin/env python3
"""NextKey Phase 3: Edge Device Optimization — Student training (with optional KD).

When alpha=0 (no teacher), trains with hard labels only.
When alpha>0 and teacher_checkpoint is provided, applies Knowledge Distillation.

Usage:
    python scripts/run_phase3_edge.py \
        --config configs/phase3_edge/distill.yaml \
        --mode smoke --device mps \
        --output-dir artifacts/phase3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nextkey.utils.config_parser import load_merged_config
from nextkey.utils.seed import seed_everything
from nextkey.data.dataset import load_examples
from nextkey.data.tokenizer import build_vocab_from_examples
from nextkey.models import create_model
from nextkey.engine.trainer import ModelTrainer, resolve_device
from nextkey.engine.evaluator import ModelEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description="NextKey Phase 3: Edge Optimization")
    parser.add_argument("--config", type=str, required=True, help="Path to distillation config yaml")
    parser.add_argument("--base-config", type=str, default="configs/base.yaml")
    parser.add_argument("--mode", choices=["smoke", "research"], default="smoke")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="artifacts/phase3")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_merged_config(args.base_config, args.config, mode=args.mode, cli_device=args.device)
    seed_everything(cfg["seed"])

    device = resolve_device(cfg["runtime"]["device"])
    training = cfg["training"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    distill_cfg = cfg.get("distillation", {})

    alpha = float(distill_cfg.get("alpha", 0.0))
    mode_label = "KD" if alpha > 0 else "Hard-Label"

    print(f"\n{'='*60}")
    print(f"  Phase 3 — Edge Optimization [{mode_label}, α={alpha}]")
    print(f"  Mode: {args.mode.upper()} | Device: {device}")
    print(f"{'='*60}\n")

    # Load data
    t0 = time.time()
    max_train = int(training["max_train_samples"]) if training.get("max_train_samples") is not None else None
    max_eval = int(training["max_eval_samples"]) if training.get("max_eval_samples") is not None else None
    train_examples = load_examples(
        data_cfg["train_path"],
        max_train,
        int(data_cfg["max_seq_len"]),
    )
    val_examples = load_examples(
        data_cfg["dev_path"],
        max_eval,
        int(data_cfg["max_seq_len"]),
    )
    print(f"  Data loaded in {time.time() - t0:.1f}s: "
          f"train={len(train_examples)}, val={len(val_examples)}")

    if not train_examples:
        raise RuntimeError("No training examples found!")

    # Build vocab & student model
    vocab = build_vocab_from_examples(train_examples)
    student = create_model(
        model_cfg["name"],
        vocab_size=vocab.input_vocab_size,
        num_target_classes=vocab.target_vocab_size,
        embed_dim=int(model_cfg["embed_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        num_layers=int(model_cfg["num_layers"]),
        dropout=float(model_cfg["dropout"]),
    )
    print(f"  Student: {student.summary()}")

    # Train (uses DistillationLoss when alpha > 0, DualHeadLoss when alpha = 0)
    output_dir = Path(args.output_dir) / f"student-a{alpha}"
    trainer = ModelTrainer(student, vocab, cfg, output_dir, device)
    summary = trainer.fit(train_examples, val_examples)

    # Model size report
    model_path = output_dir / "best_model.pt"
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        summary["model_size_mb"] = round(size_mb, 2)
        print(f"\n  Model file size: {size_mb:.2f} MB")

    # Evaluate
    print(f"\n{'─'*60}")
    print(f"  Evaluating student model...")
    evaluator = ModelEvaluator(student, vocab, device, int(training["batch_size"]))
    eval_report = evaluator.full_evaluation(cfg, output_dir / "evaluation")

    for split_name, metrics in eval_report.get("metrics", {}).items():
        print(f"  [{split_name}] CER={metrics['corpus_cer']:.4f} "
              f"BF1={metrics['boundary_f1']:.4f} "
              f"EM={metrics['exact_match']:.4f}")

    summary["evaluation"] = eval_report
    summary_path = output_dir / "phase3_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\n  ✓ Phase 3 complete → {summary_path}")


if __name__ == "__main__":
    main()
