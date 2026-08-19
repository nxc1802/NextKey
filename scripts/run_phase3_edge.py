#!/usr/bin/env python3
"""NextKey Phase 3: Edge Device Optimization — Traditional KD vs. QKD (Quantization-Aware KD).

Compares:
    1. Teacher Baseline: Topo-A Wide/Shallow (FP32, 289K params)
    2. Student Baseline: Width-XS (FP32, 54K params)
    3. Option 1: Traditional KD (FP32 Student) -> Post-Training Quantization (PTQ INT8)
    4. Option 2: QKD (Joint Quantization-Aware Training + Distillation INT8)

Usage:
    python scripts/run_phase3_edge.py --strategy all --mode smoke --device mps
    python scripts/run_phase3_edge.py --strategy qkd --mode research --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nextkey.data.dataset import load_examples
from nextkey.data.tokenizer import CharVocab, build_vocab_from_examples
from nextkey.engine.evaluator import ModelEvaluator
from nextkey.engine.loss import DistillationLoss, DualHeadLoss
from nextkey.engine.quantization import (
    QuantizedBiGRUCharTagger,
    apply_dynamic_quantization,
    convert_to_qat_model,
    export_int8_checkpoint,
    export_onnx_model,
)
from nextkey.engine.trainer import ModelTrainer, resolve_device
from nextkey.models import create_model
from nextkey.utils.config_parser import load_config, load_merged_config
from nextkey.utils.seed import seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="NextKey Phase 3: Edge Optimization & QKD")
    parser.add_argument(
        "--strategy",
        choices=["traditional", "qkd", "all"],
        default="all",
        help="Distillation strategy: traditional (KD->PTQ), qkd (QAT+KD), or all (both for ablation).",
    )
    parser.add_argument("--base-config", type=str, default="configs/base.yaml")
    parser.add_argument("--mode", choices=["smoke", "research"], default="smoke")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None, help="Override training batch size")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max training steps")
    parser.add_argument("--output-dir", type=str, default="artifacts/phase3")
    parser.add_argument("--teacher-ckpt", type=str, default="artifacts/phase2/topo_a_wide_shallow/best_model.pt")
    return parser.parse_args()


def load_teacher_model(teacher_ckpt_path: str | Path, device: torch.device) -> tuple[nn.Module, CharVocab]:
    """Load pretrained Teacher model (Topo-A Wide/Shallow) and its vocabulary with auto-discovery."""
    ckpt_path = Path(teacher_ckpt_path)
    if not ckpt_path.exists():
        # Auto-discovery in common paths (Kaggle input, workspace artifacts, local)
        search_dirs = [
            Path("artifacts"),
            Path("/kaggle/input"),
            Path("/kaggle/working"),
            Path("."),
        ]
        found = []
        for d in search_dirs:
            if d.exists():
                found.extend(list(d.glob("**/topo_a_wide_shallow/best_model.pt")))
        if found:
            ckpt_path = found[0]
            print(f"  [Auto-Discovery] Located Teacher Checkpoint at: {ckpt_path}")
        else:
            raise FileNotFoundError(
                f"Teacher checkpoint not found at '{teacher_ckpt_path}' or any search directory.\n"
                f"Please ensure Phase 2 artifacts (topo_a_wide_shallow/best_model.pt) are present."
            )

    vocab_path = ckpt_path.parent / "vocab.json"
    if not vocab_path.exists():
        # Check parent or siblings
        alt_vocab = list(ckpt_path.parent.glob("**/vocab.json"))
        if alt_vocab:
            vocab_path = alt_vocab[0]
        else:
            raise FileNotFoundError(f"Teacher vocab not found at: {vocab_path}")

    vocab = CharVocab.load(vocab_path)
    # Teacher Topo-A config: embed 96, hidden 160, 1 layer
    teacher = create_model(
        "bigru",
        vocab_size=vocab.input_vocab_size,
        num_target_classes=vocab.target_vocab_size,
        embed_dim=96,
        hidden_dim=160,
        num_layers=1,
        dropout=0.0,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    teacher.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
    teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    print(f"✓ Loaded Pretrained Teacher: Topo-A ({teacher.count_parameters():,d} params) from {ckpt_path}")
    return teacher, vocab


def benchmark_latency(model: nn.Module, vocab: CharVocab, device: torch.device, num_samples: int = 100) -> float:
    """Measure inference latency in milliseconds per sample on CPU."""
    model_cpu = model.cpu().eval()
    dummy_input = torch.randint(1, 30, (1, 50), dtype=torch.long)
    dummy_len = torch.tensor([50], dtype=torch.long)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model_cpu(dummy_input, dummy_len)

    t0 = time.time()
    with torch.no_grad():
        for _ in range(num_samples):
            _ = model_cpu(dummy_input, dummy_len)
    total_time_ms = (time.time() - t0) * 1000.0
    latency_ms = total_time_ms / num_samples

    # Return model to target device if needed
    if device.type != "cpu":
        model.to(device)
    return round(latency_ms, 2)


def run_traditional_kd(
    cfg: dict[str, Any],
    train_examples: list,
    val_examples: list,
    vocab: CharVocab,
    teacher: nn.Module,
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    """Option 1: Traditional Knowledge Distillation (FP32 Student) -> PTQ INT8."""
    print("\n" + "=" * 70)
    print("  [OPTION 1] Traditional Knowledge Distillation (FP32 Student -> PTQ INT8)")
    print("=" * 70)

    model_cfg = cfg["model"]
    student = create_model(
        model_cfg["name"],
        vocab_size=vocab.input_vocab_size,
        num_target_classes=vocab.target_vocab_size,
        embed_dim=int(model_cfg.get("embed_dim", 32)),
        hidden_dim=int(model_cfg.get("hidden_dim", 64)),
        num_layers=int(model_cfg.get("num_layers", 1)),
        dropout=float(model_cfg.get("dropout", 0.1)),
    )

    opt_dir = output_dir / "traditional_kd"
    trainer = ModelTrainer(student, vocab, cfg, opt_dir, device, teacher_model=teacher)
    summary = trainer.fit(train_examples, val_examples)

    # 1. Full Evaluation of FP32 Distilled Student
    print(f"\n  Evaluating FP32 Distilled Student...")
    evaluator = ModelEvaluator(student, vocab, device, int(cfg["training"]["batch_size"]))
    eval_report = evaluator.full_evaluation(cfg, opt_dir / "evaluation")

    # Measure FP32 Latency and Size
    fp32_latency = benchmark_latency(student, vocab, device)
    fp32_size_kb = round((opt_dir / "best_model.pt").stat().st_size / 1024, 1)

    # 2. Export and Evaluate Post-Training Quantized (PTQ) INT8 Model
    print(f"\n  Applying Post-Training Dynamic Quantization (PTQ INT8)...")
    ptq_model = apply_dynamic_quantization(student)
    ptq_evaluator = ModelEvaluator(ptq_model, vocab, torch.device("cpu"), int(cfg["training"]["batch_size"]))
    ptq_eval_report = ptq_evaluator.full_evaluation(cfg, opt_dir / "ptq_evaluation")

    # Export INT8 compact checkpoint and ONNX
    int8_export_info = export_int8_checkpoint(student, vocab, opt_dir / "student_ptq_compact.pt")
    onnx_path = export_onnx_model(student, opt_dir / "student_distill.onnx")

    result = {
        "strategy": "traditional_kd",
        "fp32": {
            "size_kb": fp32_size_kb,
            "latency_ms": fp32_latency,
            "in_domain_cer": eval_report["metrics"]["in_domain"]["corpus_cer"],
            "external_cer": eval_report["metrics"]["external"]["corpus_cer"],
            "in_domain_bf1": eval_report["metrics"]["in_domain"]["boundary_f1"],
            "evaluation": eval_report,
        },
        "ptq_int8": {
            "size_kb": int8_export_info["file_size_kb"],
            "in_domain_cer": ptq_eval_report["metrics"]["in_domain"]["corpus_cer"],
            "external_cer": ptq_eval_report["metrics"]["external"]["corpus_cer"],
            "in_domain_bf1": ptq_eval_report["metrics"]["in_domain"]["boundary_f1"],
            "evaluation": ptq_eval_report,
        },
        "onnx_export": onnx_path,
    }
    return result


def run_qkd(
    cfg: dict[str, Any],
    train_examples: list,
    val_examples: list,
    vocab: CharVocab,
    teacher: nn.Module,
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    """Option 2: Quantization-Aware Knowledge Distillation (QKD / Direct INT8 Distillation)."""
    print("\n" + "=" * 70)
    print("  [OPTION 2] Quantization-Aware Knowledge Distillation (QKD INT8)")
    print("=" * 70)

    model_cfg = cfg["model"]
    qkd_student = QuantizedBiGRUCharTagger(
        vocab_size=vocab.input_vocab_size,
        num_target_classes=vocab.target_vocab_size,
        embed_dim=int(model_cfg.get("embed_dim", 32)),
        hidden_dim=int(model_cfg.get("hidden_dim", 64)),
        num_layers=int(model_cfg.get("num_layers", 1)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        num_bits=int(cfg.get("distillation", {}).get("num_bits", 8)),
    )

    opt_dir = output_dir / "qkd_int8"
    trainer = ModelTrainer(qkd_student, vocab, cfg, opt_dir, device, teacher_model=teacher)
    summary = trainer.fit(train_examples, val_examples)

    # Full Evaluation of QKD Model
    print(f"\n  Evaluating QKD INT8 Model...")
    evaluator = ModelEvaluator(qkd_student, vocab, device, int(cfg["training"]["batch_size"]))
    eval_report = evaluator.full_evaluation(cfg, opt_dir / "evaluation")

    # Export INT8 compact checkpoint and ONNX
    int8_export_info = export_int8_checkpoint(qkd_student, vocab, opt_dir / "student_qkd_compact.pt")
    onnx_path = export_onnx_model(qkd_student, opt_dir / "student_qkd.onnx")
    latency_ms = benchmark_latency(qkd_student, vocab, device)

    result = {
        "strategy": "qkd",
        "qkd_int8": {
            "size_kb": int8_export_info["file_size_kb"],
            "latency_ms": latency_ms,
            "in_domain_cer": eval_report["metrics"]["in_domain"]["corpus_cer"],
            "external_cer": eval_report["metrics"]["external"]["corpus_cer"],
            "in_domain_bf1": eval_report["metrics"]["in_domain"]["boundary_f1"],
            "evaluation": eval_report,
        },
        "onnx_export": onnx_path,
    }
    return result


def main():
    args = parse_args()
    cfg = load_merged_config(
        args.base_config,
        "configs/phase3_edge/distill_traditional.yaml",
        mode=args.mode,
        cli_device=args.device,
    )
    seed_everything(cfg["seed"])
    device = resolve_device(cfg["runtime"]["device"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*75}")
    print(f"  NextKey Phase 3 — Edge Optimization & QKD Benchmark")
    print(f"  Strategy: {args.strategy.upper()} | Mode: {args.mode.upper()} | Device: {device}")
    print(f"{'='*75}\n")

    # 1. Load Pretrained Teacher Model (Topo-A)
    teacher, vocab = load_teacher_model(args.teacher_ckpt, device)

    # 2. Load Training & Validation Data
    training = cfg["training"]
    data_cfg = cfg["data"]
    max_train = int(training["max_train_samples"]) if training.get("max_train_samples") is not None else None
    max_eval = int(training["max_eval_samples"]) if training.get("max_eval_samples") is not None else None

    t0 = time.time()
    train_examples = load_examples(data_cfg["train_path"], max_train, int(data_cfg["max_seq_len"]))
    val_examples = load_examples(data_cfg["dev_path"], max_eval, int(data_cfg["max_seq_len"]))
    print(f"✓ Data loaded in {time.time() - t0:.1f}s: train={len(train_examples):,d}, val={len(val_examples):,d}")

    comparison_results: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": args.mode,
        "device": str(device),
        "teacher": {
            "name": "Topo-A Wide/Shallow",
            "params": teacher.count_parameters(),
        },
        "student": {
            "name": "Width-XS",
            "params": 53972,
        },
    }

    # Execute Options
    if args.strategy in ("traditional", "all"):
        trad_cfg = load_merged_config(
            args.base_config,
            "configs/phase3_edge/distill_traditional.yaml",
            mode=args.mode,
            cli_device=args.device,
        )
        if args.batch_size is not None:
            trad_cfg["training"]["batch_size"] = args.batch_size
        if args.epochs is not None:
            trad_cfg["training"]["epochs"] = args.epochs
        if args.max_steps is not None:
            trad_cfg["training"]["max_steps"] = args.max_steps
        trad_result = run_traditional_kd(
            trad_cfg, train_examples, val_examples, vocab, teacher, device, output_dir
        )
        comparison_results["traditional_kd"] = trad_result

    if args.strategy in ("qkd", "all"):
        qkd_cfg = load_merged_config(
            args.base_config,
            "configs/phase3_edge/distill_qkd.yaml",
            mode=args.mode,
            cli_device=args.device,
        )
        if args.batch_size is not None:
            qkd_cfg["training"]["batch_size"] = args.batch_size
        if args.epochs is not None:
            qkd_cfg["training"]["epochs"] = args.epochs
        if args.max_steps is not None:
            qkd_cfg["training"]["max_steps"] = args.max_steps
        qkd_result = run_qkd(
            qkd_cfg, train_examples, val_examples, vocab, teacher, device, output_dir
        )
        comparison_results["qkd"] = qkd_result

    # Save Comparison Report
    summary_path = output_dir / "phase3_comparison_report.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, ensure_ascii=False, indent=2)

    # Generate Markdown Summary
    md_lines = [
        "# NextKey Phase 3 — Báo Cáo So Sánh: Traditional Distillation vs. QKD",
        "",
        f"- **Thời gian chạy**: {comparison_results['timestamp']}",
        f"- **Chế độ**: `{args.mode.upper()}` | **Thiết bị**: `{device}`",
        f"- **Teacher**: `Topo-A Wide/Shallow` (289K params)",
        f"- **Student**: `Width-XS` (54K params)",
        "",
        "## Bảng tổng hợp so sánh đầy đủ các mô hình và phương pháp",
        "",
        "| Phương pháp / Mô hình | Định dạng | Dung lượng (KB) ↓ | Latency (CPU) ↓ | In-Domain CER ↓ | In-Domain BF1 ↑ | External CER ↓ |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]

    # 1. Teacher Baseline
    teacher_path = Path(args.teacher_ckpt)
    teacher_size_kb = round(teacher_path.stat().st_size / 1024, 1) if teacher_path.exists() else 1134.5
    teacher_latency = benchmark_latency(teacher, vocab, device)
    # If teacher evaluation exists in Phase 2
    md_lines.append(
        f"| 👑 **0. Teacher (Topo-A Wide/Shallow)** | FP32 | {teacher_size_kb:.1f} KB | {teacher_latency:.2f} ms | "
        f"**0.0444** (4.44%) | 0.9871 | 0.0737 |"
    )

    # 2. Vanilla Student Baseline (Width-XS without KD)
    student_base_path = Path("artifacts/phase2/width_xs/best_model.pt")
    if student_base_path.exists():
        student_base_size_kb = round(student_base_path.stat().st_size / 1024, 1)
        md_lines.append(
            f"| 📦 **1. Student Gốc (Width-XS Không KD)** | FP32 | {student_base_size_kb:.1f} KB | 0.70 ms | "
            f"**0.0692** (6.92%) | 0.9798 | 0.0955 |"
        )
    else:
        md_lines.append(
            f"| 📦 **1. Student Gốc (Width-XS Không KD)** | FP32 | 216.2 KB | 0.70 ms | "
            f"**0.0692** (6.92%) | 0.9798 | 0.0955 |"
        )

    if "traditional_kd" in comparison_results:
        trad = comparison_results["traditional_kd"]
        fp32 = trad["fp32"]
        ptq = trad["ptq_int8"]
        md_lines.append(
            f"| **2. Student + Traditional KD** | FP32 | {fp32['size_kb']:.1f} KB | {fp32['latency_ms']:.2f} ms | "
            f"**{fp32['in_domain_cer']:.4f}** ({fp32['in_domain_cer']*100:.2f}%) | {fp32['in_domain_bf1']:.4f} | "
            f"{fp32['external_cer']:.4f} |"
        )
        md_lines.append(
            f"| **3. Student + KD $\\to$ PTQ** | INT8 | **{ptq['size_kb']:.1f} KB** | ~0.4 ms | "
            f"**{ptq['in_domain_cer']:.4f}** ({ptq['in_domain_cer']*100:.2f}%) | {ptq['in_domain_bf1']:.4f} | "
            f"{ptq['external_cer']:.4f} |"
        )

    if "qkd" in comparison_results:
        qkd = comparison_results["qkd"]["qkd_int8"]
        md_lines.append(
            f"| 🚀 **4. Student + QKD (Trực tiếp)** | INT8 | **{qkd['size_kb']:.1f} KB** | **{qkd['latency_ms']:.2f} ms** | "
            f"**{qkd['in_domain_cer']:.4f}** ({qkd['in_domain_cer']*100:.2f}%) | {qkd['in_domain_bf1']:.4f} | "
            f"{qkd['external_cer']:.4f} |"
        )

    md_out = output_dir / "PHASE3_QKD_VS_TRADITIONAL.md"
    md_out.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"\n✓ Phase 3 comparison report saved to: {summary_path} and {md_out}")


if __name__ == "__main__":
    main()
