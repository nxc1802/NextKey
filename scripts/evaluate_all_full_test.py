#!/usr/bin/env python3
"""Run full comprehensive evaluation on 100% of test data (232,029 samples) for all trained research models."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nextkey.data.dataset import AlignedExample, iter_batches, load_examples
from nextkey.data.tokenizer import CharVocab
from nextkey.models import create_model
from nextkey.utils.config_parser import load_config
from nextkey.utils.metrics import MetricTotals


def evaluate_model_full(
    model: torch.nn.Module,
    vocab: CharVocab,
    device: torch.device,
    examples: list[AlignedExample],
    batch_size: int = 512,
    desc: str = "Test Set",
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Evaluate model on a list of examples with overall and per-domain metric breakdowns."""
    model.eval()
    totals = MetricTotals()
    domain_totals: dict[str, MetricTotals] = {}

    total_samples = len(examples)
    t0 = time.time()

    with torch.no_grad():
        for i, (source, targets, boundaries, lengths, chunk) in enumerate(
            iter_batches(examples, vocab, batch_size=batch_size, domain_balanced=False)
        ):
            source = source.to(device)
            outputs = model(source, lengths)

            char_ids = outputs["diacritic_logits"].argmax(-1).cpu()
            boundary_preds = (outputs["boundary_logits"].sigmoid() >= 0.5).long().cpu()

            for idx, length in enumerate(lengths.tolist()):
                prediction = vocab.decode(
                    char_ids[idx, :length].tolist(),
                    boundary_preds[idx, :length].tolist(),
                )
                target = vocab.decode(
                    targets[idx, :length].tolist(),
                    boundaries[idx, :length].clamp_min(0).tolist(),
                )

                totals.update(prediction, target)
                dom = chunk[idx].domain
                if dom not in domain_totals:
                    domain_totals[dom] = MetricTotals()
                domain_totals[dom].update(prediction, target)

            if (i + 1) % 50 == 0 or (i + 1) * batch_size >= total_samples:
                done = min((i + 1) * batch_size, total_samples)
                elapsed = time.time() - t0
                speed = done / max(elapsed, 0.001)
                print(f"    [{desc}] {done:>7,d}/{total_samples:,d} ({done/total_samples*100:>5.1f}%) | "
                      f"Speed: {speed:>5.0f} samples/s | Running CER: {totals.as_dict()['corpus_cer']:.4f}")

    overall_metrics = totals.as_dict()
    per_domain_metrics = {dom: tot.as_dict() for dom, tot in sorted(domain_totals.items())}
    return overall_metrics, per_domain_metrics


def main():
    device = torch.device("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    print("=" * 75)
    print(f"  NextKey Full 100% Test Data Benchmark (Device: {device})")
    print("=" * 75)

    test_in_domain_path = "data/processed/jdwr_v1/test/in_domain"
    test_external_path = "data/processed/jdwr_v1/test/external"

    print("\nLoading 100% of test datasets from disk...")
    t0 = time.time()
    in_domain_examples = load_examples(test_in_domain_path, max_samples=None)
    external_examples = load_examples(test_external_path, max_samples=None)
    print(f"✓ Datasets loaded in {time.time() - t0:.1f}s:")
    print(f"   • In-Domain Test Split:  {len(in_domain_examples):>7,d} sentences (7 distinct domains)")
    print(f"   • External Test Split:   {len(external_examples):>7,d} sentences (Sports domain)")
    print(f"   • Total Test Sentences:  {len(in_domain_examples) + len(external_examples):>7,d} sentences\n")

    models_to_eval = [
        ("Topo-A Wide/Shallow (96/160, 1L)", "artifacts/phase2/topo_a_wide_shallow", "configs/phase2_size/topo_a_wide_shallow.yaml"),
        ("BiGRU Baseline (64/128, 1L)", "artifacts/phase1/bigru", "configs/phase1_backbone/bigru.yaml"),
        ("Width-XS Edge Model (32/64, 1L)", "artifacts/phase2/width_xs", "configs/phase2_size/width_xs.yaml"),
    ]

    all_reports: list[dict[str, Any]] = []

    for label, model_dir, config_path in models_to_eval:
        dir_path = Path(model_dir)
        ckpt_path = dir_path / "best_model.pt"
        vocab_path = dir_path / "vocab.json"

        if not ckpt_path.exists() or not vocab_path.exists():
            print(f"Skipping {label}, checkpoint not found.")
            continue

        print(f"\n{'─'*75}")
        print(f"  Evaluating: {label}")
        print(f"{'─'*75}")

        vocab = CharVocab.load(vocab_path)
        cfg = load_config(config_path)
        model_cfg = dict(cfg.get("model", {}))
        model_name = model_cfg.pop("name")

        model = create_model(
            model_name,
            vocab_size=vocab.input_vocab_size,
            num_target_classes=vocab.target_vocab_size,
            **model_cfg,
        )
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
        model.to(device)

        # 1. In-Domain Evaluation
        print(f"  [1/2] Evaluating In-Domain Split (72,078 samples)...")
        in_metrics, in_domain_breakdown = evaluate_model_full(
            model, vocab, device, in_domain_examples, batch_size=512, desc="In-Domain"
        )

        # 2. External Evaluation
        print(f"  [2/2] Evaluating External Split (159,951 samples)...")
        ext_metrics, ext_domain_breakdown = evaluate_model_full(
            model, vocab, device, external_examples, batch_size=512, desc="External"
        )

        report = {
            "label": label,
            "model_name": model_name,
            "parameters": model.count_parameters(),
            "model_size_kb": round(ckpt_path.stat().st_size / 1024, 1),
            "in_domain": in_metrics,
            "external": ext_metrics,
            "in_domain_by_category": in_domain_breakdown,
            "external_by_category": ext_domain_breakdown,
            "domain_gap": {
                "cer_gap": round(ext_metrics["corpus_cer"] - in_metrics["corpus_cer"], 6),
                "wer_gap": round(ext_metrics["corpus_wer"] - in_metrics["corpus_wer"], 6),
                "bf1_gap": round(in_metrics["boundary_f1"] - ext_metrics["boundary_f1"], 6),
            },
        }

        # Save per-model full evaluation report
        eval_dir = dir_path / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)
        with open(eval_dir / "full_100pct_test_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        all_reports.append(report)

        print(f"\n  ✓ Result for {label}:")
        print(f"     • In-Domain (72K)  -> CER: {in_metrics['corpus_cer']:.4f} | WER: {in_metrics['corpus_wer']:.4f} | BF1: {in_metrics['boundary_f1']:.4f} | EM: {in_metrics['exact_match']:.4f}")
        print(f"     • External (160K)  -> CER: {ext_metrics['corpus_cer']:.4f} | WER: {ext_metrics['corpus_wer']:.4f} | BF1: {ext_metrics['boundary_f1']:.4f} | EM: {ext_metrics['exact_match']:.4f}")
        print(f"     • Domain Gap (CER) -> +{report['domain_gap']['cer_gap']:.4f}")

    # Save consolidated full benchmark
    out_json = Path("artifacts/full_100pct_test_benchmark.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, ensure_ascii=False, indent=2)

    # Generate Markdown Summary
    md_lines = [
        "# NextKey — Báo Cáo Đánh Giá Trên 100% Toàn Bộ Tập Dữ Liệu Test",
        "",
        f"Đánh giá toàn diện trên toàn bộ **232.029 câu kiểm thử** ({len(in_domain_examples):,d} In-domain + {len(external_examples):,d} External).",
        "",
        "## 1. Bảng so sánh tổng thể (100% Test Data)",
        "",
        "| Model | Params | Size (KB) | In-Domain CER ↓ | In-Domain WER ↓ | In-Domain BF1 ↑ | In-Domain EM ↑ | External CER ↓ | External WER ↓ | External BF1 ↑ | Domain Gap (CER) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for r in all_reports:
        md_lines.append(
            f"| **{r['label']}** | {r['parameters']:,d} | {r['model_size_kb']:.1f} KB | "
            f"**{r['in_domain']['corpus_cer']:.4f}** ({r['in_domain']['corpus_cer']*100:.2f}%) | "
            f"{r['in_domain']['corpus_wer']:.4f} | {r['in_domain']['boundary_f1']:.4f} | "
            f"{r['in_domain']['exact_match']*100:.2f}% | "
            f"**{r['external']['corpus_cer']:.4f}** ({r['external']['corpus_cer']*100:.2f}%) | "
            f"{r['external']['corpus_wer']:.4f} | {r['external']['boundary_f1']:.4f} | "
            f"+{r['domain_gap']['cer_gap']:.4f} |"
        )

    md_lines.extend([
        "",
        "## 2. Chi tiết hiệu năng từng miền dữ liệu (Per-Domain CER & Boundary F1)",
        "",
        "| Miền dữ liệu (Domain) | Số mẫu Test | Topo-A CER | BiGRU Baseline CER | Width-XS CER | Topo-A Boundary F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ])

    sample_report = all_reports[0]
    for dom in sorted(sample_report["in_domain_by_category"]):
        count = sample_report["in_domain_by_category"][dom]["count"]
        t_cer = all_reports[0]["in_domain_by_category"][dom]["corpus_cer"]
        b_cer = all_reports[1]["in_domain_by_category"][dom]["corpus_cer"]
        w_cer = all_reports[2]["in_domain_by_category"][dom]["corpus_cer"]
        t_bf1 = all_reports[0]["in_domain_by_category"][dom]["boundary_f1"]
        md_lines.append(f"| `{dom}` | {count:,d} | **{t_cer:.4f}** | {b_cer:.4f} | {w_cer:.4f} | {t_bf1:.4f} |")

    # External domain
    for dom in sorted(sample_report["external_by_category"]):
        count = sample_report["external_by_category"][dom]["count"]
        t_cer = all_reports[0]["external_by_category"][dom]["corpus_cer"]
        b_cer = all_reports[1]["external_by_category"][dom]["corpus_cer"]
        w_cer = all_reports[2]["external_by_category"][dom]["corpus_cer"]
        t_bf1 = all_reports[0]["external_by_category"][dom]["boundary_f1"]
        md_lines.append(f"| `external / {dom}` *(Ngoại miền)* | {count:,d} | **{t_cer:.4f}** | {b_cer:.4f} | {w_cer:.4f} | {t_bf1:.4f} |")

    md_out = Path("artifacts/FULL_TEST_REPORT.md")
    md_out.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"\n✓ Full report written to: {md_out} and {out_json}")


if __name__ == "__main__":
    main()
