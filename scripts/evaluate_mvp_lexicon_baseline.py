#!/usr/bin/env python3
"""Evaluate the MVP lexicon baseline."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from nextkey.cli import load_config
from nextkey.evaluation.mvp_metrics import MetricTotals
from nextkey.models.mvp_lexicon_baseline import iter_jsonl, load_model


def evaluate_split(
    model_path: Path,
    split_path: Path,
    predictions_path: Path,
    max_samples: int | None = None,
    max_samples_per_category: int | None = None,
) -> dict[str, Any]:
    model = load_model(model_path)
    totals = MetricTotals()
    by_category: dict[str, MetricTotals] = defaultdict(MetricTotals)
    examples: list[dict[str, str]] = []
    category_seen: dict[str, int] = defaultdict(int)

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with predictions_path.open("w", encoding="utf-8") as output:
        for index, row in enumerate(iter_jsonl(split_path), start=1):
            if max_samples is not None and totals.count >= max_samples:
                break
            category = row["category"]
            if (
                max_samples_per_category is not None
                and category_seen[category] >= max_samples_per_category
            ):
                continue
            prediction = model.predict(row["input"])
            target = row["target"]
            totals.update(prediction, target)
            by_category[category].update(prediction, target)
            category_seen[category] += 1
            output.write(
                json.dumps(
                    {
                        "sample_id": row["sample_id"],
                        "category": row["category"],
                        "input": row["input"],
                        "prediction": prediction,
                        "target": target,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            if len(examples) < 20 and prediction != target:
                examples.append(
                    {
                        "sample_id": row["sample_id"],
                        "category": row["category"],
                        "input": row["input"],
                        "prediction": prediction,
                        "target": target,
                    }
                )

    return {
        "split_path": str(split_path),
        "predictions_path": str(predictions_path),
        "max_samples": max_samples,
        "max_samples_per_category": max_samples_per_category,
        "metrics": totals.as_dict(),
        "by_category": {category: values.as_dict() for category, values in sorted(by_category.items())},
        "error_examples": examples,
    }


def render_report(report: dict[str, Any]) -> str:
    lines = [
        "# MVP Lexicon Baseline Report",
        "",
        f"- Model: `{report['model_path']}`",
        "",
        "## Overall Metrics",
        "",
        "| Split | Count | Exact | CER | WER | Token F1 | Spacing F1 | Diacritic Acc |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, payload in report["splits"].items():
        metrics = payload["metrics"]
        lines.append(
            f"| `{split}` | {metrics['count']:,} | {metrics['exact_match']:.4f} | "
            f"{metrics['cer']:.4f} | {metrics['wer']:.4f} | {metrics['token_f1']:.4f} | "
            f"{metrics['spacing_f1']:.4f} | {metrics['diacritic_accuracy']:.4f} |"
        )

    for split, payload in report["splits"].items():
        lines.extend(
            [
                "",
                f"## `{split}` By Category",
                "",
                "| Category | Count | Exact | CER | WER | Token F1 | Spacing F1 | Diacritic Acc |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for category, metrics in payload["by_category"].items():
            lines.append(
                f"| `{category}` | {metrics['count']:,} | {metrics['exact_match']:.4f} | "
                f"{metrics['cer']:.4f} | {metrics['wer']:.4f} | {metrics['token_f1']:.4f} | "
                f"{metrics['spacing_f1']:.4f} | {metrics['diacritic_accuracy']:.4f} |"
            )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This is a no-GPU lexicon + dynamic-programming baseline.",
            "- It is intended as a feasibility floor, not the final model.",
            "- The MVP target excludes punctuation and capitalization.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MVP lexicon baseline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/model/mvp_lexicon_baseline.yaml"),
        help="Path to baseline config.",
    )
    parser.add_argument(
        "--split",
        choices=["dev", "test", "both"],
        default="both",
        help="Split to evaluate.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional per-split sample limit for quick feasibility evaluation.",
    )
    parser.add_argument(
        "--max-samples-per-category",
        type=int,
        default=None,
        help="Optional per-category sample limit for balanced quick evaluation.",
    )
    args = parser.parse_args()
    config = load_config(args.config)

    data_config = config.get("data", {})
    model_config = config.get("model", {})
    evaluation_config = config.get("evaluation", {})
    model_path = Path(model_config.get("output_path", "models/checkpoints/mvp-lexicon-baseline.json"))
    predictions_dir = Path(
        evaluation_config.get("predictions_dir", "experiments/runs/mvp-lexicon-baseline")
    )
    report_path = Path(evaluation_config.get("report_path", "experiments/reports/mvp-lexicon-baseline.json"))
    report_md_path = Path(
        evaluation_config.get("report_md_path", "experiments/reports/mvp-lexicon-baseline.md")
    )

    split_paths = {
        "dev": Path(data_config.get("dev_path", "data/processed/mvp/dev.jsonl")),
        "test": Path(data_config.get("test_path", "data/processed/mvp/test.jsonl")),
    }
    selected = ["dev", "test"] if args.split == "both" else [args.split]
    max_samples = args.max_samples
    max_samples_per_category = args.max_samples_per_category
    if (
        max_samples is None
        and max_samples_per_category is None
        and "quick_max_samples_per_category" in evaluation_config
    ):
        max_samples_per_category = int(evaluation_config["quick_max_samples_per_category"])
    elif max_samples is None and "quick_max_samples" in evaluation_config:
        max_samples = int(evaluation_config["quick_max_samples"])
    report: dict[str, Any] = {"model_path": str(model_path), "splits": {}}
    for split in selected:
        report["splits"][split] = evaluate_split(
            model_path,
            split_paths[split],
            predictions_dir / f"{split}.predictions.jsonl",
            max_samples=max_samples,
            max_samples_per_category=max_samples_per_category,
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_md_path.parent.mkdir(parents=True, exist_ok=True)
    report_md_path.write_text(render_report(report), encoding="utf-8")
    print(f"Wrote {report_path}")
    print(f"Wrote {report_md_path}")


if __name__ == "__main__":
    main()
