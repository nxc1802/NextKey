"""Unified evaluation pipeline for CharTagger models.

Evaluates on in-domain and external-domain splits, computes per-domain
metrics, and generates JSON + Markdown reports.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from nextkey.data.dataset import AlignedExample, iter_batches, load_examples
from nextkey.data.tokenizer import CharVocab
from nextkey.utils.metrics import MetricTotals


class ModelEvaluator:
    """Evaluate a CharTagger model on multiple test splits."""

    def __init__(
        self,
        model: torch.nn.Module,
        vocab: CharVocab,
        device: torch.device,
        batch_size: int = 64,
    ):
        self.model = model
        self.vocab = vocab
        self.device = device
        self.batch_size = batch_size
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def evaluate_examples(
        self,
        examples: list[AlignedExample],
    ) -> tuple[dict[str, float | int], list[dict[str, str]]]:
        """Evaluate on a list of examples.

        Returns:
            (metrics_dict, list_of_prediction_rows)
        """
        totals = MetricTotals()
        rows: list[dict[str, str]] = []

        for source, targets, boundaries, lengths, chunk in iter_batches(
            examples, self.vocab, self.batch_size,
        ):
            source = source.to(self.device)
            outputs = self.model(source, lengths)

            char_ids = outputs["diacritic_logits"].argmax(-1).cpu()
            boundary_preds = (outputs["boundary_logits"].sigmoid() >= 0.5).long().cpu()

            for idx, length in enumerate(lengths.tolist()):
                prediction = self.vocab.decode(
                    char_ids[idx, :length].tolist(),
                    boundary_preds[idx, :length].tolist(),
                )
                target = self.vocab.decode(
                    targets[idx, :length].tolist(),
                    boundaries[idx, :length].clamp_min(0).tolist(),
                )
                totals.update(prediction, target)
                rows.append({
                    "input": chunk[idx].source,
                    "prediction": prediction,
                    "target": target,
                    "domain": chunk[idx].domain,
                })

        return totals.as_dict(), rows

    def evaluate_split(
        self,
        path: str | Path,
        max_samples: int,
        max_len: int = 256,
    ) -> tuple[dict[str, float | int], list[dict[str, str]]]:
        """Load examples from path and evaluate."""
        examples = load_examples(path, max_samples, max_len)
        return self.evaluate_examples(examples)

    def full_evaluation(
        self,
        cfg: dict[str, Any],
        output_dir: str | Path,
    ) -> dict[str, Any]:
        """Run evaluation on in-domain and external test splits.

        Returns a comprehensive report dict.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        data_cfg = cfg.get("data", {})
        training_cfg = cfg.get("training", {})
        max_samples = int(training_cfg.get("max_eval_samples", 5000))
        max_len = int(data_cfg.get("max_seq_len", 256))

        splits: dict[str, str | Path] = {}
        if data_cfg.get("test_in_domain_path"):
            splits["in_domain"] = data_cfg["test_in_domain_path"]
        elif data_cfg.get("dev_path"):
            splits["in_domain"] = data_cfg["dev_path"]
        if data_cfg.get("test_external_path"):
            splits["external"] = data_cfg["test_external_path"]

        report: dict[str, Any] = {"metrics": {}}

        for name, path in splits.items():
            metrics, rows = self.evaluate_split(path, max_samples, max_len)
            report["metrics"][name] = metrics

            # Save predictions
            pred_path = output_dir / f"{name}_predictions.jsonl"
            pred_path.write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                encoding="utf-8",
            )

        # Domain generalization gap
        if "in_domain" in report["metrics"] and "external" in report["metrics"]:
            report["domain_gap"] = {
                "cer_gap": round(
                    report["metrics"]["external"]["corpus_cer"]
                    - report["metrics"]["in_domain"]["corpus_cer"], 6
                ),
                "bf1_gap": round(
                    report["metrics"]["in_domain"]["boundary_f1"]
                    - report["metrics"]["external"]["boundary_f1"], 6
                ),
            }

        # Save report
        report_path = output_dir / "evaluation_report.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        return report
