#!/usr/bin/env python3
"""Evaluate the context-aware MVP char tagger baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nextkey.cli import load_config
from nextkey.evaluation.mvp_metrics import MetricTotals
from nextkey.models.mvp_chartagger import (
    CharTaggerFactory,
    load_aligned_examples,
    load_vocab,
    pad_batch,
    require_torch,
)


def render_report(report: dict) -> str:
    metrics = report["metrics"]
    return "\n".join(
        [
            "# MVP CharTagger Smoke Report",
            "",
            f"- Model: `{report['model_path']}`",
            f"- Eval samples: {metrics['count']}",
            "",
            "| Exact | CER | WER | Token F1 | Spacing F1 | Diacritic Acc |",
            "|---:|---:|---:|---:|---:|---:|",
            (
                f"| {metrics['exact_match']:.4f} | {metrics['cer']:.4f} | "
                f"{metrics['wer']:.4f} | {metrics['token_f1']:.4f} | "
                f"{metrics['spacing_f1']:.4f} | {metrics['diacritic_accuracy']:.4f} |"
            ),
            "",
            "This is the first context-aware neural MVP baseline.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MVP char tagger baseline.")
    parser.add_argument("--config", type=Path, default=Path("configs/model/mvp_chartagger_smoke.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    torch, _ = require_torch()

    data = config["data"]
    model_config = config["model"]
    training = config["training"]
    evaluation = config["evaluation"]

    vocab = load_vocab(Path(model_config["vocab_path"]))
    checkpoint = torch.load(Path(model_config["output_path"]), map_location="cpu")
    model = CharTaggerFactory.build(
        char_vocab_size=len(vocab.char_itos),
        label_vocab_size=len(vocab.label_itos),
        embedding_dim=int(model_config["embedding_dim"]),
        hidden_dim=int(model_config["hidden_dim"]),
        num_layers=int(model_config["num_layers"]),
        dropout=float(model_config["dropout"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    examples = load_aligned_examples(
        Path(data["dev_path"]),
        max_samples=int(training["max_eval_samples"]),
        max_len=int(training["max_len"]),
    )
    totals = MetricTotals()
    error_examples = []
    predictions_path = Path(evaluation["predictions_path"])
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with predictions_path.open("w", encoding="utf-8") as handle, torch.no_grad():
        for source, labels in examples:
            source_batch = pad_batch([vocab.encode_chars(source)], vocab.pad_char_id)
            logits = model(source_batch)
            label_ids = logits.argmax(-1)[0, : len(source)].tolist()
            prediction = vocab.decode_labels(label_ids)
            target = "".join(labels).strip()
            totals.update(prediction, target)
            row = {"input": source, "prediction": prediction, "target": target}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            if len(error_examples) < 10 and prediction != target:
                error_examples.append(row)

    report = {
        "model_path": model_config["output_path"],
        "vocab_path": model_config["vocab_path"],
        "metrics": totals.as_dict(),
        "examples": error_examples,
    }
    report_path = Path(evaluation["report_path"])
    report_md_path = Path(evaluation["report_md_path"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_md_path.write_text(render_report(report), encoding="utf-8")
    print(f"Wrote {report_path}")
    print(f"Wrote {report_md_path}")


if __name__ == "__main__":
    main()
