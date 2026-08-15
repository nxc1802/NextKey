#!/usr/bin/env python3
"""Evaluate JDWR v1 on in-domain and external-domain splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nextkey.cli import load_config
from nextkey.evaluation.mvp_metrics import MetricTotals
from nextkey.models.mvp_chartagger import CharTaggerFactory, batch_examples, load_aligned_examples, load_vocab, require_torch


def device_for(torch, requested: str | None):
    if requested and requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate_path(model, path: Path, vocab, device, max_samples: int, max_len: int, batch_size: int,
                  length_bucket_size: int = 0, pad_to_multiple_of: int = 1):
    torch, _ = require_torch(); examples = load_aligned_examples(path, max_samples, max_len); totals = MetricTotals(); rows = []
    model.eval()
    with torch.no_grad():
        for source, targets, boundaries, lengths, chunk in batch_examples(
                examples, vocab, batch_size, length_bucket_size=length_bucket_size,
                pad_to_multiple_of=pad_to_multiple_of):
            char_logits, boundary_logits = model(source.to(device), lengths)
            chars = char_logits.argmax(-1).cpu(); spaces = (boundary_logits.sigmoid() >= .5).long().cpu()
            for index, length in enumerate(lengths.tolist()):
                prediction = vocab.decode(chars[index, :length].tolist(), spaces[index, :length].tolist())
                target = vocab.decode(targets[index, :length].tolist(), boundaries[index, :length].clamp_min(0).tolist())
                totals.update(prediction, target)
                rows.append({"input": chunk[index].source, "prediction": prediction, "target": target})
    return totals.as_dict(), rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate JDWR v1 two-head char tagger.")
    parser.add_argument("--config", type=Path, default=Path("configs/model/mvp_chartagger_smoke.yaml"))
    args = parser.parse_args(); config = load_config(args.config)
    torch, _ = require_torch(); data, model_config, training, evaluation = config["data"], config["model"], config["training"], config["evaluation"]
    vocab = load_vocab(Path(model_config["vocab_path"])); checkpoint = torch.load(Path(model_config["output_path"]), map_location="cpu")
    model = CharTaggerFactory.build(len(vocab.char_itos), len(vocab.target_itos), int(model_config["embedding_dim"]),
                                    int(model_config["hidden_dim"]), int(model_config["num_layers"]), float(model_config["dropout"]))
    model.load_state_dict(checkpoint["state_dict"])
    device = device_for(torch, training.get("device")); model.to(device)
    paths = {"in_domain": Path(data["test_path"] if "test_path" in data else data["dev_path"])}
    if data.get("external_test_path"):
        paths["external"] = Path(data["external_test_path"])
    report, all_rows = {"model_path": model_config["output_path"], "metrics": {}}, []
    for name, path in paths.items():
        metrics, rows = evaluate_path(model, path, vocab, device, int(training["max_eval_samples"]),
                                      int(training["max_len"]), int(training["batch_size"]),
                                      int(training.get("length_bucket_size", 0)),
                                      int(training.get("pad_to_multiple_of", 1)))
        report["metrics"][name] = metrics; all_rows.extend({"split": name, **row} for row in rows)
    if "external" in report["metrics"]:
        report["domain_generalization_gap"] = {"cer": round(report["metrics"]["external"]["corpus_cer"] - report["metrics"]["in_domain"]["corpus_cer"], 6),
            "boundary_f1": round(report["metrics"]["in_domain"]["boundary_f1"] - report["metrics"]["external"]["boundary_f1"], 6)}
    predictions_path, report_path, report_md = Path(evaluation["predictions_path"]), Path(evaluation["report_path"]), Path(evaluation["report_md_path"])
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in all_rows), encoding="utf-8")
    report_path.parent.mkdir(parents=True, exist_ok=True); report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# JDWR v1 Evaluation", ""]
    for name, metrics in report["metrics"].items():
        lines.extend([f"## {name}", "", "| Exact | Corpus CER | Boundary F1 | Diacritic Acc |", "|---:|---:|---:|---:|",
                      f"| {metrics['exact_match']:.4f} | {metrics['corpus_cer']:.4f} | {metrics['boundary_f1']:.4f} | {metrics['diacritic_accuracy']:.4f} |", ""])
    if "domain_generalization_gap" in report:
        gap = report["domain_generalization_gap"]
        lines.extend(["## Domain generalization gap", "", f"- External − in-domain Corpus CER: {gap['cer']:.4f}",
                      f"- In-domain − external Boundary F1: {gap['boundary_f1']:.4f}", ""])
    report_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
