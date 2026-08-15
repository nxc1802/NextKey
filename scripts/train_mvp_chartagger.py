#!/usr/bin/env python3
"""Train the JDWR v1 two-head character tagger."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from nextkey.cli import load_config
from nextkey.evaluation.mvp_metrics import MetricTotals
from nextkey.models.mvp_chartagger import CharTaggerFactory, batch_examples, build_vocab, load_aligned_examples, require_torch, save_vocab


def device_for(torch, requested: str | None):
    if requested and requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate(model, examples, vocab, device, batch_size: int) -> dict[str, float | int]:
    torch, _ = require_torch()
    totals = MetricTotals(); model.eval()
    with torch.no_grad():
        for source, targets, boundaries, lengths, _ in batch_examples(examples, vocab, batch_size):
            char_logits, boundary_logits = model(source.to(device), lengths)
            char_ids = char_logits.argmax(-1).cpu(); boundary_ids = (boundary_logits.sigmoid() >= .5).long().cpu()
            for row, length in enumerate(lengths.tolist()):
                prediction = vocab.decode(char_ids[row, :length].tolist(), boundary_ids[row, :length].tolist())
                target = vocab.decode(targets[row, :length].tolist(), boundaries[row, :length].clamp_min(0).tolist())
                totals.update(prediction, target)
    return totals.as_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train JDWR v1 two-head char tagger.")
    parser.add_argument("--config", type=Path, default=Path("configs/model/mvp_chartagger_smoke.yaml"))
    args = parser.parse_args(); config = load_config(args.config)
    torch, nn = require_torch(); data, model_config, training = config["data"], config["model"], config["training"]
    seed = int(training["seed"]); random.seed(seed); torch.manual_seed(seed)
    device = device_for(torch, training.get("device"))
    train_examples = load_aligned_examples(data["train_path"], int(training["max_train_samples"]), int(training["max_len"]))
    dev_examples = load_aligned_examples(data["dev_path"], int(training["max_eval_samples"]), int(training["max_len"]))
    if not train_examples:
        raise RuntimeError("No aligned training examples were found.")
    vocab = build_vocab(train_examples)
    model = CharTaggerFactory.build(len(vocab.char_itos), len(vocab.target_itos), int(model_config["embedding_dim"]),
                                    int(model_config["hidden_dim"]), int(model_config["num_layers"]), float(model_config["dropout"])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training["learning_rate"]))
    char_loss = nn.CrossEntropyLoss(ignore_index=vocab.pad_target_id)
    boundary_loss = nn.BCEWithLogitsLoss(reduction="none")
    lambda_boundary = float(training.get("lambda_boundary", 1.0)); max_steps = int(training.get("max_steps", 0)) or None
    batch_size = int(training["batch_size"]); log_every = int(training["log_every"])
    best_cer, step, history = float("inf"), 0, []
    output_path, vocab_path = Path(model_config["output_path"]), Path(model_config["vocab_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True); save_vocab(vocab_path, vocab)
    for epoch in range(1, int(training["epochs"]) + 1):
        model.train(); loss_sum = 0.0; batches = 0
        for source, targets, boundaries, lengths, _ in batch_examples(
                train_examples, vocab, batch_size, bool(training.get("domain_balanced", True)),
                int(training.get("length_bucket_size", 0)), int(training.get("pad_to_multiple_of", 1))):
            source, targets, boundaries = source.to(device), targets.to(device), boundaries.to(device)
            optimizer.zero_grad(); char_logits, boundary_logits = model(source, lengths)
            loss_char = char_loss(char_logits.reshape(-1, char_logits.shape[-1]), targets.reshape(-1))
            valid = boundaries != -100
            loss_boundary = boundary_loss(boundary_logits[valid], boundaries[valid].float()).mean()
            loss = loss_char + lambda_boundary * loss_boundary; loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            step += 1; batches += 1; loss_sum += float(loss.item())
            if step % log_every == 0:
                print(json.dumps({"epoch": epoch, "step": step, "loss": round(loss_sum / batches, 5)}))
            if max_steps and step >= max_steps: break
        metrics = evaluate(model, dev_examples, vocab, device, batch_size)
        record = {"epoch": epoch, "step": step, "train_loss": round(loss_sum / max(batches, 1), 6), "dev": metrics}
        history.append(record)
        if metrics["corpus_cer"] <= best_cer:
            best_cer = float(metrics["corpus_cer"])
            torch.save({"state_dict": model.state_dict(), "config": config, "vocab_format": "jdwr-v1-two-head", "best_dev": metrics}, output_path)
        if max_steps and step >= max_steps: break
    history_path = output_path.with_name("training_history.json")
    history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"model_path": str(output_path), "vocab_path": str(vocab_path), "history_path": str(history_path),
                      "device": str(device), "train_examples": len(train_examples), "dev_examples": len(dev_examples),
                      "steps": step, "best_dev_corpus_cer": best_cer}, ensure_ascii=False))


if __name__ == "__main__":
    main()
