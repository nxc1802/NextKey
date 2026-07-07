#!/usr/bin/env python3
"""Train the context-aware MVP char tagger baseline."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from nextkey.cli import load_config
from nextkey.models.mvp_chartagger import (
    CharTaggerFactory,
    batch_examples,
    build_vocab,
    load_aligned_examples,
    require_torch,
    save_vocab,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MVP char tagger baseline.")
    parser.add_argument("--config", type=Path, default=Path("configs/model/mvp_chartagger_smoke.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    torch, nn = require_torch()

    data = config["data"]
    model_config = config["model"]
    training = config["training"]
    seed = int(training["seed"])
    random.seed(seed)
    torch.manual_seed(seed)

    examples = load_aligned_examples(
        Path(data["train_path"]),
        max_samples=int(training["max_train_samples"]),
        max_len=int(training["max_len"]),
    )
    vocab = build_vocab(examples)
    model = CharTaggerFactory.build(
        char_vocab_size=len(vocab.char_itos),
        label_vocab_size=len(vocab.label_itos),
        embedding_dim=int(model_config["embedding_dim"]),
        hidden_dim=int(model_config["hidden_dim"]),
        num_layers=int(model_config["num_layers"]),
        dropout=float(model_config["dropout"]),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=float(training["learning_rate"]))
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_label_id)
    step = 0
    losses: list[float] = []
    log_every = int(training["log_every"])
    max_steps = int(training.get("max_steps", 0)) or None

    model.train()
    for _epoch in range(int(training["epochs"])):
        for source, labels in batch_examples(examples, vocab, int(training["batch_size"])):
            optimizer.zero_grad()
            logits = model(source)
            loss = criterion(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            losses.append(float(loss.item()))
            if step % log_every == 0:
                print(json.dumps({"step": step, "loss": round(sum(losses[-log_every:]) / log_every, 4)}))
            if max_steps is not None and step >= max_steps:
                break
        if max_steps is not None and step >= max_steps:
            break

    output_path = Path(model_config["output_path"])
    vocab_path = Path(model_config["vocab_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": config,
            "training_stats": {
                "examples": len(examples),
                "steps": step,
                "final_loss": losses[-1] if losses else None,
                "char_vocab_size": len(vocab.char_itos),
                "label_vocab_size": len(vocab.label_itos),
            },
        },
        output_path,
    )
    save_vocab(vocab_path, vocab)
    print(
        json.dumps(
            {
                "model_path": str(output_path),
                "vocab_path": str(vocab_path),
                "examples": len(examples),
                "steps": step,
                "final_loss": losses[-1] if losses else None,
                "char_vocab_size": len(vocab.char_itos),
                "label_vocab_size": len(vocab.label_itos),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
