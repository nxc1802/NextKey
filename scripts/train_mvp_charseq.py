#!/usr/bin/env python3
"""Train a small char-level seq2seq MVP baseline."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from nextkey.cli import load_config
from nextkey.models.mvp_charseq import (
    CharVocab,
    EncoderDecoder,
    batch_pairs,
    load_pairs,
    require_torch,
    save_vocab,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MVP char-level seq2seq baseline.")
    parser.add_argument("--config", type=Path, default=Path("configs/model/mvp_charseq_smoke.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    torch, nn = require_torch()

    data = config["data"]
    model_config = config["model"]
    training = config["training"]
    seed = int(training.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)

    train_pairs = load_pairs(
        Path(data["train_path"]),
        max_samples=int(training["max_train_samples"]),
        max_input_len=int(training["max_input_len"]),
        max_target_len=int(training["max_target_len"]),
    )
    vocab = CharVocab.build([text for pair in train_pairs for text in pair])
    model = EncoderDecoder.build(
        vocab_size=len(vocab.itos),
        embedding_dim=int(model_config["embedding_dim"]),
        hidden_dim=int(model_config["hidden_dim"]),
        num_layers=int(model_config["num_layers"]),
        dropout=float(model_config["dropout"]),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=float(training["learning_rate"]))
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)
    batch_size = int(training["batch_size"])
    log_every = int(training.get("log_every", 100))
    step = 0
    losses: list[float] = []

    model.train()
    for epoch in range(int(training["epochs"])):
        for src, tgt in batch_pairs(train_pairs, vocab, batch_size):
            optimizer.zero_grad()
            logits = model(src, tgt, teacher_forcing_ratio=float(training["teacher_forcing_ratio"]))
            gold = tgt[:, 1:]
            loss = criterion(logits.reshape(-1, logits.shape[-1]), gold.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
            step += 1
            if step % log_every == 0:
                print(json.dumps({"step": step, "loss": round(sum(losses[-log_every:]) / log_every, 4)}))

    output_path = Path(model_config["output_path"])
    vocab_path = Path(model_config["vocab_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": config,
            "vocab_size": len(vocab.itos),
            "training_stats": {
                "train_pairs": len(train_pairs),
                "steps": step,
                "final_loss": losses[-1] if losses else None,
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
                "train_pairs": len(train_pairs),
                "steps": step,
                "final_loss": losses[-1] if losses else None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
