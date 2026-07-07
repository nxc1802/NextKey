"""Small PyTorch char-level seq2seq baseline for the MVP."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PAD = "<pad>"
SOS = "<s>"
EOS = "</s>"
UNK = "<unk>"


def require_torch():
    try:
        import torch
        from torch import nn
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyTorch is required for the neural baseline. Run: "
            "python3 -m venv .venv && source .venv/bin/activate && "
            "python -m pip install -e '.[neural]'"
        ) from exc
    return torch, nn


@dataclass
class CharVocab:
    stoi: dict[str, int]
    itos: list[str]

    @classmethod
    def build(cls, texts: list[str]) -> "CharVocab":
        chars = sorted({ch for text in texts for ch in text})
        itos = [PAD, SOS, EOS, UNK, *chars]
        stoi = {token: index for index, token in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD]

    @property
    def sos_id(self) -> int:
        return self.stoi[SOS]

    @property
    def eos_id(self) -> int:
        return self.stoi[EOS]

    @property
    def unk_id(self) -> int:
        return self.stoi[UNK]

    def encode(self, text: str, add_sos: bool = False, add_eos: bool = True) -> list[int]:
        ids = [self.stoi.get(ch, self.unk_id) for ch in text]
        if add_sos:
            ids.insert(0, self.sos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        chars: list[str] = []
        for idx in ids:
            if idx == self.eos_id:
                break
            if idx in {self.pad_id, self.sos_id}:
                continue
            token = self.itos[idx] if 0 <= idx < len(self.itos) else ""
            if token not in {UNK, PAD, SOS, EOS}:
                chars.append(token)
        return "".join(chars).strip()

    def to_json(self) -> dict[str, Any]:
        return {"itos": self.itos}

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "CharVocab":
        itos = list(payload["itos"])
        return cls(stoi={token: index for index, token in enumerate(itos)}, itos=itos)


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_pairs(
    path: Path,
    max_samples: int,
    max_input_len: int,
    max_target_len: int,
) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for row in iter_jsonl(path):
        source = row["input"]
        target = row["target"]
        if len(source) <= max_input_len and len(target) <= max_target_len:
            pairs.append((source, target))
            if len(pairs) >= max_samples:
                break
    return pairs


class EncoderDecoder:
    """Factory wrapper to avoid importing torch at module import time."""

    @staticmethod
    def build(vocab_size: int, embedding_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        torch, nn = require_torch()

        class Seq2Seq(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
                self.encoder = nn.GRU(
                    embedding_dim,
                    hidden_dim,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0.0,
                )
                self.decoder = nn.GRU(
                    embedding_dim,
                    hidden_dim,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0.0,
                )
                self.output = nn.Linear(hidden_dim, vocab_size)

            def forward(self, src, tgt_in, teacher_forcing_ratio: float = 1.0):
                _, hidden = self.encoder(self.embedding(src))
                batch_size, target_steps = tgt_in.shape
                logits = []
                decoder_input = tgt_in[:, 0:1]
                for step in range(1, target_steps):
                    decoder_output, hidden = self.decoder(self.embedding(decoder_input), hidden)
                    step_logits = self.output(decoder_output)
                    logits.append(step_logits)
                    use_teacher = random.random() < teacher_forcing_ratio
                    decoder_input = tgt_in[:, step : step + 1] if use_teacher else step_logits.argmax(-1)
                return torch.cat(logits, dim=1)

            def generate(self, src, sos_id: int, eos_id: int, max_len: int):
                _, hidden = self.encoder(self.embedding(src))
                decoder_input = torch.full(
                    (src.shape[0], 1),
                    sos_id,
                    dtype=torch.long,
                    device=src.device,
                )
                outputs = []
                for _ in range(max_len):
                    decoder_output, hidden = self.decoder(self.embedding(decoder_input), hidden)
                    next_id = self.output(decoder_output).argmax(-1)
                    outputs.append(next_id)
                    decoder_input = next_id
                return torch.cat(outputs, dim=1)

        return Seq2Seq()


def pad_batch(sequences: list[list[int]], pad_id: int):
    torch, _ = require_torch()
    max_len = max(len(seq) for seq in sequences)
    batch = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
    for row, seq in enumerate(sequences):
        batch[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return batch


def batch_pairs(pairs: list[tuple[str, str]], vocab: CharVocab, batch_size: int):
    random.shuffle(pairs)
    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start : start + batch_size]
        src = pad_batch([vocab.encode(source) for source, _ in chunk], vocab.pad_id)
        tgt = pad_batch([vocab.encode(target, add_sos=True) for _, target in chunk], vocab.pad_id)
        yield src, tgt


def save_vocab(path: Path, vocab: CharVocab) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(vocab.to_json(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_vocab(path: Path) -> CharVocab:
    return CharVocab.from_json(json.loads(path.read_text(encoding="utf-8")))
