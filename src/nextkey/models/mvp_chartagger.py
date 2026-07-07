"""Context-aware character tagging baseline for MVP restoration."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nextkey.data.mvp_dataset import compact_key, strip_accents
from nextkey.models.mvp_charseq import PAD, UNK, require_torch


@dataclass
class TaggerVocab:
    char_stoi: dict[str, int]
    char_itos: list[str]
    label_stoi: dict[str, int]
    label_itos: list[str]

    @property
    def pad_char_id(self) -> int:
        return self.char_stoi[PAD]

    @property
    def unk_char_id(self) -> int:
        return self.char_stoi[UNK]

    @property
    def pad_label_id(self) -> int:
        return self.label_stoi[PAD]

    def encode_chars(self, text: str) -> list[int]:
        return [self.char_stoi.get(ch, self.unk_char_id) for ch in text]

    def encode_labels(self, labels: list[str]) -> list[int]:
        return [self.label_stoi[label] for label in labels]

    def decode_labels(self, label_ids: list[int]) -> str:
        pieces: list[str] = []
        for label_id in label_ids:
            if label_id == self.pad_label_id:
                continue
            pieces.append(self.label_itos[label_id])
        return "".join(pieces).strip()

    def to_json(self) -> dict[str, Any]:
        return {"char_itos": self.char_itos, "label_itos": self.label_itos}

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "TaggerVocab":
        char_itos = list(payload["char_itos"])
        label_itos = list(payload["label_itos"])
        return cls(
            char_stoi={token: index for index, token in enumerate(char_itos)},
            char_itos=char_itos,
            label_stoi={token: index for index, token in enumerate(label_itos)},
            label_itos=label_itos,
        )


def target_to_labels(target: str) -> list[str]:
    labels: list[str] = []
    at_token_start = True
    for ch in target:
        if ch.isspace():
            at_token_start = True
            continue
        label = (" " + ch) if at_token_start and labels else ch
        labels.append(label)
        at_token_start = False
    return labels


def align_pair(source: str, target: str) -> tuple[str, list[str]] | None:
    source_key = compact_key(source)
    labels = target_to_labels(target)
    target_key = "".join(strip_accents(label.strip()) for label in labels)
    if source_key != target_key or len(source_key) != len(labels):
        return None
    return source_key, labels


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_aligned_examples(path: Path, max_samples: int, max_len: int) -> list[tuple[str, list[str]]]:
    examples: list[tuple[str, list[str]]] = []
    for row in iter_jsonl(path):
        aligned = align_pair(row["input"], row["target"])
        if aligned is None:
            continue
        source_key, labels = aligned
        if 0 < len(source_key) <= max_len:
            examples.append((source_key, labels))
            if len(examples) >= max_samples:
                break
    return examples


def build_vocab(examples: list[tuple[str, list[str]]]) -> TaggerVocab:
    chars = sorted({ch for source, _ in examples for ch in source})
    labels = sorted({label for _, labels_for_row in examples for label in labels_for_row})
    char_itos = [PAD, UNK, *chars]
    label_itos = [PAD, *labels]
    return TaggerVocab(
        char_stoi={token: index for index, token in enumerate(char_itos)},
        char_itos=char_itos,
        label_stoi={token: index for index, token in enumerate(label_itos)},
        label_itos=label_itos,
    )


class CharTaggerFactory:
    @staticmethod
    def build(
        char_vocab_size: int,
        label_vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ):
        _, nn = require_torch()

        class CharTagger(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(char_vocab_size, embedding_dim, padding_idx=0)
                self.encoder = nn.GRU(
                    embedding_dim,
                    hidden_dim,
                    num_layers=num_layers,
                    batch_first=True,
                    bidirectional=True,
                    dropout=dropout if num_layers > 1 else 0.0,
                )
                self.output = nn.Linear(hidden_dim * 2, label_vocab_size)

            def forward(self, source):
                encoded, _ = self.encoder(self.embedding(source))
                return self.output(encoded)

        return CharTagger()


def pad_batch(sequences: list[list[int]], pad_id: int):
    torch, _ = require_torch()
    max_len = max(len(seq) for seq in sequences)
    batch = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
    for row, seq in enumerate(sequences):
        batch[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return batch


def batch_examples(examples: list[tuple[str, list[str]]], vocab: TaggerVocab, batch_size: int):
    random.shuffle(examples)
    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        source = pad_batch([vocab.encode_chars(source) for source, _ in chunk], vocab.pad_char_id)
        labels = pad_batch([vocab.encode_labels(labels) for _, labels in chunk], vocab.pad_label_id)
        yield source, labels


def save_vocab(path: Path, vocab: TaggerVocab) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(vocab.to_json(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_vocab(path: Path) -> TaggerVocab:
    return TaggerVocab.from_json(json.loads(path.read_text(encoding="utf-8")))
