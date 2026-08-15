"""Two-head BiGRU character tagger for JDWR v1."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from nextkey.data.mvp_dataset import compact_key, normalize_mvp_target, strip_accents
from nextkey.models.mvp_charseq import PAD, UNK, require_torch


@dataclass
class JdwrExample:
    source: str
    char_target: str
    boundary_target: list[int]
    domain: str


@dataclass
class TaggerVocab:
    char_stoi: dict[str, int]
    char_itos: list[str]
    target_stoi: dict[str, int]
    target_itos: list[str]

    @property
    def pad_char_id(self) -> int:
        return self.char_stoi[PAD]

    @property
    def unk_char_id(self) -> int:
        return self.char_stoi[UNK]

    @property
    def pad_target_id(self) -> int:
        return self.target_stoi[PAD]

    def encode_chars(self, text: str) -> list[int]:
        return [self.char_stoi.get(ch, self.unk_char_id) for ch in text]

    def encode_target(self, text: str) -> list[int]:
        return [self.target_stoi.get(ch, self.target_stoi[UNK]) for ch in text]

    def decode(self, target_ids: list[int], boundary_predictions: list[int]) -> str:
        chars: list[str] = []
        for index, target_id in enumerate(target_ids):
            if target_id == self.pad_target_id:
                continue
            char = self.target_itos[target_id]
            if char in {PAD, UNK}:
                continue
            if chars and index < len(boundary_predictions) and boundary_predictions[index]:
                chars.append(" ")
            chars.append(char)
        return "".join(chars)

    def to_json(self) -> dict[str, Any]:
        return {"char_itos": self.char_itos, "target_itos": self.target_itos, "format": "jdwr-v1-two-head"}

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "TaggerVocab":
        char_itos, target_itos = list(payload["char_itos"]), list(payload["target_itos"])
        return cls({token: i for i, token in enumerate(char_itos)}, char_itos,
                   {token: i for i, token in enumerate(target_itos)}, target_itos)


def _targets_from_text(target: str) -> tuple[str, list[int]]:
    chars: list[str] = []
    boundaries: list[int] = []
    boundary = False
    for char in normalize_mvp_target(target):
        if char.isspace():
            boundary = bool(chars)
        else:
            chars.append(char); boundaries.append(int(boundary)); boundary = False
    if boundaries:
        boundaries[0] = 0
    return "".join(chars), boundaries


def align_pair(source: str, target: str) -> JdwrExample | None:
    source_key = compact_key(source)
    char_target, boundary_target = _targets_from_text(target)
    if not source_key or source_key != strip_accents(char_target):
        return None
    return JdwrExample(source_key, char_target, boundary_target, "unknown")


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _paths(path_or_paths: Path | str | Iterable[Path | str]) -> list[Path]:
    if isinstance(path_or_paths, (str, Path)):
        path = Path(path_or_paths)
        return sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    return [Path(path) for path in path_or_paths]


def load_aligned_examples(path_or_paths: Path | str | Iterable[Path | str], max_samples: int,
                          max_len: int) -> list[JdwrExample]:
    paths = _paths(path_or_paths)
    if not paths or max_samples <= 0:
        return []
    # Keep each domain represented when a bounded smoke sample is requested.
    quota = -(-max_samples // len(paths))
    examples: list[JdwrExample] = []
    for path in paths:
        path_examples: list[JdwrExample] = []
        for row in iter_jsonl(path):
            source = row.get("input", "")
            char_target = row.get("char_target")
            boundary_target = row.get("boundary_target")
            if not isinstance(char_target, str) or not isinstance(boundary_target, list):
                aligned = align_pair(source, row.get("target", ""))
                if aligned is None:
                    continue
                aligned.domain = str(row.get("domain", row.get("category", "unknown")))
            else:
                aligned = JdwrExample(compact_key(source), char_target, [int(value) for value in boundary_target],
                                      str(row.get("domain", row.get("category", "unknown"))))
                if aligned.source != strip_accents(aligned.char_target):
                    continue
            if 0 < len(aligned.source) <= max_len and len(aligned.source) == len(aligned.boundary_target):
                path_examples.append(aligned)
                if len(path_examples) >= quota:
                    break
        examples.extend(path_examples)
    return examples[:max_samples]


def build_vocab(examples: list[JdwrExample]) -> TaggerVocab:
    input_chars = sorted({char for example in examples for char in example.source})
    target_chars = sorted({char for example in examples for char in example.char_target})
    char_itos, target_itos = [PAD, UNK, *input_chars], [PAD, UNK, *target_chars]
    return TaggerVocab({token: i for i, token in enumerate(char_itos)}, char_itos,
                       {token: i for i, token in enumerate(target_itos)}, target_itos)


class CharTaggerFactory:
    @staticmethod
    def build(char_vocab_size: int, target_vocab_size: int, embedding_dim: int, hidden_dim: int,
              num_layers: int, dropout: float):
        torch, nn = require_torch()

        class TwoHeadCharTagger(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(char_vocab_size, embedding_dim, padding_idx=0)
                self.encoder = nn.GRU(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                                      bidirectional=True, dropout=dropout if num_layers > 1 else 0.0)
                self.char_head = nn.Linear(hidden_dim * 2, target_vocab_size)
                self.boundary_head = nn.Linear(hidden_dim * 2, 1)

            def forward(self, source, lengths=None):
                embeddings = self.embedding(source)
                if lengths is not None:
                    packed = nn.utils.rnn.pack_padded_sequence(embeddings, lengths.cpu(), batch_first=True,
                                                                enforce_sorted=False)
                    encoded, _ = self.encoder(packed)
                    encoded, _ = nn.utils.rnn.pad_packed_sequence(encoded, batch_first=True,
                                                                   total_length=source.shape[1])
                else:
                    encoded, _ = self.encoder(embeddings)
                return self.char_head(encoded), self.boundary_head(encoded).squeeze(-1)

        return TwoHeadCharTagger()


def pad_batch(sequences: list[list[int]], pad_id: int, pad_to_multiple_of: int = 1):
    torch, _ = require_torch()
    max_length = max(map(len, sequences))
    if pad_to_multiple_of > 1:
        max_length = -(-max_length // pad_to_multiple_of) * pad_to_multiple_of
    batch = torch.full((len(sequences), max_length), pad_id, dtype=torch.long)
    for row, sequence in enumerate(sequences):
        batch[row, :len(sequence)] = torch.tensor(sequence, dtype=torch.long)
    return batch


def _ordered_examples(examples: list[JdwrExample], domain_balanced: bool) -> list[JdwrExample]:
    if not domain_balanced:
        result = list(examples); random.shuffle(result); return result
    by_domain: dict[str, list[JdwrExample]] = defaultdict(list)
    for example in examples:
        by_domain[example.domain].append(example)
    for values in by_domain.values():
        random.shuffle(values)
    domains = sorted(by_domain)
    result: list[JdwrExample] = []
    positions = defaultdict(int)
    while True:
        added = False
        for domain in domains:
            position = positions[domain]
            if position < len(by_domain[domain]):
                result.append(by_domain[domain][position]); positions[domain] += 1; added = True
        if not added:
            return result


def batch_examples(examples: list[JdwrExample], vocab: TaggerVocab, batch_size: int,
                   domain_balanced: bool = False, length_bucket_size: int = 0,
                   pad_to_multiple_of: int = 1):
    ordered = _ordered_examples(examples, domain_balanced)
    if length_bucket_size:
        if length_bucket_size < batch_size:
            raise ValueError("length_bucket_size must be at least batch_size")
        # Sorting only bounded windows preserves the randomized/domain-balanced
        # ordering between windows while avoiding excessive padding for long rows.
        ordered = [example for start in range(0, len(ordered), length_bucket_size)
                   for example in sorted(ordered[start:start + length_bucket_size], key=lambda item: len(item.source))]
    for start in range(0, len(ordered), batch_size):
        chunk = ordered[start:start + batch_size]
        if not chunk:
            continue
        source_ids = [vocab.encode_chars(example.source) for example in chunk]
        target_ids = [vocab.encode_target(example.char_target) for example in chunk]
        lengths = [len(value) for value in source_ids]
        source = pad_batch(source_ids, vocab.pad_char_id, pad_to_multiple_of)
        targets = pad_batch(target_ids, vocab.pad_target_id, pad_to_multiple_of)
        boundaries = pad_batch([example.boundary_target for example in chunk], -100, pad_to_multiple_of)
        torch, _ = require_torch()
        yield source, targets, boundaries, torch.tensor(lengths, dtype=torch.long), chunk


def save_vocab(path: Path, vocab: TaggerVocab) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(vocab.to_json(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_vocab(path: Path) -> TaggerVocab:
    return TaggerVocab.from_json(json.loads(path.read_text(encoding="utf-8")))
