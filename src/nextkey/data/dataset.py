"""Dataset loading and batching for the CharTagger research pipeline.

Loads JSONL data produced by the JDWR v1 split pipeline and yields
padded batches suitable for CharTagger training and evaluation.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import torch

from nextkey.data.preprocessor import compact_key, strip_accents
from nextkey.data.tokenizer import CharVocab


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AlignedExample:
    """A single aligned training/evaluation example."""
    source: str              # compact input (no accents, no spaces)
    char_target: str         # accented characters (no spaces)
    boundary_target: list[int]  # [0, 0, 1, 0, ...] space-before flags
    domain: str


# ---------------------------------------------------------------------------
# JSONL loading
# ---------------------------------------------------------------------------

def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _resolve_paths(path_or_paths: Path | str | Iterable[Path | str]) -> list[Path]:
    if isinstance(path_or_paths, (str, Path)):
        path = Path(path_or_paths)
        return sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    return [Path(p) for p in path_or_paths]


def load_examples(
    path_or_paths: Path | str | Iterable[Path | str],
    max_samples: int | None = None,
    max_len: int = 256,
) -> list[AlignedExample]:
    """Load aligned examples from JSONL file(s) or directory.

    If max_samples is provided, each domain file gets a proportional quota
    to ensure balanced sampling. If max_samples is None, all valid examples are loaded.
    """
    paths = _resolve_paths(path_or_paths)
    if not paths:
        return []
    if max_samples is not None and max_samples <= 0:
        return []

    # Fair quota per file when max_samples is bounded
    quota = -(-max_samples // len(paths)) if max_samples is not None else None
    examples: list[AlignedExample] = []

    for path in paths:
        file_examples: list[AlignedExample] = []
        for row in iter_jsonl(path):
            source = row.get("input", "")
            char_target = row.get("char_target")
            boundary_target = row.get("boundary_target")

            # Fall back to re-computing alignment if pre-computed fields are missing
            if not isinstance(char_target, str) or not isinstance(boundary_target, list):
                continue

            src_key = compact_key(source)
            if not src_key or src_key != strip_accents(char_target):
                continue

            bt = [int(v) for v in boundary_target]
            if 0 < len(src_key) <= max_len and len(src_key) == len(bt):
                domain = str(row.get("domain", row.get("category", "unknown")))
                file_examples.append(AlignedExample(src_key, char_target, bt, domain))
                if quota is not None and len(file_examples) >= quota:
                    break

        examples.extend(file_examples)

    if max_samples is not None:
        return examples[:max_samples]
    return examples


# ---------------------------------------------------------------------------
# Batching utilities
# ---------------------------------------------------------------------------

def pad_sequences(sequences: list[list[int]], pad_id: int,
                  pad_to_multiple_of: int = 1) -> torch.Tensor:
    """Pad a list of integer sequences into a 2-D LongTensor."""
    max_len = max(map(len, sequences))
    if pad_to_multiple_of > 1:
        max_len = -(-max_len // pad_to_multiple_of) * pad_to_multiple_of
    batch = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
    for i, seq in enumerate(sequences):
        batch[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
    return batch


def _order_examples(examples: list[AlignedExample],
                    domain_balanced: bool) -> list[AlignedExample]:
    """Optionally interleave examples across domains."""
    if not domain_balanced:
        result = list(examples)
        random.shuffle(result)
        return result

    by_domain: dict[str, list[AlignedExample]] = defaultdict(list)
    for ex in examples:
        by_domain[ex.domain].append(ex)
    for vals in by_domain.values():
        random.shuffle(vals)

    domains = sorted(by_domain)
    result: list[AlignedExample] = []
    positions: dict[str, int] = defaultdict(int)
    while True:
        added = False
        for domain in domains:
            pos = positions[domain]
            if pos < len(by_domain[domain]):
                result.append(by_domain[domain][pos])
                positions[domain] += 1
                added = True
        if not added:
            return result


BatchType = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[AlignedExample]]


def iter_batches(
    examples: list[AlignedExample],
    vocab: CharVocab,
    batch_size: int,
    domain_balanced: bool = False,
    length_bucket_size: int = 0,
    pad_to_multiple_of: int = 1,
) -> Iterator[BatchType]:
    """Yield (source, targets, boundaries, lengths, chunk) batches.

    Returns:
        source: [B, T] LongTensor of input character IDs
        targets: [B, T] LongTensor of target character IDs
        boundaries: [B, T] LongTensor of boundary flags (padded with -100)
        lengths: [B] LongTensor of original sequence lengths
        chunk: list of AlignedExample for this batch
    """
    ordered = _order_examples(examples, domain_balanced)

    if length_bucket_size and length_bucket_size >= batch_size:
        ordered = [
            ex for start in range(0, len(ordered), length_bucket_size)
            for ex in sorted(ordered[start:start + length_bucket_size],
                             key=lambda x: len(x.source))
        ]

    for start in range(0, len(ordered), batch_size):
        chunk = ordered[start:start + batch_size]
        if not chunk:
            continue

        source_ids = [vocab.encode_input(ex.source) for ex in chunk]
        target_ids = [vocab.encode_target(ex.char_target) for ex in chunk]
        lengths = [len(s) for s in source_ids]

        source = pad_sequences(source_ids, vocab.pad_char_id, pad_to_multiple_of)
        targets = pad_sequences(target_ids, vocab.pad_target_id, pad_to_multiple_of)
        boundaries = pad_sequences(
            [ex.boundary_target for ex in chunk], -100, pad_to_multiple_of
        )

        yield source, targets, boundaries, torch.tensor(lengths, dtype=torch.long), chunk
