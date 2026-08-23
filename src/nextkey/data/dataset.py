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


@dataclass
class TriTaskAlignedExample:
    """A single aligned training/evaluation example for 3-task restoration."""
    source: str              # noisy input (typos, no accents, no spaces)
    base_target: str         # base unaccented corrected characters
    diacritic_target: str    # accented target characters
    boundary_target: list[int]  # space-before flags
    domain: str = "general"
    noise_tags: list[str] = None


# ---------------------------------------------------------------------------
# JSONL loading
# ---------------------------------------------------------------------------

def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def find_dataset_root() -> Path | None:
    """Dynamically search for jdwr_v1 dataset directory locally or in Kaggle inputs."""
    # 1. Local path
    local_path = Path("data/processed/jdwr_v1")
    if local_path.is_dir() and ((local_path / "manifest.json").exists() or any(local_path.glob("*/*.jsonl"))):
        return local_path

    # 2. Kaggle environment search
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        # Specific known user dataset paths
        for specific in [
            Path("/kaggle/input/datasets/cuongnguyen1802/nextkey-dataset/processed/jdwr_v1"),
            Path("/kaggle/input/datasets/cuongnguyen1802/nextkey-dataset/jdwr_v1"),
            Path("/kaggle/input/nextkey-dataset/processed/jdwr_v1"),
            Path("/kaggle/input/nextkey-dataset/jdwr_v1"),
        ]:
            if specific.is_dir():
                _try_symlink_local(specific)
                return specific

        # Search for manifest.json under jdwr_v1
        manifests = sorted(kaggle_input.glob("**/jdwr_v1/manifest.json"))
        if manifests:
            target = manifests[0].parent
            _try_symlink_local(target)
            return target

        # Search for train directory containing jsonl
        train_dirs = sorted(kaggle_input.glob("**/jdwr_v1/train"))
        if train_dirs:
            target = train_dirs[0].parent
            _try_symlink_local(target)
            return target

        # Generic search for any jdwr_v1 directory
        jdwr_dirs = sorted(kaggle_input.glob("**/jdwr_v1"))
        if jdwr_dirs:
            target = jdwr_dirs[0]
            _try_symlink_local(target)
            return target

    return None


def _try_symlink_local(source: Path) -> None:
    """Attempt to create a local symlink data/processed/jdwr_v1 -> source."""
    try:
        local_target = Path("data/processed/jdwr_v1")
        if not local_target.exists() and not local_target.is_symlink():
            local_target.parent.mkdir(parents=True, exist_ok=True)
            local_target.symlink_to(source.resolve(), target_is_directory=True)
    except OSError:
        pass


def _resolve_paths(path_or_paths: Path | str | Iterable[Path | str]) -> list[Path]:
    if isinstance(path_or_paths, (list, tuple, set)):
        paths: list[Path] = []
        for p in path_or_paths:
            paths.extend(_resolve_paths(p))
        return paths

    raw_path = Path(path_or_paths)

    # 1. Directly exists as dir
    if raw_path.is_dir():
        files = sorted(raw_path.glob("*.jsonl"))
        if files:
            return files
        return [raw_path]

    # 2. Directly exists as file
    if raw_path.is_file():
        return [raw_path]

    # 3. Dynamic search across local and Kaggle environments
    root = find_dataset_root()
    if root is not None:
        path_str = str(raw_path)
        # Check standard subpaths
        for subkey in ["test/in_domain", "test/external", "train", "dev", "test"]:
            if subkey in path_str:
                resolved = root / subkey
                if resolved.is_dir():
                    files = sorted(resolved.glob("*.jsonl"))
                    if files:
                        return files
                elif resolved.is_file():
                    return [resolved]

        # Check by name
        resolved = root / raw_path.name
        if resolved.is_dir():
            files = sorted(resolved.glob("*.jsonl"))
            if files:
                return files
        elif resolved.is_file():
            return [resolved]

    # 4. If path doesn't exist, raise informative FileNotFoundError
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Could not find data path: '{path_or_paths}'.\n"
            f"Checked local 'data/processed/jdwr_v1' and '/kaggle/input/**/jdwr_v1'.\n"
            "Please ensure the dataset is attached in Kaggle (e.g. 'nextkey-dataset')."
        )

    return [raw_path]


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


TriBatchType = tuple[
    torch.Tensor,  # source [B, T]
    torch.Tensor,  # corr_targets [B, T]
    torch.Tensor,  # diac_targets [B, T]
    torch.Tensor,  # boundaries [B, T]
    torch.Tensor,  # lengths [B]
    list[Any],     # chunk
]


def iter_tri_batches(
    examples: list[Any],
    vocab: CharVocab,
    batch_size: int,
    domain_balanced: bool = False,
    length_bucket_size: int = 0,
    pad_to_multiple_of: int = 1,
) -> Iterator[TriBatchType]:
    """Yield (source, corr_targets, diac_targets, boundaries, lengths, chunk) batches.

    Returns:
        source: [B, T] LongTensor of noisy input character IDs
        corr_targets: [B, T] LongTensor of base unaccented target character IDs
        diac_targets: [B, T] LongTensor of accented diacritic target character IDs
        boundaries: [B, T] LongTensor of boundary flags (padded with -100)
        lengths: [B] LongTensor of original sequence lengths
        chunk: list of examples in this batch
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
        corr_ids = [vocab.encode_corr(ex.base_target) for ex in chunk]
        diac_ids = [vocab.encode_target(ex.diacritic_target) for ex in chunk]
        lengths = [len(s) for s in source_ids]

        source = pad_sequences(source_ids, vocab.pad_char_id, pad_to_multiple_of)
        corr_targets = pad_sequences(corr_ids, vocab.pad_corr_id, pad_to_multiple_of)
        diac_targets = pad_sequences(diac_ids, vocab.pad_target_id, pad_to_multiple_of)
        boundaries = pad_sequences(
            [ex.boundary_target for ex in chunk], -100, pad_to_multiple_of
        )

        yield source, corr_targets, diac_targets, boundaries, torch.tensor(lengths, dtype=torch.long), chunk
