"""Utilities for the narrowed MVP dataset based on provided processed CSV files."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import re
import statistics
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


BRACE_PATTERN = re.compile(r"[{}]")
HTML_LIKE_PATTERN = re.compile(r"<[^>]+>|&[a-zA-Z0-9#]+;")
PUNCT_SPACE_PATTERN = re.compile(r"[\W_]+", re.UNICODE)
MVP_TARGET_SEPARATOR_PATTERN = re.compile(r"[^\wÀ-ỹ]+", re.UNICODE)
WHITESPACE_PATTERN = re.compile(r"\s+")


@dataclass(frozen=True)
class MvpSample:
    sample_id: str
    category: str
    input_raw: str
    target_raw: str
    input_normalized: str
    target_normalized: str
    has_brace_marker: bool
    has_html_like_text: bool
    normalized_alignment_match: bool


def remove_brace_markers(text: str) -> str:
    """Remove brace markers while keeping the text inside them."""
    return BRACE_PATTERN.sub("", text)


def strip_accents(text: str) -> str:
    """Remove Vietnamese diacritics while preserving base characters."""
    decomposed = unicodedata.normalize("NFD", text)
    without_marks = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", without_marks).replace("đ", "d").replace("Đ", "D")


def compact_key(text: str) -> str:
    """Normalize text for checking whether input and target contain the same core text."""
    text = html.unescape(text)
    text = remove_brace_markers(text)
    text = strip_accents(text).lower()
    return PUNCT_SPACE_PATTERN.sub("", text)


def normalize_mvp_target(text: str) -> str:
    """Keep only the MVP target: lowercase words with accents and spaces."""
    text = html.unescape(text)
    text = unicodedata.normalize("NFC", text)
    text = MVP_TARGET_SEPARATOR_PATTERN.sub(" ", text)
    text = WHITESPACE_PATTERN.sub(" ", text).strip().lower()
    return text


def has_html_like_text(text: str) -> bool:
    return bool(HTML_LIKE_PATTERN.search(text))


def category_from_path(path: Path) -> str:
    suffix = "_dataset"
    name = path.stem
    return name[: -len(suffix)] if name.endswith(suffix) else name


def iter_mvp_samples(
    processed_dir: Path,
    pattern: str = "*_dataset.csv",
    input_column: str = "Input_X",
    target_column: str = "Target_Y",
) -> Iterable[MvpSample]:
    for csv_path in sorted(processed_dir.glob(pattern)):
        category = category_from_path(csv_path)
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                continue
            missing_columns = {input_column, target_column}.difference(reader.fieldnames)
            if missing_columns:
                raise ValueError(f"{csv_path} missing columns: {sorted(missing_columns)}")

            for row_number, row in enumerate(reader, start=2):
                input_raw = row.get(input_column, "") or ""
                target_raw = row.get(target_column, "") or ""
                input_normalized = remove_brace_markers(input_raw)
                target_normalized = target_raw
                input_key = compact_key(input_normalized)
                target_key = compact_key(target_normalized)
                yield MvpSample(
                    sample_id=f"{category}:{row_number}",
                    category=category,
                    input_raw=input_raw,
                    target_raw=target_raw,
                    input_normalized=input_normalized,
                    target_normalized=target_normalized,
                    has_brace_marker=("{" in input_raw or "}" in input_raw),
                    has_html_like_text=has_html_like_text(input_raw)
                    or has_html_like_text(target_raw),
                    normalized_alignment_match=(bool(input_key) and input_key == target_key),
                )


def percentile(values: list[int], ratio: float) -> int | None:
    if not values:
        return None
    values = sorted(values)
    index = min(len(values) - 1, max(0, round((len(values) - 1) * ratio)))
    return values[index]


def length_stats(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "p50": None, "p90": None, "p95": None, "max": None, "mean": None}
    return {
        "min": min(values),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "max": max(values),
        "mean": round(statistics.fmean(values), 2),
    }


def profile_mvp_dataset(
    processed_dir: Path,
    pattern: str,
    input_column: str,
    target_column: str,
    max_samples: int,
) -> tuple[dict[str, Any], list[MvpSample]]:
    samples: list[MvpSample] = []
    category_counts: Counter[str] = Counter()
    input_lengths: list[int] = []
    target_lengths: list[int] = []
    files = sorted(processed_dir.glob(pattern))

    total_rows = 0
    empty_input = 0
    empty_target = 0
    brace_rows = 0
    html_like_rows = 0
    aligned_rows = 0

    for sample in iter_mvp_samples(processed_dir, pattern, input_column, target_column):
        total_rows += 1
        category_counts[sample.category] += 1
        if not sample.input_raw:
            empty_input += 1
        if not sample.target_raw:
            empty_target += 1
        if sample.has_brace_marker:
            brace_rows += 1
        if sample.has_html_like_text:
            html_like_rows += 1
        if sample.normalized_alignment_match:
            aligned_rows += 1
        input_lengths.append(len(sample.input_normalized))
        target_lengths.append(len(sample.target_normalized))
        if len(samples) < max_samples:
            samples.append(sample)

    valid_rows = total_rows - empty_input - empty_target
    report = {
        "processed_dir": str(processed_dir),
        "pattern": pattern,
        "files": [str(path) for path in files],
        "file_count": len(files),
        "total_rows": total_rows,
        "valid_rows": valid_rows,
        "empty_input_rows": empty_input,
        "empty_target_rows": empty_target,
        "brace_marker_rows": brace_rows,
        "html_like_rows": html_like_rows,
        "normalized_alignment_match_rows": aligned_rows,
        "normalized_alignment_match_rate": round(aligned_rows / total_rows, 4)
        if total_rows
        else 0.0,
        "brace_marker_rate": round(brace_rows / total_rows, 4) if total_rows else 0.0,
        "html_like_rate": round(html_like_rows / total_rows, 4) if total_rows else 0.0,
        "category_counts": dict(sorted(category_counts.items())),
        "input_length": length_stats(input_lengths),
        "target_length": length_stats(target_lengths),
    }
    return report, samples


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, samples: list[MvpSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(asdict(sample), ensure_ascii=False) + "\n")


def split_name(sample_id: str, train_ratio: float, dev_ratio: float) -> str:
    digest = hashlib.sha1(sample_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    if bucket < train_ratio:
        return "train"
    if bucket < train_ratio + dev_ratio:
        return "dev"
    return "test"


def export_mvp_splits(
    processed_dir: Path,
    pattern: str,
    input_column: str,
    target_column: str,
    split_dir: Path,
    train_ratio: float,
    dev_ratio: float,
    require_alignment_match: bool,
    exclude_html_like_rows: bool,
) -> dict[str, Any]:
    split_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": split_dir / "train.jsonl",
        "dev": split_dir / "dev.jsonl",
        "test": split_dir / "test.jsonl",
    }
    handles = {name: path.open("w", encoding="utf-8") for name, path in paths.items()}
    counts: Counter[str] = Counter()
    skipped: Counter[str] = Counter()

    try:
        for sample in iter_mvp_samples(processed_dir, pattern, input_column, target_column):
            if not sample.input_normalized or not sample.target_normalized:
                skipped["empty"] += 1
                continue
            if require_alignment_match and not sample.normalized_alignment_match:
                skipped["alignment_mismatch"] += 1
                continue
            if exclude_html_like_rows and sample.has_html_like_text:
                skipped["html_like"] += 1
                continue

            target = normalize_mvp_target(sample.target_normalized)
            if not target:
                skipped["empty_mvp_target"] += 1
                continue

            split = split_name(sample.sample_id, train_ratio, dev_ratio)
            row = {
                "sample_id": sample.sample_id,
                "category": sample.category,
                "input": sample.input_normalized,
                "target": target,
                "target_original": sample.target_raw,
                "has_brace_marker": sample.has_brace_marker,
            }
            handles[split].write(json.dumps(row, ensure_ascii=False) + "\n")
            counts[split] += 1
            counts["total"] += 1
    finally:
        for handle in handles.values():
            handle.close()

    return {
        "split_dir": str(split_dir),
        "paths": {name: str(path) for name, path in paths.items()},
        "counts": dict(counts),
        "skipped": dict(skipped),
        "policy": {
            "brace_markers": "remove markers and keep content",
            "target": "lowercase accent-and-spacing target; punctuation/capitalization removed",
            "require_alignment_match": require_alignment_match,
            "exclude_html_like_rows": exclude_html_like_rows,
        },
    }


def render_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# MVP Feasibility Report",
        "",
        "## Summary",
        "",
        f"- CSV files: {report['file_count']}",
        f"- Total rows: {report['total_rows']:,}",
        f"- Valid rows: {report['valid_rows']:,}",
        f"- Rows with brace markers: {report['brace_marker_rows']:,} ({report['brace_marker_rate']:.2%})",
        f"- HTML-like rows: {report['html_like_rows']:,} ({report['html_like_rate']:.2%})",
        "- Rows whose normalized target matches normalized input: "
        f"{report['normalized_alignment_match_rows']:,} "
        f"({report['normalized_alignment_match_rate']:.2%})",
        "",
        "## Category Counts",
        "",
        "| Category | Rows |",
        "|---|---:|",
    ]
    for category, count in report["category_counts"].items():
        lines.append(f"| `{category}` | {count:,} |")

    lines.extend(
        [
            "",
            "## Length Stats",
            "",
            "| Field | Min | P50 | P90 | P95 | Max | Mean |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for field in ("input_length", "target_length"):
        stats = report[field]
        lines.append(
            f"| `{field}` | {stats['min']} | {stats['p50']} | {stats['p90']} | "
            f"{stats['p95']} | {stats['max']} | {stats['mean']} |"
        )

    lines.extend(
        [
            "",
            "## Initial Feasibility Interpretation",
            "",
            "- The data volume is sufficient for an MVP feasibility experiment.",
            "- Brace markers can be removed while preserving their content for the first MVP.",
            "- HTML-like rows should be quantified and may need filtering before model training.",
            "- The normalized alignment rate estimates how often the pair cleanly matches the narrowed "
            "task of restoring diacritics and spacing.",
            "",
        ]
    )
    return "\n".join(lines)
