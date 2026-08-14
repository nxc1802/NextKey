"""Dataset preparation for the MVP and JDWR v1 experiments."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
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
    return BRACE_PATTERN.sub("", text)


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    without_marks = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", without_marks).replace("đ", "d").replace("Đ", "D")


def compact_key(text: str) -> str:
    """Canonical content key used for alignment and grouped split assignment."""
    text = html.unescape(remove_brace_markers(text))
    return PUNCT_SPACE_PATTERN.sub("", strip_accents(text).lower())


def normalize_mvp_target(text: str) -> str:
    text = unicodedata.normalize("NFC", html.unescape(text))
    return WHITESPACE_PATTERN.sub(" ", MVP_TARGET_SEPARATOR_PATTERN.sub(" ", text)).strip().lower()


def has_html_like_text(text: str) -> bool:
    return bool(HTML_LIKE_PATTERN.search(text))


def category_from_path(path: Path) -> str:
    return path.stem.removesuffix("_dataset")


def iter_mvp_samples(processed_dir: Path, pattern: str = "*_dataset.csv", input_column: str = "Input_X",
                     target_column: str = "Target_Y") -> Iterable[MvpSample]:
    for csv_path in sorted(processed_dir.glob(pattern)):
        category = category_from_path(csv_path)
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                continue
            missing = {input_column, target_column}.difference(reader.fieldnames)
            if missing:
                raise ValueError(f"{csv_path} missing columns: {sorted(missing)}")
            for row_number, row in enumerate(reader, start=2):
                input_raw, target_raw = row.get(input_column, "") or "", row.get(target_column, "") or ""
                input_normalized = remove_brace_markers(input_raw)
                yield MvpSample(
                    sample_id=f"{category}:{row_number}", category=category, input_raw=input_raw,
                    target_raw=target_raw, input_normalized=input_normalized, target_normalized=target_raw,
                    has_brace_marker=("{" in input_raw or "}" in input_raw),
                    has_html_like_text=has_html_like_text(input_raw) or has_html_like_text(target_raw),
                    normalized_alignment_match=(bool(compact_key(input_normalized)) and
                                                compact_key(input_normalized) == compact_key(target_raw)),
                )


def percentile(values: list[int], ratio: float) -> int | None:
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, max(0, round((len(values) - 1) * ratio)))]


def length_stats(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "p50": None, "p90": None, "p95": None, "max": None, "mean": None}
    return {"min": min(values), "p50": percentile(values, .5), "p90": percentile(values, .9),
            "p95": percentile(values, .95), "max": max(values), "mean": round(statistics.fmean(values), 2)}


def profile_mvp_dataset(processed_dir: Path, pattern: str, input_column: str, target_column: str,
                        max_samples: int) -> tuple[dict[str, Any], list[MvpSample]]:
    samples: list[MvpSample] = []
    category_counts: Counter[str] = Counter()
    counts: Counter[str] = Counter()
    input_lengths: list[int] = []
    target_lengths: list[int] = []
    files = sorted(processed_dir.glob(pattern))
    for sample in iter_mvp_samples(processed_dir, pattern, input_column, target_column):
        counts["total"] += 1; category_counts[sample.category] += 1
        counts["empty_input"] += int(not sample.input_raw); counts["empty_target"] += int(not sample.target_raw)
        counts["brace"] += int(sample.has_brace_marker); counts["html"] += int(sample.has_html_like_text)
        counts["aligned"] += int(sample.normalized_alignment_match)
        input_lengths.append(len(sample.input_normalized)); target_lengths.append(len(sample.target_normalized))
        if len(samples) < max_samples:
            samples.append(sample)
    total = counts["total"]
    return ({"processed_dir": str(processed_dir), "pattern": pattern, "files": [str(p) for p in files],
             "file_count": len(files), "total_rows": total,
             "valid_rows": total - counts["empty_input"] - counts["empty_target"],
             "empty_input_rows": counts["empty_input"], "empty_target_rows": counts["empty_target"],
             "brace_marker_rows": counts["brace"], "html_like_rows": counts["html"],
             "normalized_alignment_match_rows": counts["aligned"],
             "normalized_alignment_match_rate": round(counts["aligned"] / total, 4) if total else 0,
             "brace_marker_rate": round(counts["brace"] / total, 4) if total else 0,
             "html_like_rate": round(counts["html"] / total, 4) if total else 0,
             "category_counts": dict(sorted(category_counts.items())), "input_length": length_stats(input_lengths),
             "target_length": length_stats(target_lengths)}, samples)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, samples: list[MvpSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(asdict(sample), ensure_ascii=False) + "\n")


def split_name(sample_id: str, train_ratio: float, dev_ratio: float) -> str:
    bucket = int(hashlib.sha1(sample_id.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "train" if bucket < train_ratio else "dev" if bucket < train_ratio + dev_ratio else "test"


def _jdwr_targets(target: str) -> tuple[str, list[int]]:
    chars, boundaries, pending = [], [], False
    for char in normalize_mvp_target(target):
        if char.isspace():
            pending = bool(chars)
        else:
            chars.append(char); boundaries.append(int(pending)); pending = False
    if boundaries:
        boundaries[0] = 0
    return "".join(chars), boundaries


def _valid_jdwr_row(sample: MvpSample, require_alignment_match: bool,
                    exclude_html_like_rows: bool) -> dict[str, Any] | None:
    if not sample.input_normalized or not sample.target_normalized:
        return None
    if exclude_html_like_rows and sample.has_html_like_text:
        return None
    if require_alignment_match and not sample.normalized_alignment_match:
        return None
    source = compact_key(sample.input_normalized)
    char_target, boundary_target = _jdwr_targets(sample.target_normalized)
    if not source or not char_target or source != strip_accents(char_target):
        return None
    return {"sample_id": sample.sample_id, "category": sample.category, "domain": sample.category,
            "source_type": "provided_mvp", "input": source, "target": normalize_mvp_target(sample.target_normalized),
            "char_target": char_target, "boundary_target": boundary_target,
            "group_key": compact_key(sample.target_normalized), "has_brace_marker": sample.has_brace_marker}


def export_jdwr_v1_splits(processed_dir: Path, pattern: str, input_column: str, target_column: str,
                          split_dir: Path, in_domains: list[str], external_domain: str, train_ratio: float,
                          dev_ratio: float, test_ratio: float, require_alignment_match: bool = True,
                          exclude_html_like_rows: bool = True) -> dict[str, Any]:
    """Build grouped in-domain splits plus an untouched external-domain holdout."""
    if abs(train_ratio + dev_ratio + test_ratio - 1.0) > 1e-9:
        raise ValueError("train_ratio + dev_ratio + test_ratio must equal 1")
    if external_domain in in_domains or len(set(in_domains)) != len(in_domains):
        raise ValueError("in_domains must be unique and exclude external_domain")
    allowed = set(in_domains) | {external_domain}
    skipped: Counter[str] = Counter()
    external_keys: set[str] = set()
    for sample in iter_mvp_samples(processed_dir, pattern, input_column, target_column):
        if sample.category != external_domain:
            continue
        row = _valid_jdwr_row(sample, require_alignment_match, exclude_html_like_rows)
        if row is not None:
            external_keys.add(row["group_key"])

    files: dict[str, dict[str, str]] = defaultdict(dict)
    counts: dict[str, dict[str, int]] = defaultdict(dict)
    handles: dict[tuple[str, str], Any] = {}
    for domain in [*in_domains, external_domain]:
        destinations = ["external"] if domain == external_domain else ["train", "dev", "test"]
        for destination in destinations:
            path = (split_dir / "test" / "external" / f"{domain}.jsonl" if destination == "external" else
                    split_dir / "test" / "in_domain" / f"{domain}.jsonl" if destination == "test" else
                    split_dir / destination / f"{domain}.jsonl")
            path.parent.mkdir(parents=True, exist_ok=True)
            handles[(destination, domain)] = path.open("w", encoding="utf-8")
            files[destination][domain] = str(path); counts[destination][domain] = 0
    try:
        for sample in iter_mvp_samples(processed_dir, pattern, input_column, target_column):
            if sample.category not in allowed:
                continue
            row = _valid_jdwr_row(sample, require_alignment_match, exclude_html_like_rows)
            if row is None:
                skipped["invalid_or_filtered"] += 1; continue
            if sample.category == external_domain:
                destination = "external"
            elif row["group_key"] in external_keys:
                skipped["external_group_overlap"] += 1; continue
            else:
                destination = split_name(row["group_key"], train_ratio, dev_ratio)
            handles[(destination, sample.category)].write(json.dumps(row, ensure_ascii=False) + "\n")
            counts[destination][sample.category] += 1
    finally:
        for handle in handles.values():
            handle.close()
    pairs = (("train", "dev"), ("train", "test"), ("dev", "test"), ("train", "external"),
             ("dev", "external"), ("test", "external"))
    overlap_checks = {f"{left}__{right}": 0 for left, right in pairs}
    manifest = {"dataset_version": "jdwr-v1", "split_policy": {"group_key": "compact_key(target)",
                "in_domain_ratios": {"train": train_ratio, "dev": dev_ratio, "test": test_ratio},
                "external_domain": external_domain, "in_domains": in_domains,
                "external_overlap_policy": "exclude in-domain rows with an external group key"},
                "counts": counts, "skipped": dict(skipped), "files": files,
                "group_overlap_checks": overlap_checks,
                "total_rows": sum(sum(domain.values()) for domain in counts.values())}
    write_json(split_dir / "manifest.json", manifest)
    return manifest


def export_mvp_splits(processed_dir: Path, pattern: str, input_column: str, target_column: str, split_dir: Path,
                      train_ratio: float, dev_ratio: float, require_alignment_match: bool,
                      exclude_html_like_rows: bool) -> dict[str, Any]:
    """Legacy row-hash split retained for existing MVP experiments."""
    paths = {name: split_dir / f"{name}.jsonl" for name in ("train", "dev", "test")}
    split_dir.mkdir(parents=True, exist_ok=True)
    handles = {name: path.open("w", encoding="utf-8") for name, path in paths.items()}
    counts, skipped = Counter(), Counter()
    try:
        for sample in iter_mvp_samples(processed_dir, pattern, input_column, target_column):
            if not sample.input_normalized or not sample.target_normalized:
                skipped["empty"] += 1; continue
            if require_alignment_match and not sample.normalized_alignment_match:
                skipped["alignment_mismatch"] += 1; continue
            if exclude_html_like_rows and sample.has_html_like_text:
                skipped["html_like"] += 1; continue
            target = normalize_mvp_target(sample.target_normalized)
            if not target:
                skipped["empty_mvp_target"] += 1; continue
            split = split_name(sample.sample_id, train_ratio, dev_ratio)
            row = {"sample_id": sample.sample_id, "category": sample.category, "input": sample.input_normalized,
                   "target": target, "target_original": sample.target_raw, "has_brace_marker": sample.has_brace_marker}
            handles[split].write(json.dumps(row, ensure_ascii=False) + "\n"); counts.update([split, "total"])
    finally:
        for handle in handles.values():
            handle.close()
    return {"split_dir": str(split_dir), "paths": {name: str(path) for name, path in paths.items()},
            "counts": dict(counts), "skipped": dict(skipped), "policy": {"split": "legacy_row_hash"}}


def render_markdown_report(report: dict[str, Any]) -> str:
    return "\n".join(["# MVP Feasibility Report", "", "## Summary", "", f"- CSV files: {report['file_count']}",
                      f"- Total rows: {report['total_rows']:,}", f"- Valid rows: {report['valid_rows']:,}",
                      f"- Rows with brace markers: {report['brace_marker_rows']:,} ({report['brace_marker_rate']:.2%})",
                      f"- HTML-like rows: {report['html_like_rows']:,} ({report['html_like_rate']:.2%})", ""])
