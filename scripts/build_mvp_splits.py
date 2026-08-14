#!/usr/bin/env python3
"""Build leakage-safe JDWR v1 splits from the provided CSV data."""

from __future__ import annotations

import argparse
from pathlib import Path

from nextkey.cli import load_config
from nextkey.data.mvp_dataset import export_jdwr_v1_splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Build JDWR v1 JSONL splits.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data/mvp_feasibility.yaml"),
        help="Path to MVP data config.",
    )
    args = parser.parse_args()
    config = load_config(args.config)

    input_config = config.get("input", {})
    columns = config.get("columns", {})
    output = config.get("output", {})
    split_config = config.get("split", {})
    filtering = config.get("filtering", {})

    domains = config.get("domains", {})
    summary = export_jdwr_v1_splits(
        processed_dir=Path(input_config.get("processed_dir", "data/processed")),
        pattern=input_config.get("pattern", "*_dataset.csv"),
        input_column=columns.get("input", "Input_X"),
        target_column=columns.get("target", "Target_Y"),
        split_dir=Path(output.get("split_dir", "data/processed/mvp")),
        train_ratio=float(split_config.get("train_ratio", 0.8)),
        dev_ratio=float(split_config.get("dev_ratio", 0.1)),
        test_ratio=float(split_config.get("test_ratio", 0.1)),
        in_domains=list(domains.get("in_domains", [])),
        external_domain=str(domains.get("external_domain", "the_thao")),
        require_alignment_match=bool(filtering.get("require_alignment_match", True)),
        exclude_html_like_rows=bool(filtering.get("exclude_html_like_rows", True)),
    )
    summary_path = Path(output.get("split_dir", "data/processed/jdwr_v1")) / "manifest.json"
    print(f"Wrote {summary_path}")
    print(summary)


if __name__ == "__main__":
    main()
