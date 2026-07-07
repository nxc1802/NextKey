#!/usr/bin/env python3
"""Profile provided processed CSV files for the narrowed MVP."""

from __future__ import annotations

import argparse
from pathlib import Path

from nextkey.cli import load_config
from nextkey.data.mvp_dataset import (
    profile_mvp_dataset,
    render_markdown_report,
    write_json,
    write_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile MVP feasibility data.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data/mvp_feasibility.yaml"),
        help="Path to MVP feasibility YAML config.",
    )
    args = parser.parse_args()
    config = load_config(args.config)

    input_config = config.get("input", {})
    columns = config.get("columns", {})
    output = config.get("output", {})
    sampling = config.get("sampling", {})

    report, samples = profile_mvp_dataset(
        processed_dir=Path(input_config.get("processed_dir", "data/processed")),
        pattern=input_config.get("pattern", "*_dataset.csv"),
        input_column=columns.get("input", "Input_X"),
        target_column=columns.get("target", "Target_Y"),
        max_samples=int(sampling.get("max_samples", 50)),
    )

    report_json = Path(output.get("report_json", "experiments/reports/mvp-feasibility.json"))
    report_md = Path(output.get("report_md", "experiments/reports/mvp-feasibility.md"))
    sample_jsonl = Path(output.get("sample_jsonl", "data/samples/mvp-feasibility-sample.jsonl"))

    write_json(report_json, report)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text(render_markdown_report(report), encoding="utf-8")
    write_jsonl(sample_jsonl, samples)

    print(f"Wrote {report_json}")
    print(f"Wrote {report_md}")
    print(f"Wrote {sample_jsonl}")


if __name__ == "__main__":
    main()
