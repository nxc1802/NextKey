#!/usr/bin/env python3
"""Train the MVP lexicon dynamic-programming baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nextkey.cli import load_config
from nextkey.models.mvp_lexicon_baseline import save_model, train_lexicon_baseline


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MVP lexicon baseline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/model/mvp_lexicon_baseline.yaml"),
        help="Path to baseline config.",
    )
    args = parser.parse_args()
    config = load_config(args.config)

    data_config = config.get("data", {})
    model_config = config.get("model", {})
    train_path = Path(data_config.get("train_path", "data/processed/mvp/train.jsonl"))
    output_path = Path(model_config.get("output_path", "models/checkpoints/mvp-lexicon-baseline.json"))
    min_count = int(model_config.get("min_count", 1))

    model, stats = train_lexicon_baseline(train_path, min_count=min_count)
    save_model(output_path, model, stats)
    print(json.dumps({"model_path": str(output_path), "training_stats": stats}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
