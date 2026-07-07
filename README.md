# NextKey

NextKey is a research and prototype project for Vietnamese compact writing
restoration: turning noisy, shortened, accentless, typo-prone Vietnamese input
into a complete standard sentence.

The original deep research report is kept unchanged at
[`deep-research-report.md`](./deep-research-report.md). The `docs/` folder turns
that report into an implementation-oriented documentation system.

## Current Status

This repository is currently in the planning and documentation stage.

The repository now has the Task 0 foundation scaffold: Python package layout,
configs, command wrappers, artifact folders, and smoke tests. Dataset builders,
model training, evaluation metrics, and demo endpoints are still placeholders
that will be implemented in later tasks.

## Setup

Use Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e ".[dev]"
```

## Smoke Commands

These commands verify that the project scaffold is wired correctly.

```bash
python3 scripts/build_dataset.py --config configs/data/synthetic_v1.yaml
PYTHONPATH=src python3 scripts/profile_mvp_data.py --config configs/data/mvp_feasibility.yaml
PYTHONPATH=src python3 scripts/build_mvp_splits.py --config configs/data/mvp_feasibility.yaml
PYTHONPATH=src python3 scripts/train_mvp_lexicon_baseline.py --config configs/model/mvp_lexicon_baseline.yaml
PYTHONPATH=src python3 scripts/evaluate_mvp_lexicon_baseline.py --config configs/model/mvp_lexicon_baseline.yaml --split both
PYTHONPATH=src .venv/bin/python scripts/train_mvp_chartagger.py --config configs/model/mvp_chartagger_v1.yaml
PYTHONPATH=src .venv/bin/python scripts/evaluate_mvp_chartagger.py --config configs/model/mvp_chartagger_v1.yaml
python3 scripts/train.py --config configs/model/teacher_candidate.yaml
python3 scripts/evaluate.py --config configs/eval/dev_human.yaml
python3 scripts/export_model.py --checkpoint models/checkpoints/student
python3 scripts/run_demo.py
python3 scripts/check_scaffold.py
pytest
```

The commands currently emit structured scaffold status. They do not build real
datasets, train models, or launch a real demo yet.

`pytest` is part of the optional dev dependencies. `scripts/check_scaffold.py`
uses only the Python standard library and can run before dev dependencies are
installed.

## Documentation Entry Points

- [`docs/README.md`](./docs/README.md): documentation index and reading order
- [`docs/00-project/overview.md`](./docs/00-project/overview.md): project scope
- [`docs/00-project/research-contract.md`](./docs/00-project/research-contract.md): locked problem statement and research questions
- [`docs/00-project/mvp-feasibility-scope.md`](./docs/00-project/mvp-feasibility-scope.md): narrowed MVP scope for provided processed data
- [`docs/00-project/proposal.md`](./docs/00-project/proposal.md): short project proposal
- [`docs/00-project/task-breakdown.md`](./docs/00-project/task-breakdown.md): task and phase breakdown
- [`docs/01-data/dataset-build-plan.md`](./docs/01-data/dataset-build-plan.md): dataset build plan
- [`docs/01-data/mvp-feasibility-report.md`](./docs/01-data/mvp-feasibility-report.md): feasibility result for provided processed data
- [`docs/02-model/model-selection.md`](./docs/02-model/model-selection.md): open model-selection gate
- [`docs/03-evaluation/evaluation-plan.md`](./docs/03-evaluation/evaluation-plan.md): metrics, baselines, and ablations
- [`docs/03-evaluation/mvp-lexicon-baseline-report.md`](./docs/03-evaluation/mvp-lexicon-baseline-report.md): first MVP baseline result
- [`docs/03-evaluation/mvp-chartagger-baseline-report.md`](./docs/03-evaluation/mvp-chartagger-baseline-report.md): first context-aware neural baseline result
- [`docs/04-system/system-architecture.md`](./docs/04-system/system-architecture.md): target system architecture

## Core Principle

Model choice is intentionally not finalized. The project must first build the
data contract and evaluation harness, then choose models through comparable
experiments.
