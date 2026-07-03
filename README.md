# NextKey

NextKey is a research and prototype project for Vietnamese compact writing
restoration: turning noisy, shortened, accentless, typo-prone Vietnamese input
into a complete standard sentence.

The original deep research report is kept unchanged at
[`deep-research-report.md`](./deep-research-report.md). The `docs/` folder turns
that report into an implementation-oriented documentation system.

## Current Status

This repository is currently in the planning and documentation stage.

There is no source code, dataset, model checkpoint, or demo implementation yet.
The next engineering step is to scaffold the project structure described in
[`docs/07-operations/repo-structure.md`](./docs/07-operations/repo-structure.md).

## Documentation Entry Points

- [`docs/README.md`](./docs/README.md): documentation index and reading order
- [`docs/00-project/overview.md`](./docs/00-project/overview.md): project scope
- [`docs/00-project/task-breakdown.md`](./docs/00-project/task-breakdown.md): task and phase breakdown
- [`docs/01-data/dataset-build-plan.md`](./docs/01-data/dataset-build-plan.md): dataset build plan
- [`docs/02-model/model-selection.md`](./docs/02-model/model-selection.md): open model-selection gate
- [`docs/03-evaluation/evaluation-plan.md`](./docs/03-evaluation/evaluation-plan.md): metrics, baselines, and ablations
- [`docs/04-system/system-architecture.md`](./docs/04-system/system-architecture.md): target system architecture

## Core Principle

Model choice is intentionally not finalized. The project must first build the
data contract and evaluation harness, then choose models through comparable
experiments.
