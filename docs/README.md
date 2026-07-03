# Documentation Index

This folder is the operational documentation system for NextKey. It is derived
from the research direction in [`../deep-research-report.md`](../deep-research-report.md),
but it is organized for execution: tasks, phases, data contracts, experiments,
demo requirements, and risk management.

## Folder Structure

```text
docs/
  00-project/
    overview.md
    task-breakdown.md
    roadmap.md
  01-data/
    dataset-build-plan.md
    schema.md
    noise-taxonomy.md
    annotation-guideline.md
  02-model/
    model-selection.md
    training-plan.md
  03-evaluation/
    evaluation-plan.md
  04-system/
    system-architecture.md
    api-and-demo.md
  05-edge/
    edge-plan.md
  06-personalization-privacy/
    personalization-and-privacy.md
  07-operations/
    repo-structure.md
    risk-register.md
```

## Recommended Reading Order

1. [`00-project/overview.md`](./00-project/overview.md)
2. [`00-project/task-breakdown.md`](./00-project/task-breakdown.md)
3. [`00-project/roadmap.md`](./00-project/roadmap.md)
4. [`01-data/dataset-build-plan.md`](./01-data/dataset-build-plan.md)
5. [`03-evaluation/evaluation-plan.md`](./03-evaluation/evaluation-plan.md)
6. [`02-model/model-selection.md`](./02-model/model-selection.md)
7. [`04-system/system-architecture.md`](./04-system/system-architecture.md)
8. [`04-system/api-and-demo.md`](./04-system/api-and-demo.md)
9. [`05-edge/edge-plan.md`](./05-edge/edge-plan.md)
10. [`06-personalization-privacy/personalization-and-privacy.md`](./06-personalization-privacy/personalization-and-privacy.md)

## Documentation Rules

- Keep `deep-research-report.md` as the research source and evidence trail.
- Keep implementation docs in `docs/`.
- Do not finalize model choice before the model-selection gate.
- Update evaluation docs whenever a new metric, baseline, split, or ablation is
  added.
- Avoid duplicating large content from the research report; link back to it and
  summarize only the operational decision.
