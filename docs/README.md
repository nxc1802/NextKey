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
    research-contract.md
    scope-matrix.md
    acceptance-criteria.md
    use-cases.md
    proposal.md
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
2. [`00-project/research-contract.md`](./00-project/research-contract.md)
3. [`00-project/scope-matrix.md`](./00-project/scope-matrix.md)
4. [`00-project/acceptance-criteria.md`](./00-project/acceptance-criteria.md)
5. [`00-project/use-cases.md`](./00-project/use-cases.md)
6. [`00-project/proposal.md`](./00-project/proposal.md)
7. [`00-project/task-breakdown.md`](./00-project/task-breakdown.md)
8. [`00-project/roadmap.md`](./00-project/roadmap.md)
9. [`01-data/dataset-build-plan.md`](./01-data/dataset-build-plan.md)
10. [`03-evaluation/evaluation-plan.md`](./03-evaluation/evaluation-plan.md)
11. [`02-model/model-selection.md`](./02-model/model-selection.md)
12. [`04-system/system-architecture.md`](./04-system/system-architecture.md)
13. [`04-system/api-and-demo.md`](./04-system/api-and-demo.md)
14. [`05-edge/edge-plan.md`](./05-edge/edge-plan.md)
15. [`06-personalization-privacy/personalization-and-privacy.md`](./06-personalization-privacy/personalization-and-privacy.md)

## Documentation Rules

- Keep `deep-research-report.md` as the research source and evidence trail.
- Keep implementation docs in `docs/`.
- Do not finalize model choice before the model-selection gate.
- Update evaluation docs whenever a new metric, baseline, split, or ablation is
  added.
- Avoid duplicating large content from the research report; link back to it and
  summarize only the operational decision.
