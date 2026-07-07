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
    mvp-feasibility-scope.md
    proposal.md
    task-breakdown.md
    roadmap.md
  01-data/
    dataset-build-plan.md
    mvp-feasibility-report.md
    schema.md
    noise-taxonomy.md
    annotation-guideline.md
  02-model/
    model-selection.md
    training-plan.md
  03-evaluation/
    evaluation-plan.md
    mvp-lexicon-baseline-report.md
    mvp-chartagger-baseline-report.md
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
6. [`00-project/mvp-feasibility-scope.md`](./00-project/mvp-feasibility-scope.md)
7. [`00-project/proposal.md`](./00-project/proposal.md)
8. [`00-project/task-breakdown.md`](./00-project/task-breakdown.md)
9. [`00-project/roadmap.md`](./00-project/roadmap.md)
10. [`01-data/dataset-build-plan.md`](./01-data/dataset-build-plan.md)
11. [`01-data/mvp-feasibility-report.md`](./01-data/mvp-feasibility-report.md)
12. [`03-evaluation/evaluation-plan.md`](./03-evaluation/evaluation-plan.md)
13. [`03-evaluation/mvp-lexicon-baseline-report.md`](./03-evaluation/mvp-lexicon-baseline-report.md)
14. [`03-evaluation/mvp-chartagger-baseline-report.md`](./03-evaluation/mvp-chartagger-baseline-report.md)
15. [`02-model/model-selection.md`](./02-model/model-selection.md)
16. [`04-system/system-architecture.md`](./04-system/system-architecture.md)
17. [`04-system/api-and-demo.md`](./04-system/api-and-demo.md)
18. [`05-edge/edge-plan.md`](./05-edge/edge-plan.md)
19. [`06-personalization-privacy/personalization-and-privacy.md`](./06-personalization-privacy/personalization-and-privacy.md)

## Documentation Rules

- Keep `deep-research-report.md` as the research source and evidence trail.
- Keep implementation docs in `docs/`.
- Do not finalize model choice before the model-selection gate.
- Update evaluation docs whenever a new metric, baseline, split, or ablation is
  added.
- Avoid duplicating large content from the research report; link back to it and
  summarize only the operational decision.
