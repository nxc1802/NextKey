# Acceptance Criteria

This document defines what it means for the project to be complete at different
quality levels.

## MVP Acceptance

The MVP is accepted when all of the following are true:

- The project is described as compact Vietnamese input restoration.
- Dataset schema is documented and implemented.
- Noise taxonomy is documented and represented in generated samples.
- Synthetic dataset generation works for at least a small reproducible sample.
- Human-refined annotation guideline exists.
- At least one baseline can run through the evaluation harness.
- Model-selection remains open until comparable results exist.
- Restore command/API returns structured output.
- Generated artifacts are excluded from git.
- README setup and smoke commands work.

## Final Defense Acceptance

The final defense package is accepted when:

- at least two baselines are reported
- selected model candidate is justified by metrics
- final evaluation includes `dev_human` or `test_human`
- results include NLP quality, typo behavior, and top-k metrics
- limitations and failure cases are documented
- demo can run in a repeatable mode
- thesis/slides explain why the project is broader than diacritic restoration

## Strong Project Acceptance

The project is strong when:

- keyboard-aware noise ablation shows measurable impact
- top-k candidate workflow improves user-facing metrics
- edge student has measured size, RAM, and latency
- edge quality loss versus teacher is quantified
- at least one personalization layer has measured gain
- privacy controls are visible and documented

## Rejection Criteria

The project should not be considered complete if:

- the only task solved is adding Vietnamese diacritics
- final model choice is based on preference instead of benchmark evidence
- test data leaks into training or model selection
- evaluation reports only one global metric
- demo cannot run without manual notebook steps
- user logs are used without consent scope and deletion policy

## Review Checklist

Before moving from one milestone to the next, verify:

- Is the current artifact reproducible from commands?
- Is the relevant doc updated?
- Are generated artifacts ignored?
- Are evaluation splits protected?
- Is model choice still evidence-based?
- Are privacy risks controlled before using user feedback?
