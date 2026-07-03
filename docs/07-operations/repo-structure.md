# Repository Structure

This is the proposed implementation structure after the documentation stage.

```text
NextKey/
  README.md
  deep-research-report.md
  docs/
  configs/
    data/
    model/
    eval/
  data/
    raw/
    interim/
    processed/
    samples/
  notebooks/
  scripts/
    build_dataset.py
    train.py
    evaluate.py
    export_model.py
    run_demo.py
  src/
    nextkey/
      data/
      generation/
      models/
      evaluation/
      serving/
      personalization/
      privacy/
      utils/
  demo/
    backend/
    frontend/
  experiments/
    runs/
    reports/
  models/
    checkpoints/
    exported/
  tests/
```

## Git Policy

Commit:

- source code
- docs
- configs
- small sample data
- test fixtures

Do not commit:

- large raw datasets
- generated full datasets
- model checkpoints
- private user logs
- secrets
- local experiment caches

## Artifact Naming

Use versioned names:

- `synthetic-v1`
- `human-dev-v1`
- `teacher-byt5-small-exp003`
- `student-char-v2-int8`
- `eval-2026-07-03-dev-human`

## Command Shape

Planned commands:

```bash
python scripts/build_dataset.py --config configs/data/synthetic_v1.yaml
python scripts/train.py --config configs/model/teacher_candidate.yaml
python scripts/evaluate.py --config configs/eval/dev_human.yaml
python scripts/export_model.py --checkpoint models/checkpoints/student
python scripts/run_demo.py
```

## Documentation Update Rule

When implementation starts:

- update dataset docs after schema changes
- update model docs after each selection gate
- update evaluation docs after metric or split changes
- update risk register when a risk becomes real
