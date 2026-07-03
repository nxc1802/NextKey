# Model Selection

Model choice is an open task. This project must not lock the final model before
the dataset contract and evaluation harness exist.

## Candidate Roles

| Role | Candidate Pool | Purpose |
|---|---|---|
| Rule baseline | Dictionary, Telex/VNI rules, heuristic LM | Explainable lower bound |
| Lightweight baseline | Char-level seq2seq | Small model and deployability reference |
| Pretrained baseline | mT5-small | Multilingual text-to-text comparison |
| Research teacher | ByT5-small/base, ViT5, BARTpho | Quality-oriented model |
| Reranker | PhoBERT or light cross-encoder | Rank top-k candidates |
| Edge student | Char-level mini Transformer or mini seq2seq | Distilled deployable model |

## Decision Gates

### Gate 1: Smoke Run

Goal: reject candidates that are too slow, unstable, or difficult to train.

Required checks:

- training starts cleanly
- inference returns Vietnamese text
- output length is controlled
- no severe language drift
- memory use is acceptable for available hardware

### Gate 2: Comparable Benchmark

Goal: compare candidates on the same data.

Required splits:

- `dev.synthetic`
- `dev.human`
- noise-specific slices
- domain-specific slices

Required metrics:

- Exact Match
- CER
- WER
- Diacritic Accuracy
- Segmentation F1
- Typo Correction Accuracy
- Overcorrection Rate
- Top-3 Hit Rate
- latency p50/p95 where relevant

### Gate 3: Role Assignment

Goal: assign model roles.

Decisions:

- teacher model
- student model
- whether a reranker is worth keeping
- whether ViT5/BARTpho are required or deferred
- whether ByT5-base is affordable or ByT5-small is the practical maximum

## Decision Record Template

For every model candidate, record:

- model name and version
- training dataset version
- hyperparameters
- checkpoint path
- hardware
- training time
- metric table
- qualitative error examples
- latency and memory if applicable
- decision: keep, drop, or defer

## Default Bias

Use the smallest model that can satisfy the research and demo goals. Larger
models are useful as teachers only if their quality gain is measurable and the
student path remains credible.
