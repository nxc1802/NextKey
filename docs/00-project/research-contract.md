# Research Contract

This document locks the research shape of NextKey while keeping model choice
open until the model-selection gate.

## Final Problem Statement

Given a compact, noisy Vietnamese input string `x`, NextKey restores a standard
Vietnamese sentence `y`.

The input may combine:

- missing diacritics
- merged words or syllables
- informal abbreviations
- missing punctuation
- missing capitalization
- keyboard-neighbor typos
- Telex/VNI residue
- mixed Vietnamese-English domain terms

The output should restore:

- Vietnamese diacritics
- spacing
- punctuation
- capitalization
- appropriate abbreviation expansion
- typo corrections

This is not only a Vietnamese diacritic restoration project. The central task is
compact Vietnamese input restoration under mixed noise.

## Primary Use Cases

| Use Case | Description | Priority |
|---|---|---|
| Fast text input | User types compact Vietnamese and receives a restored sentence | P0 |
| Candidate selection | User chooses from top-k restored candidates | P0 |
| Typo-aware correction | System handles nearby-key and Telex/VNI residue mistakes | P0 |
| Feedback capture | User accepts, rejects, or edits output for later learning | P1 |
| Edge prototype | Student model runs with lower latency and memory | P1 |
| Personalization | System adapts to user-specific abbreviations and phrasing | P2 |

## Research Questions

| ID | Question | Evidence Required |
|---|---|---|
| RQ1 | Can a system restore standard Vietnamese from compact input containing simultaneous missing diacritics, spacing errors, abbreviations, punctuation loss, and keyboard-aware typos? | Exact Match, CER, WER, Diacritic Accuracy, Segmentation F1, Typo Correction Accuracy |
| RQ2 | Which model family is most suitable for noisy compact Vietnamese input under the project constraints? | Comparable benchmark across model candidates on the same splits |
| RQ3 | Does keyboard-aware synthetic noise improve performance on human-refined compact input compared with generic or reduced noise? | Ablation: full noise generator versus no keyboard-neighbor noise |
| RQ4 | Can a smaller student model retain enough quality for edge-style inference after distillation or compression? | Teacher versus student quality, model size, RAM, latency p50/p95 |
| RQ5 | Does top-k candidate generation and reranking improve practical usability compared with single-output decoding? | Top-3/Top-5 Hit Rate, acceptance rate, correction effort |
| RQ6 | Does lightweight personalization improve user-specific restoration behavior without unsafe logging? | Base versus personalized evaluation with consented logs |

## Open Model Policy

No model is final at this stage.

Candidate families may include:

- rule-based heuristics
- char-level seq2seq
- mT5-style multilingual text-to-text models
- byte-level text-to-text models
- Vietnamese encoder-decoder models
- rerankers based on Vietnamese language models
- distilled student models

Model roles must be decided through:

1. shared data splits
2. shared evaluation scripts
3. quality metrics
4. latency and memory measurements
5. implementation risk

The project may start with placeholder configs, but final claims must not say a
model is best until comparable evidence exists.

## Core Claim To Defend

NextKey proposes a unified pipeline for compact Vietnamese input restoration
that combines data taxonomy, synthetic and human-refined evaluation, model
selection, practical top-k UX, and an edge/personalization path.

## What Counts As Success

The project succeeds at MVP level if:

- dataset schema and noise taxonomy are implemented
- synthetic and human-refined evaluation splits exist
- at least two baselines and one selected candidate are benchmarked fairly
- the system exposes a working restoration path through CLI/API/demo
- results are reported by metric group, noise type, and domain

The project becomes strong if:

- top-k candidate UX works
- keyboard-aware noise ablation shows value
- edge student or personalization shows measurable benefit
- final defense includes error analysis and limitations
