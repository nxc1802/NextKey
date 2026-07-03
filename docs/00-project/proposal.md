# Project Proposal

## Working Title

Personalized and Edge-Efficient Vietnamese Compact Writing Restoration with
Keyboard-Aware Typo Correction

## Short Title

NextKey: Vietnamese Compact Writing Restoration

## Background

Vietnamese text input often appears in compact or noisy form, especially when
users type quickly on desktop or mobile keyboards. Common phenomena include
missing diacritics, omitted punctuation, lowercasing, merged words, informal
abbreviations, Telex/VNI residue, and nearby-key typing mistakes.

Traditional Vietnamese diacritic restoration addresses only one part of this
problem. Practical input assistance requires a broader task: restoring a
standard Vietnamese sentence from mixed compact-writing noise.

## Problem Statement

Given a noisy compact input string `x`, the system must produce a standard
Vietnamese sentence `y`.

The input may include:

- no diacritics
- merged spans
- chat-style abbreviations
- missing punctuation and capitalization
- keyboard-neighbor typos
- Telex/VNI residue
- mixed Vietnamese-English terms

The output should restore:

- diacritics
- spacing
- punctuation
- capitalization
- typo corrections
- abbreviation expansion where appropriate

## Motivation

This topic is suitable for a graduation project because it combines research
and system implementation:

- It has a clear Vietnamese NLP problem.
- It requires a data contribution through a noise taxonomy and dataset builder.
- It supports fair model comparison.
- It can produce a visible demo.
- It has practical constraints through edge deployment and privacy-aware
  personalization.

## Research Questions

1. Can a system restore standard Vietnamese from compact input with mixed noise?
2. Which model family works best under shared data and evaluation constraints?
3. Does keyboard-aware synthetic noise improve performance on human-refined
   compact input?
4. Can a smaller student model retain useful quality for edge-style inference?
5. Does top-k candidate generation improve user-facing usefulness?
6. Can lightweight personalization improve user-specific restoration while
   preserving privacy controls?

## Proposed Method

The project will be implemented in stages.

First, the project defines a canonical JSONL dataset schema and a compact-input
noise taxonomy. Clean Vietnamese source sentences are transformed into
synthetic pairs by applying controlled noise such as diacritic removal, spacing
merge, abbreviation, punctuation drop, nearby-key typo, and Telex/VNI residue.
A human-refined dev/test set is collected or prepared through a guideline so
the final evaluation is not limited to synthetic data.

Second, the project builds a shared evaluation harness. Candidate models are
compared with the same data splits and metrics. Model choice is intentionally
left open until this benchmark exists.

Third, the project implements a restoration pipeline that can generate top-k
candidates. The pipeline may include preprocessing, restoration model,
candidate generation, optional reranking, post-processing, and structured
feedback logging.

Fourth, the project explores one or both advanced directions depending on time
and resources: edge optimization through distillation/export/quantization, and
privacy-aware personalization through dictionaries, priors, reranking, or
adapter-based learning.

## Open Model Selection

The project does not preselect a final model.

Candidate roles include:

- rule-based baseline
- char-level neural baseline
- pretrained multilingual text-to-text model
- byte-level text-to-text model
- Vietnamese-specific generation model
- reranker
- distilled student model

The final decision must be based on:

- quality metrics
- top-k behavior
- robustness by noise type
- latency and memory
- training and deployment cost
- implementation risk

## Dataset Plan

The dataset consists of:

- clean source sentences
- synthetic noisy pairs
- human-refined dev/test samples
- optional user feedback logs
- optional preference pairs

Minimum targets:

- 100k synthetic pairs
- 1k human-refined dev/test samples
- domain and noise metadata for analysis

Target domains:

- content writing
- social text
- email
- education
- admin/productivity
- search queries
- casual short text

## Evaluation Plan

The evaluation will include:

- Exact Match
- CER
- WER
- Diacritic Accuracy
- Segmentation F1
- Punctuation F1
- Capitalization Accuracy
- Typo Correction Accuracy
- Overcorrection Rate
- Top-3/Top-5 Hit Rate
- latency p50/p95
- model size and RAM for edge experiments
- acceptance rate and correction effort for user-facing experiments if possible

## Expected Deliverables

Required:

- dataset schema
- noise taxonomy
- synthetic generator
- human annotation guideline
- baseline models
- evaluation harness
- model-selection report
- restoration command/API
- demo path
- final report and slides

Strong-project additions:

- top-k candidate UI
- reranking
- edge student model
- quantization benchmark
- personalization pilot
- user study pilot

## Success Criteria

Minimum success:

- end-to-end restoration path works
- baselines and selected candidate are benchmarked fairly
- human-refined evaluation is included
- project is clearly broader than diacritic restoration

Strong success:

- keyboard-aware noise shows measurable benefit
- top-k candidates improve practical usability
- edge or personalization contribution is measured
- final defense includes ablation and error analysis

## Risks

| Risk | Mitigation |
|---|---|
| Scope becomes too broad | Prioritize P0 core and defer P2 items |
| Synthetic data is unrealistic | Add human-refined evaluation early |
| Model choice becomes subjective | Use an open benchmark gate |
| Edge export is difficult | Use a small student model and ONNX-first strategy |
| Personalization lacks data | Start with dictionary and feedback schema |
| Privacy concerns | Use opt-in, hashed user IDs, deletion, and minimization |

## Initial Timeline

| Milestone | Output |
|---|---|
| M1 | project contract and repo foundation |
| M2 | dataset schema, taxonomy, generator v1 |
| M3 | baselines and evaluation harness |
| M4 | model-selection benchmark |
| M5 | restoration API/demo MVP |
| M6 | edge or personalization push |
| M7 | final evaluation and ablation |
| M8 | thesis, slides, video, demo checklist |
