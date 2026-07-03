# Scope Matrix

This file separates required work from optional work so the project can finish
end to end before adding research extensions.

## Priority Definitions

| Priority | Meaning |
|---|---|
| P0 | Required for MVP and final defensibility |
| P1 | Strong-project work after P0 is stable |
| P2 | Extension work if time, data, and compute allow |
| Out | Explicitly out of scope for this graduation project |

## Feature Scope

| Feature | Priority | Notes |
|---|---|---|
| Problem definition as compact Vietnamese input restoration | P0 | Must avoid narrowing to diacritic restoration only |
| Dataset schema | P0 | JSONL contract required before training |
| Noise taxonomy | P0 | Must cover missing diacritics, spacing, abbreviations, punctuation, typos, Telex/VNI residue |
| Synthetic generator v1 | P0 | Required for first training data |
| Human-refined dev/test protocol | P0 | Required for credible evaluation |
| Rule baseline | P0 | Lower bound and debugging baseline |
| Small neural baseline | P0 | Char-level or equivalent |
| Open model-selection benchmark | P0 | Final model choice must be evidence-based |
| Core restoration model | P0 | Produces top-1 and ideally top-k outputs |
| Evaluation harness | P0 | Shared metrics and split handling |
| API or CLI restore path | P0 | Needed for demo and reproducibility |
| Demo UI | P1 | Strongly recommended for defense |
| Top-k candidate UX | P1 | Important because multiple restorations may be valid |
| Reranker | P1 | Keep only if it improves top-k/top-1 metrics |
| Edge student | P1 | Main differentiator if feasible |
| ONNX export | P1 | Prefer before LiteRT |
| Quantization | P1 | Measure trade-off, do not assume improvement |
| User study pilot | P1 | Useful for typing-centered claims |
| Personal dictionary | P1 | Low-cost personalization |
| LoRA per-user | P2 | Needs enough user-specific data |
| DPO-lite | P2 | Only after chosen/rejected pairs exist |
| Production mobile keyboard app | Out | Too broad for first project |
| Production user account system | Out | Demo can use hashed local user IDs |
| Large-scale private log collection | Out | Privacy risk and not needed for MVP |

## Must-Have Deliverables

- project proposal
- dataset contract
- data generator
- evaluation harness
- baseline results
- model-selection result
- final benchmark table
- working demo path
- thesis/report material

## Should-Have Deliverables

- edge benchmark
- top-k candidate UI
- error analysis by noise type
- ablation table
- user study pilot

## Extension Deliverables

- personalized reranker
- LoRA adapter
- DPO-lite proof of concept
- polished offline demo

## Scope Control Rules

- Do not add optional model work before evaluation scripts exist.
- Do not collect user logs before privacy controls exist.
- Do not optimize edge before a teacher or baseline model is measurable.
- Do not claim personalization gains without base versus personalized evaluation.
- Drop P2 items first when schedule or compute becomes constrained.
