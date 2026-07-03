# Project Overview

## Problem Statement

NextKey targets Vietnamese compact writing restoration. Given a noisy input
string `x`, the system returns a standard Vietnamese sentence `y`.

Input may include:

- missing Vietnamese diacritics
- merged words or syllables
- chat-style abbreviations
- missing punctuation and capitalization
- Telex/VNI residue
- keyboard-neighbor typos
- mixed Vietnamese and English domain terms

Output should be a complete Vietnamese sentence with restored diacritics,
spacing, punctuation, capitalization, expanded abbreviations when appropriate,
and typo correction.

## Product Shape

The target prototype is an input-assist system:

- user types or pastes compact Vietnamese text
- system returns top-k candidate restorations
- user accepts one candidate or edits manually
- feedback can be logged for personalization if the user opts in
- a smaller student model can run offline or near-edge when feasible

## Research Contributions

The project should produce three contribution layers:

1. Data contribution: a compact-writing dataset contract, noise taxonomy,
   synthetic generator, and human-refined test set.
2. Model/system contribution: restoration model, keyboard-aware typo handling,
   candidate reranking, distillation, quantization, and optional personalization.
3. Evaluation contribution: NLP quality metrics, typo metrics, edge metrics,
   and human-centered typing metrics.

## Scope Levels

### P0 Core

- data schema and noise taxonomy
- synthetic dataset generator
- human-refined dev/test protocol
- baseline models
- model-selection benchmark
- restoration API and demo
- evaluation tables

### P1 Strong Project

- top-k candidate reranking
- edge student model
- ONNX or LiteRT export attempt
- quantization benchmark
- user study pilot

### P2 Extension

- per-user dictionary and phrase priors
- LoRA per-user adapter
- DPO-lite preference optimization
- richer privacy controls
- mobile/offline polished demo

## Non-Goals For The First MVP

- building a production keyboard app
- guaranteeing full offline mobile deployment for all candidate models
- training large models without a comparable baseline harness
- collecting sensitive user logs without consent and deletion controls
