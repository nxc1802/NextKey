# Training Plan

Training should follow a staged approach so each improvement can be measured.

## Common Contract

Every training run must record:

- dataset version
- split names
- model candidate
- tokenizer or byte/char encoding
- hyperparameters
- random seed
- checkpoint path
- evaluation command
- metric output path

## Stage A: Synthetic Easy/Medium

Goal:

- learn basic restoration from no-diacritic and lightly noisy input

Models:

- rule baseline
- char-level seq2seq
- mT5-small
- selected byte/seq2seq candidate

Exit:

- first benchmark table exists

## Stage B: Synthetic Full/Hard

Goal:

- learn mixed noise, keyboard-neighbor typos, abbreviations, and merged spans

Models:

- best candidates from Stage A

Exit:

- hard-noise results improve over Stage A
- overcorrection is tracked

## Stage C: Human-Refined Fine-Tuning

Goal:

- adapt model to realistic compact input distribution

Data:

- human-refined train/dev if available
- never train on final `test_human`

Exit:

- selected teacher candidate improves on `dev_human`

## Stage D: Distillation

Goal:

- compress teacher behavior into a smaller student model

Data:

- gold targets
- teacher top-k outputs
- optional teacher scores or logits if feasible

Exit:

- student quality/latency trade-off is measured

## Stage E: User-Specific SFT

Goal:

- adapt to user edits through lightweight fine-tuning or adapter training

Data:

- opt-in edited outputs
- user-specific abbreviation patterns

Exit:

- base versus personalized evaluation exists

## Stage F: DPO-Lite

Goal:

- learn from chosen/rejected candidate preferences

Prerequisite:

- enough preference pairs from feedback logs

Exit:

- DPO-lite result is compared against user-specific SFT

## Reranker Training

Reranker training should only start after top-k generation is stable.

Possible labels:

- exact ground-truth candidate
- user-chosen candidate
- edited-output similarity score
- rejection signal

Exit:

- top-1 and top-3 hit rates improve after reranking
