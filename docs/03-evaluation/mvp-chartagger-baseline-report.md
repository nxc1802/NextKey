# MVP CharTagger Neural Baseline Report

Generated on 2026-07-07 for the narrowed MVP:

- restore Vietnamese diacritics
- restore spacing
- ignore punctuation and capitalization

## Model

The neural baseline is a context-aware character tagger:

- input: compact accent-free character sequence
- output label per input character: accented character, optionally prefixed by a
  space
- architecture: embedding + bidirectional GRU + linear classifier

This formulation is better matched to the MVP than a free-form seq2seq decoder
because the current data is almost perfectly aligned after removing accents,
spaces, punctuation, and case.

## Training

Config: `configs/model/mvp_chartagger_v1.yaml`

Training run:

- train examples: 100,000
- max steps: 1,000
- batch size: 128
- embedding dim: 96
- hidden dim: 192
- final loss: 0.2714
- checkpoint: `models/checkpoints/mvp-chartagger-v1.pt`

## Evaluation

Evaluation was run on 10,000 aligned dev examples that satisfy the configured
max length. This is a quick baseline report, not the final full test report.

| Model | Count | Exact | CER | WER | Token F1 | Spacing F1 | Diacritic Acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| Lexicon DP baseline | 10,000 | 0.0199 | 0.0785 | 0.3099 | 0.6967 | 0.9804 | 0.9096 |
| CharTagger v1 | 10,000 | 0.0412 | 0.0678 | 0.2793 | 0.7335 | 0.9776 | 0.9246 |

## Interpretation

The context-aware neural baseline is already better than the lexicon baseline
on the main quality metrics:

- Exact match improves from 1.99% to 4.12%.
- CER improves from 0.0785 to 0.0678.
- WER improves from 0.3099 to 0.2793.
- Token F1 improves from 0.6967 to 0.7335.
- Diacritic accuracy improves from 0.9096 to 0.9246.

Spacing F1 is slightly lower than the lexicon baseline, but still high at
0.9776. This is acceptable for the first neural baseline and can be improved
with more training data, length bucketing, and class weighting for space-prefixed
labels.

## Decision

The neural direction is feasible and should replace lexicon DP as the main MVP
baseline candidate.

Recommended next steps:

1. Run category-balanced evaluation for CharTagger v1.
2. Train on more than 100k examples and save best checkpoint by dev CER.
3. Add length buckets so long samples are evaluated fairly.
4. Add a hybrid decoder that uses lexicon DP for spacing confidence and
   CharTagger for diacritics, if spacing remains stronger in the lexicon model.
