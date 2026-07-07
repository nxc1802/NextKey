# MVP Lexicon Baseline Report

Generated on 2026-07-07 for the narrowed MVP:

- restore Vietnamese diacritics
- restore spacing
- ignore punctuation and capitalization

## Baseline

The baseline is a no-GPU lexicon and dynamic-programming decoder.

Training:

- input: `data/processed/mvp/train.jsonl`
- rows: 707,039
- total target tokens: 17,340,471
- unique target words: 67,665
- unique compact keys: 60,491
- model artifact: `models/checkpoints/mvp-lexicon-baseline.json`

Inference:

1. Convert compact input to a punctuation-free, accent-free key.
2. Segment the key using dynamic programming.
3. For each compact key, choose the most frequent accented word from training.

This baseline does not use sentence context, so it cannot reliably choose among
Vietnamese homographs such as `giam` -> `giảm` versus `giám`.

## Evaluation Setup

The quick report uses a balanced sample:

- 1,250 samples per category
- 8 categories
- 10,000 samples per split
- splits: `dev`, `test`

Full evaluation was intentionally deferred because edit-distance metrics over
the entire 176k dev/test rows are slower and should be optimized before routine
use.

## Overall Metrics

| Split | Count | Exact | CER | WER | Token F1 | Spacing F1 | Diacritic Acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| `dev` | 10,000 | 0.0199 | 0.0785 | 0.3099 | 0.6967 | 0.9804 | 0.9096 |
| `test` | 10,000 | 0.0184 | 0.0786 | 0.3091 | 0.6973 | 0.9800 | 0.9097 |

## Interpretation

The result is useful as a feasibility floor:

- Spacing restoration is already strong with a simple lexicon baseline.
- Diacritic restoration needs context; unigram frequency is not enough.
- Exact match is low because a few wrong accents or word choices break the whole
  sentence.
- CER around 0.078 shows the baseline is not trivial, but it leaves enough room
  for a neural sequence model to improve.

## Category Notes

Best categories by CER are generally `kinh_doanh`, `the_gioi`, and `the_thao`.
Harder categories include `doi_song`, `suc_khoe`, and `phap_luat`, likely because
they contain more ambiguous common words and domain-specific vocabulary.

## Decision

Proceed to the next phase: implement a context-aware sequence baseline.

Recommended next baseline:

1. Start with a character-level encoder-decoder or Transformer small enough to
   train quickly.
2. Keep the target exactly as the MVP split defines it: lowercase accented text
   with spaces only.
3. Compare against this lexicon baseline on the same balanced quick report.
4. Optimize full dev/test evaluation after the first neural baseline is wired.
