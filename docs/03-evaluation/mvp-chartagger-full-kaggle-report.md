# JDWR v1 CharTagger — Full-data Kaggle Run

Generated on 2026-08-15 from the completed Kaggle artifact bundle.

## Run contract

- Config: `configs/model/mvp_chartagger_full_kaggle.yaml`
- Hardware: Kaggle Tesla T4, PyTorch 2.10.0+cu128
- Model: two-head bidirectional GRU for character/diacritic and whitespace tags
- Seed: 42
- Training data: 577,068 aligned examples across seven in-domain categories
- Development data: 72,153 aligned examples
- Training schedule: 12 epochs, 54,108 optimizer steps, batch size 128
- Selection rule: lowest full-development corpus CER
- Final checkpoint: `mvp-chartagger-full-kaggle.pt` (epoch 12)

The run used length buckets and padding to a multiple of 32. It did not apply
a sample or step cap: the configured limits were above all split sizes.

## Training and development trajectory

| Epoch | Steps | Train loss | Dev exact | Dev corpus CER | Dev boundary F1 | Dev diacritic accuracy |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4,509 | 0.3332 | 0.0529 | 0.0600 | 0.9833 | 0.9298 |
| 4 | 18,036 | 0.2069 | 0.0800 | 0.0504 | 0.9856 | 0.9409 |
| 8 | 36,072 | 0.1937 | 0.0884 | 0.0478 | 0.9861 | 0.9439 |
| 12 | 54,108 | 0.1887 | 0.0937 | 0.0466 | 0.9866 | 0.9452 |

Train loss fell 43.4% from epoch 1 to epoch 12, while development corpus CER
fell 22.4%. The final epoch has the best development corpus CER and was thus
saved as the production candidate. Exact match peaked marginally earlier at
epoch 11 (9.38% versus 9.37% at epoch 12), but checkpoint selection followed
the preconfigured CER criterion.

## Full held-out evaluation

All 232,029 supplied test rows were scored: 72,078 in-domain and 159,951 from
the untouched external `the_thao` domain.

| Split | Count | Exact | Corpus CER | Corpus WER | Token F1 | Boundary F1 | Diacritic accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| In-domain | 72,078 | 0.0936 | 0.0466 | 0.1947 | 0.8081 | 0.9866 | 0.9455 |
| External (`the_thao`) | 159,951 | 0.0623 | 0.0753 | 0.3286 | 0.6797 | 0.9527 | 0.9295 |

The in-domain result confirms strong whitespace recovery (boundary F1 98.66%)
and materially better diacritic recovery than the earlier quick baseline. The
external result remains usable, but has a clear domain shift:

- Corpus CER increases by 0.0287 absolute points (61.5% relative).
- Boundary F1 decreases by 0.0339 points.
- Exact match decreases by 3.13 points (33.5% relative).

The prediction bundle contains representative short ambiguous cases such as
`hoga` being predicted as `họ gá` instead of `ho gà`, and the numeric input
`12` remaining unsegmented when the target is `1 2`. These are consistent with
the residual spacing and lexical-disambiguation errors implied by the CER and
exact-match metrics.

## Relation to the quick neural baseline

The earlier CharTagger v1 report used only 100,000 training examples, 1,000
steps, and a 10,000-row development sample. Its 0.0678 corpus CER and 4.12%
exact match are not a controlled head-to-head comparison with this full test,
because the evaluated splits differ. They are nevertheless useful directional
context: the full run reaches 0.0466 in-domain corpus CER and 9.36% exact
match on a substantially larger held-out set.

## Decision and next steps

The full-data model should replace the previous quick checkpoint as the JDWR
v1 reference candidate. The main remaining risk is out-of-domain robustness,
not basic spacing restoration.

1. Add external-domain examples or domain-aware augmentation, prioritizing
   sports/news vocabulary and number segmentation.
2. Report category-level results inside the in-domain test split to identify
   the weakest source categories.
3. Evaluate a hybrid decoder or lexicon-aware postprocessor for short,
   ambiguous compact strings where character context alone is insufficient.
4. Keep corpus CER as the primary checkpoint criterion, while monitoring exact
   match so small CER gains do not mask regression on complete restorations.

## Artifact inventory

The downloaded Kaggle bundle contains the epoch-12 checkpoint, vocabulary,
training history, 232,029 predictions, and JSON/Markdown metric reports. The
binary artifacts remain local and are intentionally not committed; this report
is the versioned record of their results.
