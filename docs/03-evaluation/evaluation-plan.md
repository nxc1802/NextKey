# Evaluation Plan

Evaluation must reflect the real input-method use case. BLEU or exact match
alone is not enough.

## Metric Groups

| Group | Metrics |
|---|---|
| NLP quality | Exact Match, CER, WER, Diacritic Accuracy, Segmentation F1, Punctuation F1, Capitalization Accuracy |
| Typo behavior | Typo Detection Precision/Recall/F1, Correction Accuracy, Keyboard-neighbor Recovery |
| Safety of correction | Overcorrection Rate, keep-list violations, named-entity or acronym corruption |
| Candidate UX | Top-1 accuracy, Top-3 Hit Rate, Top-5 Hit Rate |
| Edge performance | model size, peak RAM, latency p50/p95, throughput |
| User usefulness | Keystroke Saving Rate, Correction Effort, Acceptance Rate, Time-to-final-text, Satisfaction Score |

## Required Baselines

- rule-only baseline
- char-level seq2seq baseline
- mT5-small or equivalent pretrained baseline
- selected research teacher
- student model if edge work is included
- pipeline split-step versus joint model if feasible

## Evaluation Splits

| Split | Purpose |
|---|---|
| `dev.synthetic` | quick iteration |
| `dev.human` | model selection |
| `test.human` | final reported quality |
| domain slices | compare content, social, email, education, admin, search query |
| noise slices | compare no-diacritic, space-merge, abbreviation, typo, mixed |
| `personalization_eval` | user-specific adaptation |

## Ablation Plan

| Ablation | Question |
|---|---|
| remove keyboard-neighbor noise | Does realistic typo simulation help? |
| remove human-refined data | Does human data improve real input performance? |
| remove reranker | Is beam top-1 enough? |
| remove personal dictionary | How much does simple personalization help? |
| remove LoRA per-user | Is deeper adaptation worth the cost? |
| remove DPO-lite | Does preference learning beat SFT-only? |
| mT5 versus ByT5 | Do byte-level models help noisy Vietnamese input? |
| teacher versus student | What quality is lost for edge deployment? |
| float versus INT8 | What is the quality/latency trade-off? |

## Internal KPI Targets

These are planning targets, not guaranteed claims.

| Metric | Minimum | Target |
|---|---:|---:|
| Exact Match on `test_human` | 45% | 60%+ |
| CER on `test_human` | < 0.12 | < 0.07 |
| Diacritic Accuracy | 95% | 97%+ |
| Segmentation F1 | 90% | 95%+ |
| Typo Correction Accuracy | 75% | 85%+ |
| Overcorrection Rate | < 8% | < 4% |
| Top-3 Hit Rate | 75% | 90%+ |
| Keystroke Saving Rate | 25% | 40%+ |
| Correction Effort Reduction | 20% | 35%+ |
| Student INT8 latency p50 | < 300ms | < 150ms |

## Reporting Requirements

Every experiment report should include:

- dataset version
- model version
- metric table
- domain breakdown
- noise breakdown
- representative success and failure examples
- latency/memory if relevant
- decision: keep, drop, or rerun
