# Dataset Build Plan

The dataset is the central asset of NextKey. Model work should not proceed
seriously until the schema, noise taxonomy, and evaluation splits are stable.

## Dataset Layers

| Layer | Purpose | Source |
|---|---|---|
| Clean source sentences | Ground-truth Vietnamese text | Curated corpora, articles, education/content/social/email samples |
| Synthetic pairs | Large training data | Generator applies controlled compact-writing noise |
| Human-refined pairs | Realistic dev/test data | Annotators rewrite standard sentences as fast compact input |
| User logs | Personalization data | Opt-in demo feedback |
| Preference pairs | DPO-lite or reranker learning | Chosen/rejected candidates from feedback |

## Planned Data Paths

```text
data/
  raw/
    clean_sources/
  interim/
    normalized_sources/
    generated_candidates/
  processed/
    train.synthetic.jsonl
    dev.synthetic.jsonl
    dev.human.jsonl
    test.human.jsonl
    personalization.eval.jsonl
  user_logs/
    consented_feedback.jsonl
```

Generated datasets should not be committed to git unless they are tiny samples.

## Build Phases

### Phase 1: Clean Source Collection

Tasks:

- collect short and medium Vietnamese sentences
- track source domain
- normalize Unicode
- remove duplicates
- filter very short, very long, malformed, or low-quality lines

Target:

- minimum: 200k clean sentences
- good: 500k clean sentences
- very good: 1M+ clean sentences

### Phase 2: Synthetic Generator

Tasks:

- implement deterministic rules for coverage
- implement stochastic constrained generation for variety
- generate easy, medium, and hard profiles
- attach `noise_tags`, `difficulty`, and generator metadata

Target:

- minimum: 100k synthetic pairs
- good: 500k synthetic pairs
- very good: 1-2M synthetic pairs

### Phase 3: Dataset QA

Tasks:

- verify `compact_input` differs from `standard_output`
- reject empty or corrupted text
- reject overly distorted samples
- check noise-tag consistency
- check domain distribution
- check duplicate leakage across splits

Target:

- every dataset build produces a QA report

### Phase 4: Human-Refined Set

Tasks:

- prepare annotation guideline
- ask annotators to rewrite clean sentences as realistic fast input
- collect device, typing style, and domain metadata
- cross-check 10-15 percent of samples
- freeze `dev_human` and `test_human`

Target:

- minimum: 1k dev human + 1k test human
- good: 2k dev human + 3k test human
- very good: 3k dev human + 5k test human

### Phase 5: Feedback And Preference Data

Tasks:

- log accepted candidate
- log rejected candidates
- log manually edited output
- derive chosen/rejected preference pairs
- only use opt-in data

Target:

- minimum for DPO-lite pilot: 2k preference pairs
- good: 10k preference pairs

## Dataset Freeze Rules

- Never tune model choices on the final `test_human` set.
- Keep `dev_human` for model selection and error analysis.
- Keep user-level splits separate for personalization evaluation.
- Record generator version and source corpus version for every dataset build.
