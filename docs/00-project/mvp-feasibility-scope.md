# MVP Feasibility Scope

This document narrows the first feasibility MVP to match the data currently
available in `data/processed`.

## MVP Goal

Validate whether the project is feasible using the provided CSV pairs:

- `Input_X`: compact Vietnamese input
- `Target_Y`: restored Vietnamese output

For this MVP, the model target is limited to:

1. Vietnamese diacritic restoration
2. spacing restoration

The MVP intentionally does not solve abbreviation expansion, typo correction,
punctuation generation, personalization, reranking, or edge deployment yet.

## Current Data Assumption

The available files are category-specific CSV files under `data/processed`.
Each file should contain the columns:

- `Input_X`
- `Target_Y`

The category is inferred from the filename.

## Brace Marker Policy

Some inputs contain spans in braces, for example `{tphcm}`, `{30-4}`, or
`{beckham}`. These markers appear to identify compound terms, dates, names,
foreign terms, or other grouped spans.

For the first MVP, these markers are ignored:

- remove `{`
- remove `}`
- keep the text inside the braces

Example:

```text
{sri}{lanka} -> srilanka
{30-4} -> 30-4
```

This keeps the current sequence-to-sequence framing simple. A later task can
reintroduce brace spans as structured segmentation hints.

## Feasibility Checks

The first data feasibility report should measure:

- number of CSV files
- number of rows
- number of valid rows
- empty input/target rows
- rows containing brace markers
- rows containing HTML-like text
- input and target length distribution
- ratio of target text that becomes equal to input after removing accents,
  spaces, punctuation, and case
- per-category row counts

## Data Quality Risks

Known risks:

- Some files contain HTML fragments.
- Some target rows include punctuation and capitalization, although MVP only
  focuses on diacritics and spacing.
- Some rows may contain malformed source text from old news pages.
- Some category distributions are much larger than others.

## MVP Acceptance

The MVP data stage is acceptable when:

- all provided CSV files can be read
- brace markers are normalized consistently
- a feasibility report is generated
- at least one sample output file is produced for manual inspection
- invalid or risky rows are quantified instead of silently ignored

## Next Gate

After the feasibility report, decide whether to:

- train a small baseline on all valid rows
- filter out HTML-heavy rows first
- keep only rows where normalized input and normalized target align closely
- create train/dev/test splits from the cleaned rows
