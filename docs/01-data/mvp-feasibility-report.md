# MVP Data Feasibility Report

Generated from the provided CSV files in `data/processed` on 2026-07-07.

## MVP Scope

The current MVP is limited to:

- Vietnamese diacritic restoration
- spacing restoration

The output target for generated MVP splits is lowercase Vietnamese text with
accents and spaces. Punctuation and capitalization are intentionally removed
from the MVP target so the first experiment does not accidentally expand into a
larger punctuation/capitalization restoration task.

Brace markers such as `{tphcm}`, `{30-4}`, and `{beckham}` are ignored for now
by removing only the marker characters and keeping the content inside.

## Source Data

| Category | Rows |
|---|---:|
| `chinh_tri_xa_hoi` | 116,322 |
| `doi_song` | 124,040 |
| `kinh_doanh` | 74,139 |
| `phap_luat` | 95,971 |
| `suc_khoe` | 90,690 |
| `the_gioi` | 91,281 |
| `the_thao` | 160,014 |
| `van_hoa` | 131,885 |

Total source rows: 884,342.

## Profiling Result

| Check | Value |
|---|---:|
| CSV files | 8 |
| Total rows | 884,342 |
| Valid rows | 884,342 |
| Rows with brace markers | 419,019 |
| Brace marker rate | 47.38% |
| HTML-like rows | 767 |
| HTML-like rate | 0.09% |
| Normalized alignment match rows | 884,072 |
| Normalized alignment match rate | 99.97% |

The normalized alignment check compares input and target after removing accents,
spacing, punctuation, case, and brace markers. A 99.97% match rate means the
data is highly consistent with the narrowed MVP task.

## Generated MVP Splits

Path: `data/processed/mvp/`

| Split | Rows |
|---|---:|
| `train.jsonl` | 707,039 |
| `dev.jsonl` | 88,351 |
| `test.jsonl` | 88,185 |
| Total kept | 883,575 |

Skipped rows:

| Reason | Rows |
|---|---:|
| HTML-like row | 497 |
| Alignment mismatch | 270 |

## Feasibility Conclusion

The MVP is feasible with the provided data.

Reasons:

- Data volume is far above the minimum needed for a first supervised experiment.
- The source-target relationship is almost perfectly aligned for restoration of
  accents and spacing.
- Brace markers are common but can be ignored safely for the first MVP by
  removing only `{` and `}`.
- HTML-like noise exists but is small enough to filter out before training.

## Recommended Next Step

Proceed to a first baseline experiment:

1. Use `data/processed/mvp/train.jsonl`, `dev.jsonl`, and `test.jsonl`.
2. Start with a small character-level seq2seq baseline.
3. Evaluate only accent and spacing metrics first.
4. Keep punctuation, capitalization, abbreviation expansion, typo correction,
   personalization, and edge deployment out of this MVP.
