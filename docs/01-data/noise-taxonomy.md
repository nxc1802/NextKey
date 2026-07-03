# Noise Taxonomy

The project models compact Vietnamese input as a combination of controlled noise
types. The taxonomy is part of the research contribution and must stay aligned
with the generator, dataset schema, and evaluation reports.

## Core Noise Types

| Tag | Meaning | Example Input | Standard Output |
|---|---|---|---|
| `no_diacritic` | Vietnamese accents removed | `toi dang viet bai` | `Tôi đang viết bài.` |
| `space_merge` | Words or syllables merged | `toidangvietbai` | `Tôi đang viết bài.` |
| `abbr` | Common abbreviation | `sv nam 4` | `sinh viên năm 4` |
| `domain_abbr` | Domain-specific abbreviation | `cta cho camp nay` | `CTA cho campaign này.` |
| `punct_drop` | Punctuation removed | `hom nay toi viet bai moi` | `Hôm nay tôi viết bài mới.` |
| `lowercase` | Capitalization removed | `ha noi la thu do` | `Hà Nội là thủ đô.` |
| `neighbor_sub` | Nearby-key typo | `toj` | `tôi` |
| `delete` | Missing character | `to dang viet` | `Tôi đang viết.` |
| `insert` | Extra character | `toii` | `tôi` |
| `transpose` | Adjacent character swap | `viet` -> `veit` | `viết` |
| `repeat` | Repeated character | `vieet` | `viết` |
| `telex_residue` | Leftover Telex sequence | `vietj`, `ddi` | `việt`, `đi` |
| `vni_residue` | Leftover VNI sequence | `a6`, `d9` | `â`, `đ` |
| `regional_variant` | Region or pronunciation-driven confusion | `s/x`, `ch/tr`, `n/l` | Policy-dependent |
| `mixed` | Multiple noise types combined | `t muonviet 1 bai ve ai` | `Tôi muốn viết một bài về AI.` |

## Difficulty Profiles

| Difficulty | Diacritic Drop | Space Merge | Abbreviation | Typo | Punctuation Drop | Mixed Noise |
|---|---:|---:|---:|---:|---:|---:|
| `easy` | 100% | 10% | 10% | 5% | 30% | 5% |
| `medium` | 100% | 25% | 25% | 12% | 45% | 20% |
| `hard` | 100% | 40% | 40% | 20% | 60% | 35% |

These rates are starting targets. They should be adjusted after human-refined
data reveals the real distribution of typing behavior.

## Generator Modules

Planned generator modules:

- `normalize_unicode`
- `remove_punctuation`
- `lowercase_text`
- `apply_common_abbreviations`
- `apply_domain_abbreviations`
- `strip_vietnamese_diacritics`
- `collapse_spaces_by_span`
- `introduce_keyboard_typos`
- `inject_telex_residue`
- `inject_vni_residue`
- `infer_noise_tags`

## Keyboard-Aware Typo Policy

Keyboard typo generation should use a QWERTY neighbor graph instead of random
character substitution.

Operations:

- `neighbor_sub`: replace selected character with nearby key
- `delete`: remove selected character
- `insert`: insert nearby or repeated character
- `transpose`: swap adjacent characters
- `repeat`: duplicate selected character
- `telex_residue`: leave partial Telex marker
- `vni_residue`: leave partial VNI marker

## Validation Checklist

- Each synthetic sample has at least one noise tag.
- `hard` samples should contain more mixed-noise cases than `easy` samples.
- Keyboard typo positions should be recorded when available.
- Proper nouns, acronyms, and brand terms should be protected by keep-list or
  evaluated separately for overcorrection.
