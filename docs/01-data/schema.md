# Dataset Schema

The canonical training and evaluation format is JSON Lines. Each line is one
sample.

## Required Fields

| Field | Type | Description |
|---|---|---|
| `sample_id` | string | Stable unique ID |
| `domain` | string | Domain such as `content`, `social`, `email`, `education`, `admin`, `search_query`, or `casual` |
| `source_type` | string | `synthetic`, `human_refined`, or `user_log` |
| `standard_output` | string | Clean target sentence |
| `compact_input` | string | Noisy compact input |
| `noise_tags` | array string | Applied noise categories |
| `difficulty` | string | `easy`, `medium`, or `hard` |
| `created_at` | string | ISO timestamp |

## Optional Fields

| Field | Type | Description |
|---|---|---|
| `abbr_profile` | string | `none`, `common`, `heavy`, or `user_specific` |
| `keyboard_layout` | string | `qwerty_telex`, `qwerty_vni`, or `mobile_qwerty` |
| `typo_positions` | array int | Character positions affected by typo operations |
| `typo_ops` | array string | Operations such as `neighbor_sub`, `delete`, `insert`, `transpose`, `repeat`, `telex_residue` |
| `candidate_list` | array string | Top-k candidates from inference, if available |
| `chosen_candidate` | string | Candidate accepted by user |
| `edited_output` | string | Manual correction after model output |
| `user_id_hash` | string | Hashed user identifier for opt-in personalization |
| `consent_scope` | string | `none`, `personalization_only`, or `research_only` |
| `generator_version` | string | Version of the synthetic generator |
| `source_ref` | string | Internal source reference, if allowed |

## Example

```json
{
  "sample_id": "S000001",
  "domain": "content",
  "source_type": "synthetic",
  "standard_output": "Tôi muốn viết một bài về AI cho sinh viên năm 4.",
  "compact_input": "t muonviet 1 bai ve ai cho svnam4",
  "noise_tags": ["no_diacritic", "space_merge", "abbr", "number_replace", "punct_drop"],
  "difficulty": "hard",
  "abbr_profile": "common",
  "keyboard_layout": "qwerty_telex",
  "typo_positions": [],
  "typo_ops": [],
  "created_at": "2026-07-03T00:00:00+07:00",
  "generator_version": "synthetic-v1"
}
```

## Split Naming

Use explicit split names:

- `train.synthetic.jsonl`
- `dev.synthetic.jsonl`
- `dev.human.jsonl`
- `test.human.jsonl`
- `personalization.eval.jsonl`
- `preference.train.jsonl`

## Validation Requirements

Every sample must pass:

- valid UTF-8
- non-empty `compact_input`
- non-empty `standard_output`
- `compact_input` is not identical to `standard_output` for synthetic training
- `noise_tags` match observable transformations where possible
- no raw personal identifiers in research splits
