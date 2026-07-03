# Personalization And Privacy

Personalization is valuable, but it must be layered and consent-driven.

## Personalization Ladder

| Level | Mechanism | Data Needed | Priority |
|---|---|---:|---|
| 1 | Personal dictionary | 10-50 edits | P0/P1 |
| 2 | Phrase/domain priors | 50-200 logs | P1 |
| 3 | Personalized reranker | 100-500 logs | P1 |
| 4 | LoRA per-user adapter | 200-2,000 samples | P2 |
| 5 | DPO-lite preference learning | 500-5,000 preference pairs | P2 |

Start with the dictionary. Do not begin LoRA or DPO-lite until logging, consent,
and evaluation are already working.

## Feedback Data Types

- accepted candidate
- rejected candidates
- manually edited output
- compact input
- model metadata
- minimal device/runtime metadata
- consent scope

## Consent Scopes

| Scope | Meaning |
|---|---|
| `none` | Do not store feedback beyond transient request handling |
| `personalization_only` | Use data for the user's own personalization |
| `research_only` | Use anonymized data for research evaluation |

## Privacy Requirements

- Do not store email, phone number, or raw identity in training logs.
- Use `user_id_hash` for user-level grouping.
- Provide deletion for personalization data.
- Keep raw logs for a limited retention window.
- Prefer on-device inference and local personalization where feasible.
- Redact obvious PII patterns before server-side research use.

## Evaluation

Personalization must be measured as:

- base model versus dictionary
- dictionary versus personalized reranker
- SFT/LoRA versus no adapter, if implemented
- DPO-lite versus SFT-only, if implemented

Useful metrics:

- acceptance rate
- top-k hit rate
- correction effort
- user-specific phrase accuracy
- overcorrection rate
