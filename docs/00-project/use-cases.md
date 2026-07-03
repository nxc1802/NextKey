# Use Cases

This document defines the product-facing scenarios that guide data, model, and
demo decisions.

## UC1: Restore Compact Input

User enters:

```text
t muonviet 1 bai ve ai cho svnam4
```

System returns:

```text
Tôi muốn viết một bài về AI cho sinh viên năm 4.
```

Acceptance:

- input can include multiple noise types
- output is standard Vietnamese
- response includes model/runtime metadata

## UC2: Choose From Candidates

User enters ambiguous compact input.

System returns top-k candidates:

1. best output
2. plausible alternative
3. another plausible alternative

Acceptance:

- candidates are ranked
- user can accept one
- chosen candidate can be logged with consent

## UC3: Correct Keyboard-Aware Typo

User enters text with nearby-key typo or Telex residue:

```text
hom nay toi xam dang bai moi
```

System should produce candidates that recover the likely intended sentence
without overcorrecting names, acronyms, or domain terms.

Acceptance:

- typo correction is measured separately from diacritic restoration
- overcorrection is tracked

## UC4: Compare Model Candidates

Researcher runs evaluation over a shared split.

Acceptance:

- all models use the same split and metric scripts
- result files include dataset version and model version
- model-selection decision is reproducible

## UC5: Edge Prototype

User or researcher runs a smaller student model.

Acceptance:

- student has measured model size
- RAM and latency are recorded
- quality drop versus teacher is reported

## UC6: Personalization With Consent

User opts in to personalization and edits model output.

System stores minimal feedback and improves future ranking or restoration for
that user.

Acceptance:

- consent scope is explicit
- deletion path exists
- base versus personalized behavior is measurable

## Out-Of-Scope User Stories

- User installs a production mobile keyboard.
- User syncs personalization across devices with a full account system.
- System trains from private data without explicit consent.
