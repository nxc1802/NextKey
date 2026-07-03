# Human Annotation Guideline

Human-refined data is used to measure whether the system works on realistic
compact input, not only synthetic generator output.

## Annotation Task

Given a clean Vietnamese sentence, annotators rewrite it as they would type it
quickly in a casual input box.

Annotators may:

- remove Vietnamese diacritics
- omit punctuation
- lowercase text
- use common abbreviations
- merge short spans of words
- leave Telex/VNI residue if natural
- include realistic nearby-key typo mistakes

Annotators should not:

- change the meaning of the sentence
- invent unrelated content
- create unreadable random noise
- include private personal information

## Metadata To Collect

For each annotation batch:

- device type: desktop, laptop, mobile
- keyboard style: Telex, VNI, mobile QWERTY, other
- domain: content, social, email, education, admin, search query, casual
- self-rated naturalness from 1 to 5
- optional note for unusual abbreviation or domain term

## Quality Rules

Accept a sample when:

- the compact input is plausible
- the target output still matches the original sentence
- the sample contains useful compact-writing behavior
- naturalness rating is acceptable

Reject or revise a sample when:

- meaning changes
- compact input is random or impossible to recover
- output contains personal data
- source sentence is malformed

## Cross-Check Protocol

- Re-annotate 10-15 percent of samples with a second annotator.
- Compare abbreviation choices, typo behavior, and recoverability.
- Use disagreement to improve the guideline, not to force one fixed style.

## Split Policy

- `dev_human`: model selection and error analysis
- `test_human`: final evaluation only
- `personalization_eval`: user-specific adaptation tests

Avoid user leakage between training and personalization evaluation splits.
