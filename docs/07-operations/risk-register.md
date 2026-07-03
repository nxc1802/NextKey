# Risk Register

| Risk | Level | Signal | Mitigation |
|---|---|---|---|
| Scope too broad | Very high | No end-to-end demo by midpoint | Lock core: restoration, typo correction, benchmark, demo |
| Synthetic data feels fake | High | Strong synthetic metrics but weak human-dev metrics | Increase human-refined set and tune generator from real examples |
| Model overcorrects names or acronyms | High | Proper nouns, brands, or code-like spans are changed incorrectly | Add keep-list, penalties, and overcorrection metric |
| Model choice becomes opinion-based | High | Team debates model before benchmark exists | Keep model-selection gate open and require comparable metrics |
| ByT5 or large teacher is too slow | High | Latency too high for demo or edge story | Use as teacher only and distill to student |
| Export fails | Medium | ONNX/LiteRT conversion errors or runtime too slow | Select export-friendly student and benchmark ONNX first |
| Insufficient human data | Medium | No realistic dev/test set | Run small annotation pilot early and scale only after guideline stabilizes |
| Insufficient personalization logs | Medium | Not enough accepted/rejected pairs | Start with dictionary and synthetic user styles |
| Privacy concerns | High | Users refuse logging or logs contain sensitive text | Use opt-in, deletion, hashed IDs, retention limits, and PII redaction |
| Compute shortage | Medium | Training queue blocks main experiments | Keep small baselines, use ByT5-small before base, reduce run matrix |
| Committee sees project as only diacritic restoration | High | Questions focus on "adding accents" | Emphasize compact writing, typo correction, edge, personalization, and human-centered metrics |

## Kill Criteria For Optional Work

Drop or defer optional work when:

- it does not improve the main metrics
- it blocks demo completion
- it requires sensitive logs before privacy controls exist
- it cannot be evaluated fairly in the remaining time

## Required Fallbacks

- rule baseline fallback for demo
- char-level baseline fallback for edge story
- server-side teacher fallback when export fails
- static sample demo when live model is unavailable during defense
