"""Metrics for the narrowed MVP: diacritic and spacing restoration."""

from __future__ import annotations

from dataclasses import dataclass

from nextkey.data.mvp_dataset import compact_key, strip_accents


def levenshtein(a: str, b: str) -> int:
    """Compute Levenshtein distance with O(min(n, m)) memory."""
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (char_a != char_b),
                )
            )
        previous = current
    return previous[-1]


def cer(prediction: str, target: str) -> float:
    if not target:
        return 0.0 if not prediction else 1.0
    return levenshtein(prediction, target) / len(target)


def wer(prediction: str, target: str) -> float:
    pred_tokens = prediction.split()
    target_tokens = target.split()
    if not target_tokens:
        return 0.0 if not pred_tokens else 1.0
    return levenshtein_tokens(pred_tokens, target_tokens) / len(target_tokens)


def levenshtein_tokens(a: list[str], b: list[str]) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, token_a in enumerate(a, start=1):
        current = [i]
        for j, token_b in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (token_a != token_b),
                )
            )
        previous = current
    return previous[-1]


def token_f1(prediction: str, target: str) -> float:
    pred_tokens = prediction.split()
    target_tokens = target.split()
    if not pred_tokens and not target_tokens:
        return 1.0
    if not pred_tokens or not target_tokens:
        return 0.0
    target_counts: dict[str, int] = {}
    for token in target_tokens:
        target_counts[token] = target_counts.get(token, 0) + 1
    overlap = 0
    for token in pred_tokens:
        if target_counts.get(token, 0) > 0:
            overlap += 1
            target_counts[token] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(target_tokens)
    return 2 * precision * recall / (precision + recall)


def spacing_boundaries(text: str) -> set[int]:
    """Return compact-character offsets where a space appears after a token."""
    boundaries: set[int] = set()
    offset = 0
    tokens = text.split()
    for token in tokens[:-1]:
        offset += len(compact_key(token))
        boundaries.add(offset)
    return boundaries


def spacing_f1(prediction: str, target: str) -> float:
    pred_boundaries = spacing_boundaries(prediction)
    target_boundaries = spacing_boundaries(target)
    if not pred_boundaries and not target_boundaries:
        return 1.0
    if not pred_boundaries or not target_boundaries:
        return 0.0
    overlap = len(pred_boundaries & target_boundaries)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_boundaries)
    recall = overlap / len(target_boundaries)
    return 2 * precision * recall / (precision + recall)


def boundary_counts(prediction: str, target: str) -> tuple[int, int, int]:
    """Return corpus-additive true-positive, predicted and gold boundary counts."""
    predicted = spacing_boundaries(prediction)
    gold = spacing_boundaries(target)
    return len(predicted & gold), len(predicted), len(gold)


def diacritic_accuracy(prediction: str, target: str) -> float:
    """Compare non-space characters when their accent-stripped forms align."""
    pred_chars = prediction.replace(" ", "")
    target_chars = target.replace(" ", "")
    if not target_chars:
        return 1.0 if not pred_chars else 0.0
    if strip_accents(pred_chars) != strip_accents(target_chars):
        return 0.0
    matches = sum(1 for pred_char, target_char in zip(pred_chars, target_chars) if pred_char == target_char)
    return matches / len(target_chars)


@dataclass
class MetricTotals:
    count: int = 0
    exact: int = 0
    cer_sum: float = 0.0
    wer_sum: float = 0.0
    token_f1_sum: float = 0.0
    spacing_f1_sum: float = 0.0
    diacritic_accuracy_sum: float = 0.0
    char_distance: int = 0
    char_total: int = 0
    word_distance: int = 0
    word_total: int = 0
    boundary_true_positive: int = 0
    boundary_predicted: int = 0
    boundary_gold: int = 0

    def update(self, prediction: str, target: str) -> None:
        self.count += 1
        self.exact += int(prediction == target)
        char_distance = levenshtein(prediction, target)
        target_length = len(target)
        self.cer_sum += char_distance / target_length if target_length else (0.0 if not prediction else 1.0)
        prediction_tokens = prediction.split()
        target_tokens = target.split()
        word_distance = levenshtein_tokens(prediction_tokens, target_tokens)
        target_token_count = len(target_tokens)
        self.wer_sum += (word_distance / target_token_count if target_token_count
                         else (0.0 if not prediction_tokens else 1.0))
        self.token_f1_sum += token_f1(prediction, target)
        true_positive, predicted, gold = boundary_counts(prediction, target)
        if not predicted and not gold:
            self.spacing_f1_sum += 1.0
        elif predicted and gold and true_positive:
            precision, recall = true_positive / predicted, true_positive / gold
            self.spacing_f1_sum += 2 * precision * recall / (precision + recall)
        self.diacritic_accuracy_sum += diacritic_accuracy(prediction, target)
        self.char_distance += char_distance
        self.char_total += target_length
        self.word_distance += word_distance
        self.word_total += target_token_count
        self.boundary_true_positive += true_positive
        self.boundary_predicted += predicted
        self.boundary_gold += gold

    def as_dict(self) -> dict[str, float | int]:
        if self.count == 0:
            return {
                "count": 0,
                "exact_match": 0.0,
                "cer": 0.0,
                "wer": 0.0,
                "token_f1": 0.0,
                "spacing_f1": 0.0,
                "diacritic_accuracy": 0.0,
                "corpus_cer": 0.0,
                "corpus_wer": 0.0,
                "boundary_precision": 0.0,
                "boundary_recall": 0.0,
                "boundary_f1": 0.0,
            }
        boundary_precision = self.boundary_true_positive / self.boundary_predicted if self.boundary_predicted else 0.0
        boundary_recall = self.boundary_true_positive / self.boundary_gold if self.boundary_gold else 0.0
        boundary_f1 = (2 * boundary_precision * boundary_recall / (boundary_precision + boundary_recall)
                       if boundary_precision + boundary_recall else 0.0)
        return {
            "count": self.count,
            "exact_match": round(self.exact / self.count, 6),
            "cer": round(self.cer_sum / self.count, 6),
            "wer": round(self.wer_sum / self.count, 6),
            "token_f1": round(self.token_f1_sum / self.count, 6),
            "spacing_f1": round(self.spacing_f1_sum / self.count, 6),
            "diacritic_accuracy": round(self.diacritic_accuracy_sum / self.count, 6),
            "corpus_cer": round(self.char_distance / self.char_total, 6) if self.char_total else 0.0,
            "corpus_wer": round(self.word_distance / self.word_total, 6) if self.word_total else 0.0,
            "boundary_precision": round(boundary_precision, 6),
            "boundary_recall": round(boundary_recall, 6),
            "boundary_f1": round(boundary_f1, 6),
        }
