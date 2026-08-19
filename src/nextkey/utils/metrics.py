"""Evaluation metrics for Vietnamese compact-text restoration.

Provides character-level (CER), word-level (WER), boundary F1,
diacritic accuracy, and a cumulative MetricTotals aggregator.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Text normalization helpers
# ---------------------------------------------------------------------------

def strip_accents(text: str) -> str:
    """Remove Vietnamese diacritics while preserving đ/Đ mapping."""
    decomposed = unicodedata.normalize("NFD", text)
    without_marks = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", without_marks).replace("đ", "d").replace("Đ", "D")


def compact_key(text: str) -> str:
    """Produce a canonical accent-free, space-free, lowercase key."""
    import re
    punct_space = re.compile(r"[\W_]+", re.UNICODE)
    return punct_space.sub("", strip_accents(text).lower())


# ---------------------------------------------------------------------------
# Edit distance
# ---------------------------------------------------------------------------

try:
    import Levenshtein as _c_levenshtein
except ImportError:
    _c_levenshtein = None


def levenshtein(a: str, b: str) -> int:
    """Levenshtein distance with C acceleration if available."""
    if a == b:
        return 0
    if _c_levenshtein is not None:
        return _c_levenshtein.distance(a, b)

    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (char_a != char_b),
            ))
        previous = current
    return previous[-1]


def levenshtein_tokens(a: list[str], b: list[str]) -> int:
    """Token-level Levenshtein distance with C acceleration if available."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    if _c_levenshtein is not None:
        vocab: dict[str, str] = {}
        idx = 1
        seq_a = []
        for t in a:
            if t not in vocab:
                vocab[t] = chr(idx)
                idx += 1
            seq_a.append(vocab[t])
        seq_b = []
        for t in b:
            if t not in vocab:
                vocab[t] = chr(idx)
                idx += 1
            seq_b.append(vocab[t])
        return _c_levenshtein.distance("".join(seq_a), "".join(seq_b))

    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, token_a in enumerate(a, start=1):
        current = [i]
        for j, token_b in enumerate(b, start=1):
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (token_a != token_b),
            ))
        previous = current
    return previous[-1]


# ---------------------------------------------------------------------------
# Sample-level metrics
# ---------------------------------------------------------------------------

def cer(prediction: str, target: str) -> float:
    """Character Error Rate."""
    if not target:
        return 0.0 if not prediction else 1.0
    return levenshtein(prediction, target) / len(target)


def wer(prediction: str, target: str) -> float:
    """Word Error Rate."""
    pred_tokens = prediction.split()
    target_tokens = target.split()
    if not target_tokens:
        return 0.0 if not pred_tokens else 1.0
    return levenshtein_tokens(pred_tokens, target_tokens) / len(target_tokens)


def spacing_boundaries(text: str) -> set[int]:
    """Return compact-character offsets where a space appears after a token."""
    boundaries: set[int] = set()
    offset = 0
    tokens = text.split()
    for token in tokens[:-1]:
        offset += len(compact_key(token))
        boundaries.add(offset)
    return boundaries


def boundary_counts(prediction: str, target: str) -> tuple[int, int, int]:
    """Return (true_positive, predicted, gold) boundary counts."""
    predicted = spacing_boundaries(prediction)
    gold = spacing_boundaries(target)
    return len(predicted & gold), len(predicted), len(gold)


def diacritic_accuracy(prediction: str, target: str) -> float:
    """Character-level accuracy when accent-stripped forms align."""
    pred_chars = prediction.replace(" ", "")
    target_chars = target.replace(" ", "")
    if not target_chars:
        return 1.0 if not pred_chars else 0.0
    if strip_accents(pred_chars) != strip_accents(target_chars):
        return 0.0
    matches = sum(1 for p, t in zip(pred_chars, target_chars) if p == t)
    return matches / len(target_chars)


# ---------------------------------------------------------------------------
# Corpus-level metric accumulator
# ---------------------------------------------------------------------------

@dataclass
class MetricTotals:
    """Accumulates metrics across all evaluated samples."""
    count: int = 0
    exact: int = 0
    cer_sum: float = 0.0
    wer_sum: float = 0.0
    char_distance: int = 0
    char_total: int = 0
    word_distance: int = 0
    word_total: int = 0
    diacritic_accuracy_sum: float = 0.0
    boundary_true_positive: int = 0
    boundary_predicted: int = 0
    boundary_gold: int = 0

    def update(self, prediction: str, target: str) -> None:
        self.count += 1
        self.exact += int(prediction == target)

        char_dist = levenshtein(prediction, target)
        target_len = len(target)
        self.cer_sum += char_dist / target_len if target_len else (0.0 if not prediction else 1.0)

        pred_tokens = prediction.split()
        target_tokens = target.split()
        word_dist = levenshtein_tokens(pred_tokens, target_tokens)
        target_token_count = len(target_tokens)
        self.wer_sum += (word_dist / target_token_count if target_token_count
                         else (0.0 if not pred_tokens else 1.0))

        self.diacritic_accuracy_sum += diacritic_accuracy(prediction, target)

        tp, pred_cnt, gold_cnt = boundary_counts(prediction, target)
        self.boundary_true_positive += tp
        self.boundary_predicted += pred_cnt
        self.boundary_gold += gold_cnt

        self.char_distance += char_dist
        self.char_total += target_len
        self.word_distance += word_dist
        self.word_total += target_token_count

    def as_dict(self) -> dict[str, float | int]:
        if self.count == 0:
            return {
                "count": 0, "exact_match": 0.0, "cer": 0.0, "wer": 0.0,
                "diacritic_accuracy": 0.0, "corpus_cer": 0.0, "corpus_wer": 0.0,
                "boundary_precision": 0.0, "boundary_recall": 0.0, "boundary_f1": 0.0,
            }
        bp = self.boundary_true_positive / self.boundary_predicted if self.boundary_predicted else 0.0
        br = self.boundary_true_positive / self.boundary_gold if self.boundary_gold else 0.0
        bf1 = 2 * bp * br / (bp + br) if (bp + br) else 0.0
        return {
            "count": self.count,
            "exact_match": round(self.exact / self.count, 6),
            "cer": round(self.cer_sum / self.count, 6),
            "wer": round(self.wer_sum / self.count, 6),
            "diacritic_accuracy": round(self.diacritic_accuracy_sum / self.count, 6),
            "corpus_cer": round(self.char_distance / self.char_total, 6) if self.char_total else 0.0,
            "corpus_wer": round(self.word_distance / self.word_total, 6) if self.word_total else 0.0,
            "boundary_precision": round(bp, 6),
            "boundary_recall": round(br, 6),
            "boundary_f1": round(bf1, 6),
        }
