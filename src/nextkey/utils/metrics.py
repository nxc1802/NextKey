"""Evaluation metrics for Vietnamese compact-text restoration.

Provides comprehensive metrics across 3 levels:
1. Character-Level: CER (macro/corpus), Character Diacritic Acc, Boundary F1, Typo Recovery Rate.
2. Word-Level: WER (macro/corpus), Word Accuracy, Word Precision/Recall/F1, Word Diacritic Acc, Word Typo Recovery.
3. Sentence-Level: Sentence Exact Match (EM), Sentence Near-Perfect (CER <= 5%, CER <= 10%),
   Sentence Error-Free Diacritics/Spacing, BLEU (BLEU-1, BLEU-2, BLEU-4), and ROUGE-L (F1/P/R).
"""

from __future__ import annotations

import math
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Text normalization helpers
# ---------------------------------------------------------------------------

def strip_accents(text: str) -> str:
    """Remove Vietnamese diacritics while preserving d/D mapping."""
    decomposed = unicodedata.normalize("NFD", text)
    without_marks = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", without_marks).replace("đ", "d").replace("Đ", "D")


def compact_key(text: str) -> str:
    """Produce a canonical accent-free, space-free, lowercase key."""
    import re
    punct_space = re.compile(r"[\W_]+", re.UNICODE)
    return punct_space.sub("", strip_accents(text).lower())


# ---------------------------------------------------------------------------
# Edit distance (Character and Token level)
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
# Character-level metrics
# ---------------------------------------------------------------------------

def cer(prediction: str, target: str) -> float:
    """Character Error Rate."""
    if not target:
        return 0.0 if not prediction else 1.0
    return levenshtein(prediction, target) / len(target)


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
# Word-level metrics
# ---------------------------------------------------------------------------

def wer(prediction: str, target: str) -> float:
    """Word Error Rate via token edit distance."""
    pred_tokens = prediction.split()
    target_tokens = target.split()
    if not target_tokens:
        return 0.0 if not pred_tokens else 1.0
    return levenshtein_tokens(pred_tokens, target_tokens) / len(target_tokens)


def word_overlap_metrics(prediction: str, target: str) -> tuple[float, float, float]:
    """Compute Word-level Precision, Recall, and F1 based on token multiset overlap.
    
    Returns:
        (precision, recall, f1)
    """
    pred_tokens = prediction.split()
    gold_tokens = target.split()
    if not gold_tokens:
        return (1.0, 1.0, 1.0) if not pred_tokens else (0.0, 0.0, 0.0)
    if not pred_tokens:
        return 0.0, 0.0, 0.0

    pred_counts = Counter(pred_tokens)
    gold_counts = Counter(gold_tokens)

    overlap = sum(min(count, gold_counts[token]) for token, count in pred_counts.items())
    precision = overlap / len(pred_tokens) if pred_tokens else 0.0
    recall = overlap / len(gold_tokens) if gold_tokens else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def word_diacritic_accuracy(prediction: str, target: str) -> float:
    """Percentage of aligned words where all diacritics are 100% correct."""
    pred_tokens = prediction.split()
    gold_tokens = target.split()
    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0
    if len(pred_tokens) != len(gold_tokens):
        # Fallback to multiset match
        correct_words = 0
        gold_counts = Counter(gold_tokens)
        pred_counts = Counter(pred_tokens)
        for w, cnt in pred_counts.items():
            correct_words += min(cnt, gold_counts[w])
        return correct_words / len(gold_tokens)

    correct = sum(1 for p, g in zip(pred_tokens, gold_tokens) if p == g)
    return correct / len(gold_tokens)


# ---------------------------------------------------------------------------
# Sentence-level metrics (Exact Match, BLEU, ROUGE-L)
# ---------------------------------------------------------------------------

def compute_lcs(x: list[str], y: list[str]) -> int:
    """Compute length of Longest Common Subsequence between two token sequences."""
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return 0
    dp = [0] * (n + 1)
    for i in range(1, m + 1):
        prev = 0
        for j in range(1, n + 1):
            temp = dp[j]
            if x[i - 1] == y[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def sentence_rouge_l(prediction: str, target: str, beta: float = 1.2) -> tuple[float, float, float]:
    """Compute Sentence-level ROUGE-L (Precision, Recall, F1).
    
    Returns:
        (rouge_precision, rouge_recall, rouge_f1)
    """
    p_tokens = prediction.split()
    g_tokens = target.split()
    if not g_tokens:
        return (1.0, 1.0, 1.0) if not p_tokens else (0.0, 0.0, 0.0)
    if not p_tokens:
        return 0.0, 0.0, 0.0

    lcs_len = compute_lcs(p_tokens, g_tokens)
    prec = lcs_len / len(p_tokens)
    rec = lcs_len / len(g_tokens)
    if prec + rec == 0:
        return 0.0, 0.0, 0.0
    beta_sq = beta ** 2
    f1 = ((1 + beta_sq) * prec * rec) / (beta_sq * prec + rec)
    return prec, rec, f1


def compute_ngram_counts(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    """Extract n-grams and count occurrences."""
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def sentence_bleu(prediction: str, target: str, max_n: int = 4) -> dict[str, float]:
    """Compute Sentence-level BLEU-1, BLEU-2, and BLEU-4 with brevity penalty."""
    p_tokens = prediction.split()
    g_tokens = target.split()
    if not g_tokens:
        return {"bleu_1": 1.0, "bleu_2": 1.0, "bleu_4": 1.0}
    if not p_tokens:
        return {"bleu_1": 0.0, "bleu_2": 0.0, "bleu_4": 0.0}

    # Brevity Penalty (BP)
    p_len = len(p_tokens)
    g_len = len(g_tokens)
    bp = 1.0 if p_len > g_len else math.exp(1.0 - g_len / p_len) if p_len > 0 else 0.0

    precisions: list[float] = []
    for n in range(1, max_n + 1):
        if p_len < n:
            precisions.append(0.0)
            continue
        p_ngrams = compute_ngram_counts(p_tokens, n)
        g_ngrams = compute_ngram_counts(g_tokens, n)
        match_count = sum(min(cnt, g_ngrams[ng]) for ng, cnt in p_ngrams.items())
        total_count = sum(p_ngrams.values())
        precisions.append((match_count + 1e-6) / (total_count + 1e-6))

    bleu_1 = bp * precisions[0]
    bleu_2 = bp * math.exp(0.5 * (math.log(precisions[0]) + math.log(precisions[1]))) if precisions[1] > 0 else 0.0
    
    # BLEU-4
    if all(p > 0 for p in precisions):
        bleu_4 = bp * math.exp(0.25 * sum(math.log(p) for p in precisions))
    else:
        bleu_4 = 0.0

    return {
        "bleu_1": round(bleu_1, 4),
        "bleu_2": round(bleu_2, 4),
        "bleu_4": round(bleu_4, 4),
    }


# ---------------------------------------------------------------------------
# Comprehensive Metric Accumulator (Dual-Task)
# ---------------------------------------------------------------------------

@dataclass
class MetricTotals:
    """Accumulates multi-level metrics (Character, Word, Sentence) across all samples."""
    count: int = 0
    exact: int = 0

    # Character-level
    cer_sum: float = 0.0
    char_distance: int = 0
    char_total: int = 0
    diacritic_accuracy_sum: float = 0.0
    boundary_true_positive: int = 0
    boundary_predicted: int = 0
    boundary_gold: int = 0

    # Word-level
    wer_sum: float = 0.0
    word_distance: int = 0
    word_total: int = 0
    word_precision_sum: float = 0.0
    word_recall_sum: float = 0.0
    word_f1_sum: float = 0.0
    word_diacritic_acc_sum: float = 0.0

    # Sentence-level
    sentence_cer_le_5: int = 0   # Sentences with CER <= 5% (Near-Perfect)
    sentence_cer_le_10: int = 0  # Sentences with CER <= 10% (High-Quality)
    sentence_diac_error_free: int = 0
    sentence_spacing_error_free: int = 0
    bleu_1_sum: float = 0.0
    bleu_2_sum: float = 0.0
    bleu_4_sum: float = 0.0
    rouge_l_f1_sum: float = 0.0

    def update(self, prediction: str, target: str) -> None:
        self.count += 1
        p_clean = prediction.strip()
        t_clean = target.strip()
        is_exact = int(p_clean == t_clean)
        self.exact += is_exact

        # 1. Character-level
        char_dist = levenshtein(prediction, target)
        target_len = len(target)
        s_cer = char_dist / target_len if target_len else (0.0 if not prediction else 1.0)
        self.cer_sum += s_cer
        self.char_distance += char_dist
        self.char_total += target_len

        diac_acc = diacritic_accuracy(prediction, target)
        self.diacritic_accuracy_sum += diac_acc
        if diac_acc == 1.0:
            self.sentence_diac_error_free += 1

        tp, pred_cnt, gold_cnt = boundary_counts(prediction, target)
        self.boundary_true_positive += tp
        self.boundary_predicted += pred_cnt
        self.boundary_gold += gold_cnt
        if tp == pred_cnt == gold_cnt:
            self.sentence_spacing_error_free += 1

        # 2. Word-level
        pred_tokens = prediction.split()
        target_tokens = target.split()
        word_dist = levenshtein_tokens(pred_tokens, target_tokens)
        target_token_count = len(target_tokens)
        self.wer_sum += (word_dist / target_token_count if target_token_count
                         else (0.0 if not pred_tokens else 1.0))
        self.word_distance += word_dist
        self.word_total += target_token_count

        wp, wr, wf1 = word_overlap_metrics(prediction, target)
        self.word_precision_sum += wp
        self.word_recall_sum += wr
        self.word_f1_sum += wf1
        self.word_diacritic_acc_sum += word_diacritic_accuracy(prediction, target)

        # 3. Sentence-level
        if s_cer <= 0.05:
            self.sentence_cer_le_5 += 1
        if s_cer <= 0.10:
            self.sentence_cer_le_10 += 1

        bleu_dict = sentence_bleu(prediction, target)
        self.bleu_1_sum += bleu_dict["bleu_1"]
        self.bleu_2_sum += bleu_dict["bleu_2"]
        self.bleu_4_sum += bleu_dict["bleu_4"]

        _, _, r_f1 = sentence_rouge_l(prediction, target)
        self.rouge_l_f1_sum += r_f1

    def as_dict(self) -> dict[str, float | int]:
        if self.count == 0:
            return {"count": 0}
        n = self.count
        bp = self.boundary_true_positive / self.boundary_predicted if self.boundary_predicted else 0.0
        br = self.boundary_true_positive / self.boundary_gold if self.boundary_gold else 0.0
        bf1 = 2 * bp * br / (bp + br) if (bp + br) else 0.0

        corpus_wer = self.word_distance / self.word_total if self.word_total else 0.0

        return {
            "count": self.count,
            # --- 1. Character-Level Metrics ---
            "corpus_cer": round(self.char_distance / self.char_total, 6) if self.char_total else 0.0,
            "macro_cer": round(self.cer_sum / n, 6),
            "diacritic_accuracy": round(self.diacritic_accuracy_sum / n, 6),
            "boundary_precision": round(bp, 6),
            "boundary_recall": round(br, 6),
            "boundary_f1": round(bf1, 6),
            # Backward-compatible keys:
            "cer": round(self.cer_sum / n, 6),
            "wer": round(self.wer_sum / n, 6),

            # --- 2. Word-Level Metrics ---
            "corpus_wer": round(corpus_wer, 6),
            "macro_wer": round(self.wer_sum / n, 6),
            "word_accuracy": round(1.0 - corpus_wer, 6),
            "word_precision": round(self.word_precision_sum / n, 6),
            "word_recall": round(self.word_recall_sum / n, 6),
            "word_f1": round(self.word_f1_sum / n, 6),
            "word_diacritic_accuracy": round(self.word_diacritic_acc_sum / n, 6),

            # --- 3. Sentence-Level Metrics ---
            "exact_match": round(self.exact / n, 6),
            "sentence_accuracy": round(self.exact / n, 6),
            "sentence_near_perfect_5pct": round(self.sentence_cer_le_5 / n, 6),
            "sentence_near_perfect_10pct": round(self.sentence_cer_le_10 / n, 6),
            "sentence_diac_error_free": round(self.sentence_diac_error_free / n, 6),
            "sentence_spacing_error_free": round(self.sentence_spacing_error_free / n, 6),
            "bleu_1": round(self.bleu_1_sum / n, 4),
            "bleu_2": round(self.bleu_2_sum / n, 4),
            "bleu_4": round(self.bleu_4_sum / n, 4),
            "rouge_l_f1": round(self.rouge_l_f1_sum / n, 4),
        }


# ---------------------------------------------------------------------------
# Comprehensive Metric Accumulator (Tri-Task Multi-Level)
# ---------------------------------------------------------------------------

@dataclass
class TriMetricTotals:
    """Accumulates metrics across 3 tasks: Correction, Diacritics, and Spacing.
    
    Includes Character-level, Word-level, and Sentence-level metrics.
    """
    count: int = 0
    exact: int = 0

    # Character-level
    cer_sum: float = 0.0
    char_distance: int = 0
    char_total: int = 0

    # Task 1: Correction Head (Character level)
    corr_correct: int = 0
    corr_total: int = 0
    typos_introduced: int = 0
    typos_fixed: int = 0

    # Task 2: Diacritic Head
    diac_correct: int = 0
    diac_total: int = 0

    # Task 3: Boundary Head
    boundary_tp: int = 0
    boundary_pred: int = 0
    boundary_gold: int = 0

    # Word-level
    wer_sum: float = 0.0
    word_distance: int = 0
    word_total: int = 0
    word_precision_sum: float = 0.0
    word_recall_sum: float = 0.0
    word_f1_sum: float = 0.0
    word_diacritic_acc_sum: float = 0.0
    word_typos_total: int = 0
    word_typos_restored: int = 0

    # Sentence-level
    sentence_cer_le_5: int = 0
    sentence_cer_le_10: int = 0
    sentence_diac_error_free: int = 0
    sentence_spacing_error_free: int = 0
    bleu_1_sum: float = 0.0
    bleu_2_sum: float = 0.0
    bleu_4_sum: float = 0.0
    rouge_l_f1_sum: float = 0.0

    def update_tri(
        self,
        source: str,
        pred_base: str,
        gold_base: str,
        pred_diac: str,
        gold_diac: str,
        pred_boundaries: list[int],
        gold_boundaries: list[int],
        final_prediction: str,
        gold_sentence: str,
    ) -> None:
        self.count += 1
        p_clean = final_prediction.strip()
        g_clean = gold_sentence.strip()
        is_exact = int(p_clean == g_clean)
        self.exact += is_exact

        # 1. Character-level CER
        c_dist = levenshtein(final_prediction, gold_sentence)
        t_len = len(gold_sentence)
        self.char_distance += c_dist
        self.char_total += t_len
        s_cer = c_dist / t_len if t_len else (0.0 if not final_prediction else 1.0)
        self.cer_sum += s_cer

        # Task 1: Base Character Correction
        for s_ch, p_b, g_b in zip(source, pred_base, gold_base):
            self.corr_total += 1
            if p_b == g_b:
                self.corr_correct += 1
            if s_ch != g_b:
                self.typos_introduced += 1
                if p_b == g_b:
                    self.typos_fixed += 1

        # Task 2: Diacritic Char Accuracy
        sentence_diac_ok = True
        for p_d, g_d in zip(pred_diac, gold_diac):
            self.diac_total += 1
            if p_d == g_d:
                self.diac_correct += 1
            else:
                sentence_diac_ok = False
        if sentence_diac_ok:
            self.sentence_diac_error_free += 1

        # Task 3: Boundary
        sentence_space_ok = (pred_boundaries == gold_boundaries)
        if sentence_space_ok:
            self.sentence_spacing_error_free += 1

        for p_b, g_b in zip(pred_boundaries, gold_boundaries):
            if p_b == 1 and g_b == 1:
                self.boundary_tp += 1
            if p_b == 1:
                self.boundary_pred += 1
            if g_b == 1:
                self.boundary_gold += 1

        # 2. Word-level Metrics
        p_words = final_prediction.split()
        g_words = gold_sentence.split()
        w_dist = levenshtein_tokens(p_words, g_words)
        self.word_distance += w_dist
        self.word_total += len(g_words)
        self.wer_sum += w_dist / len(g_words) if g_words else (0.0 if not p_words else 1.0)

        wp, wr, wf1 = word_overlap_metrics(final_prediction, gold_sentence)
        self.word_precision_sum += wp
        self.word_recall_sum += wr
        self.word_f1_sum += wf1
        self.word_diacritic_acc_sum += word_diacritic_accuracy(final_prediction, gold_sentence)

        # Word typo fix analysis: compare clean source words vs predicted words
        # When len(p_words) == len(g_words), word typo matches
        if len(p_words) == len(g_words):
            for pw, gw in zip(p_words, g_words):
                if strip_accents(pw) != strip_accents(gw):
                    self.word_typos_total += 1
                    if pw == gw:
                        self.word_typos_restored += 1

        # 3. Sentence-level Metrics
        if s_cer <= 0.05:
            self.sentence_cer_le_5 += 1
        if s_cer <= 0.10:
            self.sentence_cer_le_10 += 1

        bleu_dict = sentence_bleu(final_prediction, gold_sentence)
        self.bleu_1_sum += bleu_dict["bleu_1"]
        self.bleu_2_sum += bleu_dict["bleu_2"]
        self.bleu_4_sum += bleu_dict["bleu_4"]

        _, _, r_f1 = sentence_rouge_l(final_prediction, gold_sentence)
        self.rouge_l_f1_sum += r_f1

    def as_dict(self) -> dict[str, float | int]:
        if self.count == 0:
            return {}
        n = self.count
        bp = self.boundary_tp / self.boundary_pred if self.boundary_pred else 0.0
        br = self.boundary_tp / self.boundary_gold if self.boundary_gold else 0.0
        bf1 = 2 * bp * br / (bp + br) if (bp + br) else 0.0
        corr_acc = self.corr_correct / self.corr_total if self.corr_total else 0.0
        typo_rec = self.typos_fixed / self.typos_introduced if self.typos_introduced else 1.0
        diac_acc = self.diac_correct / self.diac_total if self.diac_total else 0.0
        corpus_wer = self.word_distance / self.word_total if self.word_total else 0.0

        return {
            "count": self.count,
            # --- 1. Character-Level Metrics ---
            "corpus_cer": round(self.char_distance / self.char_total, 6) if self.char_total else 0.0,
            "macro_cer": round(self.cer_sum / n, 6),
            "correction_accuracy": round(corr_acc, 6),
            "typo_recovery_rate": round(typo_rec, 6),
            "diacritic_accuracy": round(diac_acc, 6),
            "boundary_precision": round(bp, 6),
            "boundary_recall": round(br, 6),
            "boundary_f1": round(bf1, 6),
            "typos_evaluated": self.typos_introduced,
            "typos_restored": self.typos_fixed,

            # --- 2. Word-Level Metrics ---
            "corpus_wer": round(corpus_wer, 6),
            "macro_wer": round(self.wer_sum / n, 6),
            "word_accuracy": round(1.0 - corpus_wer, 6),
            "word_precision": round(self.word_precision_sum / n, 6),
            "word_recall": round(self.word_recall_sum / n, 6),
            "word_f1": round(self.word_f1_sum / n, 6),
            "word_diacritic_accuracy": round(self.word_diacritic_acc_sum / n, 6),

            # --- 3. Sentence-Level Metrics ---
            "exact_match": round(self.exact / n, 6),
            "sentence_accuracy": round(self.exact / n, 6),
            "sentence_near_perfect_5pct": round(self.sentence_cer_le_5 / n, 6),
            "sentence_near_perfect_10pct": round(self.sentence_cer_le_10 / n, 6),
            "sentence_diac_error_free": round(self.sentence_diac_error_free / n, 6),
            "sentence_spacing_error_free": round(self.sentence_spacing_error_free / n, 6),
            "bleu_1": round(self.bleu_1_sum / n, 4),
            "bleu_2": round(self.bleu_2_sum / n, 4),
            "bleu_4": round(self.bleu_4_sum / n, 4),
            "rouge_l_f1": round(self.rouge_l_f1_sum / n, 4),
        }
