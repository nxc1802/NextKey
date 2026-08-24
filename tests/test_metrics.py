from __future__ import annotations

import pytest

from nextkey.utils.metrics import (
    MetricTotals,
    TriMetricTotals,
    boundary_counts,
    cer,
    compute_lcs,
    diacritic_accuracy,
    levenshtein,
    sentence_bleu,
    sentence_rouge_l,
    wer,
    word_diacritic_accuracy,
    word_overlap_metrics,
)


def test_levenshtein():
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("abc", "abc") == 0
    assert levenshtein("", "abc") == 3


def test_cer():
    assert cer("hôm nay", "hôm nay") == 0.0
    assert cer("hom nay", "hôm nay") == 1 / 7


def test_wer():
    assert wer("hôm nay trời đẹp", "hôm nay trời đẹp") == 0.0
    assert wer("hôm qua trời đẹp", "hôm nay trời đẹp") == 1 / 4


def test_word_overlap_metrics():
    prec, rec, f1 = word_overlap_metrics("tôi đi học bài", "tôi đi học")
    assert prec == 3 / 4
    assert rec == 3 / 3
    assert round(f1, 4) == round(2 * 0.75 * 1.0 / 1.75, 4)


def test_word_diacritic_accuracy():
    assert word_diacritic_accuracy("tôi đi học", "tôi đi học") == 1.0
    assert word_diacritic_accuracy("toi đi học", "tôi đi học") == 2 / 3


def test_compute_lcs():
    assert compute_lcs(["a", "b", "c", "d"], ["a", "c", "d"]) == 3
    assert compute_lcs(["a", "b"], ["c", "d"]) == 0


def test_sentence_rouge_l():
    prec, rec, f1 = sentence_rouge_l("tôi đang học bài", "tôi đang học bài")
    assert prec == 1.0
    assert rec == 1.0
    assert f1 == 1.0


def test_sentence_bleu():
    bleu = sentence_bleu("tôi đang học bài", "tôi đang học bài")
    assert bleu["bleu_1"] == 1.0
    assert bleu["bleu_2"] == 1.0
    assert bleu["bleu_4"] == 1.0


def test_boundary_counts():
    tp, pred_cnt, gold_cnt = boundary_counts("tôi đang học", "tôi đang học")
    assert tp == 2
    assert pred_cnt == 2
    assert gold_cnt == 2


def test_diacritic_accuracy():
    assert diacritic_accuracy("tôi đang học", "tôi đang học") == 1.0
    assert diacritic_accuracy("toi dang hoc", "tôi đang học") < 1.0


def test_metric_totals_multi_level():
    totals = MetricTotals()
    totals.update("tôi đang học bài", "tôi đang học bài")
    totals.update("tôi đag học bài", "tôi đang học bài")
    metrics = totals.as_dict()

    # Character-level
    assert "corpus_cer" in metrics
    assert "macro_cer" in metrics
    assert "diacritic_accuracy" in metrics
    assert "boundary_f1" in metrics

    # Word-level
    assert "corpus_wer" in metrics
    assert "word_accuracy" in metrics
    assert "word_f1" in metrics
    assert "word_diacritic_accuracy" in metrics

    # Sentence-level
    assert "exact_match" in metrics
    assert "sentence_near_perfect_5pct" in metrics
    assert "sentence_near_perfect_10pct" in metrics
    assert "bleu_1" in metrics
    assert "bleu_4" in metrics
    assert "rouge_l_f1" in metrics


def test_tri_metric_totals_multi_level():
    tri = TriMetricTotals()
    tri.update_tri(
        source="toidaghocbai",
        pred_base="toidanghocbai",
        gold_base="toidanghocbai",
        pred_diac="tôi đang học bài",
        gold_diac="tôi đang học bài",
        pred_boundaries=[0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
        gold_boundaries=[0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
        final_prediction="tôi đang học bài",
        gold_sentence="tôi đang học bài",
    )
    res = tri.as_dict()

    assert res["exact_match"] == 1.0
    assert res["corpus_cer"] == 0.0
    assert res["word_accuracy"] == 1.0
    assert res["word_f1"] == 1.0
    assert res["sentence_near_perfect_5pct"] == 1.0
    assert res["bleu_1"] == 1.0
    assert res["rouge_l_f1"] == 1.0
