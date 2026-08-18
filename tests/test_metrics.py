from __future__ import annotations

import pytest

from nextkey.utils.metrics import (
    MetricTotals,
    boundary_counts,
    cer,
    diacritic_accuracy,
    levenshtein,
    wer,
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


def test_boundary_counts():
    tp, pred_cnt, gold_cnt = boundary_counts("tôi đang học", "tôi đang học")
    assert tp == 2
    assert pred_cnt == 2
    assert gold_cnt == 2


def test_diacritic_accuracy():
    assert diacritic_accuracy("tôi đang học", "tôi đang học") == 1.0
    assert diacritic_accuracy("toi dang hoc", "tôi đang học") < 1.0


def test_metric_totals():
    totals = MetricTotals()
    totals.update("tôi đang học", "tôi đang học")
    metrics = totals.as_dict()

    assert metrics["exact_match"] == 1.0
    assert metrics["cer"] == 0.0
    assert metrics["wer"] == 0.0
    assert metrics["boundary_f1"] == 1.0
    assert metrics["diacritic_accuracy"] == 1.0
