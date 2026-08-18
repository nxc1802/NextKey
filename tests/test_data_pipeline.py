from __future__ import annotations

import pytest
import torch

from nextkey.data.preprocessor import (
    compact_key,
    extract_char_and_boundary_targets,
    is_valid_alignment,
    normalize_target,
    strip_accents,
)
from nextkey.data.tokenizer import CharVocab, build_vocab_from_examples
from nextkey.data.dataset import AlignedExample, iter_batches, load_examples, pad_sequences


def test_strip_accents():
    assert strip_accents("Tôi đang học tiếng Việt") == "Toi dang hoc tieng Viet"
    assert strip_accents("đường đời") == "duong doi"
    assert strip_accents("Đà Nẵng") == "Da Nang"


def test_compact_key():
    assert compact_key("Tôi đang học!") == "toidanghoc"
    assert compact_key("{hôm} nay") == "homnay"


def test_normalize_target():
    assert normalize_target("Tôi  đang   học.") == "tôi đang học"


def test_extract_char_and_boundary_targets():
    chars, boundaries = extract_char_and_boundary_targets("Tôi đang học")
    assert chars == "tôiđanghọc"
    # boundary before index 3 ('đ') and index 7 ('h')
    assert boundaries == [0, 0, 0, 1, 0, 0, 0, 1, 0, 0]


def test_is_valid_alignment():
    assert is_valid_alignment("toidanghoc", "tôiđanghọc") is True
    assert is_valid_alignment("toidanghoc", "tôiđang") is False


def test_char_vocab_encode_decode():
    examples = [
        AlignedExample("toidanghoc", "tôiđanghọc", [0, 0, 0, 1, 0, 0, 0, 1, 0, 0], "alpha")
    ]
    vocab = build_vocab_from_examples(examples)
    input_ids = vocab.encode_input("toidanghoc")
    target_ids = vocab.encode_target("tôiđanghọc")

    assert len(input_ids) == 10
    assert len(target_ids) == 10

    decoded = vocab.decode(target_ids, [0, 0, 0, 1, 0, 0, 0, 1, 0, 0])
    assert decoded == "tôi đang học"


def test_pad_sequences():
    seqs = [[1, 2, 3], [4, 5]]
    padded = pad_sequences(seqs, pad_id=0, pad_to_multiple_of=4)
    assert padded.shape == (2, 4)
    assert padded[0].tolist() == [1, 2, 3, 0]
    assert padded[1].tolist() == [4, 5, 0, 0]


def test_iter_batches():
    examples = [
        AlignedExample("homnay", "hômnay", [0, 0, 0, 1, 0, 0], "domain1"),
        AlignedExample("troidep", "trờiđẹp", [0, 0, 0, 0, 1, 0, 0], "domain2"),
    ]
    vocab = build_vocab_from_examples(examples)
    batches = list(iter_batches(examples, vocab, batch_size=2))

    assert len(batches) == 1
    source, targets, boundaries, lengths, chunk = batches[0]
    assert source.shape[0] == 2
    assert targets.shape[0] == 2
    assert boundaries.shape[0] == 2
    assert len(lengths) == 2
    assert len(chunk) == 2
