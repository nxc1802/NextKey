"""Data pipeline: dataset loading, tokenization, and preprocessing."""

from nextkey.data.dataset import AlignedExample, load_examples, iter_batches
from nextkey.data.tokenizer import CharVocab, build_vocab_from_examples
from nextkey.data.preprocessor import strip_accents, compact_key, normalize_target

__all__ = [
    "AlignedExample", "load_examples", "iter_batches",
    "CharVocab", "build_vocab_from_examples",
    "strip_accents", "compact_key", "normalize_target",
]
