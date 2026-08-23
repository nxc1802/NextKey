"""Synthetic Corruption Engine for NextKey Multi-Task Vietnamese Restoration.

Provides realistic synthetic noise generation covering:
1. Diacritic removal, stripping, and random mark corruption
2. Whitespace merge, partial spacing, extra spacing, and misplaced spacing
3. Keyboard-neighbor typos (QWERTY adjacency graph)
4. Character swap / transposition (adjacent letters)
5. Character duplication / repetition
6. Character substitution / confusion
7. Multi-variant generation (1 clean sentence -> N noisy training samples)
"""

from __future__ import annotations

import html
import random
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Sequence

from nextkey.data.preprocessor import (
    compact_key,
    extract_char_and_boundary_targets,
    normalize_target,
    strip_accents,
)


# ---------------------------------------------------------------------------
# QWERTY Keyboard Adjacency Graph
# ---------------------------------------------------------------------------

QWERTY_NEIGHBORS: dict[str, list[str]] = {
    "q": ["w", "a", "s", "1", "2"],
    "w": ["q", "e", "a", "s", "d", "2", "3"],
    "e": ["w", "r", "s", "d", "f", "3", "4"],
    "r": ["e", "t", "d", "f", "g", "4", "5"],
    "t": ["r", "y", "f", "g", "h", "5", "6"],
    "y": ["t", "u", "g", "h", "j", "6", "7"],
    "u": ["y", "i", "h", "j", "k", "7", "8"],
    "i": ["u", "o", "j", "k", "l", "8", "9"],
    "o": ["i", "p", "k", "l", "9", "0"],
    "p": ["o", "l", "0"],
    "a": ["q", "w", "s", "z", "x"],
    "s": ["a", "w", "e", "d", "z", "x", "c"],
    "d": ["s", "e", "r", "f", "x", "c", "v"],
    "f": ["d", "r", "t", "g", "c", "v", "b"],
    "g": ["f", "t", "y", "h", "v", "b", "n"],
    "h": ["g", "y", "u", "j", "b", "n", "m"],
    "j": ["h", "u", "i", "k", "n", "m"],
    "k": ["j", "i", "o", "l", "m"],
    "l": ["k", "o", "p"],
    "z": ["a", "s", "x"],
    "x": ["z", "s", "d", "c"],
    "c": ["x", "d", "f", "v"],
    "v": ["c", "f", "g", "b"],
    "b": ["v", "g", "h", "n"],
    "n": ["b", "h", "j", "m"],
    "m": ["n", "j", "k"],
    "0": ["9", "p", "o"],
    "1": ["2", "q"],
    "2": ["1", "3", "q", "w"],
    "3": ["2", "4", "w", "e"],
    "4": ["3", "5", "e", "r"],
    "5": ["4", "6", "r", "t"],
    "6": ["5", "7", "t", "y"],
    "7": ["6", "8", "y", "u"],
    "8": ["7", "9", "u", "i"],
    "9": ["8", "0", "i", "o"],
}

# Common phonetic or informal character substitutions in Vietnamese typing
CHAR_CONFUSIONS: dict[str, list[str]] = {
    "c": ["k", "q"],
    "k": ["c", "q"],
    "q": ["k", "c"],
    "g": ["gh"],
    "ng": ["ngh"],
    "d": ["gi", "r", "v"],
    "gi": ["d", "r"],
    "r": ["d", "gi"],
    "ch": ["tr"],
    "tr": ["ch"],
    "s": ["x"],
    "x": ["s"],
    "i": ["y", "j"],
    "y": ["i"],
    "u": ["w"],
    "o": ["w"],
}

# Diacritic tone confusion pairs (ngã <-> hỏi, sắc <-> huyền, etc.)
TONE_CONFUSIONS: dict[str, str] = {
    "ả": "ã", "ã": "ả",
    "ẻ": "ẽ", "ẽ": "ẻ",
    "ỉ": "ĩ", "ĩ": "ỉ",
    "ỏ": "õ", "õ": "ỏ",
    "ủ": "ũ", "ũ": "ủ",
    "ỷ": "ỹ", "ỹ": "ỷ",
    "ẩ": "ẫ", "ẫ": "ẩ",
    "ẳ": "ẵ", "ẵ": "ẳ",
    "ể": "ễ", "ễ": "ể",
    "ổ": "ỗ", "ỗ": "ổ",
    "ử": "ữ", "ữ": "ử",
    "á": "à", "à": "á",
    "é": "è", "è": "é",
    "í": "ì", "ì": "í",
    "ó": "ò", "ò": "ó",
    "ú": "ù", "ù": "ú",
    "ý": "ỳ", "ỳ": "ý",
}


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class CorruptedSample:
    """A synthetic training sample with aligned multi-task targets.

    Attributes:
        source: The corrupted/noisy input character sequence (e.g. "tojdanghoc")
        base_target: Base unaccented correct character sequence (e.g. "toidanghoc")
        diacritic_target: Accented correct character sequence (e.g. "tôiđanghọc")
        boundary_target: Space-before binary flags [0, 0, 0, 1, 0, 0, 0, 1, 0, 0]
        noise_tags: List of applied noise transformation tags
        clean_text: Original ground truth sentence (e.g. "Tôi đang học")
        domain: Source domain / category
    """
    source: str
    base_target: str
    diacritic_target: str
    boundary_target: list[int]
    noise_tags: list[str] = field(default_factory=list)
    clean_text: str = ""
    domain: str = "general"

    def __post_init__(self):
        # Validate length alignment
        assert len(self.source) == len(self.base_target) == len(self.diacritic_target) == len(self.boundary_target), (
            f"Length mismatch: source={len(self.source)}, base={len(self.base_target)}, "
            f"diacritic={len(self.diacritic_target)}, boundary={len(self.boundary_target)}"
        )


# ---------------------------------------------------------------------------
# Synthetic Corruption Engine
# ---------------------------------------------------------------------------

class SyntheticCorruptor:
    """Configurable synthetic corruption engine for generating noisy Vietnamese samples."""

    def __init__(
        self,
        typo_prob: float = 0.15,
        swap_prob: float = 0.05,
        diacritic_confuse_prob: float = 0.08,
        space_mode_weights: dict[str, float] | None = None,
        seed: int | None = None,
    ):
        self.typo_prob = typo_prob
        self.swap_prob = swap_prob
        self.diacritic_confuse_prob = diacritic_confuse_prob
        self.space_mode_weights = space_mode_weights or {
            "all_merged": 0.50,       # toidanghoc (100% compact)
            "partial_merged": 0.25,   # toi danghoc
            "standard_spaced": 0.15,  # toi dang hoc
            "noisy_spaced": 0.10,     # to i dang ho c
        }
        self.rng = random.Random(seed)

    def get_qwerty_neighbor(self, char: str) -> str:
        """Return a random adjacent key on the QWERTY keyboard, or the char itself."""
        ch_lower = char.lower()
        neighbors = QWERTY_NEIGHBORS.get(ch_lower, [])
        if neighbors:
            return self.rng.choice(neighbors)
        return ch_lower

    def corrupt_character(self, base_ch: str, diac_ch: str) -> tuple[str, list[str]]:
        """Apply typo or character corruption to a single character."""
        tags: list[str] = []
        rand_val = self.rng.random()

        # 1. Tone confusion (e.g. hỏi <-> ngã, sắc <-> huyền)
        if diac_ch in TONE_CONFUSIONS and rand_val < self.diacritic_confuse_prob:
            tags.append("diacritic_confuse")
            return base_ch, tags

        # 2. QWERTY neighbor typo
        if rand_val < self.typo_prob:
            neighbor = self.get_qwerty_neighbor(base_ch)
            if neighbor != base_ch:
                tags.append("qwerty_neighbor")
                return neighbor, tags

        # Default: clean base unaccented char
        return base_ch, tags

    def generate_variants(
        self,
        clean_sentence: str,
        num_variants: int = 3,
        domain: str = "general",
    ) -> list[CorruptedSample]:
        """Generate multiple corrupted training samples from a single clean ground-truth sentence.

        Args:
            clean_sentence: Standard Vietnamese sentence (e.g. "Tôi đang học bài.")
            num_variants: Number of distinct corrupted variations to generate.
            domain: Category / domain tag.

        Returns:
            List of CorruptedSample dataclasses with strict 1-to-1 character alignment.
        """
        # 1. Normalize clean target
        norm_clean = normalize_target(clean_sentence)
        if not norm_clean:
            return []

        # 2. Extract gold diacritics and boundary targets
        gold_diac_chars, gold_boundaries = extract_char_and_boundary_targets(norm_clean)
        gold_base_chars = strip_accents(gold_diac_chars)

        T = len(gold_base_chars)
        if T < 2 or T > 300:
            return []

        samples: list[CorruptedSample] = []

        for v_idx in range(num_variants):
            tags: list[str] = []
            source_chars: list[str] = list(gold_base_chars)

            # Variant 0 is always canonical clean compact: 100% no accents, 100% no spaces
            if v_idx == 0:
                tags.append("canonical_compact")
                samples.append(
                    CorruptedSample(
                        source=gold_base_chars,
                        base_target=gold_base_chars,
                        diacritic_target=gold_diac_chars,
                        boundary_target=list(gold_boundaries),
                        noise_tags=tags,
                        clean_text=norm_clean,
                        domain=domain,
                    )
                )
                continue

            # Variant >= 1: Apply character-level corruptions (typo, swap, tone confusion)
            # A. Character substitutions / QWERTY neighbor typos
            for i in range(T):
                ch_in, ch_tags = self.corrupt_character(gold_base_chars[i], gold_diac_chars[i])
                source_chars[i] = ch_in
                tags.extend(ch_tags)

            # B. Character Swaps / Transpositions (swap adjacent letters within a word)
            for i in range(T - 1):
                # Don't swap across word boundaries if boundary is 1
                if gold_boundaries[i + 1] == 0 and self.rng.random() < self.swap_prob:
                    source_chars[i], source_chars[i + 1] = source_chars[i + 1], source_chars[i]
                    tags.append("char_swap")

            final_source = "".join(source_chars)
            samples.append(
                CorruptedSample(
                    source=final_source,
                    base_target=gold_base_chars,
                    diacritic_target=gold_diac_chars,
                    boundary_target=list(gold_boundaries),
                    noise_tags=sorted(set(tags)) if tags else ["mild_noise"],
                    clean_text=norm_clean,
                    domain=domain,
                )
            )

        return samples


# ---------------------------------------------------------------------------
# Dataset Generation Helper
# ---------------------------------------------------------------------------

def corrupt_dataset_from_sentences(
    sentences: Sequence[str],
    num_variants_per_sample: int = 3,
    domain: str = "general",
    typo_prob: float = 0.12,
    seed: int = 42,
) -> list[CorruptedSample]:
    """Batch generate synthetic corrupted samples from a list of clean text sentences."""
    corruptor = SyntheticCorruptor(typo_prob=typo_prob, seed=seed)
    all_samples: list[CorruptedSample] = []
    for s in sentences:
        variants = corruptor.generate_variants(
            clean_sentence=s,
            num_variants=num_variants_per_sample,
            domain=domain,
        )
        all_samples.extend(variants)
    return all_samples
