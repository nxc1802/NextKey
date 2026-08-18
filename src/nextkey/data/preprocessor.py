"""Text normalization and preprocessing for Vietnamese compact-text restoration."""

from __future__ import annotations

import html
import re
import unicodedata
from typing import Optional


BRACE_PATTERN = re.compile(r"[{}]")
PUNCT_SPACE_PATTERN = re.compile(r"[\W_]+", re.UNICODE)
WHITESPACE_PATTERN = re.compile(r"\s+")
MVP_TARGET_SEPARATOR_PATTERN = re.compile(r"[^\wÀ-ỹ]+", re.UNICODE)


def strip_accents(text: str) -> str:
    """Remove Vietnamese diacritics, map đ→d and Đ→D."""
    decomposed = unicodedata.normalize("NFD", text)
    without_marks = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", without_marks).replace("đ", "d").replace("Đ", "D")


def compact_key(text: str) -> str:
    """Canonical content key: accent-free, space-free, lowercase."""
    text = html.unescape(BRACE_PATTERN.sub("", text))
    return PUNCT_SPACE_PATTERN.sub("", strip_accents(text).lower())


def normalize_target(text: str) -> str:
    """Normalize a target string: NFC, collapse whitespace, lowercase."""
    text = unicodedata.normalize("NFC", html.unescape(text))
    return WHITESPACE_PATTERN.sub(" ", MVP_TARGET_SEPARATOR_PATTERN.sub(" ", text)).strip().lower()


def extract_char_and_boundary_targets(target: str) -> tuple[str, list[int]]:
    """Parse a spaced target into (chars_no_spaces, boundary_flags).

    Example:
        "tôi đang học" → ("tôiđanghọc", [0, 0, 0, 1, 0, 0, 0, 1, 0, 0])

    boundary_flags[i] == 1 means a space should be inserted *before* character i.
    """
    chars: list[str] = []
    boundaries: list[int] = []
    pending_space = False
    for char in normalize_target(target):
        if char.isspace():
            pending_space = bool(chars)
        else:
            chars.append(char)
            boundaries.append(int(pending_space))
            pending_space = False
    if boundaries:
        boundaries[0] = 0
    return "".join(chars), boundaries


def is_valid_alignment(source: str, char_target: str) -> bool:
    """Check whether compact source aligns with accent-stripped char_target."""
    return bool(source) and source == strip_accents(char_target)
