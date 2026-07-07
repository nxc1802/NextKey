"""Lexicon dynamic-programming baseline for the narrowed MVP."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nextkey.data.mvp_dataset import compact_key


@dataclass
class LexiconBaseline:
    key_to_word: dict[str, str]
    key_to_score: dict[str, float]
    max_key_len: int
    unknown_penalty: float = -18.0

    def predict(self, text: str) -> str:
        key = compact_key(text)
        if not key:
            return ""

        n = len(key)
        scores = [-math.inf] * (n + 1)
        back: list[tuple[int, str] | None] = [None] * (n + 1)
        scores[0] = 0.0

        for i in range(n):
            if scores[i] == -math.inf:
                continue

            upper = min(n, i + self.max_key_len)
            for j in range(i + 1, upper + 1):
                piece = key[i:j]
                word = self.key_to_word.get(piece)
                if word is None:
                    continue
                score = scores[i] + self.key_to_score[piece]
                if score > scores[j]:
                    scores[j] = score
                    back[j] = (i, word)

            # Unknown fallback consumes one character so inference always finishes.
            fallback_score = scores[i] + self.unknown_penalty
            if fallback_score > scores[i + 1]:
                scores[i + 1] = fallback_score
                back[i + 1] = (i, key[i : i + 1])

        words: list[str] = []
        cursor = n
        while cursor > 0:
            item = back[cursor]
            if item is None:
                words.append(key[cursor - 1 : cursor])
                cursor -= 1
                continue
            prev, word = item
            words.append(word)
            cursor = prev
        return " ".join(reversed(words))

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": "mvp_lexicon_dp",
            "max_key_len": self.max_key_len,
            "unknown_penalty": self.unknown_penalty,
            "key_to_word": self.key_to_word,
            "key_to_score": self.key_to_score,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LexiconBaseline":
        return cls(
            key_to_word=dict(payload["key_to_word"]),
            key_to_score={key: float(value) for key, value in payload["key_to_score"].items()},
            max_key_len=int(payload["max_key_len"]),
            unknown_penalty=float(payload.get("unknown_penalty", -18.0)),
        )


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def train_lexicon_baseline(train_path: Path, min_count: int = 1) -> tuple[LexiconBaseline, dict[str, Any]]:
    word_counts: Counter[str] = Counter()
    key_word_counts: dict[str, Counter[str]] = defaultdict(Counter)
    rows = 0

    for row in iter_jsonl(train_path):
        rows += 1
        for word in row["target"].split():
            key = compact_key(word)
            if not key:
                continue
            word_counts[word] += 1
            key_word_counts[key][word] += 1

    total_words = sum(word_counts.values())
    key_to_word: dict[str, str] = {}
    key_to_score: dict[str, float] = {}

    for key, candidates in key_word_counts.items():
        word, count = candidates.most_common(1)[0]
        if count < min_count:
            continue
        key_to_word[key] = word
        # Frequency score plus a small length preference reduces over-segmentation.
        key_to_score[key] = math.log(count / total_words) + 0.18 * len(key)

    model = LexiconBaseline(
        key_to_word=key_to_word,
        key_to_score=key_to_score,
        max_key_len=max((len(key) for key in key_to_word), default=1),
    )
    stats = {
        "train_path": str(train_path),
        "rows": rows,
        "total_words": total_words,
        "unique_words": len(word_counts),
        "unique_compact_keys": len(key_to_word),
        "max_key_len": model.max_key_len,
        "min_count": min_count,
    }
    return model, stats


def save_model(path: Path, model: LexiconBaseline, stats: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = model.to_dict()
    payload["training_stats"] = stats
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def load_model(path: Path) -> LexiconBaseline:
    return LexiconBaseline.from_dict(json.loads(path.read_text(encoding="utf-8")))
