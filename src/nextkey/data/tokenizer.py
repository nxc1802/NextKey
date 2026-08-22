"""Character-level vocabulary for the CharTagger dual-head architecture."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PAD = "<pad>"
UNK = "<unk>"


@dataclass
class CharVocab:
    """Unified character vocabulary for input and target character sets.

    Attributes:
        char_stoi: input character → index mapping
        char_itos: index → input character list
        target_stoi: target (accented) character → index mapping
        target_itos: index → target character list
    """
    char_stoi: dict[str, int]
    char_itos: list[str]
    target_stoi: dict[str, int]
    target_itos: list[str]

    # --- Convenience properties ---

    @property
    def pad_char_id(self) -> int:
        return self.char_stoi[PAD]

    @property
    def unk_char_id(self) -> int:
        return self.char_stoi[UNK]

    @property
    def pad_target_id(self) -> int:
        return self.target_stoi[PAD]

    @property
    def input_vocab_size(self) -> int:
        return len(self.char_itos)

    @property
    def target_vocab_size(self) -> int:
        return len(self.target_itos)

    # --- Encoding / Decoding ---

    def encode_input(self, text: str) -> list[int]:
        return [self.char_stoi.get(ch, self.unk_char_id) for ch in text]

    def encode_target(self, text: str) -> list[int]:
        return [self.target_stoi.get(ch, self.target_stoi[UNK]) for ch in text]

    def decode(self, target_ids: list[int], boundary_preds: list[int]) -> str:
        """Decode target character IDs + boundary predictions → Vietnamese text."""
        chars: list[str] = []
        for idx, target_id in enumerate(target_ids):
            if target_id == self.pad_target_id:
                continue
            char = self.target_itos[target_id]
            if char in {PAD, UNK}:
                continue
            if chars and idx < len(boundary_preds) and boundary_preds[idx]:
                chars.append(" ")
            chars.append(char)
        return "".join(chars)

    # --- Serialization ---

    def to_json(self) -> dict[str, Any]:
        return {
            "char_itos": self.char_itos,
            "target_itos": self.target_itos,
            "format": "nextkey-chartagger-v1",
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "CharVocab":
        char_itos = list(payload["char_itos"])
        target_itos = list(payload["target_itos"])
        return cls(
            char_stoi={t: i for i, t in enumerate(char_itos)},
            char_itos=char_itos,
            target_stoi={t: i for i, t in enumerate(target_itos)},
            target_itos=target_itos,
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharVocab":
        path = Path(path)
        return cls.from_json(json.loads(path.read_text(encoding="utf-8")))


def build_vocab_from_examples(examples: list) -> CharVocab:
    """Build a CharVocab from a list of AlignedExample dataclasses.

    Each example must have `.source` (input chars) and `.char_target` (target chars).
    """
    input_chars = sorted({ch for ex in examples for ch in ex.source})
    target_chars = sorted({ch for ex in examples for ch in ex.char_target})
    char_itos = [PAD, UNK, *input_chars]
    target_itos = [PAD, UNK, *target_chars]
    return CharVocab(
        char_stoi={t: i for i, t in enumerate(char_itos)},
        char_itos=char_itos,
        target_stoi={t: i for i, t in enumerate(target_itos)},
        target_itos=target_itos,
    )
