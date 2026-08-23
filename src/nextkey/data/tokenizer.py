"""Character-level vocabulary for the CharTagger dual-head and tri-head architectures."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PAD = "<pad>"
UNK = "<unk>"


@dataclass
class CharVocab:
    """Unified character vocabulary for input, correction base targets, and diacritic targets.

    Attributes:
        char_stoi: input character → index mapping
        char_itos: index → input character list
        target_stoi: diacritic target (accented) character → index mapping
        target_itos: index → diacritic target character list
        corr_stoi: correction target (base unaccented) character → index mapping
        corr_itos: index → correction target character list
    """
    char_stoi: dict[str, int]
    char_itos: list[str]
    target_stoi: dict[str, int]
    target_itos: list[str]
    corr_stoi: dict[str, int] | None = None
    corr_itos: list[str] | None = None

    def __post_init__(self):
        if self.corr_stoi is None or self.corr_itos is None:
            # Default to char_stoi / char_itos for backwards compatibility
            self.corr_stoi = dict(self.char_stoi)
            self.corr_itos = list(self.char_itos)

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
    def pad_corr_id(self) -> int:
        return self.corr_stoi[PAD]

    @property
    def input_vocab_size(self) -> int:
        return len(self.char_itos)

    @property
    def target_vocab_size(self) -> int:
        return len(self.target_itos)

    @property
    def corr_vocab_size(self) -> int:
        return len(self.corr_itos)

    # --- Encoding / Decoding ---

    def encode_input(self, text: str) -> list[int]:
        return [self.char_stoi.get(ch, self.unk_char_id) for ch in text]

    def encode_target(self, text: str) -> list[int]:
        return [self.target_stoi.get(ch, self.target_stoi[UNK]) for ch in text]

    def encode_corr(self, text: str) -> list[int]:
        return [self.corr_stoi.get(ch, self.corr_stoi[UNK]) for ch in text]

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

    def decode_tri(
        self,
        corr_ids: list[int],
        diac_ids: list[int],
        boundary_preds: list[int],
        use_corr: bool = True,
    ) -> str:
        """Decode 3-head outputs into standard Vietnamese text."""
        chars: list[str] = []
        for idx, (c_id, d_id) in enumerate(zip(corr_ids, diac_ids)):
            if d_id == self.pad_target_id:
                continue
            d_char = self.target_itos[d_id]
            if d_char in {PAD, UNK}:
                continue
            if chars and idx < len(boundary_preds) and boundary_preds[idx]:
                chars.append(" ")
            chars.append(d_char)
        return "".join(chars)

    # --- Serialization ---

    def to_json(self) -> dict[str, Any]:
        return {
            "char_itos": self.char_itos,
            "target_itos": self.target_itos,
            "corr_itos": self.corr_itos,
            "format": "nextkey-trichartagger-v1",
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "CharVocab":
        char_itos = list(payload["char_itos"])
        target_itos = list(payload["target_itos"])
        corr_itos = list(payload.get("corr_itos", char_itos))
        return cls(
            char_stoi={t: i for i, t in enumerate(char_itos)},
            char_itos=char_itos,
            target_stoi={t: i for i, t in enumerate(target_itos)},
            target_itos=target_itos,
            corr_stoi={t: i for i, t in enumerate(corr_itos)},
            corr_itos=corr_itos,
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
    """Build a CharVocab from a list of AlignedExample or CorruptedSample dataclasses."""
    input_chars: set[str] = set()
    target_chars: set[str] = set()
    corr_chars: set[str] = set()

    for ex in examples:
        if hasattr(ex, "source"):
            input_chars.update(ex.source)
        if hasattr(ex, "char_target"):
            target_chars.update(ex.char_target)
        if hasattr(ex, "diacritic_target"):
            target_chars.update(ex.diacritic_target)
        if hasattr(ex, "base_target"):
            corr_chars.update(ex.base_target)

    sorted_in = sorted(input_chars)
    sorted_tgt = sorted(target_chars)
    sorted_corr = sorted(corr_chars) if corr_chars else sorted_in

    char_itos = [PAD, UNK, *sorted_in]
    target_itos = [PAD, UNK, *sorted_tgt]
    corr_itos = [PAD, UNK, *sorted_corr]

    return CharVocab(
        char_stoi={t: i for i, t in enumerate(char_itos)},
        char_itos=char_itos,
        target_stoi={t: i for i, t in enumerate(target_itos)},
        target_itos=target_itos,
        corr_stoi={t: i for i, t in enumerate(corr_itos)},
        corr_itos=corr_itos,
    )
