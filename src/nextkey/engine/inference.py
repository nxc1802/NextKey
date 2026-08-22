"""Inference engine and prediction pipeline for NextKey models."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from nextkey.data.preprocessor import compact_key
from nextkey.data.tokenizer import CharVocab
from nextkey.models.bigru import BiGRUCharTagger


@dataclass
class CharPredictionDetail:
    """Detail for a single predicted character position."""
    index: int
    input_char: str
    predicted_char: str
    boundary_flag: bool
    boundary_prob: float
    diacritic_prob: float


@dataclass
class PredictionResult:
    """Structured result from NextKey inference."""
    request_id: str
    raw_input: str
    compact_input: str
    best_text: str
    candidates: list[dict[str, Any]]
    char_details: list[CharPredictionDetail]
    latency_ms: float
    model_name: str = "NextKey Width-XXXS"
    parameter_count: int = 17828


class NextKeyPredictor:
    """High-performance predictor for Vietnamese compact writing restoration."""

    def __init__(
        self,
        checkpoint_path: str | Path = "artifacts/phase2/width_xxxs/best_model.pt",
        vocab_path: str | Path = "artifacts/phase2/width_xxxs/vocab.json",
        device: str | torch.device | None = None,
        embed_dim: int = 16,
        hidden_dim: int = 32,
        num_layers: int = 1,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.vocab_path = Path(vocab_path)

        if not self.vocab_path.exists():
            raise FileNotFoundError(f"Vocab file not found at: {self.vocab_path}")
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found at: {self.checkpoint_path}")

        # Determine compute device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # Load Vocab
        self.vocab = CharVocab.load(self.vocab_path)

        # Build Model
        self.model = BiGRUCharTagger(
            vocab_size=self.vocab.input_vocab_size,
            num_target_classes=self.vocab.target_vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=0.0,
        )

        # Load Weights
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.num_params = self.model.count_parameters()
        self.checkpoint_size_kb = round(self.checkpoint_path.stat().st_size / 1024, 2)

    @torch.no_grad()
    def restore(
        self,
        text: str,
        top_k: int = 1,
        boundary_threshold: float = 0.5,
    ) -> PredictionResult:
        """Restore compact text into standard Vietnamese.

        Args:
            text: Raw input string (e.g. "toidanghoc", "ha noi thu do", "nguoivietnamyeunuoc").
            top_k: Number of candidate variations to return.
            boundary_threshold: Sigmoid threshold for word boundary insertion.

        Returns:
            PredictionResult containing best text, details, and latency.
        """
        start_time = time.perf_counter()
        req_id = f"req_{uuid.uuid4().hex[:8]}"

        # Normalize to compact input if spaces/accents exist, or use as is
        compact = compact_key(text)
        if not compact:
            compact = text.strip().lower()

        if not compact:
            return PredictionResult(
                request_id=req_id,
                raw_input=text,
                compact_input="",
                best_text="",
                candidates=[{"text": "", "rank": 1, "score": 1.0}],
                char_details=[],
                latency_ms=0.0,
                parameter_count=self.num_params,
            )

        # Prepare batch tensor
        input_ids = self.vocab.encode_input(compact)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        lengths = torch.tensor([len(input_ids)], dtype=torch.long)

        # Run forward pass
        outputs = self.model(input_tensor, lengths)
        diacritic_logits = outputs["diacritic_logits"].squeeze(0)  # [T, C]
        boundary_logits = outputs["boundary_logits"].squeeze(0)    # [T]

        # Probabilities
        diacritic_probs = F.softmax(diacritic_logits, dim=-1)
        boundary_probs = torch.sigmoid(boundary_logits)

        best_char_ids = diacritic_probs.argmax(dim=-1).cpu().tolist()
        best_char_probs = diacritic_probs.max(dim=-1).values.cpu().tolist()
        boundary_prob_list = boundary_probs.cpu().tolist()
        boundary_flags = [p >= boundary_threshold for p in boundary_prob_list]
        if boundary_flags:
            boundary_flags[0] = False  # Never insert space before first character

        # Decode best sequence
        best_restored = self.vocab.decode(
            best_char_ids,
            [int(b) for b in boundary_flags],
        )

        # Build character-by-character trace
        char_details: list[CharPredictionDetail] = []
        for i, ch in enumerate(compact):
            pred_id = best_char_ids[i]
            pred_ch = self.vocab.target_itos[pred_id] if pred_id < len(self.vocab.target_itos) else ch
            char_details.append(
                CharPredictionDetail(
                    index=i,
                    input_char=ch,
                    predicted_char=pred_ch,
                    boundary_flag=boundary_flags[i],
                    boundary_prob=round(boundary_prob_list[i], 4),
                    diacritic_prob=round(best_char_probs[i], 4),
                )
            )

        # Compute average sequence confidence score
        avg_score = round(
            float(sum(best_char_probs) / max(len(best_char_probs), 1)), 4
        )

        candidates = [
            {
                "text": best_restored,
                "rank": 1,
                "score": avg_score,
            }
        ]

        # Top-k alternative variations if requested (>1)
        if top_k > 1 and len(compact) > 0:
            top2_values, top2_indices = torch.topk(diacritic_probs, k=min(2, self.vocab.target_vocab_size), dim=-1)
            diff = top2_values[:, 0] - top2_values[:, 1]
            uncertain_indices = torch.argsort(diff)[: top_k - 1].cpu().tolist()

            for rank_offset, pos in enumerate(uncertain_indices, start=2):
                alt_ids = list(best_char_ids)
                alt_ids[pos] = top2_indices[pos, 1].item()
                alt_text = self.vocab.decode(alt_ids, [int(b) for b in boundary_flags])
                if alt_text != best_restored and not any(c["text"] == alt_text for c in candidates):
                    alt_score = round(avg_score * 0.95, 4)
                    candidates.append({
                        "text": alt_text,
                        "rank": len(candidates) + 1,
                        "score": alt_score,
                    })

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 3)

        return PredictionResult(
            request_id=req_id,
            raw_input=text,
            compact_input=compact,
            best_text=best_restored,
            candidates=candidates,
            char_details=char_details,
            latency_ms=elapsed_ms,
            model_name="NextKey Width-XXXS",
            parameter_count=self.num_params,
        )

    def get_metadata(self) -> dict[str, Any]:
        """Return model metadata and runtime specifications."""
        return {
            "model_name": "NextKey Width-XXXS",
            "architecture": "Dual-Head BiGRU CharTagger",
            "parameters": self.num_params,
            "parameters_human": f"{self.num_params / 1000:.1f}K",
            "checkpoint_size_kb": self.checkpoint_size_kb,
            "input_vocab_size": self.vocab.input_vocab_size,
            "target_vocab_size": self.vocab.target_vocab_size,
            "device": str(self.device),
            "status": "ready",
        }
