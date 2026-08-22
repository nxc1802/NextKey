"""Pydantic data schemas for NextKey Backend REST API."""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


class RestoreRequest(BaseModel):
    """Payload for text restoration request."""
    input: str = Field(
        ...,
        description="Raw compact Vietnamese text (e.g. 'toidanghoc', 'nguoivietnamyeunuoc')",
        examples=["toidanghoc"],
    )
    top_k: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Number of candidate variations to return",
    )
    boundary_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Threshold for space insertion boundary prediction",
    )
    user_id_hash: Optional[str] = Field(
        default=None,
        description="Optional anonymous identifier for telemetry/logging",
    )
    mode: str = Field(
        default="width_xxxs",
        description="Model variant tag (e.g. 'width_xxxs')",
    )


class Candidate(BaseModel):
    """A restored sentence candidate."""
    text: str
    rank: int
    score: float


class CharDetail(BaseModel):
    """Character-level prediction trace."""
    index: int
    input_char: str
    predicted_char: str
    boundary_flag: bool
    boundary_prob: float
    diacritic_prob: float


class ModelMeta(BaseModel):
    """Model information in response."""
    name: str
    architecture: str
    parameters_human: str
    checkpoint_size_kb: float


class RestoreResponse(BaseModel):
    """Structured response for restoration endpoint."""
    request_id: str
    raw_input: str
    compact_input: str
    best: str
    candidates: list[Candidate]
    char_details: list[CharDetail]
    model: ModelMeta
    latency_ms: float


class ModelInfoResponse(BaseModel):
    """Model metadata & specifications."""
    model_name: str
    architecture: str
    parameters: int
    parameters_human: str
    checkpoint_size_kb: float
    input_vocab_size: int
    target_vocab_size: int
    device: str
    status: str


class HealthResponse(BaseModel):
    """Service health response."""
    status: str
    model_loaded: bool
    model_name: str
