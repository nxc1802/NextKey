"""FastAPI Backend Application for NextKey Vietnamese Text Restoration."""

from __future__ import annotations

import argparse
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from BE.schemas import (
    Candidate,
    CharDetail,
    HealthResponse,
    ModelInfoResponse,
    ModelMeta,
    RestoreRequest,
    RestoreResponse,
)
from nextkey.engine.inference import NextKeyPredictor

# Global predictor instance
predictor: Optional[NextKeyPredictor] = None


def get_predictor() -> NextKeyPredictor:
    """Retrieve or initialize the NextKey predictor instance."""
    global predictor
    if predictor is None:
        # Default checkpoint and vocab paths
        checkpoint = Path("artifacts/phase2/width_xxxs/best_model.pt")
        vocab = Path("artifacts/phase2/width_xxxs/vocab.json")

        if not checkpoint.exists() or not vocab.exists():
            raise RuntimeError(
                f"Cannot find checkpoint ({checkpoint}) or vocab ({vocab}). "
                "Ensure artifacts are present in the workspace."
            )
        predictor = NextKeyPredictor(checkpoint_path=checkpoint, vocab_path=vocab)
    return predictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager to load model on startup."""
    try:
        get_predictor()
        print(" [NextKey BE] Model Width-XXXS loaded successfully!")
    except Exception as exc:
        print(f" [NextKey BE] Warning: Model could not be pre-loaded: {exc}")
    yield


app = FastAPI(
    title="NextKey API — Vietnamese Compact Text Restoration",
    description=(
        "High-performance, ultra-compact neural API for restoring accentless, "
        "spacing-free Vietnamese text using the NextKey Width-XXXS (~17.8K params) model."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["General"])
async def root():
    """Root info endpoint."""
    return {
        "project": "NextKey",
        "description": "Vietnamese Compact Writing Restoration API",
        "model": "NextKey Width-XXXS (Nano ~17.8K params)",
        "docs_url": "/docs",
        "endpoints": {
            "restore": "POST /restore",
            "model_info": "GET /model_info",
            "metrics_edge": "GET /metrics/edge",
            "health": "GET /health",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """Service health check."""
    pred = predictor
    return HealthResponse(
        status="ok",
        model_loaded=pred is not None,
        model_name="NextKey Width-XXXS (~17.8K params)",
    )


@app.get("/model_info", response_model=ModelInfoResponse, tags=["Monitoring"])
@app.get("/metrics/edge", response_model=ModelInfoResponse, tags=["Monitoring"])
async def get_model_info():
    """Get model specifications and edge deployment metrics."""
    try:
        pred = get_predictor()
        meta = pred.get_metadata()
        return ModelInfoResponse(
            model_name=meta["model_name"],
            architecture=meta["architecture"],
            parameters=meta["parameters"],
            parameters_human=meta["parameters_human"],
            checkpoint_size_kb=meta["checkpoint_size_kb"],
            input_vocab_size=meta["input_vocab_size"],
            target_vocab_size=meta["target_vocab_size"],
            device=meta["device"],
            status=meta["status"],
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch model info: {str(exc)}",
        )


@app.post("/restore", response_model=RestoreResponse, tags=["Inference"])
async def restore_text(payload: RestoreRequest):
    """Restore compact text into standard Vietnamese.

    Example input: "toidanghoc" -> "Tôi đang học"
    """
    try:
        pred = get_predictor()
        result = pred.restore(
            text=payload.input,
            top_k=payload.top_k,
            boundary_threshold=payload.boundary_threshold,
        )

        char_details = [
            CharDetail(
                index=cd.index,
                input_char=cd.input_char,
                predicted_char=cd.predicted_char,
                boundary_flag=cd.boundary_flag,
                boundary_prob=cd.boundary_prob,
                diacritic_prob=cd.diacritic_prob,
            )
            for cd in result.char_details
        ]

        candidates = [
            Candidate(
                text=c["text"],
                rank=c["rank"],
                score=c["score"],
            )
            for c in result.candidates
        ]

        meta = pred.get_metadata()

        return RestoreResponse(
            request_id=result.request_id,
            raw_input=result.raw_input,
            compact_input=result.compact_input,
            best=result.best_text,
            candidates=candidates,
            char_details=char_details,
            model=ModelMeta(
                name=meta["model_name"],
                architecture=meta["architecture"],
                parameters_human=meta["parameters_human"],
                checkpoint_size_kb=meta["checkpoint_size_kb"],
            ),
            latency_ms=result.latency_ms,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Restoration failed: {str(exc)}",
        )


def main():
    """Command-line entrypoint to launch backend server."""
    parser = argparse.ArgumentParser(description="Run NextKey FastAPI Backend")
    parser.add_argument("--host", default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    print(f"🚀 Starting NextKey Backend on http://{args.host}:{args.port}")
    uvicorn.run("BE.main:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
