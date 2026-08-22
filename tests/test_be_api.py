"""Tests for NextKey Backend REST API and Inference Engine."""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from BE.main import app
from nextkey.engine.inference import NextKeyPredictor


def test_predictor_direct_inference():
    checkpoint = Path("artifacts/phase2/width_xxxs/best_model.pt")
    vocab = Path("artifacts/phase2/width_xxxs/vocab.json")
    if not checkpoint.exists() or not vocab.exists():
        pytest.skip("Model artifacts not found, skipping predictor test")

    predictor = NextKeyPredictor(checkpoint_path=checkpoint, vocab_path=vocab)
    res = predictor.restore("toidanghoc", top_k=2)

    assert res.request_id.startswith("req_")
    assert res.raw_input == "toidanghoc"
    assert res.compact_input == "toidanghoc"
    assert len(res.best_text) > 0
    assert len(res.candidates) >= 1
    assert len(res.char_details) == len("toidanghoc")
    assert res.latency_ms > 0


def test_fastapi_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_name" in data


def test_fastapi_model_info_endpoint():
    client = TestClient(app)
    response = client.get("/model_info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "NextKey Width-XXXS"
    assert data["parameters"] == 17828
    assert data["status"] == "ready"


def test_fastapi_restore_endpoint():
    client = TestClient(app)
    payload = {
        "input": "toidanghoc",
        "top_k": 3,
        "boundary_threshold": 0.5,
    }
    response = client.post("/restore", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "best" in data
    assert len(data["best"]) > 0
    assert "latency_ms" in data
    assert len(data["candidates"]) >= 1
    assert len(data["char_details"]) == len("toidanghoc")
