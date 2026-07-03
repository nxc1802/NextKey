# Edge Deployment Plan

Edge work is a major differentiator, but it must come after a working teacher
and evaluation harness.

## Goals

- reduce model size
- reduce latency
- reduce RAM usage
- support offline or near-offline demo behavior
- preserve enough quality to remain useful

## Target Metrics

| Metric | Target | Strong |
|---|---:|---:|
| Student checkpoint size | < 150MB | < 80MB |
| INT8 model size | < 80MB | < 40MB |
| CPU inference RAM | < 800MB | < 400MB |
| Short sentence latency p50 | < 300ms | < 150ms |
| Medium sentence latency p50 | < 700ms | < 350ms |

## Phases

### Phase 1: Distillation Dataset

Create a dataset containing:

- compact input
- gold output
- teacher top-k outputs
- optional teacher scores

### Phase 2: Student Training

Train a small model optimized for deployability:

- char-level mini Transformer
- mini seq2seq model
- other export-friendly architecture selected by benchmark

### Phase 3: Export

Preferred order:

1. export float model
2. export ONNX
3. benchmark ONNX Runtime
4. attempt LiteRT only when architecture is suitable

### Phase 4: Quantization

Try:

- dynamic INT8 quantization
- static INT8 quantization if calibration is available
- fallback to float16 or dynamic quantization if full INT8 harms quality

### Phase 5: Edge Benchmark

Measure:

- model size
- startup time
- peak RAM
- p50/p95 latency
- quality delta against teacher

## Fallback Strategy

If export for the selected teacher is too difficult, keep the teacher server-side
and use the student as the edge proof. The thesis should frame this as a
teacher-student trade-off, not as a failed deployment.
