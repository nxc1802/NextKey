# NextKey Phase 3 — Báo Cáo So Sánh: Traditional Distillation vs. QKD

- **Thời gian chạy**: 2026-08-20 00:37:51
- **Chế độ**: `RESEARCH` | **Thiết bị**: `mps`
- **Teacher**: `Topo-A Wide/Shallow` (289K params)
- **Student**: `Width-XS` (54K params)

## Bảng tổng hợp so sánh độ chính xác và dung lượng

| Phương pháp / Mô hình | Định dạng | Dung lượng (KB) ↓ | Latency (CPU) ↓ | In-Domain CER ↓ | In-Domain BF1 ↑ | External CER ↓ |
|---|---|---:|---:|---:|---:|---:|
| **1. Student + Traditional KD** | FP32 | 217.7 KB | 0.70 ms | **0.1190** (11.90%) | 0.9594 | 0.1410 |
| **2. Student + KD $\to$ PTQ** | INT8 | **57.8 KB** | ~0.4 ms | **0.1192** (11.92%) | 0.9594 | 0.1414 |
| 🚀 **3. Student + QKD (Trực tiếp)** | INT8 | **57.8 KB** | **0.87 ms** | **0.1190** (11.90%) | 0.9579 | 0.1423 |
