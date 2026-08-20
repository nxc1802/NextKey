# NextKey Phase 3 — Báo Cáo So Sánh: Traditional Distillation vs. QKD

- **Thời gian chạy**: 2026-08-20 08:24:45
- **Chế độ**: `KAGGLE` | **Thiết bị**: `cuda:0 + cuda:1 (Dual GPU Parallel)`
- **Teacher**: `Topo-A Wide/Shallow` (289K params)
- **Student**: `Width-XS` (54K params)

## Bảng tổng hợp so sánh đầy đủ các mô hình và phương pháp

| Phương pháp / Mô hình | Định dạng | Dung lượng (KB) ↓ | Latency (CPU) ↓ | In-Domain CER ↓ | In-Domain BF1 ↑ | External CER ↓ |
|---|---|---:|---:|---:|---:|---:|
| 👑 **0. Teacher (Topo-A Wide/Shallow)** | FP32 | 1134.5 KB | 5.02 ms | **0.0444** (4.44%) | 0.9871 | 0.0737 |
| 📦 **1. Student Gốc (Width-XS Không KD)** | FP32 | 216.2 KB | 0.70 ms | **0.0692** (6.92%) | 0.9798 | 0.0955 |
| **2. Student + Traditional KD** | FP32 | 216.7 KB | 0.70 ms | **0.0691** (6.91%) | 0.9800 | 0.0959 |
| **3. Student + KD $\to$ PTQ** | INT8 | **57.8 KB** | ~0.4 ms | **0.0694** (6.94%) | 0.9799 | 0.0961 |
| 🚀 **4. Student + QKD (Trực tiếp)** | INT8 | **57.8 KB** | **0.87 ms** | **0.0699** (6.99%) | 0.9801 | 0.0964 |
