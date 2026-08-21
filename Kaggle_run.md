# NextKey — Kaggle Research & Training Guide

Hướng dẫn chạy huấn luyện và benchmark các phase trên **Kaggle GPU (T4 / P100 / A100)** hoặc **Local (MPS / CUDA / CPU)**.

---

## 1. Yêu cầu trước khi chạy (Setup)

1. **Bật GPU Accelerator**: Trong Notebook Settings trên Kaggle, chọn **GPU T4 x2** hoặc **GPU P100**.
2. **Attach Dataset**: Thêm Kaggle Dataset chứa thư mục `jdwr_v1` (hoặc `data/processed/jdwr_v1` có `manifest.json`).
3. **Cài đặt / Pull Code**:

```bash
# Clone hoặc pull mã nguồn NextKey mới nhất
!if [ ! -d /kaggle/working/NextKey/.git ]; then git clone https://github.com/nxc1802/NextKey.git /kaggle/working/NextKey; fi
!git -C /kaggle/working/NextKey pull origin main
%cd /kaggle/working/NextKey
!pip install -q pyyaml
```

---

## 2. Các lệnh CLI chính cho Phase 1 & Phase 2

### A. Phase 1 — Backbone Selection (5 họ kiến trúc)

Chạy so sánh 5 họ mô hình: `BiGRU`, `BiLSTM`, `CNN-TCN`, `CNN-BiGRU`, `Tiny Transformer` (ngân sách ~250K tham số).

```bash
# 1. Chạy TOÀN BỘ 5 mô hình Phase 1 + Tạo bảng so sánh Pareto (Smoke test ~30s)
!python scripts/run_phase1_backbone.py --all --mode smoke --device cuda

# 2. Chạy TOÀN BỘ 5 mô hình Phase 1 trên 100% dữ liệu (Research Mode)
!python scripts/run_phase1_backbone.py --all --mode research --device cuda

# 3. Chạy đơn lẻ từng mô hình cụ thể:
!python scripts/run_phase1_backbone.py --config configs/phase1_backbone/bigru.yaml --mode research --device cuda
!python scripts/run_phase1_backbone.py --config configs/phase1_backbone/bilstm.yaml --mode research --device cuda
!python scripts/run_phase1_backbone.py --config configs/phase1_backbone/cnn_tcn.yaml --mode research --device cuda
!python scripts/run_phase1_backbone.py --config configs/phase1_backbone/cnn_bigru.yaml --mode research --device cuda
!python scripts/run_phase1_backbone.py --config configs/phase1_backbone/tiny_transformer.yaml --mode research --device cuda
```

*Kết quả so sánh Pareto được tự động xuất ra:* `artifacts/phase1/pareto_backbone_report.md` và `.json`.

---

### B. Phase 2 — Size & Topology Search (12 biến thể kích thước)

Đánh giá ảnh hưởng của bề rộng (Width), độ sâu (Depth), và cấu trúc hình học (Topology).

#### 🚀 Chạy song song 2 Ultra-Small Models trên 2 GPU Kaggle T4x2 (Mới):
Hai model siêu nhỏ dưới 50K tham số: **Width-XXS (~34K params)** và **Width-XXXS (~18K params)**, chạy đồng thời trên `cuda:0` và `cuda:1`:

```bash
# 1. Chạy SONG SONG CẢ 2 GPU T4 (Kaggle Dual-GPU Mode — Tăng tốc x2):
#    GPU 0 chạy Width-XXS (~34K params), GPU 1 chạy Width-XXXS (~18K params) đồng thời!
!python scripts/run_phase2_size.py --sweep ultra_small --mode kaggle

# 2. Hoặc chạy qua Kaggle Runner tự động đóng gói Zip:
!python scripts/run_kaggle_training.py --phase 2 --sweep ultra_small --mode kaggle

# 3. Chạy kiểm tra nhanh 30s (Smoke Mode):
!python scripts/run_phase2_size.py --sweep ultra_small --mode smoke
```

#### Các lệnh chạy Sweep khác:

```bash
# • Chạy TOÀN BỘ cấu hình Phase 2 trên 100% dữ liệu (Research Mode)
!python scripts/run_phase2_size.py --all --mode research --device cuda

# • Width Sweep (XXXS ~18K, XXS ~34K, XS ~54K, S ~106K, M ~181K, L ~378K):
!python scripts/run_phase2_size.py --sweep width --mode research --device cuda

# • Depth Sweep (D1 ~181K, D2 ~475K, D3 ~771K):
!python scripts/run_phase2_size.py --sweep depth --mode research --device cuda

# • Topology Sweep (~300K compute budget: Wide/Shallow vs Mid/Mid vs Narrow/Deep):
!python scripts/run_phase2_size.py --sweep topo --mode research --device cuda
```

*Kết quả phân tích kích thước được tự động xuất ra:* `artifacts/phase2/size_ablation_results.md` và `.json`.

---

### C. Phase 3 — Edge Optimization & QKD Benchmark (Traditional KD vs. QKD)

So sánh trực tiếp giữa 2 chiến lược nén mô hình cho Edge Device với Teacher `Topo-A Wide/Shallow` (289K params) và Student `Width-XS` (54K params):
- **Option 1: Traditional KD** (Huấn luyện Student FP32 bằng KD $\to$ Lượng tử hóa sau PTQ INT8)
- **Option 2: QKD** (Quantization-Aware Knowledge Distillation: Huấn luyện Student INT8 trực tiếp cùng Teacher)

```bash
# 1. Chạy SONG SONG CẢ 2 GPU T4 (Kaggle Dual-GPU Mode — Tăng tốc x2):
#    GPU 0 chạy Traditional KD, GPU 1 chạy QKD đồng thời!
!python scripts/run_phase3_edge.py --strategy all --mode kaggle

# 2. Chạy tuần tự 1 GPU (Research Mode thông thường):
!python scripts/run_phase3_edge.py --strategy all --mode research --device cuda

# 3. Chạy nhanh kiểm tra luồng (Smoke Mode):
!python scripts/run_phase3_edge.py --strategy all --mode smoke --device cuda

# 4. Chạy đơn lẻ từng phương pháp cụ thể:
# • Chạy chỉ QKD (Quantization-Aware Distillation INT8 trực tiếp):
!python scripts/run_phase3_edge.py --strategy qkd --mode research --device cuda

# • Chạy chỉ Traditional KD (FP32 KD -> PTQ INT8):
!python scripts/run_phase3_edge.py --strategy traditional --mode research --device cuda
```

*Kết quả phân tích và so sánh được tự động xuất ra:* `artifacts/phase3/PHASE3_QKD_VS_TRADITIONAL.md`, `phase3_comparison_report.json`, model compact `.pt` (< 60 KB) và model ONNX.

---

### D. Chạy qua Kaggle Runner Tự Động (`run_kaggle_training.py`)

Kaggle runner tự động tìm dataset, tận dụng cả 2 GPU T4 chạy song song, chạy các phase và nén toàn bộ artifacts vào `nextkey-results.zip`:

```bash
# Chạy Phase 3 Tối ưu Edge (Chạy song song 2 GPU T4: GPU 0 -> Trad KD, GPU 1 -> QKD) và đóng gói zip
!python scripts/run_kaggle_training.py --phase 3 --strategy all --mode kaggle

# Chạy toàn bộ Phase 1 (5 backbones) và đóng gói zip
!python scripts/run_kaggle_training.py --phase 1 --all --mode kaggle

# Chạy toàn bộ Phase 2 (10 sizes) và đóng gói zip
!python scripts/run_kaggle_training.py --phase 2 --all --mode kaggle

# Chạy nhanh toàn bộ Phase 1 + 2 + 3 (Smoke mode check)
!python scripts/run_kaggle_training.py --phase all --mode smoke
```

---

## 3. Lệnh đóng gói file Zip kết quả (Zip Artifacts)

Nếu bạn chạy các script huấn luyện riêng lẻ (như `run_phase1_backbone.py` hoặc `run_phase2_size.py`), hãy chạy lệnh sau ở cell cuối cùng để gom toàn bộ kết quả vào 1 file `.zip` tải về:

```bash
# Đóng gói im lặng (không in log từng file) toàn bộ thư mục artifacts thành file zip
!zip -r -q /kaggle/working/nextkey-results.zip artifacts/
```

*Hoặc sử dụng Python (hoàn toàn không sinh log):*
```bash
!python -c "import shutil; shutil.make_archive('/kaggle/working/nextkey-results', 'zip', 'artifacts')"
```

---

## 4. Chạy trên máy Local (Apple Silicon MPS / CPU)

Chỉ cần thay cờ `--device cuda` thành `--device mps` (cho Mac) hoặc `--device cpu`:

```bash
# Local Phase 1 (Tất cả 5 model)
python scripts/run_phase1_backbone.py --all --mode smoke --device mps

# Local Phase 2 (Tất cả 10 size)
python scripts/run_phase2_size.py --all --mode smoke --device mps
```

---

## 5. Bảng tham số CLI

| Tham số | Giá trị | Mặc định | Mô tả |
|---|---|---|---|
| `--all` | cờ | `False` | Chạy toàn bộ danh sách model / size của phase |
| `--sweep` | `width`, `depth`, `topo`, `all` | `None` | Chạy một nhóm sweep cụ thể trong Phase 2 |
| `--config` | `<file.yaml>` | `None` | Đường dẫn file config tùy chỉnh |
| `--mode` | `smoke`, `research` | `smoke` (Local) / `research` (Kaggle) | 1K mẫu kiểm thử nhanh hoặc 100% dữ liệu |
| `--device` | `cuda`, `mps`, `cpu` | Auto | Thiết bị tính toán |
| `--output-dir` | `<dir>` | `artifacts/<phase>` | Thư mục lưu checkpoint & báo cáo |
| `--zip-output` | `<file.zip>` | `/kaggle/working/nextkey-results.zip` | Đường dẫn file zip xuất kết quả trên Kaggle |
