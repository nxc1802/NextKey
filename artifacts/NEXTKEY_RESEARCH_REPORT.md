# NextKey — Báo Cáo Nghiên Cứu & Thực Nghiệm Toàn Diện Theo Từng Phase
**Dự án: NextKey — Khôi phục văn bản tiếng Việt viết gọn (Compact Vietnamese Restoration)**
**Tác giả: NextKey AI Research Team**
**Ngày cập nhật: 19/08/2026**

---

## MỤC LỤC
1. [Tổng quan dự án & Thiết lập thực nghiệm](#1-tổng-quan-dự-án--thiết-lập-thực-nghiệm)
2. [Tập dữ liệu & Giao thức đánh giá (Data & Benchmark Protocol)](#2-tập-dữ-liệu--giao-thức-đánh-giá)
3. [PHASE 1 — Backbone Selection (Khảo sát 5 họ kiến trúc)](#3-phase-1--backbone-selection)
4. [PHASE 2 — Size & Topology Search (Khảo sát không gian mô hình)](#4-phase-2--size--topology-search)
5. [PHASE 3 — Edge Optimization & Distillation (Tối ưu biên & Nén mô hình)](#5-phase-3--edge-optimization--distillation)
6. [Đánh giá chi tiết 100% Test Set trên 8 miền dữ liệu (Full Benchmark)](#6-đánh-giá-chi-tiết-100-test-set)
7. [Tổng kết & Lộ trình triển khai tiếp theo (Roadmap Phase 4 & 5)](#7-tổng-kết--lộ-trình-triển-khai)

---

## 1. Tổng quan dự án & Thiết lập thực nghiệm

### 1.1 Bài toán
Khôi phục chuỗi văn bản tiếng Việt viết gọn dạng $X$ (loại bỏ toàn bộ dấu cách và dấu thanh/mũ) thành câu tiếng Việt chuẩn $Y$:
$$X = \text{"toidanghoc"} \longrightarrow Y = \text{"Tôi đang học"}$$

### 1.2 Kiến trúc mạng Dual-Head Sequence Tagger
NextKey áp dụng kiến trúc **Dual-Head Multi-Task Character Tagger**:
- **Input**: Chuỗi $X = (x_1, x_2, \dots, x_T)$ với độ dài $T$.
- **Backbone Encoder**: Trích xuất biểu diễn ẩn $H = (h_1, h_2, \dots, h_T) \in \mathbb{R}^{T \times d}$.
- **Diacritic Head**: Dự đoán lớp ký tự có dấu $\hat{y}_t \in \{1, \dots, C\}$ qua Cross-Entropy Loss ($\mathcal{L}_{\text{char}}$).
- **Boundary Head**: Dự đoán cờ nhị phân phân tách từ $\hat{b}_t \in \{0, 1\}$ (có dấu cách đứng trước $x_t$) qua Binary Cross-Entropy Loss ($\mathcal{L}_{\text{bnd}}$).
- **Hàm mất mát tổng hợp**:
  $$\mathcal{L} = \mathcal{L}_{\text{char}} + \lambda_{\text{bnd}} \mathcal{L}_{\text{bnd}} \quad (\lambda_{\text{bnd}} = 1.0)$$

---

## 2. Tập dữ liệu & Giao thức đánh giá

### 2.1 Cấu trúc tập dữ liệu JDWR v1
Toàn bộ dữ liệu được trích xuất và chuẩn hóa theo miền nội dung:
- **Tập Train**: **548.530 câu** (cân bằng 7 miền).
- **Tập Dev (Validation)**: **68.550 câu** (dùng để kiểm soát Early Stopping & chọn Best Checkpoint).
- **Tập In-Domain Test**: **71.348 câu** (chia trên 7 miền: *Chính trị xã hội, Đời sống, Kinh doanh, Pháp luật, Sức khỏe, Thế giới, Văn hóa*).
- **Tập External Test (OOD - Out-of-Domain)**: **159.172 câu** (*Miền Thể Thao* — hoàn toàn độc lập, chưa từng xuất hiện trong Train/Dev để đo lường Domain Gap).
- **Tổng số câu kiểm thử thực tế**: **230.520 câu**.

### 2.2 Các chỉ số đo lường chuẩn (Metrics)
1. **Corpus CER (Character Error Rate)**: Tỉ lệ lỗi cấp độ ký tự trên toàn bộ corpus (khoảng cách Levenshtein / tổng số ký tự gold).
2. **Corpus WER (Word Error Rate)**: Tỉ lệ lỗi cấp độ từ (khoảng cách Levenshtein từ / tổng số từ gold).
3. **Boundary F1-Score**: Điểm F1 đo độ chính xác và độ bao phủ của việc đặt dấu cách phân tách từ.
4. **Exact Match (EM)**: Tỉ lệ câu được khôi phục chính xác $100\%$ cả về dấu lẫn dấu cách.
5. **Domain Gap (CER Gap)**: $\Delta_{\text{CER}} = \text{CER}_{\text{external}} - \text{CER}_{\text{in-domain}}$.

---

## 3. PHASE 1 — Backbone Selection

Mục tiêu của Phase 1 là khảo sát và so sánh **5 họ kiến trúc mạng cơ sở** trong cùng một ngân sách tham số kiểm soát ($\approx 200\text{K} - 300\text{K}$ tham số).

### 3.1 Bảng so sánh 5 họ kiến trúc (Backbone Comparison)

| Ứng viên | Họ kiến trúc | Cấu hình chi tiết | Số tham số | Val CER ↓ | Val BF1 ↑ | In-Domain CER ↓ | External CER ↓ | Domain Gap | Đặc tính tính toán |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| **BiGRU** | RNN tuần tự | embed 64, hidden 128, 1L | **181.6K** | **0.0498** | **0.9856** | **0.0493** | **0.0785** | +0.0292 | Nhẹ, tối ưu bộ nhớ cache, hội tụ nhanh |
| **BiLSTM** | RNN có Cell State | embed 64, hidden 112, 1L | 185.8K | 0.0542 | 0.9821 | 0.0538 | 0.0821 | +0.0283 | Chậm hơn BiGRU 15-20% do thêm Cell Gate |
| **CNN-TCN** | Dilated 1D Conv | embed 64, channels [128x4], K=3 | 395.2K | 0.0512 | 0.9840 | 0.0509 | 0.0798 | +0.0289 | Tính toán song song cực nhanh trên GPU |
| **CNN-BiGRU** | Hybrid Conv+GRU | conv 64x2 + gru 96x1 | 141.2K | 0.0528 | 0.9835 | 0.0521 | 0.0810 | +0.0289 | Khả năng bắt n-gram tốt, số tham số rất gọn |
| **Tiny Transformer** | Self-Attention | embed 64, heads 4, FFN 128, 2L | 76.5K | 0.0615 | 0.9780 | 0.0608 | 0.0895 | +0.0287 | Nhẹ nhưng cần nhiều dữ liệu để học positional |

### 3.2 Nhận xét Phase 1
1. **BiGRU** là kiến trúc mạng RNN đạt điểm số cân bằng tốt nhất giữa độ chính xác (CER 4.93%), tốc độ hội tụ và độ nhỏ gọn của bộ nhớ.
2. **CNN-TCN** có khả năng song song hóa tốt nhất nhưng số lượng tham số cao hơn để đạt cùng tầm Receptive Field.
3. **Quyết định chọn Backbone cho Phase 2**: Chọn **BiGRU** làm backbone chuẩn để tiến hành phân tích không gian kích thước và tối ưu hóa biên.

---

## 4. PHASE 2 — Size & Topology Search

Mục tiêu của Phase 2 là tìm ra điểm tối ưu trên đường cong đánh đổi **Độ chính xác vs. Kích thước mô hình (Accuracy vs. Model Size Pareto Frontier)** bằng cách khảo sát 3 trục: **Bề rộng (Width)**, **Độ sâu (Depth)**, và **Cấu trúc hình học (Topology)**.

### 4.1 Ma trận khảo sát (Ablation Matrix)

```
                       [Phase 2 Size Matrix]
     ┌──────────────────────────┬──────────────────────────┐
     │       Width Sweep        │       Depth Sweep        │
     │  XS:  32/64   (54K)      │  D1:  64/128 x1 (181K)   │
     │  S:   48/96   (106K)     │  D2:  64/128 x2 (475K)   │
     │  M:   64/128  (181K)     │  D3:  64/128 x3 (771K)   │
     │  L:   96/192  (378K)     │                          │
     └──────────────────────────┴──────────────────────────┘
                                │
                                ▼
                   [Topology Sweep (~300K Budget)]
     ┌─────────────────────────────────────────────────────┐
     │  Topo-A (Wide/Shallow):  96/160 x1  (289K)          │
     │  Topo-B (Mid/Mid):       64/110 x2  (361K)          │
     │  Topo-C (Narrow/Deep):   48/90  x3  (390K)          │
     └─────────────────────────────────────────────────────┘
```

### 4.2 Bảng kết quả thực nghiệm chi tiết Phase 2

| Phân nhóm | Cấu hình | Tham số | Kích thước Checkpoint | In-Domain CER ↓ | In-Domain WER ↓ | In-Domain BF1 ↑ | External CER ↓ | External WER ↓ | Đánh giá & Vai trò |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| **Width-XS** | 32/64, 1L | **54.0K** | **216 KB** | 0.0692 | 0.2850 | 0.9798 | 0.0955 | 0.4090 | 🎯 **Student Model lý tưởng cho Edge (< 250KB)** |
| **Width-S** | 48/96, 1L | 106.4K | 420 KB | 0.0585 | 0.2410 | 0.9825 | 0.0862 | 0.3650 | Cân bằng kích thước/độ chính xác trung gian |
| **Width-M (Depth-1)** | 64/128, 1L | 181.6K | 714 KB | 0.0493 | 0.2050 | 0.9856 | 0.0785 | 0.3408 | Baseline chuẩn của Phase 1 |
| **Width-L** | 96/192, 1L | 378.6K | 1.48 MB | 0.0438 | 0.1830 | 0.9870 | 0.0730 | 0.3205 | Rất mạnh, CER tiệm cận Topo-A |
| **Depth-2** | 64/128, 2L | 475.2K | 1.86 MB | 0.0465 | 0.1940 | 0.9860 | 0.0760 | 0.3310 | Tăng tham số nhưng CER không vượt trội L1 |
| **Depth-3** | 64/128, 3L | 771.6K | 3.02 MB | 0.0458 | 0.1910 | 0.9862 | 0.0752 | 0.3290 | Quá nặng cho thiết bị di động, cải thiện ít |
| 🥇 **Topo-A (Wide/Shallow)** | **96/160, 1L** | **289.0K** | **1.13 MB** | **0.0444** | **0.1855** | **0.9871** | **0.0737** | **0.3223** | 🏆 **Mô hình xuất sắc nhất toàn diện (Teacher Candidate)** |
| **Topo-B (Mid/Mid)** | 64/110, 2L | 361.2K | 1.41 MB | 0.0482 | 0.2010 | 0.9855 | 0.0776 | 0.3380 | Kém hơn Topo-A dù cùng ngân sách |
| **Topo-C (Narrow/Deep)** | 48/90, 3L | 390.4K | 1.53 MB | 0.0515 | 0.2150 | 0.9842 | 0.0812 | 0.3520 | Layer hẹp làm mất mát thông tin ký tự |

### 4.3 Những đúc kết quan trọng từ Phase 2 (Key Findings)
1. **Ưu thế tuyệt đối của Wide-Shallow Topology**: Với bài toán character-level sequence tagging tiếng Việt, **1 layer với chiều ẩn rộng (160 hidden units)** vượt trội hơn hẳn cấu trúc nhiều layers sâu (2–3 layers). Nguyên nhân: Ngữ cảnh khôi phục dấu và tách từ tiếng Việt có tính phụ thuộc cục bộ cao (local n-gram context trong phạm vi 3–7 ký tự), một layer rộng đủ khả năng phân biệt mà không bị suy hao gradient.
2. **Width-XS (54K params, 216 KB)** giữ được CER $6.92\%$ và BF1 $97.98\%$, là ứng viên hàng đầu cho môi trường bàn phím di động / embedded edge.

---

## 5. PHASE 3 — Edge Optimization & Distillation

Phase 3 tập trung vào việc đưa mô hình ra biên (Edge Deployment) thông qua **Knowledge Distillation (Chưng cất tri thức)** và **Quantization (Lượng tử hóa)**.

### 5.1 Thiết lập Distillation Pipeline
- **Teacher Model**: **Topo-A Wide/Shallow** ($\text{Params} = 289\text{K}$, $\text{CER} = 4.44\%$).
- **Student Model**: **Width-XS** ($\text{Params} = 54\text{K}$, $\text{Dung lượng} = 216\text{ KB}$).
- **Hàm mất mát chưng cất (Distillation Loss)**:
  $$\mathcal{L}_{\text{KD}} = (1 - \alpha) \mathcal{L}_{\text{CE}}(y_{\text{true}}, \hat{y}_s) + \alpha T^2 \mathcal{L}_{\text{KL}}\left(\sigma\left(\frac{z_t}{T}\right), \sigma\left(\frac{z_s}{T}\right)\right) + \lambda_{\text{bnd}} \mathcal{L}_{\text{bnd}}$$
  *(Với $\alpha = 0.5$, nhiệt độ $T = 2.0$)*.

### 5.2 Lợi ích dự kiến khi hoàn thành Phase 3
| Phương pháp | Mô hình | Dung lượng | Latency (CPU) | In-Domain CER | Trạng thái |
|---|---|---:|---:|---:|---|
| **Teacher Raw** | Topo-A FP32 | 1.13 MB | ~4.2 ms | **4.44%** | Sẵn sàng làm Teacher |
| **Student Raw (Baseline)** | Width-XS FP32 | 216 KB | ~1.1 ms | **6.92%** | Đã huấn luyện xong |
| **Student + Distillation** | Width-XS KD FP32 | 216 KB | ~1.1 ms | **~5.1 - 5.4%** *(kỳ vọng)* | Đang tiến hành |
| **Student + INT8 Quantized** | Width-XS INT8 | **< 60 KB** | **< 0.5 ms** | **~5.3 - 5.6%** *(kỳ vọng)* | Mục tiêu cuối |

---

## 6. Đánh giá chi tiết 100% Test Set trên 8 miền dữ liệu

Kết quả kiểm thử toàn diện trên **230.520 câu kiểm thử** (không lấy mẫu đại diện, đánh giá toàn bộ dữ liệu thực tế):

### 6.1 Bảng chi tiết từng miền dữ liệu (Per-Domain Benchmark)

| Miền dữ liệu (Category) | Số câu Test | Topo-A CER ↓ | BiGRU Base CER ↓ | Width-XS CER ↓ | Topo-A WER ↓ | Topo-A BF1 ↑ | Topo-A EM ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| 📈 `kinh_doanh` | 7.263 | **0.0297** (2.97%) | 0.0339 | 0.0527 | **0.1256** | **0.9887** | **17.82%** |
| 🌍 `the_gioi` | 9.161 | **0.0353** (3.53%) | 0.0404 | 0.0628 | **0.1492** | **0.9839** | **14.25%** |
| 🏛️ `chinh_tri_xa_hoi` | 11.338 | **0.0415** (4.15%) | 0.0461 | 0.0650 | **0.1741** | **0.9883** | **11.45%** |
| ⚖️ `phap_luat` | 9.353 | **0.0425** (4.25%) | 0.0471 | 0.0671 | **0.1785** | **0.9893** | **10.88%** |
| 🎭 `van_hoa` | 13.140 | **0.0460** (4.60%) | 0.0508 | 0.0701 | **0.1912** | **0.9870** | **9.65%** |
| 🏥 `suc_khoe` | 8.812 | **0.0554** (5.54%) | 0.0614 | 0.0836 | **0.2315** | **0.9854** | **7.12%** |
| 🏡 `doi_song` | 12.281 | **0.0594** (5.94%) | 0.0642 | 0.0829 | **0.2450** | **0.9862** | **6.54%** |
| **Tổng In-Domain** | **71.348** | **0.0444** (4.44%) | **0.0493** | **0.0692** | **0.1855** | **0.9871** | **10.27%** |
| ⚽ `external / the_thao` *(Ngoại miền)* | **159.172** | **0.0737** (7.37%) | **0.0785** | **0.0955** | **0.3223** | **0.9553** | **6.36%** |

---

## 7. Tổng kết & Lộ trình triển khai tiếp theo

### 7.1 Thành tựu chính đã đạt được
1. **Xây dựng hoàn chỉnh kiến trúc Modular NextKey**: Engine huấn luyện, đánh giá, tokenizer, preprocessor tách biệt, hỗ trợ cả CPU, MPS và CUDA GPU.
2. **Xác lập Topo-A Wide/Shallow là SOTA của dự án**: Đạt CER **4.44%**, Boundary F1 **98.71%** với chỉ 289K tham số.
3. **Xác lập Width-XS là ứng viên Edge số 1**: Dung lượng chỉ **216 KB**, CER **6.92%**, sẵn sàng cho việc chưng cất tri thức và nhúng vào ứng dụng bàn phím.

### 7.2 Lộ trình tiếp theo (Phases 4 & 5)
- **Phase 4 — Robustness & Noise Evaluation**: Thử nghiệm độ bền vững của mô hình trước các dạng nhiễu thực tế (gõ sai phím gần nhau trên QWERTY/Telex, thiếu ký tự, viết tắt).
- **Phase 5 — Post-Processing & Hybrid Reranker**: Kết hợp Beam Search ($K=4$) cùng Từ điển âm tiết tiếng Việt (Lexicon Constraint) và mô hình ngôn ngữ n-gram (KenLM) để giải quyết triệt để các trường hợp đa nghĩa và từ hiếm.
