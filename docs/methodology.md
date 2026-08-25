# NextKey — Phương Pháp Luận Nghiên Cứu & Thiết Kế Hệ Thống (Comprehensive Methodology)

**Dự án: NextKey — Hệ Thống AI Khôi Phục Văn Bản Tiếng Việt Viết Gọn & Sửa Lỗi Gõ Phím Đa Nhiệm Trên Thiết Bị Biên**  
**Tác giả: NextKey AI Research Team**  
**Mã đề tài: AIP491 — Capstone Project**  

---

## 📑 MỤC LỤC

1. [Tổng Quan & Định Nghĩa Toán Học Của Bài Toán](#1-tổng-quan--định-nghĩa-toán-học-của-bài-toán)
2. [Tập Dữ Liệu JDWR v1 & Động Cơ Sinh Nhiễu Bàn Phím (Noise Engine)](#2-tập-dữ-liệu-jdwr-v1--động-cơ-sinh-nhiễu-bàn-phím-noise-engine)
3. [Khảo Sát Kiến Trúc & Đột Phá Mạng Nơ-ron Phân Tầng CascadeTriBiGRU](#3-khảo-sát-kiến-trúc--đột-phá-mạng-nơ-ron-phân-tầng-cascadetribigru)
4. [Tối Ưu Hóa Thiết Bị Biên: Chưng Cất Tri Thức (KD) & Lượng Tử Hóa INT8 (QKD/PTQ)](#4-tối-ưu-hóa-thiết-bị-biên-chưng-cất-tri-thức-kd--lượng-tử-hóa-int8-qkdptq)
5. [Khung Đánh Giá Đa Tầng 3 Cấp Độ (3-Tier Hierarchical Evaluation Framework)](#5-khung-đánh-giá-đa-tầng-3-cấp-độ-3-tier-hierarchical-evaluation-framework)

---

## 1. Tổng Quan & Định Nghĩa Toán Học Của Bài Toán

### 1.1 Đặt Vấn Đề
Trong kỷ nguyên số, nhập liệu văn bản trên thiết bị di động (bàn phím cảm ứng QWERTY/Telex) là thao tác thường nhật của hàng chục triệu người dùng. Do hạn chế về kích thước phím bấm và nhu cầu nhắn tin tốc độ cao, người dùng thường có xu hướng:
* Gõ lướt không dấu tiếng Việt (*"toidanghoc"*).
* Viết liền không khoảng trắng (*"homnaytroidep"*).
* Bấm nhầm phím lân cận do ngón tay chạm lệch trên màn hình cảm ứng (*"hocbakk"* thay vì *"hocbai"*).
* Để lại mã tồn dư Telex/VNI khi gõ nhanh (*"ddang"* hoặc *"hojc"*).
* Sử dụng từ viết tắt hoặc tiếng lóng chat (*"ko"*, *"dc"*, *"ng"*, *"trc"*).

Các bộ gõ truyền thống dựa trên luật (Rule-based như Unikey, EVKey) hoặc các mô hình gán dấu truyền thống (N-gram, CRF) hoàn toàn bất lực khi chuỗi nhập liệu bị đồng thời nhiều loại nhiễu: vừa mất dấu, dính từ, vừa có lỗi ký tự gõ sai. Ngược lại, các mô hình ngôn ngữ lớn (LLMs, Seq2Seq Transformer) lại có độ trễ suy luận lớn ($> 500\text{ ms}$) và dung lượng bộ nhớ hàng trăm MB/GB, không thể chạy trực tiếp ngoại tuyến (offline) trên chip di động với ràng buộc thời gian thực ($< 1\text{ ms}$).

**NextKey** được xây dựng nhằm giải quyết triệt để bài toán này thông qua một kiến trúc nơ-ron hồi quy phân tầng siêu nhẹ, giải quyết đồng thời 3 bài toán chỉ trong một lần suy luận (single-pass forward).

---

### 1.2 Mô Hình Hóa Toán Học (Mathematical Formulation)

Cho một chuỗi ký tự đầu vào bị nhiễu độ dài $T$:
$$X = (x_1, x_2, \dots, x_T), \quad x_t \in \mathcal{V}_{\text{input}}$$
trong đó $\mathcal{V}_{\text{input}}$ là bảng chữ cái ký tự đầu vào không dấu kèm các ký tự đặc biệt/số.

Hệ thống NextKey mô hình hóa bài toán như một quá trình gán nhãn chuỗi đồng thời (Joint Sequence Tagging) qua 3 không gian nhãn đầu ra:

1. **Nhiệm vụ 1 — Sửa lỗi ký tự (Character Typo Correction):**
   $$y^{\text{corr}}_t \in \mathcal{V}_{\text{clean}}$$
   Ánh xạ ký tự bị gõ sai $x_t$ về ký tự nguyên âm/phụ âm cơ sở chính xác (ví dụ: $x_t = \text{'k'} \to y^{\text{corr}}_t = \text{'i'}$ trong ngữ cảnh *"hocbakk"* $\to$ *"hocbai"*).

2. **Nhiệm vụ 2 — Phân tách khoảng trắng / Biên từ (Whitespace & Word Boundary Segmentation):**
   $$y^{\text{bnd}}_t \in \{0, 1\}$$
   Dự đoán xem có khoảng trắng phân cách phía trước ký tự $t$ hay không ($1$: chèn khoảng trắng, $0$: viết liền).

3. **Nhiệm vụ 3 — Gán dấu thanh tiếng Việt (Diacritic Restoration):**
   $$y^{\text{diac}}_t \in \mathcal{V}_{\text{diac}}$$
   Dự đoán dấu thanh chuẩn tiếng Việt (Không dấu, Sắc, Huyền, Hỏi, Ngã, Nặng) và biến thể mũ/móc ($a, \breve{a}, \hat{a}, e, \hat{e}, o, \hat{o}, \sigma, u, u', d$).

```
                         [MÔ HÌNH HÓA ĐẦU VÀO / ĐẦU RA]
  
  Đầu vào (X):       t   o   i   d   a   n   g   h   o   c   b   a   k   k
                     │   │   │   │   │   │   │   │   │   │   │   │   │   │
  Sửa Typo (Y_corr): t   o   i   d   a   n   g   h   o   c   b   a   i   _
  Biên từ (Y_bnd):   0   0   0   1   0   0   0   1   0   0   1   0   0   0
  Thêm dấu (Y_diac): t   ô   i   đ   a   n   g   h   ọ   c   b   à   i   _
                     ─────────────────────────────────────────────────
  Đầu ra chuẩn (Y):  "Tôi đang học bài"
```

---

### 1.3 Hàm Mất Mát Đa Nhiệm Đa Trọng Số (Multi-Task Loss Function)

Để tối ưu hóa đồng thời 3 nhiệm vụ, NextKey sử dụng hàm mất mát kết hợp đa nhiệm (Weighted Multi-Task Cross-Entropy Loss) với cơ chế cân bằng trọng số:

$$\mathcal{L}_{\text{total}} = \lambda_{\text{corr}} \mathcal{L}_{\text{corr}} + \lambda_{\text{bnd}} \mathcal{L}_{\text{bnd}} + \lambda_{\text{diac}} \mathcal{L}_{\text{diac}}$$

Trong đó:
* $\mathcal{L}_{\text{corr}} = -\frac{1}{T}\sum_{t=1}^T \sum_{c=1}^{|\mathcal{V}_{\text{clean}}|} y^{\text{corr}}_{t,c} \log \hat{y}^{\text{corr}}_{t,c}$
* $\mathcal{L}_{\text{bnd}} = -\frac{1}{T}\sum_{t=1}^T \Big[ y^{\text{bnd}}_t \log \hat{y}^{\text{bnd}}_t + (1 - y^{\text{bnd}}_t) \log (1 - \hat{y}^{\text{bnd}}_t) \Big]$
* $\mathcal{L}_{\text{diac}} = -\frac{1}{T}\sum_{t=1}^T \sum_{d=1}^{|\mathcal{V}_{\text{diac}}|} y^{\text{diac}}_{t,d} \log \hat{y}^{\text{diac}}_{t,d}$
* Bộ siêu tham số trọng số tối ưu thực nghiệm: $\lambda_{\text{corr}} = 1.0, \lambda_{\text{bnd}} = 1.0, \lambda_{\text{diac}} = 1.2$ (ưu tiên độ chính xác dấu thanh).

---

## 2. Tập Dữ Liệu JDWR v1 & Động Cơ Sinh Nhiễu Bàn Phím (Noise Engine)

### 2.1 Cấu Trúc Tập Dữ Liệu JDWR v1 (Joint Diacritic and Word Restoration)
Ngữ liệu được chuẩn hóa và trích xuất từ 8 chuyên mục báo chí tiếng Việt chính thống, phân chia nghiêm ngặt thành các phân tập độc lập:

| Phân Tập (Split) | Số Lượng Mẫu (Sentences) | Mục Đích Sử Dụng | Chuyên Mục Bao Phủ |
|---|---:|---|---|
| **Train Set** | 548,530 | Huấn luyện mô hình ($\approx 1.7\text{M}$ synthetic samples) | 7 Chuyên mục In-Domain |
| **Dev / Validation Set** | 68,550 | Điều chỉnh siêu tham số, Early Stopping | 7 Chuyên mục In-Domain |
| **In-Domain Test Set** | 71,348 | Đánh giá năng lực nội miền chuẩn | 7 Chuyên mục (*Chính trị, Kinh doanh, Đời sống, Pháp luật, Sức khỏe, Thế giới, Văn hóa*) |
| **External OOD Test Set** | 159,172 | Đánh giá độ bền vững ngoại miền (Domain Gap) | Chuyên mục *Thể Thao* (hoàn toàn mới) |
| **TỔNG CỘNG** | **847,600** | **Đảm bảo tính đại diện và độ tin cậy khoa học cao nhất** | **8 Chuyên Mục** |

---

### 2.2 Động Cơ Sinh Nhiễu Nhân Tạo Thực Tế (Realistic QWERTY Noise Engine)
Để mô hình có thể thích ứng với mọi tình huống gõ phím thực tế, NextKey thiết kế bộ sinh nhiễu nhân tạo mô phỏng hành vi người dùng:

```
                            [MA TRẬN NGUỒN NHIỄU BÀN PHÍM]
  
   ┌───────────────────────┐   ┌───────────────────────┐   ┌───────────────────────┐
   │  1. BỎ DẤU THANH TIẾNG│   │ 2. DÍNH LIỀN KHOẢNG   │   │  3. LỖI GÕ PHÍM LÂN   │
   │          VIỆT         │   │         TRẮNG         │   │     CẬN (QWERTY)      │
   │  "tiếng việt" ->      │   │  "hôm nay" ->         │   │  "u" -> "y, i, j, h"  │
   │  "tieng viet"         │   │  "homnay"             │   │  "bài" -> "bakk"      │
   └───────────────────────┘   └───────────────────────┘   └───────────────────────┘
               │                           │                           │
               └───────────────────────────┼───────────────────────────┘
                                           ▼
   ┌───────────────────────┐   ┌───────────────────────┐   ┌───────────────────────┐
   │ 4. TỒN DƯ MÃ TELEX/VNI│   │ 5. CHỮ HOA / CHỮ THƯỜNG│   │  6. VIẾT TẮT & SLANG  │
   │  "đang" -> "ddang"    │   │  "Việt Nam" ->        │   │  "không" -> "ko"      │
   │  "học" -> "hojc"      │   │  "viet nam"           │   │  "được" -> "dc"       │
   └───────────────────────┘   └───────────────────────┘   └───────────────────────┘
```

1. **Ma trận khoảng cách hình học bàn phím QWERTY (Keyboard Adjacency Matrix):**
   Mỗi ký tự $c$ có tập hợp các phím lân cận $\mathcal{N}(c)$ trên layout bàn phím chuẩn:
   - Phím `'a'`: $\mathcal{N}(\text{'a'}) = \{\text{'q'}, \text{'w'}, \text{'s'}, \text{'z'}\}$
   - Phím `'h'`: $\mathcal{N}(\text{'h'}) = \{\text{'g'}, \text{'j'}, \text{'y'}, \text{'u'}, \text{'b'}, \text{'n'}\}$
   - Xác suất nhầm phím tỉ lệ nghịch với khoảng cách Euclid giữa tâm 2 phím trên màn hình cảm ứng.
2. **Quy tắc tồn dư Telex/VNI:** Mô phỏng lỗi ấn phím lặp (`aa` $\to$ `â`, `dd` $\to$ `đ`) hoặc gõ dấu chưa ăn (`s, f, r, x, j`).
3. **Quy tắc dính từ:** Tỉ lệ xóa khoảng trắng ngẫu nhiên từ $40\% - 80\%$ trên câu để tạo các chuỗi ghép dài liên tục.

---

## 3. Khảo Sát Kiến Trúc & Đột Phá Mạng Nơ-ron Phân Tầng CascadeTriBiGRU

### 3.1 Khảo Sát Kiến Trúc Xương Sống (Backbone Search)
Nghiên cứu khảo sát 5 họ kiến trúc trong cùng điều kiện ngân sách tham số ($\approx 200\text{K}$ params):
* **BiLSTM:** 2 cổng quên và cổng nhớ phức tạp, độ trễ $1.15\text{ ms}$.
* **CNN-BiGRU:** Kết hợp trích xuất đặc trưng không gian và chuỗi thời gian.
* **CNN-TCN (Temporal Convolutional Network):** Tích chập giãn nở đa tầng, tốc độ cao nhất ($0.45\text{ ms}$) nhưng mất ngữ cảnh từ xa.
* **Tiny-Transformer:** 2 lớp Self-Attention, bị suy thoái do thiếu thông tin vị trí tuần tự cục bộ và độ trễ cao ($1.95\text{ ms}$).
* **BiGRU (Bidirectional Gated Recurrent Unit):** Đạt cân bằng tối ưu giữa khả năng mô hình hóa ngữ cảnh hai chiều tiếng Việt, ít cổng điều khiển hơn LSTM giúp tốc độ xử lý nhanh gấp đôi.

---

### 3.2 Khảo Sát Topology: Đột Phá "Rộng & Nông" (Wide & Shallow Paradigm)
Trái với trực giác thông thường rằng mạng càng sâu càng tốt, nghiên cứu NextKey đã phát hiện một quy luật khoa học quan trọng:
* **Mạng 1 Lớp Rộng (`Width-XL`, `Topo-A Wide/Shallow` $H=160, L=1$):** Đạt CER **4.44%**, Word F1 **81.80%**.
* **Mạng 3 Lớp Sâu (`Depth-3L` $H=64, L=3$):** Đạt CER **4.85%**, Word F1 **80.05%** dù số tham số lớn hơn $1.65\times$.
* **Giải thích nguyên lý:** Các đặc trưng ngữ âm tiếng Việt (vần, thanh điệu, phụ âm đầu) có tính phụ thuộc cục bộ cực kỳ mạnh. Mạng 1-Layer BiGRU với ẩn số chiều rộng cho phép biểu diễn phong phú không gian ngữ âm mà không bị suy hao gradient qua các tầng sâu.

---

### 3.3 Đột Phá Kiến Trúc `CascadeTriBiGRU` (SOTA Phase 4)

Trong bài toán 3-nhiệm vụ đồng thời, nếu sử dụng 3 đầu phân loại độc lập song song (Parallel Heads), đầu Diacritic sẽ gặp khó khăn khi gán dấu vì ký tự đầu vào bị gõ sai. Để giải quyết triệt để vấn đề này, NextKey đề xuất kiến trúc phân tầng **`CascadeTriBiGRU`**:

```
Input Chars ──► Embedding (64d) ──► Conv1D (K=3) ──► Shared BiGRU (H=128, 256d)
                                                              │
                     ┌────────────────────────────────────────┼────────────────────────────────────────┐
                     ▼                                        ▼                                        │
             [Correction Head]                        [Boundary Head]                                  │
                     │                                        │                                        │
                     ▼                                        ▼                                        │
             Corr Feature (32d)                      Bnd Feature (8d)                                  │
                     └────────────────────────┬────────────────────────────────────────────────────────┘
                                              ▼
                                     [Cross-Head Fusion]
                                   H_fused = [H_backbone, Corr_feat, Bnd_feat] (296d)
                                              │
                                              ▼
                                       [Diacritic Head]
```

1. **Khối Conv1D Cục Bộ ($K=3, C=64$):** Trích xuất tức thời các n-gram 3 ký tự (phụ âm ghép như `ngh`, `tr`, `ch`, nguyên âm đôi `oai`, `uyen`).
2. **Khối Shared BiGRU Backbone ($H=128$):** Lan truyền ngữ cảnh hai chiều toàn câu, trích xuất vector biểu diễn ẩn $H_t \in \mathbb{R}^{256}$.
3. **Hierarchical Cross-Head Feature Flow:**
   * `Correction Head` tạo vector đặc trưng sửa lỗi: $F^{\text{corr}}_t = \text{ReLU}(W_{\text{corr\_proj}} H_t) \in \mathbb{R}^{32}$.
   * `Boundary Head` tạo vector đặc trưng biên từ: $F^{\text{bnd}}_t = \text{ReLU}(W_{\text{bnd\_proj}} H_t) \in \mathbb{R}^8$.
   * `Cross-Head Fusion`: Vector nối ghép đa nhiệm $H^{\text{fused}}_t = [H_t; F^{\text{corr}}_t; F^{\text{bnd}}_t] \in \mathbb{R}^{296}$ được cấp cho `Diacritic Head`.
   * **Hiệu quả:** `Diacritic Head` được "thông báo" ký tự thực sự đã được sửa và vị trí bắt đầu từ mới, giúp độ chính xác gán dấu đạt **94.97%** ngay cả khi đầu vào chứa lỗi gõ phím nặng.

---

## 4. Tối Ưu Hóa Thiết Bị Biên: Chưng Cất Tri Thức (KD) & Lượng Tử Hóa INT8 (QKD/PTQ)

Nhằm triển khai trực tiếp mô hình trên các thiết bị di động có tài nguyên giới hạn mà không cần kết nối mạng Internet, NextKey áp dụng quy trình nén mô hình kép:

```
  [Teacher Model] ───► Knowledge Distillation (KD) ───► [Student Model] ───► Quantization (INT8) ───► [Deployable Edge Model]
  Topo-A (289K params)                                   Width-XS (54K params)                           < 58 KB Checkpoint
  1.13 MB (FP32)                                         216 KB (FP32)                                   0.25 ms Latency
```

### 4.1 Cơ Chế Chưng Cất Tri Thức (Knowledge Distillation Formulation)
Mô hình Student học cách bắt chước phân phối xác suất mềm (Soft Probabilities) của mô hình Teacher thông qua hàm mất mát Kullback-Leibler (KL) Divergence:

$$\mathcal{L}_{\text{KD}} = (1 - \alpha) \mathcal{L}_{\text{CE}}(y_{\text{true}}, \hat{y}_{\text{student}}) + \alpha \cdot T^2 \cdot \mathcal{D}_{\text{KL}}\left( \sigma\left(\frac{z_{\text{teacher}}}{T}\right) \parallel \sigma\left(\frac{z_{\text{student}}}{T}\right) \right)$$

* Nhiệt độ làm mềm: $T = 2.0$.
* Hệ số cân bằng: $\alpha = 0.5$.
* Kết quả: Student có KD đạt CER thấp hơn và giữ nguyên độ chính xác biên từ so với Student tự học.

---

### 4.2 Lượng Tử Hóa INT8 (Post-Training Quantization & QKD)
Chuyển đổi toàn bộ trọng số mạng $W \in \mathbb{R}^{M \times N}$ từ dạng số thực 32-bit (FP32) sang số nguyên 8-bit (INT8):

$$q = \text{round}\left(\frac{W}{S}\right) + Z$$

* Hệ số tỉ lệ $S = \frac{\max(W) - \min(W)}{2^8 - 1}$, điểm zero $Z = -\text{round}(\min(W) / S)$.
* Giảm dung lượng mô hình tới $75\%$ (từ $216.2\text{ KB} \to 57.8\text{ KB}$) và tăng tốc độ xử lý CPU lên gấp $4\times$.

---

## 5. Khung Đánh Giá Đa Tầng 3 Cấp Độ (3-Tier Hierarchical Evaluation Framework)

Để đánh giá toàn diện năng lực mô hình từ chi tiết ký tự đến mức độ hiểu trọn vẹn của người dùng, NextKey xây dựng hệ thống chỉ số đo lường 3 cấp độ:

```
  ┌────────────────────────────────────────────────────────────────────────────────────────┐
  │                    HỆ THỐNG ĐO LƯỜNG ĐA TẦNG 3 CẤP ĐỘ (NEXTKEY)                        │
  ├────────────────────────┬───────────────────────────────┬───────────────────────────────┤
  │   1. CẤP KÝ TỰ         │         2. CẤP TỪ             │          3. CẤP CÂU           │
  │  (Character-Level)     │        (Word-Level)           │       (Sentence-Level)        │
  ├────────────────────────┼───────────────────────────────┼───────────────────────────────┤
  │ • Corpus CER (↓)       │ • Corpus WER (↓)              │ • Sentence Exact Match (↑)    │
  │ • Diacritic Accuracy(↑)│ • Word Accuracy WAcc (↑)      │ • Near-Perfect (CER <= 5%) (↑)│
  │ • Typo Recovery Rate(↑)│ • Word Overlap F1 (↑)         │ • High-Quality(CER <= 10%)(↑) │
  │ • Boundary F1-Score (↑)│ • Word Diacritic Accuracy (↑) │ • BLEU-1, BLEU-2, BLEU-4 (↑)  │
  │                        │                               │ • ROUGE-L F1 (↑)              │
  └────────────────────────┴───────────────────────────────┴───────────────────────────────┘
```

1. **Cấp Độ Ký Tự (Character-Level):**
   * $\text{Corpus CER} = \frac{\sum \text{Levenshtein}(y, \hat{y})}{\sum |y|}$: Tỉ lệ sai lệch chỉnh sửa ký tự.
   * $\text{Typo Recovery Rate} = \frac{\text{Số ký tự typo sửa đúng}}{\text{Tổng số ký tự typo ban đầu}}$.
   * $\text{Boundary F1} = \frac{2 \cdot P_{\text{bnd}} \cdot R_{\text{bnd}}}{P_{\text{bnd}} + R_{\text{bnd}}}$: Đo độ chuẩn xác chèn khoảng trắng.
2. **Cấp Độ Từ (Word-Level):**
   * $\text{Corpus WER} = \frac{\sum \text{Levenshtein}_{\text{word}}(y, \hat{y})}{\sum |y_{\text{words}}|}$.
   * $\text{Word Accuracy} = 1 - \text{WER}$: Tỉ lệ từ khôi phục hoàn hảo 100%.
   * $\text{Word Diacritic Accuracy}$: Tỉ lệ từ có toàn bộ ký tự mang đúng dấu thanh.
3. **Cấp Độ Câu (Sentence-Level):**
   * $\text{Exact Match (EM)} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(y_i == \hat{y}_i)$: Tỉ lệ câu hoàn hảo 100%.
   * $\text{Near-Perfect Rate}$: Tỉ lệ câu có $\text{CER} \le 5\%$ (lỗi tối đa 1 ký tự, người đọc hiểu trọn vẹn ngữ nghĩa).
   * $\text{High-Quality Rate}$: Tỉ lệ câu có $\text{CER} \le 10\%$ (đạt chuẩn giao tiếp hàng ngày).
   * $\text{BLEU-4}$ & $\text{ROUGE-L}$: Đánh giá sự bảo toàn chuỗi n-gram và ngữ nghĩa toàn văn bản.
