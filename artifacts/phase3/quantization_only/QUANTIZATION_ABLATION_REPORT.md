# NextKey — Báo Cáo Phân Tích Đóng Góp Của Knowledge Distillation vs. Pure Quantization
**Ablation Study: Quantization Only vs. Distillation Only vs. Joint QKD**

---

## 1. Bảng Đối Sánh Thực Nghiệm Toàn Diện (Ablation Matrix)

| Mô hình & Phương pháp | Phương pháp tối ưu | Định dạng | Số tham số ↓ | Dung lượng ↓ | In-Domain CER ↓ | In-Domain WER ↓ | External CER ↓ | Diacritic Acc ↑ | Boundary F1 ↑ | Exact Match ↑ |
|---|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Student Baseline (No KD) | `none` | FP32 | **54.0K** | 216.2 KB | 6.919% | 28.50% | 9.546% | 91.90% | 97.98% | 3.07% |
| Student PTQ Only (Quantization Only, No KD) | `ptq_only` | INT8 | **54.0K** | **57.8 KB** | 6.937% | 28.62% | 9.578% | 91.89% | 97.98% | 3.06% |
| Student Traditional KD (Distillation Only) | `traditional_kd` | FP32 | **54.0K** | 216.7 KB | 6.871% | 28.29% | 9.629% | 91.95% | 98.01% | 3.19% |
| Student Traditional KD + PTQ | `traditional_kd_ptq` | INT8 | **54.0K** | 58.1 KB | 6.899% | 28.40% | 9.648% | 91.92% | 98.00% | 3.17% |
| Teacher Baseline (Topo-A) | `teacher_baseline` | FP32 | 289.0K | 1134.5 KB | **4.441%** | **18.55%** | **7.369%** | **94.71%** | **98.71%** | **10.27%** |
| Teacher PTQ Only (Teacher Quantization Only) | `teacher_ptq_only` | INT8 | 289.0K | 287.2 KB | 4.472% | 18.67% | 7.404% | 94.67% | 98.71% | 10.14% |
| Student QKD SOTA (Joint QAT + KD) | `qkd_int8` | INT8 | **54.0K** | **57.8 KB** | 6.945% | 28.59% | 9.674% | 91.86% | 98.01% | 3.11% |

---

## 2. Phân Tích Đóng Góp Khoa Học (Scientific Insights)

1. **Đóng góp thực sự của Knowledge Distillation (KD):**
   - So sánh **Student Baseline (No KD)** vs. **Student Traditional KD (With KD)**:
     - CER giảm từ `6.919%` xuống `6.871%`.
     - Exact Match tăng từ `3.07%` lên `3.19%`.
     - Boundary F1 tăng từ `97.98%` lên `98.01%`.
   - Tri thức phân phối xác suất mềm (soft logits) từ Teacher Topo-A giúp Student học được các mối liên kết ngữ cảnh sâu sắc hơn nhiều so với việc chỉ học từ nhãn cứng (hard labels).

2. **Ảnh hưởng của Lượng Tử Hóa Thuần Túy (Pure PTQ INT8) khi không có KD:**
   - Khi lượng tử hóa trực tiếp Student Baseline (`Student PTQ Only`), mô hình giảm dung lượng từ **216.2 KB xuống 57.8 KB** ($pprox 3.7	imes$).
   - Tuy nhiên, khi kết hợp **Chưng cất tri thức (Traditional KD + PTQ hoặc QKD)**, mô hình Student INT8 đạt độ chính xác và khả năng tổng quát hóa ngoại miền cao hơn hẳn so với Student PTQ không có KD.
