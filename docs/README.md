# NextKey Documentation Index

Hệ thống tài liệu dự án **NextKey — Hệ Thống AI Khôi Phục Văn Bản Tiếng Việt Viết Gọn & Sửa Lỗi Gõ Phím Đa Nhiệm Trên Thiết Bị Biên**.

---

## 📂 Cấu Trúc Tài Liệu Tinh Gọn (Refactored Docs Structure)

```text
docs/
├── README.md                               # Tổng quan cấu trúc tài liệu dự án
├── methodology.md                          # TÀI LIỆU PHƯƠNG PHÁP LUẬN TOÀN DIỆN
│                                           # (Toán học, Dữ liệu JDWR v1, Noise Engine,
│                                           #  CascadeTriBiGRU, Edge Distillation & QKD/PTQ,
│                                           #  Khung đánh giá 3 cấp độ)
│
├── experiments_and_benchmark_report.md     # BÁO CÁO THỰC NGHIỆM & MASTER BENCHMARK
│                                           # (Tổng hợp toàn bộ Phase 1 -> Phase 4,
│                                           #  Ablation Study, Kaggle 1.7M Dual-T4 run,
│                                           #  Đánh giá đa miền 8 categories, Phân tích lỗi)
│
├── 00-project/                             # Hồ sơ dự án & Yêu cầu học thuật
│   ├── overview.md                         # Tổng quan dự án
│   ├── proposal.md                         # Đề cương nghiên cứu
│   ├── scope-matrix.md                     # Ma trận phạm vi (P0/P1/P2/Out)
│   ├── acceptance-criteria.md              # Tiêu chí nghiệm thu (MVP / Defense)
│   ├── task-breakdown.md                   # Phân rã công việc WBS
│   └── roadmap.md                          # Lộ trình phát triển
│
├── 01-data/                                # Dữ liệu & Quy chuẩn gán nhãn
│   ├── schema.md                           # Định dạng JSONL chuẩn
│   ├── noise-taxonomy.md                   # Phân loại 6 nhóm nhiễu bàn phím
│   ├── dataset-build-plan.md               # Quy trình trích xuất và sinh mẫu
│   └── annotation-guideline.md             # Hướng dẫn gán nhãn dữ liệu chuẩn
│
├── 02-model/                               # Thiết kế mô hình & Kế hoạch huấn luyện
│   ├── model-selection.md                  # Khảo sát và lựa chọn không gian mô hình
│   └── training-plan.md                    # Kế hoạch huấn luyện các giai đoạn
│
├── 03-evaluation/                          # Quy chuẩn đánh giá & Log kiểm thử MVP
│   ├── evaluation-plan.md                  # Kế hoạch đo lường chất lượng
│   ├── mvp-lexicon-baseline-report.md      # Baseline tra cứu từ điển
│   ├── mvp-chartagger-baseline-report.md   # Baseline nơ-ron sơ khai
│   └── mvp-chartagger-full-kaggle-report.md# Baseline Kaggle sơ khởi
│
└── 05-edge/                                # Thiết bị biên & Nén mô hình
    └── edge-plan.md                        # Kế hoạch tối ưu hóa CPU/Edge
```

---

## 🎯 Tài Liệu Trọng Tâm Nên Đọc Trước

1. **[`docs/methodology.md`](./methodology.md):** Phương pháp luận khoa học, công thức toán học, thiết kế kiến trúc phân tầng `CascadeTriBiGRU`, quy trình nén mô hình `QKD/PTQ` và hệ thống chỉ số đánh giá đa tầng 3 cấp độ.
2. **[`docs/experiments_and_benchmark_report.md`](./experiments_and_benchmark_report.md):** Toàn bộ kết quả thực nghiệm, đối sánh 5 Backbone (Phase 1), 10 Topology (Phase 2), Chưng cất & Lượng tử hóa (Phase 3), và mô hình 3-Task SOTA huấn luyện trên 1.7 triệu mẫu (Phase 4).
