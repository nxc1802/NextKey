# NextKey — Báo Cáo Đánh Giá Trên 100% Toàn Bộ Tập Dữ Liệu Test

Đánh giá toàn diện trên toàn bộ **232.029 câu kiểm thử** (71,348 In-domain + 159,172 External).

## 1. Bảng so sánh tổng thể (100% Test Data)

| Model | Params | Size (KB) | In-Domain CER ↓ | In-Domain WER ↓ | In-Domain BF1 ↑ | In-Domain EM ↑ | External CER ↓ | External WER ↓ | External BF1 ↑ | Domain Gap (CER) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Topo-A Wide/Shallow (96/160, 1L)** | 289,044 | 1134.5 KB | **0.0444** (4.44%) | 0.1855 | 0.9871 | 10.27% | **0.0737** (7.37%) | 0.3223 | 0.9553 | +0.0293 |
| **BiGRU Baseline (64/128, 1L)** | 181,556 | 714.6 KB | **0.0493** (4.93%) | 0.2050 | 0.9856 | 8.30% | **0.0785** (7.85%) | 0.3408 | 0.9528 | +0.0292 |
| **Width-XS Edge Model (32/64, 1L)** | 53,972 | 216.2 KB | **0.0692** (6.92%) | 0.2850 | 0.9798 | 3.07% | **0.0955** (9.55%) | 0.4090 | 0.9457 | +0.0263 |

## 2. Chi tiết hiệu năng từng miền dữ liệu (Per-Domain CER & Boundary F1)

| Miền dữ liệu (Domain) | Số mẫu Test | Topo-A CER | BiGRU Baseline CER | Width-XS CER | Topo-A Boundary F1 |
|---|---:|---:|---:|---:|---:|
| `chinh_tri_xa_hoi` | 11,338 | **0.0415** | 0.0461 | 0.0650 | 0.9883 |
| `doi_song` | 12,281 | **0.0594** | 0.0642 | 0.0829 | 0.9862 |
| `kinh_doanh` | 7,263 | **0.0297** | 0.0339 | 0.0527 | 0.9887 |
| `phap_luat` | 9,353 | **0.0425** | 0.0471 | 0.0671 | 0.9893 |
| `suc_khoe` | 8,812 | **0.0554** | 0.0614 | 0.0836 | 0.9854 |
| `the_gioi` | 9,161 | **0.0353** | 0.0404 | 0.0628 | 0.9839 |
| `van_hoa` | 13,140 | **0.0460** | 0.0508 | 0.0701 | 0.9870 |
| `external / the_thao` *(Ngoại miền)* | 159,172 | **0.0737** | 0.0785 | 0.0955 | 0.9553 |
