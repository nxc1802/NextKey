"""Streamlit Frontend Application for NextKey Vietnamese Text Restoration.

Supports both Direct In-Memory Inference and Backend REST API modes.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Add src to python path if not present
current_dir = Path(__file__).resolve().parent
src_dir = current_dir.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import pandas as pd
import requests
import streamlit as st

from nextkey.engine.inference import NextKeyPredictor

# Set page config
st.set_page_config(
    page_title="NextKey — Phục Hồi Tiếng Việt Viết Gọn",
    page_icon="⌨️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern, polished UI
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1e3c72, #2a5298, #ff6b6b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #6c757d;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .best-card {
        background: linear-gradient(135deg, #f0f7ff 0%, #e6f0fa 100%);
        border: 1px solid #b8daff;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .best-text {
        font-size: 1.6rem;
        font-weight: 700;
        color: #004085;
        margin: 0.4rem 0;
    }
    .stat-badge {
        display: inline-block;
        background-color: #e2e3e5;
        color: #383d41;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    .success-badge {
        background-color: #d4edda;
        color: #155724;
    }
    .info-badge {
        background-color: #d1ecf1;
        color: #0c5460;
    }
    .warn-badge {
        background-color: #fff3cd;
        color: #856404;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_predictor():
    """Load in-memory predictor instance."""
    checkpoint = Path("artifacts/phase2/width_xxxs/best_model.pt")
    vocab = Path("artifacts/phase2/width_xxxs/vocab.json")
    if not checkpoint.exists() or not vocab.exists():
        return None
    return NextKeyPredictor(checkpoint_path=checkpoint, vocab_path=vocab)


# Sidebar Configuration
with st.sidebar:
    st.image("https://img.shields.io/badge/Model-Width--XXXS-blue?style=for-the-badge&logo=pytorch", use_container_width=True)
    st.markdown("### ⚙️ Cấu hình Hệ thống")

    mode = st.radio(
        "Chế độ thực thi (Inference Mode):",
        ["⚡ Trực tiếp (Direct In-Memory)", "🌐 REST API Backend"],
        index=0,
    )

    api_url = "http://localhost:8000"
    if "REST API" in mode:
        api_url = st.text_input("Backend API URL:", value="http://localhost:8000")
        try:
            r = requests.get(f"{api_url}/health", timeout=1.0)
            if r.status_code == 200:
                st.success("🟢 Đã kết nối Backend API")
            else:
                st.warning(f"🟡 Backend phản hồi mã {r.status_code}")
        except Exception:
            st.error("🔴 Không thể kết nối Backend (Hãy chạy `python scripts/run_be.py`)")

    st.markdown("---")
    st.markdown("### 🎛️ Tham số Dự đoán")
    top_k = st.slider("Số lượng ứng viên (Top-K):", min_value=1, max_value=5, value=3)
    boundary_threshold = st.slider(
        "Ngưỡng tách từ (Boundary Threshold):",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="Xác suất kích hoạt khoảng cách trắng trước ký tự",
    )

    st.markdown("---")
    st.markdown("### 📊 Thông số Model Width-XXXS")
    st.markdown(
        """
        - **Kiến trúc**: BiGRU Dual-Head
        - **Số tham số**: `17,828 (~17.8K)`
        - **Kích thước**: `~70 KB (FP32) / ~18 KB (INT8)`
        - **Boundary F1**: `97.27% (In-domain)`
        - **Ứng dụng**: Bàn phím Edge / Mobile
        """
    )


# Main Interface
st.markdown('<div class="main-title">⌨️ NextKey — Vietnamese Compact Text Restoration</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Khôi phục văn bản tiếng Việt viết liền, không dấu, viết tắt thành câu tiếng Việt chuẩn hoàn chỉnh bằng mô hình siêu nhỏ <b>NextKey Width-XXXS (~17.8K tham số)</b>.</div>',
    unsafe_allow_html=True,
)

# Preset Samples
st.markdown("**💡 Câu mẫu thử nghiệm nhanh (Bấm để chọn):**")
sample_cols = st.columns(4)

samples = [
    "toidanghoc",
    "nguoivietnamyeunuoc",
    "hanoithudo",
    "chucmungnammoiankhangthinhvuong",
    "thanhphohochiminh",
    "congnghethongtin",
    "hocsinhchihocgioi",
    "chuctatcacacbanmotngaytothanh",
]

if "input_text" not in st.session_state:
    st.session_state.input_text = "toidanghoc"

for idx, sample in enumerate(samples):
    col = sample_cols[idx % 4]
    if col.button(f"📌 {sample}", key=f"sample_{idx}", use_container_width=True):
        st.session_state.input_text = sample
        st.rerun()

# Text input
input_text = st.text_input(
    "Nhập chuỗi tiếng Việt viết gọn (không dấu, viết liền hoặc có lỗi):",
    value=st.session_state.input_text,
    placeholder="Ví dụ: toidanghoc, nguoivietnamyeunuoc, hanoithudo...",
    key="main_input",
)

col_btn1, col_btn2 = st.columns([1, 5])
with col_btn1:
    submit = st.button("🚀 Khôi phục", type="primary", use_container_width=True)

if submit or input_text:
    if not input_text.strip():
        st.warning("Vui lòng nhập văn bản cần khôi phục.")
    else:
        with st.spinner("Đang xử lý khôi phục..."):
            best_text = ""
            candidates = []
            char_details = []
            latency_ms = 0.0
            compact_input = ""
            model_info = {}

            if "REST API" in mode:
                try:
                    res = requests.post(
                        f"{api_url}/restore",
                        json={
                            "input": input_text,
                            "top_k": top_k,
                            "boundary_threshold": boundary_threshold,
                        },
                        timeout=5.0,
                    )
                    if res.status_code == 200:
                        data = res.json()
                        best_text = data["best"]
                        candidates = data.get("candidates", [])
                        char_details = data.get("char_details", [])
                        latency_ms = data.get("latency_ms", 0.0)
                        compact_input = data.get("compact_input", "")
                        model_info = data.get("model", {})
                    else:
                        st.error(f"Lỗi từ API Backend ({res.status_code}): {res.text}")
                except Exception as e:
                    st.error(f"Không thể kết nối Backend API: {e}. Vui lòng chuyển sang chế độ 'Trực tiếp (Direct In-Memory)'.")
            else:
                pred = load_predictor()
                if pred is None:
                    st.error("Không tìm thấy checkpoint mô hình tại `artifacts/phase2/width_xxxs/`!")
                else:
                    pred_res = pred.restore(
                        input_text,
                        top_k=top_k,
                        boundary_threshold=boundary_threshold,
                    )
                    best_text = pred_res.best_text
                    candidates = pred_res.candidates
                    latency_ms = pred_res.latency_ms
                    compact_input = pred_res.compact_input
                    model_info = pred.get_metadata()
                    char_details = [
                        {
                            "index": cd.index,
                            "input_char": cd.input_char,
                            "predicted_char": cd.predicted_char,
                            "boundary_flag": cd.boundary_flag,
                            "boundary_prob": cd.boundary_prob,
                            "diacritic_prob": cd.diacritic_prob,
                        }
                        for cd in pred_res.char_details
                    ]

            if best_text or candidates:
                # 1. Best Result Card
                st.markdown(
                    f"""
                    <div class="best-card">
                        <div style="font-size: 0.9rem; color: #495057; font-weight: 600;">✨ KẾT QUẢ KHÔI PHỤC TỐT NHẤT:</div>
                        <div class="best-text">{best_text}</div>
                        <div>
                            <span class="stat-badge success-badge">⚡ Độ trễ: {latency_ms:.2f} ms</span>
                            <span class="stat-badge info-badge">📦 Model: Width-XXXS (~17.8K params)</span>
                            <span class="stat-badge warn-badge">📏 Độ dài: {len(compact_input)} ký tự</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # 2. Tabs for deep inspection
                tab1, tab2, tab3 = st.tabs(["📋 Danh sách ứng viên (Top-K)", "🔍 Phân tích từng ký tự & Dấu cách", "📈 Thông số kỹ thuật"])

                with tab1:
                    if candidates:
                        for cand in candidates:
                            c_rank = cand.get("rank", 1)
                            c_text = cand.get("text", "")
                            c_score = cand.get("score", 0.0)
                            c1, c2, c3 = st.columns([1, 6, 2])
                            c1.markdown(f"**#{c_rank}**")
                            c2.markdown(f"`{c_text}`")
                            c3.progress(min(max(float(c_score), 0.0), 1.0), text=f"Điểm: {c_score:.2%}")
                    else:
                        st.info("Không có ứng viên phụ.")

                with tab2:
                    st.markdown("**Bảng phân tách ký tự (Diacritic Head) và cờ tách từ (Boundary Head):**")
                    if char_details:
                        df_details = pd.DataFrame(char_details)
                        df_details = df_details.rename(
                            columns={
                                "index": "Vị trí",
                                "input_char": "Ký tự vào",
                                "predicted_char": "Ký tự dự đoán",
                                "boundary_flag": "Dấu cách trước",
                                "boundary_prob": "Xác suất dấu cách",
                                "diacritic_prob": "Độ tự tin ký tự",
                            }
                        )
                        # Format percentages
                        df_details["Xác suất dấu cách"] = df_details["Xác suất dấu cách"].apply(lambda x: f"{x*100:.1f}%")
                        df_details["Độ tự tin ký tự"] = df_details["Độ tự tin ký tự"].apply(lambda x: f"{x*100:.1f}%")
                        st.dataframe(df_details, use_container_width=True, hide_index=True)
                    else:
                        st.info("Không có dữ liệu chi tiết.")

                with tab3:
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Thời gian suy luận (Latency)", f"{latency_ms:.2f} ms")
                    m2.metric("Số tham số (Parameters)", "17,828 (~17.8K)")
                    m3.metric("Dung lượng Checkpoint", "75.1 KB (FP32)")
                    m4.metric("Throughput ước tính", f"{int(len(compact_input) / max(latency_ms/1000, 0.0001)):,} chars/s")
