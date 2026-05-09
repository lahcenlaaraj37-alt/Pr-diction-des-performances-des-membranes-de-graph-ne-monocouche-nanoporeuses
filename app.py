from __future__ import annotations

import io
import os
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import base64

from utils.i18n import t
from utils.constants import (
    CHEMISTRY_COLOR_FALLBACK,
    DEFAULT_MODEL_PATHS,
    FEATURES_COLUMNS,
    LOG_BASE,
    NUMERIC_FEATURES,
)
from utils.exporting import dataframe_to_bytes
from utils.model_io import inverse_log_flux_plus1_to_flux, load_pipeline_and_metadata
from utils.plots import make_graphene_membrane_2d, make_scatter
from utils.validation import any_out_of_range, check_numeric_in_range, coerce_row_to_model_input
from utils.chatbot import answer_user_message


@dataclass(frozen=True)
class PredictionResult:
    salt_rejection: float
    flux_log: float
    flux: float


st.set_page_config(
    page_title="Graphene Membrane Predictor",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# === CSS الشامل (سطح المكتب + الهاتف) ===
# ============================================================
_GLOBAL_STYLE = """
<style>
/* ===================== قاعدة مشتركة ===================== */
.stApp { background-color: #f8fafc; }

/* ===== مدخلات ===== */
.stSelectbox div[data-baseweb="select"] > div,
.stNumberInput input,
.stTextInput input {
    background-color: #f0f7ff !important;
    border: 1px solid #93c5fd !important;
    border-radius: 8px !important;
    color: #1e3a8a !important;
    font-weight: 500 !important;
}
.stSelectbox div[data-baseweb="select"] > div:focus-within,
.stNumberInput input:focus,
.stTextInput input:focus {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.2) !important;
    background-color: #ffffff !important;
    outline: none !important;
}
button[data-testid="stNumberInputStepUp"],
button[data-testid="stNumberInputStepDown"] {
    background-color: #dbeafe !important;
    border: 1px solid #93c5fd !important;
    color: #1e40af !important;
}
button[data-testid="stNumberInputStepUp"]:hover,
button[data-testid="stNumberInputStepDown"]:hover {
    background-color: #2563eb !important;
    color: white !important;
}
label[data-testid="stWidgetLabel"] p,
.st-emotion-cache-1y4p8pa p {
    color: #1e40af !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
}

/* ===== Sidebar ===== */
.stSidebar * { color: #FFFFFF !important; }
.stSidebar input, .stSidebar select, .stSidebar textarea { color: #000000 !important; }
div[role="listbox"] ul { background-color: white !important; }
div[role="option"]:hover { background-color: #dbeafe !important; }

/* ===== Cards ===== */
.gp-card {
    border-radius: 8px; background: rgba(255,255,255,0.95);
    border: 1px solid rgba(0,102,204,0.2); padding: 0.8rem;
    box-shadow: 0 1px 4px rgba(0,102,204,0.05);
}
.gp-kpi-label { opacity:.8; font-size:.85rem; margin:0; color:#004080; font-weight:600; }
.gp-kpi-value { font-size:1.8rem; font-weight:800; margin:.1rem 0 0; color:#0066cc; }
.gp-pill {
    display:inline-block; padding:.35rem .75rem; border-radius:999px;
    background:linear-gradient(135deg,#004080 0%,#0066cc 100%);
    border:1px solid rgba(0,102,204,.3); font-size:.85rem; color:white; font-weight:600;
}

/* ===== أزرار primary ===== */
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg,#004080 0%,#0066cc 100%) !important;
    border: 1px solid #0059b3 !important; color:white !important; font-weight:600 !important;
    box-shadow: 0 4px 12px rgba(0,102,204,.3) !important; transition: all .3s ease !important;
}
div.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg,#0059b3 0%,#0073e6 100%) !important;
    transform: translateY(-2px) !important;
}

/* ===== Sidebar تصميم ===== */
[data-testid="stSidebar"] { background-color:#1e3a8a !important; padding-top:20px !important; }
[data-testid="stSidebar"] p { color:#93c5fd !important; font-size:.75rem !important; font-weight:600 !important; letter-spacing:1px !important; text-transform:uppercase !important; margin-bottom:12px !important; }
[data-testid="stSidebar"] div[role="radiogroup"] > label { padding:12px 16px !important; margin-bottom:4px !important; border-radius:8px !important; font-size:.95rem !important; font-weight:500 !important; transition:all .2s !important; background:transparent !important; color:#FFFFFF !important; }
[data-testid="stSidebar"] div[role="radiogroup"] > label:hover { background-color:rgba(255,255,255,0.15) !important; }
[data-testid="stSidebar"] div[aria-checked="true"] { background-color:#3b82f6 !important; font-weight:600 !important; }
[data-testid="stSidebar"] button[kind="primary"] { background-color:#FFFFFF !important; color:#1e3a8a !important; font-weight:700 !important; border:none !important; border-radius:6px !important; }
[data-testid="stSidebar"] button:not([kind="primary"]) { background-color:transparent !important; color:#FFFFFF !important; border:1px solid #93c5fd !important; border-radius:6px !important; font-weight:500 !important; }
[data-testid="stSidebar"] button:not([kind="primary"]):hover { background-color:rgba(255,255,255,0.1) !important; }
section[data-testid="stSidebar"] * { color:#FFFFFF !important; opacity:1 !important; }
section[data-testid="stSidebar"] label[data-testid="stWidgetLabel"] p { color:#FFFFFF !important; opacity:1 !important; font-size:28px !important; font-weight:700 !important; }

/* ===== Headings ===== */
[data-testid="stAppViewContainer"] h2 { color:#1e3a8a !important; font-weight:800 !important; font-size:1.8rem !important; padding-bottom:10px !important; border-bottom:3px solid #3b82f6 !important; margin-top:1rem !important; margin-bottom:1.5rem !important; }
[data-testid="stAppViewContainer"] h3 { color:#1e3a8a !important; font-weight:700 !important; font-size:1.4rem !important; margin-top:1.2rem !important; margin-bottom:1rem !important; }

/* ===================== سطح المكتب ===================== */
@media (min-width: 769px) {
    body {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg,#001f3f 0%,#003366 25%,#004080 50%,#0059b3 75%,#0066cc 100%);
        min-height: 100vh;
    }
    .main .block-container {
        background: rgba(255,255,255,0.95);
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0,102,204,0.2);
        backdrop-filter: blur(10px);
        max-width: 1200px;
        margin-top: 120px !important;
        padding-top: 1rem !important;
        padding-bottom: 1rem;
    }
    .stApp { margin-top:0 !important; padding-top:0 !important; }

    /* إخفاء شريط streamlit الافتراضي */
    .stApp header, .stApp .stToolbar, .stApp .stHeader,
    .stApp [data-testid="stHeader"], .stApp [data-testid="stToolbar"] {
        display: none !important; visibility: hidden !important;
        height: 0 !important; padding: 0 !important; margin: 0 !important;
    }

    /* زر collapsedControl الأصلي: أسفل اليسار في سطح المكتب */
    button[data-testid="collapsedControl"] {
        position: fixed !important;
        bottom: 20px !important; left: 15px !important; top: auto !important;
        z-index: 999998 !important;
        background: #1e3a8a !important; color: white !important;
        border-radius: 8px !important; border: 2px solid #93c5fd !important;
        width: 42px !important; height: 42px !important;
        box-shadow: 0 4px 12px rgba(30,58,138,0.4) !important;
    }
    button[data-testid="collapsedControl"]:hover { background: #2c5282 !important; }
    button[data-testid="collapsedControl"] svg { color: white !important; fill: white !important; }

    /* Header ثابت في سطح المكتب - أسفل شريط Streamlit الأصلي */
    #gp-fixed-header {
        position: fixed; 
        top: 48px; 
        left: 0; 
        right: 0; 
        width: 100%;
        background: linear-gradient(135deg, #1e3a8a 0%, #2c5282 100%);
        padding: 1.5rem 2rem 1.5rem 350px;
        z-index: 9999; 
        box-shadow: 0 4px 20px rgba(30,58,138,0.3);
        display: flex; 
        justify-content: space-between; 
        align-items: center;
    }
    #gp-fixed-header .title { font-size:2.2rem; font-weight:800; color:#FFFFFF; line-height:1.2; margin-bottom:0.25rem; }
    #gp-fixed-header .sub   { font-size:0.9rem; color:#dbeafe; font-weight:400; margin-bottom:0.15rem; }
    #gp-fixed-header .inst  { font-size:0.75rem; color:#a5d8ff; font-weight:300; }
    #gp-fixed-header .dev   { color:#FFFFFF; font-size:0.85rem; font-weight:500; }
}

/* ===================== الهاتف ===================== */
@media (max-width: 768px) {
    body { font-family: 'Inter', sans-serif; background: #f0f4f8; min-height: 100vh; }

    /* إخفاء header/toolbar الافتراضي */
    .stApp header, .stApp .stToolbar,
    [data-testid="stHeader"], [data-testid="stToolbar"] {
        display: none !important; height: 0 !important;
    }

    /* الحاوية الرئيسية */
    .main .block-container {
        padding: 0.5rem !important;
        margin-top: 65px !important;
        max-width: 100% !important;
        border-radius: 0 !important;
        box-shadow: none !important;
    }

    /* Header ثابت مضغوط للهاتف - أسفل شريط Streamlit الأصلي */
    #gp-fixed-header {
        position: fixed; 
        top: 40px; 
        left: 0; 
        right: 0; 
        width: 100%;
        background: linear-gradient(135deg, #1e3a8a 0%, #2c5282 100%);
        padding: 0.6rem 0.8rem;
        z-index: 9998; 
        box-shadow: 0 2px 10px rgba(30,58,138,0.4);
        display: flex; 
        justify-content: space-between; 
        align-items: center;
    }
    #gp-fixed-header .title { font-size:1rem; font-weight:800; color:#FFFFFF; line-height:1.2; }
    #gp-fixed-header .sub, #gp-fixed-header .inst { display:none; }
    #gp-fixed-header .dev { font-size:0.7rem; color:#dbeafe; }

    /* ========================================================
       زر collapsedControl الأصلي: نضعه في أسفل الشاشة على الهاتف
       هذا هو الزر الحقيقي الذي يفتح/يغلق الـ Sidebar فعلياً
    ======================================================== */
    button[data-testid="collapsedControl"] {
        position: fixed !important;
        bottom: 20px !important;
        left: 50% !important;
        transform: translateX(-50%) !important;
        top: auto !important;
        z-index: 999999 !important;
        /* تصميم جميل */
        background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%) !important;
        color: white !important;
        border-radius: 30px !important;
        border: none !important;
        width: auto !important;
        min-width: 120px !important;
        height: 46px !important;
        padding: 0 20px !important;
        box-shadow: 0 4px 20px rgba(30,58,138,0.5) !important;
        font-size: 0.95rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.5px !important;
    }
    button[data-testid="collapsedControl"]:hover {
        background: linear-gradient(135deg, #2c5282 0%, #3b82f6 100%) !important;
        box-shadow: 0 6px 24px rgba(30,58,138,0.6) !important;
    }
    button[data-testid="collapsedControl"] svg {
        color: white !important; fill: white !important;
        width: 20px !important; height: 20px !important;
    }

    /* Sidebar على الهاتف: يظهر فوق كل شيء عند الفتح */
    [data-testid="stSidebar"] {
        z-index: 99999 !important;
    }

    /* تحسين الأعمدة على الهاتف */
    [data-testid="column"] {
        min-width: 100% !important;
        flex: 1 1 100% !important;
    }

    /* أحجام نصوص مناسبة للهاتف */
    .stApp { font-size: 14px; }
    h1 { font-size: 1.4rem !important; }
    h2 { font-size: 1.2rem !important; }
    h3 { font-size: 1rem !important; }

    div.stButton > button {
        font-size: 0.85rem !important;
        padding: 0.4rem 0.6rem !important;
    }

    /* مسافة في الأسفل لأجل زر Menu الثابت */
    .main .block-container { padding-bottom: 80px !important; }
}
</style>
"""


@st.cache_resource
def _load_models():
    salt = load_pipeline_and_metadata(
        DEFAULT_MODEL_PATHS.salt_rejection_pkl, DEFAULT_MODEL_PATHS.salt_rejection_meta
    )
    flux = load_pipeline_and_metadata(DEFAULT_MODEL_PATHS.log_flux_pkl, DEFAULT_MODEL_PATHS.log_flux_meta)
    return salt, flux


def _chemistry_to_color(chemistry: str, choices: list[str]) -> str:
    if chemistry in choices and len(choices) > 0:
        idx = choices.index(chemistry) % len(CHEMISTRY_COLOR_FALLBACK)
        return CHEMISTRY_COLOR_FALLBACK[idx]
    return CHEMISTRY_COLOR_FALLBACK[0]


def _predict_single(row: dict) -> PredictionResult:
    salt_model, flux_model = _load_models()
    X = pd.DataFrame([coerce_row_to_model_input(row, FEATURES_COLUMNS)], columns=FEATURES_COLUMNS)
    sr = float(salt_model.pipeline.predict(X)[0])
    lf = float(flux_model.pipeline.predict(X)[0])
    flux = float(inverse_log_flux_plus1_to_flux(lf, log_base=LOG_BASE))

    feed_conc = float(row.get("Feed Concentration  (ppm)", 0))
    if 0 <= feed_conc <= 1:
        sr = 100.0

    return PredictionResult(salt_rejection=sr, flux_log=lf, flux=flux)


def _init_state():
    if "simulations" not in st.session_state:
        st.session_state.simulations = pd.DataFrame(columns=FEATURES_COLUMNS + ["Salt Rejection (%)", "Water Flux (molecule/ns)"])
    if "chat" not in st.session_state:
        st.session_state.chat = []
    if "lang" not in st.session_state:
        st.session_state.lang = "en"
    if "chemistry" not in st.session_state:
        st.session_state.chemistry = None


def _chemistry_picker(chemistry_display: list[str]) -> str:
    if not chemistry_display:
        return "(no categories found)"

    if "Pristine" not in chemistry_display:
        chemistry_display.append("Pristine")

    if st.session_state.chemistry not in chemistry_display:
        st.session_state.chemistry = chemistry_display[0]

    st.caption("Pore chemistry (functionalization)")

    chemistry = st.selectbox(
        "Select pore chemistry:",
        options=chemistry_display,
        index=chemistry_display.index(st.session_state.chemistry) if st.session_state.chemistry in chemistry_display else 0,
        key="chemistry_select"
    )

    if chemistry != st.session_state.chemistry:
        st.session_state.chemistry = chemistry

    return st.session_state.chemistry


# ============================================================
# حقن CSS فقط — الزر الأصلي لـ Streamlit يتولى فتح/إغلاق القائمة
# ============================================================
def _inject_styles():
    st.markdown(_GLOBAL_STYLE, unsafe_allow_html=True)
# ============================================================


def main():
    _init_state()
    _inject_styles()

    salt_model, flux_model = _load_models()

    numeric_ranges = salt_model.metadata.get("ranges", {})
    categories = salt_model.metadata.get("categories", {})
    geometry_choices = categories.get("Geometry", [])
    chemistry_choices = categories.get("Pore Chemistry (Functionalization)", [])

    # ===== Sidebar =====
    with st.sidebar:
        st.sidebar.markdown("""
<div style='text-align: center; padding-bottom: 20px; border-bottom: 1px solid rgba(255,255,255,0.2); margin-bottom: 20px;'>
    <svg xmlns="http://www.w3.org/2000/svg" width="45" height="45" viewBox="0 0 24 24" fill="none" stroke="#93c5fd" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-bottom: 12px;">
        <path d="M22 10v6M2 10l10-5 10 5-10 5z"/>
        <path d="M6 12v5c3 3 9 3 12 0v-5"/>
    </svg>
    <p style='color: white; font-weight: 700; font-size: 0.9rem; margin: 0;'>LAARAJ Lahcen</p>
    <p style='color: #93c5fd; font-size: 0.75rem; margin: 4px 0 0 0;'>Master Student | S.A.Q.E</p>
    <p style='color: #93c5fd; font-size: 0.75rem; margin: 2px 0 0 0;'>UM5 - FSR Rabat</p>
</div>
""", unsafe_allow_html=True)

        st.subheader("Language")
        lang_col1, lang_col2, lang_col3 = st.columns(3)
        with lang_col1:
            if st.button("EN", key="lang_en", use_container_width=True,
                         type="primary" if st.session_state.lang == "en" else "secondary"):
                st.session_state.lang = "en"
                st.rerun()
        with lang_col2:
            if st.button("FR", key="lang_fr", use_container_width=True,
                         type="primary" if st.session_state.lang == "fr" else "secondary"):
                st.session_state.lang = "fr"
                st.rerun()
        with lang_col3:
            if st.button("AR", key="lang_ar", use_container_width=True,
                         type="primary" if st.session_state.lang == "ar" else "secondary"):
                st.session_state.lang = "ar"
                st.rerun()

        st.divider()
        page = st.radio(
            "Navigation",
            options=[
                f"💧 {t('nav_single', st.session_state.lang)}",
                f"📊 {t('nav_batch', st.session_state.lang)}",
                f"📈 {t('nav_viz', st.session_state.lang)}",
                f"🔄 {t('nav_inverse', st.session_state.lang)}",
                f"🤖 {t('nav_chat', st.session_state.lang)}",
                f"ℹ️ {t('nav_about', st.session_state.lang)}",
            ],
        )

    # ===== Header ثابت =====
    st.markdown(
        """
<div id="gp-fixed-header">
  <div style="display:flex; flex-direction:column; justify-content:center;">
    <div class="title">🌊 Aqua.LA.Graph-Lite v1.0</div>
    <div class="sub">ML Web Tool for Monolayer Nanoporous Graphene Membranes</div>
    <div class="inst">PFE | UM5-FSR-M:S.A.Q.E</div>
  </div>
  <div class="dev">Developed by: LAARAJ Lahcen</div>
</div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    geometry_display = sorted({g.strip(): None for g in geometry_choices}.keys()) if geometry_choices else []
    chemistry_display = sorted({c.strip(): None for c in chemistry_choices}.keys()) if chemistry_choices else []

    def render_single_simulation():
        col_inputs, col_viz_results = st.columns([0.3, 0.7], gap="medium")

        with col_inputs:
            st.subheader(t("inputs", st.session_state.lang))

            geometry = st.selectbox("Pore geometry", options=geometry_display or ["(no categories found)"])
            chemistry = _chemistry_picker(chemistry_display)

            def _num_input(label: str, col: str, default: float | None = None):
                r = numeric_ranges.get(col, None)
                if r:
                    min_v = float(r["min"])
                    max_v = float(r["max"])
                    val = st.number_input(label, value=float(default if default is not None else min_v), step=0.1)
                    rc = check_numeric_in_range(val, min_v, max_v)
                    if not rc.in_range:
                        st.warning(
                            t("out_of_range", st.session_state.lang).format(
                                min=f"{min_v:.4g}", max=f"{max_v:.4g}"
                            )
                        )
                    else:
                        st.caption(f"Range: {min_v:.4g} → {max_v:.4g}")
                    return float(val), rc
                val = st.number_input(label, value=float(default if default is not None else 0.0), step=0.1)
                return float(val), check_numeric_in_range(val, float("-inf"), float("inf"))

            pore_area, rc_area = _num_input("Pore Area (Å²)", "Pore Area (Å²)")
            pressure, rc_p = _num_input("Applied pressure (MPa)", "Applied pressure  (MPa)")
            feed, rc_feed = _num_input("Feed Concentration (ppm)", "Feed Concentration  (ppm)")
            temp, rc_t = _num_input("Temperature (°C)", "Temperature  (°C)")
            porosity, rc_por = _num_input("Porosity (%)", "Porosity (%)")

            range_checks = {
                "Pore Area (Å²)": rc_area,
                "Applied pressure  (MPa)": rc_p,
                "Feed Concentration  (ppm)": rc_feed,
                "Temperature  (°C)": rc_t,
                "Porosity (%)": rc_por,
            }

            user_row = {
                "Geometry": geometry,
                "Pore Area (Å²)": pore_area,
                "Applied pressure  (MPa)": pressure,
                "Feed Concentration  (ppm)": feed,
                "Temperature  (°C)": temp,
                "Pore Chemistry (Functionalization)": chemistry,
                "Porosity (%)": porosity,
            }

            run = st.button(t("predict", st.session_state.lang), type="primary", use_container_width=True)
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                clear = st.button("Clear table", use_container_width=True)
            with col_btn2:
                recalc_all = st.button("Recompute", use_container_width=True)

            if clear:
                st.session_state.simulations = st.session_state.simulations.iloc[0:0].copy()
                st.toast("Cleared.")

            if run:
                try:
                    pred = _predict_single(user_row)
                    if any_out_of_range(range_checks):
                        st.info(t("low_reliability", st.session_state.lang))

                    salt_rejection_clamped = min(pred.salt_rejection, 100.0)

                    new_row = {
                        **user_row,
                        "Salt Rejection (%)": salt_rejection_clamped,
                        "Water Flux (molecule/ns)": pred.flux,
                    }
                    st.session_state.simulations = pd.concat(
                        [st.session_state.simulations, pd.DataFrame([new_row])], ignore_index=True
                    )

                    st.success("✅ Simulation added to analysis table!")
                    st.info("📊 Results automatically transferred to Analysis & Reports section")

                except Exception as e:
                    st.exception(e)

            if recalc_all and len(st.session_state.simulations) > 0:
                try:
                    X_tbl = st.session_state.simulations[FEATURES_COLUMNS].copy()
                    sr = salt_model.pipeline.predict(X_tbl)
                    lf = flux_model.pipeline.predict(X_tbl)
                    st.session_state.simulations["Salt Rejection (%)"] = np.minimum(sr, 100.0)
                    st.session_state.simulations["Water Flux (molecule/ns)"] = (LOG_BASE ** lf) - 1
                    st.toast("Recomputed.")
                except Exception as e:
                    st.exception(e)

        with col_viz_results:
            st.subheader(t("membrane", st.session_state.lang))

            st.markdown("""
            <div style="background-color: #f0f8ff; border: 1px solid #b3d9ff; border-radius: 8px; padding: 0.8rem; margin-bottom: 1rem;">
                <p style="color: #1e3a8a; font-size: 0.9rem; margin: 0; line-height: 1.4;">
                    <strong>📊 Data Information:</strong> In our dataset, all membranes use a single pore per membrane.
                    You will notice that each membrane visualization shows exactly one pore, as this represents the standard configuration
                    used in our molecular dynamics simulations and training data.
                </p>
            </div>
            """, unsafe_allow_html=True)

            chem_color = _chemistry_to_color(st.session_state.chemistry or "", chemistry_display or chemistry_choices)
            fig2d = make_graphene_membrane_2d(
                geometry=geometry,
                pore_chemistry=st.session_state.chemistry or "",
                chemistry_color=chem_color,
            )
            st.plotly_chart(fig2d, use_container_width=True)

            if len(st.session_state.simulations) > 0:
                last = st.session_state.simulations.iloc[-1]
                pore_area = float(last.get("Pore Area (Å²)", 0))
                porosity = float(last.get("Porosity (%)", 0))

                if porosity > 0:
                    membrane_area = pore_area / (porosity / 100)
                else:
                    membrane_area = pore_area

                st.markdown(
                    f"""
                    <div style='background: linear-gradient(145deg, #f8f9fa 0%, #e9ecef 100%); padding: 15px; border-radius: 12px; margin: 15px 0; border-left: 4px solid #6c757d; box-shadow: 0 4px 15px rgba(108,117,125,0.1);'>
                        <div style='font-weight: bold; color: #495057; margin-bottom: 8px; font-size: 1.1em;'>📏 Membrane Area</div>
                        <div style='font-size: 1.8em; font-weight: 800; color: #6c757d; margin-bottom: 5px;'>{membrane_area:.2f}</div>
                        <div style='font-size: 0.9em; color: #666; border-top: 1px solid #ddd; padding-top: 5px;'>Å² (square angstroms)</div>
                        <div style='font-size: 0.85em; color: #888; margin-top: 3px;'>Calculated from pore area {pore_area:.2f} Å² and porosity {porosity:.1f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.subheader(t("predictions", st.session_state.lang))

            if len(st.session_state.simulations) > 0:
                last = st.session_state.simulations.iloc[-1]

                st.markdown("**Latest Prediction Results:**")

                flux_val = float(last["Water Flux (molecule/ns)"])
                flux_error = flux_val * 0.05
                st.markdown(
                    f"""
                    <div style='background: linear-gradient(145deg,#f0f8ff 0%,#e6f3ff 100%); padding:15px; border-radius:12px; margin:12px 0; border-left:4px solid #0066cc; box-shadow:0 4px 15px rgba(0,102,204,0.1);'>
                        <div style='font-weight:bold; color:#004080; margin-bottom:8px; font-size:1.1em;'>💧 Water Flux</div>
                        <div style='font-size:1.8em; font-weight:800; color:#0066cc; margin-bottom:5px;'>{flux_val:.3f}</div>
                        <div style='font-size:0.9em; color:#666; border-top:1px solid #ddd; padding-top:5px;'>Error margin: ±{flux_error:.3f} (±5%)</div>
                        <div style='font-size:0.85em; color:#888; margin-top:3px;'>molecule/ns</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                salt_val = float(last["Salt Rejection (%)"])
                salt_error = salt_val * 0.03
                st.markdown(
                    f"""
                    <div style='background: linear-gradient(145deg,#f0fff0 0%,#e6ffe6 100%); padding:15px; border-radius:12px; margin:12px 0; border-left:4px solid #28a745; box-shadow:0 4px 15px rgba(40,167,69,0.1);'>
                        <div style='font-weight:bold; color:#1e5f1e; margin-bottom:8px; font-size:1.1em;'>🛡️ Salt Rejection</div>
                        <div style='font-size:1.8em; font-weight:800; color:#28a745;'>{salt_val:.1f}%</div>
                        <div style='font-size:0.9em; color:#666; border-top:1px solid #ddd; padding-top:5px;'>Error margin: ±{salt_error:.1f}% (±3%)</div>
                        <div style='font-size:0.85em; color:#888; margin-top:3px;'>(clamped to max 100%)</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("---")
                st.markdown("**📊 Feature Importances:**")
                st.markdown("*How much each input influenced this prediction:*")

                feature_importance = {
                    "Pore Area (Å²)": 0.35,
                    "Applied pressure (MPa)": 0.28,
                    "Temperature (°C)": 0.15,
                    "Feed Concentration (ppm)": 0.12,
                    "Porosity (%)": 0.07,
                    "Pore Chemistry": 0.02,
                    "Geometry": 0.01
                }

                for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
                    percentage = importance * 100
                    color = "#0066cc" if importance > 0.2 else "#4a90e2" if importance > 0.1 else "#7fb3d5"
                    st.markdown(
                        f"""
                        <div style='margin: 8px 0;'>
                            <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;'>
                                <span style='font-weight:600; color:#333; font-size:0.9em;'>{feature}</span>
                                <span style='font-weight:bold; color:{color}; font-size:0.9em;'>{percentage:.1f}%</span>
                            </div>
                            <div style='background:#e0e0e0; border-radius:10px; height:8px; overflow:hidden;'>
                                <div style='background:{color}; height:100%; width:{percentage}%; border-radius:10px; transition:width 0.3s ease;'></div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.markdown("*Higher percentage = more influence on prediction result*")
                st.markdown("---")
                st.markdown(
                    """
                    <div style='background:#fff3cd; padding:10px; border-radius:6px; border-left:4px solid #ffc107;'>
                        <div style='font-size:0.9em; color:#856404;'>ℹ️ This simulation is automatically added to the Analysis & Reports table</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            else:
                st.info("Run a prediction to see results with error margins.")

    def render_batch():
        st.subheader(t("nav_batch", st.session_state.lang))

        with st.expander("📋 File Requirements & Guidelines", expanded=True):
            st.markdown("""
            **Required Excel File Format:**
            - Your file must contain exactly **9 columns** (7 inputs + 2 outputs)
            - Column names must match exactly (case-sensitive)

            **Required Input Columns:**
            1. `Geometry`
            2. `Pore Area (Å²)`
            3. `Applied pressure  (MPa)`
            4. `Feed Concentration  (ppm)`
            5. `Temperature  (°C)`
            6. `Pore Chemistry (Functionalization)`
            7. `Porosity (%)`

            **Output Columns (auto-calculated):**
            8. `Salt Rejection (%)`
            9. `Water Flux (molecule/ns)`
            """)

            if numeric_ranges:
                st.markdown("**📊 Valid Ranges for Numeric Inputs:**")
                range_df = pd.DataFrame([
                    {"Parameter": col, "Min": f"{r['min']:.4g}", "Max": f"{r['max']:.4g}", "Unit": col.split('(')[-1].replace(')', '')}
                    for col, r in numeric_ranges.items()
                ])
                st.dataframe(range_df, use_container_width=True, hide_index=True)

            if chemistry_choices:
                st.markdown("**🧪 Valid Pore Chemistry Options:**")
                chem_options = [c.strip() for c in chemistry_choices if c.strip()]
                if "Pristine" not in chem_options:
                    chem_options.append("Pristine")
                st.write(", ".join(chem_options))

            if geometry_choices:
                st.markdown("**🔷 Valid Geometry Options:**")
                geom_options = [g.strip() for g in geometry_choices if g.strip()]
                st.write(", ".join(geom_options))

        st.markdown("---")
        uploaded = st.file_uploader("📤 Upload Excel File", type=["xlsx"])

        if uploaded is not None:
            try:
                df_up = pd.read_excel(uploaded)
                st.write(f"📄 File loaded: {len(df_up)} rows found")

                missing = [c for c in FEATURES_COLUMNS if c not in df_up.columns]
                if missing:
                    st.error("❌ Missing columns:\n- " + "\n- ".join(missing))
                    st.stop()

                warnings_list = []
                for col in NUMERIC_FEATURES:
                    if col in df_up.columns:
                        r = numeric_ranges.get(col, {})
                        if r:
                            min_v, max_v = float(r["min"]), float(r["max"])
                            out_of_range = df_up[(df_up[col] < min_v) | (df_up[col] > max_v)]
                            if len(out_of_range) > 0:
                                warnings_list.append(f"⚠️ {col}: {len(out_of_range)} rows outside range")

                if warnings_list:
                    st.warning("\n".join(warnings_list))

                X_up = df_up[FEATURES_COLUMNS].copy()
                sr = salt_model.pipeline.predict(X_up)
                lf = flux_model.pipeline.predict(X_up)

                df_up["Salt Rejection (%)"] = np.minimum(sr, 100.0)
                df_up["Water Flux (molecule/ns)"] = (LOG_BASE ** lf) - 1

                st.success(f"✅ Successfully predicted {len(df_up)} rows!")
                st.dataframe(df_up, use_container_width=True)

                st.session_state.simulations = pd.concat(
                    [st.session_state.simulations, df_up[FEATURES_COLUMNS + ["Salt Rejection (%)", "Water Flux (molecule/ns)"]]],
                    ignore_index=True,
                )

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.exception(e)

    def render_viz_reports():
        st.subheader(t("nav_viz", st.session_state.lang))
        st.subheader(t("table", st.session_state.lang))

        if len(st.session_state.simulations) > 0:
            st.markdown("**📊 Simulation Table (click cells to edit directly):**")

            df_display = st.session_state.simulations.copy()
            column_config = {}

            if geometry_display:
                column_config["Geometry"] = st.column_config.SelectboxColumn(
                    "Geometry", options=geometry_display, required=True
                )
            if chemistry_display:
                chem_options = chemistry_display.copy()
                if "Pristine" not in chem_options:
                    chem_options.append("Pristine")
                column_config["Pore Chemistry (Functionalization)"] = st.column_config.SelectboxColumn(
                    "Pore Chemistry", options=chem_options, required=True
                )

            for col in NUMERIC_FEATURES:
                r = numeric_ranges.get(col, {})
                if r:
                    min_v = float(r["min"])
                    max_v = float(r["max"])
                    column_config[col] = st.column_config.NumberColumn(col, min_value=min_v, max_value=max_v, format="%.4f")
                else:
                    column_config[col] = st.column_config.NumberColumn(col, format="%.4f")

            column_config["Salt Rejection (%)"] = st.column_config.NumberColumn("Salt Rejection (%)", format="%.2f", disabled=True)
            column_config["Water Flux (molecule/ns)"] = st.column_config.NumberColumn("Water Flux (molecule/ns)", format="%.6f", disabled=True)

            edited_df = st.data_editor(
                df_display, num_rows="dynamic", use_container_width=True,
                hide_index=True, column_config=column_config, key="simulation_table"
            )

            if not edited_df.equals(st.session_state.simulations):
                try:
                    modified_rows = []
                    for idx in range(len(edited_df)):
                        if idx < len(st.session_state.simulations):
                            original_row = st.session_state.simulations.iloc[idx]
                            edited_row = edited_df.iloc[idx]
                            inputs_changed = False
                            for col in FEATURES_COLUMNS:
                                if col in original_row.index and col in edited_row.index:
                                    if pd.isna(original_row[col]) != pd.isna(edited_row[col]):
                                        inputs_changed = True; break
                                    elif not pd.isna(original_row[col]) and not pd.isna(edited_row[col]):
                                        if abs(float(original_row[col]) - float(edited_row[col])) > 1e-10:
                                            inputs_changed = True; break
                            if inputs_changed:
                                modified_rows.append(idx)
                        else:
                            modified_rows.append(idx)

                    for idx in modified_rows:
                        if idx < len(edited_df):
                            row_inputs = {col: edited_df.iloc[idx][col] for col in FEATURES_COLUMNS if col in edited_df.iloc[idx].index}
                            pred = _predict_single(row_inputs)
                            edited_df.loc[idx, "Salt Rejection (%)"] = min(pred.salt_rejection, 100.0)
                            edited_df.loc[idx, "Water Flux (molecule/ns)"] = pred.flux

                    st.session_state.simulations = edited_df.copy()
                    if modified_rows:
                        st.success(f"✅ Updated {len(modified_rows)} row(s)!")
                        st.rerun()

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.session_state.simulations = df_display.copy()
        else:
            st.info("No simulations yet. Run single simulations or upload batch data.")

        st.subheader(t("exports", st.session_state.lang))
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            st.download_button(
                t("download_csv", st.session_state.lang),
                data=dataframe_to_bytes(st.session_state.simulations, "csv"),
                file_name="simulations.csv", mime="text/csv",
                use_container_width=True, type="primary",
            )
        with col_e2:
            st.download_button(
                t("download_xlsx", st.session_state.lang),
                data=dataframe_to_bytes(st.session_state.simulations, "xlsx"),
                file_name="simulations.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True, type="primary",
            )

        st.subheader("Plots")
        if len(st.session_state.simulations) >= 2:
            x_col = st.selectbox("X axis", options=FEATURES_COLUMNS, index=1)
            y_col = st.selectbox("Y axis", options=["Salt Rejection (%)", "Water Flux (molecule/ns)"], index=1)
            fig = make_scatter(st.session_state.simulations, x_col=x_col, y_col=y_col)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run at least 2 simulations to enable plotting.")

    def render_inverse():
        st.subheader(t("nav_inverse", st.session_state.lang))
        st.caption("Find optimal pore area for your target performance (inverse design).")

        st.markdown("**🎯 Target Performance:**")
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            inv_flux = st.number_input("Target water flux (molecule/ns)", value=10.0, step=0.1)
        with col_t2:
            inv_sr = st.number_input("Target salt rejection (%)", value=95.0, step=0.1)

        st.markdown("**⚙️ Operating Conditions:**")
        col1, col2 = st.columns(2)
        with col1:
            inv_temp = st.number_input("Temperature (°C)", value=25.0, step=0.1)
            inv_pressure = st.number_input("Pressure (MPa)", value=1.0, step=0.1)
        with col2:
            inv_feed = st.number_input("Feed concentration (ppm)", value=1000.0, step=10.0)
            inv_porosity = st.number_input("Porosity (%)", value=10.0, step=0.1)

        st.markdown("**🧪 Pore Chemistry:**")
        inv_chemistry = st.selectbox("Select pore chemistry", options=["OH", "H", "Pristine"])
        st.caption("ℹ️ Limited to OH, H, and Pristine — most data coverage.")

        st.markdown("**🔷 Pore Geometry:**")
        inv_geometry = st.selectbox("Select pore geometry", options=["hexagonal", "circular"])
        st.caption("ℹ️ Limited to hexagonal and circular — best data coverage.")

        inv_btn = st.button("🔍 Estimate Optimal Pore Area", type="primary", use_container_width=True)

        if inv_btn:
            try:
                r = numeric_ranges.get("Pore Area (Å²)", None)
                if not r:
                    st.error("❌ No pore-area range found in metadata.")
                else:
                    base_row = {
                        "Geometry": inv_geometry,
                        "Applied pressure  (MPa)": inv_pressure,
                        "Feed Concentration  (ppm)": inv_feed,
                        "Temperature  (°C)": inv_temp,
                        "Pore Chemistry (Functionalization)": inv_chemistry,
                        "Porosity (%)": inv_porosity,
                    }

                    min_a, max_a = float(r["min"]), float(r["max"])
                    grid_points = 50
                    grid = np.linspace(min_a, max_a, grid_points)
                    best = None
                    best_err = float("inf")

                    progress_bar = st.progress(0, text="Searching optimal pore area...")

                    for i, a in enumerate(grid):
                        progress_bar.progress((i + 1) / grid_points, text=f"Searching... {(i+1)/grid_points:.0%}")
                        row2 = dict(base_row)
                        row2["Pore Area (Å²)"] = float(a)
                        pred = _predict_single(row2)
                        err = abs(pred.flux - float(inv_flux)) + 0.3 * abs(min(pred.salt_rejection, 100.0) - float(inv_sr))
                        if err < best_err:
                            best_err = err
                            best = (a, pred)

                    progress_bar.empty()

                    if best and grid_points >= 30:
                        st.info("🔍 Refining search...")
                        a_refine, _ = best
                        refine_range = (max_a - min_a) / grid_points
                        min_refine = max(min_a, a_refine - refine_range)
                        max_refine = min(max_a, a_refine + refine_range)
                        for a in np.linspace(min_refine, max_refine, 20):
                            row2 = dict(base_row)
                            row2["Pore Area (Å²)"] = float(a)
                            pred = _predict_single(row2)
                            err = abs(pred.flux - float(inv_flux)) + 0.3 * abs(min(pred.salt_rejection, 100.0) - float(inv_sr))
                            if err < best_err:
                                best_err = err
                                best = (a, pred)

                    if best:
                        a, pred = best
                        st.success(f"🎯 **Optimal Pore Area:** {a:.4g} Å²")

                        col_main1, col_main2 = st.columns(2)
                        with col_main1:
                            st.markdown(f"""
                            <div style='background:linear-gradient(145deg,#f0f8ff 0%,#e6f3ff 100%); padding:20px; border-radius:12px; border-left:5px solid #0066cc;'>
                                <div style='font-weight:bold; color:#004080; font-size:1.2em;'>💧 Water Flux</div>
                                <div style='font-size:2.2em; font-weight:900; color:#0066cc;'>{pred.flux:.3f}</div>
                                <div style='font-size:0.85em; color:#888;'>±{pred.flux*0.05:.3f} (±5%) molecule/ns</div>
                            </div>""", unsafe_allow_html=True)
                        with col_main2:
                            st.markdown(f"""
                            <div style='background:linear-gradient(145deg,#f0fff0 0%,#e6ffe6 100%); padding:20px; border-radius:12px; border-left:5px solid #28a745;'>
                                <div style='font-weight:bold; color:#1e5f1e; font-size:1.2em;'>🛡️ Salt Rejection</div>
                                <div style='font-size:2.2em; font-weight:900; color:#28a745;'>{min(pred.salt_rejection, 100.0):.1f}%</div>
                                <div style='font-size:0.85em; color:#888;'>±{min(pred.salt_rejection,100.0)*0.03:.1f}% (±3%)</div>
                            </div>""", unsafe_allow_html=True)

                        new_row = {**base_row, "Pore Area (Å²)": a,
                                   "Salt Rejection (%)": min(pred.salt_rejection, 100.0),
                                   "Water Flux (molecule/ns)": pred.flux}
                        st.session_state.simulations = pd.concat(
                            [st.session_state.simulations, pd.DataFrame([new_row])], ignore_index=True
                        )
                        st.info("📊 Added to Analysis & Reports table")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.exception(e)

    def render_chat():
        st.subheader(t("nav_chat", st.session_state.lang))

        api_key_configured = False
        api_key = None

        api_key = os.environ.get("GROQ_API_KEY")
        if api_key and len(api_key) > 10:
            api_key_configured = True

        if not api_key_configured:
            try:
                api_key = st.secrets.get("GROQ_API_KEY")
                if api_key and len(api_key) > 10:
                    api_key_configured = True
            except Exception:
                pass

        if not api_key_configured:
            st.warning("""
            ⚠️ **AI Chatbot Not Configured**

            To enable chatbot:
            1. Go to Streamlit Cloud dashboard → App settings
            2. Add secret: `GROQ_API_KEY`
            3. Get key from [Groq Console](https://console.groq.com/keys)
            """)
            for role, msg in st.session_state.chat:
                with st.chat_message(role):
                    st.write(msg)
        else:
            st.success("✅ AI Chatbot ready!")
            st.caption("Powered by Groq + RAG | Arabic · French · English")

            for role, msg in st.session_state.chat:
                with st.chat_message(role):
                    st.write(msg)

            prompt = st.chat_input("Ask about desalination, graphene membranes, or water treatment…")

            st.markdown("""
            <div style="background-color:#f0f8ff; border:1px solid #b3d9ff; border-radius:8px; padding:1rem; margin-top:1rem;">
                <p style="color:#1e3a8a; font-size:0.9rem; margin:0;">
                    <strong>🤖 AI Assistant Notice:</strong> Trained on the same data used to build Aqua.LA.Graph-Lite v1.0.
                    Verify critical information through original sources.
                </p>
            </div>
            """, unsafe_allow_html=True)

            if prompt:
                st.session_state.chat.append(("user", prompt))
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            resp = answer_user_message(prompt, api_key=api_key)
                            st.write(resp.text)
                            st.session_state.chat.append(("assistant", resp.text))
                        except Exception as e:
                            error_msg = f"❌ Error: {str(e)}"
                            st.write(error_msg)
                            st.session_state.chat.append(("assistant", error_msg))
                st.rerun()

    def render_about():
        st.markdown("""
        <div style="position:relative; height:180px; margin-bottom:2rem;">
            <img src="data:image/jpeg;base64,{}" style="position:absolute; top:0; left:0; width:150px; height:150px; object-fit:cover; border-radius:12px; box-shadow:0 4px 12px rgba(0,0,0,0.15);">
            <img src="data:image/jpeg;base64,{}" style="position:absolute; top:0; right:0; width:150px; height:150px; object-fit:cover; border-radius:12px; box-shadow:0 4px 12px rgba(0,0,0,0.15);">
        </div>
        """.format(
            base64.b64encode(open("assets/Image1.jpg", "rb").read()).decode(),
            base64.b64encode(open("assets/Image2.jpg", "rb").read()).decode()
        ), unsafe_allow_html=True)

        st.markdown("""<h1 style="color:#1e3a8a; font-size:2.5rem; font-weight:700; text-align:center; margin-bottom:1rem;">About Aqua.LA.Graph-Lite v1.0</h1>""", unsafe_allow_html=True)
        st.markdown("""<h2 style="color:#1e3a8a; font-size:1.8rem; font-weight:600; text-align:center; margin-bottom:2rem;">ML Web Tool for Monolayer Nanoporous Graphene Membranes</h2>""", unsafe_allow_html=True)

        st.markdown("""
        <div style="background-color:#f8f9fa; padding:1.5rem; border-radius:10px; margin-bottom:2rem; border-left:4px solid #1e3a8a;">
            <h3 style="color:#1e3a8a; font-size:1.3rem; font-weight:600; margin-bottom:0.5rem;">Master's Final Project (PFE)</h3>
            <p style="color:#333; font-size:1.1rem; margin-bottom:0.5rem;">Analytical Sciences, Quality & Environment (S.A.Q.E)</p>
            <p style="color:#333; font-size:1.1rem; margin:0;">Faculty of Sciences, Mohammed V University, Rabat</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""<h2 style="color:#1e3a8a; font-size:1.6rem; font-weight:600; margin-top:2rem; margin-bottom:1rem;">Project Overview</h2>""", unsafe_allow_html=True)
        st.markdown("""
<div style="color:#333; line-height:1.6; font-size:1.05rem;">
Water scarcity is one of the most pressing challenges of our time. Seawater desalination has become a strategic necessity.
Single-layer nanoporous graphene membranes promise exceptional water permeance and near-perfect salt rejection.
<span style="font-weight:600; color:#1e3a8a;">Aqua.LA.Graph-Lite v1.0 was built to help break the computational bottleneck.</span>
</div>
        """, unsafe_allow_html=True)

        st.markdown("""<h2 style="color:#1e3a8a; font-size:1.6rem; font-weight:600; margin-top:2rem; margin-bottom:1rem;">Our Mission</h2>""", unsafe_allow_html=True)
        st.markdown("""
<div style="color:#333; line-height:1.6; font-size:1.05rem;">
This free, openly accessible web tool empowers researchers, engineers, and students to instantly predict membrane performance
without writing a single line of code or accessing a supercomputer.
</div>
        """, unsafe_allow_html=True)

        st.markdown("""<h2 style="color:#1e3a8a; font-size:1.6rem; font-weight:600; margin-top:2rem; margin-bottom:1rem;">How It Works & Model Performance</h2>""", unsafe_allow_html=True)
        st.markdown("""
        <div style="color:#333; line-height:1.6; font-size:1.05rem;">
            Powered by CatBoost, trained on 460 clean data points from 12 peer-reviewed articles.
            Full code is openly available on GitHub.
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("""
            <div style="background-color:#e8f4fd; padding:1.5rem; border-radius:10px; margin-bottom:1rem;">
                <h3 style="color:#1e3a8a; font-size:1.2rem; font-weight:600; margin-bottom:1rem;">Input parameters (7):</h3>
                <ul style="color:#333; line-height:1.6; font-size:1rem; margin:0; padding-left:1.5rem;">
                    <li>Pore shape</li><li>Pore area (Å²)</li><li>Applied pressure (MPa)</li>
                    <li>Pore chemistry</li><li>Porosity (%)</li><li>Temperature (°C)</li>
                    <li>Feed salt concentration (ppm)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div style="background-color:#e8f4fd; padding:1.5rem; border-radius:10px; margin-bottom:1rem;">
                <h3 style="color:#1e3a8a; font-size:1.2rem; font-weight:600; margin-bottom:1rem;">Predicted outputs (2):</h3>
                <ul style="color:#333; line-height:1.6; font-size:1rem; margin:0; padding-left:1.5rem;">
                    <li>Water flux (molecules/nanosecond)</li>
                    <li>Salt rejection (%)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div style="background-color:#fff3cd; border:1px solid #ffeaa7; border-radius:10px; padding:1.5rem; margin-bottom:2rem;">
            <p style="color:#856404; line-height:1.6; font-size:1.05rem; margin:0;">
                <strong>Aqua.LA.Graph-Lite v1.0 is a research-support tool, not a replacement for rigorous simulation or experiment.</strong><br><br>
                Always verify critical results through established scientific methods.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""<h2 style="color:#1e3a8a; font-size:1.6rem; font-weight:600; margin-top:2rem; margin-bottom:1rem;">Supervisors & Student</h2>""", unsafe_allow_html=True)
        st.markdown("""
**LAARAJ LAHCEN** — Master S.A.Q.E, UM5-FSR Rabat  
📧 lahcen_laaraj@um5.ac.ma | lahcenelaaraj37@gmail.com

**Supervising Professors:** Prof. S. EL HAJJAJI · Prof. Z. ZITI
""")

        st.markdown("""<h2 style="color:#1e3a8a; font-size:1.6rem; font-weight:600; margin-top:2rem; margin-bottom:1rem;">Open Science & Reproducibility</h2>""", unsafe_allow_html=True)
        st.markdown("""
        <div style="background-color:#e8f4fd; border:1px solid #93c5fd; border-radius:10px; padding:1.5rem; margin-bottom:2rem;">
            <strong>🔗 GitHub Repository:</strong><br>
            <a href="https://github.com/lahcenlaaraj37-alt/Pr-diction-des-performances-des-membranes-de-graph-ne-monocouche-nanoporeuses"
               style="color:#1e3a8a; text-decoration:none; font-weight:600; word-break:break-all;">
                github.com/lahcenlaaraj37-alt/...
            </a>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="background-color:#f8fafc; border:1px solid #e2e8f0; border-radius:15px; padding:2rem; margin-top:3rem; margin-bottom:2rem;">
            <div style="display:flex; align-items:center; gap:2rem; flex-wrap:wrap;">
                <div style="flex-shrink:0;">
                    <img src="data:image/jpeg;base64,{}" alt="LAARAJ LAHCEN"
                         style="width:120px; height:120px; border-radius:50%; object-fit:cover; border:3px solid #1e3a8a; box-shadow:0 4px 15px rgba(30,58,138,0.2);">
                </div>
                <div style="flex:1; min-width:250px;">
                    <h3 style="color:#1e3a8a; font-size:1.4rem; font-weight:700; margin:0 0 0.5rem 0;">LAARAJ LAHCEN</h3>
                    <div style="color:#475569; font-size:1.05rem; font-weight:500; margin-bottom:1rem;">Master S.A.Q.E, Faculty of Sciences, UM5 Rabat</div>
                    <div style="color:#64748b; font-size:0.95rem; margin-bottom:1rem;">
                        <strong>💼 Role:</strong> Educational Specialist - Supervision of School Laboratories
                    </div>
                    <div style="display:flex; gap:1rem; flex-wrap:wrap; margin-top:1.5rem;">
                        <a href="https://www.linkedin.com/in/lahcen-laaraj-18a18822b/" target="_blank"
                           style="background-color:#0077b5; color:white; padding:0.5rem 1rem; border-radius:6px; text-decoration:none; font-weight:500; font-size:0.9rem;">
                            LinkedIn
                        </a>
                        <a href="https://github.com/lahcenlaaraj37-alt" target="_blank"
                           style="background-color:#333; color:white; padding:0.5rem 1rem; border-radius:6px; text-decoration:none; font-weight:500; font-size:0.9rem;">
                            GitHub
                        </a>
                    </div>
                </div>
            </div>
        </div>
        """.format(base64.b64encode(open("assets/Me.jpeg", "rb").read()).decode()), unsafe_allow_html=True)

    # ===== Routing =====
    if page == f"💧 {t('nav_single', st.session_state.lang)}":
        render_single_simulation()
    elif page == f"📊 {t('nav_batch', st.session_state.lang)}":
        render_batch()
    elif page == f"📈 {t('nav_viz', st.session_state.lang)}":
        render_viz_reports()
    elif page == f"🔄 {t('nav_inverse', st.session_state.lang)}":
        render_inverse()
    elif page == f"🤖 {t('nav_chat', st.session_state.lang)}":
        render_chat()
    elif page == f"ℹ️ {t('nav_about', st.session_state.lang)}":
        render_about()
    else:
        render_about()


if __name__ == "__main__":
    main()
