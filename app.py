from __future__ import annotations

import datetime as dt
import io
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st

from utils.i18n import t
from utils.constants import (
    CHEMISTRY_COLOR_FALLBACK,
    DEFAULT_MODEL_PATHS,
    FEATURES_COLUMNS,
    LOG_BASE,
)
from utils.exporting import dataframe_to_bytes
from utils.model_io import inverse_log_flux_plus1_to_flux, load_pipeline_and_metadata
from utils.pdf_report import build_pdf_report
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
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

_APP_CSS = """
<style>
  .block-container { padding-top: 1.1rem; padding-bottom: 2.2rem; }
  .gp-header {
    display:flex; align-items:center; justify-content:space-between;
    gap: 1rem; padding: 0.75rem 1rem;
    border-radius: 14px;
    background: linear-gradient(90deg, rgba(37,99,235,0.10), rgba(255,255,255,0.0));
    border: 1px solid rgba(17,24,39,0.08);
  }
  .gp-title { font-size: 1.25rem; font-weight: 800; margin: 0; }
  .gp-subtitle { margin: 0.15rem 0 0; opacity: 0.85; font-size: 0.9rem; }
  .gp-card {
    border-radius: 14px;
    background: rgba(255, 255, 255, 1.0);
    border: 1px solid rgba(17,24,39,0.10);
    padding: 0.85rem 1rem;
  }
  .gp-kpi-label { opacity: 0.8; font-size: 0.85rem; margin: 0; }
  .gp-kpi-value { font-size: 1.55rem; font-weight: 800; margin: 0.1rem 0 0; }
  .gp-pill { display:inline-block; padding: 0.25rem 0.55rem; border-radius:999px;
    background: rgba(37,99,235,0.10); border:1px solid rgba(37,99,235,0.25); font-size:0.8rem; }

  /* Slightly lighter blue accents */
  div.stButton > button[kind="primary"] {
    background: #3B82F6 !important;
    border-color: #2563EB !important;
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
    # A "button grid" so each option shows its color (closest to your request).
    if not chemistry_display:
        return "(no categories found)"

    if st.session_state.chemistry not in chemistry_display:
        st.session_state.chemistry = chemistry_display[0]

    st.caption("Pore chemistry (functionalization)")
    cols = st.columns(2)
    for i, c in enumerate(chemistry_display):
        col = _chemistry_to_color(c, chemistry_display)
        btn_label = f"● {c}"
        with cols[i % 2]:
            if st.button(btn_label, key=f"chem_btn_{c}", width="stretch"):
                st.session_state.chemistry = c
            st.markdown(
                f"<div style='margin-top:-8px;margin-bottom:10px;'>"
                f"<span style='display:inline-block;width:10px;height:10px;border-radius:999px;background:{col};"
                f"margin-right:8px;'></span><span style='font-size:0.85rem;'>{c}</span></div>",
                unsafe_allow_html=True,
            )

    st.markdown(
        f"<div style='margin-top:8px;'><b>Selected:</b> {st.session_state.chemistry}</div>",
        unsafe_allow_html=True,
    )
    return st.session_state.chemistry


def main():
    _init_state()
    salt_model, flux_model = _load_models()

    numeric_ranges = salt_model.metadata.get("ranges", {})
    categories = salt_model.metadata.get("categories", {})
    geometry_choices = categories.get("Geometry", [])
    chemistry_choices = categories.get("Pore Chemistry (Functionalization)", [])

    st.markdown(_APP_CSS, unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Language")
        st.session_state.lang = st.selectbox("", options=["en", "fr", "ar"], index=["en", "fr", "ar"].index(st.session_state.lang))
        st.divider()
        page = st.radio(
            "Navigation",
            options=[
                t("nav_single", st.session_state.lang),
                t("nav_batch", st.session_state.lang),
                t("nav_viz", st.session_state.lang),
                t("nav_inverse", st.session_state.lang),
                t("nav_chat", st.session_state.lang),
                t("nav_about", st.session_state.lang),
            ],
        )

    st.markdown(
        """
<div class="gp-header">
  <div>
    <div class="gp-title">{title}</div>
    <div class="gp-subtitle">{subtitle}</div>
  </div>
  <div>
    <span class="gp-pill">Free web tool</span>
  </div>
</div>
        """.format(
            title=t("app_title", st.session_state.lang),
            subtitle=t("app_subtitle", st.session_state.lang),
        ).strip(),
        unsafe_allow_html=True,
    )

    st.write("")

    # Shared cleaned categories
    geometry_display = sorted({g.strip(): None for g in geometry_choices}.keys()) if geometry_choices else []
    chemistry_display = sorted({c.strip(): None for c in chemistry_choices}.keys()) if chemistry_choices else []

    def render_single_simulation():
        left, mid = st.columns([0.35, 0.65], gap="large")

        with left:
            st.markdown('<div class="gp-card">', unsafe_allow_html=True)
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

            run = st.button(t("predict", st.session_state.lang), type="primary", width="stretch")
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                clear = st.button("Clear table", width="stretch")
            with col_btn2:
                recalc_all = st.button("Recompute", width="stretch")

            if clear:
                st.session_state.simulations = st.session_state.simulations.iloc[0:0].copy()
                st.toast("Cleared.")

            if run:
                try:
                    pred = _predict_single(user_row)
                    if any_out_of_range(range_checks):
                        st.info(t("low_reliability", st.session_state.lang))
                    new_row = {
                        **user_row,
                        "Salt Rejection (%)": pred.salt_rejection,
                        "Water Flux (molecule/ns)": pred.flux,
                    }
                    st.session_state.simulations = pd.concat(
                        [st.session_state.simulations, pd.DataFrame([new_row])], ignore_index=True
                    )
                except Exception as e:
                    st.exception(e)

            if recalc_all and len(st.session_state.simulations) > 0:
                try:
                    X_tbl = st.session_state.simulations[FEATURES_COLUMNS].copy()
                    sr = salt_model.pipeline.predict(X_tbl)
                    lf = flux_model.pipeline.predict(X_tbl)
                    st.session_state.simulations["Salt Rejection (%)"] = sr
                    st.session_state.simulations["Water Flux (molecule/ns)"] = (LOG_BASE ** lf) - 1
                    st.toast("Recomputed.")
                except Exception as e:
                    st.exception(e)

            st.markdown("</div>", unsafe_allow_html=True)

        with mid:
            st.markdown('<div class="gp-card">', unsafe_allow_html=True)
            st.subheader(t("membrane", st.session_state.lang))
            chem_color = _chemistry_to_color(st.session_state.chemistry or "", chemistry_display or chemistry_choices)
            fig2d = make_graphene_membrane_2d(
                geometry=geometry,
                pore_chemistry=st.session_state.chemistry or "",
                chemistry_color=chem_color,
            )
            st.plotly_chart(fig2d, use_container_width=True)

            st.subheader(t("predictions", st.session_state.lang))
            if len(st.session_state.simulations) > 0:
                last = st.session_state.simulations.iloc[-1]
                kpi1, kpi2 = st.columns(2)
                with kpi1:
                    st.markdown(
                        f'<div class="gp-card"><p class="gp-kpi-label">Water flux</p>'
                        f'<p class="gp-kpi-value">{float(last["Water Flux (molecule/ns)"]):.3f}</p>'
                        f'<span class="gp-pill">molecule/ns</span></div>',
                        unsafe_allow_html=True,
                    )
                with kpi2:
                    st.markdown(
                        f'<div class="gp-card"><p class="gp-kpi-label">Salt rejection</p>'
                        f'<p class="gp-kpi-value">{float(last["Salt Rejection (%)"]):.3f}</p>'
                        f'<span class="gp-pill">%</span></div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.info("Run a prediction to see KPIs.")
            st.markdown("</div>", unsafe_allow_html=True)

    def render_batch():
        st.markdown('<div class="gp-card">', unsafe_allow_html=True)
        st.subheader(t("nav_batch", st.session_state.lang))
        st.caption("Your Excel must include the 7 feature columns with exact names.")
        uploaded = st.file_uploader("Upload Excel with many rows", type=["xlsx"])
        if uploaded is not None:
            try:
                df_up = pd.read_excel(uploaded)
                missing = [c for c in FEATURES_COLUMNS if c not in df_up.columns]
                if missing:
                    st.error("Uploaded file is missing required columns:\n- " + "\n- ".join(missing))
                else:
                    X_up = df_up[FEATURES_COLUMNS].copy()
                    sr = salt_model.pipeline.predict(X_up)
                    lf = flux_model.pipeline.predict(X_up)
                    df_up["Salt Rejection (%)"] = sr
                    df_up["Water Flux (molecule/ns)"] = (LOG_BASE ** lf) - 1
                    st.success(f"Predicted {len(df_up)} rows.")
                    st.session_state.simulations = pd.concat(
                        [st.session_state.simulations, df_up[FEATURES_COLUMNS + ["Salt Rejection (%)", "Water Flux (molecule/ns)"]]],
                        ignore_index=True,
                    )
                    st.dataframe(df_up, use_container_width=True)
            except Exception as e:
                st.exception(e)
        st.markdown("</div>", unsafe_allow_html=True)

    def render_viz_reports():
        st.markdown('<div class="gp-card">', unsafe_allow_html=True)
        st.subheader(t("nav_viz", st.session_state.lang))
        st.subheader(t("table", st.session_state.lang))
        edited = st.data_editor(
            st.session_state.simulations,
            num_rows="dynamic",
            width="stretch",
            hide_index=True,
        )
        st.session_state.simulations = edited

        st.subheader(t("exports", st.session_state.lang))
        col_e1, col_e2, col_e3 = st.columns(3)
        with col_e1:
            st.download_button(
                t("download_csv", st.session_state.lang),
                data=dataframe_to_bytes(st.session_state.simulations, "csv"),
                file_name="simulations.csv",
                mime="text/csv",
                width="stretch",
            )
        with col_e2:
            st.download_button(
                t("download_xlsx", st.session_state.lang),
                data=dataframe_to_bytes(st.session_state.simulations, "xlsx"),
                file_name="simulations.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width="stretch",
            )
        with col_e3:
            make_pdf = st.button(t("build_pdf", st.session_state.lang), width="stretch")

        st.subheader("Plots")
        fig = None
        if len(st.session_state.simulations) >= 2:
            x_col = st.selectbox("X axis", options=FEATURES_COLUMNS, index=1)
            y_col = st.selectbox("Y axis", options=["Salt Rejection (%)", "Water Flux (molecule/ns)"], index=1)
            fig = make_scatter(st.session_state.simulations, x_col=x_col, y_col=y_col)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run at least 2 simulations to enable plotting.")

        if make_pdf:
            try:
                plot_pngs = []
                if fig is not None:
                    plot_pngs.append(fig.to_image(format="png", scale=2))
                pdf_bytes = build_pdf_report(
                    title="Graphene Membrane Predictor Report",
                    simulations_df=st.session_state.simulations,
                    plot_images_png=plot_pngs,
                )
                st.download_button(
                    "Download PDF",
                    data=pdf_bytes,
                    file_name=f"report_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    width="stretch",
                )
            except Exception as e:
                st.exception(e)
        st.markdown("</div>", unsafe_allow_html=True)

    def render_inverse():
        st.markdown('<div class="gp-card">', unsafe_allow_html=True)
        st.subheader(t("nav_inverse", st.session_state.lang))
        st.caption("Estimate a pore area that matches your target outputs (simple grid search).")

        inv_flux = st.number_input("Target water flux (molecule/ns)", value=10.0, step=0.1)
        inv_sr = st.number_input("Target salt rejection (%)", value=95.0, step=0.1)
        inv_btn = st.button("Estimate pore area", width="stretch")

        # Use current selections as context
        geometry = (geometry_display[0] if geometry_display else "Circular")
        chemistry = (chemistry_display[0] if chemistry_display else "Pristine")

        if inv_btn:
            try:
                r = numeric_ranges.get("Pore Area (Å²)", None)
                if not r:
                    st.error("No pore-area range found in metadata.")
                else:
                    base_row = {
                        "Geometry": geometry,
                        "Applied pressure  (MPa)": float(numeric_ranges["Applied pressure  (MPa)"]["min"]),
                        "Feed Concentration  (ppm)": float(numeric_ranges["Feed Concentration  (ppm)"]["min"]),
                        "Temperature  (°C)": float(numeric_ranges["Temperature  (°C)"]["min"]),
                        "Pore Chemistry (Functionalization)": chemistry,
                        "Porosity (%)": float(numeric_ranges["Porosity (%)"]["min"]),
                    }
                    min_a, max_a = float(r["min"]), float(r["max"])
                    grid = np.linspace(min_a, max_a, 100)
                    best = None
                    best_err = float("inf")
                    for a in grid:
                        row2 = dict(base_row)
                        row2["Pore Area (Å²)"] = float(a)
                        pred = _predict_single(row2)
                        err = abs(pred.flux - float(inv_flux)) + 0.2 * abs(pred.salt_rejection - float(inv_sr))
                        if err < best_err:
                            best_err = err
                            best = (a, pred)
                    if best:
                        a, pred = best
                        st.success(f"Estimated pore area (Å²): {a:.4g}")
                        st.caption(f"At this area → rejection: {pred.salt_rejection:.2f}% | flux: {pred.flux:.2f}")
            except Exception as e:
                st.exception(e)
        st.markdown("</div>", unsafe_allow_html=True)

    def render_chat():
        st.markdown('<div class="gp-card">', unsafe_allow_html=True)
        st.subheader(t("nav_chat", st.session_state.lang))
        st.caption("Powered by Google Gemini (configure `GEMINI_API_KEY` in Streamlit secrets).")

        for role, msg in st.session_state.chat:
            with st.chat_message(role):
                st.write(msg)

        prompt = st.chat_input("Chat about desalination and graphene membranes…")
        if prompt:
            st.session_state.chat.append(("user", prompt))
            api_key = None
            try:
                api_key = st.secrets.get("GEMINI_API_KEY")
            except Exception:
                api_key = None
            resp = answer_user_message(prompt, api_key=api_key)
            st.session_state.chat.append(("assistant", resp.text))
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    def render_about():
        st.markdown('<div class="gp-card">', unsafe_allow_html=True)
        st.subheader(t("nav_about", st.session_state.lang))
        st.markdown(t("about_text", st.session_state.lang))
        st.markdown("</div>", unsafe_allow_html=True)

    # Route
    if page == t("nav_single", st.session_state.lang):
        render_single_simulation()
    elif page == t("nav_batch", st.session_state.lang):
        render_batch()
    elif page == t("nav_viz", st.session_state.lang):
        render_viz_reports()
    elif page == t("nav_inverse", st.session_state.lang):
        render_inverse()
    elif page == t("nav_chat", st.session_state.lang):
        render_chat()
    else:
        render_about()


if __name__ == "__main__":
    main()

