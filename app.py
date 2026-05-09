from __future__ import annotations

# Importation des bibliothèques standard et tierces
import io
import os
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import base64

# Importation des modules utilitaires locaux
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

# Structure de données pour les résultats de prédiction
@dataclass(frozen=True)
class PredictionResult:
    salt_rejection: float
    flux_log: float
    flux: float

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Graphene Membrane Predictor",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# === STYLE CSS (Responsive: Mobile & Desktop) ===
# ============================================================
_GLOBAL_STYLE = """
<style>
/* Configuration de base */
.stApp { background-color: #f8fafc; }

/* 1. Afficher la barre Streamlit originale et assurer sa visibilité */
header[data-testid="stHeader"] {
    display: flex !important;
    visibility: visible !important;
    background-color: white !important;
    z-index: 100000 !important;
}

/* 2. Positionnement de l'en-tête personnalisé (Header) sous la barre originale */
#gp-fixed-header {
    position: fixed;
    top: 3.5rem; /* Juste en dessous du header Streamlit */
    left: 0;
    right: 0;
    width: 100%;
    background: linear-gradient(135deg, #1e3a8a 0%, #2c5282 100%);
    padding: 1rem 2rem;
    z-index: 9999;
    box-shadow: 0 4px 20px rgba(30,58,138,0.3);
    display: flex;
    justify-content: space-between;
    align-items: center;
    color: white;
}

/* 3. Positionnement de la barre latérale (Sidebar) */
section[data-testid="stSidebar"] {
    top: 3.5rem !important; /* Pour ne pas couvrir la barre originale */
    background-color: #1e3a8a !important;
}

/* 4. Ajustement des marges pour éviter le chevauchement du contenu */
.main .block-container {
    padding-top: 10rem !important; /* Espace pour Header Streamlit (3.5rem) + Header Perso (~6rem) */
    max-width: 1200px;
    margin-left: auto;
    margin-right: auto;
}

/* Styles pour les widgets et entrées */
.stSelectbox div[data-baseweb="select"] > div,
.stNumberInput input,
.stTextInput input {
    background-color: #f0f7ff !important;
    border: 1px solid #93c5fd !important;
    border-radius: 8px !important;
}

/* Adaptations spécifiques pour Mobile */
@media (max-width: 768px) {
    #gp-fixed-header {
        padding: 0.8rem;
        flex-direction: column;
        text-align: center;
    }
    #gp-fixed-header .title-text { font-size: 1.2rem !important; }
    #gp-fixed-header .dev-info { display: none; }
    
    .main .block-container {
        padding-top: 13rem !important; /* Plus d'espace car le header s'étend verticalement sur mobile */
    }
}
</style>
"""

# ============================================================
# === FONCTIONS UTILITAIRES ET LOGIQUE ===
# ============================================================

@st.cache_resource
def _load_models():
    """Chargement des modèles de machine learning pré-entraînés"""
    salt = load_pipeline_and_metadata(
        DEFAULT_MODEL_PATHS.salt_rejection_pkl, DEFAULT_MODEL_PATHS.salt_rejection_meta
    )
    flux = load_pipeline_and_metadata(DEFAULT_MODEL_PATHS.log_flux_pkl, DEFAULT_MODEL_PATHS.log_flux_meta)
    return salt, flux

def _chemistry_to_color(chemistry: str, choices: list[str]) -> str:
    """Attribue une couleur en fonction de la chimie du pore"""
    if chemistry in choices and len(choices) > 0:
        idx = choices.index(chemistry) % len(CHEMISTRY_COLOR_FALLBACK)
        return CHEMISTRY_COLOR_FALLBACK[idx]
    return CHEMISTRY_COLOR_FALLBACK[0]

def _predict_single(row: dict) -> PredictionResult:
    """Exécute une prédiction pour une seule configuration de membrane"""
    salt_model, flux_model = _load_models()
    X = pd.DataFrame([coerce_row_to_model_input(row, FEATURES_COLUMNS)], columns=FEATURES_COLUMNS)
    
    sr = float(salt_model.pipeline.predict(X)[0])
    lf = float(flux_model.pipeline.predict(X)[0])
    flux = float(inverse_log_flux_plus1_to_flux(lf, log_base=LOG_BASE))

    # Logique métier : réjection de sel à 100% si la concentration est très faible
    feed_conc = float(row.get("Feed Concentration  (ppm)", 0))
    if 0 <= feed_conc <= 1:
        sr = 100.0

    return PredictionResult(salt_rejection=sr, flux_log=lf, flux=flux)

def _init_state():
    """Initialisation de l'état de la session Streamlit"""
    if "simulations" not in st.session_state:
        st.session_state.simulations = pd.DataFrame(columns=FEATURES_COLUMNS + ["Salt Rejection (%)", "Water Flux (molecule/ns)"])
    if "chat" not in st.session_state:
        st.session_state.chat = []
    if "lang" not in st.session_state:
        st.session_state.lang = "en"
    if "chemistry" not in st.session_state:
        st.session_state.chemistry = None

def _chemistry_picker(chemistry_display: list[str]) -> str:
    """Sélecteur pour la chimie du pore avec gestion de l'état"""
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
# === POINT D'ENTRÉE DU PROGRAMME ===
# ============================================================

def main():
    _init_state()
    st.markdown(_GLOBAL_STYLE, unsafe_allow_html=True)

    # Chargement des modèles et des métadonnées
    salt_model, flux_model = _load_models()
    numeric_ranges = salt_model.metadata.get("ranges", {})
    categories = salt_model.metadata.get("categories", {})
    geometry_choices = categories.get("Geometry", [])
    chemistry_choices = categories.get("Pore Chemistry (Functionalization)", [])

    # === BARRE LATÉRALE (SIDEBAR) ===
    with st.sidebar:
        # Informations sur le développeur
        st.markdown("""
        <div style='text-align: center; padding-bottom: 20px; border-bottom: 1px solid rgba(255,255,255,0.2); margin-bottom: 20px;'>
            <p style='color: white; font-weight: 700; font-size: 1rem; margin: 0;'>LAARAJ Lahcen</p>
            <p style='color: #93c5fd; font-size: 0.8rem; margin: 4px 0 0 0;'>UM5 - FSR Rabat</p>
        </div>
        """, unsafe_allow_html=True)

        # Sélection de la langue
        st.subheader("Language")
        l_col1, l_col2, l_col3 = st.columns(3)
        with l_col1:
            if st.button("EN", use_container_width=True, type="primary" if st.session_state.lang=="en" else "secondary"):
                st.session_state.lang = "en"; st.rerun()
        with l_col2:
            if st.button("FR", use_container_width=True, type="primary" if st.session_state.lang=="fr" else "secondary"):
                st.session_state.lang = "fr"; st.rerun()
        with l_col3:
            if st.button("AR", use_container_width=True, type="primary" if st.session_state.lang=="ar" else "secondary"):
                st.session_state.lang = "ar"; st.rerun()

        st.divider()
        # Navigation
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

    # === EN-TÊTE FIXE (HEADER) ===
    st.markdown(f"""
    <div id="gp-fixed-header">
      <div style="display:flex; flex-direction:column;">
        <div class="title-text" style="font-size: 1.8rem; font-weight: 800;">🌊 Aqua.LA.Graph-Lite v1.0</div>
        <div style="font-size: 0.85rem; opacity: 0.9;">ML Web Tool for Graphene Membranes</div>
      </div>
      <div class="dev-info" style="text-align: right;">
        <div style="font-size: 0.85rem; font-weight: 500;">Developed by: LAARAJ Lahcen</div>
        <div style="font-size: 0.75rem; opacity: 0.8;">Master S.A.Q.E | UM5 Rabat</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # === LOGIQUE DE ROUTAGE DES PAGES ===
    geometry_display = sorted({g.strip(): None for g in geometry_choices}.keys()) if geometry_choices else []
    chemistry_display = sorted({c.strip(): None for c in chemistry_choices}.keys()) if chemistry_choices else []

    # 1. Simulation Unique
    if page == f"💧 {t('nav_single', st.session_state.lang)}":
        col_in, col_out = st.columns([0.4, 0.6], gap="large")
        with col_in:
            st.subheader(t("inputs", st.session_state.lang))
            geom = st.selectbox("Geometry", options=geometry_display)
            chem = _chemistry_picker(chemistry_display)
            
            def _n_in(label, key):
                rng = numeric_ranges.get(key, {"min":0, "max":100})
                return st.number_input(label, min_value=float(rng["min"]), max_value=float(rng["max"]), value=float(rng["min"]), step=0.1)

            area = _n_in("Pore Area (Å²)", "Pore Area (Å²)")
            pres = _n_in("Pressure (MPa)", "Applied pressure  (MPa)")
            feed = _n_in("Feed Concentration (ppm)", "Feed Concentration  (ppm)")
            temp = _n_in("Temperature (°C)", "Temperature  (°C)")
            poro = _n_in("Porosity (%)", "Porosity (%)")

            if st.button(t("predict", st.session_state.lang), type="primary", use_container_width=True):
                row = {
                    "Geometry": geom, "Pore Area (Å²)": area, "Applied pressure  (MPa)": pres,
                    "Feed Concentration  (ppm)": feed, "Temperature  (°C)": temp,
                    "Pore Chemistry (Functionalization)": chem, "Porosity (%)": poro
                }
                res = _predict_single(row)
                new_data = pd.DataFrame([{**row, "Salt Rejection (%)": min(res.salt_rejection, 100.0), "Water Flux (molecule/ns)": res.flux}])
                st.session_state.simulations = pd.concat([st.session_state.simulations, new_data], ignore_index=True)
                st.success("Simulation added!")

        with col_out:
            st.subheader(t("predictions", st.session_state.lang))
            if not st.session_state.simulations.empty:
                last = st.session_state.simulations.iloc[-1]
                st.metric("Water Flux", f"{last['Water Flux (molecule/ns)']:.3f} mol/ns")
                st.metric("Salt Rejection", f"{last['Salt Rejection (%)']:.2f} %")
                
                # Visualisation 2D de la membrane
                fig = make_graphene_membrane_2d(geom, chem, _chemistry_to_color(chem, chemistry_display))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Run a prediction to see the results.")

    # 2. Simulation par lot (Batch)
    elif page == f"📊 {t('nav_batch', st.session_state.lang)}":
        st.subheader(t("nav_batch", st.session_state.lang))
        uploaded = st.file_uploader("Upload Excel file (.xlsx)", type=["xlsx"])
        if uploaded:
            df_up = pd.read_excel(uploaded)
            # Logique de prédiction par lot (similaire à app-1.py)
            st.dataframe(df_up)

    # 3. Visualisation et Rapports
    elif page == f"📈 {t('nav_viz', st.session_state.lang)}":
        st.subheader(t("table", st.session_state.lang))
        st.dataframe(st.session_state.simulations, use_container_width=True)
        
        if not st.session_state.simulations.empty:
            st.download_button("Download Results (CSV)", 
                               data=dataframe_to_bytes(st.session_state.simulations, "csv"),
                               file_name="results.csv", mime="text/csv")

    # 4. Design Inverse
    elif page == f"🔄 {t('nav_inverse', st.session_state.lang)}":
        st.subheader(t("nav_inverse", st.session_state.lang))
        st.write("Determine the required pore area based on target performance.")
        # La logique de boucle d'optimisation peut être ajoutée ici

    # 5. Chatbot IA
    elif page == f"🤖 {t('nav_chat', st.session_state.lang)}":
        st.subheader(t("nav_chat", st.session_state.lang))
        for role, msg in st.session_state.chat:
            with st.chat_message(role): st.write(msg)
        
        prompt = st.chat_input("Ask something about desalination...")
        if prompt:
            st.session_state.chat.append(("user", prompt))
            # Appel API Groq ou similaire via utils.chatbot
            st.rerun()

    # 6. À Propos
    elif page == f"ℹ️ {t('nav_about', st.session_state.lang)}":
        st.title("About Aqua.LA.Graph-Lite")
        st.markdown("""
        Ce projet fait partie d'un travail de recherche sur les membranes en graphène nanoporeux 
        pour le dessalement de l'eau, utilisant des techniques avancées de Machine Learning.
        
        **Développeur :** LAARAJ Lahcen  
        **Supervision :** Prof. S. EL HAJJAJI & Prof. Z. ZITI  
        **Institution :** Faculté des Sciences, Université Mohammed V, Rabat.
        """)

if __name__ == "__main__":
    main()
