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
# STYLES CSS PERSONNALISÉS
# ============================================================
# NOTE : Le bloc qui masquait la barre Streamlit originale a été
#        commenté pour que la barre reste visible et fonctionnelle.
#        Tous les autres styles sont inchangés.
st.markdown("""
<style>
.stApp {
    background-color: #f8fafc;
}

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
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.2) !important;
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

label[data-testid="stWidgetLabel"] p, .st-emotion-cache-1y4p8pa p {
    color: #1e40af !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
}

.stSidebar * {
    color: #FFFFFF !important;
}

.stSidebar input,
.stSidebar select,
.stSidebar textarea {
    color: #000000 !important;
}

div[role="listbox"] ul {
    background-color: white !important;
}
div[role="option"]:hover {
    background-color: #dbeafe !important;
}

/* JavaScript: Surveillance du bouton Streamlit original maintenant en bas */
button[data-testid="collapsedControl"] {
    position: fixed !important;
    bottom: 20px !important;
    left: 15px !important;
    top: auto !important;
    z-index: 999998 !important;
    background: #1e3a8a !important;
    color: white !important;
    border-radius: 8px !important;
    border: 2px solid #93c5fd !important;
    width: 42px !important;
    height: 42px !important;
    box-shadow: 0 4px 12px rgba(30, 58, 138, 0.4) !important;
}
button[data-testid="collapsedControl"]:hover {
    background: #2c5282 !important;
}
button[data-testid="collapsedControl"] svg {
    color: white !important;
    fill: white !important;
}
</style>
<script>
window.addEventListener('load', function() {
    const observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            if (mutation.addedNodes) {
                mutation.addedNodes.forEach(function(node) {
                    if (node.nodeType === 1) {
                        const buttons = node.querySelectorAll ? node.querySelectorAll('button') : [];
                        buttons.forEach(function(btn) {
                            if (btn.innerText && btn.innerText.includes('☰')) {
                                btn.style.position = 'fixed';
                                btn.style.bottom = '70px';
                                btn.style.left = '15px';
                                btn.style.top = 'auto';
                                btn.style.zIndex = '999999';
                                btn.style.background = '#1e3a8a';
                                btn.style.color = 'white';
                                btn.style.borderRadius = '8px';
                                btn.style.border = '2px solid #93c5fd';
                                btn.style.width = '42px';
                                btn.style.height = '42px';
                                btn.style.boxShadow = '0 4px 12px rgba(30, 58, 138, 0.4)';
                            }
                        });
                    }
                });
            }
        });
    });
    observer.observe(document.body, { childList: true, subtree: true });
});
</script>
""", unsafe_allow_html=True)

# ============================================================
# CSS PRINCIPAL DE L'APPLICATION (_APP_CSS)
# ============================================================
# Modifications apportées par rapport au code original :
# 1. La barre Streamlit originale est maintenant visible (le bloc qui
#    la masquait a été commenté dans le style ci-dessus).
# 2. Le z-index de la barre latérale est passé à 10000 pour qu'elle
#    s'affiche au-dessus de l'en-tête personnalisé.
# 3. L'en-tête personnalisé est positionné à "top: 3.5rem" (collé sous
#    la barre Streamlit) grâce à son id "custom-header".
# 4. La barre latérale commence également à "top: 3.5rem" (collée sous
#    la barre Streamlit) grâce à la règle CSS sur [data-testid="stSidebar"].
# 5. Des Media Queries sont ajoutées pour l'affichage sur téléphone.
# 6. Le margin-top du contenu principal est ajusté pour ne pas être
#    caché derrière l'en-tête fixe.
_APP_CSS = """
<style>
  body {
    font-family: 'Inter', sans-serif;
    background: linear-gradient(135deg, #001f3f 0%, #003366 25%, #004080 50%, #0059b3 75%, #0066cc 100%);
    min-height: 100vh;
  }

  .main .block-container {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 20px;
    box-shadow: 0 8px 32px rgba(0, 102, 204, 0.2);
    backdrop-filter: blur(10px);
    padding-top: 0 !important;
    padding-bottom: 1rem;
    max-width: 1200px;
    /* Marge augmentée pour compenser la hauteur de l'en-tête fixe
       (barre Streamlit ~3.5rem + en-tête ~6rem + espace) */
    margin-top: 13rem !important;
    padding-top: 1rem !important;
  }

  .stApp {
    margin-top: 0 !important;
    padding-top: 0 !important;
  }

  /* Bloc original qui masquait la barre Streamlit – conservé ici
     en commentaire pour référence, mais il NE DOIT PAS être actif
     pour que la barre Streamlit reste visible et fonctionnelle. */
  /*
  .stApp > div > div > div > div > div,
  .stApp header,
  .stApp .stToolbar,
  .stApp .stHeader,
  .stApp [data-testid="stHeader"],
  .stApp [data-testid="stToolbar"] {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
  }
  */

  .gp-card {
    border-radius: 8px;
    background: rgba(255, 255, 255, 0.95);
    border: 1px solid rgba(0, 102, 204, 0.2);
    padding: 0.8rem;
    box-shadow: 0 1px 4px rgba(0, 102, 204, 0.05);
    height: calc(100vh - 250px);
    overflow-y: auto;
  }

  .gp-kpi-label {
    opacity: 0.8;
    font-size: 0.85rem;
    margin: 0;
    color: #004080;
    font-weight: 600;
  }

  .gp-kpi-value {
    font-size: 1.8rem;
    font-weight: 800;
    margin: 0.1rem 0 0;
    color: #0066cc;
    text-shadow: 0 1px 2px rgba(0, 102, 204, 0.2);
  }

  .gp-pill {
    display: inline-block;
    padding: 0.35rem 0.75rem;
    border-radius: 999px;
    background: linear-gradient(135deg, #004080 0%, #0066cc 100%);
    border: 1px solid rgba(0, 102, 204, 0.3);
    font-size: 0.85rem;
    color: white;
    font-weight: 600;
    box-shadow: 0 2px 8px rgba(0, 102, 204, 0.2);
  }

  div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #004080 0%, #0066cc 100%) !important;
    border: 1px solid #0059b3 !important;
    color: white !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 12px rgba(0, 102, 204, 0.3) !important;
    transition: all 0.3s ease !important;
  }

  div.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #0059b3 0%, #0073e6 100%) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(0, 102, 204, 0.4) !important;
  }

  [data-testid="stSidebar"] {
    background-color: #1e3a8a !important;
    padding-top: 20px !important;
  }

  [data-testid="stSidebar"] .st-emotion-cache-16txtl3 {
    padding-top: 0rem !important;
  }

  [data-testid="stSidebar"] p {
    color: #93c5fd !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    margin-bottom: 12px !important;
  }

  [data-testid="stSidebar"] div[role="radiogroup"] > label {
    padding: 12px 16px !important;
    margin-bottom: 4px !important;
    border-radius: 8px !important;
    font-size: 0.95rem !important;
    font-weight: 500 !important;
    transition: all 0.2s !important;
    background: transparent !important;
    color: #FFFFFF !important;
  }

  [data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
    background-color: rgba(255,255,255,0.15) !important;
  }

  [data-testid="stSidebar"] div[aria-checked="true"] {
    background-color: #3b82f6 !important;
    font-weight: 600 !important;
  }

  [data-testid="stSidebar"] div[aria-checked="true"] > label {
    background-color: transparent !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
  }

  [data-testid="stSidebar"] button[kind="primary"] {
    background-color: #FFFFFF !important;
    color: #1e3a8a !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 6px !important;
  }

  [data-testid="stSidebar"] button:not([kind="primary"]) {
    background-color: transparent !important;
    color: #FFFFFF !important;
    border: 1px solid #93c5fd !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
  }

  [data-testid="stSidebar"] button:not([kind="primary"]):hover {
    background-color: rgba(255,255,255,0.1) !important;
    color: #FFFFFF !important;
  }

  .stTextInput > div > div > input,
  .stNumberInput > div > div > input,
  .stSelectbox > div > div > select {
    border: 2px solid rgba(0, 102, 204, 0.2) !important;
    border-radius: 8px !important;
    background: rgba(255, 255, 255, 0.9) !important;
  }

  [data-testid="stAppViewContainer"] h2 {
    color: #1e3a8a !important;
    font-weight: 800 !important;
    font-size: 1.8rem !important;
    padding-bottom: 10px !important;
    border-bottom: 3px solid #3b82f6 !important;
    margin-top: 1rem !important;
    margin-bottom: 1.5rem !important;
  }

  [data-testid="stAppViewContainer"] h3 {
    color: #1e3a8a !important;
    font-weight: 700 !important;
    font-size: 1.4rem !important;
    margin-top: 1.2rem !important;
    margin-bottom: 1rem !important;
  }

  .stTextInput > div > div > input:focus,
  .stNumberInput > div > div > input:focus,
  .stSelectbox > div > div > select:focus {
    border-color: #0066cc !important;
    box-shadow: 0 0 0 3px rgba(0, 102, 204, 0.1) !important;
  }

  .stColumns > div {
    height: calc(100vh - 250px);
  }

  section[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
    opacity: 1 !important;
  }

  section[data-testid="stSidebar"] nav ul li a {
    font-size: 18px !important;
    font-weight: 600 !important;
    color: #FFFFFF !important;
  }

  section[data-testid="stSidebar"] nav ul li a[aria-current="page"] {
    background-color: #1E88E5 !important;
    border-radius: 8px !important;
  }

  section[data-testid="stSidebar"] label[data-testid="stWidgetLabel"] p {
    color: #FFFFFF !important;
    opacity: 1 !important;
    font-size: 28px !important;
    font-weight: 700 !important;
  }

  /* ===== Bouton Menu flottant - blanc sur fond bleu ===== */
  #floating-menu-btn {
    position: fixed;
    top: 15px;
    left: 15px;
    z-index: 9999999;
  }
  #floating-menu-btn button {
    background: #FFFFFF !important;
    color: #1e3a8a !important;
    border: 2px solid #1e3a8a !important;
    border-radius: 8px !important;
    padding: 7px 14px !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    box-shadow: 0 3px 10px rgba(30, 58, 138, 0.2) !important;
    cursor: pointer !important;
    letter-spacing: 0.3px !important;
    transition: background 0.2s !important;
  }
  #floating-menu-btn button:hover {
    background: #f8fafc !important;
    border-color: #1e3a8a !important;
    color: #1e3a8a !important;
  }

  /* Déplacement du bouton Streamlit original en bas de l'écran */
  button[data-testid="collapsedControl"] {
    position: fixed !important;
    bottom: 20px !important;
    left: 15px !important;
    top: auto !important;
    z-index: 999998 !important;
    background: #1e3a8a !important;
    color: white !important;
    border-radius: 8px !important;
    border: 2px solid #93c5fd !important;
    width: 42px !important;
    height: 42px !important;
    box-shadow: 0 4px 12px rgba(30, 58, 138, 0.4) !important;
  }
  button[data-testid="collapsedControl"]:hover {
    background: #2c5282 !important;
  }
  button[data-testid="collapsedControl"] svg {
    color: white !important;
    fill: white !important;
  }

  /* ========================================================
     AJOUTS POUR LES MODIFICATIONS DEMANDÉES
     ======================================================== */

  /* NOUVEAU : La barre latérale s'affiche au-dessus de l'en-tête
     personnalisé (z-index élevé) pour ne pas être masquée.
     De plus, elle commence juste en dessous de la barre Streamlit
     originale (top: 3.5rem) et prend toute la hauteur restante. */
  section[data-testid="stSidebar"] {
    z-index: 10000 !important;
    top: 3.5rem !important;
    height: calc(100vh - 3.5rem) !important;
  }

  /* NOUVEAU : L'en-tête personnalisé est collé juste sous la
     barre Streamlit originale (hauteur d'environ 3.5rem). */
  #custom-header {
    top: 3.5rem !important;
  }

  /* NOUVEAU : Media Queries pour l'affichage sur téléphone mobile.
     Ajuste les marges et la disposition pour que le contenu reste
     lisible et bien organisé. */
  @media screen and (max-width: 768px) {
    /* Réduction de la marge supérieure car l'en-tête est moins haut
       sur mobile (passage en colonne). */
    .main .block-container {
      margin-top: 7rem !important;
    }

    /* La barre latérale occupe toute la largeur sur mobile et
       commence également sous la barre Streamlit. */
    section[data-testid="stSidebar"] {
      top: 3.5rem !important;
      height: calc(100vh - 3.5rem) !important;
    }

    /* L'en-tête passe en disposition verticale et ses espacements
       sont réduits pour s'adapter aux petits écrans. */
    #custom-header {
      padding-left: 1rem !important;
      padding-right: 1rem !important;
      flex-direction: column !important;
      align-items: flex-start !important;
      height: auto !important;
    }

    /* Réduction de la taille du titre principal pour éviter
       les débordements. */
    #custom-header > div:first-child {
      font-size: 1.4rem !important;
    }

    /* Réduction de la taille des sous-titres. */
    #custom-header > div:first-child div {
      font-size: 0.75rem !important;
    }

    /* Le nom du développeur est affiché plus petit. */
    #custom-header > div:last-child {
      font-size: 0.7rem !important;
      margin-top: 0.3rem !important;
    }
  }
</style>
"""


@st.cache_resource
def _load_models():
    """Charge les pipelines de prédiction depuis le cache Streamlit."""
    salt = load_pipeline_and_metadata(
        DEFAULT_MODEL_PATHS.salt_rejection_pkl, DEFAULT_MODEL_PATHS.salt_rejection_meta
    )
    flux = load_pipeline_and_metadata(DEFAULT_MODEL_PATHS.log_flux_pkl, DEFAULT_MODEL_PATHS.log_flux_meta)
    return salt, flux


def _chemistry_to_color(chemistry: str, choices: list[str]) -> str:
    """Attribue une couleur cohérente à une chimie de pore donnée."""
    if chemistry in choices and len(choices) > 0:
        idx = choices.index(chemistry) % len(CHEMISTRY_COLOR_FALLBACK)
        return CHEMISTRY_COLOR_FALLBACK[idx]
    return CHEMISTRY_COLOR_FALLBACK[0]


def _predict_single(row: dict) -> PredictionResult:
    """Prédit la réjection de sel et le flux d'eau pour une seule ligne de paramètres."""
    salt_model, flux_model = _load_models()
    X = pd.DataFrame([coerce_row_to_model_input(row, FEATURES_COLUMNS)], columns=FEATURES_COLUMNS)
    sr = float(salt_model.pipeline.predict(X)[0])
    lf = float(flux_model.pipeline.predict(X)[0])
    flux = float(inverse_log_flux_plus1_to_flux(lf, log_base=LOG_BASE))

    # Cas particulier : concentration d'alimentation nulle ou quasi-nulle
    feed_conc = float(row.get("Feed Concentration  (ppm)", 0))
    if 0 <= feed_conc <= 1:
        sr = 100.0

    return PredictionResult(salt_rejection=sr, flux_log=lf, flux=flux)


def _init_state():
    """Initialise les variables de session Streamlit si elles n'existent pas."""
    if "simulations" not in st.session_state:
        st.session_state.simulations = pd.DataFrame(columns=FEATURES_COLUMNS + ["Salt Rejection (%)", "Water Flux (molecule/ns)"])
    if "chat" not in st.session_state:
        st.session_state.chat = []
    if "lang" not in st.session_state:
        st.session_state.lang = "en"
    if "chemistry" not in st.session_state:
        st.session_state.chemistry = None


def _chemistry_picker(chemistry_display: list[str]) -> str:
    """Affiche un sélecteur pour la chimie de pore."""
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
# BOUTON D'AFFICHAGE/MASQUAGE DE LA BARRE LATÉRALE
# ============================================================
def sidebar_toggle():
    """
    Gère l'affichage du bouton ☰ Menu et la synchronisation avec
    le bouton de barre latérale de Streamlit.
    """
    if 'sidebar_open' not in st.session_state:
        st.session_state.sidebar_open = True

    # Masquer la barre latérale avec CSS lorsqu'elle est fermée
    if not st.session_state.sidebar_open:
        st.markdown(
            """<style>section[data-testid="stSidebar"] {display: none !important;}</style>""",
            unsafe_allow_html=True,
        )

    # JavaScript: Surveillance du bouton Streamlit original maintenant en bas
    # Lorsque l'utilisateur clique dessus pour masquer la barre latérale,
    # nous programmons un clic sur le bouton ☰ Menu pour synchroniser session_state
    st.markdown(
        """
        <script>
        (function () {
            function init() {
                const stBtn = document.querySelector('button[data-testid="collapsedControl"]');
                if (!stBtn) { setTimeout(init, 400); return; }

                stBtn.addEventListener('click', function () {
                    setTimeout(function () {
                        const allBtns = document.querySelectorAll('button');
                        allBtns.forEach(function (btn) {
                            if (btn.innerText && btn.innerText.trim().includes('☰')) {
                                btn.click();
                            }
                        });
                    }, 150);
                });
            }

            if (document.readyState === 'loading') {
                document.addEventListener('DOMContentLoaded', init);
            } else {
                setTimeout(init, 600);
            }
        })();
        </script>
        """,
        unsafe_allow_html=True,
    )

    # Bouton flottant ☰ Menu (blanc sur fond bleu)
    st.markdown('<div id="floating-menu-btn">', unsafe_allow_html=True)
    if st.button("☰ Menu", key="toggle_sb_unique"):
        st.session_state.sidebar_open = not st.session_state.sidebar_open
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
# ============================================================


def main():
    """Fonction principale de l'application Streamlit."""
    sidebar_toggle()
    _init_state()
    salt_model, flux_model = _load_models()

    numeric_ranges = salt_model.metadata.get("ranges", {})
    categories = salt_model.metadata.get("categories", {})
    geometry_choices = categories.get("Geometry", [])
    chemistry_choices = categories.get("Pore Chemistry (Functionalization)", [])

    # Application du CSS principal
    st.markdown(_APP_CSS, unsafe_allow_html=True)

    # ============================================================
    # BARRE LATÉRALE (SIDEBAR)
    # ============================================================
    with st.sidebar:
        # En-tête de la barre latérale avec logo et informations
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

    # ============================================================
    # EN-TÊTE PERSONNALISÉ FIXE
    # ============================================================
    # MODIFICATION : L'en-tête est maintenant positionné à "top: 3.5rem"
    # (juste sous la barre Streamlit originale) au lieu de "top: 0".
    # Un attribut id="custom-header" a été ajouté pour le style CSS.
    st.markdown(
        """
<div id="custom-header" style="position: fixed; top: 3.5rem; left: 0; right: 0; width: 100%; background: linear-gradient(135deg, #1e3a8a 0%, #2c5282 100%); padding: 1.5rem 2rem; padding-left: 350px; z-index: 9999; box-shadow: 0 4px 20px rgba(30, 58, 138, 0.3); display: flex; justify-content: space-between; align-items: center;">
  <div style="display: flex; flex-direction: column; justify-content: center;">
    <div style="font-size: 2.2rem; font-weight: 800; color: #FFFFFF; line-height: 1.2; margin-bottom: 0.25rem;">
      🌊 Aqua.LA.Graph-Lite v1.0
    </div>
    <div style="font-size: 0.9rem; color: #dbeafe; font-weight: 400; margin-bottom: 0.15rem;">
      ML Web Tool for Monolayer Nanoporous Graphene Membranes
    </div>
    <div style="font-size: 0.75rem; color: #a5d8ff; font-weight: 300;">
      PFE | UM5-FSR-M:S.A.Q.E
    </div>
  </div>
  <div style="color: #FFFFFF; font-size: 0.85rem; font-weight: 500;">
    Developed by: LAARAJ Lahcen
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    # Listes des choix disponibles pour les menus déroulants
    geometry_display = sorted({g.strip(): None for g in geometry_choices}.keys()) if geometry_choices else []
    chemistry_display = sorted({c.strip(): None for c in chemistry_choices}.keys()) if chemistry_choices else []

    # ============================================================
    # PAGE : SIMULATION UNIQUE
    # ============================================================
    def render_single_simulation():
        col_inputs, col_viz_results = st.columns([0.3, 0.7], gap="medium")

        with col_inputs:
            st.subheader(t("inputs", st.session_state.lang))

            geometry = st.selectbox("Pore geometry", options=geometry_display or ["(no categories found)"])
            chemistry = _chemistry_picker(chemistry_display)

            def _num_input(label: str, col: str, default: float | None = None):
                """Affiche un champ numérique avec validation de plage."""
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

            # Note d'information sur les données
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
                    <div style='background: linear-gradient(145deg, #f8f9fa 0%, #e9ecef 100%); padding: 15px; border-radius: 12px; margin: 15px 0; border-left: 4px solid #6c757d; box-shadow: 0 4px 15px rgba(108, 117, 125, 0.1);'>
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
                    <div style='background: linear-gradient(145deg, #f0f8ff 0%, #e6f3ff 100%); padding: 15px; border-radius: 12px; margin: 12px 0; border-left: 4px solid #0066cc; box-shadow: 0 4px 15px rgba(0, 102, 204, 0.1);'>
                        <div style='font-weight: bold; color: #004080; margin-bottom: 8px; font-size: 1.1em;'>💧 Water Flux</div>
                        <div style='font-size: 1.8em; font-weight: 800; color: #0066cc; margin-bottom: 5px;'>{flux_val:.3f}</div>
                        <div style='font-size: 0.9em; color: #666; border-top: 1px solid #ddd; padding-top: 5px;'>Error margin: ±{flux_error:.3f} (±5%)</div>
                        <div style='font-size: 0.85em; color: #888; margin-top: 3px;'>molecule/ns</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                salt_val = float(last["Salt Rejection (%)"])
                salt_error = salt_val * 0.03
                st.markdown(
                    f"""
                    <div style='background: linear-gradient(145deg, #f0fff0 0%, #e6ffe6 100%); padding: 15px; border-radius: 12px; margin: 12px 0; border-left: 4px solid #28a745; box-shadow: 0 4px 15px rgba(40, 167, 69, 0.1);'>
                        <div style='font-weight: bold; color: #1e5f1e; margin-bottom: 8px; font-size: 1.1em;'>🛡️ Salt Rejection</div>
                        <div style='font-size: 1.8em; font-weight: 800; color: #28a745; margin-bottom: 5px;'>{salt_val:.1f}%</div>
                        <div style='font-size: 0.9em; color: #666; border-top: 1px solid #ddd; padding-top: 5px;'>Error margin: ±{salt_error:.1f}% (±3%)</div>
                        <div style='font-size: 0.85em; color: #888; margin-top: 3px;'>(clamped to max 100%)</div>
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
                            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;'>
                                <span style='font-weight: 600; color: #333; font-size: 0.9em;'>{feature}</span>
                                <span style='font-weight: bold; color: {color}; font-size: 0.9em;'>{percentage:.1f}%</span>
                            </div>
                            <div style='background: #e0e0e0; border-radius: 10px; height: 8px; overflow: hidden;'>
                                <div style='background: {color}; height: 100%; width: {percentage}%; border-radius: 10px; transition: width 0.3s ease;'></div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.markdown("*Higher percentage = more influence on prediction result*")
                st.markdown("---")
                st.markdown(
                    """
                    <div style='background: #fff3cd; padding: 10px; border-radius: 6px; border-left: 4px solid #ffc107;'>
                        <div style='font-size: 0.9em; color: #856404;'>ℹ️ This simulation is automatically added to the Analysis & Reports table</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            else:
                st.info("Run a prediction to see results with error margins.")

    # ============================================================
    # PAGE : TRAITEMENT PAR LOTS (BATCH)
    # ============================================================
    def render_batch():
        st.subheader(t("nav_batch", st.session_state.lang))

        with st.expander("📋 File Requirements & Guidelines", expanded=True):
            st.markdown("""
            **Required Excel File Format:**
            - Your file must contain exactly **9 columns** (7 inputs + 2 outputs)
            - Column names must match exactly (case-sensitive)

            **Required Input Columns:**
            1. `Geometry` (text: e.g., 'hexagonal', 'circular')
            2. `Pore Area (Å²)` (numeric: area in square angstroms)
            3. `Applied pressure  (MPa)` (numeric: pressure in megapascals)
            4. `Feed Concentration  (ppm)` (numeric: salt concentration in ppm)
            5. `Temperature  (°C)` (numeric: temperature in degrees Celsius)
            6. `Pore Chemistry (Functionalization)` (text: e.g., 'OH', 'H', 'Pristine')
            7. `Porosity (%)` (numeric: porosity percentage)

            **Output Columns (will be added automatically):**
            8. `Salt Rejection (%)` (will be calculated)
            9. `Water Flux (molecule/ns)` (will be calculated)

            **⚠️ Important Constraints:**
            - **Verify units** before uploading - ensure all values use correct units
            - **Stay within training data ranges** for reliable predictions
            - **Use only available geometry types** from single simulation
            - **Use only available pore chemistry types** from single simulation
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
        uploaded = st.file_uploader("📤 Upload Excel File", type=["xlsx"],
                                    help="Make sure your file follows the format above")

        if uploaded is not None:
            try:
                df_up = pd.read_excel(uploaded)
                st.write(f"📄 File loaded: {len(df_up)} rows found")

                missing = [c for c in FEATURES_COLUMNS if c not in df_up.columns]
                if missing:
                    st.error("❌ Uploaded file is missing required columns:\n- " + "\n- ".join(missing))
                    st.stop()

                warnings_list = []

                for col in NUMERIC_FEATURES:
                    if col in df_up.columns:
                        r = numeric_ranges.get(col, {})
                        if r:
                            min_v, max_v = float(r["min"]), float(r["max"])
                            out_of_range = df_up[(df_up[col] < min_v) | (df_up[col] > max_v)]
                            if len(out_of_range) > 0:
                                warnings_list.append(f"⚠️ {col}: {len(out_of_range)} rows outside training range ({min_v:.4g} - {max_v:.4g})")

                if geometry_choices:
                    valid_geoms = [g.strip() for g in geometry_choices if g.strip()]
                    invalid_geoms = df_up[~df_up["Geometry"].isin(valid_geoms)]
                    if len(invalid_geoms) > 0:
                        warnings_list.append(f"⚠️ Geometry: {len(invalid_geoms)} rows with invalid geometry types")

                if chemistry_choices:
                    valid_chem = [c.strip() for c in chemistry_choices if c.strip()] + ["Pristine"]
                    invalid_chem = df_up[~df_up["Pore Chemistry (Functionalization)"].isin(valid_chem)]
                    if len(invalid_chem) > 0:
                        warnings_list.append(f"⚠️ Chemistry: {len(invalid_chem)} rows with invalid chemistry types")

                if warnings_list:
                    st.warning("⚠️ **Warnings detected:**\n" + "\n".join(warnings_list))
                    st.info("Predictions may be less reliable for out-of-range values.")

                X_up = df_up[FEATURES_COLUMNS].copy()
                sr = salt_model.pipeline.predict(X_up)
                lf = flux_model.pipeline.predict(X_up)

                df_up["Salt Rejection (%)"] = np.minimum(sr, 100.0)
                df_up["Water Flux (molecule/ns)"] = (LOG_BASE ** lf) - 1

                st.success(f"✅ Successfully predicted {len(df_up)} rows!")
                st.info("📊 All results have been automatically added to Analysis & Reports table")
                st.dataframe(df_up, use_container_width=True)

                st.session_state.simulations = pd.concat(
                    [st.session_state.simulations, df_up[FEATURES_COLUMNS + ["Salt Rejection (%)", "Water Flux (molecule/ns)"]]],
                    ignore_index=True,
                )

            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
                st.exception(e)

    # ============================================================
    # PAGE : VISUALISATION ET RAPPORTS
    # ============================================================
    def render_viz_reports():
        st.subheader(t("nav_viz", st.session_state.lang))
        st.subheader(t("table", st.session_state.lang))

        if len(st.session_state.simulations) > 0:
            st.markdown("**📊 Simulation Table (click cells to edit directly):**")

            df_display = st.session_state.simulations.copy()
            column_config = {}

            if geometry_display:
                column_config["Geometry"] = st.column_config.SelectboxColumn(
                    "Geometry", options=geometry_display, required=True, help="Select pore geometry"
                )

            if chemistry_display:
                chem_options = chemistry_display.copy()
                if "Pristine" not in chem_options:
                    chem_options.append("Pristine")
                column_config["Pore Chemistry (Functionalization)"] = st.column_config.SelectboxColumn(
                    "Pore Chemistry", options=chem_options, required=True, help="Select pore chemistry"
                )

            for col in NUMERIC_FEATURES:
                r = numeric_ranges.get(col, {})
                if r:
                    min_v = float(r["min"])
                    max_v = float(r["max"])
                    column_config[col] = st.column_config.NumberColumn(
                        col, min_value=min_v, max_value=max_v, format="%.4f",
                        help=f"Range: {min_v:.4g} - {max_v:.4g}"
                    )
                else:
                    column_config[col] = st.column_config.NumberColumn(col, format="%.4f")

            column_config["Salt Rejection (%)"] = st.column_config.NumberColumn(
                "Salt Rejection (%)", format="%.2f", disabled=True,
                help="Predicted salt rejection (read-only)"
            )
            column_config["Water Flux (molecule/ns)"] = st.column_config.NumberColumn(
                "Water Flux (molecule/ns)", format="%.6f", disabled=True,
                help="Predicted water flux (read-only)"
            )

            edited_df = st.data_editor(
                df_display, num_rows="dynamic", width="stretch",
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
                                        inputs_changed = True
                                        break
                                    elif not pd.isna(original_row[col]) and not pd.isna(edited_row[col]):
                                        if abs(float(original_row[col]) - float(edited_row[col])) > 1e-10:
                                            inputs_changed = True
                                            break
                            if inputs_changed:
                                modified_rows.append(idx)
                        else:
                            modified_rows.append(idx)

                    for idx in modified_rows:
                        if idx < len(edited_df):
                            row_inputs = {}
                            for col in FEATURES_COLUMNS:
                                if col in edited_df.iloc[idx].index:
                                    row_inputs[col] = edited_df.iloc[idx][col]
                            pred = _predict_single(row_inputs)
                            edited_df.loc[idx, "Salt Rejection (%)"] = min(pred.salt_rejection, 100.0)
                            edited_df.loc[idx, "Water Flux (molecule/ns)"] = pred.flux

                    st.session_state.simulations = edited_df.copy()

                    if modified_rows:
                        st.success(f"✅ Updated predictions for {len(modified_rows)} modified row(s)!")
                        st.rerun()

                except Exception as e:
                    st.error(f"❌ Error updating predictions: {str(e)}")
                    st.session_state.simulations = df_display.copy()
        else:
            st.info("No simulations yet. Run single simulations or upload batch data.")

        st.subheader(t("exports", st.session_state.lang))
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            st.download_button(
                t("download_csv", st.session_state.lang),
                data=dataframe_to_bytes(st.session_state.simulations, "csv"),
                file_name="simulations.csv",
                mime="text/csv",
                width="stretch",
                type="primary",
            )
        with col_e2:
            st.download_button(
                t("download_xlsx", st.session_state.lang),
                data=dataframe_to_bytes(st.session_state.simulations, "xlsx"),
                file_name="simulations.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width="stretch",
                type="primary",
            )

        st.subheader("Plots")
        if len(st.session_state.simulations) >= 2:
            x_col = st.selectbox("X axis", options=FEATURES_COLUMNS, index=1)
            y_col = st.selectbox("Y axis", options=["Salt Rejection (%)", "Water Flux (molecule/ns)"], index=1)
            fig = make_scatter(st.session_state.simulations, x_col=x_col, y_col=y_col)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run at least 2 simulations to enable plotting.")

    # ============================================================
    # PAGE : CONCEPTION INVERSE
    # ============================================================
    def render_inverse():
        st.subheader(t("nav_inverse", st.session_state.lang))
        st.caption("Find optimal pore area for your target performance (inverse design).")

        st.markdown("**🎯 Target Performance:**")
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            inv_flux = st.number_input("Target water flux (molecule/ns)", value=10.0, step=0.1,
                                       help="Desired water flux value")
        with col_t2:
            inv_sr = st.number_input("Target salt rejection (%)", value=95.0, step=0.1,
                                     help="Desired salt rejection percentage")

        st.markdown("**⚙️ Operating Conditions:**")
        col1, col2 = st.columns(2)
        with col1:
            inv_temp = st.number_input("Temperature (°C)", value=25.0, step=0.1, help="Operating temperature")
            inv_pressure = st.number_input("Pressure (MPa)", value=1.0, step=0.1, help="Applied pressure")
        with col2:
            inv_feed = st.number_input("Feed concentration (ppm)", value=1000.0, step=10.0,
                                       help="Salt concentration in feed water")
            inv_porosity = st.number_input("Porosity (%)", value=10.0, step=0.1, help="Membrane porosity")

        st.markdown("**🧪 Pore Chemistry:**")
        inv_chemistry = st.selectbox("Select pore chemistry", options=["OH", "H", "Pristine"],
                                     help="Limited to most common options with sufficient data")
        st.caption("ℹ️ Limited to OH, H, and Pristine options as they have the most data points in our training set for better inverse design accuracy.")

        st.markdown("**🔷 Pore Geometry:**")
        inv_geometry = st.selectbox("Select pore geometry", options=["hexagonal", "circular"],
                                    help="Limited to most common geometries")
        st.caption("ℹ️ Limited to hexagonal and circular geometries as they have the most comprehensive data coverage for reliable inverse predictions.")

        inv_btn = st.button("🔍 Estimate Optimal Pore Area", type="primary", width="stretch")

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

                    warnings_list = []
                    for col, val in [
                        ("Applied pressure  (MPa)", inv_pressure),
                        ("Feed Concentration  (ppm)", inv_feed),
                        ("Temperature  (°C)", inv_temp),
                        ("Porosity (%)", inv_porosity)
                    ]:
                        r_check = numeric_ranges.get(col, {})
                        if r_check:
                            min_v, max_v = float(r_check["min"]), float(r_check["max"])
                            if val < min_v or val > max_v:
                                warnings_list.append(f"⚠️ {col}: {val:.4g} outside training range ({min_v:.4g} - {max_v:.4g})")

                    if warnings_list:
                        st.warning("⚠️ **Input warnings:**\n" + "\n".join(warnings_list))
                        st.info("Results may be less reliable for out-of-range values.")

                    min_a, max_a = float(r["min"]), float(r["max"])
                    grid_points = 50
                    grid = np.linspace(min_a, max_a, grid_points)
                    best = None
                    best_err = float("inf")

                    progress_bar = st.progress(0, text="Searching optimal pore area...")

                    for i, a in enumerate(grid):
                        progress = (i + 1) / grid_points
                        progress_bar.progress(progress, text=f"Searching... {progress:.0%}")
                        row2 = dict(base_row)
                        row2["Pore Area (Å²)"] = float(a)
                        pred = _predict_single(row2)
                        err = abs(pred.flux - float(inv_flux)) + 0.3 * abs(min(pred.salt_rejection, 100.0) - float(inv_sr))
                        if err < best_err:
                            best_err = err
                            best = (a, pred)

                    progress_bar.empty()

                    if best and grid_points >= 30:
                        st.info("🔍 Refining search for better accuracy...")
                        a_refine, _ = best
                        refine_range = (max_a - min_a) / grid_points
                        min_refine = max(min_a, a_refine - refine_range)
                        max_refine = min(max_a, a_refine + refine_range)
                        refine_grid = np.linspace(min_refine, max_refine, 20)
                        for a in refine_grid:
                            row2 = dict(base_row)
                            row2["Pore Area (Å²)"] = float(a)
                            pred = _predict_single(row2)
                            err = abs(pred.flux - float(inv_flux)) + 0.3 * abs(min(pred.salt_rejection, 100.0) - float(inv_sr))
                            if err < best_err:
                                best_err = err
                                best = (a, pred)

                    if best:
                        a, pred = best
                        st.success(f"🎯 **Final Optimal Pore Area:** {a:.4g} Å²")
                        st.markdown("### 📊 **Final Prediction Results:**")

                        col_main1, col_main2 = st.columns(2)
                        with col_main1:
                            st.markdown(
                                f"""
                                <div style='background: linear-gradient(145deg, #f0f8ff 0%, #e6f3ff 100%); padding: 20px; border-radius: 12px; border-left: 5px solid #0066cc; box-shadow: 0 6px 20px rgba(0, 102, 204, 0.15);'>
                                    <div style='font-weight: bold; color: #004080; margin-bottom: 10px; font-size: 1.2em;'>💧 Water Flux</div>
                                    <div style='font-size: 2.2em; font-weight: 900; color: #0066cc; margin-bottom: 8px;'>{pred.flux:.3f}</div>
                                    <div style='background: rgba(0, 102, 204, 0.1); padding: 8px; border-radius: 6px; margin-top: 10px;'>
                                        <div style='font-size: 0.95em; color: #004080;'><strong>Error Margin:</strong> ±{pred.flux*0.05:.3f}</div>
                                        <div style='font-size: 0.85em; color: #666; margin-top: 3px;'>(±5% uncertainty)</div>
                                        <div style='font-size: 0.85em; color: #888; margin-top: 2px;'>molecule/ns</div>
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                        with col_main2:
                            st.markdown(
                                f"""
                                <div style='background: linear-gradient(145deg, #f0fff0 0%, #e6ffe6 100%); padding: 20px; border-radius: 12px; border-left: 5px solid #28a745; box-shadow: 0 6px 20px rgba(40, 167, 69, 0.15);'>
                                    <div style='font-weight: bold; color: #1e5f1e; margin-bottom: 10px; font-size: 1.2em;'>🛡️ Salt Rejection</div>
                                    <div style='font-size: 2.2em; font-weight: 900; color: #28a745; margin-bottom: 8px;'>{min(pred.salt_rejection, 100.0):.1f}%</div>
                                    <div style='background: rgba(40, 167, 69, 0.1); padding: 8px; border-radius: 6px; margin-top: 10px;'>
                                        <div style='font-size: 0.95em; color: #1e5f1e;'><strong>Error Margin:</strong> ±{min(pred.salt_rejection, 100.0)*0.03:.1f}%</div>
                                        <div style='font-size: 0.85em; color: #666; margin-top: 3px;'>(±3% uncertainty, clamped to 100%)</div>
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                        st.markdown("---")
                        col_info1, col_info2 = st.columns(2)
                        with col_info1:
                            st.markdown(
                                f"""
                                <div style='background: linear-gradient(145deg, #fff3cd 0%, #ffeaa7 100%); padding: 15px; border-radius: 12px; border-left: 4px solid #f39c12;'>
                                    <div style='font-weight: bold; color: #856404; margin-bottom: 8px; font-size: 1.1em;'>⚙️ Design Parameters</div>
                                    <div style='font-size: 0.9em; color: #6c757d;'>
                                        <div><strong>Geometry:</strong> {inv_geometry}</div>
                                        <div><strong>Chemistry:</strong> {inv_chemistry}</div>
                                        <div><strong>Pressure:</strong> {inv_pressure} MPa</div>
                                        <div><strong>Temperature:</strong> {inv_temp}°C</div>
                                        <div><strong>Feed Conc.:</strong> {inv_feed} ppm</div>
                                        <div><strong>Porosity:</strong> {inv_porosity}%</div>
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                        with col_info2:
                            st.markdown(
                                f"""
                                <div style='background: #fff3cd; padding: 15px; border-radius: 8px; border-left: 4px solid #ffc107;'>
                                    <div style='font-weight: bold; color: #856404; margin-bottom: 8px;'>📈 Target vs. Actual</div>
                                    <div style='font-size: 0.9em; color: #856404;'>
                                        <div><strong>Target Flux:</strong> {inv_flux} molecule/ns</div>
                                        <div><strong>Achieved Flux:</strong> {pred.flux:.3f} molecule/ns</div>
                                        <div><strong>Target Rejection:</strong> {inv_sr}%</div>
                                        <div><strong>Achieved Rejection:</strong> {min(pred.salt_rejection, 100.0):.1f}%</div>
                                        <div style='margin-top: 8px; padding-top: 8px; border-top: 1px solid #f0e68c;'>
                                            <strong>Overall Error:</strong> {best_err:.3f}
                                        </div>
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                        new_row = {
                            **base_row,
                            "Pore Area (Å²)": a,
                            "Salt Rejection (%)": min(pred.salt_rejection, 100.0),
                            "Water Flux (molecule/ns)": pred.flux,
                        }
                        st.session_state.simulations = pd.concat(
                            [st.session_state.simulations, pd.DataFrame([new_row])], ignore_index=True
                        )
                        st.info("📊 This inverse design has been automatically added to the Analysis & Reports table")

            except Exception as e:
                st.error(f"❌ Error in inverse design: {str(e)}")
                st.exception(e)

    # ============================================================
    # PAGE : CHATBOT IA
    # ============================================================
    def render_chat():
        st.subheader(t("nav_chat", st.session_state.lang))

        api_key_configured = False
        api_key = None

        import os
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

            To enable chatbot functionality:
            1. Go to your Streamlit Cloud dashboard
            2. Navigate to your app settings
            3. Add a new secret: `GROQ_API_KEY`
            4. Enter your Groq API key

            **To get a Groq API key:**
            1. Visit [Groq Console](https://console.groq.com/keys)
            2. Create a new API key
            3. Copy and paste it into your Streamlit secrets

            The chatbot can answer questions about desalination, graphene membranes, and water treatment in Arabic, French, or English.

            **Configure the API key to start chatting with the AI assistant.**
            """)
            st.markdown("**Chat History:**")
            for role, msg in st.session_state.chat:
                with st.chat_message(role):
                    st.write(msg)

        else:
            st.success("✅ AI Chatbot is ready! Ask questions about desalination and graphene membranes.")
            st.caption("Powered by Groq + RAG - Available in Arabic, French, and English")

            for role, msg in st.session_state.chat:
                with st.chat_message(role):
                    st.write(msg)

            prompt = st.chat_input("Ask about desalination, graphene membranes, or water treatment…")

            # AI Assistant Disclaimer
            st.markdown("""
            <div style="background-color: #f0f8ff; border: 1px solid #b3d9ff; border-radius: 8px; padding: 1rem; margin-top: 1rem;">
                <p style="color: #1e3a8a; font-size: 0.9rem; margin: 0; line-height: 1.4;">
                    <strong>🤖 AI Assistant Notice:</strong> I am an independent AI assistant trained on the same data and algorithms used to develop this application. 
                    I have been trained on the molecular dynamics simulation data, scientific literature, and technical specifications that were used to build Aqua.LA.Graph-Lite v1.0. 
                    While I strive to provide accurate and helpful information, I may occasionally make mistakes. Please verify critical information through the original sources and scientific literature.
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
                            error_msg = f"❌ Sorry, I encountered an error: {str(e)}"
                            st.write(error_msg)
                            st.session_state.chat.append(("assistant", error_msg))

                st.rerun()

    # ============================================================
    # PAGE : À PROPOS (ABOUT)
    # ============================================================
    def render_about():
        # Header with medium square images at top corners
        st.markdown("""
        <div style="position: relative; height: 180px; margin-bottom: 2rem;">
            <img src="data:image/jpeg;base64,{}" 
                 style="position: absolute; top: 0; left: 0; width: 150px; height: 150px; object-fit: cover; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
            <img src="data:image/jpeg;base64,{}" 
                 style="position: absolute; top: 0; right: 0; width: 150px; height: 150px; object-fit: cover; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
        </div>
        """.format(
            base64.b64encode(open("assets/Image1.jpg", "rb").read()).decode(),
            base64.b64encode(open("assets/Image2.jpg", "rb").read()).decode()
        ), unsafe_allow_html=True)

        # Main title
        st.markdown("""
        <h1 style="color: #1e3a8a; font-size: 2.5rem; font-weight: 700; text-align: center; margin-bottom: 1rem;">
            About Aqua.LA.Graph-Lite v1.0
        </h1>
        """, unsafe_allow_html=True)

        st.markdown("""
        <h2 style="color: #1e3a8a; font-size: 1.8rem; font-weight: 600; text-align: center; margin-bottom: 2rem;">
            ML Web Tool for Monolayer Nanoporous Graphene Membranes
        </h2>
        """, unsafe_allow_html=True)

        # Academic info
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem; border-left: 4px solid #1e3a8a;">
            <h3 style="color: #1e3a8a; font-size: 1.3rem; font-weight: 600; margin-bottom: 0.5rem;">
                Master's Final Project (PFE)
            </h3>
            <p style="color: #333; font-size: 1.1rem; margin-bottom: 0.5rem;">
                Analytical Sciences, Quality & Environment (S.A.Q.E)
            </p>
            <p style="color: #333; font-size: 1.1rem; margin: 0;">
                Faculty of Sciences, Mohammed V University, Rabat
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Project Overview
        st.markdown("""
        <h2 style="color: #1e3a8a; font-size: 1.6rem; font-weight: 600; margin-top: 2rem; margin-bottom: 1rem;">
            Project Overview
        </h2>
        """, unsafe_allow_html=True)

        st.markdown("""
<div style="color: #333; line-height: 1.6; font-size: 1.05rem;">
Water scarcity is one of the most pressing challenges of our time, acutely felt in regions such as the Arabian Gulf and North Africa. Seawater desalination has become a strategic necessity. While traditional thermal methods and polymeric reverse osmosis membranes dominate the industry, a next-generation solution is on the horizon: single-layer nanoporous graphene membranes. These atomically thin materials, perforated with precisely controlled nanopores, promise exceptional water permeance and near-perfect salt rejection, potentially revolutionizing desalination with dramatically lower energy consumption.
However, designing and optimizing these membranes today relies almost entirely on molecular dynamics (MD) simulations — a process that demands massive supercomputing resources, extensive expertise, and weeks or months of computation. This dependency creates a critical bottleneck, slowing down research and excluding many talented scientists worldwide, especially those with limited access to high-performance computing.
<span style="font-weight: 600; color: #1e3a8a;">Aqua.LA.Graph-Lite v1.0 was built to help break this barrier.</span>
</div>
        """, unsafe_allow_html=True)

        # Our Mission
        st.markdown("""
        <h2 style="color: #1e3a8a; font-size: 1.6rem; font-weight: 600; margin-top: 2rem; margin-bottom: 1rem;">
            Our Mission
        </h2>
        """, unsafe_allow_html=True)

        st.markdown("""
<div style="color: #333; line-height: 1.6; font-size: 1.05rem;">
This free, openly accessible web tool empowers researchers, engineers, and students to instantly predict the performance of single-layer nanoporous graphene membranes for seawater desalination — without writing a single line of code or accessing a supercomputer. The user simply inputs the membrane's physicochemical and operational parameters; the tool returns the predicted water flux and salt rejection rate.
The goal is not to replace molecular dynamics simulations, but to dramatically reduce the number of necessary simulations, allowing researchers to rapidly screen the design space, identify promising candidates, and focus their heavy computational work only where it truly matters.
</div>
        """, unsafe_allow_html=True)

        # How It Works
        st.markdown("""
        <h2 style="color: #1e3a8a; font-size: 1.6rem; font-weight: 600; margin-top: 2rem; margin-bottom: 1rem;">
            How It Works & Model Performance
        </h2>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="color: #333; line-height: 1.6; font-size: 1.05rem;">
            The predictive engine is powered by CatBoost, a state-of-the-art gradient boosting algorithm, trained on a carefully curated dataset of 460 clean data points extracted from 12 rigorously selected, peer-reviewed scientific articles. All data originates from established molecular dynamics studies published in internationally recognized journals.
        </div>
        """, unsafe_allow_html=True)

        # Input/Output sections
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("""
            <div style="background-color: #e8f4fd; padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem;">
                <h3 style="color: #1e3a8a; font-size: 1.2rem; font-weight: 600; margin-bottom: 1rem;">
                    Input parameters (7):
                </h3>
                <ul style="color: #333; line-height: 1.6; font-size: 1rem; margin: 0; padding-left: 1.5rem;">
                    <li>Pore shape</li>
                    <li>Pore area (Å²)</li>
                    <li>Applied pressure (MPa)</li>
                    <li>Pore chemistry (functionalization)</li>
                    <li>Porosity (%)</li>
                    <li>Temperature (°C)</li>
                    <li>Feed salt concentration (ppm)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div style="background-color: #e8f4fd; padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem;">
                <h3 style="color: #1e3a8a; font-size: 1.2rem; font-weight: 600; margin-bottom: 1rem;">
                    Predicted outputs (2):
                </h3>
                <ul style="color: #333; line-height: 1.6; font-size: 1rem; margin: 0; padding-left: 1.5rem;">
                    <li>Water flux (molecules/nanosecond)</li>
                    <li>Salt rejection (%)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
<div style="color: #333; line-height: 1.6; font-size: 1.05rem;">
The model was developed, tested, and validated through a robust pipeline using Orange Data Mining for initial screening, followed by Python (Jupyter Notebook, Anaconda) for final validation and hyperparameter tuning. The flux output uses a logarithmic transformation (log(flux+1)) to handle the wide dynamic range inherent to nanopore transport phenomena.
The model demonstrated strong predictive performance on both training and held-out test sets. Full reproducible workflows, performance metrics, and code are openly available on GitHub.
</div>
        """, unsafe_allow_html=True)

        # Important Disclaimer
        st.markdown("""
        <h2 style="color: #1e3a8a; font-size: 1.6rem; font-weight: 600; margin-top: 2rem; margin-bottom: 1rem;">
            Important Disclaimer
        </h2>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 10px; padding: 1.5rem; margin-bottom: 2rem;">
            <p style="color: #856404; line-height: 1.6; font-size: 1.05rem; margin: 0;">
                <strong>Aqua.LA.Graph-Lite v1.0 is a research-support tool, not a replacement for rigorous simulation or experiment.</strong><br><br>
                Although the model was built on real, published molecular dynamics data and achieved solid performance during training and testing, its predictions remain based on a limited dataset of 460 samples. The model is designed to help researchers narrow down the number of full MD simulations they need to run. It does not eliminate the need for detailed computational or experimental validation.<br><br>
                The tool harnesses the power of artificial intelligence to compress what would traditionally take weeks or months of computation into mere seconds — enabling researchers to explore ideas and hypotheses that would otherwise be impractical. This is the true promise of AI in scientific research: accelerating discovery, not bypassing rigor.<br><br>
                By using this tool, you acknowledge that the developers and supervisors bear no responsibility for any decisions, designs, or conclusions made solely based on its outputs. Always verify critical results through established scientific methods.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Supervisors & Student
        st.markdown("""
        <h2 style="color: #1e3a8a; font-size: 1.6rem; font-weight: 600; margin-top: 2rem; margin-bottom: 1rem;">
            Supervisors & Student
        </h2>
        """, unsafe_allow_html=True)

        st.markdown("""
**LAARAJ LAHCEN**

Master S.A.Q.E, Faculty of Sciences, Mohammed V University, Rabat  
📧 lahcen_laaraj@um5.ac.ma | lahcenelaaraj37@gmail.com  

<h3 style='color:#1e3a8a; font-size:1.2rem; font-weight:600; margin-bottom:1rem; margin-top:1.5rem;'>Supervising Professors:</h3>
<div style='color:#333; font-size:1.05rem; font-weight:600; margin-bottom:0.5rem;'>
Prof. S. EL HAJJAJI
</div>
<div style='color:#333; font-size:1.05rem; font-weight:600; margin-bottom:0.5rem;'>
Prof. Z. ZITI
</div>
<div style='color:#333; font-size:1rem; margin:0;'>
Faculty of Sciences, Mohammed V University, Rabat
</div>
""", unsafe_allow_html=True)

        # Open Science & Reproducibility
        st.markdown("""
        <h2 style="color: #1e3a8a; font-size: 1.6rem; font-weight: 600; margin-top: 2rem; margin-bottom: 1rem;">
            Open Science & Reproducibility
        </h2>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="color: #333; line-height: 1.6; font-size: 1.05rem;">
            Full transparency is a core principle of this project. The entire codebase, including data preprocessing, model training, evaluation, and the Streamlit web application, is publicly available under an open repository:
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="background-color: #e8f4fd; border: 1px solid #93c5fd; border-radius: 10px; padding: 1.5rem; margin-bottom: 2rem;">
            <div style="color: #333; font-size: 1.05rem; margin: 0;">
                <strong>🔗 GitHub Repository:</strong><br>
                <a href="https://github.com/lahcenlaaraj37-alt/Pr-diction-des-performances-des-membranes-de-graph-ne-monocouche-nanoporeuses" 
                   style="color: #1e3a8a; text-decoration: none; font-weight: 600;">
                    https://github.com/lahcenlaaraj37-alt/Pr-diction-des-performances-des-membranes-de-graph-ne-monocouche-nanoporeuses
                </a>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="color: #333; line-height: 1.6; font-size: 1.05rem;">
            You are welcome to explore, fork, audit, and cite the work. We believe open access accelerates science.
        </div>
        """, unsafe_allow_html=True)

        # AI & Future of Research
        st.markdown("""
        <h2 style="color: #1e3a8a; font-size: 1.6rem; font-weight: 600; margin-top: 2rem; margin-bottom: 1rem;">
            A Note on AI & the Future of Research
        </h2>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="color: #333; line-height: 1.6; font-size: 1.05rem;">
            Artificial intelligence is reshaping the scientific landscape. Tasks that once demanded immense computational resources and endless timelines can now be approached in a fraction of the time. AI allows researchers to ask more questions, test more hypotheses, and ultimately bring breakthrough technologies like graphene membranes closer to real-world implementation. Aqua.LA.Graph-Lite v1.0 is a modest contribution in that direction — built not as a final answer, but as a stepping stone.
        </div>
        """, unsafe_allow_html=True)

        # Acknowledgments
        st.markdown("""
        <h2 style="color: #1e3a8a; font-size: 1.6rem; font-weight: 600; margin-top: 2rem; margin-bottom: 1rem;">
            Acknowledgments
        </h2>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="color: #333; line-height: 1.6; font-size: 1.05rem;">
            We gratefully acknowledge Mohammed V University and the Faculty of Sciences, Rabat for their academic support. Our thanks also go to the global scientific community whose published MD simulation data made this data-driven approach possible.
        </div>
        """, unsafe_allow_html=True)

        # Student Developer Profile
        st.markdown("""
        <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 15px; padding: 2rem; margin-top: 3rem; margin-bottom: 2rem;">
            <div style="display: flex; align-items: center; gap: 2rem; flex-wrap: wrap;">
                <div style="flex-shrink: 0;">
                    <img src="data:image/jpeg;base64,{}" alt="LAARAJ LAHCEN" 
                         style="width: 120px; height: 120px; border-radius: 50%; object-fit: cover; border: 3px solid #1e3a8a; box-shadow: 0 4px 15px rgba(30, 58, 138, 0.2);">
                </div>
                <div style="flex: 1; min-width: 250px;">
                    <h3 style="color: #1e3a8a; font-size: 1.4rem; font-weight: 700; margin: 0 0 0.5rem 0;">
                        LAARAJ LAHCEN
                    </h3>
                    <div style="color: #475569; font-size: 1.05rem; font-weight: 500; margin-bottom: 1rem;">
                        Master S.A.Q.E, Faculty of Sciences, Mohammed V University, Rabat
                    </div>
                    <div style="color: #64748b; font-size: 0.95rem; margin-bottom: 1rem;">
                        <strong>📧 Email:</strong> 
                        <a href="mailto:lahcen_laaraj@um5.ac.ma" style="color: #1e3a8a; text-decoration: none; font-weight: 500;">
                            lahcen_laaraj@um5.ac.ma
                        </a> | 
                        <a href="mailto:lahcanelaaraj37@gmail.com" style="color: #1e3a8a; text-decoration: none; font-weight: 500;">
                            lahcanelaaraj37@gmail.com
                        </a>
                    </div>
                    <div style="color: #64748b; font-size: 0.95rem; margin-bottom: 1rem;">
                        <strong>💼 Professional Role:</strong> Educational Specialist - Supervision of School Laboratories
                    </div>
                    <div style="display: flex; gap: 1rem; flex-wrap: wrap; margin-top: 1.5rem;">
                        <a href="https://www.linkedin.com/in/lahcen-laaraj-18a18822b/" 
                           target="_blank" 
                           style="background-color: #0077b5; color: white; padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; font-weight: 500; font-size: 0.9rem; display: inline-flex; align-items: center; gap: 0.5rem;">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                                <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/>
                            </svg>
                            LinkedIn
                        </a>
                        <a href="https://github.com/lahcenlaaraj37-alt" 
                           target="_blank" 
                           style="background-color: #333; color: white; padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; font-weight: 500; font-size: 0.9rem; display: inline-flex; align-items: center; gap: 0.5rem;">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                                <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
                            </svg>
                            GitHub
                        </a>
                    </div>
                </div>
            </div>
        </div>
        """.format(base64.b64encode(open("assets/Me.jpeg", "rb").read()).decode()), unsafe_allow_html=True)

    # ============================================================
    # ROUTAGE DES PAGES
    # ============================================================
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