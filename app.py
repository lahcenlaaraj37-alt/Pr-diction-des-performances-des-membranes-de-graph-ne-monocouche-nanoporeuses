import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# --- 1. إعدادات الصفحة والعناوين (بالفرنسية) ---
st.set_page_config(page_title="Graphene Membrane Simulator", layout="wide")

st.markdown("""
    <h1 style='text-align: center; color: #1E3A8A;'>
    Modélisation et prédiction des performances des membranes de graphène monocouche nanoporeuses pour le dessalement de l'eau de mer.
    </h1>
    <hr>
    """, unsafe_allow_html=True)

# --- 2. تحميل النماذج والبيانات الإحصائية ---
@st.cache_resource
def load_assets():
    model_rejection = joblib.load('model_salt_rejection.pkl')
    model_flux = joblib.load('model_flux_log.pkl')
    stats_rej = joblib.load('training_stats.pkl')
    stats_flux = joblib.load('training_stats_flux.pkl')
    return model_rejection, model_flux, stats_rej, stats_flux

try:
    model_rej, model_flx, stats_rej, stats_flx = load_assets()
except:
    st.error("Erreur : Fichiers modèles (.pkl) introuvables. Veuillez exécuter la Phase 1 d'abord.")

# --- 3. إدارة حالة الجلسة (Session State) لتخزين المحاكاة ---
if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=[
        'Géométrie', 'Pore Area', 'Pressure', 'Concentration', 'Temp', 'Chemistry', 'Porosity', 
        'Salt Rejection (%)', 'Water Flux (mol/ns)'
    ])

# --- 4. الجانب الأيسر: المدخلات (Sidebar) ---
st.sidebar.header("🛠️ Paramètres d'Entrée")

# ميزة التحميل الجماعي (Batch Processing)
uploaded_file = st.sidebar.file_uploader("Charger un fichier Excel (Batch)", type=["xlsx"])

# نظام الألوان لكيمياء المسام
chem_colors = {"OH": "#3498DB", "H": "#95A5A6", "COOH": "#E74C3C", "None": "#2C3E50"}

# قراءة البارامترات من URL إن وجدت (Share Feature)
query_params = st.query_params

def get_input(label, key, min_val, max_val, default):
    val = st.sidebar.slider(label, float(min_val*0.5), float(max_val*1.5), float(default))
    if val < min_val or val > max_val:
        st.sidebar.markdown(f"<span style='color:red; font-size:12px;'>Hors plage ! [{min_val} - {max_val}]</span>", unsafe_allow_html=True)
    return val

# المدخلات السبعة
geom = st.sidebar.selectbox("Géométrie", ["Hexagonal", "Cubic"], index=0)
pore_area = get_input("Pore Area (Å²)", "pa", stats_rej['Pore Area (Å²)']['min'], stats_rej['Pore Area (Å²)']['max'], 16.3)
press = get_input("Applied pressure (MPa)", "pr", stats_rej['Applied pressure  (MPa)']['min'], stats_rej['Applied pressure  (MPa)']['max'], 100.0)
conc = get_input("Feed Concentration (ppm)", "fc", stats_rej['Feed Concentration  (ppm)']['min'], stats_rej['Feed Concentration  (ppm)']['max'], 35000.0)
temp = get_input("Température (°C)", "tp", stats_rej['Température  (°C)']['min'], stats_rej['Température  (°C)']['max'], 27.0)
chem = st.sidebar.selectbox("Pore Chemistry", list(chem_colors.keys()))
porosity = get_input("Porosity (%)", "po", stats_rej['Porosity (%)']['min'], stats_rej['Porosity (%)']['max'], 10.0)

# --- 5. الحسابات والتنبؤات (التنسيق اليميني) ---
input_df = pd.DataFrame([[geom, pore_area, press, conc, temp, chem, porosity]], 
                         columns=['Géométrie', 'Pore Area (Å²)', 'Applied pressure  (MPa)', 'Feed Concentration  (ppm)', 'Température  (°C)', 'Pore Chemistry (Functionalization)', 'Porosity (%)'])

# التنبؤ
pred_rej = model_rej.predict(input_df)
pred_log_flux = model_flx.predict(input_df)
# المعادلة العكسية: Flux = 10^y - 1
real_flux = (10 ** pred_log_flux) - 1

# تحذير النطاق العام
out_of_range = any([pore_area < stats_rej['Pore Area (Å²)']['min'], pore_area > stats_rej['Pore Area (Å²)']['max'],
                    press < stats_rej['Applied pressure  (MPa)']['min'], press > stats_rej['Applied pressure  (MPa)']['max']])

# --- 6. توزيع الواجهة الرئيسية (3 أعمدة) ---
col_inputs, col_viz, col_outputs = st.columns([1, 2])

with col_outputs:
    st.subheader("🎯 Prédictions")
    st.metric("Salt Rejection", f"{pred_rej:.2f} %")
    st.metric("Water Flux", f"{real_flux:.4f} mol/ns")
    
    if out_of_range:
        st.warning("⚠️ Attention : Entrées hors plage d'entraînement. Résultats potentiellement imprécis.")
    
    if st.button("➕ Ajouter à la simulation"):
        new_row = pd.DataFrame([[geom, pore_area, press, conc, temp, chem, porosity, pred_rej, real_flux]], 
                                columns=st.session_state.history.columns)
        st.session_state.history = pd.concat([st.session_state.history, new_row], ignore_index=True)

    if st.button("🔗 Partager Simulation"):
        st.info("Lien copié (Simulé) : Utilisez les paramètres d'URL pour partager.")

with col_viz:
    st.subheader("🖼️ Visualisation du Pore")
    fig, ax = plt.subplots(figsize=(4, 4))
    circle_size = np.sqrt(pore_area / np.pi)
    
    if geom == "Hexagonal":
        polygon = plt.Polygon(plt.Circle((0.5, 0.5), circle_size/10).get_path().vertices, closed=True, 
                              edgecolor=chem_colors[chem], facecolor='#E5E7EB', linewidth=5)
        ax.add_patch(polygon)
    else:
        rect = plt.Rectangle((0.5-circle_size/20, 0.5-circle_size/20), circle_size/10, circle_size/10, 
                             edgecolor=chem_colors[chem], facecolor='#E5E7EB', linewidth=5)
        ax.add_patch(rect)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    st.pyplot(fig)

# --- 7. أسفل الشاشة: الجدول والمبيانات ---
st.divider()
col_table, col_plots = st.columns([1])

with col_table:
    st.subheader("📋 Tableau des Simulations")
    st.dataframe(st.session_state.history.style.background_gradient(subset=['Salt Rejection (%)'], cmap='RdYlGn'))
    if st.button("🗑️ Effacer la dernière ligne"):
        st.session_state.history = st.session_state.history.iloc[:-1]

with col_plots:
    st.subheader("📊 Analyse Dynamique")
    if not st.session_state.history.empty:
        x_axis = st.selectbox("Axe X", st.session_state.history.columns[:7])
        y_axis = st.selectbox("Axe Y", ["Salt Rejection (%)", "Water Flux (mol/ns)"])
        st.scatter_chart(st.session_state.history, x=x_axis, y=y_axis)
    else:
        st.info("Ajoutez des simulations pour voir les graphiques.")

# --- 8. التذييل (Footer بالفرنسية) ---
st.markdown("<br><br><hr>", unsafe_allow_html=True)
st.markdown("""
    <div style='text-align: center; color: gray; font-size: 14px;'>
    Cette application a été réalisée dans le cadre du projet de fin d'études de Master (<b>Master S.A.Q.E</b>), 
    Faculté des Sciences, Université Mohammed V de Rabat.<br>
    <b>Par l'étudiant :</b> LAARAJ Lahcen <br>
    Tous les fichiers (Data, Codes Python, Orange Workflow) sont archivés sur <b>GitHub</b> et accessibles gratuitement.
    </div>
    """, unsafe_allow_html=True)