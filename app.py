import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# --- 1. إعدادات الصفحة والعناوين (بالفرنسية) ---
st.set_page_config(page_title="Graphene Membrane Simulator", layout="wide")

# العنوان الرئيسي كما طلبته [3]
st.markdown("""
    <h1 style='text-align: center; color: #1E3A8A; font-family: sans-serif;'>
    Modélisation et prédiction des performances des membranes de graphène monocouche nanoporeuses pour le dessalement de l'eau de mer.
    </h1>
    <hr>
    """, unsafe_allow_html=True)

# --- 2. تحميل الأصول (النماذج والإحصائيات) ---
@st.cache_resource
def load_assets():
    # تحميل الخبيرين والحدود الإحصائية [4]
    model_rejection = joblib.load('model_salt_rejection.pkl')
    model_flux = joblib.load('model_flux_log.pkl')
    stats_rej = joblib.load('training_stats.pkl')
    return model_rejection, model_flux, stats_rej

try:
    model_rej, model_flx, stats_rej = load_assets()
except Exception as e:
    st.error(f"Erreur de chargement des modèles : {e}")

# إدارة ذاكرة المحاكاة
if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=[
        'Géométrie', 'Pore Area (Å²)', 'Pressure (MPa)', 'Conc (ppm)', 'Temp (°C)', 
        'Chemistry', 'Porosity (%)', 'Salt Rejection (%)', 'Water Flux (mol/ns)'
    ])

# --- 3. الجانب الأيسر: المدخلات (Sidebar) ---
st.sidebar.header("🛠️ Paramètres d'Entrée")

# زر تحميل ملف إكسل (Batch)
uploaded_file = st.sidebar.file_uploader("Charger un fichier Excel (Batch)", type=["xlsx"])

# كيمياء المسام والألوان المرتبطة بها
chem_colors = {"OH": "#3498DB", "H": "#95A5A6", "COOH": "#E74C3C", "None": "#2C3E50"}

def validate_input(label, min_val, max_val, default):
    val = st.sidebar.number_input(label, value=float(default))
    if val < min_val or val > max_val:
        st.sidebar.markdown(f"<span style='color:red;'>Hors plage ! [{min_val} - {max_val}]</span>", unsafe_allow_html=True)
    return val

# المدخلات السبعة
geom = st.sidebar.selectbox("Géométrie", ["Hexagonal", "Cubic"])
pore_area = validate_input("Pore Area (Å²)", stats_rej['Pore Area (Å²)']['min'], stats_rej['Pore Area (Å²)']['max'], 16.3)
press = validate_input("Applied pressure (MPa)", stats_rej['Applied pressure  (MPa)']['min'], stats_rej['Applied pressure  (MPa)']['max'], 100.0)
conc = validate_input("Feed Concentration (ppm)", stats_rej['Feed Concentration  (ppm)']['min'], stats_rej['Feed Concentration  (ppm)']['max'], 35000.0)
temp = validate_input("Température (°C)", stats_rej['Température  (°C)']['min'], stats_rej['Température  (°C)']['max'], 27.0)
chem = st.sidebar.selectbox("Pore Chemistry", list(chem_colors.keys()))
porosity = validate_input("Porosity (%)", stats_rej['Porosity (%)']['min'], stats_rej['Porosity (%)']['max'], 10.0)

# --- 4. التنبؤات الرياضية ---
input_df = pd.DataFrame([[geom, pore_area, press, conc, temp, chem, porosity]], 
                         columns=['Géométrie', 'Pore Area (Å²)', 'Applied pressure  (MPa)', 'Feed Concentration  (ppm)', 'Température  (°C)', 'Pore Chemistry (Functionalization)', 'Porosity (%)'])

res_rej = model_rej.predict(input_df)
res_log_flux = model_flx.predict(input_df)
# المعادلة العكسية للوغاريتم العشري: Flux = 10^y - 1
res_water_flux = (10**res_log_flux) - 1

# --- 5. توزيع الواجهة الرئيسية (التصحيح البرمجي) ---
# تقسيم الجزء العلوي لعمودين فقط (الوسط لليفيزيوليزاسيون واليمين للنتائج)
col_viz, col_outputs = st.columns([1, 2])

with col_outputs:
    st.subheader("🎯 Prédictions Finales")
    st.info(f"**Salt Rejection:** {res_rej:.2f} %")
    st.success(f"**Water Flux:** {res_water_flux:.4f} mol/ns")
    
    # زر الإضافة للجدول
    if st.button("➕ Ajouter à la simulation"):
        new_row = pd.DataFrame([[geom, pore_area, press, conc, temp, chem, porosity, res_rej, res_water_flux]], 
                                columns=st.session_state.history.columns)
        st.session_state.history = pd.concat([st.session_state.history, new_row], ignore_index=True)

with col_viz:
    st.subheader("🖼️ Visualisation du Pore")
    fig, ax = plt.subplots(figsize=(4, 4))
    # حجم المسام بناءً على النفادية (Porosity) ومساحة المسام
    size = np.sqrt(pore_area) / 10
    if geom == "Hexagonal":
        # رسم شكل سداسي مبسط
        polygon = plt.Polygon([[-size,0], [-size/2,size], [size/2,size], [size,0], [size/2,-size], [-size/2,-size]], 
                              closed=True, edgecolor=chem_colors[chem], facecolor='#D1D5DB', linewidth=4)
        ax.add_patch(polygon)
    else:
        rect = plt.Rectangle((-size/2, -size/2), size, size, edgecolor=chem_colors[chem], facecolor='#D1D5DB', linewidth=4)
        ax.add_patch(rect)
    ax.set_xlim(-2, 2); ax.set_ylim(-2, 2); ax.axis('off')
    st.pyplot(fig)

# --- 6. الجزء السفلي: الجدول والمبيانات ---
st.divider()
col_table, col_plots = st.columns([1])

with col_table:
    st.subheader("📋 Historique des Simulations")
    st.dataframe(st.session_state.history)
    if st.button("🗑️ Supprimer"): st.session_state.history = st.session_state.history.iloc[:-1]

with col_plots:
    st.subheader("📊 Analyses Graphiques")
    if not st.session_state.history.empty:
        x_ax = st.selectbox("Axe X", st.session_state.history.columns[:7])
        y_ax = st.selectbox("Axe Y", ["Salt Rejection (%)", "Water Flux (mol/ns)"])
        st.scatter_chart(st.session_state.history, x=x_ax, y=y_ax)

# --- 7. التذييل (Footer) ---
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
    <div style='text-align: center; font-size: 13px; color: gray;'>
    Projet de fin d'études Master S.A.Q.E - Université Mohammed V de Rabat <br>
    Réalisé par : <b>LAARAJ Lahcen</b> | Fichiers disponibles sur GitHub.
    </div>
    """, unsafe_allow_html=True)
