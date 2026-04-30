import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# --- 1. إعدادات الصفحة والعناوين الاحترافية بالفرنسية ---
st.set_page_config(page_title="Graphene Membrane Simulator", layout="wide")

# العنوان الرئيسي للمشروع (Header)
st.markdown("""
    <h1 style='text-align: center; color: #1E3A8A; font-family: sans-serif; font-size: 28px;'>
    Modélisation et prédiction des performances des membranes de graphène monocouche nanoporeuses pour le dessalement de l'eau de mer.
    </h1>
    <hr>
    """, unsafe_allow_html=True)

# --- 2. تحميل النماذج والبيانات الإحصائية (الخبراء الأربعة) ---
@st.cache_resource
def load_assets():
    # تحميل الخبيرين والحدود الإحصائية المخزنة في GitHub [3]
    model_rejection = joblib.load('model_salt_rejection.pkl')
    model_flux = joblib.load('model_flux_log.pkl')
    stats_rej = joblib.load('training_stats.pkl')
    stats_flux = joblib.load('training_stats_flux.pkl')
    return model_rejection, model_flux, stats_rej, stats_flux

try:
    model_rej, model_flx, stats_rej, stats_flx = load_assets()
except Exception as e:
    st.error(f"Erreur de chargement des fichiers modèles (.pkl) : {e}")

# إدارة حالة الجلسة لتخزين المحاكات (Session State)
if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=[
        'Géométrie', 'Pore Area (Å²)', 'Pressure (MPa)', 'Conc (ppm)', 'Temp (°C)', 
        'Chemistry', 'Porosity (%)', 'Salt Rejection (%)', 'Water Flux (mol/ns)'
    ])

# --- 3. الجانب الأيسر: مدخلات المستخدم (Sidebar) ---
st.sidebar.header("🛠️ Paramètres d'Entrée")

# ميزة التحميل الجماعي للملفات (Batch Upload)
uploaded_file = st.sidebar.file_uploader("Charger un fichier Excel (Batch)", type=["xlsx"])

# نظام الألوان المرتبط بكيمياء المسام
chem_colors = {"OH": "#3498DB", "H": "#95A5A6", "COOH": "#E74C3C", "None": "#2C3E50"}

# دالة للتحقق من النطاق وعرض تحذير أحمر عند الخروج عنه
def validate_input(label, min_val, max_val, default):
    val = st.sidebar.number_input(label, value=float(default))
    if val < min_val or val > max_val:
        st.sidebar.markdown(f"<span style='color:red; font-size:12px;'>Hors plage ! La plage d'entraînement est [{min_val:.2f} - {max_val:.2f}]</span>", unsafe_allow_html=True)
    return val

# المدخلات السبعة الأساسية
geom = st.sidebar.selectbox("Géométrie", ["Hexagonal", "Cubic"])
pore_area = validate_input("Pore Area (Å²)", stats_rej['Pore Area (Å²)']['min'], stats_rej['Pore Area (Å²)']['max'], 16.3)
press = validate_input("Applied pressure (MPa)", stats_rej['Applied pressure  (MPa)']['min'], stats_rej['Applied pressure  (MPa)']['max'], 100.0)
conc = validate_input("Feed Concentration (ppm)", stats_rej['Feed Concentration  (ppm)']['min'], stats_rej['Feed Concentration  (ppm)']['max'], 35000.0)
temp = validate_input("Température (°C)", stats_rej['Température  (°C)']['min'], stats_rej['Température  (°C)']['max'], 27.0)
chem = st.sidebar.selectbox("Pore Chemistry", list(chem_colors.keys()))
porosity = validate_input("Porosity (%)", stats_rej['Porosity (%)']['min'], stats_rej['Porosity (%)']['max'], 10.0)

# --- 4. التنبؤات والتحويلات الرياضية ---
input_df = pd.DataFrame([[geom, pore_area, press, conc, temp, chem, porosity]], 
                         columns=['Géométrie', 'Pore Area (Å²)', 'Applied pressure  (MPa)', 'Feed Concentration  (ppm)', 'Température  (°C)', 'Pore Chemistry (Functionalization)', 'Porosity (%)'])

# التنبؤ واستخراج القيمة الفردية لتجنب خطأ التنسيق (Fixing TypeError Image 775) [2]
res_rej_array = model_rej.predict(input_df)
res_rej = res_rej_array 

res_log_flux_array = model_flx.predict(input_df)
res_log_flux = res_log_flux_array

# معادلة تحويل اللوغاريتم العشري: Water Flux = 10^(log_flux) - 1
res_water_flux = (10 ** res_log_flux) - 1

# تحقق عام من النطاق لعرض رسالة تحذير تحت النتائج
is_out = any([pore_area < stats_rej['Pore Area (Å²)']['min'], pore_area > stats_rej['Pore Area (Å²)']['max'],
              press < stats_rej['Applied pressure  (MPa)']['min'], press > stats_rej['Applied pressure  (MPa)']['max']])

# --- 5. توزيع الواجهة الرئيسية (التصحيح البرمجي لـ Columns) ---
# تقسيم الصفحة إلى عمودين فقط لتجنب ValueError (Fixing Image 774) [1]
col_viz, col_outputs = st.columns([4])

with col_outputs:
    st.markdown("### 🎯 Prédictions Finales")
    # عرض النتائج في صناديق ملونة
    st.info(f"**Salt Rejection:** {res_rej:.2f} %")
    st.success(f"**Water Flux:** {res_water_flux:.4f} mol/ns")
    
    if is_out:
        st.warning("⚠️ Attention : Données hors plage. Résultats potentiellement imprécis.")
    
    # زر إضافة المحاكاة للجدول
    if st.button("➕ Ajouter à la simulation"):
        new_row = pd.DataFrame([[geom, pore_area, press, conc, temp, chem, porosity, res_rej, res_water_flux]], 
                                columns=st.session_state.history.columns)
        st.session_state.history = pd.concat([st.session_state.history, new_row], ignore_index=True)

with col_viz:
    st.markdown("### 🖼️ Visualisation du Pore")
    fig, ax = plt.subplots(figsize=(4, 4))
    # حجم المسام يتأثر بمساحة المسام المدخلة
    size = np.sqrt(pore_area) / 10
    if geom == "Hexagonal":
        polygon = plt.Polygon([[-size,0], [-size/2,size], [size/2,size], [size,0], [size/2,-size], [-size/2,-size]], 
                              closed=True, edgecolor=chem_colors[chem], facecolor='#F3F4F6', linewidth=5)
        ax.add_patch(polygon)
    else:
        rect = plt.Rectangle((-size/2, -size/2), size, size, edgecolor=chem_colors[chem], facecolor='#F3F4F6', linewidth=5)
        ax.add_patch(rect)
    ax.set_xlim(-2, 2); ax.set_ylim(-2, 2); ax.axis('off')
    st.pyplot(fig)

# --- 6. الجزء السفلي: الجدول والتحليل البياني ---
st.divider()
col_table, col_plots = st.columns([1.5, 1])

with col_table:
    st.markdown("### 📋 Historique des Simulations")
    # عرض الجدول مع تلوين عمود النتائج
    st.dataframe(st.session_state.history.style.highlight_max(axis=0, color='#D1FAE5', subset=['Salt Rejection (%)']))
    if st.button("🗑️ Supprimer la dernière simulation"):
        if not st.session_state.history.empty:
            st.session_state.history = st.session_state.history.iloc[:-1]

with col_plots:
    st.markdown("### 📊 Analyses Dynamiques")
    if not st.session_state.history.empty:
        x_ax = st.selectbox("Axe X", st.session_state.history.columns[:7])
        y_ax = st.selectbox("Axe Y", ["Salt Rejection (%)", "Water Flux (mol/ns)"])
        st.scatter_chart(st.session_state.history, x=x_ax, y=y_ax)
    else:
        st.write("Ajoutez des données pour générer des graphiques.")

# --- 7. تذييل الموقع (Footer) بالفرنسية العلمية ---
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown(f"""
    <div style='text-align: center; color: #4B5563; font-size: 13px;'>
    Cette application a été réalisée dans le cadre du projet de fin d'études de Master (<b>Master S.A.Q.E</b>),<br> 
    Faculté des Sciences, Université Mohammed V de Rabat.<br>
    <b>Par l'étudiant :</b> LAARAJ Lahcen <br>
    Tous les fichiers (Data, Codes Python, Orange Workflow) هستند archivés sur <b>GitHub</b> و يمكن الوصول إليها مجاناً.
    </div>
    """, unsafe_allow_html=True)
