<div align="center">

<!-- HEADER BANNER -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:1B3A6B,100:4A90D9&height=200&section=header&text=Aqua.LA.Graph-Lite%20v1.0&fontSize=42&fontColor=ffffff&fontAlignY=38&desc=ML%20Web%20Tool%20for%20Nanoporous%20Single-Layer%20Graphene%20Membranes&descAlignY=60&descSize=16&descColor=a8c8e8" width="100%"/>

<br/>

<!-- BADGES ROW 1 -->
[![Streamlit App](https://img.shields.io/badge/🌊%20Live%20App-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://pfe-fsr-graphene-laaraj-lahcen-2026.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![CatBoost](https://img.shields.io/badge/Model-CatBoost-yellow?style=for-the-badge&logo=yandex&logoColor=black)](https://catboost.ai)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<!-- BADGES ROW 2 -->
[![Open Science](https://img.shields.io/badge/Open%20Science-100%25%20Reproducible-27AE60?style=for-the-badge&logo=opensourceinitiative&logoColor=white)]()
[![Dataset](https://img.shields.io/badge/Dataset-460%20MD%20Data%20Points-1B3A6B?style=for-the-badge&logo=microsoftexcel&logoColor=white)]()
[![Articles](https://img.shields.io/badge/Sources-12%20Peer--Reviewed%20Articles-4A90D9?style=for-the-badge&logo=googlescholar&logoColor=white)]()
[![University](https://img.shields.io/badge/Université%20Mohammed%20V-Rabat-E74C3C?style=for-the-badge)]()

<br/>

> **🎓 Projet de Fin d'Études (PFE) — Master S.A.Q.E**
> *Sciences Analytiques, Qualité & Environnement*
> Faculté des Sciences — Université Mohammed V, Rabat · 2025–2026

<br/>

**[🌊 Accéder à l'application](https://pfe-fsr-graphene-laaraj-lahcen-2026.streamlit.app/)** · **[📊 Explorer les données](DATA%20ARTCLE%201%20A%2012%20FINAL.xlsx)** · **[📓 Notebooks Python](Jupyter%20Notebook%20(ANACONDA_NAVIGATOR(Paython))%20Salt_Rejection%20AND%20log(flux%2B1))** · **[🟠 Workflows Orange](Model%20Orange%20Data%20Mining%20.ows)**

</div>

---

## 📋 Table des matières

- [🌍 Contexte et motivation](#-contexte-et-motivation)
- [🎯 À propos de l'outil](#-à-propos-de-loutil)
- [✨ Fonctionnalités](#-fonctionnalités)
- [🔬 Méthodologie scientifique](#-méthodologie-scientifique)
  - [Collecte des données](#collecte-des-données--de-50-à-12-articles)
  - [Jeu de données](#jeu-de-données)
  - [Modélisation — Orange Data Mining](#modélisation--orange-data-mining)
  - [Validation Python](#validation-python--anaconda--jupyter-notebook)
- [🏆 Performances des modèles](#-performances-des-modèles)
- [🖥️ Application Web — Les 6 modules](#️-application-web--les-6-modules)
- [📁 Structure du dépôt](#-structure-du-dépôt)
- [🚀 Installation et exécution locale](#-installation-et-exécution-locale)
- [📦 Dépendances](#-dépendances)
- [⚠️ Avertissement d'utilisation](#️-avertissement-dutilisation)
- [📚 Sources scientifiques](#-sources-scientifiques)
- [👨‍🎓 Auteur et encadrement](#-auteur-et-encadrement)
- [📄 Licence](#-licence)

---

## 🌍 Contexte et motivation

La **pénurie d'eau potable** est l'un des défis les plus critiques de notre époque, particulièrement ressentie en **Afrique du Nord** et dans le **Golfe Arabique**. Le dessalement de l'eau de mer est devenu une nécessité stratégique mondiale.

```
Technologies thermiques  ──►  Osmose inverse polymère  ──►  Graphène monocouche nanoperforé
   (>10 kWh/m³)               (2.5–5 kWh/m³)               (potentiel révolutionnaire) ✨
```

Parmi les technologies émergentes, le **graphène monocouche nanoperforé** (Single-Layer Nanoporous Graphene) représente une rupture technologique : membrane d'un seul atome d'épaisseur, percée de nanopores calibrés, combinant un flux d'eau exceptionnel et un taux de rejet du sel quasi parfait.

> ⚠️ **Cette technologie est exclusivement numérique aujourd'hui.** Son étude repose sur des simulations de dynamique moléculaire (LAMMPS, NAMD, GROMACS, AMBER) exigeant des supercalculateurs, des semaines de calcul et des budgets prohibitifs.

**Aqua.LA.Graph-Lite v1.0** brise ce verrou : prédictions en **quelques secondes**, sans code, sans supercalculateur.

---

## 🎯 À propos de l'outil

**Aqua.LA.Graph-Lite v1.0** est un outil web gratuit et open-source propulsé par l'apprentissage automatique (CatBoost), permettant de prédire instantanément les performances des membranes de graphène monocouche nanoporeuses pour le dessalement de l'eau de mer.

| | |
|---|---|
| **Entrées** | 7 paramètres physico-chimiques et opératoires |
| **Sorties** | Taux de rejet du sel (%) + Flux d'eau (molécule/ns) |
| **Algorithme** | CatBoost (Gradient Boosting) |
| **Données** | 460 points issus de 12 articles MD peer-reviewed |
| **Accès** | Gratuit, sans inscription, sans code |
| **Code** | 100% open-source (MIT) |

---

## ✨ Fonctionnalités

```
┌─────────────────────────────────────────────────────────────────┐
│                    Aqua.LA.Graph-Lite v1.0                       │
├──────────────┬──────────────┬────────────┬───────────────────────┤
│  💧 Single   │ 📊 Batch     │ 📈 Vizual. │  🔄 Inverse           │
│  Simulation  │ Processing   │ & Reports  │  Prediction           │
├──────────────┴──────────────┴────────────┴───────────────────────┤
│           🤖 AI Chatbot (RAG)    +    ℹ️ About                   │
└─────────────────────────────────────────────────────────────────┘
```

- **💧 Simulation individuelle** — Prédiction instantanée avec visualisation 2D du nanopore + Feature Importance
- **📊 Traitement par lot** — Import Excel pour des centaines de configurations simultanées
- **📈 Visualisation & Rapports** — Tableaux, graphiques interactifs, export CSV/Excel
- **🔄 Prédiction inverse** — Trouver la surface de pore optimale pour des performances cibles
- **🤖 Chatbot IA (RAG)** — Assistant spécialisé basé sur GROQ + LangChain + FAISS
- **ℹ️ À propos** — Documentation complète du projet

---

## 🔬 Méthodologie scientifique

### Collecte des données — De 50 à 12 articles

Le jeu de données a été constitué exclusivement à partir de la **littérature scientifique internationale**, en l'absence de données expérimentales industrielles disponibles.

```
50 articles candidats identifiés
        │
        ▼ Analyse via NotebookLM (Google AI)
        │
38 exclus ──► matériau différent (MoS₂, BN, MXène...)
             ► données incomplètes ou incompatibles
             ► hors du domaine de validité
        │
        ▼
12 articles retenus  ──►  465 lignes brutes  ──►  460 points propres ✅
```

### Jeu de données

**460 points de données** — extraits manuellement de 12 articles MD peer-reviewed.

#### Variables d'entrée (7 features)

| # | Variable | Type | Description |
|---|----------|------|-------------|
| 1 | `Geometry` | Catégorielle | Forme du nanopore : Circular, Hexagonal, Rhombic, Triangular, Ozark |
| 2 | `Pore Area (Å²)` | Numérique | Surface du nanopore en angströms carrés |
| 3 | `Applied Pressure (MPa)` | Numérique | Pression hydrostatique appliquée |
| 4 | `Feed Concentration (ppm)` | Numérique | Concentration en sel de l'eau d'alimentation |
| 5 | `Temperature (°C)` | Numérique | Température de fonctionnement |
| 6 | `Pore Chemistry (Functionalization)` | Catégorielle | Chimie du pore : H, OH, Pristine, Si, N, NH, B, et 10 autres |
| 7 | `Porosity (%)` | Numérique | Pourcentage de porosité de la membrane |

#### Variables cibles (2 outputs)

| # | Variable | Modèle interne | Description |
|---|----------|---------------|-------------|
| 1 | `Salt Rejection (%)` | Direct | Taux de rejet du sel |
| 2 | `Water Flux (molécule/ns)` | `log(flux+1)` → reconversion | Flux d'eau (transformation log pour distribution large) |

> **Pourquoi `log(flux+1)` ?** Le flux varie de quelques molécules/ns à >1 500 molécules/ns. La transformation logarithmique normalise cette amplitude dynamique extrême et améliore significativement les performances de prédiction.

#### Chimies de pore disponibles

```
1-Ethyl  |  2-Ethyl  |  2-Methyl  |  4-Methyl  |  4F4H  |  8F8H  |  B
COO⁻     |  FH       |  H         |  N         |  NH    |  NO    |  OH
Pristine |  Si       |  Si(OH)₂   |  SiH₂
```

---

### Modélisation — Orange Data Mining

**Pipeline de prétraitement** appliqué aux deux modèles :

```python
Pipeline:
  1. Normalisation (μ=0, σ²=1)          ── variables numériques
  2. Encodage One-Hot                    ── variables catégorielles
  3. Imputation (moyenne/mode)           ── valeurs manquantes
```

**6 algorithmes comparés** en validation croisée 10-plis :

| Algorithme | Config. principale |
|-----------|-------------------|
| **CatBoost** ⭐ | 1 000 arbres · lr=0.050 · λ=3 · depth=6 |
| **XGBoost** | 1 000 arbres · lr=0.050 · λ=1 · depth=6 |
| Random Forest | 100 arbres · depth=10 |
| SVM | — |
| Decision Tree | — |
| Linear Regression | — |

---

### Validation Python — Anaconda / Jupyter Notebook

Validation indépendante en Python pour confirmer les résultats d'Orange Data Mining :

- ✅ **Protection contre le data leakage** via Pipeline scikit-learn
- ✅ **Validation croisée 10-folds** sur 100% des données
- ✅ **Test indépendant** split 80/20 (données jamais vues)
- ✅ **Concordance parfaite** du classement entre Orange et Python

---

## 🏆 Performances des modèles

### Modèle 1 — Salt Rejection (%)

| Phase | Modèle | R² | RMSE | MAE | MAPE |
|-------|--------|----|------|-----|------|
| **CV 10-folds** | **CatBoost** ⭐ | **0.885** | **7.114** | **3.637** | **5.907%** |
| CV 10-folds | XGBoost | 0.877 | 7.353 | 3.518 | 6.193% |
| CV 10-folds | Random Forest | 0.860 | 7.841 | 4.332 | 9.759% |
| **Test 20%** | **CatBoost** ⭐ | **0.901** | **7.033** | **3.519** | **4.690%** |
| Test 20% | XGBoost | 0.803 | 7.837 | 3.428 | 4.602% |
| Test 20% | Random Forest | 0.804 | 7.815 | 4.014 | 5.921% |

### Modèle 2 — log(flux+1) → Water Flux (molécule/ns)

| Phase | Modèle | R² | RMSE | MAE | MAPE |
|-------|--------|----|------|-----|------|
| **CV 10-folds** | **CatBoost** ⭐ | **0.940** | **0.169** | **0.093** | **14.69%** |
| CV 10-folds | XGBoost | 0.927 | 0.185 | 0.110 | 12.50% |
| CV 10-folds | Random Forest | 0.886 | 0.232 | 0.160 | 29.43% |
| **Test 20%** | **CatBoost** ⭐ | **0.934** | **0.181** | **0.102** | **10.60%** |
| Test 20% | XGBoost | 0.918 | 0.202 | 0.111 | 8.026% |
| Test 20% | Random Forest | 0.830 | 0.291 | 0.143 | 11.39% |

> ✅ **Classement identique dans Orange Data Mining ET Python** → Robustesse confirmée par double validation indépendante.

---

## 🖥️ Application Web — Les 6 modules

### 💧 Module 1 · Single Simulation

Prédiction individuelle à partir des 7 paramètres d'entrée.

- Interface intuitive avec sliders et menus déroulants
- Visualisation 2D animée de la membrane graphène et du nanopore
- Résultats instantanés : Salt Rejection (%) + Water Flux (molécule/ns)
- Graphique d'**importance des caractéristiques** : contribution de chaque input à la prédiction

---

### 📊 Module 2 · Batch Processing & Data

Traitement simultané de centaines de configurations via import Excel.

**Format requis du fichier Excel :**

```
9 colonnes exactes (7 inputs + 2 outputs calculés automatiquement)
```

| Colonne | Type | Exemple |
|---------|------|---------|
| `Geometry` | texte | `hexagonal`, `circular` |
| `Pore Area (Å²)` | numérique | `23.5` |
| `Applied pressure  (MPa)` | numérique | `100` |
| `Feed Concentration  (ppm)` | numérique | `35000` |
| `Temperature  (°C)` | numérique | `300` |
| `Pore Chemistry (Functionalization)` | texte | `OH`, `H`, `Pristine` |
| `Porosity (%)` | numérique | `8.5` |
| `Salt Rejection (%)` | *(calculé)* | — |
| `Water Flux (molecule/ns)` | *(calculé)* | — |

> ⚠️ Les noms de colonnes doivent correspondre **exactement** (sensible à la casse, espaces inclus).

---

### 📈 Module 3 · Data Visualization & Reports

Tableau agrégé de **toutes** les simulations effectuées (modules 1, 2 et 4).

- Export en **Excel (.xlsx)** ou **CSV**
- Générateur de graphiques bivariés : sortie = f(entrée au choix)
- Analyse comparative multi-configurations

---

### 🔄 Module 4 · Inverse Prediction

**Approche inverse** : définir les performances souhaitées → trouver la surface de pore optimale.

```
Input utilisateur:
  ├── Salt Rejection cible (%)
  ├── Water Flux cible (molécule/ns)
  ├── Température, Concentration, Porosité, Pression
  ├── Géométrie : Circular ou Hexagonal uniquement ✅
  └── Chimie : H, OH ou Pristine uniquement ✅

Output:
  └── Pore Area (Å²) optimal + performances réellement prédites
```

> Les géométries et chimies recommandées donnent les prédictions les plus fiables dans ce module.

---

### 🤖 Module 5 · AI Chatbot

Assistant conversationnel spécialisé, basé sur l'architecture **RAG** :

```
Utilisateur ──► Question
                    │
                    ▼
             FAISS Vector Store
           (knowledge_base.txt embeddings)
                    │
                    ▼
            LangChain Orchestration
                    │
                    ▼
              GROQ LPU (inférence)
                    │
                    ▼
            Réponse précise & sourcée
```

- **FAISS** — Base vectorielle pour la recherche sémantique
- **LangChain** — Orchestration du pipeline RAG
- **GROQ LPU** — Inférence ultra-rapide
- **Source exclusive** : [`knowledge_base.txt`](knowledge_base.txt)

---

### ℹ️ Module 6 · About

Documentation complète du projet : contexte, méthodologie, performances, avertissements, équipe et liens.

---

## 📁 Structure du dépôt

```
📦 Pr-diction-des-performances-des-membranes/
│
├── 📂 .devcontainer/                     # Configuration environnement dev
├── 📂 .streamlit/                        # Configuration Streamlit Cloud
│
├── 📂 Jupyter Notebook (ANACONDA_NAVIGATOR(Paython)) Salt_Rejection AND log(flux+1)/
│   └── 📓 *.ipynb                        # Notebooks de validation Python complets
│
├── 📂 Model Orange Data Mining .ows/     # Workflows Orange Data Mining complets
│   ├── 🟠 *Salt_Rejection*.ows           # Workflow rejet du sel
│   └── 🟠 *log(flux+1)*.ows             # Workflow flux d'eau
│
├── 📂 assets/                            # Ressources visuelles de l'application
├── 📂 models/                            # Modèles CatBoost entraînés
│   ├── 🤖 catboost_salt_rejection.pkl    # Pipeline Salt Rejection
│   ├── 🤖 catboost_log_flux.pkl          # Pipeline log(flux+1)
│   └── 📊 metadata.json                 # Plages de validité + catégories
│
├── 📂 utils/                             # Modules utilitaires partagés
│   ├── validation.py                     # Validation des entrées utilisateur
│   ├── plotting.py                       # Graphiques et visualisations
│   ├── export.py                         # Export Excel/CSV
│   └── chatbot.py                        # Interface chatbot RAG
│
├── 📄 app.py                             # Application Streamlit principale
├── 📊 DATA ARTCLE 1 A 12 FINAL.xlsx     # Jeu de données complet (460 points)
├── 📝 knowledge_base.txt                 # Base de connaissances du chatbot
├── 📋 requirements.txt                   # Dépendances Python
├── 🔒 .gitignore
└── 📖 README.md
```

---

## 🚀 Installation et exécution locale

### Prérequis

- Python 3.10+
- pip ou conda

### 1. Cloner le dépôt

```bash
git clone https://github.com/lahcenlaaraj37-alt/Pr-diction-des-performances-des-membranes-de-graph-ne-monocouche-nanoporeuses.git
cd Pr-diction-des-performances-des-membranes-de-graph-ne-monocouche-nanoporeuses
```

### 2. Créer un environnement virtuel (recommandé)

```bash
# Avec conda (Anaconda Navigator)
conda create -n aqua-graphene python=3.10
conda activate aqua-graphene

# Ou avec venv
python -m venv venv
source venv/bin/activate       # Linux/macOS
venv\Scripts\activate          # Windows
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Configurer les secrets (chatbot)

Créer `.streamlit/secrets.toml` :

```toml
GROQ_API_KEY = "votre_clé_groq_ici"
```

> 💡 Obtenez une clé gratuite sur [console.groq.com](https://console.groq.com)

### 5. Lancer l'application

```bash
streamlit run app.py
```

L'application s'ouvre automatiquement sur `http://localhost:8501`

### 6. (Optionnel) Ré-entraîner les modèles

Les modèles entraînés sont déjà disponibles dans `models/`. Si vous souhaitez les ré-entraîner depuis le jeu de données :

```bash
python train_salt_rejection_model.py
python train_log_flux_model.py
```

---

## 📦 Dépendances

```
streamlit          # Framework web
catboost           # Algorithme principal de prédiction
scikit-learn       # Pipeline ML, prétraitement
pandas             # Manipulation des données
numpy              # Calcul numérique
plotly             # Visualisations interactives
matplotlib         # Graphiques statiques
openpyxl           # Export Excel
langchain          # Orchestration RAG
faiss-cpu          # Base vectorielle sémantique
groq               # Inférence LLM ultra-rapide
```

Voir [`requirements.txt`](requirements.txt) pour les versions exactes.

---

## ⚠️ Avertissement d'utilisation

> **Aqua.LA.Graph-Lite v1.0 est un outil de support à la recherche — non un substitut aux simulations rigoureuses.**

- Les prédictions sont basées sur **460 points de données** — couverture limitée de l'espace de conception
- Les résultats ne sont fiables **qu'à l'intérieur des plages de validité** des données d'entraînement
- Toute décision scientifique ou technique basée sur cet outil **doit être validée** par des simulations de dynamique moléculaire ou des expériences en laboratoire
- Les développeurs et superviseurs **déclinent toute responsabilité** pour des décisions prises sur la seule base des sorties de l'outil

**La véritable valeur ajoutée** : compresser des semaines de calcul en quelques secondes pour identifier rapidement les configurations prometteuses à soumettre à une validation computationnelle approfondie.

---

## 📚 Sources scientifiques

Les 12 articles ayant fourni les données d'entraînement (sélectionnés parmi 50 candidats via analyse critique avec NotebookLM) :

| # | DOI | Titre (abrégé) |
|---|-----|----------------|
| 1 | [10.1021/nl3012853](https://doi.org/10.1021/nl3012853) | Water Desalination across Nanoporous Graphene |
| 2 | [10.1063/1.5143069](https://doi.org/10.1063/1.5143069) | Charged nanoporous graphene membranes for water desalination |
| 3 | [10.2166/wst.2024.334](https://doi.org/10.2166/wst.2024.334) | Optimizing water transport through graphene-based membranes |
| 4 | [10.1016/j.molliq.2020.112478](https://doi.org/10.1016/j.molliq.2020.112478) | Influence of pore functionalization on water flux |
| 5 | [10.1021/acs.jctc.8b00226](https://doi.org/10.1021/acs.jctc.8b00226) | Silicon-based pore chemistry in graphene membranes |
| 6 | [10.1021/acsnano.1c05345](https://doi.org/10.1021/acsnano.1c05345) | Machine Learning Assisted Screening of 2D Materials |
| 7 | [10.1021/acs.jpcb.1c06327](https://doi.org/10.1021/acs.jpcb.1c06327) | Ozark Graphene Nanopore for Efficient Water Desalination |
| 8 | [10.1016/j.commatsci.2016.07.036](https://doi.org/10.1016/j.commatsci.2016.07.036) | Computational design of porous graphenes for separation |
| 9 | [10.1016/j.carbon.2017.01.099](https://doi.org/10.1016/j.carbon.2017.01.099) | Interface-induced affinity sieving in nanoporous graphenes |
| 10 | [10.1038/s41699-021-00246-9](https://doi.org/10.1038/s41699-021-00246-9) | Single-layer graphene as a potential filtration membrane |
| 11 | [10.1039/C7NR07963J](https://doi.org/10.1039/C7NR07963J) | Effective ion exclusion through functionalized graphene pores |
| 12 | [10.1007/s11242-022-01870-9](https://doi.org/10.1007/s11242-022-01870-9) | Transport of Water with Various Ions Through Nanoporous Graphene |

---

## 👨‍🎓 Auteur et encadrement

<div align="center">

### 🎓 Étudiant

**LAARAJ LAHCEN**
*Master S.A.Q.E — Sciences Analytiques, Qualité & Environnement*
Faculté des Sciences, Université Mohammed V, Rabat

*Spécialiste pédagogique — Supervision des laboratoires scolaires*

[![Email](https://img.shields.io/badge/Email-lahcen__laaraj%40um5.ac.ma-D44638?style=flat-square&logo=gmail&logoColor=white)](mailto:lahcen_laaraj@um5.ac.ma)
[![Email](https://img.shields.io/badge/Email-lahcenlaaraj37%40gmail.com-D44638?style=flat-square&logo=gmail&logoColor=white)](mailto:lahcenlaaraj37@gmail.com)
[![GitHub](https://img.shields.io/badge/GitHub-lahcenlaaraj37--alt-181717?style=flat-square&logo=github)](https://github.com/lahcenlaaraj37-alt)

<br/>

### 👨‍🏫 Encadrants

| | |
|:---:|:---:|
| **Prof. S. EL HAJJAJI** | **Prof. Z. ZITI** |
| Faculté des Sciences | Faculté des Sciences |
| Université Mohammed V, Rabat | Université Mohammed V, Rabat |

<br/>

### 🏛️ Institution

**Faculté des Sciences — Université Mohammed V, Rabat**
*Année universitaire 2025–2026*

</div>

---

## 🙏 Remerciements

Nous remercions chaleureusement :

- **L'Université Mohammed V** et la **Faculté des Sciences de Rabat** pour leur soutien académique
- **La communauté scientifique internationale** dont les études de dynamique moléculaire publiées ont rendu cette approche data-driven possible
- Les équipes de **CatBoost (Yandex)**, **Streamlit**, **Orange Data Mining**, **GROQ**, **LangChain** et **FAISS** pour leurs outils open-source remarquables

---

## 📄 Licence

Ce projet est distribué sous licence **MIT**. Vous êtes libre d'explorer, forker, auditer et citer ce travail.

```
MIT License — Copyright (c) 2026 LAARAJ LAHCEN
```

> 💡 **Science ouverte** : La reproductibilité totale est un principe fondamental de ce projet.
> Toutes les données, modèles, notebooks et le code source sont librement accessibles.

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:4A90D9,100:1B3A6B&height=100&section=footer" width="100%"/>

**⭐ Si ce projet vous est utile, n'hésitez pas à lui attribuer une étoile sur GitHub !**

*Aqua.LA.Graph-Lite v1.0 — Accelerating graphene membrane research, one prediction at a time.*

[![Live App](https://img.shields.io/badge/🌊%20Try%20the%20App-pfe--fsr--graphene--laaraj--lahcen--2026.streamlit.app-FF4B4B?style=for-the-badge)](https://pfe-fsr-graphene-laaraj-lahcen-2026.streamlit.app/)

</div>
