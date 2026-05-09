from __future__ import annotations

from typing import Literal

Lang = Literal["en", "fr", "ar"]


STRINGS: dict[str, dict[Lang, str]] = {
    "app_title": {
        "en": "GrapheneMem",
        "fr": "GrapheneMem",
        "ar": "GrapheneMem",
    },
    "app_subtitle": {
        "en": "Modeling and predicting single-layer graphene membrane performance for seawater desalination.",
        "fr": "Modélisation et prédiction des performances des membranes en graphène monocouche pour le dessalement.",
        "ar": "نمذجة وتنبؤ بأداء أغشية الغرافين أحادية الطبقة النانوية في تحلية مياه البحر.",
    },
    "nav_single": {"en": "Single Simulation", "fr": "Simulation unique", "ar": "المحاكاة الفردية"},
    "nav_batch": {"en": "Batch Processing & Data", "fr": "Traitement par lot & données", "ar": "محاكاة الدفعات والبيانات"},
    "nav_viz": {"en": "Data Visualization & Reports", "fr": "Visualisation & rapports", "ar": "التحليل البياني والتقارير"},
    "nav_inverse": {"en": "Inverse Prediction", "fr": "Prédiction inverse", "ar": "التصميم العكسي"},
    "nav_chat": {"en": "AI Chatbot", "fr": "Chatbot IA", "ar": "المساعد الذكي"},
    "nav_about": {"en": "About", "fr": "À propos", "ar": "من نحن"},
    "inputs": {"en": "Inputs", "fr": "Entrées", "ar": "المدخلات"},
    "predict": {"en": "Predict", "fr": "Prédire", "ar": "تنبؤ"},
    "membrane": {"en": "Membrane (2D)", "fr": "Membrane (2D)", "ar": "رسم ثنائي الأبعاد للغشاء"},
    "predictions": {"en": "Predictions", "fr": "Prédictions", "ar": "نتائج التنبؤ (المخرجات)"},
    "table": {"en": "Simulation table", "fr": "Tableau de simulation", "ar": "جدول المحاكاة"},
    "exports": {"en": "Exports", "fr": "Exports", "ar": "تصدير"},
    "download_csv": {"en": "Download CSV", "fr": "Télécharger CSV", "ar": "تحميل CSV"},
    "download_xlsx": {"en": "Download Excel", "fr": "Télécharger Excel", "ar": "تحميل Excel"},
    "build_pdf": {"en": "Build PDF", "fr": "Générer PDF", "ar": "تصدير PDF"},
    "out_of_range": {
        "en": "Out of range: enter a value between {min} and {max}.",
        "fr": "Hors plage : entrez une valeur entre {min} et {max}.",
        "ar": "خارج النطاق: أدخل قيمة بين {min} و {max}.",
    },
    "low_reliability": {
        "en": "Prediction computed, but one or more inputs are outside the training data range (lower reliability).",
        "fr": "Prédiction calculée, mais une ou plusieurs entrées sont hors de la plage d'entraînement (fiabilité moindre).",
        "ar": "تم حساب التنبؤ، لكن بعض المدخلات خارج نطاق بيانات التدريب (دقة أقل).",
    },
    "about_text": {
        "en": (
            "This web tool was developed as a Master's final-year project (S.A.Q.E — Analytical Sciences, Quality & Environment), "
            "Faculty of Sciences, Mohammed V University.\n\n"
            "Student: **LAARAJ LAHCEN**.\n\n"
            "Goal: modeling and predicting the performance of single-layer graphene nanopore membranes for seawater desalination, "
            "to reduce dependency on heavy computational software and high-end hardware that are not accessible to everyone."
        ),
        "fr": (
            "Cet outil web a été développé dans le cadre d'un projet de fin d'études (Master S.A.Q.E — Sciences analytiques, Qualité & Environnement), "
            "Faculté des Sciences, Université Mohammed V.\n\n"
            "Étudiant : **LAARAJ LAHCEN**.\n\n"
            "Objectif : modéliser et prédire les performances des membranes en graphène monocouche nanostructuré pour le dessalement, "
            "afin de réduire la dépendance aux logiciels lourds et aux ordinateurs coûteux."
        ),
        "ar": (
            "تم تطوير هذه الأداة في إطار مشروع نهاية السنة الدراسية (سلك الماستر S.A.Q.E — علوم التحليلية الجودة والبيئة)، "
            "كلية العلوم بجامعة محمد الخامس.\n\n"
            "الطالب: **LAARAJ LAHCEN**.\n\n"
            "الهدف: نمذجة وتنبؤ بأداء أغشية الغرافين أحادية الطبقة النانوية في تحلية مياه البحر، "
            "لتقليل الاعتماد على البرامج الحاسوبية الثقيلة والحواسيب الضخمة التي تحتاج وقتًا طويلًا وليست في متناول الجميع."
        ),
    },
}


def t(key: str, lang: Lang) -> str:
    if key not in STRINGS:
        return key
    return STRINGS[key].get(lang, STRINGS[key]["en"])

