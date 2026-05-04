import json
from pathlib import Path

import pandas as pd
from catboost import CatBoostRegressor
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path(__file__).with_name("DATA ARTCLE 1 A 12 FINAL.xlsx")
OUT_DIR = Path(__file__).with_name("models")
MODEL_PATH = OUT_DIR / "salt_rejection_pipeline.pkl"
META_PATH = OUT_DIR / "salt_rejection_metadata.json"


FEATURES_COLUMNS = [
    "Geometry",
    "Pore Area (Å²)",
    "Applied pressure  (MPa)",
    "Feed Concentration  (ppm)",
    "Temperature  (°C)",
    "Pore Chemistry (Functionalization)",
    "Porosity (%)",
]
TARGET_COLUMN = "Salt Rejection (%)"

NUMERIC_FEATURES = [
    "Pore Area (Å²)",
    "Applied pressure  (MPa)",
    "Feed Concentration  (ppm)",
    "Temperature  (°C)",
    "Porosity (%)",
]
CATEGORICAL_FEATURES = ["Geometry", "Pore Chemistry (Functionalization)"]


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Excel not found: {DATA_PATH}")

    df = pd.read_excel(DATA_PATH)

    missing = [c for c in (FEATURES_COLUMNS + [TARGET_COLUMN]) if c not in df.columns]
    if missing:
        raise KeyError(
            "Missing required columns in Excel:\n"
            + "\n".join(f"- {c!r}" for c in missing)
            + "\n\nAvailable columns:\n"
            + "\n".join(f"- {c!r}" for c in df.columns)
        )

    X = df[FEATURES_COLUMNS]
    y = df[TARGET_COLUMN]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        l2_leaf_reg=3,
        depth=6,
        random_seed=42,
        verbose=0,
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(X, y)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dump(pipeline, MODEL_PATH)

    meta = {
        "features_columns": FEATURES_COLUMNS,
        "target_column": TARGET_COLUMN,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "ranges": {
            col: {"min": float(pd.to_numeric(df[col], errors="coerce").min()), "max": float(pd.to_numeric(df[col], errors="coerce").max())}
            for col in NUMERIC_FEATURES
        },
        "categories": {
            col: sorted([x for x in df[col].dropna().astype(str).unique().tolist()])
            for col in CATEGORICAL_FEATURES
        },
    }
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metadata: {META_PATH}")


if __name__ == "__main__":
    main()

