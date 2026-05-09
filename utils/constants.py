from __future__ import annotations

from dataclasses import dataclass
from typing import Final


FEATURES_COLUMNS: Final[list[str]] = [
    "Geometry",
    "Pore Area (Å²)",
    "Applied pressure  (MPa)",
    "Feed Concentration  (ppm)",
    "Temperature  (°C)",
    "Pore Chemistry (Functionalization)",
    "Porosity (%)",
]

TARGET_SALT_REJECTION: Final[str] = "Salt Rejection (%)"
TARGET_LOG_FLUX_PLUS1: Final[str] = "log(flux+1)"

NUMERIC_FEATURES: Final[list[str]] = [
    "Pore Area (Å²)",
    "Applied pressure  (MPa)",
    "Feed Concentration  (ppm)",
    "Temperature  (°C)",
    "Porosity (%)",
]

CATEGORICAL_FEATURES: Final[list[str]] = [
    "Geometry",
    "Pore Chemistry (Functionalization)",
]


@dataclass(frozen=True)
class ModelPaths:
    salt_rejection_pkl: str
    salt_rejection_meta: str
    log_flux_pkl: str
    log_flux_meta: str


DEFAULT_MODEL_PATHS: Final[ModelPaths] = ModelPaths(
    salt_rejection_pkl="models/salt_rejection_pipeline.pkl",
    salt_rejection_meta="models/salt_rejection_metadata.json",
    log_flux_pkl="models/log_flux_plus1_pipeline.pkl",
    log_flux_meta="models/log_flux_plus1_metadata.json",
)


# Used for UI conversion (you requested base-10, not ln)
LOG_BASE: Final[int] = 10


# Chemistry → color mapping (fallback palette).
# We will refine once we lock the exact categories from metadata in the app.
CHEMISTRY_COLOR_FALLBACK: Final[list[str]] = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

