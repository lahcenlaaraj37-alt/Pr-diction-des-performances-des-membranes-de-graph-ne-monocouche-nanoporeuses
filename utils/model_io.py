from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from joblib import load


@dataclass(frozen=True)
class LoadedModel:
    pipeline: Any
    metadata: dict


def load_pipeline_and_metadata(pkl_path: str, metadata_path: str) -> LoadedModel:
    pkl = Path(pkl_path)
    meta = Path(metadata_path)
    if not pkl.exists():
        raise FileNotFoundError(f"Model file not found: {pkl_path}")
    if not meta.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    pipeline = load(pkl)
    metadata = json.loads(meta.read_text(encoding="utf-8"))
    return LoadedModel(pipeline=pipeline, metadata=metadata)


def inverse_log_flux_plus1_to_flux(log_flux_plus1: float, log_base: int = 10) -> float:
    # You specified log10, not natural log.
    return (log_base**log_flux_plus1) - 1.0

