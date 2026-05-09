from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RangeCheck:
    in_range: bool
    min_value: float | None
    max_value: float | None


def check_numeric_in_range(value: float | int | None, min_value: float, max_value: float) -> RangeCheck:
    if value is None:
        return RangeCheck(in_range=False, min_value=min_value, max_value=max_value)
    return RangeCheck(in_range=(min_value <= float(value) <= max_value), min_value=min_value, max_value=max_value)


def any_out_of_range(range_checks: dict[str, RangeCheck]) -> bool:
    return any(not rc.in_range for rc in range_checks.values())


def coerce_row_to_model_input(row: dict[str, Any], features_columns: list[str]) -> dict[str, Any]:
    # Keeps only required features and preserves column order later in DataFrame construction.
    return {col: row.get(col, None) for col in features_columns}

