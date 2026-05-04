from __future__ import annotations

import io
from typing import Literal

import pandas as pd


def dataframe_to_bytes(df: pd.DataFrame, fmt: Literal["csv", "xlsx"] = "csv") -> bytes:
    if fmt == "csv":
        return df.to_csv(index=False).encode("utf-8")

    buffer = io.BytesIO()
    # Use openpyxl to avoid requiring xlsxwriter in local environments.
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="simulations")
    return buffer.getvalue()

