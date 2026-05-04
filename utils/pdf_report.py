from __future__ import annotations

import io
from typing import Iterable

import pandas as pd
from reportlab.lib.utils import ImageReader
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas


def build_pdf_report(
    *,
    title: str,
    simulations_df: pd.DataFrame,
    plot_images_png: Iterable[bytes] = (),
) -> bytes:
    """
    Creates a simple PDF report containing:
    - a title
    - a table (as text, first rows if too long)
    - optional plot images (PNG bytes)

    We will enhance layout later to match the final UI.
    """
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4

    y = height - 2 * cm
    c.setFont("Helvetica-Bold", 16)
    c.drawString(2 * cm, y, title)
    y -= 1.2 * cm

    c.setFont("Helvetica", 10)
    c.drawString(2 * cm, y, f"Rows: {len(simulations_df)}  |  Columns: {len(simulations_df.columns)}")
    y -= 0.8 * cm

    # Table as text (lightweight)
    preview = simulations_df.head(25).copy()
    lines = [", ".join(map(str, preview.columns.tolist()))]
    for _, row in preview.iterrows():
        lines.append(", ".join(map(str, row.tolist())))

    c.setFont("Helvetica", 7)
    for line in lines:
        if y < 3 * cm:
            c.showPage()
            y = height - 2 * cm
            c.setFont("Helvetica", 7)
        c.drawString(2 * cm, y, line[:180])
        y -= 0.35 * cm

    # Add plots on new pages
    for img in plot_images_png:
        c.showPage()
        y = height - 2 * cm
        c.setFont("Helvetica-Bold", 12)
        c.drawString(2 * cm, y, "Selected Plot")
        y -= 1 * cm

        # place image
        img_reader = ImageReader(io.BytesIO(img))
        c.drawImage(img_reader, 2 * cm, 4 * cm, width=17 * cm, height=20 * cm, preserveAspectRatio=True, mask="auto")

    c.save()
    return buf.getvalue()

