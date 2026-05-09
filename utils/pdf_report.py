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

    We will enhance layout later to match final UI.
    """
    try:
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

        # Add timestamp
        import datetime
        c.setFont("Helvetica", 9)
        c.drawString(2 * cm, y, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        y -= 1.0 * cm

        # Table as text (lightweight)
        if len(simulations_df) > 0:
            preview = simulations_df.head(25).copy()
            
            # Format header
            header_line = " | ".join([str(col)[:15] for col in preview.columns])
            c.setFont("Helvetica-Bold", 8)
            c.drawString(2 * cm, y, header_line)
            y -= 0.5 * cm
            
            # Add separator line
            c.drawString(2 * cm, y, "-" * min(len(header_line), 180))
            y -= 0.4 * cm
            
            # Format data rows
            c.setFont("Helvetica", 7)
            for _, row in preview.iterrows():
                row_line = " | ".join([str(val)[:15] for val in row.tolist()])
                if y < 3 * cm:
                    c.showPage()
                    y = height - 2 * cm
                    c.setFont("Helvetica-Bold", 8)
                    c.drawString(2 * cm, y, header_line)
                    y -= 0.5 * cm
                    c.drawString(2 * cm, y, "-" * min(len(header_line), 180))
                    y -= 0.4 * cm
                    c.setFont("Helvetica", 7)
                c.drawString(2 * cm, y, row_line[:180])
                y -= 0.35 * cm
        else:
            c.setFont("Helvetica", 10)
            c.drawString(2 * cm, y, "No simulation data available")
            y -= 1.0 * cm

        # Add plots on new pages
        for i, img in enumerate(plot_images_png):
            c.showPage()
            y = height - 2 * cm
            c.setFont("Helvetica-Bold", 12)
            c.drawString(2 * cm, y, f"Plot {i+1}")
            y -= 1 * cm

            # place image with error handling
            try:
                img_reader = ImageReader(io.BytesIO(img))
                c.drawImage(img_reader, 2 * cm, 4 * cm, width=17 * cm, height=20 * cm, preserveAspectRatio=True, mask="auto")
            except Exception as img_error:
                c.setFont("Helvetica", 10)
                c.drawString(2 * cm, y, f"Error loading plot {i+1}: {str(img_error)}")

        c.save()
        return buf.getvalue()
    
    except Exception as e:
        # Return a simple error PDF
        try:
            buf = io.BytesIO()
            c = canvas.Canvas(buf, pagesize=A4)
            c.setFont("Helvetica-Bold", 16)
            c.drawString(2 * cm, height - 3 * cm, "PDF Generation Error")
            c.setFont("Helvetica", 12)
            c.drawString(2 * cm, height - 5 * cm, f"Error: {str(e)}")
            c.setFont("Helvetica", 10)
            c.drawString(2 * cm, height - 7 * cm, "Please try again or contact support.")
            c.save()
            return buf.getvalue()
        except Exception:
            # Last resort: return empty bytes
            return b"PDF generation failed"
