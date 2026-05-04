from __future__ import annotations

import math
from typing import Any

import numpy as np
import plotly.graph_objects as go


def _regular_polygon(n: int, radius: float, rotation: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0, 2 * math.pi, n, endpoint=False) + rotation
    return radius * np.cos(t), radius * np.sin(t)


def _shape_outline(geometry: str, radius: float = 1.05) -> tuple[np.ndarray, np.ndarray]:
    g = (geometry or "").strip().lower()
    theta = np.linspace(0, 2 * math.pi, 240)

    if "circle" in g:
        return radius * np.cos(theta), radius * np.sin(theta)
    if "hex" in g:
        x, y = _regular_polygon(6, radius, rotation=math.pi / 6)
        return np.r_[x, x[0]], np.r_[y, y[0]]
    if "tri" in g:
        x, y = _regular_polygon(3, radius, rotation=-math.pi / 2)
        return np.r_[x, x[0]], np.r_[y, y[0]]
    if "rhomb" in g:
        # diamond
        x = np.array([0.0, radius, 0.0, -radius, 0.0])
        y = np.array([radius * 0.85, 0.0, -radius * 0.85, 0.0, radius * 0.85])
        return x, y
    if "ozark" in g:
        # ellipse-ish
        return radius * 1.2 * np.cos(theta), radius * 0.8 * np.sin(theta)

    # fallback
    return radius * np.cos(theta), radius * np.sin(theta)


def make_graphene_membrane_2d(
    *,
    geometry: str,
    pore_chemistry: str,
    chemistry_color: str,
    lattice_half_size: float = 5.5,
) -> go.Figure:
    """
    2D graphene-like hex lattice (black carbon atoms) with a single pore shape cut-out.
    The pore rim is colored by chemistry, and the geometry name is shown on the plot.
    """
    # Build a simple honeycomb lattice in a SQUARE window (to match your reference UI).
    a = 0.32  # spacing
    points = []
    for i in range(-80, 81):
        for j in range(-80, 81):
            x = a * (i + j / 2)
            y = a * (j * math.sqrt(3) / 2)
            if abs(x) <= lattice_half_size and abs(y) <= lattice_half_size:
                points.append((x, y))
    pts = np.array(points, dtype=float)

    # Bonds: connect near neighbors (distance ~ a)
    # Use a lightweight grid-hash to avoid O(n^2).
    cell = a * 1.1
    grid: dict[tuple[int, int], list[int]] = {}
    for idx, (x, y) in enumerate(pts):
        key = (int(x // cell), int(y // cell))
        grid.setdefault(key, []).append(idx)

    bonds_x: list[float] = []
    bonds_y: list[float] = []
    thr = a * 1.05
    for idx, (x, y) in enumerate(pts):
        gx, gy = int(x // cell), int(y // cell)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for jdx in grid.get((gx + dx, gy + dy), []):
                    if jdx <= idx:
                        continue
                    x2, y2 = pts[jdx]
                    d = math.hypot(x2 - x, y2 - y)
                    if d <= thr:
                        bonds_x += [x, x2, None]
                        bonds_y += [y, y2, None]

    # Remove atoms + bonds inside pore cut radius (simple cut; shape outline is shown separately)
    pore_cut_r = 1.12
    keep = (pts[:, 0] ** 2 + pts[:, 1] ** 2) >= (pore_cut_r**2)
    pts_kept = pts[keep]

    bonds = go.Scatter(
        x=bonds_x,
        y=bonds_y,
        mode="lines",
        line=dict(color="rgba(17,24,39,0.35)", width=1),
        hoverinfo="skip",
        showlegend=False,
    )

    atoms = go.Scatter(
        x=pts_kept[:, 0],
        y=pts_kept[:, 1],
        mode="markers",
        marker=dict(size=5, color="#111111", opacity=0.95),
        hoverinfo="skip",
        showlegend=False,
    )

    outline_x, outline_y = _shape_outline(geometry, radius=pore_cut_r)
    pore_outline = go.Scatter(
        x=outline_x,
        y=outline_y,
        mode="lines",
        line=dict(color=chemistry_color, width=6),
        hoverinfo="skip",
        showlegend=False,
    )

    fig = go.Figure(data=[bonds, atoms, pore_outline])
    fig.update_layout(
        margin=dict(l=0, r=0, t=35, b=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(range=[-lattice_half_size, lattice_half_size])
    fig.update_yaxes(range=[-lattice_half_size, lattice_half_size])

    fig.add_annotation(
        text=f"<b>Geometry:</b> {geometry} &nbsp;&nbsp; <b>Chemistry:</b> {pore_chemistry}",
        x=0.0,
        y=1.12,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=12, color="#1D4ED8"),
        align="left",
    )
    return fig


def make_scatter(df, x_col: str, y_col: str, color_col: str | None = None) -> go.Figure:
    marker: dict[str, Any] = dict(size=9, opacity=0.85, line=dict(width=0.5, color="rgba(0,0,0,0.4)"))
    if color_col:
        marker["color"] = df[color_col]
        marker["colorscale"] = "Viridis"
        marker["showscale"] = True

    fig = go.Figure(
        data=[
            go.Scatter(
                x=df[x_col],
                y=df[y_col],
                mode="markers",
                marker=marker,
            )
        ]
    )
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), xaxis_title=x_col, yaxis_title=y_col)
    return fig

