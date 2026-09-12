"""Track-map and spatial-separation visualizations.

Generates two publication-quality figures:
1. Circuit map with driver trajectories, closing phase, closest approach,
   and overtake point annotated.
2. Spatial-separation profile along the circuit distance.

Uses actual FastF1 X/Y coordinates with equal-aspect ratio so the circuit
shape is geometrically correct.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server/script use
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from .telemetry_alignment import AlignedTelemetry
from .models import OvertakeResult


# ---------------------------------------------------------------------------
# Colour palette (F1-inspired dark theme)
# ---------------------------------------------------------------------------
_BG       = "#1a1a2e"
_TRACK    = "#333355"
_LEC_CLR  = "#e10600"   # Ferrari red
_RUS_CLR  = "#27f4d2"   # Mercedes teal
_CLOSE_CLR = "#ffd700"  # gold for closing phase
_OV_CLR   = "#ff4ecb"   # magenta for overtake
_STAR_CLR = "#ffffff"    # white star


def plot_track_map(
    aligned: AlignedTelemetry,
    spatial_dist: np.ndarray,
    result: OvertakeResult,
    *,
    driver_a_code: str = "LEC",
    driver_b_code: str = "RUS",
    title: str = "2026 AUSTRALIAN GP — OVERTAKE ANALYSIS",
    save_path: Optional[str] = None,
    dpi: int = 200,
) -> plt.Figure:
    """Render the circuit map with overtake annotations.

    Parameters
    ----------
    aligned : AlignedTelemetry
        Interpolated telemetry for both drivers.
    spatial_dist : np.ndarray
        Euclidean X/Y separation at each grid point.
    result : OvertakeResult
        Detection output.
    save_path : str, optional
        If given, save the figure to this path.
    """
    fig, ax = plt.subplots(figsize=(14, 10), facecolor=_BG)
    ax.set_facecolor(_BG)

    # --- Track outline (all of driver A's trajectory as a faint line) ---
    ax.plot(aligned.a_x, aligned.a_y, color=_TRACK, linewidth=12,
            solid_capstyle="round", zorder=1)

    # --- Closing phase highlight ---
    cp = result.closing_phase
    mask_close = (aligned.distance_m >= cp.start_distance_m) & \
                 (aligned.distance_m <= cp.end_distance_m)
    if mask_close.any():
        ax.plot(aligned.a_x[mask_close], aligned.a_y[mask_close],
                color=_CLOSE_CLR, linewidth=14, alpha=0.35,
                solid_capstyle="round", zorder=2, label="Closing phase")

    # --- Driver trajectories ---
    ax.plot(aligned.a_x, aligned.a_y, color=_LEC_CLR, linewidth=2.5,
            alpha=0.9, zorder=3, label=driver_a_code)
    ax.plot(aligned.b_x, aligned.b_y, color=_RUS_CLR, linewidth=2.5,
            alpha=0.9, zorder=3, label=driver_b_code)

    # --- Starting positions (first grid point) ---
    ax.scatter(aligned.a_x[0], aligned.a_y[0], color=_LEC_CLR, s=80,
               edgecolors="white", linewidth=1.5, zorder=5)
    ax.scatter(aligned.b_x[0], aligned.b_y[0], color=_RUS_CLR, s=80,
               edgecolors="white", linewidth=1.5, zorder=5)

    # --- Closest approach ---
    ca = result.closest_approach
    ax.scatter(ca.x, ca.y, color=_CLOSE_CLR, s=120, marker="D",
               edgecolors="white", linewidth=1.5, zorder=6,
               label=f"Closest: {ca.spatial_separation_m:.1f} m")

    # --- Overtake point (star) ---
    ax.scatter(result.overtake_x, result.overtake_y, color=_OV_CLR,
               s=300, marker="*", edgecolors="white", linewidth=1,
               zorder=7, label="OVERTAKE")

    # --- Annotation box at overtake ---
    ann_text = (
        f"OVERTAKE\n"
        f"Track dist: {result.overtake_distance_m:.0f} m\n"
        f"Separation: {result.overtake_spatial_separation_m:.1f} m\n"
        f"{driver_a_code} speed: {result.attacker_speed_kmh:.0f} km/h\n"
        f"{driver_b_code} speed: {result.defender_speed_kmh:.0f} km/h"
    )
    ax.annotate(
        ann_text,
        xy=(result.overtake_x, result.overtake_y),
        xytext=(40, 40), textcoords="offset points",
        fontsize=8, fontfamily="monospace",
        color="white", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor=_OV_CLR, alpha=0.85,
                  edgecolor="white"),
        arrowprops=dict(arrowstyle="->", color="white", lw=1.5),
        zorder=8,
    )

    # --- Closing phase annotation ---
    close_text = (
        f"Closing phase\n"
        f"{cp.start_distance_m:.0f} m → {cp.end_distance_m:.0f} m"
    )
    mid_close = mask_close.nonzero()[0]
    if len(mid_close) > 0:
        mid_idx = mid_close[len(mid_close) // 2]
        ax.annotate(
            close_text,
            xy=(aligned.a_x[mid_idx], aligned.a_y[mid_idx]),
            xytext=(-60, -50), textcoords="offset points",
            fontsize=7, fontfamily="monospace",
            color="white",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=_CLOSE_CLR,
                      alpha=0.7, edgecolor="white"),
            arrowprops=dict(arrowstyle="->", color=_CLOSE_CLR, lw=1),
            zorder=8,
        )

    # --- Layout ---
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=16, fontweight="bold", color="white",
                 fontfamily="monospace", pad=20)
    ax.legend(loc="lower right", fontsize=9, facecolor="#2a2a4a",
              edgecolor="white", labelcolor="white", framealpha=0.9)
    ax.axis("off")
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, facecolor=_BG, bbox_inches="tight")
        print(f"[SAVED] {save_path}")

    return fig


def plot_spatial_separation(
    aligned: AlignedTelemetry,
    spatial_dist: np.ndarray,
    result: OvertakeResult,
    *,
    driver_a_code: str = "LEC",
    driver_b_code: str = "RUS",
    title: str = "SPATIAL SEPARATION — LEC vs RUS",
    save_path: Optional[str] = None,
    dpi: int = 200,
) -> plt.Figure:
    """Plot spatial separation vs circuit distance."""
    fig, ax = plt.subplots(figsize=(14, 5), facecolor=_BG)
    ax.set_facecolor(_BG)

    # Main separation curve
    ax.fill_between(aligned.distance_m, 0, spatial_dist,
                    color=_RUS_CLR, alpha=0.15)
    ax.plot(aligned.distance_m, spatial_dist, color=_RUS_CLR,
            linewidth=2, alpha=0.9)

    # Closing phase highlight
    cp = result.closing_phase
    mask_close = (aligned.distance_m >= cp.start_distance_m) & \
                 (aligned.distance_m <= cp.end_distance_m)
    if mask_close.any():
        ax.fill_between(aligned.distance_m[mask_close], 0,
                        spatial_dist[mask_close],
                        color=_CLOSE_CLR, alpha=0.25, label="Closing phase")

    # Closest approach
    ca = result.closest_approach
    ax.axvline(ca.distance_m, color=_CLOSE_CLR, linestyle="--",
               alpha=0.7, linewidth=1)
    ax.scatter([ca.distance_m], [ca.spatial_separation_m],
               color=_CLOSE_CLR, s=100, marker="D", edgecolors="white",
               zorder=5, label=f"Closest: {ca.spatial_separation_m:.1f} m")

    # Overtake point
    ov_d = result.overtake_distance_m
    ov_sep = result.overtake_spatial_separation_m
    ax.axvline(ov_d, color=_OV_CLR, linestyle="--", alpha=0.7, linewidth=1)
    ax.scatter([ov_d], [ov_sep], color=_OV_CLR, s=200, marker="*",
               edgecolors="white", zorder=5,
               label=f"Overtake: {ov_d:.0f} m")

    # Styling
    ax.set_xlabel("Circuit Distance (m)", fontsize=11, color="white",
                  fontfamily="monospace")
    ax.set_ylabel("Spatial Separation (m)", fontsize=11, color="white",
                  fontfamily="monospace")
    ax.set_title(title, fontsize=14, fontweight="bold", color="white",
                 fontfamily="monospace", pad=15)
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("white")
    ax.spines["left"].set_color("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", fontsize=9, facecolor="#2a2a4a",
              edgecolor="white", labelcolor="white", framealpha=0.9)
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, facecolor=_BG, bbox_inches="tight")
        print(f"[SAVED] {save_path}")

    return fig
