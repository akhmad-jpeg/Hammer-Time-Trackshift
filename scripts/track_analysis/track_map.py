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
    """Render the circuit map with overtake annotations and off-track sidebar."""
    fig = plt.figure(figsize=(16, 7.5), facecolor=_BG)
    gs = fig.add_gridspec(1, 2, width_ratios=[3.0, 1.15], wspace=0.08)

    ax = fig.add_subplot(gs[0], facecolor=_BG)
    ax_sidebar = fig.add_subplot(gs[1], facecolor=_BG)

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
    ax.scatter([aligned.a_x[0]], [aligned.a_y[0]], color=_STAR_CLR, s=70,
               marker="|", linewidths=3, zorder=6)

    # --- Closest approach ---
    ca = result.closest_approach
    ax.scatter(ca.x, ca.y, color=_CLOSE_CLR, s=120, marker="D",
               edgecolors="white", linewidth=1.5, zorder=6,
               label=f"Closest: {ca.spatial_separation_m:.1f} m")

    # --- Overtake point (star) ---
    ax.scatter(result.overtake_x, result.overtake_y, color=_OV_CLR,
               s=300, marker="*", edgecolors="white", linewidth=1,
               zorder=7, label="OVERTAKE")

    # --- Compact annotation tag at overtake point ---
    ax.annotate(
        f"★ Turn 3 Overtake ({result.overtake_distance_m:.0f}m)",
        xy=(result.overtake_x, result.overtake_y),
        xytext=(30, 25), textcoords="offset points",
        fontsize=8, fontfamily="monospace",
        color="white", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor=_OV_CLR, alpha=0.9,
                  edgecolor="white", linewidth=1),
        arrowprops=dict(arrowstyle="->", color="white", lw=1.2),
        zorder=8,
    )

    # --- Layout for track axis (clean and unobstructed) ---
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=13, fontweight="bold", color="white",
                 fontfamily="monospace", pad=12)
    ax.axis("off")

    # --- SIDEBAR (Legend and telemetry metrics shifted completely off track) ---
    ax_sidebar.axis("off")
    ax_sidebar.text(0.05, 0.95, "SESSION METADATA", transform=ax_sidebar.transAxes,
                    fontsize=9.5, fontfamily="monospace", color="#8e9bb0", fontweight="bold")
    meta_text = (
        f"Race: {result.race}\n"
        f"Session: Lap {result.lap} (Albert Park)\n"
        f"Attacker: {driver_a_code} (Ferrari)\n"
        f"Defender: {driver_b_code} (Mercedes)\n"
        f"Status: Overtake Confirmed ✓"
    )
    ax_sidebar.text(0.05, 0.76, meta_text, transform=ax_sidebar.transAxes,
                    fontsize=8.5, fontfamily="monospace", color="white",
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="#10121d", edgecolor="#282c40", linewidth=1))

    ax_sidebar.text(0.05, 0.69, "TELEMETRY METRICS", transform=ax_sidebar.transAxes,
                    fontsize=9.5, fontfamily="monospace", color="#8e9bb0", fontweight="bold")
    metrics_text = (
        f"Overtake Point:    {result.overtake_distance_m:5.0f} m\n"
        f"Pass Separation:   {result.overtake_spatial_separation_m:5.2f} m\n"
        f"{driver_a_code} Speed:         {result.attacker_speed_kmh:5.1f} km/h\n"
        f"{driver_b_code} Speed:         {result.defender_speed_kmh:5.1f} km/h\n"
        f"Closest Approach:  {ca.distance_m:5.0f} m ({ca.spatial_separation_m:.2f}m)\n"
        f"Closing Phase:     {cp.start_distance_m:.0f}m -> {cp.end_distance_m:.0f}m"
    )
    ax_sidebar.text(0.05, 0.44, metrics_text, transform=ax_sidebar.transAxes,
                    fontsize=8.5, fontfamily="monospace", color="white",
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="#10121d", edgecolor="#282c40", linewidth=1))

    ax_sidebar.text(0.05, 0.38, "TRACK MAP LEGEND", transform=ax_sidebar.transAxes,
                    fontsize=9.5, fontfamily="monospace", color="#8e9bb0", fontweight="bold")
    legend_elements = [
        plt.Line2D([0], [0], color=_LEC_CLR, linewidth=2.5, label=f"{driver_a_code} (Chaser -> P1)"),
        plt.Line2D([0], [0], color=_RUS_CLR, linewidth=2.5, label=f"{driver_b_code} (Leader -> P2)"),
        plt.Line2D([0], [0], color=_OV_CLR, marker="*", markersize=11, markeredgecolor="white",
                   linestyle="None", label="Overtake Point (Turn 3)"),
        plt.Line2D([0], [0], color=_CLOSE_CLR, marker="D", markersize=7, markeredgecolor="white",
                   linestyle="None", label=f"Closest ({ca.spatial_separation_m:.2f} m)"),
        plt.Line2D([0], [0], color=_CLOSE_CLR, linewidth=6, alpha=0.6, label="Closing Phase Zone"),
        plt.Line2D([0], [0], color=_STAR_CLR, marker="|", markersize=9, markeredgewidth=2,
                   linestyle="None", label="Start / Finish Line"),
    ]
    ax_sidebar.legend(handles=legend_elements, loc="lower left", bbox_to_anchor=(0.04, 0.02),
                      fontsize=8, facecolor="#10121d", edgecolor="#282c40",
                      labelcolor="white", framealpha=0.95)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.04, wspace=0.08)

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
    """Plot spatial separation vs circuit distance (wide layout matching track map)."""
    fig, ax = plt.subplots(figsize=(16, 4.8), facecolor=_BG)
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
