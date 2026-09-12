"""Single-lap overtake simulation animation with spatial separation profile.

Renders an animation of two cars (Chaser LEC vs Leader RUS) racing against
each other for a single lap (Lap 3, 2026 Australian GP). The chaser hunts
down and overtakes the leader. Upon completion of the lap, the spatial
separation profile is generated and displayed.

Usage:
    python -m scripts.track_analysis.animate_overtake
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure scripts/ is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from track_analysis.fastf1_loader import load_session, load_driver_lap
from track_analysis.telemetry_alignment import align_by_distance
from track_analysis.overtake_detector import (
    calculate_spatial_distance,
    detect_overtake,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
YEAR = 2026
GP = "Australia"
SESSION_TYPE = "R"
DRIVER_A = "LEC"   # Attacker / Chaser (Ferrari Red)
DRIVER_B = "RUS"   # Defender / Leader (Mercedes Teal)
LAP = 3

_BG        = "#0d0f18"
_PANEL_BG  = "#141724"
_TRACK     = "#282c40"
_TRACK_ACC = "#3d4360"
_LEC_CLR   = "#e10600"   # Ferrari red
_RUS_CLR   = "#27f4d2"   # Mercedes teal
_CLOSE_CLR = "#ffd700"   # Gold for closing phase
_OV_CLR    = "#ff4ecb"   # Magenta for overtake
_WHITE     = "#ffffff"

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "outputs"
DASHBOARD_STATIC = Path(__file__).resolve().parent.parent / "dashboard" / "static"


def prepare_simulation_data(grid_step_m: float = 5.0):
    """Load and align single-lap telemetry for the animation."""
    session = load_session(YEAR, GP, SESSION_TYPE)
    tel_a = load_driver_lap(session, DRIVER_A, LAP)
    tel_b = load_driver_lap(session, DRIVER_B, LAP)

    aligned = align_by_distance(tel_a, tel_b, grid_step_m=grid_step_m)
    spatial_dist = calculate_spatial_distance(aligned)

    result = detect_overtake(
        aligned, spatial_dist,
        attacker=DRIVER_A,
        defender=DRIVER_B,
        race=f"{YEAR} Australian Grand Prix",
        session="Race",
        lap=LAP,
        attacker_is_a=True,
    )
    return aligned, spatial_dist, result


def create_animation(aligned, spatial_dist, result, output_gif_path: str, n_frames: int = 120):
    """Generate the single-lap race animation and post-completion separation profile."""
    total_points = len(aligned.distance_m)
    
    # We allocate frames:
    # 0 to 82% of frames: Racing lap
    # 82% to 100% of frames: Completed lap state with full separation profile revealed
    race_frames = int(n_frames * 0.82)

    fig = plt.figure(figsize=(16, 11), facecolor=_BG)
    gs_main = fig.add_gridspec(2, 1, height_ratios=[2.1, 1.0], hspace=0.24)
    gs_top = gs_main[0].subgridspec(1, 2, width_ratios=[3.0, 1.15], wspace=0.08)

    ax_track = fig.add_subplot(gs_top[0], facecolor=_PANEL_BG)
    ax_info = fig.add_subplot(gs_top[1], facecolor=_PANEL_BG)
    ax_profile = fig.add_subplot(gs_main[1], facecolor=_PANEL_BG)

    # ------------------------------------------------------------------
    # 1. Track layout setup (Unobstructed View)
    # ------------------------------------------------------------------
    # Draw full circuit background
    ax_track.plot(aligned.a_x, aligned.a_y, color=_TRACK, linewidth=8,
                  solid_capstyle="round", zorder=1)
    ax_track.plot(aligned.a_x, aligned.a_y, color=_TRACK_ACC, linewidth=2,
                  linestyle="--", alpha=0.6, zorder=2)

    # Start / Finish line
    ax_track.scatter([aligned.a_x[0]], [aligned.a_y[0]], color=_WHITE, s=60,
                     marker="|", linewidths=3, zorder=3)
    ax_track.text(aligned.a_x[0] + 180, aligned.a_y[0] + 120, "S/F",
                  color="#aaa", fontsize=8, fontfamily="monospace", zorder=4)

    # Closing phase highlight on track
    cp = result.closing_phase
    mask_close = (aligned.distance_m >= cp.start_distance_m) & \
                 (aligned.distance_m <= cp.end_distance_m)
    if mask_close.any():
        ax_track.plot(aligned.a_x[mask_close], aligned.a_y[mask_close],
                      color=_CLOSE_CLR, linewidth=10, alpha=0.35, zorder=2)

    # Overtake point star
    ax_track.scatter([result.overtake_x], [result.overtake_y], color=_OV_CLR,
                     s=180, marker="*", edgecolors=_WHITE, linewidth=1, zorder=5)
    ax_track.text(result.overtake_x + 180, result.overtake_y - 120, "★ Turn 3 Overtake",
                  color=_OV_CLR, fontsize=8.5, fontfamily="monospace", fontweight="bold", zorder=6)

    # Dynamic elements for cars
    trail_len = 18
    trail_a, = ax_track.plot([], [], color=_LEC_CLR, linewidth=2.5, alpha=0.7, zorder=6)
    trail_b, = ax_track.plot([], [], color=_RUS_CLR, linewidth=2.5, alpha=0.7, zorder=6)

    car_a, = ax_track.plot([], [], marker="o", markersize=10, color=_LEC_CLR,
                           markeredgecolor=_WHITE, markeredgewidth=1.5, zorder=8)
    car_b, = ax_track.plot([], [], marker="o", markersize=10, color=_RUS_CLR,
                           markeredgecolor=_WHITE, markeredgewidth=1.5, zorder=8)

    ax_track.set_aspect("equal", adjustable="box")
    ax_track.set_title("2026 AUSTRALIAN GP — SINGLE-LAP OVERTAKE SIMULATION (LAP 3)",
                       fontsize=12, fontweight="bold", color=_WHITE, fontfamily="monospace", pad=12)
    ax_track.axis("off")

    # ------------------------------------------------------------------
    # Telemetry & Legend Sidebar (Next to Track — Unobstructed Track View)
    # ------------------------------------------------------------------
    ax_info.axis("off")

    ax_info.text(0.5, 0.96, "LIVE TELEMETRY & HUD", transform=ax_info.transAxes,
                 fontsize=10.5, fontfamily="monospace", color=_WHITE, fontweight="bold", ha="center", va="top")
    ax_info.text(0.5, 0.91, "ALBERT PARK CIRCUIT · 5,231 M", transform=ax_info.transAxes,
                 fontsize=8, fontfamily="monospace", color="#8e9bb0", ha="center", va="top")

    # Dynamic Battle Status Box
    status_box = ax_info.text(
        0.5, 0.83, "INITIALIZING...", transform=ax_info.transAxes,
        fontsize=9, fontfamily="monospace", color=_WHITE, fontweight="bold",
        ha="center", va="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#080a14", alpha=0.95, edgecolor="#27f4d2", linewidth=1.5)
    )

    # Dynamic Telemetry Metrics Box
    hud_box = ax_info.text(
        0.05, 0.71, "", transform=ax_info.transAxes,
        fontsize=8.5, fontfamily="monospace", color=_WHITE, fontweight="bold",
        ha="left", va="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#080a14", alpha=0.9, edgecolor="#2b3145", linewidth=1)
    )

    # Track Map Legend Box (Shifted off track into sidebar)
    ax_info.text(0.05, 0.38, "TRACK MAP LEGEND", transform=ax_info.transAxes,
                 fontsize=8.5, fontfamily="monospace", color="#8e9bb0", fontweight="bold", ha="left", va="top")

    legend_elements = [
        plt.Line2D([0], [0], color=_LEC_CLR, marker="o", markersize=8, markeredgecolor=_WHITE,
                   linestyle="None", label=f"{DRIVER_A} (Chaser -> P1)"),
        plt.Line2D([0], [0], color=_RUS_CLR, marker="o", markersize=8, markeredgecolor=_WHITE,
                   linestyle="None", label=f"{DRIVER_B} (Leader -> P2)"),
        plt.Line2D([0], [0], color=_OV_CLR, marker="*", markersize=11, markeredgecolor=_WHITE,
                   linestyle="None", label=f"Overtake ({result.overtake_distance_m:.0f}m)"),
        plt.Line2D([0], [0], color=_CLOSE_CLR, linewidth=6, alpha=0.6,
                   label="Closing Phase"),
        plt.Line2D([0], [0], color=_WHITE, marker="|", markersize=9, markeredgewidth=2,
                   linestyle="None", label="Start / Finish Line"),
    ]
    ax_info.legend(handles=legend_elements, loc="lower left", bbox_to_anchor=(0.04, 0.02),
                   fontsize=8, facecolor="#080a14", edgecolor="#282d40",
                   labelcolor=_WHITE, framealpha=0.95)

    # ------------------------------------------------------------------
    # 2. Separation profile setup (Underneath the track map)
    # ------------------------------------------------------------------
    ax_profile.set_title("SPATIAL SEPARATION PROFILE (GENERATED UPON LAP COMPLETION)",
                         fontsize=11, fontweight="bold", color="#8888aa", fontfamily="monospace", pad=10)
    ax_profile.set_xlabel("Circuit Distance (m)", fontsize=9, color=_WHITE, fontfamily="monospace")
    ax_profile.set_ylabel("Separation (m)", fontsize=9, color=_WHITE, fontfamily="monospace")
    ax_profile.set_xlim(0, aligned.distance_m[-1])
    ax_profile.set_ylim(0, spatial_dist.max() * 1.15)
    ax_profile.tick_params(colors=_WHITE, labelsize=8)
    for sp in ["bottom", "left"]:
        ax_profile.spines[sp].set_color("#444")
    for sp in ["top", "right"]:
        ax_profile.spines[sp].set_visible(False)
    ax_profile.grid(True, linestyle=":", alpha=0.25, color="#888")

    # Profile placeholder lines
    prof_line, = ax_profile.plot([], [], color=_RUS_CLR, linewidth=2, zorder=3)
    prof_fill = [None]
    cursor_line = ax_profile.axvline(0, color=_WHITE, linestyle="--", alpha=0.5, zorder=4)

    # Completion overlay in the profile panel
    completion_text = ax_profile.text(
        0.5, 0.5, "RACING LAP IN PROGRESS...\nSPATIAL PROFILE GENERATING AT FINISH",
        transform=ax_profile.transAxes, fontsize=11, fontfamily="monospace",
        color="#666688", fontweight="bold", ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#080a12", alpha=0.9, edgecolor="#222")
    )

    def init():
        trail_a.set_data([], [])
        trail_b.set_data([], [])
        car_a.set_data([], [])
        car_b.set_data([], [])
        hud_box.set_text("")
        status_box.set_text("")
        prof_line.set_data([], [])
        cursor_line.set_xdata([0])
        return car_a, car_b, trail_a, trail_b, hud_box, status_box, prof_line

    def update(frame):
        if frame < race_frames:
            idx = int((frame / (race_frames - 1)) * (total_points - 1))
            lap_finished = False
        else:
            idx = total_points - 1
            lap_finished = True

        cur_dist = aligned.distance_m[idx]
        sep = spatial_dist[idx]
        spd_a = aligned.a_speed[idx]
        spd_b = aligned.b_speed[idx]

        # Update cars on track
        car_a.set_data([aligned.a_x[idx]], [aligned.a_y[idx]])
        car_b.set_data([aligned.b_x[idx]], [aligned.b_y[idx]])

        # Update motion trails
        start_t = max(0, idx - trail_len)
        trail_a.set_data(aligned.a_x[start_t:idx+1], aligned.a_y[start_t:idx+1])
        trail_b.set_data(aligned.b_x[start_t:idx+1], aligned.b_y[start_t:idx+1])

        # Determine overtake & battle state
        ov_dist = result.overtake_distance_m
        if cur_dist < ov_dist - 150:
            state = f"STALKING ({DRIVER_B} LEADING)"
            state_color = "#27f4d2"
        elif cur_dist < ov_dist + 50:
            state = f"⚡ OVERTAKE: {DRIVER_A} PASSES {DRIVER_B}!"
            state_color = "#ff4ecb"
        elif cur_dist < cp.end_distance_m:
            state = "CLOSING PHASE BATTLE"
            state_color = "#ffd700"
        else:
            state = f"{DRIVER_A} LEADING (PULLING AWAY)"
            state_color = "#e10600"

        hud_text = (
            f"DISTANCE:\n  {cur_dist:5.0f} m / {aligned.distance_m[-1]:.0f} m\n\n"
            f"SEPARATION:\n  {sep:5.1f} m\n\n"
            f"SPEEDS:\n  {DRIVER_A}: {spd_a:3.0f} km/h\n  {DRIVER_B}: {spd_b:3.0f} km/h"
        )
        hud_box.set_text(hud_text)
        status_box.set_text(state)
        status_box.get_bbox_patch().set_edgecolor(state_color)

        cursor_line.set_xdata([cur_dist])

        # If lap completed -> Generate full spatial separation profile
        if lap_finished:
            ax_profile.set_title("✓ SPATIAL SEPARATION PROFILE GENERATED (LAP COMPLETED)",
                                 fontsize=11, fontweight="bold", color="#52c41a", fontfamily="monospace", pad=10)
            completion_text.set_visible(False)

            # Draw full profile
            prof_line.set_data(aligned.distance_m, spatial_dist)

            # Only add fill and annotations once
            if prof_fill[0] is None:
                f = ax_profile.fill_between(aligned.distance_m, 0, spatial_dist,
                                            color=_RUS_CLR, alpha=0.18, zorder=2)
                prof_fill[0] = f

                # Highlight closing phase
                if mask_close.any():
                    ax_profile.fill_between(
                        aligned.distance_m[mask_close], 0, spatial_dist[mask_close],
                        color=_CLOSE_CLR, alpha=0.35, zorder=2, label="Closing Phase"
                    )

                # Closest approach marker
                ca = result.closest_approach
                ax_profile.scatter([ca.distance_m], [ca.spatial_separation_m],
                                   color=_CLOSE_CLR, s=90, marker="D", edgecolors=_WHITE,
                                   zorder=5, label=f"Closest ({ca.spatial_separation_m:.2f}m)")

                # Overtake point marker
                ax_profile.scatter([result.overtake_distance_m],
                                   [result.overtake_spatial_separation_m],
                                   color=_OV_CLR, s=160, marker="*", edgecolors=_WHITE,
                                   zorder=6, label=f"Overtake ({result.overtake_distance_m:.0f}m)")

                ax_profile.legend(loc="upper right", fontsize=8, facecolor="#10121d",
                                  edgecolor="#333", labelcolor=_WHITE, framealpha=0.9)
        else:
            completion_text.set_visible(True)
            # Partially trace the profile up to current distance
            prof_line.set_data(aligned.distance_m[:idx+1], spatial_dist[:idx+1])

        return car_a, car_b, trail_a, trail_b, hud_box, status_box, prof_line

    print(f"Rendering {n_frames} frames to {output_gif_path}...", flush=True)
    anim = FuncAnimation(fig, update, init_func=init, frames=n_frames,
                         interval=60, blit=False)

    writer = PillowWriter(fps=15)
    anim.save(output_gif_path, writer=writer, dpi=90)
    plt.close(fig)
    print(f"[SAVED ANIMATION] {output_gif_path}")


def export_simulation_json(aligned, spatial_dist, result, output_json_path: str):
    """Export lightweight simulation data for the web UI interactive canvas."""
    # Subsample to ~400 points for ultra-fast 60fps browser rendering
    step = max(1, len(aligned.distance_m) // 400)
    indices = list(range(0, len(aligned.distance_m), step))
    if indices[-1] != len(aligned.distance_m) - 1:
        indices.append(len(aligned.distance_m) - 1)

    points = []
    for i in indices:
        points.append({
            "d": round(float(aligned.distance_m[i]), 1),
            "ax": round(float(aligned.a_x[i]), 1),
            "ay": round(float(aligned.a_y[i]), 1),
            "bx": round(float(aligned.b_x[i]), 1),
            "by": round(float(aligned.b_y[i]), 1),
            "as": round(float(aligned.a_speed[i]), 1),
            "bs": round(float(aligned.b_speed[i]), 1),
            "at": round(float(aligned.a_time_s[i]), 2),
            "bt": round(float(aligned.b_time_s[i]), 2),
            "sep": round(float(spatial_dist[i]), 2),
        })

    # Full separation curve downsampled for the post-completion chart
    sep_curve = [
        {"d": round(float(aligned.distance_m[i]), 1), "sep": round(float(spatial_dist[i]), 2)}
        for i in indices
    ]

    payload = {
        "race": result.race,
        "lap": result.lap,
        "attacker": result.attacker,
        "defender": result.defender,
        "total_distance_m": float(aligned.distance_m[-1]),
        "overtake_distance_m": float(result.overtake_distance_m),
        "overtake_x": float(result.overtake_x),
        "overtake_y": float(result.overtake_y),
        "overtake_separation_m": float(result.overtake_spatial_separation_m),
        "closest_approach": {
            "distance_m": float(result.closest_approach.distance_m),
            "separation_m": float(result.closest_approach.spatial_separation_m),
            "x": float(result.closest_approach.x),
            "y": float(result.closest_approach.y),
        },
        "closing_phase": {
            "start_distance_m": float(result.closing_phase.start_distance_m),
            "end_distance_m": float(result.closing_phase.end_distance_m),
        },
        "samples": points,
        "separation_curve": sep_curve,
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[SAVED SIMULATION JSON] {output_json_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DASHBOARD_STATIC.mkdir(parents=True, exist_ok=True)

    print("Loading Australian GP 2026 Lap 3 telemetry...")
    aligned, spatial_dist, result = prepare_simulation_data(grid_step_m=5.0)

    # 1. Export Web UI JSON
    json_path = OUTPUT_DIR / "australian_gp_2026_simulation_data.json"
    export_simulation_json(aligned, spatial_dist, result, str(json_path))

    # Also copy to dashboard static
    import shutil
    shutil.copy2(json_path, DASHBOARD_STATIC / json_path.name)

    # 2. Render animation GIF
    gif_path = OUTPUT_DIR / "australian_gp_2026_overtake_simulation.gif"
    create_animation(aligned, spatial_dist, result, str(gif_path), n_frames=100)

    # Copy to static for web serving
    shutil.copy2(gif_path, DASHBOARD_STATIC / gif_path.name)
    print(f"[COPIED TO STATIC] {DASHBOARD_STATIC / gif_path.name}")
    print("[SUCCESS] Animation and simulation data generated.")


if __name__ == "__main__":
    main()
