"""Trackshift overtake demo — 2026 Australian GP, LEC vs RUS.

Usage (from repository root):
    python -m scripts.track_analysis.overtake_demo

Outputs:
    outputs/australian_gp_2026_lec_rus_overtake.png   — track map
    outputs/australian_gp_2026_lec_rus_distance.png   — separation profile
    outputs/australian_gp_2026_lec_rus_result.json    — structured result
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure scripts/ is on sys.path so relative imports work when invoked
# via ``python -m scripts.track_analysis.overtake_demo``
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from track_analysis.fastf1_loader import load_session, load_driver_lap
from track_analysis.telemetry_alignment import align_by_distance
from track_analysis.overtake_detector import (
    calculate_spatial_distance,
    find_closest_approach,
    detect_overtake,
)
from track_analysis.track_map import plot_track_map, plot_spatial_separation

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
YEAR = 2026
GP = "Australia"
SESSION_TYPE = "R"
DRIVER_A = "LEC"   # attacker — was P2 at start of lap 3, ended P1
DRIVER_B = "RUS"   # defender — was P1 at start of lap 3, ended P2

# Lap 3 is the documented on-track overtake where LEC retakes the lead
# from RUS after the lap-2 swap.  Position data confirms:
#   Lap 2: RUS P1, LEC P2
#   Lap 3: LEC P1, RUS P2
LAP = 3

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "outputs"


def main() -> None:
    print("=" * 60)
    print("TRACKSHIFT OVERTAKE ANALYSIS")
    print("=" * 60)
    print(f"\nLoading {YEAR} {GP} {SESSION_TYPE} session...", flush=True)

    session = load_session(YEAR, GP, SESSION_TYPE)
    race_name = f"{YEAR} Australian Grand Prix"
    print(f"Session: {session}\n")

    # --- Load telemetry -------------------------------------------------
    print(f"Loading telemetry: {DRIVER_A} lap {LAP}...", flush=True)
    tel_a = load_driver_lap(session, DRIVER_A, LAP)
    print(f"  -> {len(tel_a)} samples, distance "
          f"{tel_a['Distance'].min():.0f}-{tel_a['Distance'].max():.0f} m")

    print(f"Loading telemetry: {DRIVER_B} lap {LAP}...", flush=True)
    tel_b = load_driver_lap(session, DRIVER_B, LAP)
    print(f"  -> {len(tel_b)} samples, distance "
          f"{tel_b['Distance'].min():.0f}-{tel_b['Distance'].max():.0f} m")

    # --- Validation -----------------------------------------------------
    for name, tel in [(DRIVER_A, tel_a), (DRIVER_B, tel_b)]:
        assert not tel["X"].isna().all(), f"{name}: X is all NaN"
        assert not tel["Y"].isna().all(), f"{name}: Y is all NaN"
        assert len(tel) > 10, f"{name}: too few telemetry rows ({len(tel)})"

    # --- Alignment ------------------------------------------------------
    print("\nAligning telemetry by track distance (1 m grid)...", flush=True)
    aligned = align_by_distance(tel_a, tel_b, grid_step_m=1.0)
    print(f"  -> {len(aligned.distance_m)} grid points, "
          f"{aligned.distance_m[0]:.0f}–{aligned.distance_m[-1]:.0f} m")

    # --- Spatial distance -----------------------------------------------
    spatial_dist = calculate_spatial_distance(aligned)
    assert (spatial_dist >= 0).all(), "Spatial distance must be non-negative"
    assert np.all(np.isfinite(spatial_dist)), "Non-finite spatial distances"
    print(f"  Min separation: {spatial_dist.min():.2f} m")
    print(f"  Max separation: {spatial_dist.max():.2f} m")

    # --- Overtake detection ---------------------------------------------
    print("\nDetecting overtake...", flush=True)
    result = detect_overtake(
        aligned, spatial_dist,
        attacker=DRIVER_A,
        defender=DRIVER_B,
        race=race_name,
        session="Race",
        lap=LAP,
        attacker_is_a=True,
    )

    # Validate the overtake point lies within the lap distance range
    assert (aligned.distance_m[0] <= result.overtake_distance_m
            <= aligned.distance_m[-1]), \
        "Overtake point outside lap distance range"

    # --- Console summary ------------------------------------------------
    ov_label = (f"{result.attacker} -> past {result.defender}"
                if result.overtake_completed
                else f"NO OVERTAKE (closest approach only)")

    print()
    print("=" * 60)
    print("TRACKSHIFT OVERTAKE ANALYSIS")
    print("=" * 60)
    print()
    print(f"  Race:       {result.race}")
    print(f"  Lap:        {result.lap}")
    print()
    print(f"  Attacker:   {result.attacker}")
    print(f"  Defender:   {result.defender}")
    print()
    print(f"  Overtake:   {ov_label}")
    print()
    print(f"  Track distance:       {result.overtake_distance_m:.0f} m")
    print(f"  Spatial separation:   {result.overtake_spatial_separation_m:.2f} m")
    print()
    print(f"  {DRIVER_A} speed:            {result.attacker_speed_kmh:.1f} km/h")
    print(f"  {DRIVER_B} speed:            {result.defender_speed_kmh:.1f} km/h")
    print()
    ca = result.closest_approach
    print(f"  Closest approach:     {ca.distance_m:.0f} m "
          f"({ca.spatial_separation_m:.2f} m separation)")
    cp = result.closing_phase
    print(f"  Closing phase:        {cp.start_distance_m:.0f} m -> "
          f"{cp.end_distance_m:.0f} m")
    print()
    print("=" * 60)

    # --- Output ---------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    map_path = str(OUTPUT_DIR / "australian_gp_2026_lec_rus_overtake.png")
    sep_path = str(OUTPUT_DIR / "australian_gp_2026_lec_rus_distance.png")
    json_path = str(OUTPUT_DIR / "australian_gp_2026_lec_rus_result.json")

    print(f"\nGenerating track map -> {map_path}", flush=True)
    plot_track_map(
        aligned, spatial_dist, result,
        driver_a_code=DRIVER_A, driver_b_code=DRIVER_B,
        save_path=map_path,
    )

    print(f"Generating separation profile -> {sep_path}", flush=True)
    plot_spatial_separation(
        aligned, spatial_dist, result,
        driver_a_code=DRIVER_A, driver_b_code=DRIVER_B,
        save_path=sep_path,
    )

    print(f"Writing result JSON -> {json_path}", flush=True)
    with open(json_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    # Also copy images to the dashboard static folder for the UI
    dashboard_static = (Path(__file__).resolve().parent.parent
                        / "dashboard" / "static")
    if dashboard_static.exists():
        import shutil
        for src in [map_path, sep_path]:
            dst = dashboard_static / Path(src).name
            shutil.copy2(src, dst)
            print(f"[COPIED] {dst}")

    print("\n[OK] Done.")


if __name__ == "__main__":
    main()
