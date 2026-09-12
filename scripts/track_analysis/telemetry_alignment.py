"""Distance-based telemetry alignment for two-driver comparison.

When comparing two drivers on the same lap, their telemetry rows do NOT
correspond 1:1 — the cars sample at different instants and distances.
This module creates a common distance grid and interpolates both drivers
onto it so that every index represents the same point along the circuit.

Coordinate semantics
--------------------
    Track Distance → canonical alignment axis (metres from lap start)
    X / Y          → spatial geometry for Euclidean separation
    Time           → event timing (seconds from lap start)
    Speed          → instantaneous speed (km/h)

Complexity: O(N) — a single pass through each driver's telemetry via
NumPy's interp (piecewise-linear, already-sorted input).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class AlignedTelemetry:
    """Two drivers' telemetry interpolated onto a shared distance grid.

    All arrays have the same length (= len(distance_m)).
    """

    distance_m: np.ndarray   # common distance axis

    a_x: np.ndarray          # driver A  X positions
    a_y: np.ndarray          # driver A  Y positions
    a_time_s: np.ndarray     # driver A  time (seconds from lap start)
    a_speed: np.ndarray      # driver A  speed (km/h)

    b_x: np.ndarray          # driver B  X positions
    b_y: np.ndarray          # driver B  Y positions
    b_time_s: np.ndarray     # driver B  time (seconds from lap start)
    b_speed: np.ndarray      # driver B  speed (km/h)


def _timedelta_to_seconds(series: pd.Series) -> np.ndarray:
    """Convert a pandas Timedelta series to float64 seconds."""
    return series.dt.total_seconds().to_numpy(dtype=np.float64)


def align_by_distance(
    tel_a: pd.DataFrame,
    tel_b: pd.DataFrame,
    *,
    grid_step_m: float = 1.0,
) -> AlignedTelemetry:
    """Align two drivers' telemetry onto a common distance grid.

    Parameters
    ----------
    tel_a, tel_b : pd.DataFrame
        Raw telemetry frames (from ``fastf1_loader.load_driver_lap``).
        Must contain Distance, X, Y, Speed, Time columns.
    grid_step_m : float
        Spacing of the common distance grid in metres (default 1 m).

    Returns
    -------
    AlignedTelemetry
        Both drivers interpolated onto the shared grid.

    Notes
    -----
    The common grid covers only the *overlapping* distance range so that
    no extrapolation is needed.  Typical Albert Park lap length ≈ 5.2 km.
    """
    # --- raw arrays (already sorted by distance within a single lap) ---
    dist_a = tel_a["Distance"].to_numpy(dtype=np.float64)
    dist_b = tel_b["Distance"].to_numpy(dtype=np.float64)

    x_a = tel_a["X"].to_numpy(dtype=np.float64)
    y_a = tel_a["Y"].to_numpy(dtype=np.float64)
    x_b = tel_b["X"].to_numpy(dtype=np.float64)
    y_b = tel_b["Y"].to_numpy(dtype=np.float64)

    speed_a = tel_a["Speed"].to_numpy(dtype=np.float64)
    speed_b = tel_b["Speed"].to_numpy(dtype=np.float64)

    time_a = _timedelta_to_seconds(tel_a["Time"])
    time_b = _timedelta_to_seconds(tel_b["Time"])

    # --- overlapping distance range (no extrapolation) ---
    d_min = max(dist_a.min(), dist_b.min())
    d_max = min(dist_a.max(), dist_b.max())
    if d_max <= d_min:
        raise ValueError(
            f"No overlapping distance range: A=[{dist_a.min():.0f}, "
            f"{dist_a.max():.0f}], B=[{dist_b.min():.0f}, {dist_b.max():.0f}]"
        )

    grid = np.arange(d_min, d_max, grid_step_m)

    # --- vectorised interpolation: O(N) for sorted input ---
    return AlignedTelemetry(
        distance_m=grid,
        a_x=np.interp(grid, dist_a, x_a),
        a_y=np.interp(grid, dist_a, y_a),
        a_time_s=np.interp(grid, dist_a, time_a),
        a_speed=np.interp(grid, dist_a, speed_a),
        b_x=np.interp(grid, dist_b, x_b),
        b_y=np.interp(grid, dist_b, y_b),
        b_time_s=np.interp(grid, dist_b, time_b),
        b_speed=np.interp(grid, dist_b, speed_b),
    )
