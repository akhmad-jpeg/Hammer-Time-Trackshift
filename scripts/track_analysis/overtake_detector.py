"""Overtake detection from aligned two-driver telemetry.

Algorithm overview
------------------
1. **Spatial separation** (Euclidean X/Y distance) — measures straight-line
   separation between the two cars at each point along the circuit.
   NOTE: this is NOT distance along the circuit; it is the crow-flies gap.

2. **Closing phase** — the contiguous region where spatial separation is
   decreasing significantly (gradient < threshold).

3. **Closest approach** — argmin of spatial separation.

4. **Overtake point** — determined by the *relative timing* delta, NOT by
   minimum spatial distance.  We compute:

       delta_t(d) = time_defender(d) - time_attacker(d)

   where ``defender`` is the car initially ahead.  A sign change in delta_t
   indicates the attacker has moved ahead on the circuit at that distance.

Complexity: O(N) — all operations are vectorised NumPy.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .telemetry_alignment import AlignedTelemetry
from .models import OvertakeResult, ClosestApproach, ClosingPhase


# ---------------------------------------------------------------------------
# Core calculations — all O(N), fully vectorised
# ---------------------------------------------------------------------------

def calculate_spatial_distance(aligned: AlignedTelemetry) -> np.ndarray:
    """Euclidean X/Y separation at every grid point.

    d = sqrt((x_a - x_b)^2 + (y_a - y_b)^2)

    This measures straight-line spatial separation between the two vehicles
    in the FastF1 X/Y coordinate system.  It is NOT the same as distance
    along the circuit.
    """
    return np.sqrt(
        (aligned.a_x - aligned.b_x) ** 2 +
        (aligned.a_y - aligned.b_y) ** 2
    )


def find_closest_approach(
    aligned: AlignedTelemetry,
    spatial_dist: np.ndarray,
) -> ClosestApproach:
    """Find the point of minimum spatial separation."""
    idx = int(np.argmin(spatial_dist))
    return ClosestApproach(
        distance_m=float(aligned.distance_m[idx]),
        spatial_separation_m=float(spatial_dist[idx]),
        x=float(aligned.a_x[idx]),
        y=float(aligned.a_y[idx]),
        driver_a_speed_kmh=float(aligned.a_speed[idx]),
        driver_b_speed_kmh=float(aligned.b_speed[idx]),
        driver_a_time_s=float(aligned.a_time_s[idx]),
        driver_b_time_s=float(aligned.b_time_s[idx]),
    )


def find_closing_phase(
    aligned: AlignedTelemetry,
    spatial_dist: np.ndarray,
    *,
    smoothing_window: int = 50,
    gradient_threshold: float = -0.02,
) -> ClosingPhase:
    """Identify the contiguous closing region before the closest approach.

    The closing phase is where the smoothed gradient of the spatial distance
    is consistently negative (the cars are converging).

    Parameters
    ----------
    smoothing_window : int
        Convolution kernel width for smoothing the gradient.
    gradient_threshold : float
        Gradient value below which we consider the cars to be closing.
    """
    closest_idx = int(np.argmin(spatial_dist))

    # Smooth the spatial-distance signal to avoid noise spikes
    kernel = np.ones(smoothing_window) / smoothing_window
    smoothed = np.convolve(spatial_dist, kernel, mode="same")

    # Numerical gradient (O(N))
    grad = np.gradient(smoothed, aligned.distance_m)

    # Walk backwards from closest approach to find the start of closing
    start_idx = closest_idx
    for i in range(closest_idx, -1, -1):
        if grad[i] < gradient_threshold:
            start_idx = i
        else:
            break

    return ClosingPhase(
        start_distance_m=float(aligned.distance_m[start_idx]),
        end_distance_m=float(aligned.distance_m[closest_idx]),
    )


def detect_overtake(
    aligned: AlignedTelemetry,
    spatial_dist: np.ndarray,
    *,
    attacker: str,
    defender: str,
    race: str = "",
    session: str = "Race",
    lap: int = 0,
    attacker_is_a: bool = True,
) -> OvertakeResult:
    """Full overtake detection using relative timing.

    Parameters
    ----------
    attacker_is_a : bool
        If True, driver A (the first telemetry passed to alignment) is the
        attacker (the car starting behind).  The relative-time delta is
        then computed as:  delta_t = defender_time - attacker_time.
        When delta_t flips from positive to negative, the attacker has
        moved ahead.
    """
    if attacker_is_a:
        att_time = aligned.a_time_s
        def_time = aligned.b_time_s
        att_speed = aligned.a_speed
        def_speed = aligned.b_speed
        att_x, att_y = aligned.a_x, aligned.a_y
    else:
        att_time = aligned.b_time_s
        def_time = aligned.a_time_s
        att_speed = aligned.b_speed
        def_speed = aligned.a_speed
        att_x, att_y = aligned.b_x, aligned.b_y

    # Relative timing: positive means defender passed this distance first
    # (attacker is behind).  A sign change → overtake.
    delta_t = def_time - att_time

    # Find zero-crossing (sign change)
    sign_changes = np.where(np.diff(np.sign(delta_t)))[0]

    if len(sign_changes) > 0:
        # Use the first sign change as the overtake point
        ov_idx = int(sign_changes[0])
        overtake_completed = True
    else:
        # No sign change — check if attacker ended up ahead anyway
        # (could happen if they were already ahead the entire lap)
        if delta_t[-1] < 0:
            # Attacker was ahead the whole time — use closest approach
            ov_idx = int(np.argmin(spatial_dist))
            overtake_completed = True
        else:
            # No overtake completed — report the closest approach
            ov_idx = int(np.argmin(spatial_dist))
            overtake_completed = False

    closest = find_closest_approach(aligned, spatial_dist)
    closing = find_closing_phase(aligned, spatial_dist)

    return OvertakeResult(
        race=race,
        session=session,
        lap=lap,
        attacker=attacker,
        defender=defender,
        overtake_completed=overtake_completed,
        overtake_distance_m=float(aligned.distance_m[ov_idx]),
        closest_approach=closest,
        closing_phase=closing,
        overtake_x=float(att_x[ov_idx]),
        overtake_y=float(att_y[ov_idx]),
        overtake_spatial_separation_m=float(spatial_dist[ov_idx]),
        attacker_speed_kmh=float(att_speed[ov_idx]),
        defender_speed_kmh=float(def_speed[ov_idx]),
        attacker_time_s=float(att_time[ov_idx]),
        defender_time_s=float(def_time[ov_idx]),
    )
