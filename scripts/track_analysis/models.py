"""Dataclasses for structured overtake-analysis results.

These are plain data containers — no I/O, no side effects.  Every field is
documented so a reviewer can understand the output schema without reading the
detection code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class ClosestApproach:
    """Point where the two drivers are spatially nearest (Euclidean X/Y)."""

    distance_m: float          # track distance along the lap (metres)
    spatial_separation_m: float  # Euclidean X/Y separation (metres)
    x: float                   # FastF1 X coordinate
    y: float                   # FastF1 Y coordinate
    driver_a_speed_kmh: float
    driver_b_speed_kmh: float
    driver_a_time_s: float     # seconds from lap start
    driver_b_time_s: float


@dataclass
class ClosingPhase:
    """Track-distance window where the trailing car is actively closing."""

    start_distance_m: float
    end_distance_m: float


@dataclass
class OvertakeResult:
    """Full analysis output for a single-lap two-driver comparison.

    `attacker` is the driver who was behind at the start of the lap and
    completed (or attempted) the pass.  `defender` was initially ahead.

    The `overtake_distance_m` marks where the relative timing flips —
    i.e. where the attacker effectively moves ahead on the circuit.
    This is NOT necessarily the point of minimum spatial separation.
    """

    race: str
    session: str
    lap: int
    attacker: str              # driver code, e.g. "LEC"
    defender: str              # driver code, e.g. "RUS"
    overtake_completed: bool   # did the attacker finish ahead?
    overtake_distance_m: float
    closest_approach: ClosestApproach
    closing_phase: ClosingPhase
    # Telemetry at the overtake point
    overtake_x: float
    overtake_y: float
    overtake_spatial_separation_m: float
    attacker_speed_kmh: float
    defender_speed_kmh: float
    attacker_time_s: float
    defender_time_s: float

    def to_dict(self) -> dict:
        """Serialise to a JSON-friendly dict (no numpy/timedelta types)."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
