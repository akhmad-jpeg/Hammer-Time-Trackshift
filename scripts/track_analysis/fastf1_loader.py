"""FastF1 session and lap telemetry loader.

Wraps the FastF1 API with the project's cache directory and returns raw
DataFrames without destructive modification.

NOTE: The X/Y coordinates from FastF1 are telemetry-derived racing
trajectories, NOT official CAD/survey circuit geometry.  They are accurate
enough for spatial-separation analysis but should not be treated as
survey-grade.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    # pyrefly: ignore [missing-import]
    import fastf1
except ImportError:
    fastf1 = None

# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
_CACHE_DIR = str(Path(__file__).resolve().parent.parent.parent / "f1_cache")


def _ensure_cache() -> None:
    """Enable the FastF1 disk cache in the project's f1_cache/ folder."""
    if fastf1 is None:
        raise ImportError("fastf1 is required — pip install fastf1")
    os.makedirs(_CACHE_DIR, exist_ok=True)
    fastf1.Cache.enable_cache(_CACHE_DIR)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_session(year: int, gp: str, session_type: str = "R"):
    """Load and return a FastF1 session (with telemetry).

    Parameters
    ----------
    year : int
        Season year, e.g. 2026.
    gp : str
        Grand-prix identifier accepted by FastF1 (name or round number).
    session_type : str
        "R" (Race), "Q" (Qualifying), etc.

    Returns
    -------
    fastf1.core.Session
        Fully loaded session object.
    """
    _ensure_cache()
    session = fastf1.get_session(year, gp, session_type)
    session.load(telemetry=True, laps=True, weather=False)
    return session


def load_driver_lap(session, driver_code: str, lap_number: int) -> pd.DataFrame:
    """Return the raw telemetry DataFrame for one driver's single lap.

    The returned DataFrame is a *copy* — the caller can mutate it freely
    without affecting the cached session object.

    Key columns used downstream:
        Distance  – metres along the lap (canonical alignment axis)
        X, Y      – spatial position (telemetry-derived, not survey)
        Speed     – km/h
        Time      – timedelta from the start of the lap

    Parameters
    ----------
    session : fastf1.core.Session
        A loaded session (from `load_session`).
    driver_code : str
        Three-letter driver abbreviation, e.g. "LEC".
    lap_number : int
        1-indexed lap number.

    Returns
    -------
    pd.DataFrame
        Telemetry with at least Distance, X, Y, Speed, Time columns.

    Raises
    ------
    ValueError
        If the driver or lap is not found, or telemetry is empty.
    """
    laps = session.laps.pick_drivers(driver_code)
    if laps.empty:
        raise ValueError(f"No laps found for driver {driver_code}")

    lap = laps.pick_laps(lap_number)
    if lap.empty:
        raise ValueError(f"Lap {lap_number} not found for {driver_code}")

    tel = lap.get_telemetry()
    if tel.empty:
        raise ValueError(f"Empty telemetry for {driver_code} lap {lap_number}")

    required = {"Distance", "X", "Y", "Speed", "Time"}
    missing = required - set(tel.columns)
    if missing:
        raise ValueError(f"Telemetry missing columns: {missing}")

    return tel.copy()
