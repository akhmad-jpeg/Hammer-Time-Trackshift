"""Synthetic ERS energy simulator (crude v2 — 2026-regulation shaped).

F1 does not broadcast battery state-of-charge, so the deck's 'Energy_Remaining'
metric cannot be read from any feed -- it has to be *synthesised*.  This module
is that synthesiser, deliberately crude but self-consistent and shaped to the
FIA 2026 power-unit regulations:

  * per-lap regeneration is estimated from the speed trace actually stored in
    `telemetry` (kinetic energy lost between consecutive samples) -- braking
    pedal data is useless at the ~6 samples/lap the importer stores, but speed
    drops survive the sampling.  The trace is capped by the regulation's
    per-lap electrical harvest limit (C5.2.10: 8.5 MJ/lap for the 2026 PU;
    2 MJ/lap for the 2014-2025 PU) and by the store's headroom -- a full
    battery cannot accept regen, so surplus is wasted to the friction brakes;
  * per-lap deployment is a *management choice* (mode = Push / Balanced /
    Lift & Coast) but it is bounded by physics and the regs, never a flat
    demand that drains a fantasy 20 MJ battery:
      - the Energy Store holds a maximum of 4 MJ of *usable* energy at any
        one time (unchanged from the previous era);
      - deployment comes from the small store, so a mode can only sustain
        spending more than the lap harvests for a few laps before the store
        hits its floor.  From the floor the car lives off what it recovers,
        and because part of each lap must then be spent re-charging
        (super-clipping / lift-off windows), a depleted car re-deploys only
        ~85% of that lap's flow -- the store slowly rebuilds and the battery
        "yo-yos" in a low band instead of sitting dead at 0%, which is the
        2026 behaviour described in race reporting;
      - a management reserve (10% of usable capacity) protects the cell:
        the battery is never projected to literal zero -- real teams manage
        to a target window, not to empty.
  * modes therefore differ in *recovery discipline* and in how the driver
    moves SOC against pace:
      - push:      late braking, little lift-off -> harvests ~15% less of the
                   lap's braking energy, asks the era's full deployment
                   ceiling every lap -> drains the store to a low band within
                   a lap or two and then lives energy-limited off what it
                   recovers (the deck's "paradox": fast now, limited later);
      - balanced:  spends roughly what each lap recovers, steering SOC gently
                   back toward ~55% -> the battery floats in a soft 30-80%
                   band and never sits dead (the default trace);
      - liftcoast: early lift-off opens a longer regen window (+20% harvest)
                   and the driver banks the battery toward ~90% as insurance.
  * the battery starts the race FULL -- a car leaves the grid with the usable
    Energy Store fully charged -- and the opening laps burn the surplus down
    into the working band (a full store cannot accept any regeneration, so
    the driver spends the excess rather than bank it): ~100% at lights-out,
    draining over the first few laps, then settling into the soft 30-80%
    band for the rest of the stint;
  * within the band, SOC is tied to *pace*, not pinned by it: laps faster
    than the driver's average spend stored energy (the line can dip below the
    ~30% guide during an attack) and slower / management laps bank it (the
    line can climb above ~80%).  The 30-80% guides are where steady laps
    mostly cycle -- a soft preference, not a hard limit.

Everything is written to `race_state`
(energy_start_mj / energy_deployed_mj / energy_harvested_mj / energy_end_mj),
the columns scaffolded for exactly this.  These rows are a scenario overlay for
the strategy / what-if layer -- NOT ground truth, and the lap-time ML model
must not be trained on them as if it were.

Regulation sources (authoritative journalism citing the FIA 2026 Technical
Regulations, which the FIA publishes as PDFs this tool cannot parse inline):
  * MGU-K 350 kW deploy AND recover (up ~3x from the 120 kW of 2014-2025).
    May-2026 refinement: 350 kW kept in acceleration/overtaking zones, limited
    to 250 kW elsewhere; manual Boost capped at +150 kW over current.
  * ES usable capacity 4 MJ ("permitted to store a maximum of 4MJ of usable
    energy at any one time").
  * Max per-lap recharge 8.5 MJ/lap baseline (TR C5.2.10), circuit-adjusted
    5-9 MJ/lap in qualifying; up to 9 MJ/lap reported for energy-rich venues.
    Deployment: multiple bursts (each up to ~4 MJ at 350 kW) per lap, as many
    as the battery can deliver -- no separate fixed per-lap deployment quota
    in 2026 (unlike 2014-2025, which capped deploy at 4 MJ/lap and recharge at
    2 MJ/lap; those live in the legacy spec below).
  * Recharge sources: braking, part-throttle, lift-and-coast (full 350 kW,
    disables active aero), and super-clipping (on-throttle recharging).

Usage:
    python scripts/energy_simulator.py                 # latest 2026 race, balanced
    python scripts/energy_simulator.py --session 78 --mode push
    python scripts/energy_simulator.py --dry-run --mode liftcoast
"""

import argparse
import sys
import re
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import get_db_connection

# ---------------------------------------------------------------------------
# Track-name resolution for per-track energy pace profiles.
#
# The sessions table and the training artifact (ml_models/energy_pace.json)
# disagree on a handful of track names, so an exact-string lookup silently
# misses measured profiles and falls back to the flat global constant:
#   * case only -- 'Circuit de Barcelona-Catalunya' vs 'Circuit De
#     Barcelona-Catalunya' (same for Monza's di/Di, Imola's e/E, Mugello's
#     del/Del);
#   * short/full pairs -- the DB stores some seasons under the short name
#     ('Monaco', 'Miami Gardens') and others under the full name ('Circuit de
#     Monaco', 'Miami International Autodrome').  The training artifact holds
#     BOTH spellings as separate entries with different measured s/MJ (same
#     circuit, different telemetry samples), so without an alias the same
#     circuit silently uses two different values depending on which season
#     you look at.
#
# Resolution: NFC-normalise + casefold + collapse whitespace on both sides
# (fixes all four case-only tracks), then alias the short/full pairs to a
# single canonical entry.  Canonical entry per pair = the larger measured
# sample (laps count in the artifact): Monaco 464 > 225, Miami Gardens
# 297 > 190.
# ---------------------------------------------------------------------------
TRACK_NAME_ALIASES = {
    'circuit de monaco': 'monaco',               # -> 'Monaco' (464 laps, 0.1543)
    'miami international autodrome': 'miami gardens',  # -> 'Miami Gardens' (297 laps)
}


def normalize_track_name(name):
    """Canonical form for track-name lookups: NFC, casefold, collapse spaces."""
    s = unicodedata.normalize('NFC', str(name or '').strip())
    return ' '.join(s.lower().split())


def canonical_track_name(name):
    """Normalised track name with short/full-name aliases resolved.

    'Circuit de Monaco' -> 'monaco', 'Miami International Autodrome' ->
    'miami gardens'.  Use this (not normalize_track_name) when matching a
    DB track name against a keyed profile/table that may hold either
    spelling.
    """
    return TRACK_NAME_ALIASES.get(normalize_track_name(name), normalize_track_name(name))


def resolve_track_profile(track_name, per_track):
    """The measured profile entry for a DB track name, or None.

    `per_track` is the 'per_track' dict from ml_models/energy_pace.json.
    Exact-name lookup misses the case-variant and short/full-name tracks
    (see TRACK_NAME_ALIASES), silently falling back to the flat constant
    -- which skews race-time projections by ~30-45% on those circuits.
    """
    if not per_track:
        return None
    n = canonical_track_name(track_name)
    for key, entry in per_track.items():
        if canonical_track_name(key) == n:
            return entry
    return None

# ---------------------------------------------------------------------------
# Power-unit specs, per regulation era.  A session's year picks its spec, so a
# 2021 race is simulated with the 2014-2025 PU numbers and a 2026 race with
# the current ones -- never an anachronistic mix.
# ---------------------------------------------------------------------------
PU_SPECS = {
    # 2026 "New Generation" (current regulation, incl. the May-2026
    # refinement): single 350 kW MGU-K (no MGU-H), 4 MJ usable ES, harvest
    # capped at 8.5 MJ/lap (C5.2.10), deployment battery/boost-limited.
    # ~768 kg minimum car mass (down from 800 kg).
    "newgen_2026": {
        "label": "2026 New Generation PU",
        "mgu_k_kw": 350.0,
        "capacity_mj": 4.0,
        "harvest_limit_mj": 8.5,
        "deploy_ceiling_mj": 8.5,
        "car_mass_kg": 768.0,
    },
    # 2014-2025 hybrid era (the cars the 2020/21/25 sessions actually ran):
    # 120 kW MGU-K (+MGU-H), 4 MJ usable ES, hard 4 MJ/lap deployment cap and
    # 2 MJ/lap recovery cap.
    "legacy_2014_2025": {
        "label": "2014-2025 hybrid PU (pre-reboot)",
        "mgu_k_kw": 120.0,
        "capacity_mj": 4.0,
        "harvest_limit_mj": 2.0,
        "deploy_ceiling_mj": 4.0,
        "car_mass_kg": 795.0,
    },
}

DEFAULT_SPEC = "newgen_2026"

# Module-level mirrors of the DEFAULT (2026) spec, kept for callers that work
# on the current-regulation default (dashboard %, clamps, feasibility).
BATTERY_CAPACITY_MJ = PU_SPECS[DEFAULT_SPEC]["capacity_mj"]
HARVEST_LIMIT_MJ = PU_SPECS[DEFAULT_SPEC]["harvest_limit_mj"]

# Management reserve: the battery is never projected below 10% of usable
# capacity.  This is a team-policy placeholder (real cells are protected from
# deep discharge and teams manage SOC to a window, not to empty), NOT an FIA
# figure.
RESERVE_FRACTION = 0.10
BATTERY_MIN_MJ = round(BATTERY_CAPACITY_MJ * RESERVE_FRACTION, 3)  # 0.4 MJ

# ---------------------------------------------------------------------------
# The energy model the dashboard's SOC chart is built to reflect:
#
#   * the usable Energy Store holds 4 MJ  ->  100% SOC = 4.0 MJ, 1% = 0.04 MJ;
#   * a car leaves the grid with that store FULL: the simulation starts at
#     100% and the opening laps burn the surplus down into the working band,
#     because a full store cannot accept any regeneration;
#   * deploy and recovery power are both capped at 120 kW, i.e.
#     DEPLOY_RATE_MJ_S = RECOVER_RATE_MJ_S = 0.12 MJ/s;
#   * per-lap deployment is era-capped: the 2014-2025 PU had a hard 4 MJ/lap
#     deployment quota, while the 2026 PU has no fixed quota (deployment is
#     store-limited bursts), so its higher spec ceiling applies;
#   * SOC is managed into a *soft* 30-80% working band (SOC_WINDOW_MIN/MAX,
#     the dashed guides on the chart): steady laps mostly cycle inside it,
#     but the line is not clamped to it -- laps faster than the driver's
#     average spend stored energy (the line can dip below ~30%) and slower /
#     management laps bank it (it can climb above ~80%).
#
# These are the pre-2026 hybrid-PU figures the imported seasons actually ran
# (see PU_SPECS), with the 2026 deployment ceiling following the current
# regulations; the per-lap caps below keep the synthetic trace regulation-
# shaped.
# ---------------------------------------------------------------------------
RECOVER_FLOW_CAP_MJ = 4.0   # modelled per-lap recovery flow cap (MJ)
DEPLOY_RATE_MJ_S = 0.12     # 120 kW max deploy power  ->  0.12 MJ/s
RECOVER_RATE_MJ_S = 0.12    # 120 kW max recovery power ->  0.12 MJ/s

# Preferred SOC band (fraction of usable capacity) -- where steady laps
# mostly cycle.  Soft, not a hard limit: the reconstructed line moves freely
# inside it and may leak out (attack dips below, banking climbs above).
SOC_WINDOW_MIN = 0.30
SOC_WINDOW_MAX = 0.80

# A race starts with the usable Energy Store FULL (100%): cars leave the grid
# fully charged and burn the surplus down into the working band over the
# first laps, since a full store cannot accept any regeneration until
# deployment opens headroom.
DEFAULT_START_SOC_MJ = round(BATTERY_CAPACITY_MJ, 3)  # 4.0 MJ = 100%

# SOC UNCERTAINTY BAND — the synthetic state of charge is an ESTIMATE, not
# telemetry (F1 does not broadcast the battery), so every SOC value the
# system reports carries an explicit ± band instead of fake point precision.
#
#   * floor ±2% — reconstruction error from ~6 telemetry samples/lap and the
#     downsampled speed-drop regen estimate, even at the last trusted anchor;
#   * +0.5% per projected lap of drift — each modelled lap compounds the
#     estimate's error (pace deviations, regen estimate noise);
#   * cap ±8% — beyond ~12 untrusted laps the estimate is honest about being
#     almost useless; growing past that adds no information.
#
# One shared function (battery_uncertainty_band) is THE definition — the
# dashboard chart envelope, the live call and the policy engine all read the
# same constants, so a band shown anywhere is reproducible everywhere.
SOC_BAND_FLOOR_PCT = 2.0     # ±% at the anchor lap (best case)
SOC_BAND_PER_LAP_PCT = 0.5   # ±% added per lap since the trusted anchor
SOC_BAND_CAP_PCT = 8.0       # ±% ceiling


def battery_uncertainty_band(laps_since_anchor: float,
                             capacity_mj: float = BATTERY_CAPACITY_MJ) -> dict:
    """The ± uncertainty band (in % and MJ) for a synthesized SOC estimate.

    `laps_since_anchor` counts laps since the SOC value was anchored by a
    simulator write (or race start).  Deterministic; the single source of
    truth for every SOC band shown in the product.
    """
    band_pct = min(SOC_BAND_CAP_PCT,
                   SOC_BAND_FLOOR_PCT
                   + SOC_BAND_PER_LAP_PCT * max(0.0, float(laps_since_anchor)))
    return {
        "band_pct": round(band_pct, 2),
        "band_mj": round(band_pct / 100.0 * capacity_mj, 4),
        "floor_pct": SOC_BAND_FLOOR_PCT,
        "per_lap_pct": SOC_BAND_PER_LAP_PCT,
        "cap_pct": SOC_BAND_CAP_PCT,
    }


# Gentle SOC steering for the 'hold' modes: each lap the driver nudges the
# battery this fraction of the way back toward the mode's SOC target.  Kept
# deliberately soft -- a strong controller would pin SOC to the target and
# the chart would flat-line, which is exactly what the pace coupling below
# is meant to prevent.
SOC_REGAIN_FRACTION = 0.15

# Opening burn-down: a full (or over-band) store cannot accept regeneration --
# every braking event's surplus is wasted to the friction brakes until
# deployment opens headroom -- so while SOC sits ABOVE the working band the
# driver burns the surplus down rather than banking it.  Each lap drains this
# fraction of the excess above the band ceiling (a full 4.0 MJ store falls to
# ~80% in 2-3 laps), which is why a race trace reads: ~100% off the grid,
# draining through the first few laps, then settling into the 30-80% band.
# Lift & coast's target (90%) sits above the band, so its ceiling is its own
# target -- it burns down to ~90% and banks there.
OPENING_BURN_DOWN_FRACTION = 0.35

# Pace coupling: SOC is tied to *pace*.  Each lap the driver spends down the
# store by PACE_NET_MJ_PER_DEV MJ per unit of lap-time deviation from the
# lap's expected pace (positive deviation = faster lap), and banks the same
# amount when the lap is slower than expected.  A ~1% faster lap moves the
# store ~0.45 MJ (~11%), so a sustained attack (several laps a percent or
# more faster than the driver's recent average) dips the line below the 30%
# guide while a sustained save banks it above 80%.  On a race whose pace is
# very steady, the line simply stays inside the band -- the leaks appear when
# the data actually shows a push or a save.
PACE_NET_MJ_PER_DEV = 45.0

# Sanity bound on a single lap's net bank or drain: a full store cannot
# accept more than this, and no single lap should swing the SOC wildly.
NET_MOVE_CAP_MJ = 1.0

# Intra-lap soft-band damping: inside a lap the reconstructed SOC line moves
# freely through the 30-80% guides and only a fraction of any flow beyond the
# guides passes -- the line can leak out of the band, but stays mostly
# between the guides on steady laps.
SOC_BAND_LEAK_FRACTION = 0.30

# Intra-lap wiggle damping: the per-lap deploy/harvest totals (several MJ on
# an energy-rich lap) vastly exceed the 4 MJ store, so redistributing them at
# face value swings the reconstructed line across almost the whole store
# every lap.  The path is compressed toward the straight start->end line by
# this factor so steady laps show modest in-band movement around their
# anchors (deployment dips, harvest humps) while a lap with a genuine net
# move still sweeps in that direction.
INTRA_LAP_WIGGLE_COMPRESS = 0.45

# When a lap is energy-limited (the mode wanted more than the store could
# carry), part of the lap is spent re-charging (super-clipping / lift-off
# windows) rather than deploying, so only this fraction of the available flow
# is actually re-deployed that lap.  The store therefore slowly rebuilds and
# the battery cycles in a low band instead of pinning dead at the floor.
FLOOR_REDEPLOY_FRACTION = 0.85

# Crude pace benefit of deployment, used by the advisor's what-if to convert
# each mode's energy trace into an estimated remaining-race time: ~0.35 s per
# MJ deployed (order-of-magnitude for the 2026 PU; tune).  A lap that was
# energy-limited deploys at reduced effectiveness -- energy that arrives
# around re-charge windows, not at the optimum power points -- so its credit
# is scaled by LIMITED_LAP_PACE_EFFECTIVENESS.
DEPLOY_PACE_S_PER_MJ = 0.35
LIMITED_LAP_PACE_EFFECTIVENESS = 0.70

# Driver modes.  regen_style scales the lap's trace-estimated braking energy
# (push brakes late / never lifts -> energy shed to the friction brakes;
# lift & coast lifts early -> a longer recovery window).  Deployment:
#   "ceiling" -> ask for the era's full deployment ceiling every lap;
#   "hold"    -> per-lap net SOC control: spend roughly the lap's own
#                harvest, shifted by the pace coupling (faster than the
#                driver's average drains the store, slower banks it) plus a
#                gentle nudge (SOC_REGAIN_FRACTION) back toward soc_target --
#                the ECU-style management that keeps steady laps cycling
#                inside the soft 30-80% band without pinning SOC flat.
MODES = {
    "push":      {"regen_style": 0.85, "deploy": "ceiling"},
    "balanced":  {"regen_style": 1.00, "deploy": "hold", "soc_target": 0.55},
    "liftcoast": {"regen_style": 1.20, "deploy": "hold", "soc_target": 0.90},
}

# The importer stores only ~6 telemetry rows/lap, so the sampled speed trace
# under-detects braking: raw kinetic-loss sums come out at ~1-3.6 MJ/lap where
# a real Spa lap recovers ~5-7 MJ.  REGEN_SAMPLING_SCALE compensates for that
# under-sampling (crude); traces still vary lap-to-lap, which is the point.
REGEN_SAMPLING_SCALE = 2.2

# Used only when a lap's trace cannot detect ANY speed drop (missing samples).
REGEN_FALLBACK_MJ = 5.0


def spec_for_year(year: int) -> str:
    """PU spec key for a session's year (2026+ -> newgen, else legacy)."""
    return DEFAULT_SPEC if (year or 0) >= 2026 else "legacy_2014_2025"


def _spec(spec_key: str) -> dict:
    if spec_key not in PU_SPECS:
        raise ValueError(f"unknown PU spec '{spec_key}' — use one of {sorted(PU_SPECS)}")
    return PU_SPECS[spec_key]


def _speed_drop_regen(samples: list[dict], spec_key: str = DEFAULT_SPEC) -> float:
    """Regeneration (MJ) from kinetic energy lost on detected speed drops.

    Each consecutive sample pair with v_{i+1} < v_i is treated as a braking
    event; the energy recovered is eta * 0.5 * m * (v_i^2 - v_{i+1}^2).
    The raw sum is scaled by REGEN_SAMPLING_SCALE to compensate for the
    coarse ~6 samples/lap (see note above) and capped by the era's per-lap
    harvest limit (C5.2.10 for 2026; 2 MJ for the legacy PU).
    """
    spec = _spec(spec_key)
    mass = spec["car_mass_kg"]
    harvest_limit = spec["harvest_limit_mj"]
    recovered = 0.0
    if len(samples) >= 2:
        speeds = [float(s.get("speed") or 0.0) / 3.6 for s in samples]  # km/h -> m/s
        for a, b in zip(speeds, speeds[1:]):
            if b < a:
                recovered += 0.5 * mass * (a * a - b * b) * 0.8  # REGEN_EFFICIENCY
    recovered = recovered / 1e6 * REGEN_SAMPLING_SCALE
    if recovered <= 0.0:
        return min(REGEN_FALLBACK_MJ, harvest_limit)
    return min(recovered, harvest_limit)


def project_energy_trace(mode: str, battery_start_mj: float,
                         regen_mj_per_lap: list,
                         spec_key: str = DEFAULT_SPEC,
                         pace_dev_per_lap: list | None = None) -> dict:
    """Project the battery over ``len(regen_mj_per_lap)`` future laps under a mode.

    Pure stepping engine shared by the CLI backfill (``simulate_session_energy``)
    and the strategy advisor's energy what-if, so both stay consistent.

    Rules (see the module docstring and the PU spec for the regulation
    numbers): harvest is capped by the era's per-lap harvest limit and by the
    store's headroom; deployment is capped by the era's per-lap ceiling and by
    what the small store can actually carry (never below the management
    reserve).

    For the 'hold' modes the driver asks for a *net* SOC move each lap rather
    than a fixed energy demand: ``pace_dev_per_lap`` gives each lap's deviation
    from the session-average lap time (positive = faster than average), so
    faster laps spend stored energy and slower laps bank it, plus a gentle
    SOC_REGAIN_FRACTION pull back toward the mode's target.  When the ask
    exceeds what the store can carry the lap is ``limited`` -- it deploys only
    FLOOR_REDEPLOY_FRACTION of what is available, so the store rebuilds and
    the battery cycles in a low band (2026 "yo-yo") rather than pinning at
    zero.

    Returns ``{"laps": [...], "summary": {...}}`` -- laps in order with
    start/deployed/harvested/end MJ per lap; summary carries totals, the
    number of limited laps, the minimum and final battery (MJ and %).
    """
    spec = _spec(spec_key)
    mode = mode.lower()
    if mode not in MODES:
        raise ValueError(f"unknown mode '{mode}' — use one of {sorted(MODES)}")
    profile = MODES[mode]
    capacity = spec["capacity_mj"]
    # Per-lap deployment ceiling comes from the spec itself: the 2014-2025 PU
    # carried a hard 4 MJ/lap quota, while the 2026 PU has no fixed quota
    # (deployment is store-limited bursts), so its higher ceiling applies.
    ceiling = spec["deploy_ceiling_mj"]
    harvest_limit = min(spec["harvest_limit_mj"], RECOVER_FLOW_CAP_MJ)
    reserve = capacity * RESERVE_FRACTION
    regen_style = profile["regen_style"]
    hold = profile["deploy"] == "hold"
    target = capacity * profile.get("soc_target", 0.55)
    # Above this level the driver cannot usefully bank (the store is too full
    # to accept the lap's regen) so surplus is burned down.  Modes whose own
    # target sits above the band (lift & coast -> ~90%) bank up to that
    # target instead.
    band_high_mj = capacity * max(SOC_WINDOW_MAX, profile.get("soc_target", 0.0))

    # Per-lap deviation from the session-average lap time (positive = faster
    # than average).  Optional: the CLI backfill passes real deviations so
    # SOC tracks pace; callers that omit it (strategy what-if) get 0.0.
    n = len(regen_mj_per_lap)
    if pace_dev_per_lap is None:
        devs = [0.0] * n
    else:
        devs = [float(pace_dev_per_lap[i] if i < len(pace_dev_per_lap) else 0.0)
                for i in range(n)]

    laps = []
    battery = float(battery_start_mj)
    deployed_total = harvested_total = 0.0
    limited_laps = 0
    min_battery = battery

    for i, raw_regen in enumerate(regen_mj_per_lap):
        start_mj = battery
        # 1) harvest: mode recovery discipline, regulation cap, headroom.
        h = min(max(0.0, raw_regen) * regen_style, harvest_limit)

        # 2) deployment demand for the lap.
        if not hold:  # push: ask for the era's full per-lap ceiling.
            request = ceiling
        else:
            # The driver targets a *net* SOC move for the lap: pace-coupled
            # (faster-than-average laps spend stored energy -> the store
            # drains; slower laps bank it) plus a gentle pull back toward the
            # mode's SOC target.  The move is bounded so a single lap never
            # swings the whole store.
            dev = devs[i]
            net = -PACE_NET_MJ_PER_DEV * dev + SOC_REGAIN_FRACTION * (target - battery)
            # Opening burn-down: while the store is above the band ceiling the
            # surplus is spent (a full battery cannot bank -- there is nowhere
            # for the regen to go), so the SOC descends from ~100% into the
            # band over the first few laps instead of sitting pinned at full
            # until a fast lap happens to drain it.  A battery sitting exactly
            # at capacity ignores any banking ask and burns the surplus.
            if battery > band_high_mj:
                burn = -OPENING_BURN_DOWN_FRACTION * (battery - band_high_mj)
                net += burn
                if battery >= capacity - 1e-9 and net > 0.0:
                    net = burn
            net = max(-NET_MOVE_CAP_MJ, min(NET_MOVE_CAP_MJ, net))
            target_end = max(reserve, min(capacity, battery + net))
            # Deploy enough this lap that the store lands on target_end: less
            # than the lap harvests when banking (net > 0), more when the
            # driver is spending stored energy (net < 0).
            request = battery + h - target_end

        # 3) era ceiling + what the store can actually carry this lap (never
        #    below the management reserve).
        request = max(0.0, min(request, ceiling))
        available = max(0.0, battery - reserve) + h
        limited = request > available + 1e-9
        if limited:
            # Depleted: part of the lap is spent re-charging, so only a
            # fraction of the available flow is re-deployed -- the store
            # rebuilds instead of pinning dead at the floor.
            deploy = available * FLOOR_REDEPLOY_FRACTION
        else:
            deploy = max(0.0, request)

        # 4) regen that can actually be stored: the store bottoms out at
        #    (battery - deploy) mid-lap, so headroom is capacity minus the
        #    mid-lap low point.  deploy never exceeds battery + h (available
        #    is bounded by it), and harvested is capped by this headroom, so
        #    end = battery - deploy + harvested stays >= 0 by construction.
        headroom = capacity - (battery - deploy)
        harvested = min(h, headroom)

        end_mj = min(capacity, battery - deploy + harvested)
        if limited:
            limited_laps += 1

        laps.append({
            "start_mj": round(start_mj, 4),
            "deployed_mj": round(deploy, 4),
            "harvested_mj": round(harvested, 4),
            "end_mj": round(end_mj, 4),
            "limited": limited,
            "soc_band_pct": battery_uncertainty_band(i)["band_pct"],
        })
        battery = end_mj
        deployed_total += deploy
        harvested_total += harvested
        min_battery = min(min_battery, battery)

    return {
        "laps": laps,
        "summary": {
            "deployed_total": round(deployed_total, 4),
            "harvested_total": round(harvested_total, 4),
            "limited_laps": limited_laps,
            "min_battery_mj": round(max(min_battery, 0.0), 4),
            "final_battery_mj": round(battery, 4),
            "final_battery_pct": round(battery / capacity * 100.0, 1),
            "final_soc_band_pct": battery_uncertainty_band(n - 1)["band_pct"],
        },
    }


def intra_lap_battery_curve(samples: list[dict], start_mj: float, deployed_mj: float,
                           harvested_mj: float, end_mj: float,
                           capacity_mj: float = BATTERY_CAPACITY_MJ,
                           reserve_mj: float = BATTERY_MIN_MJ,
                           lap_time_s: float | None = None) -> list[dict]:
    """Reconstruct the battery movement *inside* one lap at sample resolution.

    The per-lap model stores one start- and end-of-lap SOC per lap: for a
    Balanced trace those anchors wander gently with pace (see
    project_energy_trace).  Real battery telemetry also moves *within* the
    lap: SOC falls while energy is deployed (full-throttle stretches) and
    rises while it is harvested (braking / lift-and-coast).  This function
    rebuilds that shape so the dashboard can show a single tracking line that
    drops under deployment and climbs under harvesting while staying anchored
    to the stored per-lap values.

    The lap's stored policy totals (start / deployed / harvested / end) are
    redistributed over the sample-to-sample segments: a segment is typed from
    the speed trace (speed drop => harvest; otherwise throttle => deploy),
    and each segment gets a share proportional to the throttle reached
    (deploy) or the kinetic energy lost (harvest).  SOC steps through the
    segments.

    Two dashboard facts shape the path:

    * power is capped at 120 kW both ways, so when ``lap_time_s`` is given a
      segment can move the store by at most 0.12 MJ/s x its share of the lap
      (deploying OR recovering -- the energy cannot teleport);
    * SOC is managed into a *soft* 30-80% working window (SOC_WINDOW_MIN/MAX):
      the line moves freely inside the band and only a damped fraction
      (SOC_BAND_LEAK_FRACTION) of any flow beyond its edges passes, so steady
      laps mostly cycle between the guides while a genuinely banking (lift &
      coast toward ~90%) or draining (push, attack) lap can still leak out
      instead of flat-topping on the guides.  The lap anchors (start / end
      from race_state) are kept exact.

    The lap anchors (fraction 0 and 1) sit exactly on the stored start/end
    SOC so the chart never disagrees with race_state or jumps at lap
    boundaries.

    Returns [{"fraction": float 0..1, "soc_mj": float}, ...].
    """
    n = len(samples)
    if n < 2 or (deployed_mj <= 0.0 and harvested_mj <= 0.0):
        return [{"fraction": 0.0, "soc_mj": float(start_mj)},
                {"fraction": 1.0, "soc_mj": float(end_mj)}]

    v = [float(s.get("speed") or 0.0) / 3.6 for s in samples]      # km/h -> m/s
    thr = [float(s.get("throttle") or 0.0) for s in samples]       # 0..1
    nseg = n - 1
    d_w = [0.0] * nseg   # deploy weight per segment (full-throttle stretches)
    h_w = [0.0] * nseg   # harvest weight per segment (deceleration stretches)
    for j in range(nseg):
        if v[j + 1] < v[j] - 0.3:      # speed dropped -> braking/regen segment
            h_w[j] = max(0.0, v[j] * v[j] - v[j + 1] * v[j + 1])
        else:                           # accelerating/cruising -> deploy segment
            d_w[j] = max(0.0, thr[j + 1])
    if sum(d_w) <= 0.0:                 # no throttle signal: spread deploy evenly
        d_w = [1.0] * nseg
    if sum(h_w) <= 0.0:                 # no decel detected: spread harvest evenly
        h_w = [1.0] * nseg
    sd, sh = sum(d_w), sum(h_w)

    # Segment-level 120 kW (0.12 MJ/s) flow cap: no single segment may move
    # the store by more energy than the deploy/recover rate allows over its
    # share of the lap.  (Barely binds at the importer's ~6 samples/lap -- it
    # just keeps the synthetic path honest about the power limit.)
    seg_rate_mj = 0.0
    if lap_time_s and lap_time_s > 0.0 and nseg > 0:
        seg_rate_mj = max(DEPLOY_RATE_MJ_S, RECOVER_RATE_MJ_S) * lap_time_s / nseg

    # Soft 30-80% band in MJ: movement is free inside it; beyond either edge
    # only SOC_BAND_LEAK_FRACTION of the flow passes, so the line can leak out
    # (attack dips, banking climbs) without flat-topping on the guides.
    w_lo = capacity_mj * SOC_WINDOW_MIN
    w_hi = capacity_mj * SOC_WINDOW_MAX

    def _rise(level, add):
        room = max(0.0, w_hi - level)
        return level + add if add <= room else w_hi + SOC_BAND_LEAK_FRACTION * (add - room)

    def _fall(level, sub):
        room = max(0.0, level - w_lo)
        return level - sub if sub <= room else w_lo - SOC_BAND_LEAK_FRACTION * (sub - room)

    soc = float(start_mj)
    pts = [{"fraction": 0.0, "soc_mj": soc}]
    for j in range(nseg):
        add = harvested_mj * h_w[j] / sh if h_w[j] > 0.0 else 0.0
        sub = deployed_mj * d_w[j] / sd if d_w[j] > 0.0 else 0.0
        if seg_rate_mj > 0.0:
            add = min(add, seg_rate_mj)
            sub = min(sub, seg_rate_mj)
        if add > 0.0:
            soc = _rise(soc, add)
        if sub > 0.0:
            soc = _fall(soc, sub)
        pts.append({"fraction": round((j + 1) / nseg, 6), "soc_mj": soc})

    # Compress the interior oscillation around the straight start->end anchor
    # line (see INTRA_LAP_WIGGLE_COMPRESS) so steady laps show modest in-band
    # movement instead of a full-store sawtooth every lap.  The lap anchors
    # (fraction 0 / 1) keep their stored values exactly, so the chart never
    # jumps at lap boundaries; interior points are NOT pinned to the managed
    # 30-80% window, so genuine policy moves (a draining attack lap, a
    # banking lift-and-coast lap) still show.
    for i, p in enumerate(pts):
        if i == 0 or i == len(pts) - 1:
            p["soc_mj"] = float(start_mj if i == 0 else end_mj)
            continue
        anchor = float(start_mj) + (float(end_mj) - float(start_mj)) * p["fraction"]
        p["soc_mj"] = anchor + (p["soc_mj"] - anchor) * INTRA_LAP_WIGGLE_COMPRESS
    for p in pts:
        p["soc_mj"] = max(reserve_mj, min(capacity_mj, p["soc_mj"]))
    return pts


def simulate_session_energy(session_id: int, mode: str, dry_run: bool = False) -> dict:
    """Simulate every lap of a session under an energy mode; persist to race_state."""
    mode = mode.lower()
    if mode not in MODES:
        raise SystemExit(f"[ERROR] unknown mode '{mode}' — use one of {sorted(MODES)}")
    spec_key = DEFAULT_SPEC
    spec = _spec(spec_key)

    conn = get_db_connection()
    cur = conn.cursor(dictionary=True)

    cur.execute("""
        SELECT s.session_id, s.driver_id, s.track_name, s.date
        FROM sessions s WHERE s.session_id = %s
    """, (session_id,))
    session = cur.fetchone()
    if session is None:
        raise SystemExit(f"[ERROR] Session {session_id} not found.")

    spec_key = spec_for_year(session["date"].year if session["date"] else 0)
    spec = _spec(spec_key)

    cur.execute("""
        SELECT l.lap_id, l.lap_number, l.lap_time_ms
        FROM laps l WHERE l.session_id = %s AND l.lap_time_ms > 0
        ORDER BY l.lap_number
    """, (session_id,))
    laps = cur.fetchall()
    if not laps:
        raise SystemExit(f"[ERROR] Session {session_id} has no timed laps.")

    telem = {}
    if laps:
        ids = [l["lap_id"] for l in laps]
        ph = ",".join(["%s"] * len(ids))
        cur.execute(f"SELECT lap_id, speed FROM telemetry WHERE lap_id IN ({ph}) ORDER BY telemetry_id", ids)
        for row in cur.fetchall():
            telem.setdefault(row["lap_id"], []).append(row)

    # Per-lap regeneration from the speed traces, then the shared projection
    # engine (the strategy advisor uses the same function for 'remaining laps'
    # what-if queries, so both stay consistent).  The battery starts the race
    # FULL -- cars leave the grid with a charged Energy Store -- and the
    # opening laps burn it down into the working band.
    regen_list = [_speed_drop_regen(telem.get(lap["lap_id"], []), spec_key)
                  for lap in laps]
    # Pace coupling: each lap's deviation from its *expected* pace (positive
    # = faster than expected).  Expected pace is the median of the lap's ~+
    # /-5 neighbours -- a driver's 'average' drifts with fuel load and tyre
    # age, so using the whole-session average would wrongly mark whole stints
    # as fast or slow.  Faster-than-expected laps spend stored energy and
    # drag SOC below the soft 30-80% band; slower / management laps bank it
    # back above -- see project_energy_trace.
    lap_times = [float(l["lap_time_ms"]) for l in laps]
    half = 5
    baseline = []
    n_lap = len(lap_times)
    for i, _ in enumerate(lap_times):
        window = sorted(lap_times[max(0, i - half): min(n_lap, i + half + 1)])
        baseline.append(window[len(window) // 2])
    pace_dev = [(base - t) / base for base, t in zip(baseline, lap_times)]
    trace = project_energy_trace(mode, DEFAULT_START_SOC_MJ, regen_list,
                                 spec_key, pace_dev_per_lap=pace_dev)

    rows = []
    summary = []
    for lap, rec in zip(laps, trace["laps"]):
        summary.append((lap["lap_number"], rec["start_mj"], rec["deployed_mj"],
                        rec["harvested_mj"], rec["end_mj"], rec["limited"]))
        rows.append((session_id, session["driver_id"], lap["lap_number"],
                     round(rec["start_mj"], 4), round(rec["deployed_mj"], 4),
                     round(rec["harvested_mj"], 4), round(rec["end_mj"], 4)))

    battery = trace["summary"]["final_battery_mj"]
    total_deployed = trace["summary"]["deployed_total"]
    total_harvested = trace["summary"]["harvested_total"]
    energy_limited_laps = trace["summary"]["limited_laps"]

    pct = lambda v: v / spec["capacity_mj"] * 100.0  # noqa: E731
    print("=" * 78)
    print(f"ENERGY SIMULATION (crude model) — session {session_id}  [{mode}]")
    # Per-lap caps come from the era's spec (2014-2025: hard 4 MJ/lap deploy
    # quota; 2026: store-limited bursts up to its spec ceiling).
    har_cap = min(spec["harvest_limit_mj"], RECOVER_FLOW_CAP_MJ)
    dep_cap = spec["deploy_ceiling_mj"]
    print(f"  {session['track_name']}  {session['date']}  driver #{session['driver_id']}  "
          f"{len(laps)} laps   spec: {spec['label']} (MGU-K {spec['mgu_k_kw']:.0f} kW, "
          f"ES {spec['capacity_mj']:.0f} MJ usable, deploy/recover caps {dep_cap:.1f}/{har_cap:.1f} MJ/lap, "
          f"120 kW = 0.12 MJ/s, starts full (burn-down into the soft 30-80% "
          f"band over the opening laps))")
    print("=" * 78)
    print(f"  {'LAP':>4} {'START':>7} {'DEPLOY':>8} {'HARVEST':>9} {'END':>7} {'BATT%':>6}  load")
    for lap_no, start, dep, har, end, limited in summary:
        print(f"  {lap_no:>4} {start:>7.2f} {dep:>8.3f} {har:>9.3f} {end:>7.2f} "
              f"{pct(end):>6.1f}  {'energy-limited' if limited else ''}")
    print("-" * 78)
    print(f"  total deployed {total_deployed:.2f} MJ | harvested {total_harvested:.2f} MJ")
    print(f"  final battery {battery:.2f} MJ ({pct(battery):.1f}%)  "
          f"| {energy_limited_laps} energy-limited lap(s) (store at its management reserve)")

    if dry_run:
        print("\n[dry run] nothing written to race_state.")
        cur.close(); conn.close()
        return {"rows": len(rows), "energy_limited": energy_limited_laps}

    cur.execute("DELETE FROM race_state WHERE session_id = %s", (session_id,))
    cur.executemany(
        """INSERT INTO race_state
           (session_id, driver_id, lap_number,
            energy_start_mj, energy_deployed_mj, energy_harvested_mj, energy_end_mj)
           VALUES (%s, %s, %s, %s, %s, %s, %s)""",
        rows,
    )
    conn.commit()
    print(f"\n[OK] wrote {len(rows)} synthetic energy rows to race_state "
          f"(session {session_id}, mode '{mode}', spec {spec_key}).")
    cur.close(); conn.close()
    return {"rows": len(rows), "energy_limited": energy_limited_laps}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(
        description="Synthesise regulation-shaped energy traces for a session's laps "
                    "(crude v2; writes race_state unless --dry-run).")
    parser.add_argument("--session", type=int, default=None,
                        help="session_id (default: latest 2026 race session)")
    parser.add_argument("--mode", default="balanced", choices=sorted(MODES),
                        help="deployment strategy: push / balanced / liftcoast")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the trace but do not touch the database")
    args = parser.parse_args()

    if args.session is None:
        conn = get_db_connection(); cur = conn.cursor(dictionary=True)
        cur.execute("""
            SELECT s.session_id FROM sessions s
            WHERE YEAR(s.date) = 2026 AND s.session_type = 'Race'
            ORDER BY s.date DESC, s.session_id DESC LIMIT 1
        """)
        row = cur.fetchone()
        cur.close(); conn.close()
        if row is None:
            raise SystemExit("[ERROR] No 2026 race session found — import one first "
                             "(launcher 02/03).")
        args.session = row["session_id"]
        print(f"[auto] using latest 2026 session id {args.session}")

    simulate_session_energy(args.session, args.mode, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
