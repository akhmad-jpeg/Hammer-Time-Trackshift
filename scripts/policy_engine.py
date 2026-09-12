"""Multi-Policy Tactical Decision Engine (IMPLEMENTATION_PLAN.md, P1-1).

The core challenge deliverable: instead of a passive projection plus a
threshold verdict, this module evaluates FIVE competing tactical policies
against the same race state and returns a ranked, constrained
recommendation — the ACTION card:

    {
      "action": "TACTICAL STALK",
      "deploy_lap": 23,
      "energy_pct": ...,
      "expected_gap_s": ...,
      "overtake_probability": ...,
      "energy_cost_mj": ...,
      "expected_finish_delta_s": ...,
      "battery_margin_pct": ...,
      "confidence": ...,
      "reason": "..."
    }

Design (why deterministic policy search, not DP/MC/LLM):

  * The existing machinery is already deterministic and re-derivable —
    `check_energy_benchmark_drift.py` proves the simulator reproduces its
    committed artifacts exactly.  A decision layer must inherit that
    property, not break it.
  * The decision space is small (5 postures x a horizon of laps).  Rolling
    each policy through `overtake_inference.simulate_live_call` costs tens
    of ms each, so the full comparison fits the interactive <200 ms demo
    budget without any search algorithm.
  * The scoring function is explicit and inspectable — every coefficient
    below is a named constant a strategist can challenge.  A DP/MPC layer
    would add machinery without a decision payoff at this horizon; it is
    the extension, not the base.

Policies (each maps to an ERS lever — the same per-sector MJ deltas the
live call and the Energy Sandbox already accept):

  1. GREEDY ATTACK    — full positive lever now; pass as early as possible.
  2. BALANCED HOLD    — no lever (store-neutral); the tyre edge does the work.
  3. TACTICAL STALK   — bank energy first (negative lever), strike on a
                        later lap (positive lever from the deploy lap).
  4. SAVE & DEFEND    — bank hard to the flag; protect the store.
  5. UNDERCUT PREP    — moderate push now (pit-window pace), balanced after.

Scoring (per policy):

    score(pi) = pass_reward(pi)                      # P(pass) x pass value
              - LAMBDA_BATT * soc_deficit(pi)        # over-deployment cost
              - LAMBDA_WEAR * wear_delta(pi)         # push now = tyre cost
              - LAMBDA_RISK * p_fail(pi) * RISK_PENALTY_S
              - LAMBDA_LATENCY * deferred_value(pi)  # waiting is not free

Hard constraint (never a soft penalty): any policy whose projected store
touches the management floor at or before the pass lap is marked
INFEASIBLE — "reserve breach" — and can never win, exactly like the
energy-mode recommender's feasibility rule.  Infeasible policies are
REPORTED, not hidden: seeing WHY a policy loses is the demo's point.

Import-safe (no side effects at import), same convention as
driver_comparison.py / overtake_inference.py.

Run standalone:
    python scripts/policy_engine.py                 # demo state, live models
    python scripts/policy_engine.py --battery 0.31  # low-battery scenario
"""

from __future__ import annotations

import copy
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import overtake_inference as oi  # noqa: E402
from overtake_inference import (  # noqa: E402
    ERS_MAX_MJ_LAP,
    ERS_STORE_MJ,
    ERS_DEFAULT_START_PCT,
    LIVE_ATTACK_MIN_SOC_PCT,
)
from energy_simulator import battery_uncertainty_band  # noqa: E402
import pit_strategy  # noqa: E402

# ---------------------------------------------------------------------------
# Tunable scoring constants — the whole utility function in one block, so a
# strategist (or a judge) can challenge each weight in isolation.
# ---------------------------------------------------------------------------

# Value of a converted pass, in race-time seconds.  Anchored to the
# energy-mode recommender's own scale (a full Monaco closing phase is worth
# ~1.9 s): a track-position swap at Monza-class tracks is worth ~5-8 s of
# race time (pass + the defensive delta it removes).  Conservative mid-point.
PASS_VALUE_S = 6.0

# Battery opportunity cost: s of race time charged per MJ of store the
# policy spends BELOW its post-policy reserve target.  The energy pace
# profile runs ~0.15-0.5 s/MJ on pace; over-spending the store costs
# considerably more later (you defend with what is left), hence the
# conservative 2.0 s/MJ default.
LAMBDA_BATT = 2.0          # s per MJ below reserve target
RESERVE_TARGET_MJ = 1.6    # 40% of the 4 MJ store — the post-pass cushion

# Tyre cost of pushing: s per extra health-% burned vs the Balanced policy.
# Pushing roughly doubles the tyre-wear slope for the laps it runs; the
# cliff penalty makes the tail expensive, which this term prices in.
LAMBDA_WEAR = 0.08         # s per health-% vs baseline
CLIFF_HEALTH = 40.0        # below this the Pirelli cliff dominates (s/lap)
CLIFF_RISK_S = 1.5         # expected s/lap on the cliff per policy, capped

# Failed-attack risk: P(one-lap pass attempts fail) proxy = 1 - cum P.
# An failed attack leaves the chaser exposed in dirty air with a drained
# store — priced at DIRTY_AIR_PENALTY_S per unit of failure probability.
LAMBDA_RISK = 1.0
DIRTY_AIR_PENALTY_S = 4.0

# Waiting is not free: a deferred strike trades certainty for battery, so
# its horizon gains are discounted (a pass 2 laps later at the same P is
# worth slightly less than one now — the race may end, the tyres may cliff).
LAMBDA_LATENCY = 0.10      # s per lap of deferral vs the earliest pass

# Confidence: how much of the scored gap between the top two policies is
# decisive rather than coin-flip.  Maps the margin onto [0, 1].
CONFIDENCE_SPAN_S = 1.5    # margin (s) at which confidence saturates

# Determinism guard for the latency budget.  MEASURED (2026-09-09, this
# machine): warm runs 0.6-1.4 s for the full 5-policy comparison (7
# simulator walks — the stalk runs two phases; variance is machine load);
# first call in a fresh process ~2-3.5 s while the driver pace models load
# (cached afterwards).  The IMPLEMENTATION_PLAN's original "<200 ms" guess
# did not survive measurement; per the audit's own rule the budget is
# restated to what is actually delivered and the payload reports its own
# latency_ms, so the UI can never claim faster than delivered.  Re-measure
# on presentation hardware before quoting a number on stage.
LATENCY_BUDGET_MS = 1500.0

# ---------------------------------------------------------------------------
# Engine cache for the unified call.
#
# evaluate_call runs BOTH engines; a repeated Evaluate press on the same
# race state (and the seat flip at default batteries — the leader engine
# maps a None battery onto ERS_DEFAULT_START_PCT, so (None, 62.5) and
# (62.5, None) are the same computation) would otherwise re-simulate every
# walk.  Each engine's evaluation is therefore memoised against its full
# input tuple and handed out as a deep copy: an identical call reuses the
# leader-engine walk set instead of re-simulating it.  Within one call the
# two engines share no walks by construction (the leader engine models the
# attacking threat via chaser_ers + an explicit leader battery; the chaser
# engine levers the chaser against the coupled posture), so the cache unit
# is the engine, not the walk.  The payload discloses which engines were
# reused (see evaluate_call's engine_cache) — a cached latency is never
# presented as a fresh walk time.  Eviction is bounded FIFO (insertion
# order) rather than LRU on purpose: the touch would need a pop-after-get
# that races under concurrent Flask requests, and FIFO is exactly right
# for the demo pattern (a handful of recent states).  Concurrent requests
# may compute the same engine twice — the cache is a best-effort memo,
# never a correctness dependency.
# ---------------------------------------------------------------------------
CALL_ENGINE_CACHE_MAX = 8        # most recent engine evaluations per process
_CALL_ENGINE_CACHE: dict[tuple, dict] = {}


def _engine_cache_key(engine: str, *, posture=None, **kw) -> tuple:
    """Full input tuple for one engine's evaluation.

    Battery slots are normalised exactly as the engines normalise them
    (None -> ERS_DEFAULT_START_PCT), so a seat flip at default batteries
    is a cache hit rather than a recomputation of an identical state.
    The slider banks are normalised via _shape_vector (the same
    normalisation the engines apply), so two raw vectors that collapse
    to the same walked shape share an entry — while genuinely different
    shapes (a different spread, or none) never collide.
    """

    def _batt(v):
        return ERS_DEFAULT_START_PCT if v is None else float(v)

    shape = _shape_vector(kw.get("chaser_shape"))
    l_shape = _shape_vector(kw.get("leader_shape"))
    t_shape = _shape_vector(kw.get("threat_shape"))
    return (
        engine,
        str(kw["leader_code"]).upper(), str(kw["chaser_code"]).upper(),
        str(kw["track_name"]),
        int(kw["start_lap"]), int(kw["race_length"]),
        round(float(kw["gap_before_s"]), 4),
        str(kw["leader_tyre_compound"]), str(kw["chaser_tyre_compound"]),
        float(kw["leader_tyre_age"]), float(kw["chaser_tyre_age"]),
        kw["year"],
        _batt(kw["leader_batt"]) if engine == "leader" else None,
        _batt(kw["chaser_batt"]),
        float(kw["reserve"]),
        posture if engine == "chaser" else None,
        (tuple(shape) if shape is not None else None),
        (tuple(l_shape) if l_shape is not None else None),
        (tuple(t_shape) if t_shape is not None else None),
        str(kw.get("race_event") or "green").lower(),
        str(kw.get("traffic_level") or "Clear"),
    )


def _engine_cache_get(key: tuple):
    hit = _CALL_ENGINE_CACHE.get(key)
    if hit is None:
        return None
    return copy.deepcopy(hit)


def _engine_cache_put(key: tuple, payload: dict) -> None:
    _CALL_ENGINE_CACHE[key] = copy.deepcopy(payload)
    while len(_CALL_ENGINE_CACHE) > CALL_ENGINE_CACHE_MAX:
        _CALL_ENGINE_CACHE.pop(next(iter(_CALL_ENGINE_CACHE)))


# ---------------------------------------------------------------------------
# Policy table — the decision space, in one inspectable place.
# ---------------------------------------------------------------------------

def _lever(net_mj: float) -> list[float]:
    """Flat ERS lever (same MJ to every sector) from a net MJ/lap ask.

    Mirrors simulate_live_call's chaser_ers shortcut: the lever is a
    per-sector deploy-delta vector; the live call clamps and spend-caps it
    against the store itself.
    """
    spread = max(-8.5, min(8.5, net_mj / 3.0))
    return [spread, spread, spread]


def _shape_vector(shape) -> list[float] | None:
    """Normalise a caller-supplied per-sector shape to [3 floats] or None.

    The Energy Sandbox's slider vocabulary: three MJ/lap deploy deltas,
    one per sector, clamped per-sector to +-8.5 MJ (same clamp
    simulate_live_call applies).  A vector entirely below the 0.01 MJ
    noise floor (the sliders step at 0.05) collapses to None — an inert
    shape IS the historical no-lever walk — but a zero-SUM reallocation
    (+a, -a, 0) stays active: it is fully delivered and store-neutral,
    exactly the sandbox's semantics.
    """
    if shape is None:
        return None
    try:
        v = [float(x) for x in list(shape)]
    except (TypeError, ValueError):
        return None
    if len(v) != 3:
        return None
    v = [max(-8.5, min(8.5, x)) for x in v]
    if max(abs(x) for x in v) < 0.01:
        return None
    return v


def _shape_net(vec: list[float] | None) -> float:
    """Net MJ/lap of a normalised shape (0.0 for the no-shape walk)."""
    return sum(vec) if vec is not None else 0.0


def _scale_shape(vec: list[float] | None, factor: float) -> list[float] | None:
    """Scale a shape's magnitude for a policy phase, preserving its SPREAD.

    A policy's phase lever is net MJ/lap; a caller shape spreads its net
    over the sectors.  The scaled shape keeps the caller's sector
    proportions (the s/MJ per sector still prices each delta) while its
    net matches the phase's ask; net-0 reallocations pass through
    unscaled (they are pure pace shapes).  A zero net on a zero-sum
    shape, or a scaled-to-zero shape, collapses to None.
    """
    if vec is None:
        return None
    net = _shape_net(vec)
    if abs(net) < 1e-9:
        return vec            # reallocation shape: active at every phase
    scaled = [x * factor for x in vec]
    if all(abs(x) < 1e-9 for x in scaled):
        return None
    return scaled


def _phase_vector(shape, net_mj: float) -> list[float] | None:
    """One phase's walked per-sector vector from a shape + a net ask.

    A shape with real net expresses the policy's ask THROUGH it (scaled,
    spread preserved).  A zero-SUM reallocation cannot carry the ask, so
    the ask rides the flat lever ON TOP of the reallocation — the shape's
    reallocation pace and the policy's magnitude are independent inputs.
    No shape at all degrades to the historical flat lever — and a zero ask
    with no shape is the EXPLICIT store-neutral posture ([0, 0, 0]): the
    simulator walks the store and reports its SOC (the battery margin is
    defined) while keeping Balanced pace.  A caller shape passes through
    unchanged at every phase.
    """
    vec = _shape_vector(shape)
    if vec is None:
        if abs(net_mj) < 1e-9:
            return [0.0, 0.0, 0.0]
        return _lever(net_mj)
    net0 = _shape_net(vec)
    if abs(net0) > 1e-9:
        scaled = _scale_shape(vec, net_mj / net0)
        # A real shape scaled to a zero ask lands on the same EXPLICIT
        # store-neutral posture as the no-shape zero ask below: the SOC
        # walk stays on (battery margin defined) at Balanced pace.  A
        # collapse to None would silently turn the walk OFF and fabricate
        # an unmodelled trajectory.
        if scaled is None:
            return [0.0, 0.0, 0.0]
        return scaled
    flat = _lever(net_mj) if net_mj else None
    if flat is None:
        return vec
    merged = [max(-8.5, min(8.5, v + f)) for v, f in zip(vec, flat)]
    if all(abs(x) < 1e-9 for x in merged):
        return None
    return merged


# (name, phase-1 net MJ/lap, phase-2 net MJ/lap, deploy-lap offset).
# Phase 1 runs from the current lap to the deploy lap; phase 2 from the
# deploy lap to the flag.  deploy_offset=0 means "strike now" (single
# phase).  Net MJ/lap is expressed as a fraction of ERS_MAX_MJ_LAP so the
# table reads as fractions of the car's realistic per-lap lever authority
# (the live call's own -100..100 slider maps to +-ERS_MAX_MJ_LAP).
POLICIES: list[dict[str, Any]] = [
    {
        "name": "GREEDY ATTACK",
        "phase1_mj": 1.0 * ERS_MAX_MJ_LAP,
        "phase2_mj": 1.0 * ERS_MAX_MJ_LAP,
        "deploy_offset": 0,
        "why": "deploy full boost now — convert the earliest window",
    },
    {
        "name": "BALANCED HOLD",
        "phase1_mj": 0.0,
        "phase2_mj": 0.0,
        "deploy_offset": 0,
        "why": "store-neutral — let the tyre edge open the window",
    },
    {
        "name": "TACTICAL STALK",
        "phase1_mj": -0.75 * ERS_MAX_MJ_LAP,
        "phase2_mj": 1.0 * ERS_MAX_MJ_LAP,
        "deploy_offset": 2,
        "why": "bank energy, strike on lap N+2 with a fuller store",
    },
    {
        "name": "SAVE & DEFEND",
        "phase1_mj": -1.0 * ERS_MAX_MJ_LAP,
        "phase2_mj": -1.0 * ERS_MAX_MJ_LAP,
        "deploy_offset": 0,
        "why": "bank to the flag — protect the store and the position",
    },
    {
        "name": "UNDERCUT PREP",
        "phase1_mj": 0.5 * ERS_MAX_MJ_LAP,
        "phase2_mj": 0.0,
        "deploy_offset": 0,
        "why": "moderate push now to build pit-window margin, then manage",
    },
]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _run_phase(leader_code, chaser_code, track_name, start_lap, race_length,
               gap_before_s, tyres, year, battery_pct, net_mj,
               leader_posture="balanced", chaser_shape=None):
    """One simulate_live_call pass at a given net lever; returns (sim, ms).

    ``chaser_shape`` is the caller's per-sector deploy-delta vector (the
    Energy Sandbox slider bank).  With a shape supplied, the policy's net
    ask is expressed THROUGH it — each phase's lever scales the shape,
    preserving its sector spread — so the ranked policies and the manual
    sliders share one vocabulary instead of two flat levers.
    """
    t0 = time.perf_counter()
    sim = oi.simulate_live_call(
        leader_code=leader_code, chaser_code=chaser_code,
        track_name=track_name, start_lap=start_lap,
        race_length=race_length, gap_before_s=max(0.05, gap_before_s),
        leader_tyre_compound=tyres["leader"]["compound"],
        chaser_tyre_compound=tyres["chaser"]["compound"],
        leader_tyre_age=tyres["leader"]["age"],
        chaser_tyre_age=tyres["chaser"]["age"],
        year=year,
        chaser_ers_deltas=_phase_vector(chaser_shape, net_mj),
        chaser_battery_pct=battery_pct,
        leader_posture=leader_posture,
    )
    return sim, (time.perf_counter() - t0) * 1000.0


def _phase2_gap(sim1, phase1_laps):
    """State the phase-1 walk leaves the pair at, for phase 2 to inherit.

    simulate_live_call already walks to race_length; for two-phase policies
    (the chaser's TACTICAL STALK and the leader's BANK & STRIKE) phase 2 is
    re-run from the deploy lap with the gap the phase-1 projection reached
    at that lap and BOTH cars' tyre ages ticked from their OWN starting
    ages, so phase 2 starts where phase 1 genuinely ended rather than from
    a hand-waved state.

    Returns (gap_before_s, chaser_age, leader_age).  The leader's age is
    returned separately: the old implementation ticked only the chaser's
    age and stamped it onto both cars, so the leader ran phase 2 on the
    wrong tyre age whenever the two stints started at different laps.
    """
    laps = sim1.get("laps") or []
    row = None
    for l in laps:
        if l["lap"] <= phase1_laps:
            row = l
    if row is None:
        row = laps[0] if laps else None
    if row is None:
        return None, None, None
    # Age tick: phase 2 starts one lap after phase 1's last walked lap, and
    # each car ticks from its own phase-1 age.
    meta = sim1["meta"]
    laps_run = (row["lap"] - meta["start_lap"]) + 1
    c_age2 = int(meta["tyres"]["chaser"]["age"] + laps_run)
    l_age2 = int(meta["tyres"]["leader"]["age"] + laps_run)
    return max(0.05, float(row["gap_before_s"])), c_age2, l_age2


def _battery_pct(mj: float) -> float:
    return 100.0 * mj / ERS_STORE_MJ


def evaluate_tactical_policies(
        leader_code: str, chaser_code: str, track_name: str,
        start_lap: int, race_length: int, gap_before_s: float,
        leader_tyre_compound: str = "Medium",
        chaser_tyre_compound: str = "Medium",
        leader_tyre_age: float = 10.0, chaser_tyre_age: float = 10.0,
        year: int | None = None, chaser_battery_pct: float | None = None,
        reserve_target_mj: float | None = None,
        leader_posture: str = "balanced",
        chaser_shape=None,
        race_event: str = "green",
        traffic_level: str = "Clear",
) -> dict[str, Any]:
    """Evaluate all five policies against one race state.

    ``leader_posture`` is the game-theoretic lever: 'balanced' assumes the
    leader runs its own race; 'defensive_boost' has the leader counter-
    deploy (its own 4 MJ store, same floor) so every closing-based policy
    is scored against an opponent that actually reacts.

    ``chaser_shape`` is the strategist's own per-sector deploy-delta bank
    (Energy Sandbox sliders, MJ/lap per sector).  When supplied, every
    policy expresses its lever THROUGH that shape (magnitude scaled per
    phase, sector spread preserved) and the payload discloses it under
    ``ers_shape``; the net-0 reallocation case stays active and
    store-neutral, turning the SOC walk on for BALANCED HOLD too.

    ``race_event`` ('green', 'vsc', 'safety_car') and ``traffic_level``
    ('Clear', 'Light', 'Heavy') inject circuit-specific pit loss times,
    Safety Car intervention likelihood, and fresh-tyre undercut window math.

    Returns the full comparison payload: per-policy rows (score components,
    pass lap, energy cost, SOC trajectory stats, feasibility), the ranked
    ACTION card for the winner, and the elapsed timing.  Raises ValueError
    with the underlying reason when the pair cannot be scored at all (same
    driver, uncovered track/tyre) — the route maps that to HTTP 400.
    """
    t_start = time.perf_counter()
    tyres = {
        "leader": {"compound": leader_tyre_compound, "age": leader_tyre_age},
        "chaser": {"compound": chaser_tyre_compound, "age": chaser_tyre_age},
    }
    batt_pct = (float(chaser_battery_pct)
                if chaser_battery_pct is not None else ERS_DEFAULT_START_PCT)
    reserve = (RESERVE_TARGET_MJ if reserve_target_mj is None
               else float(reserve_target_mj))
    horizon = max(0, race_length - start_lap)

    # Pit lane delta & Safety Car evaluations
    pit_summary = pit_strategy.calculate_pit_loss(track_name, event=race_event, traffic=traffic_level)
    undercut_eval = pit_strategy.evaluate_undercut_window(
        gap_s=gap_before_s,
        chaser_tyre=chaser_tyre_compound,
        chaser_age=chaser_tyre_age,
        leader_tyre=leader_tyre_compound,
        leader_age=leader_tyre_age,
        track_name=track_name,
        event=race_event,
    )
    sc_eval = pit_strategy.evaluate_safety_car_opportunity(
        track_name=track_name,
        event=race_event,
        laps_remaining=horizon,
        current_tyre_age=chaser_tyre_age,
        gap_ahead_s=gap_before_s,
    )

    # ---- Baseline (no lever): the reference every policy is scored against.
    base_sim, _ = _run_phase(leader_code, chaser_code, track_name, start_lap,
                             race_length, gap_before_s, tyres, year,
                             batt_pct, 0.0, leader_posture)
    base = base_sim["summary"]
    base_call = base_sim["call"]

    rows = []
    chaser_shape_vec = _shape_vector(chaser_shape)
    for pol in POLICIES:
        deploy_lap = start_lap + pol["deploy_offset"]
        net1, net2 = pol["phase1_mj"], pol["phase2_mj"]
        batt_start_phase2_pct = batt_pct

        if pol["deploy_offset"] == 0:
            sim, ms = _run_phase(leader_code, chaser_code, track_name,
                                 start_lap, race_length, gap_before_s,
                                 tyres, year, batt_pct, net1, leader_posture,
                                 chaser_shape=chaser_shape)
            call, summ = sim["call"], sim["summary"]
            soc_end_pct = summ.get("chaser_soc_end_pct")
            # Only laps the SOC walk actually recorded — a row whose walk
            # is off has no trajectory to report (never fabricate a 100%
            # minimum the car never modelled).
            c_soc_rows = [l["chaser_soc_pct"] for l in sim["laps"]
                          if l.get("chaser_soc_pct") is not None]
            min_soc_pct = min(c_soc_rows) if c_soc_rows else None
            deployed = summ.get("ers_deployed_mj", 0.0)
            banked = summ.get("ers_banked_mj", 0.0)
            energy_limited = summ.get("ers_energy_limited_laps", 0)
            wear_delta = _est_extra_wear(net1, len(sim.get("laps") or []),
                                         chaser_tyre_compound)
            ms_total = ms
        else:
            # Two-phase policy: bank until the deploy lap, then strike.
            phase1_laps = deploy_lap  # walk phase 1 UP TO (and incl.) this lap
            sim1, ms1 = _run_phase(leader_code, chaser_code, track_name,
                                   start_lap, deploy_lap, gap_before_s,
                                   tyres, year, batt_pct, net1, leader_posture,
                                   chaser_shape=chaser_shape)
            gap2, c_age2, l_age2 = _phase2_gap(sim1, phase1_laps)
            # Battery carried into phase 2 = the phase-1 end SOC.
            soc1 = sim1.get("summary", {}).get("chaser_soc_end_pct")
            if soc1 is not None:
                batt_start_phase2_pct = soc1
            # Each car's tyre age ticks from its OWN phase-1 age.
            tyres2 = {
                "leader": {"compound": leader_tyre_compound,
                           "age": (l_age2 if l_age2 is not None
                                   else leader_tyre_age + 1)},
                "chaser": {"compound": chaser_tyre_compound,
                           "age": (c_age2 if c_age2 is not None
                                   else chaser_tyre_age + 1)},
            }
            sim2, ms2 = _run_phase(leader_code, chaser_code, track_name,
                                   deploy_lap + 1, race_length, gap2,
                                   tyres2, year, batt_start_phase2_pct, net2,
                                   leader_posture, chaser_shape=chaser_shape)
            # Wear is priced per phase: the banking phase burns nothing
            # (net <= 0), the strike pays at full lever on the chaser's
            # compound — the old code priced phase-1's lever only, so
            # two-phase strikes rode free.
            wear_delta = (
                _est_extra_wear(net1, len(sim1.get("laps") or []),
                                chaser_tyre_compound)
                + _est_extra_wear(net2, len(sim2.get("laps") or []),
                                  chaser_tyre_compound))
            call, summ = sim2["call"], sim2["summary"]
            soc_end_pct = summ.get("chaser_soc_end_pct")
            soc1_rows = [l.get("chaser_soc_pct") for l in sim1["laps"]
                         if l.get("chaser_soc_pct") is not None]
            soc2_rows = [l.get("chaser_soc_pct") for l in sim2["laps"]
                         if l.get("chaser_soc_pct") is not None]
            min_soc_pct = (min(soc1_rows + soc2_rows)
                           if (soc1_rows or soc2_rows) else None)
            deployed = (sim1["summary"].get("ers_deployed_mj", 0.0)
                        + summ.get("ers_deployed_mj", 0.0))
            banked = (sim1["summary"].get("ers_banked_mj", 0.0)
                      + summ.get("ers_banked_mj", 0.0))
            energy_limited = (sim1["summary"].get("ers_energy_limited_laps", 0)
                              + summ.get("ers_energy_limited_laps", 0))
            ms_total = ms1 + ms2

        pass_lap = call.get("pass_lap")
        cum = float(call.get("cumulative_probability") or 0.0)
        drained = (min_soc_pct is not None
                   and min_soc_pct <= LIVE_ATTACK_MIN_SOC_PCT + 0.05)
        # The reserve bar exists to reject policies that SPEND below the
        # management target.  A zero-spend row (store-neutral hold) never
        # dips — its store sits where the car's is, below target or not —
        # so it must stay recommendable (with the honest negative margin).
        no_spend = deployed <= 1e-9 and banked <= 1e-9
        reserve_breach = (min_soc_pct is not None and not no_spend
                          and min_soc_pct < _battery_pct(reserve) - 0.05)

        # ---- Store-neutral rows: a hold asks no lever, but its SOC walk
        # is still ON ([0, 0, 0] posture) — the projected store is modelled
        # and reported, so the card's battery margin is defined even for
        # the no-slider default (an honest NEGATIVE margin when the car
        # already sits below the target).  The store path is untouched
        # (energy_cost/banked stay 0, min_soc keeps its no-walk value), so
        # a no-spend posture is never priced as a reserve breach: the
        # breach rule rejects policies that SPEND below the target — a car
        # sitting under target must still get the hold advice.
        if soc_end_pct is None:
            # Mirror the walk's own start clamp (30-100%): a reported store
            # never sits below the floor the walk itself enforces.
            soc_end_pct = round(min(100.0, max(LIVE_ATTACK_MIN_SOC_PCT,
                                               batt_pct)), 1)

        # ---- SOC uncertainty (the battery is a synthesized estimate).
        # The band grows with laps since the projection's anchor (race
        # start for the walk); the policy's worst-case battery state is
        # the mean MINUS the band, and the best case the mean PLUS it.
        band = battery_uncertainty_band(horizon)
        band_pct = band["band_pct"]
        soc_end_lo = (max(0.0, soc_end_pct - band_pct)
                      if soc_end_pct is not None else None)
        soc_end_hi = (min(100.0, soc_end_pct + band_pct)
                      if soc_end_pct is not None else None)
        # Worst-case battery enters the score: over-deployment is priced
        # at the LOW end of the band (you defend with the battery you
        # actually have, not the one the model hopes for).
        soc_end_mj = (soc_end_lo / 100.0 * ERS_STORE_MJ
                      if soc_end_lo is not None else None)
        soc_deficit = (max(0.0, reserve - soc_end_mj)
                       if soc_end_mj is not None else 0.0)
        batt_cost = LAMBDA_BATT * soc_deficit

        # ---- Score components (all in race-time seconds; lower is better
        # except the pass reward, which enters negatively as a gain).
        pass_gain = cum * PASS_VALUE_S

        # Undercut & Safety Car / Pit Window integration:
        why_text = pol["why"]
        if pol["name"] == "UNDERCUT PREP":
            if pit_summary.get("is_cheap_stop"):
                # Pitting under VSC / SC saves significant race time (~9-12s)
                time_saved = pit_summary.get("time_saved_s", 0.0)
                pass_gain += time_saved
                why_text = f"BOX UNDER {pit_summary.get('event', 'VSC')} — exploit cheap stop saving ~{time_saved:.1f}s on transit"
            elif undercut_eval.get("status") in ("OPEN_FAVORABLE", "MARGINAL"):
                undercut_gain = max(0.0, undercut_eval.get("net_exit_margin_s", 0.0)) * 1.5
                pass_gain += undercut_gain
                why_text = f"undercut window open: fresh out-lap edge +{undercut_eval.get('fresh_tyre_outlap_gain_s', 1.8):.1f}s clears {gap_before_s:.1f}s gap (+{undercut_eval.get('net_exit_margin_s', 0.0):.1f}s on exit)"

        # Tyre cost: extra health burned vs the Balanced baseline, priced
        # per phase in the branch above (on the CHASER's compound).
        wear_cost = LAMBDA_WEAR * max(0.0, wear_delta)
        # Cliff risk applies to whichever phase carries the positive lever
        # (see _push_lever): a two-phase strike that only converts by
        # draining the store used to escape the cliff penalty entirely.
        push_lever = _push_lever(pol)
        cliff_cost = CLIFF_RISK_S * (
            1.0 if (min_soc_pct is not None and drained
                    and push_lever > 0) else 0.0)

        fail_prob = max(0.0, 1.0 - cum) if pass_lap else 0.0
        risk_cost = LAMBDA_RISK * fail_prob * DIRTY_AIR_PENALTY_S

        deferral = (pass_lap - start_lap) if pass_lap else horizon
        latency_cost = LAMBDA_LATENCY * deferral

        # Hard constraint: reserve breach or a pass that only converts by
        # draining the store to its floor can never be recommended.
        infeasible = reserve_breach or (pass_lap and drained
                                        and push_lever > 0)
        score = (batt_cost + wear_cost + cliff_cost + risk_cost
                 + latency_cost) - (0.0 if infeasible else pass_gain)

        rows.append({
            "policy": pol["name"],
            "why": why_text,
            "deploy_lap": deploy_lap if pol["deploy_offset"] else None,
            "verdict": call.get("verdict"),
            "verdict_reason": call.get("verdict_reason"),
            "pass_lap": pass_lap,
            "overtake_probability": round(cum, 4),
            "energy_cost_mj": round(deployed, 3),
            "energy_banked_mj": round(banked, 3),
            "energy_limited_laps": energy_limited,
            "soc_end_pct": soc_end_pct,
            "soc_band_pct": band_pct,
            "soc_end_range_pct": ([soc_end_lo, soc_end_hi]
                                  if soc_end_pct is not None else None),
            "min_soc_pct": min_soc_pct,
            "battery_margin_pct": (round(soc_end_pct - _battery_pct(reserve), 1)
                                   if soc_end_pct is not None else None),
            "battery_margin_worst_pct": (round(soc_end_lo - _battery_pct(reserve), 1)
                                         if soc_end_lo is not None else None),
            "score_components": {
                "pass_gain_s": round(pass_gain, 3),
                "battery_cost_s": round(batt_cost, 3),
                "wear_cost_s": round(wear_cost, 3),
                "cliff_cost_s": round(cliff_cost, 3),
                "risk_cost_s": round(risk_cost, 3),
                "latency_cost_s": round(latency_cost, 3),
            },
            "score_s": round(score, 3),
            "feasible": not infeasible,
            "infeasible_reason": (
                "reserve breach — projected store dips below the "
                f"{_battery_pct(reserve):.0f}% management target"
                if reserve_breach else
                ("pass converts only by draining the store to its floor"
                 if (pass_lap and drained and push_lever > 0) else None)),
            "phase_laps_ms": round(ms_total, 1),
            # The strategist's slider bank, as actually walked (per-phase
            # scaling included) — disclosure, so a rendered card can be
            # reproduced from the payload alone.
            "ers_shape": (chaser_shape_vec
                          if chaser_shape_vec is not None else None),
        })

    feasible = [r for r in rows if r["feasible"]]
    ranked = sorted(feasible or rows, key=lambda r: r["score_s"])
    best, second = (ranked[0], ranked[1] if len(ranked) > 1 else None)
    # Decision margin = the gap to the next DISTINCT score.  A duplicated
    # score is a mirror of the same walk, not an alternative — two policies
    # that price identically ARE the same plan, so the tie cannot make the
    # call look like a coin flip (the runner-up slot still discloses it).
    next_scores = [r["score_s"] for r in ranked[1:]
                   if r["score_s"] > best["score_s"] + 1e-9]
    margin = ((min(next_scores) - best["score_s"])
              if next_scores else
              ((second["score_s"] - best["score_s"])
               if second is not None else CONFIDENCE_SPAN_S))
    # Confidence = decision margin, DEGRADED by the SOC uncertainty band:
    # a wide band means the battery itself is uncertain, so even a large
    # score margin is less trustworthy.  band spans [floor, cap] = [2, 8];
    # map it onto a 1.0 -> 0.6 multiplier (never below 0.6: the ranking
    # still carries information even at the cap).
    band_state = battery_uncertainty_band(horizon)
    band_mult = 1.0 - 0.4 * ((band_state["band_pct"] - band_state["floor_pct"])
                             / max(1e-9, band_state["cap_pct"]
                                   - band_state["floor_pct"]))
    confidence = round(min(1.0, max(0.0, (margin / CONFIDENCE_SPAN_S) * band_mult)), 2)

    # ---- ACTION card (the challenge's required output shape).
    sc = best["score_components"]
    action = {
        "action": best["policy"],
        "deploy_lap": best["deploy_lap"] or start_lap,
        "energy_pct": (round(best["energy_cost_mj"] / ERS_STORE_MJ * 100.0, 1)
                       if best["energy_cost_mj"] else 0.0),
        "expected_gap_s": round((base.get("projected_final_gap_s") or 0.0)
                                - (best["overtake_probability"] * gap_before_s),
                                2),
        "overtake_probability": best["overtake_probability"],
        "energy_cost_mj": best["energy_cost_mj"],
        "expected_finish_delta_s": round(-sc["pass_gain_s"]
                                         + sc["battery_cost_s"]
                                         + sc["wear_cost_s"]
                                         + sc["risk_cost_s"]
                                         + sc["latency_cost_s"], 2),
        "battery_margin_pct": best["battery_margin_pct"],
        "battery_margin_worst_pct": best["battery_margin_worst_pct"],
        "soc_band_pct": best["soc_band_pct"],
        "confidence": confidence,
        "feasible": best["feasible"],
        "reason": _reason(best, rows, batt_pct, reserve),
    }

    elapsed_ms = (time.perf_counter() - t_start) * 1000.0
    return {
        "perspective": "chaser",
        "state": {
            "leader": leader_code, "chaser": chaser_code,
            "track": track_name, "start_lap": start_lap,
            "race_length": race_length, "gap_before_s": gap_before_s,
            "tyres": tyres, "year": year,
            "battery_pct": batt_pct,
            "ers_shape": (list(chaser_shape_vec)
                          if chaser_shape_vec is not None else None),
            "leader_posture": leader_posture,
            "battery_band_pct": band_state["band_pct"],
            "battery_range_pct": [round(max(0.0, batt_pct - band_state["band_pct"]), 1),
                                   round(min(100.0, batt_pct + band_state["band_pct"]), 1)],
            "reserve_target_pct": round(_battery_pct(reserve), 1),
        },
        "baseline": {
            "verdict": base_call.get("verdict"),
            "pass_lap": base_call.get("pass_lap"),
            "cumulative_probability": base_call.get("cumulative_probability"),
            "avg_pace_gap_s": base.get("avg_pace_gap_s"),
            "projected_final_gap_s": base.get("projected_final_gap_s"),
            "leader_defense": base.get("leader_defense"),
        },
        "policies": rows,
        "recommendation": {
            "action_card": action,
            "runner_up": ({k: second[k] for k in
                           ("policy", "score_s", "overtake_probability",
                            "pass_lap")} if second is not None else None),
            "decision_margin_s": round(margin, 3),
            "confidence": confidence,
            "infeasible_policies": [r["policy"] for r in rows
                                    if not r["feasible"]],
        },
        "scoring_constants": {
            "pass_value_s": PASS_VALUE_S,
            "lambda_battery_s_per_mj": LAMBDA_BATT,
            "reserve_target_mj": reserve,
            "lambda_wear_s_per_health_pct": LAMBDA_WEAR,
            "lambda_risk": LAMBDA_RISK,
            "dirty_air_penalty_s": DIRTY_AIR_PENALTY_S,
            "lambda_latency_s_per_lap": LAMBDA_LATENCY,
            "soc_band": band_state,
            "confidence_band_multiplier": round(band_mult, 3),
        },
        "latency_ms": round(elapsed_ms, 1),
        "latency_budget_ms": LATENCY_BUDGET_MS,
        "pit_analysis": {
            "race_event": pit_summary["event"],
            "traffic_level": traffic_level,
            "effective_pit_loss_s": pit_summary["effective_pit_loss_s"],
            "baseline_green_s": pit_summary["baseline_green_s"],
            "time_saved_s": pit_summary["time_saved_s"],
            "is_cheap_stop": pit_summary["is_cheap_stop"],
            "sc_probability": pit_summary["sc_probability"],
            "sc_risk_tier": pit_summary["sc_risk_tier"],
            "undercut": undercut_eval,
            "safety_car": sc_eval,
        },
    }


def _push_lever(pol: dict) -> float:
    """The lever of the phase that actually PUSHES (positive net MJ/lap).

    Single-phase policies push with phase 1; two-phase (bank-then-strike)
    policies push with phase 2 — their phase-1 lever is negative (banking).
    The cliff-risk penalty and the floor-drain infeasibility must key off
    THIS lever: the old code tested phase 1 for both, which exempted every
    two-phase policy from the cliff penalty and the drain rule.
    """
    if pol["deploy_offset"] == 0:
        return pol["phase1_mj"]
    return pol["phase2_mj"]


def _est_extra_wear(net_mj: float, laps: int, compound: str = "Medium") -> float:
    """Extra tyre health-% burned by a positive lever vs Balanced.

    Pushing spends deploy MJ that Balanced would bank; the tyre cost is
    modelled from the wear meter itself: full lever ≈ double the Balanced
    heat load on those laps.  Health-% per lap at full lever ≈ the
    COMPOUND's loss rate (via tyre_health's linear model) — reused here so
    the engine does not invent a second wear model.  The compound matters:
    the old hard-coded Medium priced a Soft-tyre attack at Medium's heat
    load (Soft wears ~2.5x faster).
    """
    if net_mj <= 0 or laps <= 0:
        return 0.0
    from tyre_degradation import tyre_health
    h0 = tyre_health(compound, 0)
    h1 = tyre_health(compound, 1)
    per_lap_pct = max(0.0, float(h0) - float(h1))          # ≈ balanced heat
    lever_frac = min(1.0, net_mj / ERS_MAX_MJ_LAP)
    return per_lap_pct * lever_frac * laps                  # extra % burned


def _reason(best: dict, rows: list, batt_pct: float, reserve_mj: float) -> str:
    """One strategist-grade sentence: WHY this policy, why not the rest."""
    bits = []
    if best["pass_lap"]:
        bits.append(f"converts the pass on lap {best['pass_lap']} "
                    f"(P {best['overtake_probability']:.0%})")
    else:
        bits.append("no policy converts before the flag — this one wastes "
                    "the least energy on a dead battle")
    if best["energy_banked_mj"]:
        bits.append(f"banks {best['energy_banked_mj']:.1f} MJ")
    if best["battery_margin_pct"] is not None:
        bits.append(f"keeps +{best['battery_margin_pct']:.0f}% store margin")
    infeasible = [r["policy"] for r in rows if not r["feasible"]]
    if infeasible:
        bits.append(f"rejected {', '.join(infeasible)} — reserve breach")
    else:
        loser = next((r for r in rows
                      if r["policy"] != best["policy"] and r["feasible"]), None)
        if loser:
            bits.append(f"beats {loser['policy']} by "
                        f"{abs(loser['score_s'] - best['score_s']):.1f} s of "
                        "expected race time")
    return "; ".join(bits) + "."


# ---------------------------------------------------------------------------
# LEADER PERSPECTIVE — defending the position against the chaser.
#
# The chaser engine above answers "which deployment buys the pass?".  The
# leader engine answers the mirrored question: "the car behind is attacking
# — which response keeps the position WITHOUT buying it with the battery?"
# Same machinery, same constraints, same honesty rules; the objective
# flips: score = P(attack converts) x LOST_POSITION_S + energy + wear.
# ---------------------------------------------------------------------------

# The threat model: from the leader's seat the chaser is ASSUMED to be
# attacking (that is the situation worth advising on).  Disclosed constant,
# not a hidden assumption: every leader walk rolls the chaser at this
# fixed push posture (fraction of the chaser's full lever authority).
LEADER_THREAT_CHASER_ERS = 75.0

# Race-time cost of losing the position (same scale as PASS_VALUE_S: the
# swap itself plus the defensive delta it hands over).
LOST_POSITION_S = PASS_VALUE_S

# Value of each lap the defence buys before the swap: track position pays
# every lap (points, tyre management, the next pit window).  A defence that
# delays the conversion 5 laps earns this credit even if the swap is only
# postponed — and battery banked while conceding funds the counter-attack
# from the other seat.  Disclosed constant, same spirit as LAMBDA_LATENCY.
HELD_LAP_CREDIT_S = 0.2

LEADER_POLICIES: list[dict[str, Any]] = [
    {
        "name": "HOLD & MANAGE",
        "lever": None,           # no leader lever: tyre edge does the work
        "deploy_offset": 0,
        "why": "no counter-deployment — make the chaser convert on merit "
               "or not at all",
    },
    {
        "name": "COUNTER-DEPLOY",
        "lever": 1.0 * ERS_MAX_MJ_LAP,
        "deploy_offset": 0,
        "why": "match the attack boost-for-boost — close the door now",
    },
    {
        "name": "REACTIVE DEFENSE",
        "preset": "defensive_boost",   # the posture preset: reacts to the
                                       # attack it sees (2x chaser slider)
        "lever": 0.0,
        "deploy_offset": 0,
        "why": "answer only as hard as the attack — conserve what the "
               "chaser's push does not force",
    },
    {
        "name": "BANK & STRIKE",
        "lever": -1.0 * ERS_MAX_MJ_LAP,
        "strike_lever": 1.0 * ERS_MAX_MJ_LAP,
        "deploy_offset": 2,
        "why": "concede ground early, refill, counter-deploy hard from "
               "lap N+2 where the chaser's own store is fading",
    },
    {
        "name": "BANK & HOPE",
        "lever": -1.0 * ERS_MAX_MJ_LAP,
        "deploy_offset": 0,
        "why": "bank to the flag — concede the lap time, protect the "
               "store, force a low-percentage lunge",
    },
]


def _leader_walk(leader_code, chaser_code, track_name, start_lap, race_length,
                 gap_before_s, tyres, year, chaser_batt_pct, leader_batt_pct,
                 lever, preset=None, leader_shape=None, threat_shape=None):
    """One leader-side simulate_live_call pass; returns (sim, ms).

    ``leader_shape`` is the strategist's own per-sector bank (Energy
    Sandbox sliders): when supplied, each policy's lever is expressed
    through it, magnitude scaled, spread preserved.  ``threat_shape`` is
    the attacking car's per-sector bank — the attack the defence is
    scored against runs the chaser's shape, not a flat lever.
    """
    kwargs = {}
    if preset:
        kwargs["leader_posture"] = preset
    else:
        vec = _phase_vector(leader_shape, lever)
        if vec is not None:
            kwargs["leader_ers_deltas"] = vec
    threat_vec = _shape_vector(threat_shape)
    if threat_vec is not None:
        kwargs["chaser_ers_deltas"] = threat_vec
    t0 = time.perf_counter()
    sim = oi.simulate_live_call(
        leader_code=leader_code, chaser_code=chaser_code,
        track_name=track_name, start_lap=start_lap,
        race_length=race_length, gap_before_s=max(0.05, gap_before_s),
        leader_tyre_compound=tyres["leader"]["compound"],
        chaser_tyre_compound=tyres["chaser"]["compound"],
        leader_tyre_age=tyres["leader"]["age"],
        chaser_tyre_age=tyres["chaser"]["age"],
        year=year,
        chaser_ers=LEADER_THREAT_CHASER_ERS,
        chaser_battery_pct=chaser_batt_pct,
        leader_battery_pct=leader_batt_pct,
        **kwargs,
    )
    return sim, (time.perf_counter() - t0) * 1000.0


def _leader_push_lever(pol: dict) -> float:
    """Leader-side twin of _push_lever: the lever of the phase that PUSHES.

    Single-phase leader policies push with their lever (COUNTER-DEPLOY);
    REACTIVE DEFENSE pushes via the posture preset (scored with lever 0.0
    here, so a drained reactive defence escapes the cliff penalty by
    construction, not by accident — its drain shows up as infeasibility,
    which is the honest signal).  Two-phase BANK & STRIKE pushes with its
    strike_lever; banking rows never push.  Without this keying, a drained
    two-phase counter-strike paid no cliff risk at all.
    """
    if pol.get("preset") is not None:
        return 0.0            # preset-driven: drain => infeasible, not cliff
    if pol["deploy_offset"] == 0:
        return pol.get("lever") or 0.0
    return pol.get("strike_lever") or 0.0


def evaluate_leader_policies(
        leader_code: str, chaser_code: str, track_name: str,
        start_lap: int, race_length: int, gap_before_s: float,
        leader_tyre_compound: str = "Medium",
        chaser_tyre_compound: str = "Medium",
        leader_tyre_age: float = 10.0, chaser_tyre_age: float = 10.0,
        year: int | None = None, chaser_battery_pct: float | None = None,
        leader_battery_pct: float | None = None,
        reserve_target_mj: float | None = None,
        leader_shape=None, threat_shape=None,
        race_event: str = "green",
        traffic_level: str = "Clear",
) -> dict[str, Any]:
    """Evaluate five DEFENCE policies for the leader against one threat.

    The chaser is modelled as attacking (LEADER_THREAT_CHASER_ERS posture);
    each policy is the leader's answer.  Rows are scored on the probability
    the position SURVIVES, priced against the leader's own battery and
    tyre costs, with the same structural constraints as the chaser engine
    (reserve breach / floor-drain => INFEASIBLE, never recommended).

    ``leader_shape`` expresses the leader's own slider bank through every
    policy lever; ``threat_shape`` runs the attacking chaser on its slider
    bank instead of the flat default lever.  Both may be net-0
    reallocations (active, store-neutral).

    ``race_event`` and ``traffic_level`` inject real pit lane delta loss
    and Safety Car intervention likelihood.

    The ACTION card answers: how do I keep this position, what does the
    threat look like under that answer, and what does the defence cost?
    """
    t_start = time.perf_counter()
    tyres = {
        "leader": {"compound": leader_tyre_compound, "age": leader_tyre_age},
        "chaser": {"compound": chaser_tyre_compound, "age": chaser_tyre_age},
    }
    lbatt = (float(leader_battery_pct)
             if leader_battery_pct is not None else ERS_DEFAULT_START_PCT)
    cbatt = (float(chaser_battery_pct)
             if chaser_battery_pct is not None else ERS_DEFAULT_START_PCT)
    reserve = (RESERVE_TARGET_MJ if reserve_target_mj is None
               else float(reserve_target_mj))
    horizon = max(0, race_length - start_lap)
    band = battery_uncertainty_band(horizon)

    pit_summary = pit_strategy.calculate_pit_loss(track_name, event=race_event, traffic=traffic_level)

    rows = []
    base_summ = None
    leader_shape_vec = _shape_vector(leader_shape)
    threat_shape_vec = _shape_vector(threat_shape)
    for pol in LEADER_POLICIES:
        deploy_lap = start_lap + pol["deploy_offset"]
        preset = pol.get("preset")
        lever = pol.get("lever") or 0.0

        if pol["deploy_offset"] == 0:
            sim, ms = _leader_walk(
                leader_code, chaser_code, track_name, start_lap, race_length,
                gap_before_s, tyres, year, cbatt, lbatt,
                (lever if pol.get("preset") is None else 0.0)
                if preset is None else 0.0,
                preset=preset, leader_shape=leader_shape_vec,
                threat_shape=threat_shape_vec)
            summ = sim["summary"]
            wear_delta = _est_extra_wear(
                lever if preset is None else 0.0,
                len(sim.get("laps") or []), leader_tyre_compound)
            ms_total = ms
        else:
            # Two-phase: concede + refill until the deploy lap, then
            # counter-deploy.  Phase 2 inherits the REAL phase-1 gap/SOC.
            sim1, ms1 = _leader_walk(
                leader_code, chaser_code, track_name, start_lap, deploy_lap,
                gap_before_s, tyres, year, cbatt, lbatt, lever,
                leader_shape=leader_shape_vec, threat_shape=threat_shape_vec)
            gap2, c_age2, l_age2 = _phase2_gap(sim1, deploy_lap)
            soc1 = sim1.get("summary", {}).get("leader_soc_end_pct")
            batt2 = soc1 if soc1 is not None else lbatt
            # Each car's tyre age ticks from its OWN phase-1 age.
            tyres2 = {
                "leader": {"compound": leader_tyre_compound,
                           "age": (l_age2 if l_age2 is not None
                                   else leader_tyre_age + 1)},
                "chaser": {"compound": chaser_tyre_compound,
                           "age": (c_age2 if c_age2 is not None
                                   else chaser_tyre_age + 1)},
            }
            sim2, ms2 = _leader_walk(
                leader_code, chaser_code, track_name, deploy_lap + 1,
                race_length, gap2, tyres2, year, cbatt, batt2,
                pol.get("strike_lever") or 0.0,
                leader_shape=leader_shape_vec, threat_shape=threat_shape_vec)
            # Wear per phase on the LEADER's compound: the concede/bank
            # phase burns nothing (lever <= 0), the counter-strike pays.
            wear_delta = (
                _est_extra_wear(lever, len(sim1.get("laps") or []),
                                leader_tyre_compound)
                + _est_extra_wear(pol.get("strike_lever") or 0.0,
                                  len(sim2.get("laps") or []),
                                  leader_tyre_compound))
            summ = sim2["summary"]
            # Stitch phase-1 SOC rows in for the constraint check.
            stitched = [l["leader_soc_pct"] for l in sim1["laps"]
                        if l.get("leader_soc_pct") is not None]
            summ = dict(summ)
            summ["_phase1_soc_rows"] = stitched
            # Phase-1's store flow counts in the row's energy figures
            # (mirrors the chaser engine's two-phase accumulation): BANK &
            # STRIKE banks in phase 1 and strikes in phase 2 — dropping
            # phase-1's banked MJ made the row's deployed/banked figures
            # disagree with its own soc_end_pct.
            d1 = sim1["summary"].get("leader_defense") or {}
            d2 = summ.get("leader_defense") or {}
            head = d2 or d1
            summ["leader_defense"] = {
                "posture": head.get("posture"),
                "net_mj_per_lap": head.get("net_mj_per_lap"),
                "deployed_mj": round(float(d1.get("deployed_mj", 0.0))
                                     + float(d2.get("deployed_mj", 0.0)), 3),
                "banked_mj": round(float(d1.get("banked_mj", 0.0))
                                   + float(d2.get("banked_mj", 0.0)), 3),
                "energy_limited_laps": (int(d1.get("energy_limited_laps", 0))
                                        + int(d2.get("energy_limited_laps",
                                                     0))),
                "pace_s_per_lap": head.get("pace_s_per_lap"),
            }
            ms_total = ms1 + ms2

        cum = float(sim["call"].get("cumulative_probability") or 0.0)
        p_attack = cum
        p_defend = 1.0 - cum
        pass_lap = sim["call"].get("pass_lap")
        converted = pass_lap is not None
        laps_held = ((pass_lap - start_lap) if converted else horizon)

        # Leader SOC trajectory + constraints (mirrors the chaser rules).
        soc_rows = [l.get("leader_soc_pct") for l in sim["laps"]
                    if l.get("leader_soc_pct") is not None]
        soc_rows += summ.get("_phase1_soc_rows", [])
        min_soc_pct = min(soc_rows) if soc_rows else None
        soc_end_pct = summ.get("leader_soc_end_pct")
        # Store-neutral rows: a hold asks no lever, but its SOC walk is
        # still ON ([0, 0, 0] posture) — the projected store is modelled
        # and reported (an honest NEGATIVE margin when the car already
        # sits below the target).  The store path is untouched, so a
        # no-spend posture is never priced as a reserve breach: the
        # breach rule rejects policies that SPEND below the target.
        if soc_end_pct is None:
            # Mirror the walk's own start clamp (30-100%), as chaser-side.
            soc_end_pct = round(min(100.0, max(LIVE_ATTACK_MIN_SOC_PCT,
                                               lbatt)), 1)
        deployed = summ.get("leader_defense", {}) or {}
        deployed_mj = float(deployed.get("deployed_mj", 0.0))
        banked_mj = float(deployed.get("banked_mj", 0.0))
        energy_limited = int(deployed.get("energy_limited_laps", 0))
        lever_on = deployed_mj > 0 or banked_mj > 0

        drained = (min_soc_pct is not None
                   and min_soc_pct <= LIVE_ATTACK_MIN_SOC_PCT + 0.05)
        # Mirror of the chaser rule: the reserve bar rejects defences
        # that SPEND below the target; a zero-spend hold never dips, so
        # it stays recommendable (honest negative margin when below).
        no_spend = deployed_mj <= 1e-9 and banked_mj <= 1e-9
        reserve_breach = (min_soc_pct is not None and not no_spend
                          and min_soc_pct < _battery_pct(reserve) - 0.05)

        # Worst-case battery prices the cost terms (same rule as chaser).
        soc_end_lo = (max(0.0, soc_end_pct - band["band_pct"])
                      if soc_end_pct is not None else None)
        soc_end_mj = (soc_end_lo / 100.0 * ERS_STORE_MJ
                      if soc_end_lo is not None else None)
        soc_deficit = (max(0.0, reserve - soc_end_mj)
                       if soc_end_mj is not None else 0.0)
        batt_cost = LAMBDA_BATT * soc_deficit

        # Tyre cost of counter-deploying: priced per phase in the branch
        # above (on the LEADER's compound) — the old code priced the
        # phase-1 lever only, so BANK & STRIKE's full-lever strike was free.
        wear_cost = LAMBDA_WEAR * max(0.0, wear_delta)

        # Cliff risk: same rule as the chaser engine, keyed off the phase
        # that actually pushes (the counter-strike for BANK & STRIKE).  A
        # defence that only survives by draining the store to its floor
        # must price the cliff — the old code exempted every two-phase
        # defence from the penalty entirely (it barred the drain but
        # scored it as free).
        push_lever = _leader_push_lever(pol)
        cliff_cost = CLIFF_RISK_S * (
            1.0 if (min_soc_pct is not None and drained
                    and push_lever > 0) else 0.0)

        infeasible = reserve_breach or drained
        # Score: the attack converting is the leader's loss (a walk that
        # never converts leaves only its residual cum as risk); every lap
        # the defence buys earns HELD_LAP_CREDIT_S.  Lower is better.
        # Infeasible rows keep their honest score but are barred from
        # winning (and flagged on the card if EVERY policy is barred).
        position_risk = (LOST_POSITION_S if converted
                         else p_attack * LOST_POSITION_S)
        hold_credit = HELD_LAP_CREDIT_S * laps_held
        score = (position_risk + batt_cost + wear_cost + cliff_cost
                 - hold_credit)

        rows.append({
            "policy": pol["name"],
            "why": pol["why"],
            "deploy_lap": deploy_lap if pol["deploy_offset"] else None,
            "converted": converted,
            "pass_lap": pass_lap,
            "laps_held": laps_held,
            "overtake_probability": round(cum, 4),
            "hold_probability": round(p_defend, 4),
            "energy_cost_mj": round(deployed_mj, 3),
            "energy_banked_mj": round(banked_mj, 3),
            "energy_limited_laps": energy_limited,
            "soc_end_pct": soc_end_pct,
            "soc_band_pct": band["band_pct"],
            "soc_end_range_pct": ([max(0.0, soc_end_pct - band["band_pct"]),
                                   min(100.0, soc_end_pct + band["band_pct"])]
                                  if soc_end_pct is not None else None),
            "min_soc_pct": min_soc_pct,
            "battery_margin_pct": (round(soc_end_pct - _battery_pct(reserve), 1)
                                   if soc_end_pct is not None else None),
            "battery_margin_worst_pct": (
                round(soc_end_lo - _battery_pct(reserve), 1)
                if soc_end_lo is not None else None),
            "score_components": {
                "position_risk_s": round(position_risk, 3),
                "hold_credit_s": round(hold_credit, 3),
                "battery_cost_s": round(batt_cost, 3),
                "wear_cost_s": round(wear_cost, 3),
                "cliff_cost_s": round(cliff_cost, 3),
            },
            "score_s": round(score, 3),
            "feasible": not infeasible,
            "infeasible_reason": (
                "reserve breach — the defence itself dips below the "
                f"{_battery_pct(reserve):.0f}% management target"
                if reserve_breach else
                ("counter-deployment drains the store to its 30% floor — "
                 "the door reopens exactly where defence is needed"
                 if drained else None)),
            "phase_laps_ms": round(ms_total, 1),
            # The strategist's own slider bank as walked, disclosed like
            # the chaser engine's ers_shape.
            "ers_shape": (list(leader_shape_vec)
                          if leader_shape_vec is not None else None),
            "threat_shape": (list(threat_shape_vec)
                             if threat_shape_vec is not None else None),
        })
        if pol["name"] == "HOLD & MANAGE":
            base_summ = summ

    feasible = [r for r in rows if r["feasible"]]
    ranked = sorted(feasible or rows, key=lambda r: r["score_s"])
    best, second = (ranked[0], ranked[1] if len(ranked) > 1 else None)
    # Decision margin = the gap to the next DISTINCT score (a duplicated
    # score mirrors the same walk — it is not an alternative plan).
    next_scores = [r["score_s"] for r in ranked[1:]
                   if r["score_s"] > best["score_s"] + 1e-9]
    margin = ((min(next_scores) - best["score_s"])
              if next_scores else
              ((second["score_s"] - best["score_s"])
               if second is not None else CONFIDENCE_SPAN_S))
    band_mult = 1.0 - 0.4 * ((band["band_pct"] - band["floor_pct"])
                             / max(1e-9, band["cap_pct"]
                                   - band["floor_pct"]))
    confidence = round(min(1.0, max(0.0,
                        (margin / CONFIDENCE_SPAN_S) * band_mult)), 2)

    # Honest no-hope disclosure: when even the best feasible defence faces
    # a near-certain conversion, say so instead of selling a policy.
    no_hope = best["feasible"] and best["overtake_probability"] >= 0.75

    sc = best["score_components"]
    action = {
        "action": best["policy"],
        "deploy_lap": best["deploy_lap"] or start_lap,
        "energy_pct": (round(best["energy_cost_mj"] / ERS_STORE_MJ * 100.0, 1)
                       if best["energy_cost_mj"] else 0.0),
        "threat_probability": best["overtake_probability"],
        "hold_probability": best["hold_probability"],
        "converted": best["converted"],
        "pass_lap": best["pass_lap"],
        "laps_held": best["laps_held"],
        "energy_cost_mj": best["energy_cost_mj"],
        "position_at_risk_s": round(LOST_POSITION_S, 1),
        "expected_finish_delta_s": round(-(sc["position_risk_s"]
                                           - sc["hold_credit_s"]
                                           + sc["battery_cost_s"]
                                           + sc["wear_cost_s"]), 2),
        "battery_margin_pct": best["battery_margin_pct"],
        "battery_margin_worst_pct": best["battery_margin_worst_pct"],
        "soc_band_pct": best["soc_band_pct"],
        "confidence": confidence,
        "feasible": best["feasible"],
        "reason": _reason_leader(best, rows, no_hope),
    }

    elapsed_ms = (time.perf_counter() - t_start) * 1000.0
    base_row = next((r for r in rows if r["policy"] == "HOLD & MANAGE"),
                    None)
    return {
        "perspective": "leader",
        "state": {
            "leader": leader_code, "chaser": chaser_code,
            "track": track_name, "start_lap": start_lap,
            "race_length": race_length, "gap_before_s": gap_before_s,
            "tyres": tyres, "year": year,
            "battery_pct": lbatt,
            "ers_shape": (list(leader_shape_vec)
                          if leader_shape_vec is not None else None),
            "threat_shape": (list(threat_shape_vec)
                             if threat_shape_vec is not None else None),
            "battery_band_pct": band["band_pct"],
            "battery_range_pct": [round(max(0.0, lbatt - band["band_pct"]), 1),
                                   round(min(100.0, lbatt + band["band_pct"]), 1)],
            "threat_assumption": {
                "chaser_ers": LEADER_THREAT_CHASER_ERS,
                "chaser_battery_pct": cbatt,
            },
            "reserve_target_pct": round(_battery_pct(reserve), 1),
        },
        "baseline": ({
            "policy": "HOLD & MANAGE",
            "hold_probability": base_row["hold_probability"],
            "overtake_probability": base_row["overtake_probability"],
        } if base_row is not None else None),
        "policies": rows,
        "recommendation": {
            "action_card": action,
            "runner_up": ({k: second[k] for k in
                           ("policy", "score_s", "hold_probability",
                            "overtake_probability")}
                          if second is not None else None),
            "decision_margin_s": round(margin, 3),
            "confidence": confidence,
            "no_hope_disclosure": no_hope,
            "infeasible_policies": [r["policy"] for r in rows
                                    if not r["feasible"]],
        },
        "scoring_constants": {
            "lost_position_s": LOST_POSITION_S,
            "lambda_battery_s_per_mj": LAMBDA_BATT,
            "reserve_target_mj": reserve,
            "lambda_wear_s_per_health_pct": LAMBDA_WEAR,
            "held_lap_credit_s": HELD_LAP_CREDIT_S,
            "threat_chaser_ers": LEADER_THREAT_CHASER_ERS,
            "soc_band": band,
            "confidence_band_multiplier": round(band_mult, 3),
        },
        "latency_ms": round(elapsed_ms, 1),
        "latency_budget_ms": LATENCY_BUDGET_MS,
        "pit_analysis": {
            "race_event": pit_summary["event"],
            "traffic_level": traffic_level,
            "effective_pit_loss_s": pit_summary["effective_pit_loss_s"],
            "baseline_green_s": pit_summary["baseline_green_s"],
            "time_saved_s": pit_summary["time_saved_s"],
            "is_cheap_stop": pit_summary["is_cheap_stop"],
            "sc_probability": pit_summary["sc_probability"],
            "sc_risk_tier": pit_summary["sc_risk_tier"],
        },
    }


def _reason_leader(best: dict, rows: list, no_hope: bool) -> str:
    """Strategist-grade sentence for the leader's card."""
    bits = []
    if best.get("converted"):
        bits.append(f"the attack lands on lap {best['pass_lap']} — this "
                    f"posture holds for {best['laps_held']} laps "
                    f"(threat P {best['overtake_probability']:.0%})")
    elif no_hope or best["overtake_probability"] >= 0.75:
        bits.append(f"no conversion before the flag, but the threat stays "
                    f"at {best['overtake_probability']:.0%} — this posture "
                    "limits the damage")
    else:
        bits.append(f"holds the position — the attack stalls at "
                    f"{best['overtake_probability']:.0%}")
    if best["energy_banked_mj"]:
        bits.append(f"banks {best['energy_banked_mj']:.1f} MJ for the "
                    "next battle")
    elif best["energy_cost_mj"]:
        bits.append(f"spends {best['energy_cost_mj']:.1f} MJ defending")
    if best["battery_margin_pct"] is not None:
        bits.append(f"keeps +{best['battery_margin_pct']:.0f}% store margin")
    infeasible = [r["policy"] for r in rows if not r["feasible"]]
    if infeasible:
        bits.append(f"rejected {', '.join(infeasible)} — defence costs "
                    "battery the car does not have")
    return "; ".join(bits) + "."


# ---------------------------------------------------------------------------
# UNIFIED CALL — one decision from BOTH engines.
#
# The chaser engine answers "which attack policy?"; the leader engine
# answers "which defence?".  Neither is a complete call alone: an attack
# pick only means something against the defence the opponent will actually
# run, and a defence pick only means something against the threat it faces.
# `evaluate_call` runs BOTH and couples them:
#
#   1. the leader engine scores its five defences against a modelled
#      attacking chaser and names its winner;
#   2. that winner sets the posture the chaser engine is then scored
#      against (COUNTER-DEPLOY / REACTIVE DEFENSE -> 'defensive_boost',
#      anything else -> 'balanced'), so the attack pick has to beat the
#      defence the OPPONENT's engine actually recommends rather than a
#      posture a human toggled;
#   3. the seat picks whose call is surfaced as THE CALL — but both engines
#      always run, and every policy from both is returned, so the whole
#      decision space stays inspectable.
# ---------------------------------------------------------------------------

# Merged-call latency budget: two full engine evaluations (the leader walk
# set PLUS the chaser walk set) rather than one.  Disclosed and re-stated
# from measurement, the same honesty rule as LATENCY_BUDGET_MS — the payload
# reports its own latency_ms and the UI never claims faster than delivered.
#
# MEASURED (2026-09-10, this machine), fresh walk sets: ~1.2-1.3 s warm per
# coupled call (leader engine ~0.6 s + chaser engine ~0.65 s, 8+5 walks
# between them).  The engine cache does not change the budget — it covers
# the REPEATED call (same state / seat flip), which is a different service
# than the first evaluation of a state, and the payload's engine_cache
# block discloses which side was reused.  Re-measure on presentation
# hardware before quoting a number on stage.
CALL_LATENCY_BUDGET_MS = 6000.0

# Which leader defence forces the chaser engine to score against a reacting
# leader.  The posture lever only has two settings (see
# overtake_inference.LIVE_LEADER_DEFENSE_POSTURES), so the two defensive
# postures that actually spend the leader's store map onto the one that
# makes the leader faster.  The value-lever defences (banking) leave the
# chaser's Balanced baseline honest.
_DEFENCE_TO_POSTURE = {
    "COUNTER-DEPLOY": "defensive_boost",
    "REACTIVE DEFENSE": "defensive_boost",
}


def evaluate_call(
        leader_code: str, chaser_code: str, track_name: str,
        start_lap: int, race_length: int, gap_before_s: float,
        leader_tyre_compound: str = "Medium",
        chaser_tyre_compound: str = "Medium",
        leader_tyre_age: float = 10.0, chaser_tyre_age: float = 10.0,
        year: int | None = None,
        battery_pct: float | None = None,
        threat_battery_pct: float | None = None,
        reserve_target_mj: float | None = None,
        perspective: str = "chaser",
        chaser_shape=None, leader_shape=None,
        race_event: str = "green",
        traffic_level: str = "Clear") -> dict[str, Any]:
    """One merged call: both engines run, coupled, one decision out.

    ``battery_pct`` is OUR car's battery (the seat's own car) and
    ``threat_battery_pct`` the other car's — the caller does not have to
    know which engine consumes which, the seat decides that mapping.

    ``chaser_shape`` / ``leader_shape`` are the strategist's per-sector
    slider banks (Energy Sandbox vocabulary, MJ/lap per sector, clamped
    to +-8.5).  The chaser bank rides every chaser-engine walk and the
    leader bank every leader-engine walk; each engine's recommendation
    carries the shapes it was scored under, and the cache keys on the
    normalised vectors so a changed slider bank can never serve a stale
    call.

    ``race_event`` ('green', 'vsc', 'safety_car') and ``traffic_level``
    ('Clear', 'Light', 'Heavy') inject circuit-specific pit loss times,
    Safety Car intervention likelihood, and fresh-tyre undercut window math.

    Returns the single ``final_call`` card for the chosen seat (carrying the
    opponent engine's answer and the projected race window), every policy
    from BOTH engines tagged with its ``seat``, and each engine's own
    recommendation for drill-down.  Raises ValueError for an unusable
    state, exactly like the two underlying engines.
    """
    t_start = time.perf_counter()
    seat = str(perspective or "chaser").strip().lower()
    if seat not in ("chaser", "leader"):
        raise ValueError("perspective must be 'chaser' or 'leader'")

    # Seat -> engine inputs.  From the chaser's seat OUR battery is the
    # chaser's; from the leader's seat OUR battery is the leader's and the
    # other car is the attacking threat.
    if seat == "leader":
        leader_batt, chaser_batt = battery_pct, threat_battery_pct
    else:
        leader_batt, chaser_batt = None, battery_pct
    chaser_shape_vec = _shape_vector(chaser_shape)
    leader_shape_vec = _shape_vector(leader_shape)

    common = dict(
        leader_code=leader_code, chaser_code=chaser_code,
        track_name=track_name, start_lap=start_lap,
        race_length=race_length, gap_before_s=gap_before_s,
        leader_tyre_compound=leader_tyre_compound,
        chaser_tyre_compound=chaser_tyre_compound,
        leader_tyre_age=leader_tyre_age, chaser_tyre_age=chaser_tyre_age,
        year=year, reserve_target_mj=reserve_target_mj,
        race_event=race_event, traffic_level=traffic_level,
    )
    reserve = (RESERVE_TARGET_MJ if reserve_target_mj is None
               else float(reserve_target_mj))

    # Engine-level cache: a repeated call on the same race state (and the
    # seat flip at default batteries — the leader engine maps None onto
    # ERS_DEFAULT_START_PCT) reuses a stored engine evaluation instead of
    # re-simulating its walks.  Cached payloads are deep copies, so a hit
    # cannot leak mutations into the caller's dict; each engine's stored
    # latency_ms is kept but the payload discloses the reuse via
    # engine_cache (a cached latency is never presented as fresh walk
    # time).
    leader_key = _engine_cache_key(
        "leader", leader_code=leader_code, chaser_code=chaser_code,
        track_name=track_name, start_lap=start_lap,
        race_length=race_length, gap_before_s=gap_before_s,
        leader_tyre_compound=leader_tyre_compound,
        chaser_tyre_compound=chaser_tyre_compound,
        leader_tyre_age=leader_tyre_age, chaser_tyre_age=chaser_tyre_age,
        year=year, leader_batt=leader_batt, chaser_batt=chaser_batt,
        reserve=reserve, leader_shape=leader_shape_vec,
        threat_shape=chaser_shape_vec,
        race_event=race_event, traffic_level=traffic_level)
    leader_out = _engine_cache_get(leader_key)
    leader_cached = leader_out is not None
    if leader_out is None:
        leader_out = evaluate_leader_policies(
            **common, leader_battery_pct=leader_batt,
            chaser_battery_pct=chaser_batt,
            leader_shape=leader_shape_vec, threat_shape=chaser_shape_vec)
        _engine_cache_put(leader_key, leader_out)

    # 2. COUPLE the engines: the defence the leader engine recommends sets
    #    the posture the attack engine is scored against.  The coupling is
    #    derived from the (possibly cached) leader payload — the cached
    #    decision IS the decision, so the posture it produces is stable.
    best_defence = leader_out["recommendation"]["action_card"]["action"]
    posture = _DEFENCE_TO_POSTURE.get(best_defence, "balanced")
    chaser_key = _engine_cache_key(
        "chaser", leader_code=leader_code, chaser_code=chaser_code,
        track_name=track_name, start_lap=start_lap,
        race_length=race_length, gap_before_s=gap_before_s,
        leader_tyre_compound=leader_tyre_compound,
        chaser_tyre_compound=chaser_tyre_compound,
        leader_tyre_age=leader_tyre_age, chaser_tyre_age=chaser_tyre_age,
        year=year, leader_batt=leader_batt, chaser_batt=chaser_batt,
        reserve=reserve, posture=posture, chaser_shape=chaser_shape_vec,
        race_event=race_event, traffic_level=traffic_level)
    chaser_out = _engine_cache_get(chaser_key)
    chaser_cached = chaser_out is not None
    if chaser_out is None:
        chaser_out = evaluate_tactical_policies(
            **common, chaser_battery_pct=chaser_batt, leader_posture=posture,
            chaser_shape=chaser_shape_vec)
        _engine_cache_put(chaser_key, chaser_out)

    chaser_card = chaser_out["recommendation"]["action_card"]
    leader_card = leader_out["recommendation"]["action_card"]

    if seat == "leader":
        card, opp_card = leader_card, chaser_card
        opp_seat, opp_engine = "chaser", chaser_out["recommendation"]
        own_engine = leader_out["recommendation"]
    else:
        card, opp_card = chaser_card, leader_card
        opp_seat, opp_engine = "leader", leader_out["recommendation"]
        own_engine = chaser_out["recommendation"]

    # The race-call projection is the simulator's own no-lever walk — the
    # timing evidence behind the manoeuvre, not a second opinion.
    base = chaser_out["baseline"]

    final_call = {
        "seat": seat,
        "action": card.get("action"),
        "reason": card.get("reason"),
        "deploy_lap": card.get("deploy_lap"),
        "overtake_probability": chaser_card.get("overtake_probability"),
        "threat_probability": leader_card.get("threat_probability"),
        "hold_probability": leader_card.get("hold_probability"),
        "energy_cost_mj": card.get("energy_cost_mj"),
        "expected_finish_delta_s": card.get("expected_finish_delta_s"),
        "battery_margin_pct": card.get("battery_margin_pct"),
        "battery_margin_worst_pct": card.get("battery_margin_worst_pct"),
        "soc_band_pct": card.get("soc_band_pct"),
        "confidence": card.get("confidence"),
        # The surfaced engine's own decision margin (s to the next DISTINCT
        # score) — the number the confidence is built from, disclosed on the
        # card so a 0-confidence call can be audited without the payload.
        "engine_margin_s": own_engine.get("decision_margin_s"),
        "feasible": card.get("feasible"),
        "ers_shape": (list(chaser_shape_vec)
                      if chaser_shape_vec is not None else None),
        "leader_ers_shape": (list(leader_shape_vec)
                             if leader_shape_vec is not None else None),
        "opponent": {
            "seat": opp_seat,
            "action": opp_card.get("action"),
            "reason": opp_card.get("reason"),
            "confidence": opp_card.get("confidence"),
        },
        "projection": {
            "verdict": base.get("verdict"),
            "pass_lap": base.get("pass_lap"),
            "cumulative_probability": base.get("cumulative_probability"),
            "avg_pace_gap_s": base.get("avg_pace_gap_s"),
            "projected_final_gap_s": base.get("projected_final_gap_s"),
        },
        "engine_coupling": {
            "our_engine": ("leader" if seat == "leader" else "chaser"),
            "our_engine_runner_up": own_engine.get("runner_up"),
            "opponent_engine_recommendation": best_defence,
            "chaser_posture_used": posture,
            "note": (
                "The attack policies are scored against the defence the "
                f"opponent's own engine recommends ({best_defence} -> "
                f"'{posture}' posture), not a hand-set toggle. "
                "Changing the seat re-uses the same coupled decision."),
        },
    }

    # Every policy from BOTH engines, tagged with the seat it belongs to.
    rows = ([dict(r, seat="chaser") for r in chaser_out["policies"]]
            + [dict(r, seat="leader") for r in leader_out["policies"]])

    elapsed_ms = (time.perf_counter() - t_start) * 1000.0
    return {
        "perspective": seat,
        "perspective_label": ("LEADER — defending" if seat == "leader"
                              else "CHASER — attacking"),
        "state": chaser_out["state"],
        "leader_state": leader_out["state"],
        "final_call": final_call,
        "policies": rows,
        "chaser": {
            "recommendation": chaser_out["recommendation"],
            "baseline": chaser_out["baseline"],
            "latency_ms": chaser_out["latency_ms"],
        },
        "leader": {
            "recommendation": leader_out["recommendation"],
            "baseline": leader_out["baseline"],
            "latency_ms": leader_out["latency_ms"],
        },
        "scoring_constants": {
            "chaser": chaser_out["scoring_constants"],
            "leader": leader_out["scoring_constants"],
        },
        "latency_ms": round(elapsed_ms, 1),
        "latency_budget_ms": CALL_LATENCY_BUDGET_MS,
        "pit_analysis": chaser_out.get("pit_analysis"),
        "engine_cache": {
            "leader_reused": leader_cached,
            "chaser_reused": chaser_cached,
            "note": ("reused the stored engine evaluation for the repeated "
                     "call — latency_ms reflects the cache hit, not a fresh "
                     "simulation"
                     if (leader_cached or chaser_cached) else
                     "fresh simulation — both engines walked this state"),
        },
    }


# ---------------------------------------------------------------------------
# CLI (smoke/demo): run the engine on the audit's canonical state.
# ---------------------------------------------------------------------------

def main() -> int:
    import argparse
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--leader", default="VER")
    ap.add_argument("--chaser", default="HAM")
    ap.add_argument("--track", default="Autodromo Nazionale di Monza")
    ap.add_argument("--start-lap", type=int, default=20)
    ap.add_argument("--race-length", type=int, default=50)
    ap.add_argument("--gap", type=float, default=0.8)
    ap.add_argument("--battery", type=float, default=None,
                    help="chaser battery override, percent")
    ap.add_argument("--leader-posture", default="balanced",
                    choices=list(oi.LIVE_LEADER_DEFENSE_POSTURES),
                    help="leader stance: balanced or defensive_boost")
    ap.add_argument("--year", type=int, default=2026)
    args = ap.parse_args()
    out = evaluate_tactical_policies(
        leader_code=args.leader, chaser_code=args.chaser,
        track_name=args.track, start_lap=args.start_lap,
        race_length=args.race_length, gap_before_s=args.gap,
        year=args.year, chaser_battery_pct=args.battery,
        leader_posture=args.leader_posture)
    import json
    print(json.dumps(out, indent=1))
    card = out["recommendation"]["action_card"]
    print(f"\nACTION: {card['action']}  (confidence {card['confidence']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
