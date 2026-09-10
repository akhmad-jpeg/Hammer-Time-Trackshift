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
               leader_posture="balanced"):
    """One simulate_live_call pass at a given net lever; returns (sim, ms)."""
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
        chaser_ers_deltas=_lever(net_mj) if net_mj else None,
        chaser_battery_pct=battery_pct,
        leader_posture=leader_posture,
    )
    return sim, (time.perf_counter() - t0) * 1000.0


def _phase2_gap(sim1, phase1_laps):
    """Gap (s) the phase-1 walk leaves the pair at, for phase 2 to inherit.

    simulate_live_call already walks to race_length; for the stalk policy we
    re-run from the deploy lap with the gap its phase-1 projection reached
    at that lap (and tyre ages ticked accordingly), so phase 2 starts where
    phase 1 genuinely ended rather than from a hand-waved state.
    """
    laps = sim1.get("laps") or []
    row = None
    for l in laps:
        if l["lap"] <= phase1_laps:
            row = l
    if row is None:
        row = laps[0] if laps else None
    if row is None:
        return None, None
    # Age tick: phase 2 starts one lap after phase 1's last walked lap.
    age0 = sim1["meta"]["tyres"]["chaser"]["age"]
    ticked = int(age0 + (row["lap"] - sim1["meta"]["start_lap"]) + 1)
    return max(0.05, float(row["gap_before_s"])), ticked


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
) -> dict[str, Any]:
    """Evaluate all five policies against one race state.

    ``leader_posture`` is the game-theoretic lever: 'balanced' assumes the
    leader runs its own race; 'defensive_boost' has the leader counter-
    deploy (its own 4 MJ store, same floor) so every closing-based policy
    is scored against an opponent that actually reacts.

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

    # ---- Baseline (no lever): the reference every policy is scored against.
    base_sim, _ = _run_phase(leader_code, chaser_code, track_name, start_lap,
                             race_length, gap_before_s, tyres, year,
                             batt_pct, 0.0, leader_posture)
    base = base_sim["summary"]
    base_call = base_sim["call"]

    rows = []
    for pol in POLICIES:
        deploy_lap = start_lap + pol["deploy_offset"]
        net1, net2 = pol["phase1_mj"], pol["phase2_mj"]
        batt_start_phase2_pct = batt_pct

        if pol["deploy_offset"] == 0:
            sim, ms = _run_phase(leader_code, chaser_code, track_name,
                                 start_lap, race_length, gap_before_s,
                                 tyres, year, batt_pct, net1, leader_posture)
            call, summ = sim["call"], sim["summary"]
            soc_end_pct = summ.get("chaser_soc_end_pct")
            min_soc_pct = min((l.get("chaser_soc_pct", 100.0)
                               for l in sim["laps"]), default=None)
            deployed = summ.get("ers_deployed_mj", 0.0)
            banked = summ.get("ers_banked_mj", 0.0)
            energy_limited = summ.get("ers_energy_limited_laps", 0)
            ms_total = ms
        else:
            # Two-phase policy: bank until the deploy lap, then strike.
            phase1_laps = deploy_lap  # walk phase 1 UP TO (and incl.) this lap
            sim1, ms1 = _run_phase(leader_code, chaser_code, track_name,
                                   start_lap, deploy_lap, gap_before_s,
                                   tyres, year, batt_pct, net1, leader_posture)
            gap2, age2 = _phase2_gap(sim1, phase1_laps)
            # Battery carried into phase 2 = the phase-1 end SOC.
            soc1 = sim1.get("summary", {}).get("chaser_soc_end_pct")
            if soc1 is not None:
                batt_start_phase2_pct = soc1
            tyres2 = {
                "leader": {"compound": leader_tyre_compound,
                           "age": age2 or leader_tyre_age + 1},
                "chaser": {"compound": chaser_tyre_compound,
                           "age": age2 or chaser_tyre_age + 1},
            }
            sim2, ms2 = _run_phase(leader_code, chaser_code, track_name,
                                   deploy_lap + 1, race_length, gap2,
                                   tyres2, year, batt_start_phase2_pct, net2,
                                   leader_posture)
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
        reserve_breach = (min_soc_pct is not None
                          and min_soc_pct < _battery_pct(reserve) - 0.05)

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

        # Tyre cost: extra health burned vs the Balanced baseline for the
        # laps the policy spends on a positive lever.
        active_push_laps = len(sim.get("laps") or []) \
            if pol["deploy_offset"] == 0 else len(sim1.get("laps") or [])
        wear_delta = max(0.0, _est_extra_wear(net1, active_push_laps))
        wear_cost = LAMBDA_WEAR * wear_delta
        cliff_cost = CLIFF_RISK_S * (
            1.0 if (min_soc_pct is not None and drained
                    and pol["deploy_offset"] == 0
                    and pol["phase1_mj"] > 0) else 0.0)

        fail_prob = max(0.0, 1.0 - cum) if pass_lap else 0.0
        risk_cost = LAMBDA_RISK * fail_prob * DIRTY_AIR_PENALTY_S

        deferral = (pass_lap - start_lap) if pass_lap else horizon
        latency_cost = LAMBDA_LATENCY * deferral

        # Hard constraint: reserve breach or a pass that only converts by
        # draining the store to its floor can never be recommended.
        infeasible = reserve_breach or (pass_lap and drained
                                        and net1 > 0)
        score = (batt_cost + wear_cost + cliff_cost + risk_cost
                 + latency_cost) - (0.0 if infeasible else pass_gain)

        rows.append({
            "policy": pol["name"],
            "why": pol["why"],
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
                 if (pass_lap and drained and net1 > 0) else None)),
            "phase_laps_ms": round(ms_total, 1),
        })

    feasible = [r for r in rows if r["feasible"]]
    ranked = sorted(feasible or rows, key=lambda r: r["score_s"])
    best, second = (ranked[0], ranked[1] if len(ranked) > 1 else None)
    margin = ((second["score_s"] - best["score_s"])
              if second is not None else CONFIDENCE_SPAN_S)
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
    }


def _est_extra_wear(net_mj: float, laps: int) -> float:
    """Extra tyre health-% burned by a positive lever vs Balanced.

    Pushing spends deploy MJ that Balanced would bank; the tyre cost is
    modelled from the wear meter itself: full lever ≈ double the Balanced
    heat load on those laps.  Health-% per lap at full lever ≈ the compound
    loss rate (via tyre_health's linear model) — reused here so the engine
    does not invent a second wear model.
    """
    if net_mj <= 0 or laps <= 0:
        return 0.0
    from tyre_degradation import tyre_health
    h0 = tyre_health("Medium", 0)
    h1 = tyre_health("Medium", 1)
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
                 lever, preset=None):
    """One leader-side simulate_live_call pass; returns (sim, ms)."""
    kwargs = {}
    if preset:
        kwargs["leader_posture"] = preset
    elif lever:
        kwargs["leader_ers_deltas"] = _lever(lever)
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


def evaluate_leader_policies(
        leader_code: str, chaser_code: str, track_name: str,
        start_lap: int, race_length: int, gap_before_s: float,
        leader_tyre_compound: str = "Medium",
        chaser_tyre_compound: str = "Medium",
        leader_tyre_age: float = 10.0, chaser_tyre_age: float = 10.0,
        year: int | None = None, chaser_battery_pct: float | None = None,
        leader_battery_pct: float | None = None,
        reserve_target_mj: float | None = None,
) -> dict[str, Any]:
    """Evaluate five DEFENCE policies for the leader against one threat.

    The chaser is modelled as attacking (LEADER_THREAT_CHASER_ERS posture);
    each policy is the leader's answer.  Rows are scored on the probability
    the position SURVIVES, priced against the leader's own battery and
    tyre costs, with the same structural constraints as the chaser engine
    (reserve breach / floor-drain => INFEASIBLE, never recommended).

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

    rows = []
    base_summ = None
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
                preset=preset)
            summ = sim["summary"]
            ms_total = ms
        else:
            # Two-phase: concede + refill until the deploy lap, then
            # counter-deploy.  Phase 2 inherits the REAL phase-1 gap/SOC.
            sim1, ms1 = _leader_walk(
                leader_code, chaser_code, track_name, start_lap, deploy_lap,
                gap_before_s, tyres, year, cbatt, lbatt, lever)
            gap2, age2 = _phase2_gap(sim1, deploy_lap)
            soc1 = sim1.get("summary", {}).get("leader_soc_end_pct")
            batt2 = soc1 if soc1 is not None else lbatt
            tyres2 = {
                "leader": {"compound": leader_tyre_compound,
                           "age": age2 or leader_tyre_age + 1},
                "chaser": {"compound": chaser_tyre_compound,
                           "age": age2 or chaser_tyre_age + 1},
            }
            sim2, ms2 = _leader_walk(
                leader_code, chaser_code, track_name, deploy_lap + 1,
                race_length, gap2, tyres2, year, cbatt, batt2,
                pol.get("strike_lever") or 0.0)
            summ = sim2["summary"]
            # Stitch phase-1 SOC rows in for the constraint check.
            stitched = [l["leader_soc_pct"] for l in sim1["laps"]
                        if l.get("leader_soc_pct") is not None]
            summ = dict(summ)
            summ["_phase1_soc_rows"] = stitched
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
        deployed = summ.get("leader_defense", {}) or {}
        deployed_mj = float(deployed.get("deployed_mj", 0.0))
        banked_mj = float(deployed.get("banked_mj", 0.0))
        energy_limited = int(deployed.get("energy_limited_laps", 0))
        lever_on = deployed_mj > 0 or banked_mj > 0

        drained = (min_soc_pct is not None
                   and min_soc_pct <= LIVE_ATTACK_MIN_SOC_PCT + 0.05)
        reserve_breach = (min_soc_pct is not None
                          and min_soc_pct < _battery_pct(reserve) - 0.05)

        # Worst-case battery prices the cost terms (same rule as chaser).
        soc_end_lo = (max(0.0, soc_end_pct - band["band_pct"])
                      if soc_end_pct is not None else None)
        soc_end_mj = (soc_end_lo / 100.0 * ERS_STORE_MJ
                      if soc_end_lo is not None else None)
        soc_deficit = (max(0.0, reserve - soc_end_mj)
                       if soc_end_mj is not None else 0.0)
        batt_cost = LAMBDA_BATT * soc_deficit

        # Tyre cost of counter-deploying (push laps only, vs baseline).
        push_laps = len(sim.get("laps") or [])
        wear_cost = LAMBDA_WEAR * max(
            0.0, _est_extra_wear(lever if preset is None else 0.0,
                                 push_laps))

        infeasible = reserve_breach or drained
        # Score: the attack converting is the leader's loss (a walk that
        # never converts leaves only its residual cum as risk); every lap
        # the defence buys earns HELD_LAP_CREDIT_S.  Lower is better.
        # Infeasible rows keep their honest score but are barred from
        # winning (and flagged on the card if EVERY policy is barred).
        position_risk = (LOST_POSITION_S if converted
                         else p_attack * LOST_POSITION_S)
        hold_credit = HELD_LAP_CREDIT_S * laps_held
        score = position_risk + batt_cost + wear_cost - hold_credit

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
        })
        if pol["name"] == "HOLD & MANAGE":
            base_summ = summ

    feasible = [r for r in rows if r["feasible"]]
    ranked = sorted(feasible or rows, key=lambda r: r["score_s"])
    best, second = (ranked[0], ranked[1] if len(ranked) > 1 else None)
    margin = ((second["score_s"] - best["score_s"])
              if second is not None else CONFIDENCE_SPAN_S)
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
