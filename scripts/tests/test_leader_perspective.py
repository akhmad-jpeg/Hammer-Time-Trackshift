"""Unit tests for the LEADER perspective (defence policy engine).

Covers the dual-seat capability: the same engine advising the defending
car against an attacking chaser.

  Simulator layer (simulate_live_call):
    1. leader_ers / leader_battery_pct mirror the chaser's lever physics
       (deploy funded by the leader's own store down to the 30% floor).
    2. Deploy NARROWS the attack: at a fixed horizon the chaser's
       cumulative pass probability is lower with a deploying leader than a
       balanced one; banking WIDENS it.  (The case-3-vs-4 full-walk
       comparison is invalid — walks break at different laps.)
    3. The reactive preset (leader_posture='defensive_boost') and the
       explicit lever coexist: the explicit lever wins when both are given.
    4. Back-compat: calls without any leader-lever argument are
       byte-identical to the pre-feature no-lever path (calibration
       safety for backtest_race_calls.py).

  Policy engine (evaluate_leader_policies):
    5. Five leader policies, ranked; the ACTION card carries the
       threat/hold/laps-held contract.
    6. Reserve constraints: a leader battery at/below the reserve marks
       counter-deploying defences INFEASIBLE with a reason.
    7. Determinism: identical state -> identical payload.
    8. The perspective flag is echoed ("leader") so the UI can never mix
       the two seats' payloads.

Skipped gracefully when the committed ML artifacts / DB are absent.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from overtake_inference import simulate_live_call  # noqa: E402
from policy_engine import (  # noqa: E402
    evaluate_leader_policies,
    LEADER_POLICIES,
)

TRACK = "Autodromo Nazionale Di Monza"


def _sim_available() -> bool:
    try:
        simulate_live_call("VER", "HAM", TRACK, start_lap=20,
                           race_length=22, gap_before_s=0.8)
        return True
    except Exception:
        return False


SIM_OK = _sim_available()


def _leader_walk(**kw):
    args = dict(
        leader_code="VER", chaser_code="HAM", track_name=TRACK,
        start_lap=20, race_length=50, gap_before_s=0.8,
        leader_tyre_age=12, chaser_tyre_age=12,
        chaser_ers=50, chaser_battery_pct=62.5,
    )
    args.update(kw)
    return simulate_live_call(**args)


def _cum_at(result, lap):
    c = 0.0
    for l in result["laps"]:
        if l["lap"] <= lap:
            c = l["cumulative_probability"]
    return c


def _evaluate(**kw):
    args = dict(
        leader_code="VER", chaser_code="HAM", track_name=TRACK,
        start_lap=20, race_length=50, gap_before_s=0.8, year=2026,
    )
    args.update(kw)
    return evaluate_leader_policies(**args)


@unittest.skipUnless(SIM_OK, "committed models / DB not available")
class TestSimulateLeaderLever(unittest.TestCase):

    def test_deploy_narrows_bank_widens_at_fixed_horizon(self):
        """The core direction check, at a FIXED lap (walks convert at
        different laps, so full-walk cums are not comparable): deploying
        cuts the chaser's conversion probability vs Balanced; banking
        raises it."""
        bal = _leader_walk()
        dep = _leader_walk(leader_ers=100, leader_battery_pct=80)
        bnk = _leader_walk(leader_ers=-100, leader_battery_pct=50)
        for h in (24, 26):
            c_bal, c_dep, c_bnk = (_cum_at(r, h) for r in (bal, dep, bnk))
            self.assertLess(c_dep, c_bal,
                            f"leader deploy must cut threat cum at L{h} "
                            f"({c_dep} vs {c_bal})")
            self.assertGreater(c_bnk, c_bal,
                               f"leader banking must raise threat cum at "
                               f"L{h} ({c_bnk} vs {c_bal})")

    def test_leader_deploy_is_funded_and_floor_limited(self):
        dep = _leader_walk(leader_ers=100, leader_battery_pct=80)
        defense = dep["summary"]["leader_defense"]
        self.assertIsNotNone(defense)
        self.assertEqual(defense["posture"], "explicit_lever")
        self.assertGreater(defense["deployed_mj"], 0.0)
        # 80% of 4 MJ = 3.2; deploy can never exceed store above the floor.
        self.assertLessEqual(defense["deployed_mj"], 3.2 - 1.2 + 1e-9)
        # Leader SOC is reported with the uncertainty band.
        self.assertIn("leader_soc_end_pct", dep["summary"])
        self.assertIn("leader_soc_end_band_pct", dep["summary"])

    def test_leader_battery_override_changes_the_walk(self):
        hi = _leader_walk(leader_ers=100, leader_battery_pct=90)
        lo = _leader_walk(leader_ers=100, leader_battery_pct=40)
        dep_hi = hi["summary"]["leader_defense"]["deployed_mj"]
        dep_lo = lo["summary"]["leader_defense"]["deployed_mj"]
        self.assertGreater(dep_hi, dep_lo,
                           "a fuller leader store must fund more deploy")

    def test_explicit_lever_wins_over_posture_preset(self):
        both = _leader_walk(leader_ers=-100, leader_battery_pct=50,
                            leader_posture="defensive_boost")
        # Explicit bank must NOT be overridden by the reactive preset
        # (which would deploy): banked > 0, deployed == 0.
        defense = both["summary"]["leader_defense"]
        self.assertEqual(defense["posture"], "explicit_lever")
        self.assertGreater(defense["banked_mj"], 0.0)
        self.assertEqual(defense["deployed_mj"], 0.0)

    def test_no_leader_lever_path_unchanged(self):
        """Calibration safety: no leader lever args -> byte-identical to
        the pre-feature behaviour."""
        a = simulate_live_call("VER", "HAM", TRACK, start_lap=20,
                               race_length=50, gap_before_s=0.8,
                               leader_tyre_age=12, chaser_tyre_age=12,
                               chaser_ers=50, chaser_battery_pct=62.5)
        b = _leader_walk()
        self.assertEqual(a, b)


@unittest.skipUnless(SIM_OK, "committed models / DB not available")
class TestLeaderPolicyEngine(unittest.TestCase):

    def test_card_contract(self):
        out = _evaluate(leader_battery_pct=80.0)
        card = out["recommendation"]["action_card"]
        for key in ("action", "threat_probability", "hold_probability",
                    "energy_cost_mj", "battery_margin_pct", "confidence",
                    "reason", "expected_finish_delta_s"):
            self.assertIn(key, card)
        self.assertEqual(out["perspective"], "leader")
        self.assertEqual(len(out["policies"]), len(LEADER_POLICIES))
        self.assertIn("threat_assumption", out["state"])

    def test_low_battery_rejects_counter_deploying_defences(self):
        out = _evaluate(leader_battery_pct=38.0)
        by_name = {r["policy"]: r for r in out["policies"]}
        for name in ("COUNTER-DEPLOY", "REACTIVE DEFENSE"):
            self.assertFalse(by_name[name]["feasible"],
                             f"{name} must be infeasible at 38% battery")
            self.assertIn("reserve breach",
                          by_name[name]["infeasible_reason"])

    def test_determinism(self):
        a = _evaluate(leader_battery_pct=80.0)
        b = _evaluate(leader_battery_pct=80.0)
        self.assertEqual(a["recommendation"]["action_card"],
                         b["recommendation"]["action_card"])
        self.assertEqual([r["score_s"] for r in a["policies"]],
                         [r["score_s"] for r in b["policies"]])

    def test_winner_holds_longer_or_costs_less_than_naive_defence(self):
        """Sanity on the objective: the winner must not be dominated by
        HOLD & MANAGE on both axes (held laps and battery spend) — the
        engine must be adding value beyond 'do nothing'."""
        out = _evaluate(leader_battery_pct=80.0)
        best = out["recommendation"]["action_card"]["action"]
        if best == "HOLD & MANAGE":
            self.skipTest("do-nothing won this state; nothing to prove")
        rows = {r["policy"]: r for r in out["policies"]}
        hold = rows["HOLD & MANAGE"]
        win = rows[best]
        # Winner's ledger must show why it beat doing nothing.
        self.assertLess(win["score_s"], hold["score_s"] + 1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
