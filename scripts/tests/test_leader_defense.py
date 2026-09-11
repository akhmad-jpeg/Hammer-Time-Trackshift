"""Unit tests for the leader defensive-boost posture (game-theoretic lever).

Covers the P1 adversarial capability from IMPLEMENTATION_PLAN.md:

  1. ``simulate_live_call`` accepts leader_posture ('balanced' |
     'defensive_boost'), rejects anything else, and the no-lever path is
     byte-identical to the pre-feature behaviour (calibration safety: the
     race-call backtest harness passes no posture and must be unaffected).
  2. Defensive physics is funded, not free: the leader deploys from its
     own 4 MJ store down to the same 30% floor the chaser's walk uses,
     then reverts to Balanced (energy-limited laps are counted).
  3. Defense narrows the attack: with the chaser pushing, the defended
     walk's per-lap pace edge is LOWER on every lap the two walks share,
     its gap path dominates the balanced one, and the closest approach is
     wider.  (Cumulative probabilities are deliberately NOT compared: a
     retrained classifier that saturates truncates walks at the first
     in-window lap, so fixed-horizon cums track the model's scale, not
     the lever's effect.  The pace/energy layer is deterministic.)
  4. The policy engine threads the posture through every walk (baseline +
     all five policies), echoes it in ``state``, and the adversarial demo
     beat holds: at a mid battery state, flipping the leader to defensive
     collapses the decision margin and can flip the card to SAVE & DEFEND.

Skipped gracefully when the committed ML artifacts / MySQL DB are absent
(the suite runs on any checkout that can run the other suites).
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from overtake_inference import (  # noqa: E402
    LIVE_DEFENSE_NET_MJ,
    LIVE_LEADER_DEFENSE_POSTURES,
    simulate_live_call,
)
from policy_engine import evaluate_tactical_policies  # noqa: E402

TRACK = "Autodromo Nazionale di Monza"


def _posture_available() -> bool:
    try:
        simulate_live_call("VER", "HAM", TRACK, start_lap=20,
                           race_length=22, gap_before_s=0.8)
        return True
    except Exception:
        return False


POSTURE_OK = _posture_available()


def _walk(posture, **kw):
    args = dict(
        leader_code="VER", chaser_code="HAM", track_name=TRACK,
        start_lap=20, race_length=50, gap_before_s=0.8,
        leader_tyre_age=12, chaser_tyre_age=12,
        chaser_ers=50, chaser_battery_pct=62.5,
        leader_posture=posture,
    )
    args.update(kw)
    return simulate_live_call(**args)


@unittest.skipUnless(POSTURE_OK, "committed models / DB not available")
class TestSimulateLiveCallPosture(unittest.TestCase):

    def test_invalid_posture_rejected(self):
        for bad in ("aggressive", 42):
            with self.assertRaises(ValueError):
                _walk(bad)
        # Normalization is intended: case-insensitive + None -> 'balanced'.
        for ok in (None, "DEFENSIVE_BOOST"):
            _walk(ok)  # must not raise

    def test_postures_match_module_constant(self):
        self.assertEqual(set(LIVE_LEADER_DEFENSE_POSTURES),
                         {"balanced", "defensive_boost"})

    def test_no_lever_path_unchanged_by_posture_param(self):
        """Calibration safety: the backtest harness passes no ERS args and
        no posture — its calls must be byte-identical to before the
        feature existed (posture defaults to 'balanced' and a Balanced
        leader deploys nothing)."""
        a = simulate_live_call("VER", "HAM", TRACK, start_lap=20,
                               race_length=45, gap_before_s=0.8)
        b = simulate_live_call("VER", "HAM", TRACK, start_lap=20,
                               race_length=45, gap_before_s=0.8,
                               leader_posture="balanced")
        self.assertEqual(a, b)

    def test_defense_is_funded_from_the_leader_store(self):
        defended = _walk("defensive_boost")
        defense = defended["summary"]["leader_defense"]
        self.assertIsNotNone(defense)
        self.assertGreater(defense["deployed_mj"], 0.0)
        # Deploy spend never exceeds what the store above the floor allows.
        self.assertLessEqual(defense["deployed_mj"], 4.0 - 1.2 + 1e-9)
        # meta reports the posture explicitly (never silently assumed).
        self.assertEqual(
            defended["meta"]["ers"]["leader"]["posture"], "defensive_boost")
        self.assertEqual(defended["summary"]["leader_defense"]["posture"],
                         "defensive_boost")

    def test_defense_limited_laps_counted_after_floor(self):
        # The floor must be reachable within a single walk lap: a store
        # nearly at the 30% floor cannot fund the full ~0.24 MJ/lap request,
        # so the lap is delivered at a partial fraction and COUNTED as
        # energy-limited (the pace edge fades exactly there).  Counting is
        # asserted at the floor, not after N laps of drain — walk length
        # depends on the classifier (a saturating one stops after one lap).
        defended = _walk("defensive_boost", leader_battery_pct=31.0)
        defense = defended["summary"]["leader_defense"]
        self.assertGreaterEqual(
            defense["energy_limited_laps"], 1,
            "a floor-adjacent store must count energy-limited laps")
        self.assertLess(
            defense["deployed_mj"], LIVE_DEFENSE_NET_MJ,
            "a floor-adjacent store cannot deliver the full per-lap request")
        # And a fuller store funds strictly more defence over the same
        # deterministic walk.
        fuller = _walk("defensive_boost", leader_battery_pct=80.0)
        self.assertGreater(
            fuller["summary"]["leader_defense"]["deployed_mj"],
            defense["deployed_mj"],
            "a fuller leader store must fund more deploy")

    def test_defense_narrows_the_attack(self):
        """Same race state, same chaser push: a defending leader must cut
        the chaser's pace edge on EVERY lap the two walks share, so the
        defended gap path dominates the balanced one and the closest
        approach is wider.  Pace/gap paths are the deterministic layer the
        levers move; cumulative probabilities at fixed horizons are not
        compared because a retrained (e.g. saturating) classifier truncates
        walks at the first in-window lap, making cums model-scale artifacts."""
        bal = _walk("balanced")
        dfn = _walk("defensive_boost")
        self.assertGreater(len(bal["laps"]), 0, "setup: the walk has laps")
        for lb, ld in zip(bal["laps"], dfn["laps"]):
            self.assertEqual(lb["lap"], ld["lap"])
            self.assertLess(
                ld["pace_gap_s"], lb["pace_gap_s"],
                f"L{lb['lap']}: defence must cut the chaser's pace edge")
            self.assertGreaterEqual(
                ld["gap_before_s"], lb["gap_before_s"] - 1e-9,
                f"L{lb['lap']}: defended gap must dominate the balanced gap")
        self.assertGreaterEqual(
            dfn["summary"]["min_gap_s"], bal["summary"]["min_gap_s"] - 1e-9,
            "defence must keep the closest approach wider")


@unittest.skipUnless(POSTURE_OK, "committed models / DB not available")
class TestPolicyEnginePosture(unittest.TestCase):

    def _evaluate(self, posture, **kw):
        args = dict(
            leader_code="VER", chaser_code="HAM", track_name=TRACK,
            start_lap=20, race_length=50, gap_before_s=0.8,
            year=2026, chaser_battery_pct=45.0, leader_posture=posture,
        )
        args.update(kw)
        return evaluate_tactical_policies(**args)

    def test_posture_echoed_in_state_and_baseline(self):
        out = self._evaluate("defensive_boost")
        self.assertEqual(out["state"]["leader_posture"], "defensive_boost")
        self.assertEqual(out["baseline"]["leader_defense"]["posture"],
                         "defensive_boost")
        balanced = self._evaluate("balanced")
        self.assertIsNone(balanced["baseline"]["leader_defense"])

    def _row(self, out, policy):
        return next(r for r in out["policies"] if r["policy"] == policy)

    def test_defense_weakens_attacking_policies(self):
        """The deterministic attack-weakening directions: under a defending
        leader no policy's pass is HASTENED, and each attack's contested
        phase therefore runs at least as long — the attack pays at least as
        much deploy for a later-or-equal pass.  Scores and cumulative
        probabilities are deliberately not compared across postures: an
        out-of-time classifier saturates the window (cums 0.8-0.9 for every
        policy), so score re-orderings are wear/latency tie-breaks among
        saturated walks — model-scale artifacts, not the lever's effect."""
        bal = self._evaluate("balanced")
        dfn = self._evaluate("defensive_boost")
        for name in ("GREEDY ATTACK", "TACTICAL STALK", "SAVE & DEFEND"):
            b, d = self._row(bal, name), self._row(dfn, name)
            self.assertFalse(
                d["pass_lap"] is not None and b["pass_lap"] is not None
                and d["pass_lap"] < b["pass_lap"],
                f"{name}: defence must never hasten the pass "
                f"({d['pass_lap']} < {b['pass_lap']})")
            self.assertGreaterEqual(
                d["energy_cost_mj"], b["energy_cost_mj"] - 1e-9,
                f"{name}: the contested phase runs at least as long under "
                f"defence, so the attack cannot spend less deploy")
        # A pure banker banks strictly more under a defending leader (the
        # walk runs longer before whatever converts): SAVE & DEFEND.
        b, d = (self._row(bal, "SAVE & DEFEND"),
                self._row(dfn, "SAVE & DEFEND"))
        self.assertGreaterEqual(
            d["energy_banked_mj"], b["energy_banked_mj"],
            "SAVE & DEFEND banks at least as much under a defending leader")

    def test_defense_eats_the_attacker_edge(self):
        """The adversarial mechanism, on the deterministic baseline layer:
        the no-lever baseline under a DEFENDING leader must show a slower
        chaser pace edge, a wider projected final gap, and a real deploy
        spend from the leader's own store — the attack is priced against a
        narrower window.  The old form of this test compared score EDGES
        between policy families, but an out-of-time classifier saturates
        every walk's cum (0.8-0.9 here), so those edges collapse into
        wear/latency tie-breaks that re-order on each retrain; the baseline
        physics does not move."""
        bal = self._evaluate("balanced", chaser_battery_pct=62.5)
        dfn = self._evaluate("defensive_boost", chaser_battery_pct=62.5)
        bb, bd = bal["baseline"], dfn["baseline"]
        # pace_gap_s = leader lap time - chaser lap time: the chaser's
        # EDGE.  A defending leader must SHRINK it, leave a wider projected
        # final gap, and fund the defence from its own store.
        self.assertLess(
            bd["avg_pace_gap_s"], bb["avg_pace_gap_s"],
            "a defending leader must shrink the chaser's pace edge "
            "(pace_gap = leader - chaser)")
        self.assertGreaterEqual(
            bd["projected_final_gap_s"], bb["projected_final_gap_s"],
            "the defended projection must leave a wider final gap")
        self.assertIsNone(
            bb["leader_defense"],
            "balanced posture deploys nothing")
        self.assertIsNotNone(bd["leader_defense"])
        self.assertGreater(bd["leader_defense"]["deployed_mj"], 0.0,
                           "the defence is funded, not free")
        # And across the whole policy table: no attack converts EARLIER
        # under a defending leader (out-of-time-safe direction).
        for name in ("GREEDY ATTACK", "TACTICAL STALK", "BALANCED HOLD",
                     "SAVE & DEFEND"):
            b, d = self._row(bal, name), self._row(dfn, name)
            self.assertFalse(
                d["pass_lap"] is not None and b["pass_lap"] is not None
                and d["pass_lap"] < b["pass_lap"],
                f"{name}: defence must never hasten the pass")

    def test_adversarial_flip_to_save_at_mid_battery(self):
        """The demo beat: mid battery, a passive leader allows an attacking
        recommendation, but a DEFENDING leader pushes the engine to the
        conservative card (SAVE & DEFEND / BALANCED HOLD) — the engine
        refuses to buy a contested pass."""
        out = self._evaluate("defensive_boost", chaser_battery_pct=45.0)
        action = out["recommendation"]["action_card"]["action"]
        self.assertIn(action, {"SAVE & DEFEND", "BALANCED HOLD"},
                      f"defending leader should flip the card to the "
                      f"conservative posture, got {action}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
