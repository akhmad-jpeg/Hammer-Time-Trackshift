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
     walk's cumulative pass probability is LOWER than the balanced one at
     the same race state.
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


def _cum_at(result, lap):
    """Cumulative pass probability at a fixed lap of the walk."""
    c = 0.0
    for l in result["laps"]:
        if l["lap"] <= lap:
            c = l["cumulative_probability"]
    return c


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
        defended = _walk("defensive_boost")
        defense = defended["summary"]["leader_defense"]
        # A 30-lap defensive walk at ~0.24 MJ/lap must exhaust the ~2.8 MJ
        # above the floor and revert: energy-limited laps are counted.
        self.assertGreaterEqual(defense["energy_limited_laps"], 1)

    def test_defense_narrows_the_attack(self):
        """Same race state, same chaser push, FIXED horizon: a defending
        leader must cut the cumulative pass probability (less closing, no
        compensating energy edge) and push the closest approach wider and
        later.  Fixed-horizon cum is the right invariant — the balanced
        walk converts and STOPS early while the defended walk keeps
        rolling in-window laps, so full-horizon cums are not comparable."""
        bal = _walk("balanced")
        dfn = _walk("defensive_boost")
        horizons = (24, 26)
        for h in horizons:
            cum_bal = _cum_at(bal, h)
            cum_dfn = _cum_at(dfn, h)
            self.assertLess(
                cum_dfn, cum_bal,
                f"defended cum at fixed horizon L{h} must be lower "
                f"({cum_dfn} vs {cum_bal})")
        self.assertGreater(
            dfn["summary"]["closest_lap"], bal["summary"]["closest_lap"],
            "defense must slow the close: the closest approach must come "
            "later than under a passive leader")


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
        """GREEDY ATTACK (single-phase, same horizon both postures) must
        lose pass probability and score under a defending leader.
        TACTICAL STALK is score-compared only: it is a two-phase policy,
        so its phase-2 horizon shifts with the posture and its cumulative
        P is not comparable across postures."""
        bal = self._evaluate("balanced")
        dfn = self._evaluate("defensive_boost")
        b, d = (self._row(bal, "GREEDY ATTACK"),
                self._row(dfn, "GREEDY ATTACK"))
        self.assertLess(
            d["overtake_probability"], b["overtake_probability"],
            "defense must cut GREEDY ATTACK's pass probability")
        self.assertGreater(
            d["score_s"], b["score_s"],
            "defense must worsen GREEDY ATTACK's score")
        bs, ds = (self._row(bal, "TACTICAL STALK"),
                  self._row(dfn, "TACTICAL STALK"))
        self.assertGreater(
            ds["score_s"], bs["score_s"],
            "defense must worsen TACTICAL STALK's score")

    def test_defense_collapses_decision_margin(self):
        """The adversarial mechanism, quantified: a defending leader eats
        the attacker's edge, so the score gap between the best and second
        policies collapses (measured: 0.73 s -> 0.19 s at the default
        state) — the call becomes a coin flip the strategist must see."""
        bal = self._evaluate("balanced", chaser_battery_pct=62.5)
        dfn = self._evaluate("defensive_boost", chaser_battery_pct=62.5)
        self.assertLess(
            dfn["recommendation"]["decision_margin_s"],
            bal["recommendation"]["decision_margin_s"],
            "defense must collapse the decision margin")

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
