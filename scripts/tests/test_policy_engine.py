"""Unit tests for the multi-policy decision engine (P1-1).

Covers the behaviors IMPLEMENTATION_PLAN.md requires before submission:

  * Policy determinism — same state, same recommendation (the simulator is
    deterministic; the decision layer must inherit that).
  * Constraint enforcement — no feasible policy ever dips below the
    management reserve; policies that do are marked infeasible and can
    never win; a low-battery state must yield a conserving recommendation.
  * Decision intelligence — the recommendation actually FLIPS with the
    race state (battery-rich vs battery-low), and the battery gate from the
    P0 fixes propagates into policy feasibility.
  * ACTION-card contract — the output carries the exact field set the
    challenge brief demands, with sane values.
  * Latency honesty — the payload's stated budget is >= its own measured
    latency (never claim faster than delivered).

The tests hit the real trained models + the real simulator (same
convention as test_p0_fixes.py) — no mocks, so they are slow-ish
(~1.5 s cold + ~0.7 s per evaluate call).

Run:
    python -m unittest scripts.tests.test_policy_engine -v
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import policy_engine as pe          # noqa: E402

_LEADER, _CHASER = "VER", "HAM"
_TRACK = "Autodromo Nazionale di Monza"


def _evaluate(battery_pct=None, gap=0.8, start_lap=20, race_length=50,
              reserve_target_mj=None):
    return pe.evaluate_tactical_policies(
        leader_code=_LEADER, chaser_code=_CHASER, track_name=_TRACK,
        start_lap=start_lap, race_length=race_length, gap_before_s=gap,
        leader_tyre_compound="Medium", chaser_tyre_compound="Medium",
        leader_tyre_age=10, chaser_tyre_age=10, year=2026,
        chaser_battery_pct=battery_pct,
        reserve_target_mj=reserve_target_mj,
    )


class Determinism(unittest.TestCase):
    """Same state in, same decision out — the engine must be re-derivable."""

    def test_identical_state_identical_recommendation(self):
        a = _evaluate(battery_pct=62.5)
        b = _evaluate(battery_pct=62.5)
        self.assertEqual(a["recommendation"]["action_card"]["action"],
                         b["recommendation"]["action_card"]["action"])
        self.assertEqual([p["score_s"] for p in a["policies"]],
                         [p["score_s"] for p in b["policies"]])
        self.assertEqual(a["recommendation"]["action_card"]["confidence"],
                         b["recommendation"]["action_card"]["confidence"])


class ConstraintEnforcement(unittest.TestCase):
    """The hard constraint: reserve breaches can never win."""

    def test_no_feasible_policy_below_reserve(self):
        r = _evaluate(battery_pct=90)
        for p in r["policies"]:
            if p["feasible"] and p["min_soc_pct"] is not None:
                self.assertGreaterEqual(
                    p["min_soc_pct"],
                    r["state"]["reserve_target_pct"] - 0.5,
                    f"{p['policy']} marked feasible but dipped below reserve")

    def test_low_battery_marks_push_policies_infeasible(self):
        # 31% start: everything that spends the store breaches the reserve
        # target and must be rejected — the audit's canonical demo beat.
        r = _evaluate(battery_pct=31)
        self.assertIn("GREEDY ATTACK",
                      r["recommendation"]["infeasible_policies"])
        card = r["recommendation"]["action_card"]
        self.assertTrue(card["feasible"])
        self.assertEqual(card["action"], "BALANCED HOLD")
        # The winner genuinely conserves: no store spend.
        self.assertEqual(card["energy_cost_mj"], 0.0)

    def test_infeasible_policy_never_ranked_first(self):
        r = _evaluate(battery_pct=31)
        ranked = sorted(r["policies"], key=lambda p: p["score_s"])
        self.assertFalse(ranked[0]["infeasible_reason"],
                         "an infeasible policy topped the ranking")

    def test_tight_reserve_target_rejects_more_policies(self):
        # Loose target (25% of store): all policies keep it.  Tight target
        # (62.5%): the push policies dip below it and are rejected.
        loose = _evaluate(battery_pct=62.5, reserve_target_mj=1.0)
        tight = _evaluate(battery_pct=62.5, reserve_target_mj=2.5)
        self.assertEqual(loose["recommendation"]["infeasible_policies"], [])
        self.assertGreaterEqual(
            len(tight["recommendation"]["infeasible_policies"]), 3)
        self.assertIn("GREEDY ATTACK",
                      tight["recommendation"]["infeasible_policies"])


class DecisionIntelligence(unittest.TestCase):
    """The engine must change its mind when the race state changes."""

    def test_battery_rich_vs_battery_low_flips_decision(self):
        rich = _evaluate(battery_pct=90)
        low = _evaluate(battery_pct=31)
        self.assertNotEqual(
            rich["recommendation"]["action_card"]["action"],
            low["recommendation"]["action_card"]["action"],
            "recommendation identical at 90% and 31% battery — the engine "
            "is not state-aware")

    def test_low_battery_confidence_is_decisive(self):
        # The battery-low flip is the demo's headline beat: the reserve
        # breach is structural (4 of 5 policies rejected), so confidence
        # should be high, not a coin flip.
        r = _evaluate(battery_pct=31)
        self.assertGreaterEqual(
            r["recommendation"]["action_card"]["confidence"], 0.5)

    def test_every_policy_reports_score_components(self):
        r = _evaluate(battery_pct=62.5)
        for p in r["policies"]:
            sc = p["score_components"]
            for key in ("pass_gain_s", "battery_cost_s", "wear_cost_s",
                        "risk_cost_s", "latency_cost_s"):
                self.assertIn(key, sc, f"{p['policy']} missing {key}")
            self.assertIsInstance(p["feasible"], bool)


class ActionCardContract(unittest.TestCase):
    """The challenge brief's required output shape, field by field."""

    def test_action_card_has_required_fields(self):
        card = _evaluate(battery_pct=62.5)["recommendation"]["action_card"]
        for field in ("action", "deploy_lap", "energy_pct",
                      "expected_gap_s", "overtake_probability",
                      "energy_cost_mj", "expected_finish_delta_s",
                      "battery_margin_pct", "confidence", "reason"):
            self.assertIn(field, card, f"ACTION card missing '{field}'")

    def test_action_card_values_sane(self):
        card = _evaluate(battery_pct=62.5)["recommendation"]["action_card"]
        self.assertTrue(card["action"])
        self.assertGreater(card["deploy_lap"], 0)
        self.assertTrue(0.0 <= card["overtake_probability"] <= 1.0)
        self.assertTrue(0.0 <= card["confidence"] <= 1.0)
        self.assertTrue(card["reason"])
        self.assertLess(len(card["reason"]), 400)   # a sentence, not an essay

    def test_payload_reports_latency_honestly(self):
        # First call may pay model loading; use the second (warm) run for
        # the budget comparison, mirroring how the demo is actually run.
        _evaluate(battery_pct=62.5)
        r = _evaluate(battery_pct=62.5)
        self.assertGreater(r["latency_ms"], 0.0)
        # Never claim a budget smaller than the measured warm delivery.
        self.assertGreaterEqual(r["latency_budget_ms"], r["latency_ms"])
        self.assertEqual(r["latency_budget_ms"], pe.LATENCY_BUDGET_MS)

    def test_exactly_five_policies_evaluated(self):
        r = _evaluate(battery_pct=62.5)
        self.assertEqual(len(r["policies"]), 5)
        self.assertEqual(len({p["policy"] for p in r["policies"]}), 5)


if __name__ == "__main__":
    unittest.main()
