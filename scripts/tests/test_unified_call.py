"""Unit tests for the UNIFIED call (policy_engine.evaluate_call).

The merged deliverable: BOTH decision engines run on the same race state,
the leader engine's recommended DEFENCE conditions the posture the chaser
engine's attack policies are scored against, and the seat switch only picks
whose call is surfaced.  Covers:

  * Coupling — the chaser posture is derived from the leader engine's own
    winner (not a hand-set toggle), and both seats agree on it.
  * Determinism — same state, same merged call.
  * Contract — the payload carries the single final call, ten policy rows
    tagged with their seat, per-engine drill-downs and honest latency.
  * Seat symmetry — flipping the seat keeps the same underlying engine
    recommendations and only changes whose call is surfaced.
  * Battery mapping — the seat's own battery overrides the engine default.

Tests hit the real trained models + the real simulator (no mocks — same
convention as test_policy_engine.py), so they are slow (~2 s per evaluate_call).

Run:
    python -m unittest scripts.tests.test_unified_call -v
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import policy_engine as pe          # noqa: E402

_LEADER, _CHASER = "HAM", "VER"
_TRACK = "Autodromo Nazionale di Monza"


def _evaluate(**kw):
    args = dict(
        leader_code=_LEADER, chaser_code=_CHASER, track_name=_TRACK,
        start_lap=20, race_length=50, gap_before_s=0.8,
        leader_tyre_compound="Medium", chaser_tyre_compound="Medium",
        leader_tyre_age=10, chaser_tyre_age=8, year=2026,
        battery_pct=62.5, reserve_target_mj=1.6,
        perspective="chaser",
    )
    args.update(kw)
    return pe.evaluate_call(**args)


class EngineCoupling(unittest.TestCase):
    """The merged call must reflect BOTH engines' recommendations."""

    @classmethod
    def setUpClass(cls):
        cls.out = _evaluate()

    def test_both_engines_recommendations_agree_with_coupling(self):
        fc = self.out["final_call"]
        leader_card = self.out["leader"]["recommendation"]["action_card"]
        self.assertEqual(
            fc["engine_coupling"]["opponent_engine_recommendation"],
            leader_card["action"],
            "the disclosed coupling must be the leader engine's own winner")

    def test_posture_maps_from_leader_winner(self):
        used = self.out["final_call"]["engine_coupling"]["chaser_posture_used"]
        self.assertIn(used, ("balanced", "defensive_boost"))

    def test_policies_carry_seat_tags(self):
        rows = self.out["policies"]
        self.assertEqual(len(rows), 10)
        seats = {r["seat"] for r in rows}
        self.assertEqual(seats, {"chaser", "leader"})
        for r in rows:
            self.assertIn("policy", r)
            self.assertIn("score_s", r)
            self.assertIn("feasible", r)

    def test_single_final_call_with_opponent(self):
        fc = self.out["final_call"]
        self.assertIn(fc["seat"], ("chaser", "leader"))
        self.assertTrue(fc["action"])
        self.assertIn("action", fc["opponent"])
        self.assertIn("verdict", fc["projection"])

    def test_latency_is_honest_against_budget(self):
        self.assertGreater(self.out["latency_ms"], 0.0)
        self.assertGreaterEqual(self.out["latency_budget_ms"],
                                pe.CALL_LATENCY_BUDGET_MS)


class SeatSwitch(unittest.TestCase):
    """Flipping the seat re-uses the engines; only the surfaced call moves."""

    @classmethod
    def setUpClass(cls):
        cls.chaser = _evaluate(perspective="chaser")
        cls.leader = _evaluate(perspective="leader")

    def test_seat_perspective_flips(self):
        self.assertEqual(self.chaser["perspective"], "chaser")
        self.assertEqual(self.leader["perspective"], "leader")
        self.assertEqual(self.chaser["final_call"]["seat"], "chaser")
        self.assertEqual(self.leader["final_call"]["seat"], "leader")

    def test_engine_recommendations_are_seat_invariant(self):
        # The engines see the same state regardless of the caller's seat —
        # the leader card must be identical, and the chaser card too (the
        # battery mapping maps 62.5% onto each engine's own default car).
        self.assertEqual(
            self.chaser["leader"]["recommendation"]["action_card"]["action"],
            self.leader["leader"]["recommendation"]["action_card"]["action"])
        self.assertEqual(
            self.chaser["chaser"]["recommendation"]["action_card"]["action"],
            self.leader["chaser"]["recommendation"]["action_card"]["action"])

    def test_surfaced_call_matches_own_seat_engine(self):
        self.assertEqual(
            self.chaser["final_call"]["action"],
            self.chaser["chaser"]["recommendation"]["action_card"]["action"])
        self.assertEqual(
            self.leader["final_call"]["action"],
            self.leader["leader"]["recommendation"]["action_card"]["action"])

    def test_seat_flips_the_surfaced_perspective_label(self):
        self.assertIn("chaser", self.chaser["perspective_label"].lower())
        self.assertIn("attacking", self.chaser["perspective_label"].lower())
        self.assertIn("leader", self.leader["perspective_label"].lower())
        self.assertIn("defending", self.leader["perspective_label"].lower())


class Determinism(unittest.TestCase):
    """Same state in, same merged call out — the coupling must not add noise."""

    @classmethod
    def setUpClass(cls):
        cls.a = _evaluate()
        cls.b = _evaluate()

    def test_identical_state_identical_merged_call(self):
        self.assertEqual(self.a["final_call"]["action"],
                         self.b["final_call"]["action"])
        self.assertEqual(self.a["final_call"]["opponent"]["action"],
                         self.b["final_call"]["opponent"]["action"])
        self.assertEqual(self.a["final_call"]["engine_coupling"]["chaser_posture_used"],
                         self.b["final_call"]["engine_coupling"]["chaser_posture_used"])
        self.assertEqual([p["score_s"] for p in self.a["policies"]],
                         [p["score_s"] for p in self.b["policies"]])


class EngineCache(unittest.TestCase):
    """evaluate_call memoises each engine's evaluation against its inputs.

    A repeated call reuses the stored walk sets (disclosed in the payload's
    engine_cache block — a cached latency is never presented as a fresh
    simulation), a changed state misses, and a cached payload is isolated
    from caller mutations.
    """

    @classmethod
    def setUpClass(cls):
        pe._CALL_ENGINE_CACHE.clear()

    def setUp(self):
        pe._CALL_ENGINE_CACHE.clear()

    def tearDown(self):
        pe._CALL_ENGINE_CACHE.clear()

    def test_repeat_is_a_disclosed_cache_hit(self):
        first = _evaluate(battery_pct=62.5)
        self.assertFalse(first["engine_cache"]["leader_reused"])
        self.assertFalse(first["engine_cache"]["chaser_reused"])
        self.assertIn("fresh", first["engine_cache"]["note"])

        second = _evaluate(battery_pct=62.5)
        self.assertTrue(second["engine_cache"]["leader_reused"])
        self.assertTrue(second["engine_cache"]["chaser_reused"])
        self.assertIn("reused", second["engine_cache"]["note"])
        # The cached decision is the decision.
        self.assertEqual(first["final_call"], second["final_call"])
        self.assertEqual([p["score_s"] for p in first["policies"]],
                         [p["score_s"] for p in second["policies"]])

    def test_seat_flip_at_defaults_hits_the_same_engines(self):
        """The leader engine maps None batteries onto the default, so a
        seat flip at default batteries is the same computation — the
        cache key normalises it the same way."""
        pe._CALL_ENGINE_CACHE.clear()
        chaser_seat = _evaluate(perspective="chaser")     # battery_pct=None
        leader_seat = _evaluate(perspective="leader")     # battery_pct=None
        self.assertTrue(leader_seat["engine_cache"]["leader_reused"])
        self.assertTrue(leader_seat["engine_cache"]["chaser_reused"])
        self.assertEqual(
            chaser_seat["chaser"]["recommendation"]["action_card"]["action"],
            leader_seat["chaser"]["recommendation"]["action_card"]["action"])
        self.assertEqual(
            chaser_seat["leader"]["recommendation"]["action_card"]["action"],
            leader_seat["leader"]["recommendation"]["action_card"]["action"])

    def test_changed_state_misses_and_recomputes(self):
        """A changed battery is a different state: the cache must miss and
        both engines must walk again (the miss flags are the proof)."""
        _evaluate(battery_pct=62.5)
        changed = _evaluate(battery_pct=40.0)
        self.assertFalse(changed["engine_cache"]["leader_reused"])
        self.assertFalse(changed["engine_cache"]["chaser_reused"])

    def test_cached_payload_is_mutation_isolated(self):
        first = _evaluate(battery_pct=62.5)
        victim = first["policies"][0]
        victim["score_s"] = -999.0
        again = _evaluate(battery_pct=62.5)
        self.assertNotEqual(again["policies"][0]["score_s"], -999.0,
                            "a caller mutation leaked into the cache")

    def test_cache_is_bounded(self):
        for b in (30.0, 40.0, 50.0, 62.5, 80.0, 95.0,
                  31.0, 41.0, 51.0, 63.0, 81.0, 96.0):
            _evaluate(battery_pct=b)
        self.assertLessEqual(len(pe._CALL_ENGINE_CACHE),
                             pe.CALL_ENGINE_CACHE_MAX)


class BatteryMapping(unittest.TestCase):
    """The seat's own battery override reaches the right engine."""

    def test_low_battery_surfaced_call_differs_from_default(self):
        # 31% is the audit's canonical demo beat: the attack engine must
        # prune the spend policies.  From the CHASER seat the surfaced call
        # must come from the constrained attack engine.
        low = _evaluate(perspective="chaser", battery_pct=31)
        rows = [r for r in low["policies"] if r["seat"] == "chaser"]
        pruned = [r["policy"] for r in rows if not r["feasible"]]
        self.assertIn("GREEDY ATTACK", pruned,
                      "31% battery must prune the full-attack policy")
        self.assertEqual(low["final_call"]["seat"], "chaser")

    def test_invalid_perspective_raises(self):
        with self.assertRaises(ValueError):
            _evaluate(perspective="pitwall")


if __name__ == "__main__":
    unittest.main()
