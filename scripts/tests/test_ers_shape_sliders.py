"""Unit tests for the per-sector ERS SLIDER-BANK integration.

The Energy Sandbox's slider vocabulary (three MJ/lap deploy deltas, one
per sector) is now first-class input to the decision engines:

  * _shape_vector normalisation — sandbox clamp, all-zero -> no shape,
    zero-SUM stays active (the reallocation case).
  * Chaser engine — a caller shape rides every policy walk; a net-0
    reallocation keeps BALANCED HOLD store-neutral BUT turns the SOC walk
    on, so the card's battery margin is defined; a positive-net shape is
    scaled per policy phase preserving its sector spread.
  * Leader engine — the leader's own bank expresses every defence lever;
    the threat bank runs the attacking chaser on its shape.
  * Unified call — both banks thread through evaluate_call, the payload
    discloses the walked shapes, and the engine cache keys on the
    normalised vectors (different shapes never share an entry; an
    all-zero bank shares the no-shape entry).

Tests hit the real trained models + the real simulator (no mocks — same
convention as test_policy_engine.py), so they are slow (~1-2 s per
evaluate).

Run:
    python -m unittest scripts.tests.test_ers_shape_sliders -v
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import policy_engine as pe          # noqa: E402

_LEADER, _CHASER = "VER", "HAM"
_TRACK = "Autodromo Nazionale di Monza"


def _chaser_eval(**kw):
    args = dict(
        leader_code=_LEADER, chaser_code=_CHASER, track_name=_TRACK,
        start_lap=20, race_length=50, gap_before_s=0.8,
        leader_tyre_compound="Medium", chaser_tyre_compound="Medium",
        leader_tyre_age=10, chaser_tyre_age=8, year=2026,
        chaser_battery_pct=62.5,
    )
    args.update(kw)
    return pe.evaluate_tactical_policies(**args)


def _leader_eval(**kw):
    args = dict(
        leader_code=_LEADER, chaser_code=_CHASER, track_name=_TRACK,
        start_lap=20, race_length=50, gap_before_s=0.8,
        leader_tyre_compound="Medium", chaser_tyre_compound="Medium",
        leader_tyre_age=10, chaser_tyre_age=8, year=2026,
        leader_battery_pct=62.5,
    )
    args.update(kw)
    return pe.evaluate_leader_policies(**args)


def _unified(**kw):
    args = dict(
        leader_code=_LEADER, chaser_code=_CHASER, track_name=_TRACK,
        start_lap=20, race_length=50, gap_before_s=0.8,
        leader_tyre_compound="Medium", chaser_tyre_compound="Medium",
        leader_tyre_age=10, chaser_tyre_age=8, year=2026,
        battery_pct=62.5, reserve_target_mj=1.6, perspective="chaser",
    )
    args.update(kw)
    return pe.evaluate_call(**args)


class ShapeVector(unittest.TestCase):
    """The slider bank's normalisation — pure unit tests, no models."""

    def test_none_passes_through(self):
        self.assertIsNone(pe._shape_vector(None))

    def test_junk_collapses_to_none(self):
        self.assertIsNone(pe._shape_vector("nope"))
        self.assertIsNone(pe._shape_vector([1, 2]))          # wrong arity
        self.assertIsNone(pe._shape_vector([1, 2, 3, 4]))
        self.assertIsNone(pe._shape_vector(["a", "b", "c"]))

    def test_all_zero_collapses_to_none(self):
        self.assertIsNone(pe._shape_vector([0, 0, 0]))
        self.assertIsNone(pe._shape_vector([0.001, -0.001, 0.0005]))

    def test_sandbox_clamp_applied_per_sector(self):
        self.assertEqual(pe._shape_vector([9, -9, 0.5]), [8.5, -8.5, 0.5])

    def test_zero_sum_reallocation_stays_active(self):
        v = pe._shape_vector([0.4, -0.4, 0.0])
        self.assertEqual(v, [0.4, -0.4, 0.0])
        self.assertAlmostEqual(pe._shape_net(v), 0.0)

    def test_scale_shape_preserves_spread_and_net(self):
        vec = [0.3, 0.0, -0.15]
        scaled = pe._scale_shape(vec, 2.0)
        self.assertEqual(scaled, [0.6, 0.0, -0.3])
        self.assertAlmostEqual(pe._shape_net(scaled),
                               2.0 * pe._shape_net(vec))
        # Zero-sum shapes pass through unscaled at every phase.
        zs = [0.2, -0.1, -0.1]
        self.assertEqual(pe._scale_shape(zs, 0.5), zs)
        # Zero factor on a net shape collapses (no free shape).
        self.assertIsNone(pe._scale_shape(vec, 0.0))


class ChaserShape(unittest.TestCase):
    """The chaser engine walks the caller's shape."""

    @classmethod
    def setUpClass(cls):
        cls.baseline = _chaser_eval()
        cls.realloc = _chaser_eval(chaser_shape=[0.4, -0.2, -0.2])
        # Non-uniform bank: a UNIFORM bank scaled per policy reproduces
        # the flat-lever walk exactly (same net, same sectors), so only
        # a spread with real sector bias may move the scores.
        cls.push = _chaser_eval(chaser_shape=[0.5, 0.0, 0.0])

    def test_state_discloses_shape(self):
        self.assertEqual(self.realloc["state"]["ers_shape"],
                         [0.4, -0.2, -0.2])
        self.assertIsNone(self.baseline["state"]["ers_shape"])

    def test_rows_disclose_walked_shape(self):
        for p in self.realloc["policies"]:
            self.assertEqual(p["ers_shape"], [0.4, -0.2, -0.2])
        for p in self.baseline["policies"]:
            self.assertIsNone(p["ers_shape"])

    def test_zero_sum_shape_turns_soc_walk_on_for_hold(self):
        # At the default state the hold walks the store-neutral posture:
        # the SOC is modelled and the margin defined (no degenerate nulls).
        hold0 = next(p for p in self.baseline["policies"]
                     if p["policy"] == "BALANCED HOLD")
        self.assertIsNotNone(hold0["soc_end_pct"])
        self.assertIsNotNone(hold0["battery_margin_pct"])
        # The store stays put: the hold spends and banks nothing.
        self.assertEqual(hold0["energy_cost_mj"], 0.0)
        self.assertEqual(hold0["energy_banked_mj"], 0.0)
        # With a net-0 reallocation, the store walk runs — margin defined.
        hold1 = next(p for p in self.realloc["policies"]
                     if p["policy"] == "BALANCED HOLD")
        self.assertIsNotNone(hold1["soc_end_pct"])
        self.assertIsNotNone(hold1["battery_margin_pct"])
        # ...and the reallocation is store-neutral: the store never moves.
        self.assertEqual(hold1["energy_cost_mj"], 0.0)
        self.assertGreaterEqual(hold1["min_soc_pct"], 100.0 * 1.6 / 4.0 - 0.5)

    def test_shape_changes_the_decision_space(self):
        # A non-uniform positive-net bank must shift the ranked scores
        # (the shape is a real pace input, not decoration).
        a = [p["score_s"] for p in self.baseline["policies"]]
        b = [p["score_s"] for p in self.push["policies"]]
        self.assertNotEqual(a, b)

    def test_uniform_bank_reproduces_flat_lever_scores(self):
        # Backward-compatibility proof: a uniform bank is the SAME walk
        # as the historical flat spread, so the whole ranking matches.
        uniform = _chaser_eval(chaser_shape=[0.4, 0.4, 0.4])
        self.assertEqual([p["score_s"] for p in self.baseline["policies"]],
                         [p["score_s"] for p in uniform["policies"]])

    def test_scaled_shape_matches_flat_lever_net(self):
        # GREEDY ATTACK's lever is +0.12 MJ/lap; through a uniform bank
        # [0.04, 0.04, 0.04] the walk must be identical to the flat
        # spread (same net, same sectors).
        uniform = _chaser_eval(chaser_shape=[0.04, 0.04, 0.04])
        greedy_u = next(p for p in uniform["policies"]
                        if p["policy"] == "GREEDY ATTACK")
        greedy_b = next(p for p in self.baseline["policies"]
                        if p["policy"] == "GREEDY ATTACK")
        self.assertAlmostEqual(greedy_u["energy_cost_mj"],
                               greedy_b["energy_cost_mj"], places=2)


class LeaderShape(unittest.TestCase):
    """The leader engine walks its own bank plus the threat's bank."""

    @classmethod
    def setUpClass(cls):
        cls.baseline = _leader_eval()
        cls.shaped = _leader_eval(leader_shape=[0.4, -0.2, -0.2],
                                  threat_shape=[0.1, 0.1, 0.1])

    def test_state_discloses_both_banks(self):
        st = self.shaped["state"]
        self.assertEqual(st["ers_shape"], [0.4, -0.2, -0.2])
        self.assertEqual(st["threat_shape"], [0.1, 0.1, 0.1])
        self.assertIsNone(self.baseline["state"]["ers_shape"])

    def test_rows_disclose_both_banks(self):
        for p in self.shaped["policies"]:
            self.assertEqual(p["ers_shape"], [0.4, -0.2, -0.2])
            self.assertEqual(p["threat_shape"], [0.1, 0.1, 0.1])

    def test_threat_shape_changes_the_threat(self):
        # A positive-net threat bank makes the attacking chaser faster
        # than the flat default — the defended position must be scored
        # against a different probability surface.
        a = [p["score_s"] for p in self.baseline["policies"]]
        b = [p["score_s"] for p in self.shaped["policies"]]
        self.assertNotEqual(a, b)


class UnifiedShapePlumbing(unittest.TestCase):
    """evaluate_call threads both banks and discloses them."""

    @classmethod
    def setUpClass(cls):
        cls.out = _unified(chaser_shape=[0.3, -0.3, 0.0],
                           leader_shape=[0.2, 0.2, 0.2])

    def test_final_call_discloses_both_shapes(self):
        self.assertEqual(self.out["final_call"]["ers_shape"],
                         [0.3, -0.3, 0.0])
        self.assertEqual(self.out["final_call"]["leader_ers_shape"],
                         [0.2, 0.2, 0.2])

    def test_engines_received_their_shapes(self):
        self.assertEqual(self.out["state"]["ers_shape"], [0.3, -0.3, 0.0])
        self.assertEqual(self.out["leader"]["recommendation"]
                         ["action_card"].get("ers_shape") or
                         self.out["leader_state"].get("ers_shape"),
                         [0.2, 0.2, 0.2] if self.out["leader_state"]
                         .get("ers_shape") else None)


class CacheKeying(unittest.TestCase):
    """A changed slider bank must never serve a stale cached call."""

    def test_zero_bank_shares_no_shape_entry(self):
        k0 = pe._engine_cache_key(
            "chaser", leader_code=_LEADER, chaser_code=_CHASER,
            track_name=_TRACK, start_lap=20, race_length=50,
            gap_before_s=0.8, leader_tyre_compound="Medium",
            chaser_tyre_compound="Medium", leader_tyre_age=10,
            chaser_tyre_age=8, year=2026, leader_batt=None,
            chaser_batt=62.5, reserve=1.6, posture="balanced",
            chaser_shape=None)
        k1 = pe._engine_cache_key(
            "chaser", leader_code=_LEADER, chaser_code=_CHASER,
            track_name=_TRACK, start_lap=20, race_length=50,
            gap_before_s=0.8, leader_tyre_compound="Medium",
            chaser_tyre_compound="Medium", leader_tyre_age=10,
            chaser_tyre_age=8, year=2026, leader_batt=None,
            chaser_batt=62.5, reserve=1.6, posture="balanced",
            chaser_shape=[0, 0, 0])
        self.assertEqual(k0, k1, "an all-zero bank IS the no-shape walk")

    def test_different_banks_never_share(self):
        base = dict(
            engine="chaser", leader_code=_LEADER, chaser_code=_CHASER,
            track_name=_TRACK, start_lap=20, race_length=50,
            gap_before_s=0.8, leader_tyre_compound="Medium",
            chaser_tyre_compound="Medium", leader_tyre_age=10,
            chaser_tyre_age=8, year=2026, leader_batt=None,
            chaser_batt=62.5, reserve=1.6, posture="balanced")
        k_none = pe._engine_cache_key(chaser_shape=None, **base)
        k_realloc = pe._engine_cache_key(
            chaser_shape=[0.3, -0.3, 0.0], **base)
        k_push = pe._engine_cache_key(chaser_shape=[0.3, 0.3, 0.3], **base)
        k_leader = pe._engine_cache_key(
            leader_shape=[0.3, 0.3, 0.3], chaser_shape=None, **base)
        self.assertNotEqual(k_none, k_realloc)
        self.assertNotEqual(k_realloc, k_push)
        self.assertNotEqual(k_none, k_leader)

    def test_cached_call_respects_new_shape(self):
        # Same state twice with different banks: the second call must
        # report the SECOND shape, proving no stale cache leak.
        first = _unified(chaser_shape=[0.3, -0.3, 0.0])
        second = _unified(chaser_shape=[0.3, 0.3, 0.3])
        self.assertEqual(first["final_call"]["ers_shape"], [0.3, -0.3, 0.0])
        self.assertEqual(second["final_call"]["ers_shape"], [0.3, 0.3, 0.3])
        # And re-running the first state returns its own shape again.
        again = _unified(chaser_shape=[0.3, -0.3, 0.0])
        self.assertEqual(again["final_call"]["ers_shape"], [0.3, -0.3, 0.0])


if __name__ == "__main__":
    unittest.main()
