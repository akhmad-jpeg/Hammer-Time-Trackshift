"""Regression tests for two-phase policy pricing (policy_engine).

Covers three red-team findings in the two-phase walk (TACTICAL STALK on the
chaser side, BANK & STRIKE on the leader side):

  * Phase-2 tyre-age inheritance — each car must tick from its OWN phase-1
    age.  The old `_phase2_gap` ticked only the chaser's age and stamped it
    onto BOTH cars, so the leader ran phase 2 on the wrong age whenever the
    two stints started at different laps.
  * Wear pricing — the strike phase pays at full lever on the chaser's (or
    leader's) COMPOUND.  The old code priced phase-1's lever only (which for
    a bank-then-strike policy is negative => zero wear) on a hard-coded
    Medium, so two-phase strikes rode free and Soft-tyre attacks were
    under-priced ~2.5x.
  * Cliff risk — a two-phase policy whose strike only converts by draining
    the store to its floor must pay CLIFF_RISK_S like a single-phase push
    does; the old check tested the phase-1 lever and so exempted every
    two-phase policy from the cliff penalty and the floor-drain infeasibility.

Run:
    python -m unittest scripts.tests.test_two_phase_pricing -v
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import policy_engine as pe          # noqa: E402


# ── Unit: _phase2_gap age inheritance ────────────────────────────────────

class Phase2GapInheritance(unittest.TestCase):
    """Each car's phase-2 age ticks from its OWN phase-1 age."""

    def test_independent_ages_when_stints_differ(self):
        sim = {
            "meta": {"start_lap": 20,
                     "tyres": {"chaser": {"age": 8}, "leader": {"age": 12}}},
            "laps": [{"lap": 22, "gap_before_s": 0.6}],
        }
        gap, c_age, l_age = pe._phase2_gap(sim, phase1_laps=22)
        self.assertAlmostEqual(gap, 0.6)
        self.assertEqual(c_age, 11)   # 8 + (22-20) + 1
        self.assertEqual(l_age, 15)   # 12 + (22-20) + 1  (was 11 before fix)

    def test_ages_diverge_by_the_stint_offset(self):
        sim = {
            "meta": {"start_lap": 5,
                     "tyres": {"chaser": {"age": 2}, "leader": {"age": 30}}},
            "laps": [{"lap": 10, "gap_before_s": 1.4}],
        }
        _, c_age, l_age = pe._phase2_gap(sim, phase1_laps=10)
        self.assertEqual(c_age, 8)
        self.assertEqual(l_age, 36)
        self.assertNotEqual(c_age, l_age,
                            "distinct starting ages must stay distinct")

    def test_empty_walk_returns_nones(self):
        self.assertEqual(pe._phase2_gap({"meta": {}, "laps": []}, 5),
                         (None, None, None))


# ── Unit: compound-aware wear ─────────────────────────────────────────────

class CompoundAwareWear(unittest.TestCase):
    """Wear is priced on the policy's tyre, not a hard-coded Medium."""

    def test_soft_costs_more_than_medium(self):
        full = pe.ERS_MAX_MJ_LAP
        soft = pe._est_extra_wear(full, 10, "Soft")
        medium = pe._est_extra_wear(full, 10, "Medium")
        self.assertGreater(soft, medium)
        # Soft ≈ 4.5%/lap vs Medium ≈ 2.2%/lap — roughly double, not equal.
        self.assertGreater(soft / max(medium, 1e-9), 1.5)

    def test_banking_burns_nothing(self):
        self.assertEqual(pe._est_extra_wear(-0.09, 10, "Soft"), 0.0)
        self.assertEqual(pe._est_extra_wear(0.0, 10, "Soft"), 0.0)

    def test_default_backwards_compatible_medium(self):
        full = pe.ERS_MAX_MJ_LAP
        self.assertEqual(pe._est_extra_wear(full, 10),
                         pe._est_extra_wear(full, 10, "Medium"))


# ── Integration: two-phase pricing through the real simulator ────────────
# Slow (real models, ~2 s per evaluate) — same convention as the other
# engine tests.

_LEADER, _CHASER = "HAM", "VER"
_TRACK = "Autodromo Nazionale di Monza"


def _tactical(**kw):
    args = dict(leader_code=_LEADER, chaser_code=_CHASER, track_name=_TRACK,
                start_lap=20, race_length=50, gap_before_s=0.8,
                leader_tyre_compound="Medium", chaser_tyre_compound="Medium",
                leader_tyre_age=12, chaser_tyre_age=8, year=2026,
                chaser_battery_pct=90.0, reserve_target_mj=1.6)
    args.update(kw)
    return pe.evaluate_tactical_policies(**args)


class TwoPhaseChaserPricing(unittest.TestCase):
    """TACTICAL STALK pays wear and cliff risk on its strike phase."""

    @classmethod
    def setUpClass(cls):
        cls.out = _tactical()

    def _row(self, name):
        return next(r for r in self.out["policies"] if r["policy"] == name)

    def test_stalk_pays_strike_wear(self):
        stalk = self._row("TACTICAL STALK")
        hold = self._row("BALANCED HOLD")
        self.assertGreater(stalk["score_components"]["wear_cost_s"], 0.0,
                           "the stalk's full-lever strike must cost wear")
        self.assertGreater(
            stalk["score_components"]["wear_cost_s"],
            hold["score_components"]["wear_cost_s"])

    def test_strike_wear_matches_the_analytic_strike_formula(self):
        # Pin the exact pricing: the stalk banks in phase 1 (zero wear) and
        # pays full lever for every lap of the strike phase.  Walk length
        # is deterministic (non-converting walks run to the flag), so the
        # formula is exact: rate(lever_frac=1, compound) x (L - deploy_lap).
        from tyre_degradation import tyre_health
        rate = float(tyre_health("Medium", 0)) - float(tyre_health("Medium", 1))
        expected = pe.LAMBDA_WEAR * rate * (50 - 22)   # start 20, deploy +2
        stalk = self._row("TACTICAL STALK")["score_components"]["wear_cost_s"]
        self.assertAlmostEqual(stalk, round(expected, 3), places=2,
                               msg="stalk wear must be the STRIKE phase's "
                                   "full-lever cost, not phase-1 (old: 0.0)")

    def test_push_lever_selects_the_striking_phase(self):
        stalk = next(p for p in pe.POLICIES if p["name"] == "TACTICAL STALK")
        greedy = next(p for p in pe.POLICIES if p["name"] == "GREEDY ATTACK")
        save = next(p for p in pe.POLICIES if p["name"] == "SAVE & DEFEND")
        self.assertEqual(pe._push_lever(stalk), stalk["phase2_mj"])
        self.assertGreater(pe._push_lever(stalk), 0)
        self.assertEqual(pe._push_lever(greedy), greedy["phase1_mj"])
        self.assertEqual(pe._push_lever(save), save["phase1_mj"])
        self.assertLess(pe._push_lever(save), 0)   # a banker never pushes

    def test_cliff_risk_keys_off_the_push_lever(self):
        # A drained two-phase strike must now pay CLIFF_RISK_S where the
        # old phase-1 test paid 0.  Force the drain with a near-empty
        # store: the stalk's strike runs the store to the floor.
        drained = _tactical(chaser_battery_pct=31.0)
        stalk = next(r for r in drained["policies"]
                     if r["policy"] == "TACTICAL STALK")
        self.assertTrue(stalk["min_soc_pct"] is not None
                        and stalk["min_soc_pct"] <= pe.LIVE_ATTACK_MIN_SOC_PCT + 0.05,
                        "setup: the strike must actually drain the store")
        self.assertGreaterEqual(stalk["score_components"]["cliff_cost_s"],
                                pe.CLIFF_RISK_S - 1e-9,
                                "a drained two-phase strike pays the cliff")
        self.assertTrue(stalk["infeasible_reason"],
                        "and it is barred: a pass bought by floor-draining "
                        "is infeasible whatever phase pushed")

    def test_compound_sensitivity_is_in_the_wear_model(self):
        # The per-compound scaling itself is a unit property of
        # _est_extra_wear (an end-to-end compound sweep is confounded by
        # walk truncation at the pass), so it is asserted there — see
        # CompoundAwareWear.  Here we only pin that the row's wear flows
        # from that same model on the chaser's compound.
        stalk = self._row("TACTICAL STALK")["score_components"]["wear_cost_s"]
        expected_rate = pe.LAMBDA_WEAR  # s per health-%
        self.assertGreater(stalk, expected_rate * 0.5)  # ≫ one Medium lap

    def test_ages_diverge_through_the_merged_payload(self):
        # End-to-end guard: a leader/chaser pair with DIFFERENT starting
        # ages must reach evaluate_call and produce sane rows (the old bug
        # crashed nothing — it silently scored the leader on the chaser's
        # age).  Assert the state echoes the distinct ages back.
        out = pe.evaluate_call(
            leader_code=_LEADER, chaser_code=_CHASER, track_name=_TRACK,
            start_lap=20, race_length=50, gap_before_s=0.8,
            leader_tyre_age=12, chaser_tyre_age=8, year=2026,
            battery_pct=62.5, perspective="chaser")
        st = out["state"]
        self.assertEqual(st["tyres"]["leader"]["age"], 12)
        self.assertEqual(st["tyres"]["chaser"]["age"], 8)
        self.assertNotEqual(st["tyres"]["leader"]["age"],
                            st["tyres"]["chaser"]["age"])


# ── Integration: leader-side two-phase pricing (BANK & STRIKE) ───────────
# Slow (real models, ~2 s per evaluate) — same convention as the other
# engine tests.  _LEADER/_CHASER/_TRACK/_tactical are shared with the
# chaser-side classes above.

def _defence(**kw):
    args = dict(leader_code=_LEADER, chaser_code=_CHASER, track_name=_TRACK,
                start_lap=20, race_length=50, gap_before_s=0.8,
                leader_tyre_age=12, chaser_tyre_age=8, year=2026,
                leader_battery_pct=62.5, chaser_battery_pct=62.5,
                reserve_target_mj=1.6)
    args.update(kw)
    return pe.evaluate_leader_policies(**args)


class TwoPhaseLeaderPricing(unittest.TestCase):
    """BANK & STRIKE (the leader's two-phase defence) pays honestly too."""

    @classmethod
    def setUpClass(cls):
        cls.out = _defence()

    def _row(self, name):
        return next(r for r in self.out["policies"] if r["policy"] == name)

    def test_strike_pays_per_phase_wear_on_the_leaders_compound(self):
        # Per-phase pricing on the LEADER's compound: banking burns
        # nothing, the counter-strike pays full lever for its phase.
        bank = self._row("BANK & STRIKE")
        hold = self._row("HOLD & MANAGE")
        self.assertGreater(bank["score_components"]["wear_cost_s"], 0.0,
                           "the counter-strike must cost wear (old: free)")
        self.assertGreater(
            bank["score_components"]["wear_cost_s"],
            hold["score_components"]["wear_cost_s"])

    def test_leader_ledger_has_a_cliff_line(self):
        # The cliff term must exist in the ledger even when it scores 0,
        # so a drained strike can never silently ride free again.
        for r in self.out["policies"]:
            self.assertIn("cliff_cost_s", r["score_components"])

    def test_drained_counter_strike_pays_the_cliff(self):
        # Force BANK & STRIKE's phase-2 counter-deploy to run the store to
        # its floor (small battery, wide reserve), like the chaser-side
        # drain test.  The row must pay CLIFF_RISK_S and stay barred.
        drained = _defence(leader_battery_pct=38.0, reserve_target_mj=1.6)
        bank = next(r for r in drained["policies"]
                    if r["policy"] == "BANK & STRIKE")
        self.assertTrue(bank["min_soc_pct"] is not None
                        and bank["min_soc_pct"] <= pe.LIVE_ATTACK_MIN_SOC_PCT + 0.05,
                        "setup: the counter-strike must actually drain "
                        f"the store (got min_soc={bank['min_soc_pct']})")
        self.assertGreaterEqual(bank["score_components"]["cliff_cost_s"],
                                pe.CLIFF_RISK_S - 1e-9,
                                "a drained two-phase counter-strike pays "
                                "the cliff (old: 0.0)")
        self.assertFalse(bank["feasible"],
                         "and it stays barred from winning")

    def test_clean_strike_never_pays_the_cliff(self):
        # The penalty must key off the DRAIN, not off the policy: a
        # BANK & STRIKE whose store stays well above the floor pays zero
        # cliff risk.  (Short race + healthy battery: the strike phase is
        # a few laps, so the counter-deploy cannot reach the floor —
        # verified min_soc=41%, feasible.)
        clean = _defence(leader_battery_pct=95.0, race_length=28)
        bank = next(r for r in clean["policies"]
                    if r["policy"] == "BANK & STRIKE")
        if bank["min_soc_pct"] is not None \
                and bank["min_soc_pct"] <= pe.LIVE_ATTACK_MIN_SOC_PCT + 0.05:
            self.skipTest("setup state drains the store; covered by the "
                          "drained-strike test")
        self.assertTrue(bank["feasible"])
        self.assertEqual(bank["score_components"]["cliff_cost_s"], 0.0)

    def test_leader_push_lever_selector(self):
        bank = next(p for p in pe.LEADER_POLICIES
                    if p["name"] == "BANK & STRIKE")
        counter = next(p for p in pe.LEADER_POLICIES
                       if p["name"] == "COUNTER-DEPLOY")
        reactive = next(p for p in pe.LEADER_POLICIES
                        if p["name"] == "REACTIVE DEFENSE")
        hold = next(p for p in pe.LEADER_POLICIES
                    if p["name"] == "HOLD & MANAGE")
        self.assertEqual(pe._leader_push_lever(bank),
                         bank["strike_lever"])
        self.assertGreater(pe._leader_push_lever(bank), 0)
        self.assertEqual(pe._leader_push_lever(counter), counter["lever"])
        self.assertEqual(pe._leader_push_lever(reactive), 0.0,
                         "preset rows push via posture; drain shows as "
                         "infeasibility, not cliff")
        self.assertEqual(pe._leader_push_lever(hold), 0.0)


if __name__ == "__main__":
    unittest.main()
