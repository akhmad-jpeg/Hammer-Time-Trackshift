"""Unit tests for the P0 audit fixes (IMPLEMENTATION_PLAN.md, 2026-09-09).

Covers the four red-team findings the live audit proved against the real
dashboard + models:

  1. energy-sandbox 500-crash  — near-zero deploy budget + all-negative
     sector deltas divided by zero (dashboard.energy_sandbox).  Now returns
     HTTP 200 with result.status='degraded'.
  2. chaser_battery_pct ignored — a 31% battery override must change the
     projection (SOC trajectory AND, when it drains to the floor, the
     verdict), not just the meta bookkeeping.
  3. posture-blind ATTACK verdict — a full-save posture (chaser_ers < 0)
     and a push posture that drains the store to its 30% floor must NOT
     both print "attack"; the gate downgrades them to "hold" with a reason.
  4. energy_diff_mj hardcoded to 0.0 in the live forward sim — the P0
     classifier must receive the projected chaser-minus-leader store delta
     (clipped to the training domain) so battery state moves P(pass).

Run:
    python -m unittest discover -s scripts/tests -v
    (or) python -m unittest scripts.tests.test_p0_fixes -v
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import overtake_inference as oi            # noqa: E402
import dashboard                            # noqa: E402
from dashboard import app                   # noqa: E402  (loads models + config)

def _find_sandbox_session():
    try:
        conn = dashboard.get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT s.session_id
            FROM sessions s
            JOIN laps l ON s.session_id = l.session_id
            WHERE l.lap_time_ms > 0
            GROUP BY s.session_id
            HAVING count(l.lap_id) >= 10
            ORDER BY s.session_id DESC
            LIMIT 1
        """)
        row = cursor.fetchone()
        cursor.close()
        conn.close()
        if row:
            return row['session_id']
    except Exception:
        pass
    return 989

SANDBOX_SESSION = _find_sandbox_session()

_LEADER, _CHASER = "VER", "HAM"
_TRACK = "Autodromo Nazionale di Monza"


def _live_call(battery_pct=None, ers=None, gap=0.8, start_lap=20,
               race_length=50):
    return oi.simulate_live_call(
        leader_code=_LEADER, chaser_code=_CHASER, track_name=_TRACK,
        start_lap=start_lap, race_length=race_length, gap_before_s=gap,
        leader_tyre_compound="Medium", chaser_tyre_compound="Medium",
        leader_tyre_age=10, chaser_tyre_age=10, year=2026,
        chaser_ers=ers, chaser_battery_pct=battery_pct,
    )


class SandboxReallocHelper(unittest.TestCase):
    """Fix 1a: the extracted reallocation helper directly."""

    def test_all_negative_deltas_with_budget_falls_back_to_even_split(self):
        shifted, eff, degraded = dashboard._sandbox_realloc(
            3.0, [1.0, 1.0, 1.0], [-3.0, -3.0, -3.0])
        self.assertIsNone(degraded)
        self.assertEqual(shifted, [1.0, 1.0, 1.0])

    def test_zero_budget_all_negative_deltas_is_degraded_not_crash(self):
        # Pre-fix: ZeroDivisionError (0/0 at the normalisation divide).
        shifted, eff, degraded = dashboard._sandbox_realloc(
            0.0, [0.0, 0.0, 0.0], [-3.0, -3.0, -3.0])
        self.assertEqual(shifted, [0.0, 0.0, 0.0])
        self.assertIsNotNone(degraded)
        self.assertIn("no budget to reallocate", degraded)

    def test_zero_budget_zero_deltas_is_degraded(self):
        shifted, eff, degraded = dashboard._sandbox_realloc(
            0.0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        self.assertEqual(shifted, [0.0, 0.0, 0.0])
        self.assertIsNotNone(degraded)

    def test_redistribution_is_zero_sum(self):
        base = [1.2, 0.9, 1.55]
        shifted, eff, degraded = dashboard._sandbox_realloc(
            3.65, base, [0.5, -0.2, 0.0])
        self.assertIsNone(degraded)
        self.assertAlmostEqual(sum(shifted), 3.65, places=6)

    def test_partial_clamp_redistributes_remainder(self):
        base = [1.0, 1.0, 1.0]
        shifted, eff, degraded = dashboard._sandbox_realloc(
            3.0, base, [-0.5, 0.5, 0.0])
        self.assertIsNone(degraded)
        self.assertAlmostEqual(sum(shifted), 3.0, places=6)
        # req = [0.5, 1.5, 1.0] -> each shifted = budget * req / tot
        self.assertAlmostEqual(shifted[0], 3.0 * 0.5 / 3.0, places=6)
        self.assertAlmostEqual(shifted[1], 3.0 * 1.5 / 3.0, places=6)


class SandboxDegradedPayload(unittest.TestCase):
    """Fix 1: all-negative deltas with a ~zero budget must not 500."""

    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()

    def _post(self, payload):
        resp = self.client.post("/api/strategy/energy-sandbox", json=payload)
        return resp.status_code, resp.get_json()

    def test_near_empty_battery_all_negative_deltas_is_200(self):
        # Pre-fix this raised float division by zero -> HTTP 500 (verified
        # against the live server; traceback pointed at the normalisation
        # divide in energy_sandbox).  The route must answer 200 either way.
        code, body = self._post({
            "session_id": SANDBOX_SESSION, "current_lap": 10,
            "mode": "balanced",
            "deltas_mj": {"s1": -3.0, "s2": -3.0, "s3": -3.0},
            "battery_mj": 0.4,
        })
        self.assertEqual(code, 200, f"expected 200, got {code}: {body}")
        result = body["result"]
        if result["status"] == "degraded":
            self.assertTrue(result["degraded_reason"])
            warn_codes = {w["code"] for w in result["warnings"]}
            self.assertIn("no_budget_to_reallocate", warn_codes)
            self.assertAlmostEqual(result["deploy_mj"], 0.0, places=3)
        else:
            # The mode still found a budget this lap: fine — the point is
            # the route never 500s and the payload stays coherent.
            self.assertEqual(result["status"], "ok")
            self.assertIsNone(result["degraded_reason"])
            self.assertGreaterEqual(result["deploy_mj"], 0.0)

    def test_healthy_lap_negative_deltas_still_ok(self):
        code, body = self._post({
            "session_id": SANDBOX_SESSION, "current_lap": 10,
            "mode": "balanced",
            "deltas_mj": {"s1": -0.5, "s2": -0.5, "s3": -0.5},
        })
        self.assertEqual(code, 200)
        self.assertEqual(body["result"]["status"], "ok")
        self.assertIsNone(body["result"]["degraded_reason"])

    def test_zero_budget_zero_deltas_does_not_crash(self):
        # budget ~0 with NO deltas: the old guard reset req to budget/3 == 0
        # and divided by the stale tot_req == 0 even without slider input.
        code, body = self._post({
            "session_id": SANDBOX_SESSION, "current_lap": 10,
            "mode": "balanced", "deltas_mj": {}, "battery_mj": 0.4,
        })
        self.assertEqual(code, 200, body)


class BatteryOverrideChangesProjection(unittest.TestCase):
    """Fix 2: chaser_battery_pct must flow into the projection."""

    def test_low_battery_changes_soc_trajectory(self):
        hi = _live_call(battery_pct=90, ers=40)
        lo = _live_call(battery_pct=31, ers=40)
        soc_hi = [l["chaser_soc_pct"] for l in hi["laps"]]
        soc_lo = [l["chaser_soc_pct"] for l in lo["laps"]]
        self.assertNotEqual(soc_hi, soc_lo,
                            "battery override did not change SOC trajectory")
        self.assertLess(min(soc_lo), min(soc_hi))
        # The override actually landed in meta.
        self.assertEqual(hi["meta"]["ers"]["chaser_battery_start_pct"], 90.0)
        self.assertEqual(lo["meta"]["ers"]["chaser_battery_start_pct"], 31.0)

    def test_low_battery_draining_to_floor_downgrades_to_hold(self):
        # 31% start + +40% ERS: the audit-measured run ends AT the 30%
        # floor while still projecting a pass — that is not a clean attack.
        r = _live_call(battery_pct=31, ers=40)
        gate = r["summary"]["attack_gate"]
        if gate["drained_at_pass"]:
            self.assertEqual(r["call"]["verdict"], "hold")
            self.assertIn("verdict_reason", r["call"])
        else:
            # If the posture genuinely keeps reserve, attack is legitimate.
            self.assertEqual(r["call"]["verdict"], "attack")
            self.assertFalse(gate["drained_at_pass"])


class VerdictGate(unittest.TestCase):
    """Fix 3: the verdict must respect posture and battery, not just cum."""

    def test_saving_posture_is_never_attack(self):
        for ers in (-100, -50):
            with self.subTest(ers=ers):
                r = _live_call(ers=ers)
                self.assertEqual(r["call"]["verdict"], "hold")
                self.assertIn("saving", r["call"]["verdict_reason"])
                self.assertTrue(r["summary"]["attack_gate"]["saving_posture"])

    def test_push_with_reserve_is_attack(self):
        r = _live_call(battery_pct=90, ers=40)
        self.assertEqual(r["call"]["verdict"], "attack")
        self.assertFalse(r["summary"]["attack_gate"]["drained_at_pass"])
        self.assertFalse(r["summary"]["attack_gate"]["saving_posture"])

    def test_no_ers_verdict_unchanged_and_gate_absent(self):
        # Lever off: historical behaviour (attack/attempt/no_window) and no
        # gate fields — the backtest harness runs exactly this path, so its
        # calibration stats are untouched by the fix.
        r = _live_call()
        self.assertIn(r["call"]["verdict"], ("attack", "attempt", "no_window"))
        self.assertNotIn("attack_gate", r["summary"])
        self.assertEqual(r["meta"]["ers"]["chaser"], "balanced")

    def test_gate_constants_sane(self):
        self.assertEqual(oi.LIVE_ATTACK_MIN_SOC_PCT, 30.0)
        self.assertLess(oi.LIVE_SOC_EPS_PCT, 0.5)


class EnergyDiffCoupling(unittest.TestCase):
    """Fix 4: the P0 classifier must see the projected energy delta."""

    def test_no_lever_imputes_zero_and_flags_it(self):
        r = _live_call()
        self.assertTrue(all(l["energy_diff_mj"] == 0.0 for l in r["laps"]))
        self.assertTrue(all(l["energy_imputed"] for l in r["laps"]))
        self.assertTrue(r["meta"]["energy_feature"]["imputed"])

    def test_lever_on_feeds_real_delta_into_window_laps(self):
        r = _live_call(battery_pct=62.5, ers=100)
        window_laps = [l for l in r["laps"] if l["in_window"]]
        self.assertTrue(window_laps)
        # Deployment spends the store: the feature goes NEGATIVE vs the
        # Balanced leader (within the trained clip domain).
        self.assertTrue(any(l["energy_diff_mj"] < 0.0 for l in window_laps))
        self.assertFalse(r["meta"]["energy_feature"]["imputed"])
        self.assertTrue(all(not l["energy_imputed"] for l in window_laps))

    def test_energy_diff_stays_inside_training_domain(self):
        for ers in (-100, -50, 50, 100):
            with self.subTest(ers=ers):
                r = _live_call(ers=ers)
                for l in r["laps"]:
                    self.assertGreaterEqual(l["energy_diff_mj"],
                                            oi.ENERGY_DIFF_CLIP_MIN)
                    self.assertLessEqual(l["energy_diff_mj"],
                                         oi.ENERGY_DIFF_CLIP_MAX)

    def test_push_changes_overtake_probability_vs_no_lever(self):
        # The coupling must actually move the classifier output, not just
        # decorate the rows: compare in-window probabilities with and
        # without the lever (pace also changes, but the energy feature is
        # the point under test — assert the deltas differ somewhere).
        off = _live_call()
        on = _live_call(ers=100)
        off_rows = [(l["gap_before_s"], l["pace_gap_s"], l["overtake_probability"])
                    for l in off["laps"] if l["in_window"]]
        on_rows = [(l["gap_before_s"], l["pace_gap_s"], l["overtake_probability"])
                   for l in on["laps"] if l["in_window"]]
        self.assertNotEqual(
            [r[2] for r in off_rows], [r[2] for r in on_rows],
            "overtake probabilities identical with and without the ERS lever")


if __name__ == "__main__":
    unittest.main()
