"""Unit tests for SOC uncertainty bands (P1-3).

The battery is a synthesized estimate — F1 does not broadcast SOC — so the
system must never show fake point precision:

  * one band model (energy_simulator.battery_uncertainty_band): ±2% floor,
    +0.5%/lap drift, ±8% cap — the single source of truth;
  * the energy trace attaches a per-lap band so the chart can shade it;
  * the live call reports SOC end as mean ± band with an explicit range;
  * the policy engine prices the battery at the WORST case of the band and
    degrades recommendation confidence as the band widens.

Run:
    python -m unittest scripts.tests.test_soc_uncertainty -v
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import energy_simulator as es          # noqa: E402
import overtake_inference as oi        # noqa: E402
import policy_engine as pe             # noqa: E402
import dashboard                       # noqa: E402
from dashboard import app              # noqa: E402

_LEADER, _CHASER = "VER", "HAM"
_TRACK = "Autodromo Nazionale di Monza"

def _find_energy_session():
    try:
        conn = dashboard.get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT session_id
            FROM race_state
            WHERE energy_start_mj IS NOT NULL
            LIMIT 1
        """)
        row = cursor.fetchone()
        if row:
            cursor.close()
            conn.close()
            return row['session_id']
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
            sid = row['session_id']
            es.simulate_session_energy(sid, mode='balanced')
            return sid
    except Exception:
        pass
    return 989

_SESSION = _find_energy_session()


class BandModel(unittest.TestCase):
    """The single source of truth: floor, per-lap growth, cap."""

    def test_floor_at_anchor(self):
        b = es.battery_uncertainty_band(0)
        self.assertEqual(b["band_pct"], 2.0)
        self.assertAlmostEqual(b["band_mj"], 0.08, places=3)  # 2% of 4 MJ

    def test_monotone_growth_and_cap(self):
        b1 = es.battery_uncertainty_band(1)["band_pct"]
        b12 = es.battery_uncertainty_band(12)["band_pct"]
        b50 = es.battery_uncertainty_band(50)["band_pct"]
        self.assertGreater(b1, 2.0)
        self.assertAlmostEqual(b12, 8.0, places=6)   # 2 + 12*0.5 hits the cap
        self.assertEqual(b50, 8.0)                    # capped, never more
        self.assertLessEqual(b50, es.SOC_BAND_CAP_PCT)

    def test_negative_laps_clamped(self):
        self.assertEqual(es.battery_uncertainty_band(-5)["band_pct"], 2.0)

    def test_trace_laps_carry_band(self):
        tr = es.project_energy_trace('balanced', 4.0, [0.7] * 20,
                                     'legacy_2014_2025',
                                     pace_dev_per_lap=[0.0] * 20)
        self.assertIn("soc_band_pct", tr["laps"][0])
        self.assertLess(tr["laps"][0]["soc_band_pct"],
                        tr["laps"][19]["soc_band_pct"])
        self.assertEqual(tr["summary"]["final_soc_band_pct"],
                         tr["laps"][19]["soc_band_pct"])


class TraceEndpointBand(unittest.TestCase):
    """The chart payload carries the band per point."""

    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()
        resp = cls.client.get(f"/api/session/{_SESSION}/energy")
        cls.rows = resp.get_json()
        if not cls.rows:
            es.simulate_session_energy(_SESSION, mode='balanced')
            resp = cls.client.get(f"/api/session/{_SESSION}/energy")
            cls.rows = resp.get_json()

    def test_points_carry_band(self):
        self.assertTrue(self.rows)
        for r in self.rows[:5]:
            self.assertIn("band_pct", r)
        # Band grows by lap (lap 1 anchored at the write, later laps drift).
        first = next(r for r in self.rows if r["lap_number"] == 1)
        later = next(r for r in self.rows if r["lap_number"] >= 10)
        self.assertLess(first["band_pct"], later["band_pct"])

    def test_band_within_declared_bounds(self):
        for r in self.rows:
            self.assertGreaterEqual(r["band_pct"], es.SOC_BAND_FLOOR_PCT)
            self.assertLessEqual(r["band_pct"], es.SOC_BAND_CAP_PCT)


class LiveCallBand(unittest.TestCase):
    """SOC at the flag is reported as mean ± band with an explicit range."""

    def test_band_reported_with_lever(self):
        r = oi.simulate_live_call(
            leader_code=_LEADER, chaser_code=_CHASER, track_name=_TRACK,
            start_lap=20, race_length=50, gap_before_s=0.8,
            leader_tyre_compound="Medium", chaser_tyre_compound="Medium",
            leader_tyre_age=10, chaser_tyre_age=10, year=2026,
            chaser_ers=40)
        s = r["summary"]
        self.assertIn("chaser_soc_end_band_pct", s)
        self.assertIn("chaser_soc_end_range_pct", s)
        lo, hi = s["chaser_soc_end_range_pct"]
        self.assertLessEqual(lo, s["chaser_soc_end_pct"] <= hi and hi)
        # 30-lap walk: band = 2 + 0.5*30 -> capped at 8.
        self.assertEqual(s["chaser_soc_end_band_pct"], 8.0)
        self.assertEqual(s["attack_gate"]["band_pct"], 8.0)


class PolicyEngineBand(unittest.TestCase):
    """Worst-case battery pricing + confidence degradation."""

    def _evaluate(self, battery_pct=None, gap=0.8, start_lap=20,
                  race_length=50):
        return pe.evaluate_tactical_policies(
            leader_code=_LEADER, chaser_code=_CHASER, track_name=_TRACK,
            start_lap=start_lap, race_length=race_length, gap_before_s=gap,
            year=2026, chaser_battery_pct=battery_pct)

    def test_state_carries_band_and_range(self):
        r = self._evaluate()
        st = r["state"]
        self.assertIn("battery_band_pct", st)
        self.assertIn("battery_range_pct", st)
        lo, hi = st["battery_range_pct"]
        self.assertLess(lo, st["battery_pct"])
        self.assertLess(st["battery_pct"], hi)

    def test_policies_report_worst_case_margin(self):
        r = self._evaluate()
        for p in r["policies"]:
            self.assertIn("soc_band_pct", p)
            self.assertIn("battery_margin_worst_pct", p)
            if p["battery_margin_worst_pct"] is not None:
                self.assertLessEqual(p["battery_margin_worst_pct"],
                                     p["battery_margin_pct"],
                                     "worst-case margin must not exceed mean")

    def test_longer_horizon_degrades_confidence(self):
        # Same state, longer walk -> wider band -> lower confidence ceiling.
        short = self._evaluate(start_lap=20, race_length=30)
        long_ = self._evaluate(start_lap=20, race_length=100)
        # The band multiplier is the only difference driver here; the
        # confidence must not INCREASE with the wider band on identical
        # margins.  (Scores shift slightly with horizon, so compare the
        # engine's own band multiplier, which is deterministic.)
        self.assertLessEqual(
            long_["scoring_constants"]["confidence_band_multiplier"],
            short["scoring_constants"]["confidence_band_multiplier"])

    def test_band_multiplier_floor(self):
        # Even at the cap the multiplier never drops below 0.6.
        b = es.battery_uncertainty_band(1000)
        mult = 1.0 - 0.4 * ((b["band_pct"] - b["floor_pct"])
                            / (b["cap_pct"] - b["floor_pct"]))
        self.assertGreaterEqual(mult, 0.6)


if __name__ == "__main__":
    unittest.main()
