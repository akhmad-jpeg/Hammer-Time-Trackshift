"""Unit tests for Dynamic Safety Car (VSC / Full SC) & Pit Loss Delta Model.

Verifies:
  - Circuit pit loss lookup and normalization across 32 circuits
  - Effective pit loss calculations under Green, VSC, and Full SC
  - Fresh tyre undercut window evaluation and exit margin math
  - Safety car historical probability risk tiers and opportunity value
  - Policy Engine integration with race_event ('vsc', 'safety_car') and traffic_level
  - AI Race Engineer pit-to-car radio transmission under VSC/SC conditions
  - Flask API /api/strategy/call integration with pit parameters
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import pit_strategy
import policy_engine
import ai_race_engineer
from dashboard import app


class TestPitStrategy(unittest.TestCase):
    """Test suite for pit strategy mechanics, circuit catalog, and policy engine coupling."""

    def test_circuit_lookup(self):
        """Verify circuit pit loss and historical SC probabilities for major venues."""
        monaco = pit_strategy.get_circuit_pit_profile("Monaco")
        self.assertEqual(monaco["canonical"], "Monaco")
        self.assertAlmostEqual(monaco["pit_loss_green_s"], 19.4, places=1)
        self.assertAlmostEqual(monaco["pit_loss_vsc_s"], 10.8, places=1)
        self.assertAlmostEqual(monaco["pit_loss_sc_s"], 8.5, places=1)
        self.assertGreaterEqual(monaco["sc_probability"], 0.70)
        self.assertEqual(monaco["sc_risk_tier"], "HIGH")

        silverstone = pit_strategy.get_circuit_pit_profile("Silverstone Circuit")
        self.assertEqual(silverstone["canonical"], "Silverstone Circuit")
        self.assertAlmostEqual(silverstone["pit_loss_green_s"], 22.2, places=1)

        singapore = pit_strategy.get_circuit_pit_profile("Marina Bay Street Circuit")
        self.assertEqual(singapore["canonical"], "Marina Bay Street Circuit")
        self.assertAlmostEqual(singapore["sc_probability"], 1.00, places=1)
        self.assertEqual(singapore["sc_risk_tier"], "HIGH")

        # Unknown track fallback
        unknown = pit_strategy.get_circuit_pit_profile("Nonexistent Grand Prix")
        self.assertAlmostEqual(unknown["pit_loss_green_s"], 21.5, places=1)
        self.assertEqual(unknown["sc_risk_tier"], "MEDIUM")

    def test_calculate_pit_loss(self):
        """Test net pit loss and seconds saved under Green, VSC, and Safety Car."""
        green = pit_strategy.calculate_pit_loss("Silverstone", event="green", traffic="Light")
        self.assertFalse(green["is_cheap_stop"])
        self.assertAlmostEqual(green["time_saved_s"], 0.0, places=1)
        self.assertAlmostEqual(green["effective_pit_loss_s"], 22.2, places=1)

        vsc = pit_strategy.calculate_pit_loss("Silverstone", event="vsc", traffic="Light")
        self.assertTrue(vsc["is_cheap_stop"])
        self.assertAlmostEqual(vsc["effective_pit_loss_s"], 12.8, places=1)
        self.assertAlmostEqual(vsc["time_saved_s"], 9.4, places=1)

        sc = pit_strategy.calculate_pit_loss("Silverstone", event="safety_car", traffic="Light")
        self.assertTrue(sc["is_cheap_stop"])
        self.assertAlmostEqual(sc["effective_pit_loss_s"], 9.8, places=1)
        self.assertAlmostEqual(sc["time_saved_s"], 12.4, places=1)

        # Traffic multiplier check
        heavy = pit_strategy.calculate_pit_loss("Silverstone", event="green", traffic="Heavy")
        self.assertGreater(heavy["effective_pit_loss_s"], green["effective_pit_loss_s"])

    def test_evaluate_undercut_window(self):
        """Test undercut feasibility based on gap and tyre age delta."""
        # Gap 0.8s, chaser tyres 12 laps old vs leader 12 laps: fresh rubber gives ~1.8s edge -> OPEN_FAVORABLE
        open_window = pit_strategy.evaluate_undercut_window(
            gap_s=0.8,
            chaser_tyre="Medium",
            chaser_age=12.0,
            leader_tyre="Medium",
            leader_age=12.0,
            track_name="Silverstone",
        )
        self.assertEqual(open_window["status"], "OPEN_FAVORABLE")
        self.assertGreater(open_window["net_exit_margin_s"], 0.0)

        # Gap 4.5s: too far to undercut in 1 lap without leader pit error
        closed_window = pit_strategy.evaluate_undercut_window(
            gap_s=4.5,
            chaser_tyre="Medium",
            chaser_age=12.0,
            leader_tyre="Medium",
            leader_age=12.0,
            track_name="Silverstone",
        )
        self.assertEqual(closed_window["status"], "CLOSED_TOO_FAR")
        self.assertLess(closed_window["net_exit_margin_s"], 0.0)

    def test_policy_engine_vsc_coupling(self):
        """Test that policy_engine evaluates VSC cheap stops and awards time savings."""
        call = policy_engine.evaluate_call(
            leader_code="RUS",
            chaser_code="LEC",
            track_name="Silverstone Circuit",
            start_lap=25,
            race_length=52,
            gap_before_s=1.2,
            leader_tyre_compound="Medium",
            chaser_tyre_compound="Medium",
            leader_tyre_age=18.0,
            chaser_tyre_age=18.0,
            perspective="chaser",
            race_event="vsc",
            traffic_level="Clear",
        )
        self.assertIn("pit_analysis", call)
        pa = call["pit_analysis"]
        self.assertEqual(pa["race_event"].lower(), "vsc")
        self.assertTrue(pa["is_cheap_stop"])
        self.assertGreater(pa["time_saved_s"], 8.0)

        # Under VSC, UNDERCUT PREP should reflect the cheap stop in why text and score
        chaser_policies = [p for p in call["policies"] if p.get("policy") == "UNDERCUT PREP" and p.get("seat") == "chaser"]
        self.assertTrue(len(chaser_policies) > 0)
        prep = chaser_policies[0]
        self.assertIn("BOX UNDER VSC", prep["why"])
        self.assertGreater(prep["score_components"]["pass_gain_s"], 8.0)

    def test_ai_race_engineer_vsc_radio(self):
        """Verify AI race engineer deterministic radio calls prioritize boxing under VSC."""
        state = {
            "leader_code": "RUS",
            "chaser_code": "LEC",
            "track_name": "Silverstone Circuit",
            "start_lap": 25,
            "race_length": 52,
            "gap_before_s": 1.2,
            "perspective": "chaser",
            "race_event": "vsc",
        }
        call_result = policy_engine.evaluate_call(
            leader_code="RUS",
            chaser_code="LEC",
            track_name="Silverstone Circuit",
            start_lap=25,
            race_length=52,
            gap_before_s=1.2,
            perspective="chaser",
            race_event="vsc",
        )
        radio_out = ai_race_engineer.generate_deterministic_call(state, call_result, perspective="chaser")
        self.assertIn("Box, box, box under VSC", radio_out["radio_transmission"])
        self.assertTrue(any("Pit Delta:" in p for p in radio_out["telemetry_proof"]))
        self.assertTrue(any("BOX THIS LAP" in d for d in radio_out["driver_directives"]))

    def test_dashboard_api_strategy_call(self):
        """Test Flask API /api/strategy/call with race_event and traffic_level."""
        client = app.test_client()
        res = client.post("/api/strategy/call", json={
            "leader_code": "VER",
            "chaser_code": "HAM",
            "track_name": "Silverstone Circuit",
            "start_lap": 30,
            "race_length": 52,
            "gap_before_s": 1.5,
            "perspective": "chaser",
            "race_event": "vsc",
            "traffic_level": "Light",
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("pit_analysis", data)
        self.assertEqual(data["pit_analysis"]["race_event"].lower(), "vsc")
        self.assertTrue(data["pit_analysis"]["is_cheap_stop"])


if __name__ == "__main__":
    unittest.main()
