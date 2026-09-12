"""Unit tests for AI Race Engineer pit wall strategy transceiver.

Verifies:
  - Telemetry context extraction from trained artifacts (fuel burn, tyre wear, isotonic calibration)
  - Chief Race Strategist system and user prompt formulation
  - Deterministic racecraft fallback generation for both Chaser and Leader seats
  - Structured output schemas (radio transmission, definitive call, telemetry proof matrix, directives, contingency)
  - Multi-provider fallback and offline fallback behavior
  - Flask API endpoint /api/ai-race-engineer integration
"""

import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import ai_race_engineer
from dashboard import app


class TestAiRaceEngineer(unittest.TestCase):
    """Test suite for AI Race Engineer transceiver logic."""

    def setUp(self):
        self.state = {
            "leader_code": "RUS",
            "chaser_code": "LEC",
            "track_name": "Albert Park Circuit",
            "year": 2026,
            "start_lap": 20,
            "race_length": 57,
            "gap_before_s": 0.8,
            "leader_tyre_compound": "Medium",
            "chaser_tyre_compound": "Medium",
            "leader_tyre_age": 12.0,
            "chaser_tyre_age": 8.0,
            "battery_pct": 62.5,
            "battery_band_pct": 4.0,
            "reserve_target_pct": 40.0,
            "reserve_target_mj": 1.6,
            "perspective": "chaser",
        }
        self.call_result = {
            "final_call": {
                "seat": "chaser",
                "action": "TACTICAL STALK",
                "reason": "Wait for tyre wear window, execute with full battery margin",
                "action_card": {
                    "action": "TACTICAL STALK",
                    "recommended_lap": 22,
                    "confidence_pct": 85,
                    "calibrated_probability": 0.78,
                    "battery_margin_pct": 22.0,
                    "projected_finish_delta_s": -1.45,
                }
            },
            "chaser": {
                "policies": [
                    {
                        "action": "TACTICAL STALK",
                        "feasible": True,
                        "score_s": -1.45,
                        "projected_finish_delta_s": -1.45,
                        "calibrated_probability": 0.78,
                        "projected_pass_lap": 22,
                        "battery_margin_worst_pct": 18.0,
                    },
                    {
                        "action": "GREEDY ATTACK",
                        "feasible": False,
                        "infeasible_reason": "Drains battery below 10% management reserve floor",
                        "score_s": 4.5,
                    }
                ]
            },
            "leader": {
                "policies": [
                    {
                        "action": "BALANCED HOLD",
                        "feasible": True,
                        "score_s": 0.0,
                    }
                ]
            },
            "coupling": {
                "chaser_posture": "balanced",
                "leader_posture": "defend",
            }
        }

    def test_load_model_telemetry_context(self):
        """Verify telemetry grounding extraction from artifacts."""
        ctx = ai_race_engineer.load_model_telemetry_context("Albert Park")
        self.assertAlmostEqual(ctx["fuel_burn_rate_s_per_lap"], -0.1355, places=4)
        self.assertIn("Medium", ctx["compound_wear_rates"])
        self.assertIn("Soft", ctx["compound_wear_rates"])
        self.assertIn("Hard", ctx["compound_wear_rates"])
        self.assertIn("overtake_model_info", ctx)
        self.assertIn("track_energy_profile", ctx)

    def test_build_prompts(self):
        """Verify prompt construction with full grounding."""
        sys_prompt = ai_race_engineer.build_system_prompt()
        self.assertIn("Chief Race Strategist", sys_prompt)
        self.assertIn("pit-to-car radio", sys_prompt)

        user_prompt = ai_race_engineer.build_user_prompt(
            self.state, self.call_result, perspective="chaser"
        )
        self.assertIn("Albert Park Circuit", user_prompt)
        self.assertIn("LEC", user_prompt)
        self.assertIn("RUS", user_prompt)
        self.assertIn("TACTICAL STALK", user_prompt)
        self.assertIn("10% management reserve floor", user_prompt)

    def test_deterministic_call_chaser(self):
        """Verify deterministic fallback for Chaser seat."""
        res = ai_race_engineer.generate_deterministic_call(
            self.state, self.call_result, perspective="chaser"
        )
        self.assertIn("definitive_call", res)
        self.assertIn("radio_transmission", res)
        self.assertIn("tactical_rationale", res)
        self.assertIn("telemetry_proof", res)
        self.assertIn("driver_directives", res)
        self.assertIn("contingency_protocol", res)
        self.assertEqual(len(res["telemetry_proof"]), 4)
        self.assertEqual(len(res["driver_directives"]), 3)
        self.assertIn("LEC", res["radio_transmission"])

    def test_deterministic_call_leader(self):
        """Verify deterministic fallback for Leader seat."""
        res = ai_race_engineer.generate_deterministic_call(
            self.state, self.call_result, perspective="leader"
        )
        self.assertIn("definitive_call", res)
        self.assertIn("radio_transmission", res)
        self.assertIn("RUS", res["definitive_call"])

    def test_consult_without_api_key(self):
        """Verify consulting without key returns deterministic fallback with api_key_configured=False."""
        res = ai_race_engineer.consult_ai_race_engineer(
            self.state, self.call_result, api_key=None
        )
        self.assertFalse(res["api_key_configured"])
        self.assertEqual(res["mode"], "deterministic_offline")
        self.assertIn("radio_transmission", res)

    def test_api_route_ai_race_engineer(self):
        """Verify the Flask /api/ai-race-engineer endpoint."""
        client = app.test_client()
        payload = {
            "state": self.state,
            "call_result": self.call_result,
            "api_key": "",
            "provider": "auto"
        }
        resp = client.post(
            "/api/ai-race-engineer",
            data=json.dumps(payload),
            content_type="application/json"
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("definitive_call", data)
        self.assertIn("radio_transmission", data)
        self.assertIn("tactical_rationale", data)
        self.assertIn("telemetry_proof", data)


if __name__ == "__main__":
    unittest.main()
