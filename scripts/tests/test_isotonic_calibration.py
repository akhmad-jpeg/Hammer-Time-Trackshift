"""Unit tests for Isotonic Calibration & Reliability Table (P1-4).

Verifies:
  1. Overtake model isotonic calibrator loading & prediction.
  2. Prediction contract: raw score stays in ``overtake_probability``
     (the scale every decision threshold is tuned on) and the isotonic
     mapping lands in ``calibrated_probability``.
  3. Calibration improves Brier/ECE on the held-out split.
  4. Reliability bins structure and calibration metrics.
  5. /api/overtake/reliability route returns 200 with valid schema.
  6. Graceful degradation when calibrator is absent.
  7. TIME-ORDERED split: the calibrator is fitted on LATER seasons the
     model never saw (no future leakage), as recorded in model_info.json
     and surfaced through the endpoint.
"""

import sys
import json
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import overtake_inference as oi
from dashboard import app


class TestIsotonicCalibration(unittest.TestCase):

    def test_load_reliability_table(self):
        cal_info, pkl_exists = oi.load_reliability_table()
        self.assertIsNotNone(cal_info, "model_info.json should contain isotonic_calibration")
        self.assertTrue(pkl_exists, "isotonic_calibrator.pkl should exist in ml_models/overtake/")
        self.assertIn("brier_raw", cal_info)
        self.assertIn("brier_calibrated", cal_info)
        self.assertIn("reliability_bins", cal_info)
        self.assertGreater(len(cal_info["reliability_bins"]), 0)
        # Brier score should improve or stay within tolerance
        self.assertLessEqual(cal_info["brier_calibrated"], cal_info["brier_raw"] + 1e-5)
        # ECE likewise
        self.assertLessEqual(cal_info["ece_calibrated"], cal_info["ece_raw"] + 1e-5)
        for b in cal_info["reliability_bins"]:
            self.assertIn("bin_lo", b)
            self.assertIn("bin_hi", b)
            self.assertIn("mean_predicted", b)
            self.assertIn("fraction_positive", b)
            self.assertIn("n_samples", b)
            self.assertIn("thin", b)

    def test_predict_overtake_two_scale_contract(self):
        """Raw score stays in overtake_probability; calibrator output lands
        in calibrated_probability.  Thresholds (0.5 trigger etc.) are tuned
        on the raw scale, so the two must never silently swap."""
        res = oi.predict_overtake(
            gap_before_s=0.8,
            pace_gap_s=-0.3,
            chaser_tyre_age=5,
            leader_tyre_age=15,
            chaser_tyre_compound="Soft",
            leader_tyre_compound="Medium",
            fuel_diff_kg=-5.0,
            energy_diff_mj=0.8,
            lap_number=20,
            track_name="Monza",
            year=2026,
        )
        for key in ("overtake_probability", "raw_overtake_probability",
                    "calibrated_probability", "calibrated", "track_covered"):
            self.assertIn(key, res)
        self.assertTrue(res["calibrated"])
        self.assertAlmostEqual(res["raw_overtake_probability"],
                               res["overtake_probability"], places=9)
        for p in (res["overtake_probability"], res["calibrated_probability"]):
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)
        # The isotonic map is monotone and (on this model) strictly
        # compressive above ~0.1 raw; at minimum the two scales must be
        # distinguishable fields, not a single overwritten value.
        self.assertIsNot(res["calibrated_probability"], None)

    def test_time_ordered_split_recorded(self):
        """The trainer must document an out-of-time split: the isotonic
        calibrator is fitted on LATER seasons than the model trained on.
        A random split would let future-regime races leak into the fit and
        make the reliability table describe interpolation, not the live
        use case (predicting forward).  Falls back to recording the random
        fallback honestly if the DB ever lacks enough held-out seasons."""
        info_path = oi.OVERTAKE_MODEL_DIR / "model_info.json"
        if not info_path.exists():
            self.skipTest("model_info.json not present")
        with open(info_path, encoding="utf-8") as f:
            meta = json.load(f)
        split = meta.get("split") or {}
        cal = meta.get("isotonic_calibration") or {}
        if split.get("type") != "time_ordered":
            self.fail(
                f"model was trained on a '{split.get('type')}' split — "
                "retrain with the time-ordered split (train < cutoff year, "
                "hold out later seasons) so the reliability table reflects "
                "true out-of-time calibration")
        cutoff = split["cutoff_year"]
        self.assertTrue(all(y < cutoff for y in split["train_years"]),
                        "train years must precede the cutoff")
        self.assertTrue(all(y >= cutoff for y in split["test_years"]),
                        "held-out years must be at/after the cutoff")
        self.assertGreater(split["train_positives"], 0)
        self.assertGreater(split["test_positives"], 0)
        # The calibrator's own provenance must name the held-out seasons.
        fitted_on = str(cal.get("fitted_on", ""))
        self.assertIn("time-ordered", fitted_on)
        for y in (split["test_years"][0], split["test_years"][-1]):
            self.assertIn(str(y), fitted_on,
                          f"fitted_on must name the held-out season {y}")

    def test_reliability_endpoint_surfaces_split(self):
        client = app.test_client()
        resp = client.get("/api/overtake/reliability")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        if not data.get("isotonic_fitted"):
            self.skipTest("calibrator not fitted")
        self.assertIn("time-ordered", data.get("fitted_on", ""))

    def test_reliability_endpoint(self):
        client = app.test_client()
        resp = client.get("/api/overtake/reliability")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data.get("isotonic_fitted"))
        self.assertTrue(data.get("calibrated"))
        self.assertIn("reliability_bins", data)
        self.assertIn("brier_improvement", data)
        self.assertGreater(len(data["reliability_bins"]), 0)
        for b in data["reliability_bins"]:
            self.assertIn("bin_lo", b)
            self.assertIn("fraction_positive", b)
            self.assertIn("n_samples", b)

    def test_reliability_endpoint_graceful_on_missing_dir(self):
        """The route reads the DEFAULT model dir; point the module constant
        at an empty dir and confirm it degrades instead of raising."""
        import dashboard
        original = oi.OVERTAKE_MODEL_DIR
        try:
            oi.OVERTAKE_MODEL_DIR = Path("/nonexistent/dir")
            client = app.test_client()
            resp = client.get("/api/overtake/reliability")
            self.assertEqual(resp.status_code, 200)
            data = resp.get_json()
            self.assertFalse(data.get("isotonic_fitted"))
            self.assertIn("message", data)
        finally:
            oi.OVERTAKE_MODEL_DIR = original

    def test_graceful_degradation_on_missing_dir(self):
        info, pkl = oi.load_reliability_table(models_dir="/nonexistent/dir")
        self.assertIsNone(info)
        self.assertFalse(pkl)


if __name__ == "__main__":
    unittest.main()
