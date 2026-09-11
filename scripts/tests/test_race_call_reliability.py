"""Unit tests for the race-call reliability report math.

Covers the pure functions of scripts/race_call_reliability.py — threshold
sweep, calibration bins, confusion counts, gap-walk aggregation and the
computed mid-range reading — with hand-checked fixtures.  No DB, no models.

Run:
    python -m unittest scripts.tests.test_race_call_reliability -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from race_call_reliability import (       # noqa: E402
    threshold_table, calibration_bins, confusion, gap_walk_error,
    flat_midrange_note, MIN_BUCKET_N,
)


def _cp(cum, passed, year=2021, mae=None, bias=None):
    c = {"cum_probability": cum, "actual_pass": passed, "year": year}
    if mae is not None:
        c["gap_mae_s"] = mae
        c["gap_bias_s"] = bias
    return c


class ThresholdTableTest(unittest.TestCase):

    def test_pass_rate_and_lift_per_threshold(self):
        # 10 in-battle checkpoints, 2 real passes => base rate 0.2.
        cps = [
            _cp(0.95, 1), _cp(0.85, 1),          # the two passes, both >= 0.8
            _cp(0.6, 0), _cp(0.4, 0), _cp(0.35, 0),
            _cp(0.3, 0), _cp(0.2, 0), _cp(0.15, 0),
            _cp(0.1, 0), _cp(0.05, 0),
        ]
        table, base = threshold_table(cps)
        self.assertAlmostEqual(base, 0.2)
        row80 = next(r for r in table if r["threshold"] == 0.8)
        self.assertEqual(row80["checkpoints"], 2)
        self.assertEqual(row80["actual_passes"], 2)
        self.assertAlmostEqual(row80["pass_rate"], 1.0)
        self.assertAlmostEqual(row80["lift_vs_base"], 5.0)   # 1.0 / 0.2
        row30 = next(r for r in table if r["threshold"] == 0.3)
        self.assertEqual(row30["checkpoints"], 6)
        self.assertEqual(row30["actual_passes"], 2)
        # Artifacts carry 4-dp numbers (clean JSON), so compare the rounded
        # form — this pins the report's actual output, not an idealization.
        self.assertAlmostEqual(row30["pass_rate"], round(2 / 6, 4))

    def test_empty_group_gives_none_not_crash(self):
        table, base = threshold_table([_cp(0.1, 0)])
        self.assertAlmostEqual(base, 0.0)
        top = next(r for r in table if r["threshold"] == 0.8)
        self.assertEqual(top["checkpoints"], 0)
        self.assertIsNone(top["pass_rate"])
        self.assertIsNone(top["lift_vs_base"])

    def test_thin_flag_uses_min_bucket(self):
        cps = [_cp(0.9, 1) for _ in range(MIN_BUCKET_N - 1)]
        table, _ = threshold_table(cps)
        row = next(r for r in table if r["threshold"] == 0.9)
        self.assertTrue(row["thin"])
        cps.append(_cp(0.9, 0))
        table, _ = threshold_table(cps)
        row = next(r for r in table if r["threshold"] == 0.9)
        self.assertFalse(row["thin"])

    def test_default_threshold_present(self):
        # The product's operating point (0.8) must always appear in the table.
        table, _ = threshold_table([_cp(0.85, 1)])
        self.assertIn(0.8, [r["threshold"] for r in table])


class CalibrationBinsTest(unittest.TestCase):

    def test_bins_cover_low_confidence_region(self):
        cps = [_cp(0.05, 0), _cp(0.15, 0), _cp(0.3, 1), _cp(0.6, 0), _cp(0.9, 1)]
        bins = calibration_bins(cps)
        by_label = {b["bin"]: b for b in bins}
        self.assertEqual(by_label["[0.0,0.2)"]["checkpoints"], 2)
        self.assertEqual(by_label["[0.2,0.5)"]["checkpoints"], 1)
        self.assertEqual(by_label["[0.5,0.8)"]["checkpoints"], 1)
        self.assertEqual(by_label["[0.8,1.0)"]["checkpoints"], 1)
        self.assertAlmostEqual(by_label["[0.8,1.0)"]["pass_rate"], 1.0)


class ConfusionTest(unittest.TestCase):

    def test_window_confusion_counts_and_rates(self):
        cps = [
            {"pred_window": True, "actual_window": True},
            {"pred_window": True, "actual_window": True},
            {"pred_window": True, "actual_window": False},
            {"pred_window": False, "actual_window": True},
            {"pred_window": False, "actual_window": False},
            {"pred_window": False, "actual_window": False},
        ]
        c = confusion(cps, "pred_window", "actual_window")
        self.assertEqual((c["tp"], c["fp"], c["fn"], c["tn"]), (2, 1, 1, 2))
        self.assertAlmostEqual(c["precision"], round(2 / 3, 4))
        self.assertAlmostEqual(c["recall"], round(2 / 3, 4))

    def test_no_predicted_positives_gives_none_not_crash(self):
        # No predicted positives AND no actual positives: both 0/0 -> None
        # (a 0/1 recall would be a defined zero, not an undefined rate).
        c = confusion([{"pred_window": False, "actual_window": False}],
                      "pred_window", "actual_window")
        self.assertIsNone(c["precision"])
        self.assertIsNone(c["recall"])

    def test_defined_zero_recall_is_zero_not_none(self):
        # Predicted nothing, but a pass happened: recall is a real 0.0.
        c = confusion([{"pred_window": False, "actual_window": True}],
                      "pred_window", "actual_window")
        self.assertIsNone(c["precision"])
        self.assertEqual(c["recall"], 0.0)


class GapWalkTest(unittest.TestCase):

    def test_mean_mae_and_bias(self):
        cps = [_cp(0.5, 0, mae=1.0, bias=0.5), _cp(0.5, 0, mae=2.0, bias=-0.5)]
        g = gap_walk_error(cps)
        self.assertAlmostEqual(g["mean_mae_s_per_lap"], 1.5)
        self.assertAlmostEqual(g["mean_bias_s_per_lap"], 0.0)
        self.assertEqual(g["checkpoints"], 2)

    def test_none_when_no_scored_laps(self):
        self.assertIsNone(gap_walk_error([_cp(0.5, 0)]))


class FlatMidrangeReadingTest(unittest.TestCase):

    def _table(self, rates):
        return [{"threshold": t, "pass_rate": r}
                for t, r in zip((0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95),
                                rates)]

    def test_flat_mid_and_rising_top_is_flagged(self):
        # 0.30..0.80 within 2pp of each other, 0.95 clearly higher -> note.
        note = flat_midrange_note(self._table(
            [0.18, 0.18, 0.18, 0.18, 0.18, 0.185, 0.19, 0.21]))
        self.assertIsNotNone(note)
        self.assertIn("coarse filter", note)

    def test_monotone_curve_gets_no_caveat(self):
        self.assertIsNone(flat_midrange_note(self._table(
            [0.10, 0.12, 0.14, 0.16, 0.20, 0.26, 0.34, 0.45])))

    def test_flat_everywhere_including_top_gets_no_caveat(self):
        # Flat across the WHOLE sweep is a different (worse) finding — the
        # note is specifically about a coarse-mid/strong-top shape.
        self.assertIsNone(flat_midrange_note(self._table(
            [0.18, 0.18, 0.18, 0.18, 0.18, 0.185, 0.19, 0.19])))


if __name__ == "__main__":
    unittest.main()
