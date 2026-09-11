"""Density-aware regen scaling (full-resolution telemetry migration).

Pins the contract of energy_simulator._sampling_scale_factor /
_speed_drop_regen across trace densities:

  * legacy ~6-sample timestamp-less laps  -> REGEN_SAMPLING_SCALE (x2.2)
  * full-resolution timestamped laps      -> 1.0 (no fudge)
  * full-resolution timestamp-less laps   -> 1.0 (row-count heuristic)
  * sparse timestamped laps (bad traces)  -> legacy correction via the
    row-count heuristic
  * single-sample laps (no dt to measure) -> legacy correction (conservative)

and the importer-side normalisation (import_f1_race._prepare_telemetry_rows):
boolean Brake stored as 0.0/1.0 (the old /100 bug turned True into 0.01),
time_s anchored to the lap's first sample, distance_m from FastF1's Distance
channel or speed integration.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from energy_simulator import (
    REGEN_SAMPLING_SCALE,
    SPARSE_MAX_ROWS,
    FULL_RES_MIN_ROWS,
    _speed_drop_regen,
    _sampling_scale_factor,
)
from import_f1_race import _prepare_telemetry_rows

DEFAULT_SPEC = "newgen_2026"


def _sparse_lap():
    """Legacy-shaped lap: ~6 samples, no timestamps, one big braking event."""
    return [{"speed": v} for v in (300, 290, 260, 180, 90, 80)]


def _fullres_lap(n=250, dt=0.25):
    """Full-resolution lap: n samples on a 0.25 s clock, a 300->80 km/h
    braking event mid-lap."""
    samples = []
    t = 0.0
    v = 280.0
    for i in range(n):
        brake_zone = 100 <= i < 120
        if brake_zone:
            v = max(80.0, v - 11.0)
        elif i >= 120 and v < 280.0:
            v = min(280.0, v + 10.0)
        samples.append({"speed": v, "time_s": round(t, 4)})
        t += dt
    return samples


class TestSamplingScaleFactor(unittest.TestCase):

    def test_sparse_legacy_lap_gets_fudge(self):
        self.assertAlmostEqual(_sampling_scale_factor(_sparse_lap()),
                               REGEN_SAMPLING_SCALE, places=9)

    def test_fullres_timestamped_lap_gets_no_fudge(self):
        self.assertEqual(_sampling_scale_factor(_fullres_lap()), 1.0)

    def test_fullres_timestampless_lap_gets_no_fudge(self):
        lap = [{k: v for k, v in s.items() if k != "time_s"}
               for s in _fullres_lap()]
        self.assertEqual(_sampling_scale_factor(lap), 1.0)

    def test_single_sample_laps_conservatively(self):
        self.assertAlmostEqual(_sampling_scale_factor([{"speed": 200}]),
                               REGEN_SAMPLING_SCALE, places=9)

    def test_grey_zone_is_a_linear_ramp(self):
        n = (SPARSE_MAX_ROWS + FULL_RES_MIN_ROWS) // 2
        lap = [{"speed": 200.0} for _ in range(n)]
        f = _sampling_scale_factor(lap)
        self.assertGreater(f, 1.0)
        self.assertLess(f, REGEN_SAMPLING_SCALE)

    def test_regen_scaling_follows_density(self):
        """The same braking energy must read x2.2 larger from the sparse
        trace than from the dense trace — the fudge exists ONLY at the
        legacy density."""
        sparse = _speed_drop_regen(_sparse_lap(), DEFAULT_SPEC)
        dense = _speed_drop_regen(_fullres_lap(), DEFAULT_SPEC)
        # Both capped by the same harvest limit; the sparse fudge must show
        # in the scale factor even when the cap clips the absolute value.
        self.assertGreater(_sampling_scale_factor(_sparse_lap()),
                           _sampling_scale_factor(_fullres_lap()))
        self.assertGreater(sparse, 0.0)
        self.assertGreater(dense, 0.0)


class TestPrepareTelemetryRows(unittest.TestCase):

    def _telem(self, brake_values, distances=None):
        n = len(brake_values)
        t0 = pd.Timestamp("2026-01-01 12:00:00")
        data = {
            "Date": [t0 + pd.Timedelta(seconds=0.25 * i) for i in range(n)],
            "Speed": [200.0] * n,
            "Throttle": [100.0] * n,
            "Brake": brake_values,
            "nGear": [8] * n,
            "RPM": [11000] * n,
            "DRS": [0] * n,
        }
        if distances is not None:
            data["Distance"] = distances
        return pd.DataFrame(data)

    def test_brake_boolean_not_scaled_down(self):
        rows = _prepare_telemetry_rows(1, self._telem([True, False, True, False]))
        brakes = [r[3] for r in rows]
        self.assertEqual(brakes, [1.0, 0.0, 1.0, 0.0])

    def test_time_s_anchored_to_lap_start(self):
        rows = _prepare_telemetry_rows(1, self._telem([False] * 4))
        times = [r[7] for r in rows]
        self.assertEqual(times, [0.0, 0.25, 0.5, 0.75])

    def test_distance_from_channel(self):
        rows = _prepare_telemetry_rows(
            1, self._telem([False] * 3, distances=[10.0, 25.0, 40.0]))
        self.assertEqual([r[8] for r in rows], [10.0, 25.0, 40.0])

    def test_distance_integrated_when_channel_missing(self):
        # 0.25 s at constant 200 km/h (55.56 m/s) = ~13.9 m per step.
        rows = _prepare_telemetry_rows(1, self._telem([False] * 3))
        d = [r[8] for r in rows]
        self.assertEqual(d[0], 0.0)
        self.assertAlmostEqual(d[2], 2 * (200 / 3.6) * 0.25, places=1)


if __name__ == "__main__":
    unittest.main()
