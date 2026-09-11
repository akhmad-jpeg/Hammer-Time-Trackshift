# Trackshift — "Hammer Time"

A dual-car F1 strategy engine that turns race state into **overtake probabilities** and
**energy-management strategy** — a coupled attack/defence decision panel the pit wall can
actually interrogate. Every number the product and the pitch deck show is backed by a
committed, re-runnable artifact in this repo. This README is the map from claim to receipt.

---

## Every claim has a receipt

| Claim (as shown in the deck / product) | Backing artifact | Re-run it with |
|---|---|---|
| **"−1.8 s AI strategy vs Flat-Out"** — Monaco 2023, closing phase | `backtests/monaco_2023_energy.json` → closing-phase mean **−1.875 s**, `deck_claim_reproduced: true`. Re-verified on **full-resolution telemetry**: `backtests/monaco_2023_energy_fullres.json` → **−1.875 s, bit-identical** at ~687 samples/lap | `python scripts/benchmark_energy_strategy.py` |
| **The strategy generalises** — not one cherry-picked race | `backtests/energy_fleet_summary.json` → **140 races**, closing-phase mean **−4.49 s**, full-race mean −16.41 s | `python scripts/benchmark_energy_fleet.py` |
| **Deterministic backtests** — same data → same numbers, or the build fails | `scripts/check_energy_benchmark_drift.py` (tolerance 0.02 s) | `python scripts/check_energy_benchmark_drift.py --scope anchor` |
| **"We rank attack windows, we don't certify probabilities"** — confidence is worth 18 % vs the 10 % base rate of an arbitrary in-battle checkpoint (2.12× at P ≥ 0.95) | `scripts/backtests/race_call_reliability.json` + human-readable `race_call_reliability_slide.md` — 1,717 checkpoints, 527 battle segments, 84 weekends, 2020–2026 | `python scripts/race_call_reliability.py` |
| **Attack-window detection quality, published not hidden** — precision 38 % · recall 65 % (TP 240 / FP 394 / FN 132) | `window_detection` block of the same artifact | same as above |
| **Calibrated probabilities** — isotonic mapping fitted **out-of-time** (held-out 2024–2026): Brier 0.0317 → **0.0077**, ECE 0.123 → **0.000** | `ml_models/overtake/model_info.json` (`isotonic_calibration`), served live at `/api/overtake/reliability` and rendered in the dashboard's Model Calibration & Reliability panel | `python -m unittest scripts.tests.test_isotonic_calibration -v` (from `scripts/`) |
| **"Linear Regression over black-box networks"** — interpretable by choice | overtake model = `LinearRegression`, 43 features, 5,396 training samples (`model_info.json`); lap-time model benchmarked against alternatives | `python scripts/benchmark_models.py` |
| **"Refuses any move the battery can't pay for"** — hard feasibility pruning against a **self-imposed 10 % management reserve** (no rulebook value encoded) | `scripts/policy_engine.py` feasibility layer; out-of-time-safe directions pinned in tests | `python -m unittest scripts.tests.test_policy_engine -v` |
| **"Synthesizes battery state because F1 does not broadcast it"** — with honest uncertainty, never fake precision: ±2 % floor, +0.5 %/lap drift, ±8 % cap; engine prices the **worst case** and degrades confidence as the band widens | `energy_simulator.battery_uncertainty_band` (single source of truth), band shown per lap in the UI | `python -m unittest scripts.tests.test_soc_uncertainty -v` |
| **"Simulates two cars simultaneously"** — coupled attack + defence engines, leader posture (deploy/bank/defend) funded from the leader's own store | `scripts/overtake_inference.py` (`simulate_live_call`), `scripts/policy_engine.py` | `python -m unittest scripts.tests.test_leader_defense scripts.tests.test_leader_perspective -v` |
| **Full-resolution telemetry** — the importer keeps every ~0.25 s sample with intra-lap time and distance; regen scaling becomes density-aware (the legacy ×2.2 sparse fudge disengages at ≥2 Hz) | `scripts/import_f1_race.py`, `scripts/migrate_telemetry_fullres.py`, density tests; Monaco 2023 re-import: **200,881 samples, ~644/lap, 0 failures** — headline unchanged | `python -m unittest scripts.tests.test_fullres_regen -v` |
| **No marketing fiction** — four families of unverifiable phrasing (lap-count accuracy tolerances, single-digit-millisecond latency, rulebook battery attribution, per-corner spatial claims) are **banned from the repo by a guard that exits non-zero** | `scripts/verify_claims.py` | `python scripts/verify_claims.py` |

The pitch-deck copy itself is audited against this repo in **`DECK_FIXES.md`** — every
unverifiable phrase has a replacement the artifacts can actually back.

---

## What we deliberately do *not* claim

- **No lap-count accuracy tolerance.** The models quote MAE / R² / AUC / Brier from
  committed backtests; a "±N laps" figure was never measured, so it is never written.
- **No sub-10 ms latency.** Measured budgets live in code and payloads: single-pair live
  calls are ~200 ms class, policy rollouts seconds, the energy sandbox warm in tens of ms.
  The dashboard discloses which one you are seeing.
- **No guaranteed battery outcomes.** The battery is a reconstruction with a published
  uncertainty band; the product's promise is *refusal*, not certainty — it refuses any
  move the store cannot pay for.
- **Small-label honesty.** Overtakes are rare: the model's precision is low (0.19) while
  recall is 0.65, and the reliability table *is* the calibration statement. Nothing is
  hidden to make a slide look better.

---

## Quick start

Requirements: Python 3.13, MySQL 8, and `pip install -r requirements.txt`
(pinned: fastf1 3.8.3, Flask 3.1.3, scikit-learn 1.9.0, …).

```bash
# 1. Configure — copy the template, fill in your MySQL credentials
cp .env.example .env

# 2. Create the schema (fresh install already carries time_s / distance_m)
mysqlsh --sql -u root -p --file=database/f1_strategy.sql     # or: mysql < database/f1_strategy.sql

# 3. Import history (FastF1, 2020–2026) and train the models
launchers/02_import_race.bat        # or: python scripts/import_f1_race.py --year 2023 --race Monaco --session R --driver HAM
launchers/05_train_model.bat        # trains lap + overtake models, fits the isotonic calibrator

# 4. Run the dashboard
launchers/06_run_server.bat         # or: python scripts/run_server.py  (PORT/HOST via .env)
```

`ml_models/` and `f1_cache/` are machine-local (gitignored) and fully regenerable from
steps 3–4. `database/f1_strategy.sql` creates the schema plus a small seed set.

Existing databases gain the full-resolution columns idempotently:

```bash
python scripts/migrate_telemetry_fullres.py            # add time_s / distance_m (no-op if present)
python scripts/migrate_telemetry_fullres.py --purge-telemetry   # optional: wipe + fresh re-import
```

---

## Running the test suite

128 tests across 12 suites (policy engine, leader defence/perspective, two-phase pricing,
SoC uncertainty, isotonic calibration, full-resolution regen, race-call reliability, …).
Run from the `scripts/` directory:

```bash
cd scripts
python -m unittest discover -s tests -p "test_*.py"
```

The decision-engine tests are deliberately pinned to the **deterministic pace/energy
layer** (per-lap pace gaps, gap-path dominance, feasibility gating) rather than to
classifier-scale artifacts, so they stay green across model retrains.

---

## Repository map

```
scripts/                    application code
  policy_engine.py          multi-policy decision engine (two-phase pricing, feasibility, confidence)
  overtake_inference.py     coupled two-car live call (attack + leader defence postures)
  energy_simulator.py       energy projection, density-aware regen, SoC uncertainty bands
  ml_overtake_predictions.py overtake model training + isotonic calibration
  import_f1_race.py         FastF1 importer (full-resolution telemetry)
  capture_telemetry.py      live game-telemetry UDP client (in development)
  race_call_reliability.py  replay every stored race -> reliability table
  benchmark_energy_*.py     Monaco anchor + 140-race fleet energy backtests
  verify_claims.py          claim guard (banned unverifiable phrasing)
  dashboard.py + dashboard/ Flask app: race call, what-if sandbox, calibration panel
  tests/                    12 test suites, 128 tests
backtests/                  committed artifacts: Monaco anchor (legacy + full-res), fleet summary
scripts/backtests/          race-call reliability artifact + slide
database/f1_strategy.sql    schema + seed (includes time_s / distance_m)
launchers/                  numbered Windows launchers (setup -> import -> train -> serve -> test)
DECK_FIXES.md               deck copy audit: every claim checked against this repo
```

## Data provenance

Seven seasons of real F1 history (2020–2026) imported per driver from FastF1 — every
leader/chaser duel replays both cars' actual laps. 571 sessions / 31k+ laps in the demo
database; the Monaco anchor has additionally been re-imported and re-verified at full
telemetry resolution. The live UDP capture client is built and tested but has not yet
recorded a session — it is labelled "in development" everywhere it appears.
