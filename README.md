# Trackshift — "Hammer Time"

A real-time F1 race strategy engine that turns race state into **ranked tactical
decisions**, **overtake probabilities**, and **energy-management strategy** — a coupled
attack/defence decision panel with a dynamic pit & Safety Car model and an AI Race
Engineer transceiver. Sub-200 ms from telemetry input to radio directive.

Every number the product and the pitch deck show is backed by a committed, re-runnable
artifact in this repo. This README is the map from claim to receipt.

---

## Architecture

```
Raw telemetry
  → pit_strategy     (circuit-specific pit loss, VSC/SC savings, undercut feasibility)
  → policy_engine    (5-policy coupled scoring, game-theoretic leader/chaser)
  → ai_race_engineer (deterministic radio call synthesis)
  → dashboard API    (Flask/Waitress, 37 endpoints, <200 ms)
  → Live HUD         (ranked policy table, Safety Car & Pit Radar, AI radio panel)
```

**Core decision engine**: 5 competing strategies — Greedy Attack, Balanced Hold,
Tactical Stalk, Save & Defend, Undercut Prep — evaluated simultaneously for both
the leader (defence) and chaser (attack). The leader's recommended defence feeds the
chaser's attack scoring: 15 trajectories (5 × 5), coupled as a game, not in isolation.

---

## Every claim has a receipt

| Claim (as shown in the deck / product) | Backing artifact | Re-run it with |
|---|---|---|
| **"−1.8 s AI strategy vs Flat-Out"** — Monaco 2023, closing phase | `backtests/monaco_2023_energy.json` → closing-phase mean **−1.875 s**, `deck_claim_reproduced: true`. Re-verified on **full-resolution telemetry**: `backtests/monaco_2023_energy_fullres.json` → **−1.875 s, bit-identical** at ~687 samples/lap | `python scripts/benchmark_energy_strategy.py` |
| **The strategy generalises** — not one cherry-picked race | `backtests/energy_fleet_summary.json` → **140 races**, closing-phase mean **−4.49 s**, full-race mean −16.41 s | `python scripts/benchmark_energy_fleet.py` |
| **Deterministic backtests** — same data → same numbers, or the build fails | `scripts/check_energy_benchmark_drift.py` (tolerance 0.02 s) | `python scripts/check_energy_benchmark_drift.py --scope anchor` |
| **"We rank attack windows, we don't certify probabilities"** — confidence is worth 18 % vs the 10 % base rate (2.12× at P ≥ 0.95) | `scripts/backtests/race_call_reliability.json` — 1,717 checkpoints, 527 battle segments, 84 weekends, 2020–2026 | `python scripts/race_call_reliability.py` |
| **Calibrated probabilities** — isotonic mapping fitted **out-of-time** (held-out 2024–2026): Brier 0.0317 → **0.0077**, ECE 0.123 → **0.000** | `ml_models/overtake/model_info.json` (`isotonic_calibration`), served live at `/api/overtake/reliability` | `python -m unittest scripts.tests.test_isotonic_calibration -v` |
| **Overtake model: RandomForestClassifier** — 200 trees, max_depth=8, AUC 0.967, accuracy 0.970; closing speed regressor: RandomForestRegressor, 200 trees; lap-time model: LinearRegression, 83 features | `ml_models/overtake/model_info.json`, `ml_models/model_info.json` | `python scripts/benchmark_models.py` |
| **"Refuses any move the battery can't pay for"** — hard feasibility pruning against a **self-imposed 10 % management reserve** (no rulebook value encoded) | `scripts/policy_engine.py` feasibility layer | `python -m unittest scripts.tests.test_policy_engine -v` |
| **"Synthesizes battery state because F1 does not broadcast it"** — with honest uncertainty: ±2 % floor, +0.5 %/lap drift, ±8 % cap | `energy_simulator.battery_uncertainty_band` (single source of truth), band shown per lap in the UI | `python -m unittest scripts.tests.test_soc_uncertainty -v` |
| **"Simulates two cars simultaneously"** — coupled attack + defence engines, leader posture funded from the leader's own store | `scripts/policy_engine.py`, `scripts/overtake_inference.py` | `python -m unittest scripts.tests.test_leader_defense scripts.tests.test_leader_perspective -v` |
| **32-circuit pit catalogue** — circuit-specific pit loss (green/VSC/SC), traffic multipliers (Clear ×0.95, Light ×1.0, Heavy ×1.25), historical SC probability tiers | `scripts/pit_strategy.py` (`CIRCUIT_PIT_PROFILES`) | `python -m unittest scripts.tests.test_pit_strategy -v` |
| **No marketing fiction** — four families of unverifiable phrasing are **banned from the repo by a guard that exits non-zero** | `scripts/verify_claims.py` | `python scripts/verify_claims.py` |

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
- **Small-label honesty.** Overtakes are rare: the model's precision is 0.19 while
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

**169 unit tests** across **15 suites** + **165 live API tests** (all 37 dashboard endpoints).

```bash
# Unit tests (from scripts/ directory)
cd scripts
python -m unittest discover -s tests -p "test_*.py"

# Live API tests (requires server running on port 5000)
python -X utf8 scripts/tests/test_dashboard_api.py
```

| Test suite | Coverage |
|---|---|
| `test_policy_engine.py` | 5 policies, scoring, feasibility, coupling |
| `test_pit_strategy.py` | 32 circuits, traffic factors, undercut, SC/VSC/red flag |
| `test_ai_race_engineer.py` | Radio call synthesis, directive templates |
| `test_unified_call.py` | Coupled leader+chaser call end-to-end |
| `test_leader_defense.py` | Leader defence perspective policies |
| `test_leader_perspective.py` | Leader seat switching |
| `test_soc_uncertainty.py` | ±2 % floor, 0.5 %/lap growth, 8 % cap |
| `test_isotonic_calibration.py` | Probability calibrator monotonicity |
| `test_fullres_regen.py` | ERS regen from full-resolution speed traces |
| `test_ers_shape_sliders.py` | Energy sandbox per-sector redistribution |
| `test_two_phase_pricing.py` | Two-phase scoring function |
| `test_race_call_reliability.py` | Edge cases in race calls |
| `test_p0_fixes.py` | Regression fixes |
| `test_verify_claims.py` | Claims scanner self-validation |
| `test_dashboard_api.py` | 165 live HTTP tests across 37 endpoints |

The decision-engine tests are deliberately pinned to the **deterministic pace/energy
layer** (per-lap pace gaps, gap-path dominance, feasibility gating) rather than to
classifier-scale artifacts, so they stay green across model retrains.

---

## Validation layers

| Layer | Script | What it checks |
|---|---|---|
| Source data physics | `validate_features.py` | Lap times in plausible ranges, tyre age monotonic, fuel decreasing |
| Database schema contract | `check_db_schema.py` | 11 required tables + columns present, 5 deprecated tables absent |
| Benchmark reproducibility | `check_energy_benchmark_drift.py` | Re-derives Monaco anchor & fleet, fails if any value drifts > 0.02 s |
| Claims compliance | `verify_claims.py` | Zero banned marketing phrases across all source/doc files |
| Unit tests | 15 test suites | 169 tests covering every module |
| Live API tests | `test_dashboard_api.py` | 165 HTTP tests, every endpoint, correct + error inputs |

---

## ML models

| Model | Algorithm | Features | Key metrics |
|---|---|---|---|
| Lap time predictor | `LinearRegression` | 83 (tyre age, compound, track, weather) | Committed MAE / R² in `model_info.json` |
| Per-sector time (×3) | `LinearRegression` × 3 | Same | Energy sandbox sector resolution |
| Overtake classifier | `RandomForestClassifier` (200 trees, depth 8) | 43 | AUC 0.967, accuracy 0.970 |
| Closing speed regressor | `RandomForestRegressor` (200 trees, depth 10) | 43 | MAE 0.475 s, RMSE 0.944 s |
| Probability calibrator | `IsotonicRegression` | — | Brier 0.0077, ECE 0.000 (out-of-time) |
| Energy pace profile | Measured lookup table | 32 circuits | Per-track s/MJ + sector splits |

---

## Repository map

```
scripts/                         application code (38 modules)
  policy_engine.py               multi-policy decision engine (5 policies, coupled scoring, feasibility)
  overtake_inference.py          coupled two-car live call (attack + leader defence postures)
  energy_simulator.py            ERS battery simulation, density-aware regen, SoC uncertainty bands
  pit_strategy.py                32-circuit pit loss catalogue, VSC/SC cheap stops, undercut evaluation
  ai_race_engineer.py            deterministic radio call synthesis (AI Race Engineer transceiver)
  tyre_degradation.py            Pirelli-style health model, per-track abrasion, cliff detection
  race_calendar.py               official F1 calendars 2020–2026 (single source of truth)
  stint_analysis.py              stint detrending (fuel-adjusted degradation curves)
  ml_overtake_predictions.py     overtake model training + isotonic calibration
  ml_lap_predictions.py          lap-time model training + feature engineering
  import_f1_race.py              FastF1 importer (full-resolution telemetry, ~200–300 samples/lap)
  capture_telemetry.py           live game-telemetry UDP client (in development)
  driver_comparison.py           driver-vs-driver head-to-head analysis
  fuel_estimation.py             fuel load estimation from lap times
  feature_pipeline.py            ML feature construction + validation
  race_call_reliability.py       replay every stored race → reliability table
  benchmark_energy_strategy.py   Monaco anchor energy backtest
  benchmark_energy_fleet.py      140-race fleet energy backtest
  check_energy_benchmark_drift.py  scheduled drift check (daily, tolerance 0.02 s)
  check_db_schema.py             database schema contract verification
  validate_features.py           end-to-end feature validation harness
  verify_claims.py               claim guard (banned unverifiable phrasing)
  dashboard.py + dashboard/      Flask app: 37 API endpoints, race call, what-if sandbox, calibration panel
  tests/                         15 test suites, 169 unit tests + 165 live API tests
backtests/                       committed artifacts: Monaco anchor (legacy + full-res), fleet summary
scripts/backtests/               race-call reliability artifact + slide
ml_models/                       trained models + energy pace profiles (gitignored, regenerable)
  overtake/                      RF classifier, RF regressor, isotonic calibrator, training pairs
database/f1_strategy.sql         schema + seed (includes time_s / distance_m)
launchers/                       13 numbered Windows launchers (setup → import → train → serve → test)
outputs/                         generated analysis outputs (overtake analysis, simulation data)
DECK_FIXES.md                    deck copy audit: every claim checked against this repo
FORMULAS.txt                     physics and scoring formulas reference
```

---

## Dashboard — 37 API endpoints

**Strategy & Decision**
- `POST /api/strategy/policies` — 5-policy ranked scorecard (chaser or leader perspective)
- `POST /api/strategy/call` — coupled leader+chaser final call
- `POST /api/strategy/analyze` — tyre strategy advisor (pit timing, compound selection)
- `POST /api/strategy/energy-analyze` — energy-aware strategy extension
- `POST /api/strategy/energy-sandbox` — per-sector ERS reallocation simulator
- `POST /api/ai-race-engineer` — AI Race Engineer radio call synthesis

**Overtake & Battle**
- `POST /api/overtake/sim` — overtake probability simulation
- `POST /api/overtake/live` — live battle analysis
- `GET /api/overtake/options` — available drivers/tracks for overtake scenarios
- `GET /api/overtake/sessions` — session lookup for battle pairs
- `GET /api/overtake/calibration` — model calibration data
- `GET /api/overtake/reliability` — reliability table (precision, recall, lift)
- `GET /api/overtake_analysis` — pre-computed overtake analysis
- `GET /api/overtake_simulation_data` — animated track map data

**Telemetry & Session**
- `GET /api/sessions` — session list (pagination, driver/year/track filters)
- `GET /api/session/<id>/laps` — lap data for a session
- `GET /api/session/<id>/energy` — synthetic ERS battery trace
- `POST /api/session/<id>/energy-simulate` — run energy simulator for a session
- `GET /api/session/<id>/tyre-degradation` — tyre health + fuel-adjusted degradation
- `GET /api/latest-lap` — most recent live-captured lap
- `GET /api/live/battle-state` — reconstructed live gap between two drivers

**Reference & Comparison**
- `GET /api/drivers`, `GET /api/drivers/list`, `GET /api/drivers/compare`
- `GET /api/predict/options`, `POST /api/predict`
- `GET /api/comparison/years`, `/tracks`, `/drivers`, `/race`
- `GET /api/calendar`
- `GET /api/benchmark/energy`, `/fleet`, `/race`, `POST .../run`

---

## Data provenance

Seven seasons of real F1 history (2020–2026) imported per driver from FastF1 — every
leader/chaser duel replays both cars' actual laps. The Monaco anchor has additionally
been re-imported and re-verified at full telemetry resolution (~687 samples/lap,
200,881 total samples, headline unchanged). The live UDP capture client is built
and tested but has not yet recorded a session — it is labelled "in development"
everywhere it appears.

---

## Physics engine

The ERS battery simulator (`energy_simulator.py`) synthesises state-of-charge from
regulation-grounded constants:

- **4.0 MJ** usable Energy Store capacity (FIA 2026 Technical Regulations)
- **8.5 MJ/lap** harvest limit (C5.2.10); **350 kW** MGU-K (2026); **120 kW** (legacy)
- **Regen from kinetic energy**: `0.5 × m × (v₁² − v₂²) × η` per braking event,
  density-aware sampling correction (full-res ×1.0, sparse legacy ×2.2)
- **Three driver modes**: Push (greedy deploy), Balanced (pace-coupled SOC hold at 55%),
  Lift & Coast (bank to 90%)
- **Pace coupling**: faster-than-average laps spend stored energy, slower laps bank it
- **Management reserve**: 10% floor — the battery never projects to empty

The tyre model (`tyre_degradation.py`) provides a Pirelli-style health curve
(logistic degradation) scaled by per-track abrasion factors and compound-specific
cliff points.
