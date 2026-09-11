# PROJECT_GUIDE.md — Trackshift "Hammer Time": The Complete Reference

Everything about this project: what it is, how it was built, every formula and
constant, why each decision was made, what advantages it buys, and where every
number lives. Written from the code itself (September 2026). Companion files:
`README.md` (claim → artifact map for judges) and `DECK_FIXES.md` (deck copy audit).

---

## 1. WHAT THIS PROJECT IS

**Trackshift — "Hammer Time"** (Team Underdogs) is a dual-car F1 strategy engine:
a pit-wall co-pilot that converts race state into *overtake probabilities* and
*energy-management strategy*, as a coupled attack/defence decision panel.

One-sentence pitch: **"It refuses any overtake the battery can't pay for."**

- **Origin:** built on top of a Motorsports Telemetry Platform (FastF1 history
  ingestion + telemetry DB + Flask dashboard), extended with a strategy brain.
- **Data:** 7 seasons of real F1 history (2020–2026) via FastF1 — 571 sessions,
  31,384 laps, 181,015 telemetry samples in the demo database.
- **Status numbers:** 128 tests / 12 suites green · deterministic backtests with
  a drift checker · claim guard enforcing honest copy · reliability table from
  1,717 replayed checkpoints · Monaco 2023 anchor reproduced on both legacy and
  full-resolution telemetry.
- **Stack:** Python 3.13, Flask 3.1.3, MySQL 8, FastF1 3.8.3, scikit-learn 1.9.0,
  pandas, waitress (prod WSGI), vanilla JS + Plotly front end. All pinned in
  `requirements.txt`.

### The four-layer architecture

```
DATA        import_f1_race.py (FastF1, full-res) · capture_telemetry.py (UDP, dev)
  │         MySQL: sessions → laps → telemetry, race_state, strategy_events
  ▼
PHYSICS     energy_simulator.py (battery synthesis) · tyre_degradation.py
  │         fuel_estimation.py · measure_tyre_wear.py (per-track calibration)
  ▼
ML          ml_lap_predictions.py (lap times) · ml_overtake_predictions.py
  │         (overtakes + isotonic calibration) · feature_pipeline.py
  ▼
DECISION    overtake_inference.py (live 2-car call) · policy_engine.py
            (multi-policy scoring) · dashboard (what-if sandbox)
```

---

## 2. DESIGN PHILOSOPHY — THE FIVE PRINCIPLES

These five principles explain almost every implementation choice:

1. **Deterministic or disclosed.** Same inputs must produce the same outputs —
   backtests have a drift checker that fails the build at >0.02 s delta; the
   tyre model is deliberately a closed-form curve, not a black box.
2. **Synthesize honestly.** F1 broadcasts no battery SOC, so the system
   synthesizes it — and attaches a published uncertainty band (±2%→±8%) instead
   of fake point precision.
3. **Refusal is the product.** Hard feasibility: a policy that breaches the
   10% management reserve or drains the store to its floor is *never*
   recommended — pruned, not merely scored worse.
4. **Pirelli-realistic physics.** Measured data where available (156 stints),
   published team knowledge as fallback (loss rates, cliff at 40% health,
   pit window 40–50%), everything clamped so displays can't go impossible.
5. **Claims must be backed.** A repo-wide guard (`verify_claims.py`) exits
   non-zero on four families of marketing phrasing. The README maps every
   claim to a re-runnable artifact.

---

## 3. THE DATA LAYER

### 3.1 Database schema (`database/f1_strategy.sql`)

MySQL 8, `utf8mb4`. Parent tables: `drivers`, `seasons`, `tracks`
(`canonical_name`, `short_code`), `regulations`, `data_sources` (kept across
re-imports so IDs stay stable). Child tables (deleted first on purge):
`sessions` (driver_id, track_name, date, session_type R/Q/FP1-3) → `laps`
(lap_number, lap_time_ms, validity, stint, compound…) → `telemetry`
(lap_id, speed, throttle, brake, gear, rpm, drs, **time_s**, **distance_m**)
plus `race_state` (energy_start_mj / energy_deployed_mj / energy_harvested_mj /
energy_end_mj per lap — the synthesized battery scenario rows) and
`strategy_events` (pit stops, SC/VSC/red-flag). Seeded with a small 2026-era
set; the full 571-session demo DB was imported per driver via FastF1.

### 3.2 Historical ingestion — `import_f1_race.py`

Per-driver FastF1 import (year/race/session/driver CLI). Key behaviors:

- **Full-resolution telemetry** (`TELEMETRY_FULL_RESOLUTION = True`, cap 2,000
  rows/lap): every ~0.25 s sample stored with `time_s` (intra-lap clock,
  anchored to the lap's first sample) and `distance_m` (FastF1's Distance
  channel, else integrated from speed). Result on Monaco 2023: **~644
  samples/lap (687 densest), 200,881 samples for 4 drivers, 0 failures** —
  versus ~6 samples/lap in the legacy pipeline.
- **Brake fix:** FastF1's Brake channel is BOOLEAN — the old pipeline divided
  it by 100, turning True into 0.01. Now stored 0.0/1.0.
- **Race-control parsing:** red-flag/SC/VSC lap classification from message
  patterns; pit-stop detection with min-duration 15 s; tyre-compound mapping.
- **Dedupe by default:** re-importing an existing session is skipped unless
  `--allow-duplicate`.

### 3.3 Live ingestion — `capture_telemetry.py`

UDP game-telemetry client (port **20777**, 60 Hz sample rate), threaded
parse→DB-queue→writer, lap validity windows (60–180 s), SC detection at
<120 km/h, VSC at 1.15–1.30× race-pace ratio, red flag at <60 km/h, heartbeat
every 5 s. **Status: built and tested, zero recorded sessions — labelled
"in development" everywhere it appears** (a deliberate honesty point, see
`DECK_FIXES.md` Fix 4).

### 3.4 Migration — `migrate_telemetry_fullres.py`

Idempotent schema migration adding `time_s`/`distance_m` (checked via
`information_schema` before each ALTER). Optional `--purge-telemetry` deletes
child-first (telemetry → strategy_events → race_state → laps → sessions) for a
clean re-import, keeping reference tables. Verified end-to-end in a sandbox DB
(`f1_strategy_fullres`, created via the documented `DB_NAME` env override):
full-res re-import of Monaco 2023 reproduced the deck's energy headline
bit-identically (see §6.3).

### 3.5 Related tooling

`check_db_schema.py` (schema smoke test with required/forbidden tables),
`db_transfer.py` (full backup/restore with table binning), `backfill_sector_times.py`
(FastF1 sector backfill), `race_calendar.py` (official FIA calendars 2020–2026
with DB-coverage annotation), `cleanup_pit_events.py` (purges pit events
<15 s, re-validates laps).

---

## 4. THE ML LAYER

### 4.1 Lap-time model — `ml_lap_predictions.py` / `benchmark_models.py`

- **Model: LinearRegression** (`ml_models/best_model.pkl`), **83 features**.
- **Features:** continuous `tyre_age` + `fuel_load` + weather/circuit context;
  one-hot compounds (Medium/Soft/Intermediate; Hard = base) and track. Fuel
  enters as the shared synthetic proxy (§5.2) so training and inference agree.
- **Measured performance (within-track): MAE 1.61 s · RMSE 2.89 s · R² 0.932**
  (`ml_models/model_info.json`).
- **Why LinearRegression over Random Forest:** smooth, monotonic tyre-wear
  response — no "staircase" artifacts when the simulator extrapolates long
  stints; interpretable coefficients (`tyre_age_coefficient`,
  `tyre_wear_slopes_per_compound` stored in model_info.json). `benchmark_models.py`
  runs the selection comparison; `driver_comparison.py` fits per-driver models
  for head-to-head pace deltas.

### 4.2 Overtake model — `ml_overtake_predictions.py` (P0)

The core ML asset. **Construction:** for every stored race, leader/chaser
battle pairs are built lap-by-lap (same track/date, adjacent positions):
- **Pair features:** gap_before_s, per-lap pace gap (leader − chaser lap time),
  tyre ages and compounds, fuel diff, **energy_diff_mj** (the synthesized
  battery difference, from `race_state` — this is where the energy layer enters
  ML), lap number, track one-hots.
- **Labels:** a pass = the chaser completes the overtake within
  `OVERTAKE_GAP_MAX_S = 5.0 s` of the battle window (`GAP_WINDOW_S = 10.0`);
  laps under SC/VSC/red-flag are excluded (lap-time ratio > 1.30);
  `STINT_WARMUP_LAPS = 2` excluded; `MIN_PAIR_LAPS = 40` per race.
- **Split: TEMPORAL, cutoff 2024** — trained on 2020–2023 (5,396 samples,
  73 positives), tested on 2024–2026 (6,403 samples, 60 positives). No future
  leakage; the reliability story describes *forward* prediction.
- **Metrics (out-of-time): AUC 0.967, accuracy 0.970, precision 0.188,
  recall 0.65.** Precision is honestly low because overtakes are rare
  (small-label reality) — the reliability table *is* the calibration statement.

### 4.3 Isotonic calibration (P1-4)

A post-hoc isotonic regressor maps raw scores → calibrated probabilities,
**fitted only on the held-out later seasons (2024–2026)**. Two-scale contract:
`overtake_probability` stays the raw scale (every threshold — e.g. the 0.80
pass trigger — is tuned on it); the isotonic output lives in
`calibrated_probability`. Results: **Brier 0.0317 → 0.0077 · ECE 0.123 →
0.000**. Served at `/api/overtake/reliability` and rendered in the dashboard's
Model Calibration & Reliability panel. Tests: `test_isotonic_calibration.py`.

### 4.4 Feature pipeline — `feature_pipeline.py`

Shared feature engineering: lap-phase buckets (edges at laps 13/26/42 →
early/mid1/mid2/late), regulation-era buckets (edges 2021/2022/2024 →
18_20/21/22_23/24_26), compound/track encoding, vector alignment for model
ingestion. **Deliberately consumes NO energy feature** for lap times — energy
enters ML only via the overtake model's `energy_diff_mj` (the deck once claimed
otherwise; see `DECK_FIXES.md` bonus 2).

---

## 5. THE PHYSICS LAYER

### 5.1 Power-unit specs — `energy_simulator.PU_SPECS`

Era-aware simulation — a session's year selects its spec, never an
anachronistic mix:

| Spec | MGU-K | Usable ES | Harvest limit | Deploy ceiling | Car mass |
|---|---|---|---|---|---|
| `legacy_2014_2025` | 120 kW (+MGU-H) | 4 MJ | 2 MJ/lap | 4 MJ/lap | 795 kg |
| `newgen_2026` (default) | 350 kW, no MGU-H | 4 MJ | 8.5 MJ/lap (TR C5.2.10) | 8.5 MJ/lap | 768 kg |

Sources: FIA 2026 Technical Regulations via authoritative race reporting
(docstring cites the regulation clauses). 2026 nuances modeled: manual boost
capped at +150 kW over current, 250 kW limit outside acceleration/overtaking
zones, super-clipping (on-throttle recharge).

### 5.2 Synthetic fuel — `fuel_estimation.py`

```
fuel_load(kg) = max(0, 110.0 − 2.0 × lap_number)
```
Deliberately  synthetic and disclosed as such (no real fuel sensors in either
  feed) — but *consistent* between training and inference, which is what the
  model needs. Measured fleet fuel effect (`tyre_age_coefficient` in
  `ml_models/model_info.json` = `measure_tyre_wear.FUEL_BURN_RATE_S_PER_LAP` =
  **−0.0703 s/lap per kg burned off**): each kg carried on board costs ≈
  +0.0703 s/lap — the advisor adds this back flat to every stint before
  isolating tyre wear.

### 5.3 Tyre model — `tyre_degradation.py`

Deterministic, monotonic, closed-form — chosen over a fitted model so the
chart is explainable ("Softs at 34% on lap 18") and the physics-guardrail pitch
holds:

```
raw_health        = 100 − loss_per_lap(compound, track) × age
tyre_health       = max(10, raw_health)                      # HEALTH_FLOOR, never shows 0%
tyre_cliff_penalty= 0 if raw ≥ 40 else 2.5 × (40 − raw)/40   # s/lap, CLIFF_HEALTH=40, MAX=2.5
usable_life       = 100 / loss_per_lap                        # laps to 0%
TYRE_LIFE         = {c: round(100/rate)}                      # derived per compound
```

- **Base loss rates (%/lap, Pirelli-published usable-life bands, tuned to
  reproduce reported health after 20 race laps):** Soft 4.5 (cliff ~lap 15),
  Medium 2.25 (~55% at lap 20), Hard 1.35 (~73%), Intermediate 2.0, Wet 1.8.
  Fallback for unknown compounds: 2.25 (Medium-like).
- **Per-track calibration (two-tier):** measured cells come from
  `measure_tyre_wear.py` — median fuel-corrected within-stint wear slope per
  (track × compound) relative to fleet median, **shrunk toward 1.0 with
  SHRINK_K = 6.0 for thin cells, clamped [0.5, 2.0]**, min stint 5 laps,
  outliers ±2.5 s/lap dropped; stored in `ml_models/tyre_wear_per_track.json`.
  Unmeasured cells fall back to the researched TRACK_ABRASION map (Monaco 1.15,
  Marina Bay 1.30, Bahrain 1.25, Silverstone 0.90, Red Bull Ring 0.90, …).
- **Published-max clamp:** no cell may exceed the top of Pirelli's published
  band (Soft 5.0 / Medium 2.8 / Hard 1.6 %/lap) — a Bahrain Soft can never
  wear twice as fast as real rubber.
- **Strategy logic embedded:** teams pit at 40–50% health (1–2 laps before the
  cliff); the cliff itself is the 0→2.5 s/lap penalty below 40% health, which
  is *why* pit windows sit where they do. Advisor pace deltas:
  TYRE_PACE_DELTA Soft 0.0 / Medium +0.30 / Hard +0.60 s/lap.

### 5.4 Energy simulator — `energy_simulator.py` (the battery synthesizer)

F1 does not broadcast SOC; this module synthesizes it, shaped to the FIA
regulations. Full behavior (all verified in tests):

- **Regen from the real speed trace:** kinetic energy lost between consecutive
  samples × efficiency, capped by the spec's per-lap harvest limit and the
  store's headroom (a full battery wastes surplus to friction brakes).
- **Density-aware sampling compensation** (`_sampling_scale_factor`):
  timestamped traces at median dt ≤ 0.5 s → scale **1.0**; timestamp-less
  traces use row-count (≥80 rows → 1.0, ≤12 rows → **×2.2**
  REGEN_SAMPLING_SCALE, linear ramp between); single-sample laps → 2.2. The
  ×2.2 exists because legacy ~6-sample laps under-detected braking (raw 1–3.6
  MJ/lap where a real Spa lap recovers 5–7 MJ). Fallback for undetectable
  drops: REGEN_FALLBACK_MJ = 5.0. **Verified on real data:** legacy laps scale
  2.2; full-res Monaco laps scale 1.0.
- **Three deployment modes** (`MODES`):
  - `push` — regen_style **0.85** (late braking, short regen window), deploys
    the spec ceiling every lap → drains to the low band in 1–2 laps, then
    lives energy-limited (the deck's "paradox": fast now, limited later);
  - `balanced` — regen_style 1.00, spends ≈ what the lap harvests, SOC steered
    toward **55%** (soft band 30–80%: SOC_WINDOW_MIN/MAX);
  - `liftcoast` — regen_style **1.20** (early lift-off opens a longer regen
    window), banks SOC toward **90%** as insurance.
- **Store dynamics:** battery starts FULL (OPENING_BURN_DOWN_FRACTION 0.35
  shapes the opening-lap drain into the working band). Below the floor the car
  lives off recovery, re-deploying only **FLOOR_REDEPLOY_FRACTION = 0.85** of
  the lap's flow (part of each lap must re-charge — super-clipping/lift-off),
  so the battery yo-yos in a low band instead of dying at 0%. SOC↔pace
  coupling: PACE_NET_MJ_PER_DEV = 45 MJ per pace deviation unit, NET_MOVE_CAP_MJ
  = 1.0/lap, SOC_BAND_LEAK_FRACTION = 0.30, intra-lap wiggle compression 0.45.
  Management reserve **RESERVE_FRACTION = 0.10** → BATTERY_MIN_MJ = 0.4 MJ —
  the battery is never projected to literal zero (team policy, not an FIA
  figure; the claim guard bans rulebook attribution).
- **Pace pricing:** DEPLOY_PACE_S_PER_MJ = **0.35 s/MJ** crude energy-pace
  equivalence (ERS_ANCHOR_S_PER_MJ, fleet full-throttle share 0.6703);
  energy-limited laps deliver only LIMITED_LAP_PACE_EFFECTIVENESS = **0.70**.
  Per-track measured pace profiles live in `ml_models/energy_pace.json`
  (s/MJ, measured s/lap per MJ deployed — e.g. Monaco 0.1543 from 464 laps);
  alias-resolved so 'Monaco' and 'Circuit de Monaco' share one entry.
- **SoC uncertainty band** (single source of truth,
  `battery_uncertainty_band`):
  ```
  band_pct = min(8.0, 2.0 + 0.5 × laps_since_anchor)      # floor 2, +0.5/lap, cap 8
  band_mj  = band_pct/100 × 4.0
  ```
  Attached per lap in every projection, reported as mean ± band with explicit
  ranges; the policy engine prices battery at the **worst case** (mean − band)
  and degrades recommendation confidence as the band widens (multiplier
  ≥ 0.6 floor). Tests: `test_soc_uncertainty.py`.

---

## 6. THE DECISION LAYER

### 6.1 Multi-policy engine — `policy_engine.py`

`evaluate_tactical_policies` scores every policy on one shared forward
projection (walker) of the same race state; output: scored policy table,
recommendation + action card, uncertainty bounds, scoring constants echoed
(`scoring_constants` — full auditability). **Five chaser policies:**

| Policy | Shape |
|---|---|
| BALANCED HOLD | zero-deploy baseline |
| SAVE & DEFEND | bank energy, single-phase |
| GREEDY ATTACK | deploy from the first lap, single-phase |
| TACTICAL STALK | two-phase: bank (phase 1) → full-lever strike (phase 2) |
| LATE SURGE | defer the attack to the closing laps |

**The scoring formula** (all components in race-time seconds, lower better):

```
score = ( batt_cost + wear_cost + cliff_cost + risk_cost + latency_cost )
        − (0 if infeasible else pass_gain)

pass_gain    = cum × PASS_VALUE_S            (PASS_VALUE_S = 6.0 s)
batt_cost    = LAMBDA_BATT × soc_deficit     (λ = 2.0 s/MJ; deficit vs
                                              RESERVE_TARGET_MJ = 1.6 = 40% store,
                                              priced at the band's WORST case)
wear_cost    = LAMBDA_WEAR × max(0, wear_delta)   (λ = 0.08 s per health-%
                                              vs the Balanced baseline)
cliff_cost   = CLIFF_RISK_S × 1{drained AND push_lever > 0}   (1.5 s, capped)
risk_cost    = LAMBDA_RISK × (1 − cum) × DIRTY_AIR_PENALTY_S  (λ = 1.0, penalty 4.0 s)
latency_cost = LAMBDA_LATENCY × deferral      (λ = 0.10 s/lap vs the earliest pass;
                                              deferral = pass_lap − start_lap, else horizon)

infeasible   = reserve_breach OR (pass requires draining the store to floor with a push lever)
```

**Hard constraints** (never recommended regardless of score): reserve breach
(min SOC < 10%-reserve − 0.05) and floor-draining strikes. Wear/cliff pricing
is per-phase for two-phase policies (phase 1 banks at zero wear — negative
lever clamps to 0; phase 2 pays full lever per lap), with cliff risk applied to
whichever phase carries the positive lever. Confidence: margins mapped through
CONFIDENCE_SPAN_S = 1.5 (saturation span) × the SoC-band multiplier (1.0 at
±2% → 0.6 at ±8%).

- **`LAMBDA_LATENCY = 0.10`** (waiting is not free) and **`pass_gain =
  cum × PASS_VALUE_S`** are the engine's deliberate constants — disclosed in
  the payload (`lambda_latency_s_per_lap`). Tests that once hard-coded
  artifacts of a single classifier snapshot were rewritten to pin the
  deterministic layer instead (commit `7df34d7`).
- **Latency budgets:** LATENCY_BUDGET_MS = 1500 (coupled call), engine cache
  CALL_ENGINE_CACHE_MAX = 8.

### 6.2 Live two-car call — `overtake_inference.py` (`simulate_live_call`)

The dual-agent API (P0). A coupled walk projects leader and chaser lap-by-lap
using real stored traces: per-lap `pace_gap_s` (leader − chaser), gap paths,
cumulative pass probability (window trigger RACE_TRIGGER_PROB = 0.5, walk ends
at LIVE_PASS_CUM = 0.8), energy-led verdicts.

- **Chaser levers:** `chaser_ers` (−100…+100, MJ/lap at **ERS_MAX_MJ_LAP =
  0.12 MJ per full deflection**), battery pct, funded from the chaser's 4 MJ
  store with a 30% sustain floor (ERS_FLOOR_MJ = 1.2; default start 62.5%).
- **Leader postures** (P1): `balanced` / `defensive_boost` — the leader deploys
  from its own store (net 2.0 MJ/lap at full defence = LIVE_DEFENSE_NET_MJ)
  down to the same 30% floor, then reverts; energy-limited laps counted at the
  floor. Defence narrows the attack on the deterministic layer (per-lap pace
  edge, gap dominance, wider closest approach). Explicit levers override the
  preset; back-compat preserved (no lever args = legacy behavior).
- **Attack gate:** chaser SOC must stay above LIVE_ATTACK_MIN_SOC_PCT = 30;
  walks hard-capped at LIVE_MAX_SINGLE_STINT = 42 laps (beyond = extrapolation).
- **Braking-zone rendering:** the model's per-lap probability is redistributed
  over 12 lap-fraction segments (RACE_SEGMENTS) weighted by interpolated speed
  drops (BRAKE_WEIGHT 0.5) — honest per-lap output, spatially *rendered*,
  never claimed as per-corner measurement (the guard bans that phrasing).

### 6.3 Benchmarks & reliability

- **`benchmark_energy_strategy.py`** — the Monaco 2023 anchor: per-lap energy
  projection per mode, strategy = fastest feasible mode (final battery must
  clear the reserve + FEASIBLE_MARGIN_MJ = 0.05), flat-out = push, closing
  phase = final 15 laps. **Result: closing-phase −1.875 s (deck claim −1.8,
  reproduced), full-race −9.44 s** — and **bit-identical on full-resolution
  telemetry** (`backtests/monaco_2023_energy_fullres.json`; ~644 samples/lap,
  density scale 1.0; only the pit-lap harvest changed, 2.0→1.24 MJ, real
  measurement replacing the sparse-cap).
- **`benchmark_energy_fleet.py`** — the same projection across every stored
  race: committed summary = **140 races, closing-phase mean −4.49 s, full-race
  mean −16.41 s, biggest gain −12.85 s (Red Bull Ring 2026)**. (Artifact
  predates ~100 later DB additions — refresh pending; drift check flags it.)
- **`check_energy_benchmark_drift.py`** — CI-grade determinism guard: re-runs
  the anchor (and optionally fleet) and fails at |Δ| > 0.02 s. Anchor passes
  bit-exact.
- **`race_call_reliability.py`** — replays all stored races through the same
  forward projection the live call uses: **1,717 checkpoints, 527 battle
  segments, 84 race weekends.** Confidence ≥ 0.80 → **18% real pass rate vs
  the 10% base rate** of an arbitrary in-battle checkpoint (1.85–1.88× through
  the mid-range, **2.12× at P ≥ 0.95**). Attack-window detection published
  honestly: **precision 38%, recall 65%** (TP 240 / FP 394 / FN 132). Windows
  end at the next pit stop (pit-strategy passes out of scope); MIN_BUCKET_N 15
  for reliability bins. Artifacts: `scripts/backtests/race_call_reliability.json`
  + a slide-shaped `.md`.

---

## 7. VALIDATION & GUARANTEES

### 7.1 Test suites (12 files, 128 tests) — run from `scripts/`:
`python -m unittest discover -s tests -p "test_*.py"`

| Suite | Pins |
|---|---|
| `test_policy_engine.py` | constraint enforcement (reserve/floor pruning), decision intelligence (battery state gates the feasible set: pushes feasible at 90%, structurally barred at 31% with a zero-spend conservative winner), score components incl. `pass_gain_s` |
| `test_two_phase_pricing.py` | exact analytic strike-wear formula (rate × strike laps, reconstructed from the row — exact under ANY classifier), phase-2 selection |
| `test_leader_defense.py` | defence funded from the leader store, floor-counted energy-limited laps, deterministic attack-narrowing (pace edge, gap dominance, closest approach), adversarial flip at mid battery |
| `test_leader_perspective.py` | deploy narrows / banking widens on the deterministic layer; explicit lever > preset; back-compat |
| `test_unified_call.py` | the coupled dual-seat decision panel |
| `test_soc_uncertainty.py` | band model (floor 2, +0.5/lap, cap 8), trace/live-call/policy-engine integration, worst-case pricing, confidence degradation with horizon |
| `test_isotonic_calibration.py` | two-scale contract (raw vs calibrated), out-of-time split recorded, endpoint schema, graceful degradation |
| `test_fullres_regen.py` | density-aware scaling across all trace regimes, brake-boolean fix, time anchor, distance integration |
| `test_race_call_reliability.py` | reliability artifact structure + determinism |
| `test_p0_fixes.py` | the P0 red-team fixes on live calls |
| `test_verify_claims.py` | the claim guard itself (banned families + clean samples) |
| `test_backtest*` / others | live-call backtesting, remaining contracts |

Design principle: decision tests pin the **deterministic pace/energy layer**
(per-lap pace gaps, gap dominance, feasibility), *not* classifier-scale
artifacts — so retrains don't churn the suite.

### 7.2 The claim guard — `verify_claims.py`

Scans every text file (skip-lists for VCS/caches/DB dumps, files >5 MB,
binaries; minimal exemptions) for four banned families, exit 1 on any hit:
1. **lap-count accuracy tolerances** ("±N laps") — never measured;
   quote committed MAE/R² instead.
2. **single-digit-millisecond latency** — measured budgets are ~200 ms-class
   live calls / seconds-class rollouts; fiction banned.
3. **rulebook battery attribution** — the 10% floor is SELF-IMPOSED; no FIA
   value encoded or cited.
4. **per-corner spatial claims** — output is braking-zone pass-mass
   attribution, model-shaped and labelled.

### 7.3 Self-audit — `DECK_FIXES.md`

Two-pass audit of the pitch deck against the repo: 4 fixes (per-corner
phrasing, unscoped 284 ms, "guaranteeing the battery", live-UDP wording) plus
8 second-pass items (another single-digit-ms chip on the accuracy slide,
`schema.sql` filename, stale "23 tracks",
non-code energy constants, 6 mode chips vs 3 implemented, the unbacked
Lap-12/Lap-13 anecdote, missing LR-vs-RF numbers, repo URL check).

---

## 8. DASHBOARD

`scripts/dashboard.py` (Flask, ~3,000 lines) + `dashboard/` (HTML + 5 JS
modules). Served by `run_server.py` on waitress; host/port via `.env`
(`HOST`/`PORT`, env vars win).

- **Race Call view** — the unified dual-seat panel: coupled call, leader
  posture, policy table with score ledgers (`policies.js` renders the pass
  gain / battery / wear / latency breakdown of the winning policy).
- **Energy Sandbox** — interactive what-if sliders with warm recalculation in
  tens of ms (the deck's "284 ms" is this endpoint, scoped); battery push
  sliders bounded by real physics (`d_max = min(d_req, DEPLOY_RATE × t,
  soc − reserve)`).
- **Model Calibration & Reliability panel** — `/api/overtake/reliability`
  renders the isotonic curve, reliability bins, Brier/ECE; predictions show
  raw + calibrated scales with a fitted/RAW badge.
- **Live projected leaderboard** (what-if), tyre/strategy advisor, energy
  traces with shaded uncertainty bands, driver comparison views.

---

## 9. COMPLETE FILE INVENTORY

**Core application (`scripts/`)**
| File | Role |
|---|---|
| `config.py` | env/.env-based DB config; refuses the placeholder password (fails loudly, never guesses) |
| `policy_engine.py` | multi-policy decision engine: 5 policies, scoring formula (§6.1), hard feasibility, confidence, leader-threat pricing (LOST_POSITION_S = 6.0, HELD_LAP_CREDIT_S = 0.2) |
| `overtake_inference.py` | dual-agent live call: coupled walks, chaser levers, leader postures, braking-zone rendering, calibrated two-scale probabilities |
| `energy_simulator.py` | battery synthesizer: PU specs, modes, density-aware regen, SOC bands (§5.4) |
| `tyre_degradation.py` | closed-form tyre health, measured per-track wear, cliff penalty (§5.3) |
| `fuel_estimation.py` | shared synthetic fuel feature (§5.2) |
| `feature_pipeline.py` | shared features: lap phases, eras, encodings |
| `ml_lap_predictions.py` | lap-time model training (LR, 83 features) |
| `ml_overtake_predictions.py` | overtake model training + isotonic calibration (§4.2–4.3) |
| `benchmark_models.py` | model selection benchmark (LR vs alternatives) |
| `driver_comparison.py` | per-driver lap-time models, head-to-head |
| `stint_analysis.py` | per-stint lap-time analysis (dashboard + CLI) |
| `measure_tyre_wear.py` | per-track tyre-wear measurement (shrinkage, clamps) |
| `import_f1_race.py` | FastF1 importer, full-resolution telemetry |
| `import_f1_dataset.py` | bulk multi-race import |
| `capture_telemetry.py` | live UDP game-telemetry client (in development) |
| `migrate_telemetry_fullres.py` | idempotent full-res schema migration (+purge) |
| `predict_lap_times.py` | CLI lap-time predictions |
| `backfill_sector_times.py` | FastF1 sector-time backfill |
| `race_calendar.py` | FIA calendars 2020–2026 + coverage annotation |
| `cleanup_pit_events.py` | pit-event hygiene |
| `check_db_schema.py` | schema smoke test |
| `db_transfer.py` | full DB backup/restore |
| `benchmark_energy_strategy.py` | Monaco anchor benchmark |
| `benchmark_energy_fleet.py` | fleet-wide energy benchmark |
| `check_energy_benchmark_drift.py` | determinism drift guard |
| `race_call_reliability.py` | reliability backtest (§6.3) |
| `backtest_race_calls.py` | live-call backtesting harness |
| `validate_features.py` | feature-validation harness |
| `verify_claims.py` | the claim guard (§7.2) |
| `run_server.py` | waitress WSGI runner |
| `dashboard.py` | Flask app (~50 routes: race call, sandbox, calibration, advisor) |

**Front end** — `dashboard/dashboard.html`; JS modules: `core.js` (boot/nav),
`dashboard.js` (live telemetry/energy charts), `drivers.js` (comparisons),
`policies.js` (policy table + score ledgers), `whatif.js` (sandbox + reliability panel).

**Tests** — `scripts/tests/` (12 suites, 128 tests, §7.1).

**Artifacts** — `backtests/` (Monaco anchor legacy + full-res, fleet summary),
`scripts/backtests/` (reliability JSON + slide), `ml_models/` (gitignored,
regenerable: best_model.pkl, overtake/, energy_pace.json,
tyre_wear_per_track.json, drivers/, comparisons/),
`ml_models/tyre_wear_per_track.json` + `energy_pace.json` (measured
calibrations), `database/f1_strategy.sql` (schema + seed, carries time_s /
distance_m).

**Docs** — `README.md` (judge-facing claim→artifact map), `DECK_FIXES.md`
(deck audit), `PROJECT_GUIDE.md` (this file), `launchers/README.md`
(numbered Windows launchers 01–13: setup → import → train → serve → test).

---

## 10. WHY THINGS WERE MADE THIS WAY (decision log)

| Decision | Why |
|---|---|
| LinearRegression over RF/boosting | monotonic tyre response (no staircase extrapolation), interpretable coefficients; AUC 0.967 shows the linear overtake model suffices |
| Deterministic tyre model, not fitted | explainable charts; "physics guardrails" pitch survives judge scrutiny |
| Two-scale calibrated probability | thresholds stay tuned on the raw scale; calibration evidence without breaking the decision layer |
| Temporal (not random) split | the product predicts *forward*; random splits leak future regimes |
| Synthesized battery with published bands | no SOC feed exists; fake precision would be worse than an honest band |
| Refusal-based feasibility (prune, don't downscore) | "guarantee" claims are unfalsifiable; "it refuses" is demonstrable in one demo beat |
| Density-aware regen (×2.2 legacy → 1.0 full-res) | removes the sparse-sampling fudge exactly where data becomes real; legacy numbers unchanged |
| Era-aware PU specs | 2020–2025 races simulated with legacy PU regs, 2026 with current — never anachronistic |
| Latency & wear penalties in the score | encodes real opportunity cost (waiting loses track position; tyres pay later) |
| Claim guard as a test | credibility is a build property, not a review step; the deck audit found banned strings that the guard now keeps out |
| Deterministic-layer tests | classifier retrains must not churn CI; physics is the stable contract |
| Drift checker on artifacts | committed numbers are load-bearing (deck quotes them); silent drift would be a credibility bug |
| Sandbox-via-env (DB_NAME override) | destructive pipeline work provable without touching the demo dataset (exercised for the full-res proof) |

---

## 11. ADVANTAGES (what this buys over a naive build)

1. **Every slide number is reproducible** from the repo in one command — most
   competitors' numbers die at "how was that measured?"
2. **Calibrated, disclosed uncertainty** — rare in student projects; the
   reliability table converts "trust us" into "here is the measured pass rate."
3. **Two-car coupling** — attack *and* defence priced from the leader's
   perspective; naive simulators model one car's fantasy.
4. **Regulation-shaped physics** — era-correct PU limits (4 MJ store, 2/8.5 MJ
   harvest caps, 120/350 kW) instead of fantasy 20 MJ batteries.
5. **Refusal as a feature** — the demo beat ("watch it say no at 31% battery")
   is memorable and honest.
6. **Determinism** — replayed backtests produce identical numbers (drift
   guard), so any judge re-run matches the deck.
7. **Honest small-label handling** — precision/recall published, "we rank
   windows, we don't certify probabilities" — disarming the toughest questions.
8. **Full-resolution upgrade path already proven** — the headline survives the
   100× denser telemetry bit-identically.

---

## 12. KNOWN LIMITATIONS & ROADMAP (all disclosed, none hidden)

1. **Full-res is sandbox-proven, not production-wide** — production DB rows are
   still legacy ~6 samples/lap; promote via
   `migrate_telemetry_fullres.py --purge-telemetry` + full re-import +
   retraining when ready.
2. **Live UDP capture: zero recorded sessions** — client is built/tested;
   labelled "in development" (deck updated accordingly).
3. **Fleet artifact freshness** — committed summary covers 140 races; the DB
   has grown (~100 races added since). Re-run `benchmark_energy_fleet.py` to
   refresh; drift check currently flags 3 races for this reason only.
4. **Saturation re-pin** — if/when the classifier is retrained to
   de-saturate, re-pin the rich-vs-low battery action-card flip in
   `test_policy_engine.py` (documented in the test).
5. **Overtake precision is low (0.19)** — inherent to rare events; recall 0.65
   and the reliability table are the honest framing.
6. **Intra-lap time absent from live capture rows** — the UDP client has no
   intra-lap clock yet (columns stay NULL-compatible).

---

## 13. NUMBERS CHEAT SHEET (verified, quote-ready)

| Number | Value | Source artifact |
|---|---|---|
| Monaco 2023 closing-phase improvement | **−1.875 s** (deck: −1.8) | `backtests/monaco_2023_energy.json` (+ `_fullres.json` bit-identical) |
| Monaco full-race improvement | −9.44 s (legacy) / −9.49 s (full-res) | same |
| Fleet (140 races) closing-phase mean | −4.49 s | `backtests/energy_fleet_summary.json` |
| Reliability: P ≥ 0.80 real pass rate | **18% vs 10% base** | `scripts/backtests/race_call_reliability.json` |
| Reliability: P ≥ 0.95 lift | **2.12×** | same |
| Attack-window precision / recall | 38% / 65% | same |
| Checkpoints / segments / weekends | 1,717 / 527 / 84 | same |
| Isotonic Brier / ECE (raw → cal) | 0.0317 → 0.0077 · 0.123 → 0.000 | `ml_models/overtake/model_info.json` |
| Overtake AUC / accuracy / precision / recall | 0.967 / 0.970 / 0.19 / 0.65 | same |
| Lap model MAE / R² (within-track) | 1.61 s / 0.932 | `ml_models/model_info.json` |
| Track coverage | lap model 32, overtake model 29 | both model_info.json |
| Full-res Monaco re-import | 200,881 samples, ~644/lap, 0 fails | session 125–128 |
| Tests | 128 / 12 suites, green | `scripts/tests/` |
| Key constants | reserve 10% · deploy 0.12 MJ/s · store 4 MJ · harvest 2 / 8.5 MJ/lap · tyre cliff 40% / 2.5 s/lap · PASS_VALUE_S 6.0 · LAMBDA_LATENCY 0.10 | code (§5–6) |
