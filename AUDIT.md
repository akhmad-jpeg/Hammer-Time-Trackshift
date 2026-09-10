# HAMMER TIME / TRACKSHIFT — WINNING-LEVEL AUDIT
*Audited as CTO + principal ML engineer + race strategist + red-team engineer + hackathon judge. Every verdict below is grounded in code actually read and endpoints actually exercised on the live DB (2026-09-09). No code was modified.*

---

## 1. EXECUTIVE VERDICT

**Would I award first place right now? NO.**

What exists is genuinely more than a hackathon dashboard: a real three-layer stack (FastF1/UDP → MySQL → Flask), trained models with committed artifacts, a deterministic energy simulator with era-correct PU specs, a dual-agent overtake model, a 140-race deterministic fleet benchmark, a drift checker, and a 58/58-passing end-to-end validation harness. That foundation beats most submissions before they open their mouths.

But the product has a **decision-shaped hole**. There are three separate decision surfaces (pit advisor, energy-mode recommender, live overtake call) that never talk to each other, never compare counterfactual policies side-by-side, and never quantify uncertainty. The one number judges will poke first — the overtake "probability" — comes from a classifier trained on **26 positive labels**, with precision 0.30, presented without calibration. The deck's latency claims are **false as measured** (P0 inference: 43 ms, not <2 ms; full-race P1 replay: 789 ms, not <500 ms). Red-team probing found a 500-crash in the energy sandbox, a silently-ignored battery override, and a live call that says **ATTACK under every ERS posture including full save**. None of these are fatal individually; together they mean the headline claims collapse under three minutes of competent questioning.

The fix is not more features. It is one decision object, honest uncertainty, and four bug fixes — all achievable before submission.

---

## 2. CURRENT WIN SCORE: **54 / 100**

| Dimension | Score | Why |
|---|---|---|
| Problem clarity | 8 | The "synthesize what F1 doesn't broadcast" framing is excellent |
| Technical depth | 8 | DP stint optimizer, era-correct PU specs, hazard accumulation, mixed-model benchmarks |
| AI/ML quality | 5 | 26 overtake labels; no calibration curve artifact; unseen-track R² 0.953 > within-track 0.932 is unexplained |
| Novelty | 7 | Dual-agent P0/P1 framing + sector-weighted pass probability is distinctive |
| **Decision intelligence** | **3** | Three siloed recommenders; no policy comparison; ATTACK in all postures |
| Physical realism | 6 | Constraints enforced in-sim (floors, headroom, flow caps); pace credit is an unvalidated anchor |
| Data credibility | 7 | Synthetic layers loudly labeled; imputation counts recorded |
| Validation | 5 | 58/58 harness + drift check are real; no tests/ folder; no chronological split |
| Real-time capability | 4 | Measured 43 ms P0 / 789 ms P1 vs claimed 2 ms / 500 ms |
| Demo quality | 6 | Sandbox is interactive; no adversarial flip demo |
| Explainability | 8 | Reasons attached to every recommendation; honest modes table |
| Robustness | 4 | Sandbox 500-crash; ignored inputs; no tests |
| Rule compliance | 5 | Simulator enforces self-imposed floors; "FIA Battery Minimums" is a mislabel the code itself contradicts |
| UX | 7 | Split dashboard is clean |
| Differentiation | 5 | The differentiator exists (counterfactuals) but is not assembled |
| **Overall** | **54** | Strong bones, unsupported headlines, no unified decision |

---

## 3. PHASE 0 — REPOSITORY AUTOPSY

### A. Architecture map (verified)

```
INGEST      import_f1_race.py (1196) / import_f1_dataset.py (330)   FastF1 → MySQL (~6 samples/lap)
            capture_telemetry.py (1119)                             F1 2017/18 UDP → same schema, ~1 Hz
STORE       database/f1_strategy.sql (382)                          sessions, laps, telemetry, race_state,
                                                                    strategy_events, drivers, tracks…
FEATURES    feature_pipeline.py (266)                               one-hots, era/phase buckets, interactions
MODELS      ml_lap_predictions.py (919)                             LinearRegression (deployed) vs RF; per-driver,
                                                                    per-year, 3 sector models → ml_models/
            ml_overtake_predictions.py (604)                        RF closing-rate + Logistic P(pass) → ml_models/overtake/
PHYSICS     energy_simulator.py (783)                               synthetic ERS: regen from speed drops, mode policy,
                                                                    era PU specs, intra-lap SOC path → race_state
            tyre_degradation.py (264) + measure_tyre_wear.py (211)  Pirelli-style health curve, measured wear cells
INFERENCE   overtake_inference.py (2256)                            P0 pair predict, P1 full-race sim, live-call
                                                                    projection, pit-stop derivation, calibration
STRATEGY    dashboard.py (3352, routes 1552–2268)                   DP stint optimizer + advisor + energy-mode
                                                                    recommender + energy sandbox
PRESENT     dashboard/ (Flask+Chart.js)                             25+ API routes
PROVE       benchmark_energy_strategy.py (419) / fleet (248) /
            backtest_race_calls.py (476) / check_energy_benchmark_drift.py (166) / validate_features.py (318)
```

### B. Critical execution path (exercised live)
`POST /api/overtake/live` → per-driver career pace models → gap walk → P0 classifier inside 1.2 s window → hazard `1−Π(1−pᵢ)` → verdict at cum ≥ 0.8. Measured: **196 ms** end-to-end.
`POST /api/strategy/energy-analyze` → regen from stored speed traces → `project_energy_trace` per mode → pace credit = measured s/MJ × deployed MJ → fastest feasible mode. Measured: **93 ms**.
`POST /api/strategy/analyze` → DP stint plans × measured wear + cliff penalty + pit-loss-by-event → ranked strategies + reason. Measured: **702 ms**.

### C. What actually works (exercised)
- `validate_features.py`: **58/58 PASS** against the real DB/models (lap predict, advisor, comparisons, live call, energy sim, tyre endpoint).
- `check_energy_benchmark_drift.py`: fresh re-run of all 140 races matches committed artifacts exactly (deterministic — genuinely impressive).
- Energy-mode recommender returns a feasible-mode decision with an explicit battery-floor rule (Monaco 2023 deck claim reproduced: −1.88 s closing phase vs Flat-Out).
- Live-call ERS lever genuinely moves pace (0.87 vs 0.55 s/lap) and SOC (53→44% vs 71→80%) — the *mechanics* of the sandbox work.
- Sandbox clamps over-ask: requesting 3 MJ with a near-empty store delivered only 1.48 MJ, min-SOC floored at the reserve.

### D. What only appears to work
- `chaser_battery_pct` parameter: accepted, stored in meta as `None`, **output identical for 90% vs 31% battery** — the projection ignores it.
- Sandbox net-deployment deltas: `s1=3.0` came back with `effective_deltas_mj = {0,0,0}` — explicit "deploy MORE" is silently normalized to a pure reallocation. The sandbox can only reshuffle the lap budget, never raise or lower it.
- Live-call "verdict": it is just `cum_probability ≥ 0.8` — the same ATTACK verdict (0.90 and 0.81) fired for both push and save postures.
- `tests/` folder referenced by launcher 11 does not exist; "DB access is mocked, so this also works" describes a suite that isn't there.

### E. What is simulated (and labeled as such — to the project's credit)
Battery SOC (whole race_state layer), fuel load (110 − 2·lap), intra-lap SOC path (wiggle-compressed), pace-credit s/MJ (anchor 0.35 × FT share), tyre health curve (measured *relative* wear + Pirelli-researched absolute levels).

### F. What is genuinely predictive
- Lap-time model: within-track MAE 1.609 s, R² 0.932 (12,018 laps, 32 tracks) — artifact committed.
- Overtake classifier: AUC 0.977, **precision 0.30 / recall 0.75 on 26 positive labels** — artifact committed, honest, tiny.
- Race-call backtest machinery (1,942-checkpoint capability) — scripts exist; the supporting JSON artifacts were deleted in the last commit, so the calibration numbers (24.7% pass rate at P≥0.8) are currently **assertions, not reproducible artifacts**.

**Dead code / duplication:** `_normalise_track_name` duplicated (overtake_inference + feature_pipeline/dashboard); track-name aliasing implemented three times (energy_simulator, overtake_inference, ml_overtake_predictions); `strategy_recommendations` table checked by check_db_schema but absent from schema; pace s/MJ lookup re-implemented in dashboard.py, overtake_inference.py, and benchmark scripts.

---

## 4. PHASE 1 — CLAIM VS REALITY

| CLAIM | FILE | IMPLEMENTATION | EVIDENCE (exercised) | RISK | VERDICT |
|---|---|---|---|---|---|
| Corner-by-corner overtake probability | overtake_inference.py (S1/2/3 split + 12-seg heatmap), dashboard `corner_heat` | Probabilities split by gain-weight + braking evidence; P1 sim returns `corner_heat` | `full_race.corner_heat` present in live response | Low — it's a weighted *attribution*, not corner-labeled training | **PARTIALLY PROVEN** (call it "sector-weighted attribution", not corner-level learning) |
| Dual-agent leader/chaser modeling | ml_overtake_predictions.py, P0/P1 | Real pair rows, role swap, gap walk | Live call VER/HAM worked end-to-end | Low | **PROVEN** |
| Energy sandbox | dashboard.py:1923, whatif.js | Per-sector MJ sliders, SOC path | Works, 59–88 ms | **High** — net deploy silently zeroed; all-negative deltas → HTTP 500 | **SIMULATED + BUGGY** |
| Physics guardrails | energy_simulator clamps, tyre floor/cliff | Floors, headroom, flow caps, ±1 MJ/lap | Low-battery probe clamped correctly; save-lap returned min_soc 0% (below the 10% floor) in the crashed-probe default display | Medium | **PARTIALLY PROVEN** (in-sim yes; decision layer no) |
| Synthesized ENERGY_REMAINING | energy_simulator.py, race_state | Explicitly synthetic, labeled everywhere | UI label + docstrings verified | Low — keep the label | **PROVEN (as synthetic)** |
| **<2 ms inference** | deck p.8 | — | Measured **43.1 ms/call** P0 (200-run avg) | **Fatal if challenged** | **INCORRECT** |
| **<500 ms simulation** | deck p.8 | P1 replay | Measured **789 ms** for 26 laps | High | **INCORRECT** (on demo hardware) |
| ±1 lap prediction | deck p.8 | Only appears in Q&A as a *proposed* tyre-model validation | No artifact, no test | High | **NOT IMPLEMENTED** |
| −1.8 s race-time improvement | benchmark_energy_strategy.py | Deterministic repro: −1.88 s closing phase, Monaco 2023 | Drift checker re-verified vs committed artifact | Medium — it's model-vs-model, same simulator on both sides | **PROVEN (in-model)** — must be presented as such |
| FIA battery compliance | deck | Simulator floors at 10% reserve | Code comment (energy_simulator.py:194) itself says the reserve is management practice, **"NOT an FIA rule"** | High — a knowledgeable judge kills this | **INCORRECT (mislabel)** |
| Real-world validation | deck | Fleet sweep = the same synthetic projection on real lap times | 140 races, deterministic | Medium | **SIMULATED** |
| Model accuracy (MAE 1.609 s, R² 0.932) | model_info.txt | Committed artifact | Read directly | Low | **PROVEN** |
| Overtake AUC 0.977 / P 0.30 / R 0.75 | ml_models/overtake/model_info.json | Committed artifact | Read directly | Medium (26 labels) | **PROVEN but fragile** |
| Unseen-track R² 0.953 | model_info.txt | Session-grouped split | Odd: better than within-track; unexplained | Medium — invite to a leakage question | **PROVEN but suspicious** |

**Claims to remove before the judges see them:** `<2 ms`, `<500 ms` (re-measure and restate), `±1 lap`, `FIA battery minimums compliance` (relabel "self-imposed management reserve, regulation-shaped"), and any sentence implying SOC is measured.

---

## 5. PHASE 2 — THE MOST IMPORTANT QUESTION

**Are we answering "which action NOW"? No — we answer three unrelated questions with three unrelated recommenders.**

- Pit advisor (`/api/strategy/analyze`): "Stay Out … would end age 39 on this track" — tyre-only. It has *no energy state in its objective*.
- Energy recommender (`/api/strategy/energy-analyze`): "liftcoast, finishes 89.7%" — battery-only. It has *no overtake or position term*.
- Live call (`/api/overtake/live`): "attack, pass lap 31" — overtake-only. It has *no future race-time accounting* and currently ignores the battery you pass it.

The ACTION card the challenge demands (ATTACK / window / expected gap / probability / energy cost / finish delta / margin / confidence / reason) **cannot be assembled today from any single endpoint**, and no endpoint compares DEFEND/BALANCED/SAVE/WAIT against the current policy. That comparison object — not another chart — is the missing product.

---

## 6. PHASE 3 — DECISION ENGINE DESIGN (what to build)

Use a **heuristic-constrained rollout scorer**: for each candidate policy p ∈ {ATTACK, BALANCED, SAVE, LIFT&COAST, WAIT-1-LAP}, roll the existing `simulate_live_call` + `project_energy_trace` forward to the flag, then score

```
V(p) = E[Δposition·pts] + E[Δrace_time] − λ·E[max(0, reserve_deficit)]
       − P(failed attack)·cost − rule_violation·BIG
```

and emit the argmax **with all five trajectories side by side**. Rationale: the components (deterministic simulator + calibrated classifier + per-lap credit model) already exist; DP/MPC would require a joint state space (SOC × tyre × gap) you can't calibrate from 26 overtake labels. Rollout is Monte-Carlo-free, deterministic (demo-safe), reuses 100% of existing code, and its output *is* the counterfactual table. The "why" is then computed from the score components — explainability falls out for free.

---

## 7. PHASE 4 — ENERGY MODEL RED TEAM (findings)

1. **Consumption is a policy, not a physics function.** Deployment = "ask the mode's ceiling" or "track a SOC target". Throttle/speed/gear do not drive consumption; only regen does (speed drops). Flag it as such.
2. **Pace credit is an uncalibrated anchor** (0.35 s/MJ × full-throttle share). The measured part is the *share*, not the s/MJ itself. No lap-time-delta validation exists. This is why the fleet sweep shows −57 s full-race improvements — a number a motorsport judge will find implausible on its face.
3. **Path dependence and recovery ARE modeled correctly**: headroom-limited harvest, reserve floor, 0.85 redeploy on limited laps, 30–80% soft band. This is the model's strongest suit.
4. **Exploit found:** `save everything` deltas → ZeroDivisionError (HTTP 500) at dashboard.py:2087 (`tot_req == 0`).
5. **Exploit found:** net deployment deltas are normalized away — the sandbox cannot answer "what if I push 7% more now", the single most important counterfactual in the challenge brief.
6. **Uncertainty: absent.** Battery is a point estimate everywhere. Because SOC is synthetic, ±band is not optional — it is the honesty layer. Propagate: SOC ± (regen estimation error + policy drift), grow the band each projected lap, shrink confidence of any decision whose margin < band.

**Minimum credible upgrade:** per-lap SOC band (`±(1.5 + 0.35·laps_ahead)%`), clamp to [reserve, 100], and require `V(attack) − V(save) > 2·band` before recommending ATTACK. Plus: fix the two sandbox bugs.

---

## 8. PHASE 5 — OVERTAKE MODEL RED TEAM

- Target: binary swap-from-≤5 s label on battle laps. **26 positives out of 2,179.** Precision 0.30 means 70% of "attack" calls are false alarms — the project knows this (threshold raised 0.5→0.8 from backtest evidence) but the deck doesn't say it.
- Output is an uncalibrated LogisticRegression probability; the hazard accumulation converts per-lap scores to a cumulative number that is *not* a calibrated probability of a pass by lap X.
- DRS: not a feature. Defence: not a feature. Braking-zone geometry: only via the sector-weighting heuristic.
- The 0.8 cumulative threshold and its calibration-bucket justification (24.7% vs 12.9%) is exactly the right *kind* of evidence — but the backtest artifacts backing it were deleted from the repo.

**Minimum credible fix:** (1) re-run `backtest_race_calls.py` and commit the calibration JSON as an artifact; (2) add a reliability table (predicted-bucket vs realized-rate) generated from the backtest; (3) relabel the UI column "P(overtake) — uncalibrated, 26 training labels" or fit isotonic calibration on held-out races (~20 lines with sklearn). Do not add SMOTE; do not add features you can't defend.

---

## 9. PHASES 6–8 — ADVERSARIAL SCENARIOS, COUNTERFACTUALS, UNCERTAINTY (as tested)

Exercised probes and outcomes:

| Scenario | Probe | System behavior | Naive? |
|---|---|---|---|
| Battery advantage vs disadvantage | `chaser_battery_pct` 90 vs 31 | **Identical output** — ignored | Broken |
| Push vs save posture | `chaser_ers ±100` | Pace + SOC move correctly, but **verdict ATTACK in both** | Broken (verdict layer) |
| Over-ask vs empty battery | sandbox +3 MJ @ 0.5 MJ store | Clamped to 1.48 MJ, floor held | Good |
| Save-everything | sandbox −3/−3/−3 | **HTTP 500** | Broken |
| Deploy-more | sandbox +3 MJ net | Silently normalized to zero effect | Broken |
| SC/VSC/red flag | — | Handled as pit-loss discounts + lap exclusion; no state machine | Partial |
| Sensor dropout / latency / wet | — | Not modeled | Gap |
| Final-lap desperation | — | No "opportunity cost of not attacking now" term | Gap |

The scenario suite a submission needs: the 20 scenarios in the challenge brief map to **one missing object** — a policy comparison table under perturbed inputs. Build that once and scenarios 1–20 become demo talking points instead of liabilities.

---

## 10. PHASES 9–12 — GUARDRAILS, LEAKAGE, ABLATION, BASELINES

- **Guardrails:** enforced *inside* the simulator (verified), absent *after* prediction (the ignored battery override proves the seam). Rule: any parameter a user can set must fail loudly if the constraint it feeds is violated — never silently no-op.
- **Leakage:** random within-track 80/20 split (lap-level → same-stint leakage, acknowledged in Q&A). The mystery unseen-track R² 0.953 > within-track 0.932 must be explained or the informational metric dropped. No chronological split exists anywhere. `validate_features.py` passing 58/58 is integration checking, not generalization evidence.
- **Ablation/baselines:** the fleet sweep *is* an ablation of energy modes vs Flat-Out (140 races, deterministic) — genuinely good. There is no ablation of the overtake model (pace-gap-only heuristic vs full P0) and no combined-baseline table (flat-out / always-save / greedy-attack / ours). The energy benchmark's honest framing — "shared baseline cancels; delta is model-internal" — should be reused for a race-call table.

---

## 11. PHASE 13 — JUDGE SIMULATION

**Judge A (ML engineer, 3 min):** Impressed by artifacts + drift checker + honest imputation counts. Skeptical of 26 labels, unexplained unseen-track R², no chronological split, no calibration curve. Will ask: *"Your 73% — how often was it actually 73%?"*. Kills us if: `<2 ms` appears on a slide. Remembers us if: reliability table + conformal-style SOC band.

**Judge B (motorsport engineer):** Impressed by era-correct PU specs, VSC/SC pit-loss discounts, pit-window 40–50% band. Skeptical of −57 s full-race swings, anchor 0.35 s/MJ, "FIA battery minimums" (there is no such sporting rule — your own code says so). Will ask: *"What is your energy model calibrated against?"*. Remembers us if: the live-call flips recommendation when one variable changes.

**Judge C (startup/hackathon):** Impressed by a working product in 3 minutes. Skeptical of dashboards without decisions. Will ask: *"So what does it tell the pit wall to DO?"*. Remembers us if: the ACTION card with five policy trajectories.

**Scoring summary:** see the table in §2 — Decision intelligence (3) and Robustness (4) are the anchors dragging an 8-depth project to 54.

---

## 12. PHASE 14 — THE 1% DIFFERENTIATOR

**Counterfactual race-policy comparison under uncertainty** — one endpoint that returns the five policies' trajectories (finish gap, energy floor margin, P(overtake), risk, confidence) so the pit wall sees *why not* each alternative, not just the winner. Nothing else on the challenge will show regret-avoidance with error bands.

---

## 13. PHASE 15 — DEMO SCRIPT (3:00)

1. **0:00–0:20** Live-call: HAM 0.8 s behind VER, lap 30. Card: ATTACK, pass lap 31, P 0.90, energy cost 0.72 MJ. *(Runs today.)*
2. **0:20–0:45** Flip battery to 31%: with the fix, card flips to **WAIT 1 LAP — margin 2.1% < uncertainty band ±3.4%**. One sentence: "the model knows what it doesn't know."
3. **0:45–1:15** Policy table: five trajectories side by side; highlight SAVE's battery floor and ATTACK's failed-attack risk. "We compare futures, not predictions."
4. **1:15–1:45** Adversarial event: opponent +0.25 s/lap → table re-ranks live (789 ms sim → present as "sub-second"). Show the gap never closing → HOLD.
5. **1:45–2:15** Provenance: `check_energy_benchmark_drift.py` — "our headline number is a committed artifact that re-derives itself; if the model drifts, CI fails."
6. **2:15–2:40** Honesty slide: synthetic layers labeled, 26 labels, precision 0.30, threshold raised from backtest evidence.
7. **2:40–3:00** One-liner: "F1 doesn't broadcast the battery. We rebuilt it, bounded it, and made decisions that admit their own error."

Every step is a re-run of existing machinery + the one new endpoint.

---

## 14–20. FOUR-DIMENSION GRADES (audit skill)

### SPEC — **7/10**
Working capability verified end-to-end (58/58 harness, deterministic benchmarks, live DB). Highest-value gaps:
1. No unified decision object (three siloed recommenders; the ACTION card cannot be assembled from any response payload).
2. No uncertainty representation anywhere (point-estimate synthetic SOC, uncalibrated P).
3. No counterfactual policy comparison (the closest thing — the sandbox — cannot even express "deploy more").

### DESIGN — **4/10**
Pure modules (`energy_simulator`, `tyre_degradation`, `feature_pipeline`) are genuinely well-separated and are the praise-worthy parts. Violations:
1. `scripts/dashboard.py` (3,352 lines) owns routes + DP stint optimizer + advisor reasoning + formatting + caching — six concerns, one file; a change to pit-loss logic lands among HTTP scaffolding (routes 1552–2268 vs plotting helpers vs benchmark passthrough all interleaved).
2. `scripts/overtake_inference.py` (2,256 lines) mixes inference API, backtest helpers, live projection, pit-stop derivation, and its own track-alias/pace-lookup copies (duplicating `energy_simulator` and `dashboard`).
3. Track-name canonicalization implemented 3×, pace s/MJ lookup 3×, `_normalise_track_name` 2× — one owner each is missing.

### CORRECTNESS — **5/10**
Graded only from exercised behavior: 58/58 integration pass, drift check deterministic-match, live-call/energy-analyze/advisor/P1 all functional. But red-team probing found real breakage: sandbox 500-crash on all-negative deltas (dashboard.py:2087), net-deployment deltas silently zeroed, `chaser_battery_pct` silently ignored, ATTACK verdict in every ERS posture, measured latencies contradicting deck claims (43 ms vs 2 ms; 789 ms vs 500 ms).

### QUALITY — **5/10**
~20.2k lines carry more machinery than the demonstrated behavior needs: duplicated track/pace logic (3 copies), dead `strategy_recommendations` schema check, launcher referencing a nonexistent `tests/` folder, and a 2,206-line Q_and_A.txt asserting metrics whose supporting backtest artifacts were deleted. Same behavior plausibly fits in ~16k.

### Single most valuable next pass
**Assemble the decision engine: one endpoint that rolls the five policies through the existing simulator, emits the policy-comparison table with SOC uncertainty bands and score components, and feeds the live-call verdict from it — after fixing the four red-team bugs (sandbox crash, net-delta normalization, ignored battery override, posture-blind verdict).** This single pass converts the project's three silos into the challenge's actual deliverable, fixes every exercised correctness failure, and is ~300 lines because all the physics and ML already exist.

---

## FINAL DELIVERABLES (condensed)

**Top 10 weaknesses (by judge impact):**
1. No policy comparison / "why not attack" (decision hole) 2. `<2 ms` / `<500 ms` claims false as measured 3. "FIA Battery Minimums" mislabel 4. Uncalibrated probability from 26 labels 5. `chaser_battery_pct` ignored 6. Sandbox net-deploy silently zeroed 7. Sandbox 500-crash 8. ±1-lap claim with no artifact 9. No uncertainty bands on synthetic SOC 10. No tests/ folder + deleted backtest artifacts.

**Top 3 high-leverage changes:**
- **P0-a (half day):** Fix the four red-team bugs; re-measure and restate latencies; re-run the race-call backtest and commit calibration artifacts.
- **P0-b (one day):** Policy-comparison decision endpoint (§6 design) + SOC ±band + margin-vs-band gate; wire the live-call verdict to it; add unit tests for the scorer and constraint clamps.
- **P1 (half day):** Isotonic calibration of the overtake classifier on held-out races + reliability table in UI; chronological-split validation run of the lap model committed as artifact.

**Claims we must remove:** `<2 ms inference`, `<500 ms simulation` (as stated), `±1 lap`, `FIA battery minimums compliance`, any implication battery SOC is measured.

**Claims we can prove (with locations):** MAE 1.609 s / R² 0.932 (`ml_models/model_info.txt`); AUC 0.977, P 0.30, R 0.75, 26 labels (`ml_models/overtake/model_info.json`); −1.88 s Monaco closing phase vs Flat-Out, deterministic (`backtests/monaco_2023_energy.json` + `check_energy_benchmark_drift.py` re-run 2026-09-09); 140-race fleet sweep deterministic (`backtests/energy_fleet_summary.json`); 58/58 feature validation (`scripts/validate_features.py` run 2026-09-09); era-correct PU specs (`scripts/energy_simulator.py`).

**Final win conditions (measurable):**
1. Every recommendation on screen carries a confidence derived from propagated uncertainty.
2. The demo shows ≥2 recommendation flips caused by single-variable changes.
3. Every number on the slide regenerates from a committed artifact in <5 min on a clean checkout.
4. Overtake headline restated as a calibrated interval with its reliability table.
5. The policy table shows our engine beating flat-out/save/greedy on the fleet sweep — not just charts.
