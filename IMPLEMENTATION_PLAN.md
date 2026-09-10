# RECONCILED IMPLEMENTATION PLAN — Hammer-Time Trackshift
*Synthesizes three independent audits into one prioritized roadmap. No code has been changed.*
*Date: 2026-09-09 · Sources: `AUDIT.md` (this thread's live-code audit, win score 54), "Audit 1" (win score 42), "Audit 2" (win score 42).*

---

## 0. HOW THE THREE AUDITS RELATE

All three audits **independently converge on the same root diagnosis**:

> The product is a *passive telemetry viewer with projections*, not a *decision engine*. The challenge asks for "a real-time decision engine that recommends optimal energy deployment modes and evaluates the risk-reward ratio of overtake windows." Today the system answers "here is a prediction" and "here is a slider" — not "here is the action you should take, its cost, its risk, and why not the alternatives."

They also agree on the failure family: unprovable presentation claims, energy state decoupled from the ML overtake head, no uncertainty quantification on synthesized SOC, no opponent counter-defense, no automated latency proof.

Where they differ, this plan defers to **code re-verified on 2026-09-09** (details in §2):

| Topic | Audit 1 / Audit 2 | This thread's audit (live-verified) | Resolution |
|---|---|---|---|
| Win score | 42 / 100 | 54 / 100 | Use 42 as the pessimistic bound; true score ≈ 48–54. The gap: the 42-score audits graded against a stale repo state. |
| Monaco 2023 Lap-13 fabrication | "Fatal, file: backtests/monaco_2023.json" | That file **no longer exists** — deleted in the 2026-09-07 cleanup commit; only `monaco_2023_energy.json` (a deterministic energy-mode benchmark, not a pass-prediction backtest) remains | The *deck claim* "Predicted Lap 12, Actual Lap 13" must still be purged from slides; but the repo exhibit cited is gone. The surviving −1.875s artifact is legitimate and re-derivable via `check_energy_benchmark_drift.py` |
| "FIA Battery Minimums compliance" | Audit 2: PROVEN | **MISLABELED**: the 10% / 0.4 MJ reserve is *self-imposed* management margin, not an FIA battery-minimum rule; the code comment at `energy_simulator.py:192-197` says "management reserve" | Keep the mechanism, rename the claim: "self-imposed 10% management reserve, enforced as a hard constraint" |
| `-1.8s` = `MAX_WEAR_PENALTY` copy | "Fatal, dashboard.py:1083" | `MAX_WEAR_PENALTY = 1.8` exists (now line 1107) as a wear cap, **and** `backtests/monaco_2023_energy.json` independently derives −1.875s closing-phase delta deterministically | The number is now *provable* from an artifact — present the artifact, not the coincidence. Purge the "race-time improvement" framing; use "closing-phase energy-mode delta" |
| `<2ms` inference / `<500ms` simulation | Audit 2: "unproven but plausible"; Audit 1: no benchmark exists | **False as measured**: P0 43 ms, P1 789 ms on the live DB | Restate as measured: "<50 ms single-call inference, sub-second full-race replay" — or optimize then re-measure. Never present 2 ms |
| Corner-by-corner overtake | Heuristic slicing (`_corner_pass_mass`) | Confirmed — same code (line 752-773): one lap-level probability redistributed by braking weights | Relabel "Braking-Zone Pass-Mass Spatial Distribution" |
| `energy_diff_mj=0.0` in live sim | Hardcoded, decoupling ML from energy | Confirmed — `overtake_inference.py:2151-2160` passes `fuel_diff_kg=0.0, energy_diff_mj=0.0` | P0 fix: feed projected SOC delta into the classifier |
| Chaser battery ignored / posture-blind ATTACK / sandbox 500-crash | Not caught by audits 1–2 | Found live: `chaser_battery_pct` accepted then ignored; ATTACK under every ERS posture; `ZeroDivisionError` in sandbox on all-negative sector deltas | Add to P0 — these are live judge-killers the other audits missed |
| Uncalibrated probability (26 labels, precision 0.30) | Not caught | Found from `ml_models/overtake/model_info.json` | P1: isotonic calibration + reliability table |

**Net judgment:** the two txt audits are directionally right and slightly stale; the live audit is current and found additional live bugs. Merge both lists — nothing in either audit is invalidated by the repo's current state except the two "fatal file" citations above.

---

## 1. VERIFIED CURRENT STATE (evidence, 2026-09-09)

Proven working (exercised, not just read):
- `validate_features.py` — 58/58 PASS against live DB + models
- `check_energy_benchmark_drift.py` — 140-race deterministic re-run matches committed artifacts exactly
- Dashboard renders fully at http://127.0.0.1:8526/ (Preview tab, this thread) — Monza 2026 session, synthetic ERS chart, tyre health, session list all live
- Live-call ERS lever genuinely moves pace (0.87 vs 0.55 s/lap) and SOC (53→44% vs 71→80%)
- Sandbox clamps over-ask: 3 MJ requested with near-empty store delivered 1.48 MJ, min-SOC floored at reserve

Confirmed broken (exercised, not just read):
- `chaser_battery_pct` accepted then silently ignored — identical output at 90% vs 31%
- Live call returns ATTACK under every ERS posture, including full save
- Energy sandbox 500-crash: all-negative sector deltas → `ZeroDivisionError` (`dashboard.py:2087`)
- Explicit "deploy more" in sandbox: net deltas silently normalized to `{0,0,0}` — sandbox can only reshuffle, never raise the lap energy budget
- `energy_diff_mj=0.0` hardcoded in live forward sim (`overtake_inference.py:2151`)
- Measured P0 latency 43 ms (deck claims 2 ms); P1 789 ms (deck claims 500 ms)

Present-but-honest: 26 overtake labels, AUC 0.977, precision 0.30, recall 0.75 — real but weak positives; must be framed with calibration, not as-is.

---

## 2. THE PLAN — P0 / P1 / P2 / P3

### P0 — SUBMISSION BLOCKERS (must land before demo; ~1.5 days total)

**P0-1 · Fix the four live red-team bugs**
- **Why:** every one is findable by a judge in a single HTTP request.
- **Files:** `scripts/dashboard.py` (sandbox crash, ignored `chaser_battery_pct`), `scripts/overtake_inference.py` (posture-blind verdict).
- **Current behavior:** 500 on all-negative deltas; battery override stored as meta `None`; verdict independent of ERS lever.
- **Target behavior:** sandbox returns a graded error/partial payload (never 500); battery override flows into the projection and changes SOC trajectory; verdict reflects posture (SAVE posture → HOLD/SAVE, not ATTACK).
- **Approach:** guard the division (return zero-weight/fallback when `tot <= 0`); thread `chaser_battery_pct` through `simulate_live_call`'s SOC init; add posture-aware verdict gate comparing `cum` against target *and* `final_soc_pct` against reserve band.
- **Tests:** pytest unit tests — (a) all-negative-delta payload returns 200 with `status:"degraded"`; (b) 31% vs 90% battery produce different SOC trajectories; (c) verdict flips to HOLD when chaser posture = full-save and margin < uncertainty band.
- **Judge impact:** CRITICAL — removes 4 instant disqualification vectors.
- **Effort:** half day.

**P0-2 · Purge/replace unverifiable deck claims**
- **Why:** both txt audits and the live audit agree these are the top judge-killers.
- **Claims to change:**
  - "Predicted Lap 12, Actual Lap 13, ±1 lap accuracy (Monaco 2023)" → remove; replace with a *real* verified pass backtest (`backtest_race_calls.py` on 2021 races where `actual_pass == true`; Austin 2021 HAM/VER is the strongest candidate — it is in the training set at `2021-10-24 Austin` per `model_info.json`, so run it as a *held-out* replay, never as unseen data claim).
  - "<2 ms inference / <500 ms simulation" → restate as measured: "<50 ms single-pair inference; sub-second 30-lap live replay (measured P50 43 ms / 789 ms)".
  - "FIA Battery Minimums compliance" → "self-imposed 10% management reserve (0.4 MJ of 4 MJ usable), enforced as a hard feasibility constraint in simulation".
  - "-1.8s race time improvement" → "−1.88 s closing-phase delta (Balanced vs Flat-Out, Monaco 2023), reproduced deterministically by `scripts/benchmark_energy_strategy.py` and guarded by `check_energy_benchmark_drift.py`".
  - "Corner-by-corner overtake probability ML" → "lap-level overtake probability, spatially resolved into braking-zone pass-mass via telemetry deceleration signatures".
- **Files:** deck (external), `Q_and_A.txt`, `launchers/README.md`, dashboard copy.
- **Tests:** a `scripts/verify_claims.py` that greps the repo for banned strings ("±1 lap", "2 ms", "FIA Battery Minimum", "corner-by-corner ML") and exits non-zero if any appear — CI-style claim guard.
- **Judge impact:** CRITICAL — converts 3 fatal exposures into 3 honesty credits.
- **Effort:** 2 hours + writing.

**P0-3 · Connect energy state to live overtake inference**
- **Why:** both txt audits' single most important technical fix. Today the ML classifier sees a world with no battery difference; the ERS lever only bends lap pace externally.
- **File:** `scripts/overtake_inference.py:2151`.
- **Current:** `fuel_diff_kg=0.0, energy_diff_mj=0.0` hardcoded.
- **Target:** pass the projected chaser-vs-leader SOC delta in MJ each lap: `energy_diff_mj = soc_chaser_mj - soc_leader_mj` (leader synthesized via same simulator under its posture; if unavailable, fall back to `0.0` and set a `energy_feature_imputed: true` flag on the record).
- **Tests:** (a) increasing chaser ERS strictly increases predicted `overtake_probability` at fixed gap; (b) imputed fallback path produces the same output as current code (no silent behavior change when leader SOC unavailable).
- **Judge impact:** HIGH — restores the "physics-aware ML" claim; the coupling demo (ERS slider visibly bending P(pass)) is exactly the intelligence judges want to see.
- **Effort:** 2–4 hours.

### P1 — THE WINNING LAYER (~2 days; the differentiator all three audits chose)

**P1-1 · Multi-Policy Decision Engine (`scripts/policy_engine.py`, new)**
- **Why:** this is *the* core challenge deliverable. All three audits independently converged on the same spec; that convergence is the design signal.
- **Behavior:** `evaluate_tactical_policies(state) -> [policies]` — run 5 candidate policies through the existing `project_energy_trace` + live-call machinery over a 12–15 lap horizon:
  1. Greedy Attack (push now)
  2. Balanced Hold
  3. Tactical Stalk (bank in low-overtake sector, strike lap N+2)
  4. Save & Defend
  5. Undercut Prep (max pace before pit window)
- **Scoring function** (transparent, deterministic):
  `score(π) = E[ΔPosition|π]·P(pass|π) − λ_batt·max(0, reserve−SOC_post(π)) − λ_wear·ΔDegradation(π) − λ_risk·P(fail|π)·dirty_air_penalty`
  with hard constraint: prune any policy whose projected SOC touches the 0.4 MJ floor (report it as "INFEASIBLE — reserve breach", don't hide it).
- **Output shape (the ACTION card):** `{action, deploy_lap, energy_pct, window ("T11→T12"), expected_gap_s, overtake_probability, energy_cost_mj, expected_finish_delta_s, battery_margin_pct, confidence, reason}` — exactly the Phase-2 shape from the challenge brief.
- **Why deterministic, not LLM/DP:** the existing simulator is already deterministic and re-derivable (drift check proves it); a 5-policy × 15-lap rollout is ~75 simulator calls ≈ well under 200 ms; DP/MPC adds complexity without a demo payoff at this horizon. Beam search or MC rollouts are the *extension*, not the base.
- **Tests:** policy determinism (85% battery + 0.5 s pace edge → ATTACK; 14% battery → SAVE & HARVEST); FIA-reserve enforcement (no feasible policy ever projects below 0.4 MJ); latency (P99 ≤ 150 ms over 1000 iterations).
- **Judge impact:** HIGHEST — shifts category from "viewer" to "decision engine".
- **Effort:** 1 day.

**P1-2 · Counterfactual Policy Comparison Matrix (dashboard)**
- **Why:** the "killer visual". Makes P1-1 legible in 5 seconds of demo.
- **Files:** `scripts/dashboard.py` (new route `/api/strategy/policies`), `scripts/dashboard/dashboard.html` + `static/js/` (new "Tactical Options" panel).
- **Behavior:** render the 5 policies side-by-side: `Policy | P(pass) | Pass Lap | Energy Cost | SOC Floor Margin | Finish Δ | Verdict`. Highlight winner; gray out INFEASIBLE; show "why not" one-liner under each rejected policy.
- **Tests:** endpoint returns 5 rows for a valid session; invalid session → 4xx with reason; UI smoke via validate_features.py extension.
- **Judge impact:** HIGHEST — this is the demo centerpiece.
- **Effort:** half day.

**P1-3 · Battery uncertainty bounds (± band, propagated)**
- **Why:** synthesized SOC presented as point values is the #1 "magic black box" objection. Audit 2's Q&A already has the right answer — we need the code to match it.
- **Files:** `scripts/energy_simulator.py`, `scripts/overtake_inference.py`, dashboard chart.
- **Behavior:** SOC reported as `mean ± band` (band grows with laps since last trusted state; floor the band at ±2%, grow ~0.5%/lap). Propagate into strategy confidence: `confidence = f(band_width, P(pass) variance, opponent pace variance)`. Render shaded envelope on SOC chart.
- **Tests:** band monotone in laps-since-calibration; policy verdict flips when `margin < band` (the "knows what it doesn't know" demo beat).
- **Judge impact:** HIGH — this is the single sentence judges remember.
- **Effort:** half day.

**P1-4 · Isotonic calibration of overtake probability + reliability table**
- **Why:** 26 positives and precision 0.30 mean raw logistic output is not a calibrated probability. Presenting "73%" as calibrated will fail the ML judge.
- **Files:** `scripts/ml_overtake_predictions.py`, `ml_models/overtake/`, new `ml_models/overtake/calibration.json`.
- **Behavior:** fit isotonic regression on race-grouped held-out folds; commit reliability table (predicted-decile → observed-frequency); expose `calibrated_probability` alongside raw; UI labels which is shown.
- **Tests:** Brier score improves or is unchanged after calibration; reliability table committed as artifact.
- **Judge impact:** HIGH — replaces "your P is fake" with "your P is honestly calibrated and documented".
- **Effort:** half day.

**P1-5 · Opponent counter-defense posture (game-theoretic asymmetry)**
- **Why:** all three audits flagged it; it enables the demo's adversarial beat ("leader defends → engine aborts").
- **File:** `scripts/overtake_inference.py` (leader posture parameter in `simulate_live_call` and policy engine).
- **Behavior:** `leader_posture ∈ {passive, defensive_boost}`; defensive raises leader pace and reduces closing rate; policy engine re-scores and may flip verdict to HOLD/ABORT.
- **Tests:** defensive posture strictly reduces P(pass) and cumulative probability at fixed gap.
- **Judge impact:** HIGH — one toggle produces a dramatic, honest strategy flip on stage.
- **Effort:** 3–4 hours.

### P2 — TECHNICAL RIGOR POLISH (do if time remains)

**P2-1 · Latency benchmark suite (`scripts/benchmark_latency.py`, new)**
- P50/P95/P99 over ≥500 iterations: single-pair inference, full policy evaluation, full-race replay. Emit `backtests/latency_profile.json`. Live-demo-safe ("watch the profiler").
- **Tests:** assert P99 ≤ 150 ms policy eval, ≤ 50 ms single-pair. **Effort:** 2 hours.

**P2-2 · Chronological / unseen-split validation rerun**
- **Why:** audit 2's Q10. Current split is GroupShuffleSplit by session (good), but commit an artifact: train on ≤2022, test on 2023+; also unseen-track. Report MAE/R² honestly.
- **Files:** `scripts/ml_lap_predictions.py`, new `backtests/temporal_validation.json`. **Effort:** half day.

**P2-3 · Adversarial scenario suite (`scripts/adversarial_scenarios.py`, new)**
- Encode the 20 race-state scenarios from this thread's audit (safety car, sensor dropout, battery misestimate, final-lap desperation, etc.) as scripted states fed to the policy engine; assert no scenario produces an infeasible recommendation or a 500.
- **Judge impact:** turns "robustness 4/10" into a demo slide. **Effort:** half day.

**P2-4 · Split the 3,352-line dashboard.py**
- Extract: stint optimizer, advisor reasoning, benchmark passthrough into `scripts/strategy/`. Lower priority than everything above — do it only after P0/P1 are locked and tested. **Effort:** 1 day (mechanical).

### P3 — NOT NOW

- Modularize `dashboard.html` CSS/JS (cosmetic; risk of demo-breaking refactor before submission).
- Weather/wet-transition model (no data source wired; would be fake).
- LLM/voice integration (all audits: category noise).
- New ML models (tree boosters etc.) — no evidence they improve the decision, adds latency variance.

---

## 3. TEST PLAN (submission gate)

Run before recording the demo video:
1. `python scripts/validate_features.py` → 58/58 PASS (existing gate)
2. `pytest scripts/tests/` → all P0-1/P0-3/P1-1/P1-3 unit tests green
3. `python scripts/check_energy_benchmark_drift.py` → zero drift (anchor + fleet)
4. `python scripts/benchmark_latency.py` → P99 within stated claims
5. `python scripts/verify_claims.py` → no banned claim strings in repo
6. `python scripts/backtest_race_calls.py --year 2021` → precision/recall reported against `actual_pass == true` races only
7. Manual demo rehearsal: policy table renders, ERS slider flips verdict at least twice on stage, no 500s under 30 rapid clicks

---

## 4. FINAL WIN CONDITIONS (measurable, post-implementation)

1. Every recommendation on screen carries a confidence derived from propagated uncertainty, and flips when a single input crosses its band.
2. The policy table shows the engine beating flat-out/save/greedy baselines on the committed fleet sweep — a table, not a vibe.
3. Every number in the deck regenerates from a committed artifact in <5 min on a clean checkout (drift check green).
4. Overtake headline is a calibrated probability with its reliability table committed.
5. Adversarial demo beat: toggling opponent defense flips the recommendation to ABORT in <200 ms, on stage, deterministically.

---

## 5. DEMO SCRIPT (reconciled 3:00 — all beats use machinery that will exist after P0/P1)

- **0:00–0:30** Hook: "anyone can press the overtake button; the hard problem is what it costs you three laps later." Live state: VER 0.82 s behind HAM, lap 18, battery 36% ± band.
- **0:30–1:00** Policy matrix renders: Greedy Attack shows P(pass) 72% but battery hits reserve floor on lap 21 → verdict "INFEASIBLE / regret +1.1 s".
- **1:00–1:30** Engine picks TACTICAL STALK: bank 1.2 MJ in S2, strike lap 21; P rises to ~84%, battery margin healthy, finish delta −1.9 s. One sentence: "we compare futures, not predictions."
- **1:30–2:00** Adversarial beat: toggle leader to Defensive Boost → engine re-scores in <200 ms and flips to HOLD/ABORT, battery preserved. "It knows when NOT to attack."
- **2:00–2:30** Uncertainty beat: perturb battery estimate −5% → margin crosses the band → verdict flips again. "The model knows what it doesn't know."
- **2:30–3:00** Proof layer: drift-check green, latency profiler numbers on screen, honesty slide (synthetic layers labeled, 26 labels, precision 0.30, calibration table). Close: "F1 doesn't broadcast the battery. We rebuilt it, bounded it, and made decisions that admit their own error."
