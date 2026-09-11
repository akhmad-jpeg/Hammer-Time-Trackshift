# Deck ↔ Implementation Correction Sheet

Every claim below was checked against the code (Sept 2026) and, where a number
is quoted, re-measured on the running system. Fix these in the deck's source
design (Canva/Figma/PowerPoint) and re-export the PDF before judging.

---

## FIX 1 — "corner-by-corner overtake probabilities" (Page 3)

**Deck says:**
> "A real-time ML engine that transforms raw telemetry into **corner-by-corner overtake probabilities**."
> and: "03 Advise — Dynamic output for **every corner** on the subsequent lap."

**The problem.** Telemetry is stored at ~5–6 samples per lap (25,085 laps at 6
samples, 6,101 at 5 — no timestamps, no position). The "corner" view is the
model's per-lap probability redistributed over 12 lap-fraction segments,
weighted by interpolated speed drops. It is a model-shaped rendering, not a
measurement. One sharp judge question ("how many telemetry points per lap?")
collapses this claim and takes the credibility of everything else with it.

**Replace with:**
> "A real-time ML engine that turns race state into **per-lap overtake
> probabilities, projected onto where passes actually happen — braking
> zones**."

And on the "03 Advise" card:
> "Dynamic output for **every phase of the lap** on the subsequent lap."

**If a judge asks anyway**, the honest answer is strong: "Five to six samples
per lap — the corner view is model-shaped and labelled as such. Full-resolution
re-import is roadmap item one, and the pipeline is already built for it."

---

## FIX 2 — "RESPONSE TIME 284ms" / "under 300ms" (Page 4)

**Deck says:**
> "2. Energy Sandbox — Integrated interactive sliders allow strategists to
> simulate 'What-If' scenarios with backend recalculations **in under 300ms**."
> Badge: "RESPONSE TIME **284ms**"

**The problem.** The number is real — but only for the Energy Sandbox endpoint
(`/api/strategy/energy-sandbox`, measured **44 ms warm** this week). Unscoped,
it implies the whole intelligence layer answers in ~300 ms. The headline
coupled race call (both engines, ten policies) is **1.2–1.8 s fresh**, ~24 ms
cached. A judge timing the demo against this badge will conclude the deck is
dishonest even though every individual number is true.

**Replace with (scoped and still fast):**
> "2. Energy Sandbox — Interactive What-If sliders with backend recalculation
> in **under 300 ms** (sandbox). The full dual-engine race call runs in
> **~1.5 s fresh / cached instantly**, and discloses which one you're seeing."

Badge:
> "SANDBOX RESPONSE 284ms · FULL CALL ~1.5s FRESH / <50ms CACHED"

(If the badge must stay short: "284ms SANDBOX · DISCLOSED LATENCY, ALWAYS".)

---

## FIX 3 — "guaranteeing the car finishes with enough battery" (Page 2)

**Deck says (strategic question quote):**
> "What if the pit wall had a 'What-If' simulator that predicted the exact lap
> to overtake, while **guaranteeing** the car finishes with enough battery?"

**The problem.** The engine is probabilistic with hard feasibility pruning — it
*refuses* passes that drain the store below its 10% management reserve and
demotes attacks that rely on floor-draining — but nothing is guaranteed:
battery state itself is a reconstruction carrying a ±2–8% uncertainty band.
"Guarantee" is the one word in the deck the code cannot back, and it invites
the worst possible judge frame ("they promise certainty about a number they
admit is synthetic").

**Replace with (stronger, because it's true):**
> "What if the pit wall had a 'What-If' simulator that predicted the lap to
> attack — and **refused any move the battery can't pay for?**"

The refusal is the product. Say the true thing; it's more memorable.

---

## FIX 4 — UDP listener / live telemetry (Pages 3, 5)

**Deck says:**
> Page 3: "01 Ingest — **Captures live telemetry** for Leader (Car A) and
> Chaser (Car B). UDP LISTENER · FASTF1 IMPORTERS"
> Page 5 architecture diagram: "DATA INPUT — capture_telemetry.py ·
> **Live UDP Stream**" alongside "import_f1_race.py · FastF1 Historical Data"

**The problem.** The UDP game-telemetry client exists (`capture_telemetry.py`,
fully built) but has **never captured a session — zero rows in the database**.
All 571 stored sessions are FastF1 history (2020–2026). Presented as-is, the
ingestion story is half dead code; if a judge asks to see the live stream,
there is nothing to show.

**Replace Page 3 with:**
> "01 Ingest — **Seven seasons of real race history** (2020–2026) imported
> per driver, so every leader/chaser duel replays both cars' actual laps.
> FASTF1 IMPORTERS · live-game capture in development"

**Replace Page 5 "DATA INPUT" with:**
> "import_f1_race.py — FastF1 Historical Data (2020–2026, 571 sessions)
> capture_telemetry.py — live game telemetry (in development)"

Showing the roadmap honestly costs nothing; claiming a live feed you can't
demo costs the room.

---

## BONUS MISMATCHES FOUND (same audit, fix while you're in there)

1. **Page 8 still says "±1 LAP ACCURACY" and "Full compliance with FIA Battery
   Minimums."** These are the exact strings the codebase's own claims guard
   (`scripts/verify_claims.py`) now bans — the PDF is the last place they
   live. Replace with: "Projected the Monaco 2023 attack window to within one
   lap on stored races" and "self-imposed 10% management reserve maintained".
2. **Page 7: "The Energy_Remaining column is injected into feature_pipeline.py,
   enabling precise ML strategy forecasting."** Not accurate — the lap-time
   pipeline consumes no energy feature; energy enters through the overtake
   model's `energy_diff_mj` (built in `ml_overtake_predictions.py` from
   `race_state`). Rephrase: "the synthesized battery state feeds the overtake
   model's energy-difference feature and the strategy engine's feasibility
   checks."
3. **Page 7/9 "real-time"**: the system is a forward projection with cached
   re-evaluation, not a live feed. "Instant" / "on demand" is accurate and
   still impressive.
4. **Page 5 "LATENCY: < 5MS" / "ENGINE: V6-HYBRID-SIM"** decoration: the first
   is unanchored, the second is fictional flavor text. Delete or replace the
   latency chip with a real, scoped one (see Fix 2).

## Verified-true claims you can safely keep

- "Linear Regression over black-box networks" — real, and defensible.
- "Synthesizes battery state because F1 does not broadcast it" — real, with
  ±2–8% uncertainty bands disclosed in the product.
- "Simulates two cars simultaneously" — real (coupled attack + defence engines).
- "−1.8 s AI strategy vs Flat-Out (Monaco 2023, closing phase)" — reproduced
  by `benchmark_energy_strategy.py`, drift-checked; keep the closing-phase
  qualifier visible.
- "Under 300 ms" — true for the sandbox once scoped (Fix 2).

---

## SECOND-PASS AUDIT (PDF re-read against the repo, Sept 11)

A direct re-extraction of the exported PDF found more items the first pass missed.
All verified against the committed code/artifacts on this date.

5. **Page 8: "<2ms INFERENCE SPEED" chip.** This is the exact single-digit-millisecond
   phrasing the claim guard bans (`verify_claims.py` exits non-zero on it) — the first
   pass caught page 5's "<5MS" but missed this one. Also "<500ms SIMULATION SPEED / 20-lap
   full race simulation": a single-mode 20-lap projection is that fast, but the full
   multi-policy rollout is seconds-class. Replace with the scoped, real budget:
   "LIVE CALL ~200ms · POLICY ROLLOUT seconds · SANDBOX warm in tens of ms".

6. **Page 5: "MySQL utilizing schema.sql".** The file is `database/f1_strategy.sql`.
   A judge opening the repo will search for "schema.sql" and find nothing. Rename the
   chip (or actually add a schema.sql alias) — trivial but visible.

7. **Pages 5+6: "23 tracks / TRACK CIRCUITS 23 Units".** Stale. The trained lap-time
   model covers **32 tracks** (83 features, `ml_models/model_info.json`); the overtake
   model covers **29** (`ml_models/overtake/model_info.json`). Say "30+ circuits" or the
   exact counts — the committed artifacts make this a free upgrade, not a risk.

8. **Page 7: energy constants "CONSUMPTION −2%/sec @ 100% throttle" and "RECOVERY +0.5%".
   ** Not the code's physics. The engine is MJ-based: deploy/recover rate **0.12 MJ/s
   (120 kW)** against a **4 MJ store** (= 3%/s at full deploy, not 2%), recovery gated by
   a 4 MJ/lap flow cap and the 2 MJ/lap harvest limit, plus the 10% management reserve.
   Replace the two chips with: "0.12 MJ/s deploy · 4 MJ store · 10% reserve".

9. **Page 3: six mode chips (PUSH / BALANCED / LIFT & COAST / HIGH DRAIN / OPTIMUM /
   RECOVERY).** The simulator implements exactly **three** modes — push, balanced,
   lift & coast (see `benchmark_energy_strategy.py` output). Cut the extra chips or be
   ready for "what is OPTIMUM mode?" — there is no such mode in the code.

10. **Page 8: the Monaco backtest anecdote ("first 5 laps only → AI predicted Lap 12,
    actual Lap 13").** No committed artifact backs this specific story, and it is the
    source of the "±1 LAP ACCURACY" chip. Strongest honest replacement (and a better
    slide): the reliability table — "1,717 checkpoints replayed across 84 weekends;
    confidence ≥ 0.80 carried an 18% real pass rate vs the 10% base rate (2.12× at
    P ≥ 0.95)" — `scripts/backtests/race_call_reliability_slide.md` is already
    slide-shaped.

11. **Page 6: the LR-vs-RF slide has no numbers.** The choice is defensible; make it
    un-assailable by adding the committed metrics: lap-time model MAE **1.61 s**,
    R² **0.93** within-track (`ml_models/model_info.json`).

12. **Page 9: "github.com/akhmad-jpeg/Motorsports-telemetry".** RESOLVED — the actual
    public repo is **github.com/akhmad-jpeg/Hammer-Time-Trackshift** (verified live:
    the push target of this machine). Update the slide to that URL; as printed, it
    would 404 if a judge types it during Q&A. When they open the right one, the
    README's claim→receipt table is waiting for them.
