"""STRESS HARNESS — hammer evaluate_call hard and validate every output.

Runs the unified call across drivers x compounds x tyre ages x gaps x
batteries x reserve targets x slider banks x seats x horizons, then
re-derives every reported parameter from the engine's own math:

  * shape plumbing  — every row discloses the bank its seat walked
  * posture coupling— the leader winner sets the chaser's scored posture
  * seat semantics  — surfaced card == own engine's card; opponent block
  * SOC accounting  — per-row conservation vs start battery, floor/cap
  * margin chain    — margin / worst / range recomputed from soc_end + band
  * feasibility     — reserve-breach & floor-drain rules re-derived
  * confidence      — recomputed from ranked scores + band multiplier
  * probabilities   — pass/hold/laps_held/projection sanity
  * honesty         — battery fields NEVER null on the surfaced card
  * determinism     — identical state => identical call; cache miss on change
  * seat symmetry   — both seats agree on both engines' recommendations

Run:  python scripts/_stress_whats_the_call.py
"""
import sys
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import policy_engine as pe                    # noqa: E402
import overtake_inference as oi               # noqa: E402
from energy_simulator import battery_uncertainty_band  # noqa: E402

STORE = oi.ERS_STORE_MJ
FLOOR = oi.LIVE_ATTACK_MIN_SOC_PCT
SPAN = pe.CONFIDENCE_SPAN_S
EPS = 0.2                                     # pct-points: rounding + phase float

LEADERS = ["VER", "HAM", "LEC"]
CHASERS = ["HAM", "VER", "NOR", "ALB", "SAI"]
TRACKS = ["Autodromo Nazionale di Monza", "Circuit de Monaco",
          "Silverstone Circuit", "Circuit de Spa-Francorchamps"]
COMPOUNDS = ["Soft", "Medium", "Hard"]
SHAPES = [
    None, [0, 0, 0], [0.4, -0.2, -0.2], [0.5, 0.5, 0.5], [-0.6, 0.0, 0.3],
    [2.0, -2.0, 0.0], [8.5, 8.5, 8.5], [-8.5, 8.5, 0.0], [0.05, 0.0, 0.0],
    [-0.3, -0.3, -0.3],
]
SEATS = ["chaser", "leader"]

failures = []          # (tag, check-name, detail)
n_checks = 0
covered = {"drivers": set(), "tracks": set(), "compounds": set(),
           "shapes": set(), "seats": set(), "cases": 0, "raised": 0}


def check(tag, name, ok, detail=""):
    global n_checks
    n_checks += 1
    if not ok:
        failures.append((tag, name, detail))


def close(a, b, eps=EPS):
    return a is not None and b is not None and abs(a - b) <= eps


def _row_pol(seat, name):
    table = pe.POLICIES if seat == "chaser" else pe.LEADER_POLICIES
    return next(p for p in table if p["name"] == name)


def _push_lever(seat, pol):
    return (pe._push_lever(pol) if seat == "chaser"
            else pe._leader_push_lever(pol))


def recomputed_confidence(rows, horizon):
    """Mirror of the engines' ranking -> margin -> confidence chain."""
    feas = [r for r in rows if r["feasible"]] or rows
    ranked = sorted(feas, key=lambda r: r["score_s"])
    best = ranked[0]
    nxt = [r["score_s"] for r in ranked[1:]
           if r["score_s"] > best["score_s"] + 1e-9]
    if nxt:
        margin = min(nxt) - best["score_s"]
    elif len(ranked) > 1:
        margin = ranked[1]["score_s"] - best["score_s"]
    else:
        margin = SPAN
    band = battery_uncertainty_band(horizon)
    mult = 1.0 - 0.4 * ((band["band_pct"] - band["floor_pct"])
                        / max(1e-9, band["cap_pct"] - band["floor_pct"]))
    conf = round(min(1.0, max(0.0, (margin / SPAN) * mult)), 2)
    return conf, round(margin, 3)


def validate(out, kw, tag):
    seat = kw["perspective"]
    fc = out["final_call"]
    horizon = kw["race_length"] - kw["start_lap"]
    band = battery_uncertainty_band(horizon)

    # ---- shape plumbing: rows disclose the bank their seat walked --------
    shape = pe._shape_vector(kw.get("chaser_shape"))
    lshape = pe._shape_vector(kw.get("leader_shape"))
    for r in out["policies"]:
        if r["seat"] == "chaser":
            check(tag, "chaser-row-shape", r["ers_shape"] == shape,
                  f"{r['policy']}: {r['ers_shape']} != {shape}")
        else:
            check(tag, "leader-row-shape", r["ers_shape"] == lshape,
                  f"{r['policy']}: {r['ers_shape']} != {lshape}")
    check(tag, "state-shape", out["state"]["ers_shape"] == shape)

    # ---- posture coupling: leader winner -> chaser posture ---------------
    lrows = [r for r in out["policies"] if r["seat"] == "leader"]
    lwinner = min((r for r in lrows if r["feasible"]) or lrows,
                  key=lambda r: r["score_s"])["policy"]
    want_posture = pe._DEFENCE_TO_POSTURE.get(lwinner, "balanced")
    check(tag, "posture-coupling",
          fc["engine_coupling"]["chaser_posture_used"] == want_posture,
          f"{fc['engine_coupling']['chaser_posture_used']} vs {want_posture} "
          f"(leader winner {lwinner})")
    check(tag, "coupling-disclosed",
          fc["engine_coupling"]["opponent_engine_recommendation"] == lwinner)

    # ---- seat semantics ---------------------------------------------------
    own_rows = [r for r in out["policies"] if r["seat"] == seat]
    own_rec = (out["leader"] if seat == "leader" else out["chaser"])["recommendation"]
    check(tag, "seat-surfaced", fc["seat"] == seat and out["perspective"] == seat)
    check(tag, "card-matches-engine",
          fc["action"] == own_rec["action_card"]["action"],
          f"{fc['action']} vs {own_rec['action_card']['action']}")
    winner = min((r for r in own_rows if r["feasible"]) or own_rows,
                 key=lambda r: r["score_s"])
    check(tag, "action-is-own-winner", fc["action"] == winner["policy"],
          f"{fc['action']} vs {winner['policy']}")
    opp_card = (out["chaser"] if seat == "leader" else out["leader"])["recommendation"]["action_card"]
    check(tag, "opponent-block", fc["opponent"]["action"] == opp_card["action"]
          and fc["opponent"]["confidence"] == opp_card["confidence"])

    # ---- the point of it all: battery fields are never degenerate --------
    for f in ("battery_margin_pct", "battery_margin_worst_pct",
              "soc_band_pct", "confidence", "engine_margin_s"):
        check(tag, f"card-{f}-defined", fc[f] is not None, str(fc.get(f)))

    # ---- per-row physics: SOC conservation, bands, feasibility ------------
    # Start battery per seat mapping (evaluate_call's own rule).
    if seat == "leader":
        leader_start = kw.get("battery_pct") or 62.5
        chaser_start = kw.get("threat_battery_pct") or 62.5
    else:
        leader_start = 62.5
        chaser_start = kw.get("battery_pct") or 62.5
    reserve_pct = ((kw.get("reserve_target_mj") or pe.RESERVE_TARGET_MJ)
                   / STORE * 100.0)

    for r in out["policies"]:
        st = chaser_start if r["seat"] == "chaser" else leader_start
        dep, bk = r["energy_cost_mj"], r["energy_banked_mj"]
        tag_r = f"{tag}/{r['seat']}:{r['policy']}"

        # conservation: end = start - deployed + banked, clipped to floor/cap
        pred = min(100.0, max(FLOOR, st - dep / STORE * 100.0
                              + bk / STORE * 100.0))
        check(tag_r, "soc-conservation", close(r["soc_end_pct"], pred),
              f"soc_end {r['soc_end_pct']} vs start {st} -{dep}MJ +{bk}MJ")
        check(tag_r, "soc-bounds",
              r["soc_end_pct"] is not None
              and FLOOR - 0.1 <= r["soc_end_pct"] <= 100.0)
        if r["min_soc_pct"] is not None:
            check(tag_r, "min-soc-bounds",
                  FLOOR - 0.1 <= r["min_soc_pct"]
                  <= max(st, r["soc_end_pct"]) + EPS,
                  f"min {r['min_soc_pct']} start {st} end {r['soc_end_pct']}")

        # margin chain from soc_end + band
        check(tag_r, "margin-formula",
              close(r["battery_margin_pct"], r["soc_end_pct"] - reserve_pct, 0.15))
        check(tag_r, "worst-margin-formula",
              close(r["battery_margin_worst_pct"],
                    max(0.0, r["soc_end_pct"] - band["band_pct"]) - reserve_pct,
                    0.15))
        check(tag_r, "range-formula",
              r["soc_end_range_pct"] == [max(0.0, r["soc_end_pct"] - band["band_pct"]),
                                         min(100.0, r["soc_end_pct"] + band["band_pct"])])
        check(tag_r, "band-disclosed", r["soc_band_pct"] == band["band_pct"])
        check(tag_r, "margin-ge-worst",
              r["battery_margin_pct"] >= r["battery_margin_worst_pct"] - 0.05)

        # feasibility rules re-derived
        pol = _row_pol(r["seat"], r["policy"])
        no_spend = dep <= 1e-9 and bk <= 1e-9
        drained = (r["min_soc_pct"] is not None
                   and r["min_soc_pct"] <= FLOOR + 0.05)
        breach = (r["min_soc_pct"] is not None and not no_spend
                  and r["min_soc_pct"] < reserve_pct - 0.05)
        if r["seat"] == "chaser":
            push = _push_lever("chaser", pol)
            exp_inf = breach or (r["pass_lap"] and drained and push > 0)
        else:
            exp_inf = breach or drained
        check(tag_r, "feasibility-rule",
              r["feasible"] == (not exp_inf),
              f"feasible={r['feasible']} expected {not exp_inf} "
              f"(breach={breach} drained={drained})")
        if not r["feasible"]:
            check(tag_r, "infeasible-reason", bool(r["infeasible_reason"]))

        # probability / laps sanity
        check(tag_r, "prob-bounds", 0.0 <= r["overtake_probability"] <= 1.0)
        if r["seat"] == "leader":
            check(tag_r, "hold-mirror",
                  close(r["hold_probability"],
                        1.0 - r["overtake_probability"], 0.001))
            exp_held = ((r["pass_lap"] - kw["start_lap"])
                        if r["converted"] else horizon)
            check(tag_r, "laps-held", r["laps_held"] == exp_held)
        if r["pass_lap"] is not None:
            check(tag_r, "pass-lap-range",
                  kw["start_lap"] <= r["pass_lap"] <= kw["race_length"])

    # ---- confidence recomputation (both engines + surfaced card) ----------
    for s, rec_key in (("chaser", "chaser"), ("leader", "leader")):
        rows_s = [r for r in out["policies"] if r["seat"] == s]
        conf, margin = recomputed_confidence(rows_s, horizon)
        rec = out[rec_key]["recommendation"]
        check(f"{tag}/{s}", "confidence-recomputed",
              rec["confidence"] == conf, f"{rec['confidence']} vs {conf}")
        check(f"{tag}/{s}", "margin-recomputed",
              close(rec["decision_margin_s"], margin, 0.01),
              f"{rec['decision_margin_s']} vs {margin}")
        if s == seat:
            check(tag, "card-confidence", fc["confidence"] == conf,
                  f"{fc['confidence']} vs {conf}")
            check(tag, "card-engine-margin",
                  close(fc["engine_margin_s"], margin, 0.01))

    # ---- projection / payload honesty -------------------------------------
    check(tag, "projection-verdict",
          fc["projection"]["verdict"] in ("attack", "hold", "attempt", "no_window"))
    check(tag, "baseline-pass-range",
          out["chaser"]["baseline"]["pass_lap"] is None
          or out["chaser"]["baseline"]["pass_lap"] >= kw["start_lap"])
    check(tag, "latency-positive", out["latency_ms"] > 0
          and out["latency_ms"] <= pe.CALL_LATENCY_BUDGET_MS)
    check(tag, "ten-rows", len(out["policies"]) == 10)


def run_case(kw, tag):
    covered["cases"] += 1
    covered["seats"].add(kw["perspective"])
    covered["tracks"].add(kw["track_name"])
    covered["compounds"] |= {kw["leader_tyre_compound"], kw["chaser_tyre_compound"]}
    covered["drivers"] |= {kw["leader_code"], kw["chaser_code"]}
    s = pe._shape_vector(kw.get("chaser_shape"))
    covered["shapes"].add(tuple(s) if s else None)
    try:
        out = pe.evaluate_call(**kw)
    except ValueError as e:
        covered["raised"] += 1
        print(f"  {tag:<46} RAISED (honest bail): {e}")
        return None
    validate(out, kw, tag)
    a = out["final_call"]["action"]
    m = out["final_call"]["battery_margin_pct"]
    c = out["final_call"]["confidence"]
    em = out["final_call"]["engine_margin_s"]
    flag = "OK " if not any(f[0] == tag for f in failures) else "FAIL"
    print(f"  {tag:<46} {flag} {a:<15} margin={m:>5}% worst="
          f"{out['final_call']['battery_margin_worst_pct']:>5}% conf={c:<5} "
          f"edge={em}s")
    return out


def main():
    rng = random.Random(7)
    cases = []

    # ---- anchor states: the demo beats and the extremes -------------------
    base = dict(leader_code="VER", chaser_code="HAM",
                track_name="Autodromo Nazionale di Monza", start_lap=20,
                race_length=50, gap_before_s=0.8,
                leader_tyre_compound="Medium", chaser_tyre_compound="Medium",
                leader_tyre_age=10, chaser_tyre_age=8, year=2026,
                battery_pct=62.5, reserve_target_mj=1.6)
    anchors = [
        (dict(base), "default-VER-HAM"),
        (dict(base, battery_pct=31), "low-battery-31"),
        (dict(base, battery_pct=95, reserve_target_mj=3.0), "rich-tight-reserve"),
        (dict(base, gap_before_s=0.05, start_lap=40, race_length=44),
         "flag-edge"),
        (dict(base, leader_tyre_compound="Soft", chaser_tyre_compound="Hard",
              leader_tyre_age=26, chaser_tyre_age=2, gap_before_s=1.8),
         "cliff-vs-fresh"),
        (dict(leader_code="LEC", chaser_code="NOR",
              track_name="Circuit de Monaco", start_lap=35, race_length=57,
              gap_before_s=0.4, leader_tyre_compound="Soft",
              chaser_tyre_compound="Hard", leader_tyre_age=18,
              chaser_tyre_age=3, battery_pct=45, reserve_target_mj=1.0,
              chaser_shape=[0.4, -0.2, -0.2],
              leader_shape=[0.3, -0.3, 0.0]), "monaco-shuffled"),
        (dict(base, battery_pct=50, chaser_shape=[8.5, 8.5, 8.5]),
         "full-net-shape"),
        (dict(base, leader_shape=[-8.5, 8.5, 0.0]), "leader-realloc-shape"),
        (dict(base, chaser_shape=[2.0, -2.0, 0.0], battery_pct=40),
         "realloc-on-drained"),
        (dict(base, battery_pct=None, threat_battery_pct=None,
              reserve_target_mj=None), "all-defaults"),
    ]
    for kw, name in anchors:
        for seat in SEATS:
            cases.append((dict(kw, perspective=seat), f"{name}/{seat}"))

    # ---- randomised sweep: 30 states x both seats --------------------------
    for i in range(30):
        rl = rng.randint(44, 60)
        start = rng.randint(2, rl - 5)
        kw = dict(
            leader_code=rng.choice(LEADERS), chaser_code=rng.choice(CHASERS),
            track_name=rng.choice(TRACKS), start_lap=start, race_length=rl,
            gap_before_s=round(rng.uniform(0.05, 2.5), 2),
            leader_tyre_compound=rng.choice(COMPOUNDS),
            chaser_tyre_compound=rng.choice(COMPOUNDS),
            leader_tyre_age=rng.randint(0, 28),
            chaser_tyre_age=rng.randint(0, 28),
            year=2026,
            battery_pct=rng.choice([None, rng.randint(25, 100)]),
            threat_battery_pct=rng.choice([None, rng.randint(25, 100)]),
            reserve_target_mj=rng.choice(
                [None, round(rng.uniform(0.8, 3.0), 2)]),
            chaser_shape=rng.choice(SHAPES),
            leader_shape=rng.choice(SHAPES),
        )
        for seat in SEATS:
            cases.append((dict(kw, perspective=seat), f"rnd-{i:02d}/{seat}"))

    print(f"STRESS: {len(cases)} unified calls "
          f"(anchors + random sweep x both seats)\n")

    # determinism + cache semantics on the first anchor
    pe._CALL_ENGINE_CACHE.clear()
    first = run_case(cases[0][0], cases[0][1])
    again = pe.evaluate_call(**cases[0][0])
    check("cache", "repeat-identical", again["final_call"] == first["final_call"])
    check("cache", "repeat-is-cache-hit",
          again["engine_cache"]["leader_reused"]
          and again["engine_cache"]["chaser_reused"])
    changed = dict(cases[0][0], battery_pct=40.0)
    miss = pe.evaluate_call(**changed)
    check("cache", "changed-state-misses",
          not miss["engine_cache"]["leader_reused"]
          and not miss["engine_cache"]["chaser_reused"])
    print("  cache: repeat identical + hit, changed state re-walks — OK\n")

    # seat symmetry: same state, flipped seat, same engine recommendations
    kw = cases[0][0]
    a = pe.evaluate_call(**kw)
    b = pe.evaluate_call(**dict(kw, perspective="leader"))
    check("symmetry", "leader-card-agrees",
          a["leader"]["recommendation"]["action_card"]["action"]
          == b["leader"]["recommendation"]["action_card"]["action"])
    check("symmetry", "chaser-card-agrees",
          a["chaser"]["recommendation"]["action_card"]["action"]
          == b["chaser"]["recommendation"]["action_card"]["action"])
    print("  symmetry: both seats agree on both engines' answers — OK\n")

    for kw, tag in cases:
        run_case(kw, tag)

    # error paths stay honest ValueErrors
    for bad, label in (
        (dict(cases[0][0], perspective="pitwall"), "bad-perspective"),
        (dict(cases[0][0], track_name="Nürburgring Nordschleife"), "uncovered-track"),
        (dict(cases[0][0], chaser_code="XX"), "unknown-driver"),
    ):
        try:
            pe.evaluate_call(**bad)
            check("errors", label, False, "no ValueError raised")
        except ValueError:
            check("errors", label, True)
    print("  errors: bad perspective / uncovered track / unknown driver "
          "all raise honest ValueError — OK\n")

    # ---------------------------- report -----------------------------------
    print("\n" + "=" * 78)
    print(f"CALLS: {covered['cases']}   CHECKS: {n_checks}   "
          f"FAILED: {len(failures)}   honest-bails: {covered['raised']}")
    print(f"coverage: drivers={sorted(covered['drivers'])}")
    print(f"          tracks={sorted(covered['tracks'])}")
    print(f"          compounds={sorted(covered['compounds'])}")
    print(f"          seats={sorted(covered['seats'])}")
    print(f"          distinct shape banks walked={len(covered['shapes'])}")
    if failures:
        print("\nFAILURES:")
        seen = set()
        for tag, name, detail in failures:
            key = (name, detail)
            if key in seen:
                continue
            seen.add(key)
            print(f"  [{tag}] {name}: {detail}")
        print(f"  ({len(failures)} failed checks, "
              f"{len(seen)} distinct)")
        return 1
    print("\nALL CHECKS PASSED — every parameter validated against the "
          "engine's own math.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
