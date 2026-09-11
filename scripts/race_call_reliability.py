"""Reliability report for the live Race Call — pass-rate by confidence.

Builds the project's honesty artifact: replay stored races through the SAME
forward projection the dashboard runs (reusing backtest_race_calls' machinery
verbatim), then report what the model's confidence is actually WORTH:

  * threshold table  — when cumulative P(overtake) reached T, how often did a
                       real on-track pass follow before the next pit stop?
                       This is the empirical precision of the confidence
                       signal at each operating point, from ONE replay run
                       (each checkpoint stores its cumulative probability and
                       the real outcome, so the sweep costs nothing).
  * calibration bins — the same curve in fixed confidence buckets (including
                       the low-confidence region, so the table shows the
                       signal is monotone rather than cherry-picked).
  * window detection — precision/recall of "the pair gets inside ~1.2 s".
  * gap trajectory   — mean |error| of the forward gap walk vs the real gap.

FRAMING (the slide's title line, and the product's honest position):
    "We rank attack windows, we don't certify probabilities."
The overtake classifier is trained on a small number of positive labels, so
its raw probability is a RANKING signal.  What the backtest establishes is how
much confidence is worth in practice — the number a pit wall actually acts on.

Caveats carried from the backtest (restated in every artifact):
  * a checkpoint's scoring window ends at the NEXT pit stop by either driver
    (undercut/overcut is strategy, not on-track racecraft);
  * safety-car compression inside a window counts as real convergence;
  * the conditioning gap is the cumulative-race-clock gap, matching training.

Outputs (deterministic: same DB -> same numbers):
  backtests/race_call_reliability.json        — full report
  backtests/race_call_reliability_slide.md    — paste-ready slide markdown

Usage:
    python scripts/race_call_reliability.py                     # all races
    python scripts/race_call_reliability.py --year 2021
    python scripts/race_call_reliability.py --pair HAM,VER
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import get_db_connection
from backtest_race_calls import (
    find_race_sessions, driver_series, pit_in_laps, run_checkpoint,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Confidence operating points for the threshold table.  Ascending; the default
# live-call threshold (LIVE_PASS_CUM = 0.8) must appear in this list.
THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)

# Fixed calibration bins (same edges the backtest's console report uses).
CALIBRATION_BINS = ((0.0, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 1.01))

# Buckets with fewer scored checkpoints than this are flagged "thin" in the
# artifacts rather than silently presented as evidence.
MIN_BUCKET_N = 15


def threshold_table(checkpoints, thresholds=THRESHOLDS):
    """Empirical pass rate conditioned on cumulative confidence >= T.

    One replay run yields the whole sweep: every checkpoint carries its
    cumulative P(overtake) and whether a real pass followed, so each row is a
    re-binning of the same scored checkpoints (no re-simulation).
    """
    scored = [c for c in checkpoints if c.get("cum_probability") is not None]
    base_passes = sum(1 for c in scored if c["actual_pass"])
    base_rate = (base_passes / len(scored)) if scored else None
    rows = []
    for t in thresholds:
        grp = [c for c in scored if c["cum_probability"] >= t]
        n = len(grp)
        passes = sum(1 for c in grp if c["actual_pass"])
        rate = (passes / n) if n else None
        rows.append({
            "threshold": t,
            "checkpoints": n,
            "actual_passes": passes,
            "pass_rate": round(rate, 4) if rate is not None else None,
            # How much this confidence level beats an arbitrary in-battle
            # checkpoint (None when the base rate itself is 0).
            "lift_vs_base": (round(rate / base_rate, 2)
                             if rate is not None and base_rate else None),
            "thin": n < MIN_BUCKET_N,
        })
    return rows, (round(base_rate, 4) if base_rate is not None else None)


def calibration_bins(checkpoints, bins=CALIBRATION_BINS):
    """Fixed-bucket reliability curve, including the low-confidence region."""
    scored = [c for c in checkpoints if c.get("cum_probability") is not None]
    out = []
    for lo, hi in bins:
        grp = [c for c in scored if lo <= c["cum_probability"] < hi]
        n = len(grp)
        passes = sum(1 for c in grp if c["actual_pass"])
        out.append({
            "bin": f"[{lo:.1f},{min(hi, 1.0):.1f})",
            "checkpoints": n,
            "actual_passes": passes,
            "pass_rate": round(passes / n, 4) if n else None,
            "thin": n < MIN_BUCKET_N,
        })
    return out


def confusion(checkpoints, pred_key, act_key):
    """Confusion counts + precision/recall for a binary call vs reality."""
    tp = sum(1 for c in checkpoints if c.get(pred_key) and c.get(act_key))
    fp = sum(1 for c in checkpoints if c.get(pred_key) and not c.get(act_key))
    fn = sum(1 for c in checkpoints if not c.get(pred_key) and c.get(act_key))
    tn = sum(1 for c in checkpoints if not c.get(pred_key) and not c.get(act_key))
    prec = (tp / (tp + fp)) if (tp + fp) else None
    rec = (tp / (tp + fn)) if (tp + fn) else None
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(prec, 4) if prec is not None else None,
        "recall": round(rec, 4) if rec is not None else None,
    }


def gap_walk_error(checkpoints):
    scored = [c for c in checkpoints if c.get("gap_mae_s") is not None]
    if not scored:
        return None
    mae = sum(c["gap_mae_s"] for c in scored) / len(scored)
    bias = sum(c.get("gap_bias_s") or 0.0 for c in scored) / len(scored)
    return {"mean_mae_s_per_lap": round(mae, 3),
            "mean_bias_s_per_lap": round(bias, 3),
            "checkpoints": len(scored)}


def collect_checkpoints(conn, args):
    """The backtest's replay loop, verbatim in behavior (one pass, default
    threshold) — checkpoints are then re-binned for the whole sweep."""
    races = find_race_sessions(conn, year=args.year, track_sub=args.track)
    pair_filter = None
    if args.pair:
        pair_filter = tuple(c.strip().upper() for c in args.pair.split(","))
        if len(pair_filter) != 2:
            sys.exit("--pair needs two codes, e.g. HAM,VER")

    checkpoints = []
    segments_used = 0
    race_count = 0
    for (track, date) in sorted(races):
        drivers = races[(track, date)]
        codes = sorted(drivers)
        pairs = [(codes[i], codes[j])
                 for i in range(len(codes))
                 for j in range(i + 1, len(codes))]
        if pair_filter:
            if pair_filter not in pairs and \
                    (pair_filter[1], pair_filter[0]) not in pairs:
                continue
            pairs = [pair_filter]

        year = int(str(date)[:4]) if isinstance(date, str) else date.year
        for a_code, b_code in pairs:
            sa, ca = driver_series(conn, drivers[a_code])
            sb, cb = driver_series(conn, drivers[b_code])
            pits_a = pit_in_laps(conn, drivers[a_code])
            pits_b = pit_in_laps(conn, drivers[b_code])
            got = run_checkpoint(track, str(date), a_code, b_code,
                                 sa, sb, ca, cb, pits_a, pits_b, year,
                                 args.every, args.min_gap, args.max_gap,
                                 args.min_span, args.limit_checkpoints)
            for c in got:
                if "skip" not in c:
                    checkpoints.append(c)
            if got:
                segments_used += 1
            if args.max_segments and segments_used >= args.max_segments:
                break
        if got:
            race_count += 1
        if args.max_segments and segments_used >= args.max_segments:
            break
    return checkpoints, segments_used, race_count


def build_report(checkpoints, segments_used, race_count, args):
    years = sorted({c["year"] for c in checkpoints if c.get("year")})
    table, base_rate = threshold_table(checkpoints)
    bins = calibration_bins(checkpoints)
    win = confusion(checkpoints, "pred_window", "actual_window")
    pas = confusion(checkpoints, "pred_pass", "actual_pass")
    gap = gap_walk_error(checkpoints)
    by80 = next((r for r in table if r["threshold"] == 0.8), None)
    by50 = next((r for r in table if r["threshold"] == 0.5), None)

    return {
        "meta": {
            "generated_at": datetime.now().isoformat(),
            "script": "scripts/race_call_reliability.py",
            "reuses": "scripts/backtest_race_calls.py replay machinery",
            "projection": "overtake_inference.simulate_live_call (the live call)",
            "scope": {
                "race_weekends": race_count,
                "pair_segments": segments_used,
                "checkpoints": len(checkpoints),
                "year_span": ([years[0], years[-1]] if years else None),
                "filters": {
                    "year": args.year, "track": args.track, "pair": args.pair,
                    "checkpoint_every_laps": args.every,
                    "gap_window_s": [args.min_gap, args.max_gap],
                    "min_segment_laps": args.min_span,
                },
            },
            "framing": "We rank attack windows, we don't certify probabilities.",
        },
        "method": {
            "conditioning": ("a checkpoint is scored when the pair's real "
                             "clock-gap is inside the battle window; its "
                             "cumulative P(overtake) comes from the same "
                             "forward projection the dashboard's Race Call "
                             "runs; reality is an on-track order swap before "
                             "either car's next pit stop"),
            "base_pass_rate": base_rate,
            "base_pass_rate_meaning": ("how often a real pass followed an "
                                       "arbitrary in-battle checkpoint — the "
                                       "null the confidence lift is measured "
                                       "against"),
        },
        "threshold_table": table,
        "reading": flat_midrange_note(table),
        "calibration_bins": bins,
        "window_detection": win,
        "pass_detection_at_default_threshold": pas,
        "gap_walk": gap,
        "headline": {
            "pass_rate_at_default_0_8": by80["pass_rate"] if by80 else None,
            "checkpoints_at_0_8": by80["checkpoints"] if by80 else None,
            "pass_rate_at_0_5": by50["pass_rate"] if by50 else None,
            "checkpoints_at_0_5": by50["checkpoints"] if by50 else None,
            "default_threshold_source": "overtake_inference.LIVE_PASS_CUM",
        },
        "caveats": [
            "A checkpoint's scoring window ends at the next pit stop by "
            "either driver; undercut/overcut outcomes are out of scope.",
            "Safety-car compression inside a window counts as convergence.",
            "Gaps are cumulative-race-clock gaps (training semantics), not "
            "timing-feed track gaps.",
            "Buckets flagged thin have fewer than "
            f"{MIN_BUCKET_N} scored checkpoints.",
            "The overtake classifier's positive labels are few; these rates "
            "are the honest operating evidence, not a certification.",
        ],
    }


def flat_midrange_note(table):
    """Honest read of the sweep's shape, computed not asserted.

    When the real pass rate barely moves across the mid thresholds (0.30
    through 0.80) but climbs at the top, say exactly that: the confidence
    acts as a coarse filter, not a fine-grained dial.  Returns None when the
    curve IS monotone enough that no caveat is needed.
    """
    def rate_at(t):
        row = next((r for r in table if r["threshold"] == t), None)
        return row["pass_rate"] if row and row["pass_rate"] is not None else None
    mid_lo, mid_hi, top = rate_at(0.5), rate_at(0.8), rate_at(0.95)
    if None in (mid_lo, mid_hi, top):
        return None
    if abs(mid_hi - mid_lo) <= 0.02 and top > mid_hi + 0.01:
        top_row = next(r for r in table if r["threshold"] == 0.95)
        lift_base = top_row.get("lift_vs_base")
        lift_txt = (f"{lift_base}× the base rate" if lift_base
                    else f"{top / mid_lo:.1f}× the mid-range rate")
        return (f"Read it honestly: the rate is flat through the mid-range "
                f"({100 * mid_lo:.0f}% at P≥0.50 vs {100 * mid_hi:.0f}% at "
                f"P≥0.80) and rises only at the very top ({100 * top:.0f}% "
                f"at P≥0.95) — confidence acts as a **coarse filter**, not a "
                f"fine-grained dial ({lift_txt}).  That is what our data "
                f"supports today.")
    return None


def render_slide(report):
    """Paste-ready slide markdown: the reliability story on one screen."""
    t = report["threshold_table"]
    w = report["window_detection"]
    scope = report["meta"]["scope"]
    base = report["method"]["base_pass_rate"]
    span = scope.get("year_span")
    span_txt = (f"{span[0]}–{span[1]}" if span else "all stored seasons")

    def pct(v):
        return f"{100 * v:.0f}%" if isinstance(v, (int, float)) else "—"

    rows = []
    for r in t:
        if r["checkpoints"] == 0:
            continue
        flag = " *" if r["thin"] else ""
        rows.append(
            f"| P ≥ {r['threshold']:.2f} | {r['checkpoints']}{flag} | "
            f"{r['actual_passes']} | **{pct(r['pass_rate'])}** | "
            f"{r['lift_vs_base']}× |")

    lines = [
        "# RELIABILITY: WHAT OUR CONFIDENCE IS WORTH",
        "",
        "**We rank attack windows. We don't certify probabilities.**",
        "",
        f"Every stored race was replayed through the *same* forward "
        f"projection the Race Call runs ({scope['checkpoints']} checkpoints, "
        f"{scope['pair_segments']} battle segments, "
        f"{scope['race_weekends']} race weekends, {span_txt}).",
        "When the model's cumulative confidence reached each level, we scored "
        "whether a real on-track pass followed before the next stop:",
        "",
        "| Model confidence | Checkpoints | Real passes | Real pass rate | Lift vs base |",
        "|---|---|---|---|---|",
        *rows,
        "",
        f"Base rate — an arbitrary in-battle checkpoint — is "
        f"**{pct(base)}**.  Confidence at the default call threshold "
        f"(P ≥ 0.80) is worth **{pct((t[5]['pass_rate'] if len(t) > 5 else None))}** "
        f"— the operating evidence behind the product's threshold, "
        f"published rather than asserted.",
        "",
        *([flat_midrange_note(t)] if flat_midrange_note(t) else []),
        "",
        f"**Attack-window detection:** precision {pct(w['precision'])} · "
        f"recall {pct(w['recall'])} "
        f"(TP {w['tp']} / FP {w['fp']} / FN {w['fn']}).",
        "",
        "Honest by construction:",
        "- windows end at the next pit stop (pit-strategy passes out of scope);",
        "- small-label reality stated up front — the table IS the calibration;",
        "- deterministic: same races → same table (regenerate any time).",
        "",
        "\\* fewer than 15 scored checkpoints — shown, not hidden.",
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(
        description="Pass-rate-by-confidence reliability report for the "
                    "live Race Call (replays stored races).")
    ap.add_argument("--year", type=int, default=None)
    ap.add_argument("--track", default=None)
    ap.add_argument("--pair", default=None)
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--min-gap", type=float, default=0.4)
    ap.add_argument("--max-gap", type=float, default=8.0)
    ap.add_argument("--min-span", type=int, default=5)
    ap.add_argument("--limit-checkpoints", type=int, default=0)
    ap.add_argument("--max-segments", type=int, default=0)
    ap.add_argument("--out", default="backtests/race_call_reliability.json")
    ap.add_argument("--slide", default="backtests/race_call_reliability_slide.md")
    ap.add_argument("--from-json", default=None,
                    help="re-render the slide from a saved report instead "
                         "of replaying races (no DB access)")
    args = ap.parse_args()

    if args.from_json:
        report = json.loads(Path(args.from_json).read_text(encoding="utf-8"))
        slide = Path(args.slide)
        slide.parent.mkdir(parents=True, exist_ok=True)
        slide.write_text(render_slide(report), encoding="utf-8")
        print(f"[SAVED] {slide} (re-rendered from {args.from_json})")
        return

    conn = get_db_connection()
    try:
        checkpoints, segments_used, race_count = collect_checkpoints(conn, args)
    finally:
        conn.close()

    if not checkpoints:
        print("No scored checkpoints — loosen --max-gap/--every or check "
              "model coverage.")
        sys.exit(1)

    report = build_report(checkpoints, segments_used, race_count, args)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")

    slide = Path(args.slide)
    slide.parent.mkdir(parents=True, exist_ok=True)
    slide.write_text(render_slide(report), encoding="utf-8")

    # Console summary = the slide table (same numbers, no re-computation).
    print("=" * 72)
    print("RACE CALL RELIABILITY — pass rate by cumulative confidence")
    print("=" * 72)
    print(f"checkpoints {len(checkpoints)} · segments {segments_used} · "
          f"weekends {race_count} · base pass rate "
          f"{report['method']['base_pass_rate']}")
    print(f"{'P >= T':>8} {'n':>6} {'passes':>7} {'rate':>8} {'lift':>6}  note")
    for r in report["threshold_table"]:
        note = "thin" if r["thin"] else ""
        rate = f"{100 * r['pass_rate']:.1f}%" if r["pass_rate"] is not None else "-"
        lift = f"{r['lift_vs_base']}x" if r["lift_vs_base"] is not None else "-"
        print(f"{r['threshold']:>8.2f} {r['checkpoints']:>6} "
              f"{r['actual_passes']:>7} {rate:>8} {lift:>6}  {note}")
    w = report["window_detection"]
    print(f"\nWindow detection: TP {w['tp']} FP {w['fp']} FN {w['fn']} "
          f"TN {w['tn']}  precision {w['precision']} recall {w['recall']}")
    g = report["gap_walk"]
    if g:
        print(f"Gap walk: MAE {g['mean_mae_s_per_lap']}s/lap "
              f"(bias {g['mean_bias_s_per_lap']:+}s/lap)")
    print(f"\n[SAVED] {out}")
    print(f"[SAVED] {slide}")


if __name__ == "__main__":
    main()
