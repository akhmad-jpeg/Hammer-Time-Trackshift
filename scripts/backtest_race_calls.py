"""Backtest the live Race Call against races that actually happened.

The whole point of a race-call model is that you can replay history and ask
"would the call have been right?".  This script does exactly that, using the
same forward projection the dashboard runs (overtake_inference.simulate_live_call):
for every eligible stored race it takes pairs of drivers with real race
sessions, drops checkpoints through the race at regular intervals (the cars'
stored tyre state and true gap at that lap), replays the call to the pair's
next pit stop, and scores the prediction against what the real race did:

  * WINDOW  — did the model say the pair would get inside ~1.2 s before the
              next pit stop, and did the real gap actually get that close?
  * PASS    — did the model project a clean overtake, and did the order
              actually swap on track (pit stops excluded)?
  * GAP MAE — how far the model's forward gap walk drifts from the real gap
              trajectory (this is the check on whether pace-model closing is
              realistic at all).
  * CALIBRATION — buckets of the model's cumulative P(overtake) vs how often
              a real pass actually followed.

Caveats baked in:
  * The comparison window ends at the next pit stop by either driver — an
    undercut / overcut is strategy, and this model only claims on-track
    racecraft.  Passes via pit stops are therefore not scored as misses.
  * The model runs tyres to the window end (no stop model), exactly like the
    live Race Call does.
  * Actual passes are detected as order swaps across a single RACING lap
    (pit in/out laps removed) from a small gap, mirroring the trainer's
    label rule (OVERTAKE_GAP_MAX_S = 5.0).
  * Safety-car gaps are not specially filtered; with the pit-stop horizon most
    SC compression lands inside a single segment and is a real convergence.

Usage examples:
    python scripts/backtest_race_calls.py                      # all races
    python scripts/backtest_race_calls.py --year 2021
    python scripts/backtest_race_calls.py --pair HAM,VER --every 2
    python scripts/backtest_race_calls.py --max-gap 6 --out backtests/out.json
    python scripts/backtest_race_calls.py --pass-cum 0.8   # higher cumulative
                                                           # P(overtake) target
                                                           # before calling a pass

Results print to stdout and optionally to --out as JSON.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import get_db_connection
from overtake_inference import load_session_race_laps, simulate_live_call

WINDOW_S = 1.2          # must match LIVE_WINDOW_S used by the live call
PASS_GAP_MAX_S = 5.0    # trainer OVERTAKE_GAP_MAX_S
PIT_MIN_S = 15.0        # trainer: plausible pit stop (or unknown) counts


# ---------------------------------------------------------------------------
# Race discovery
# ---------------------------------------------------------------------------
def find_race_sessions(conn, year=None, track_sub=None):
    """Rows for per-driver Race sessions with >= 25 timed laps.

    A 'race' here is a (track_name, date) with two or more per-driver
    sessions — the schema stores one session per driver per race.
    """
    q = """
        SELECT s.session_id, d.driver_code, s.track_name,
               DATE(s.date) AS race_date
        FROM sessions s
        JOIN drivers d ON s.driver_id = d.driver_id
        WHERE s.session_type = 'Race'
          AND (SELECT COUNT(*) FROM laps l
               WHERE l.session_id = s.session_id
                 AND l.lap_time_ms > 0) >= 25
    """
    params = []
    if year:
        q += " AND YEAR(s.date) = %s"
        params.append(int(year))
    cur = conn.cursor(dictionary=True)
    cur.execute(q, params)
    rows = cur.fetchall()
    cur.close()

    races = defaultdict(dict)   # (track, date) -> {code: session_id}
    for r in rows:
        if track_sub and track_sub.lower() not in str(r['track_name']).lower():
            continue
        key = (str(r['track_name']).strip(), str(r['race_date']))
        races[key][str(r['driver_code']).upper()] = int(r['session_id'])
    return {k: v for k, v in races.items() if len(v) >= 2}


def pit_in_laps(conn, session_id):
    """Lap numbers where this driver made a plausible stop."""
    cur = conn.cursor()
    cur.execute("""
        SELECT l.lap_number
        FROM strategy_events se
        JOIN laps l ON l.lap_id = se.lap_id
        WHERE se.event_type = 'PitStop'
          AND l.session_id = %s
          AND (se.duration_sec IS NULL OR se.duration_sec >= %s)
    """, (session_id, PIT_MIN_S))
    laps = {int(r[0]) for r in cur.fetchall()}
    cur.close()
    return laps


def driver_series(conn, session_id):
    """(per-lap dict by lap_number, cumulative race clock) for a session."""
    laps = load_session_race_laps(session_id, conn=conn)
    by_lap = {}
    cum = {}
    total = 0.0
    for l in laps:
        n = int(l['lap_number'])
        by_lap[n] = l
        total += float(l['lap_time_s'] or 0.0)
        cum[n] = total
    return by_lap, cum


def gap_at(cum_a, cum_b, lap):
    """Real gap entering `lap` = |cum A - cum B| after lap-1."""
    pa, pb = cum_a.get(lap - 1), cum_b.get(lap - 1)
    if pa is None or pb is None:
        return None
    return abs(pa - pb)


# ---------------------------------------------------------------------------
# Ground truth over one racing segment (start_lap, end_lap]
# ---------------------------------------------------------------------------
def analyse_segment(sa, sb, ca, cb, pits_a, pits_b, start_lap, end_lap):
    """Actual window / pass facts across racing laps (start_lap, end_lap].

    end_lap is the lap BEFORE either driver's next pit stop.  A racing lap is
    one where neither driver is pitting in or out.  Returns None when there
    is too little clean racing to score.
    """
    # Racing-lap gap series (gap entering the lap).
    racing = []
    for L in range(start_lap + 1, end_lap + 1):
        if L not in sa or L not in sb:
            continue
        if L in pits_a or L in pits_b:
            continue
        if (L - 1) in pits_a or (L - 1) in pits_b:
            continue
        g = gap_at(ca, cb, L)
        if g is not None:
            racing.append((L, g))
    if len(racing) < 2:
        return None

    window_lap = next((L for L, g in racing if g <= WINDOW_S), None)
    min_gap = min(g for _, g in racing)
    min_gap_lap = min(racing, key=lambda x: x[1])[0]

    # Pass = order (cumulative clocks) flips across a single racing lap that
    # started from a small gap (same rule as the overtake trainer).
    pass_lap = None
    for L, g in racing:
        if g > PASS_GAP_MAX_S:
            continue
        d_before = (ca[L - 1] - cb[L - 1])
        d_after = (ca[L] - cb[L])
        if d_before * d_after < 0:
            pass_lap = L
            break
    return {
        "racing_laps": len(racing),
        "actual_window": window_lap is not None,
        "actual_window_lap": window_lap,
        "actual_min_gap_s": round(min_gap, 3),
        "actual_min_gap_lap": min_gap_lap,
        "actual_pass": pass_lap is not None,
        "actual_pass_lap": pass_lap,
    }


# ---------------------------------------------------------------------------
# One (race, pair) -> checkpoints
# ---------------------------------------------------------------------------
def run_checkpoint(track, date, a_code, b_code, sa, sb, ca, cb,
                   pits_a, pits_b, year, every, min_gap, max_gap,
                   min_span, limit_checkpoints, pass_cum=None):
    """Score every `every`-th lap of one pairing's shared racing laps."""
    common = sorted(set(sa) & set(sb))
    if len(common) < 8:
        return []
    out = []
    n_checked = 0
    pits_both = pits_a | pits_b
    all_pits = sorted(pits_both)

    for L0 in common:
        if L0 < 5:
            continue
        if L0 % every:
            continue   # every Nth lap gets a checkpoint
        later_pits = [p for p in all_pits if p > L0]
        horizon_excl = min(later_pits) if later_pits else max(common)
        end_lap = horizon_excl - 1            # last lap before the stop
        if end_lap - L0 < min_span:
            continue

        gap0 = gap_at(ca, cb, L0)
        if gap0 is None or not (min_gap <= gap0 <= max_gap):
            continue
        l0_a, l0_b = sa.get(L0), sb.get(L0)
        if not l0_a or not l0_b:
            continue
        if not l0_a.get('tyre_compound') or not l0_b.get('tyre_compound'):
            continue

        # Leader / chaser by the real order entering lap L0.
        if ca[L0 - 1] < cb[L0 - 1]:
            lead_code, chase_code = a_code, b_code
            lead_ser, chase_ser, lead_cum, chase_cum = sa, sb, ca, cb
        else:
            lead_code, chase_code = b_code, a_code
            lead_ser, chase_ser, lead_cum, chase_cum = sb, sa, cb, ca
        ll, cl = lead_ser[L0], chase_ser[L0]
        l_comp, c_comp = ll['tyre_compound'], cl['tyre_compound']
        l_age, c_age = float(ll['tyre_age'] or 0.0), \
            float(cl['tyre_age'] or 0.0)

        actuals = analyse_segment(sa, sb, ca, cb, pits_a, pits_b, L0, end_lap)
        if actuals is None:
            continue

        try:
            sim = simulate_live_call(
                leader_code=lead_code, chaser_code=chase_code,
                track_name=track, start_lap=L0, race_length=end_lap,
                gap_before_s=round(gap0, 3),
                leader_tyre_compound=l_comp, chaser_tyre_compound=c_comp,
                leader_tyre_age=l_age, chaser_tyre_age=c_age,
                year=year, pass_cum=pass_cum,
            )
        except Exception as exc:
            out.append({
                "checkpoint_lap": L0, "skip": "model",
                "skip_detail": str(exc)[:160],
            })
            n_checked += 1
            if limit_checkpoints and n_checked >= limit_checkpoints:
                break
            continue

        call, summary, _meta = sim['call'], sim['summary'], sim['meta']
        pred_pass_lap = call.get('pass_lap') \
            if call.get('verdict') == 'attack' else None

        # Gap-trajectory error: model vs real, lap by lap (pre-pass only).
        mae_s = bias_s = 0.0
        n = 0
        for lrec in sim['laps']:
            L = lrec['lap']
            if L <= L0 or L > end_lap:
                continue
            real = gap_at(ca, cb, L)
            if real is None:
                continue
            err = lrec['gap_before_s'] - real
            mae_s += abs(err)
            bias_s += err
            n += 1

        out.append({
            "race_track": track,
            "race_date": date,
            "year": year,
            "leader": lead_code,
            "chaser": chase_code,
            "checkpoint_lap": L0,
            "horizon_lap": end_lap,
            "gap0_s": round(gap0, 3),
            "leader_tyre": f"{l_comp} {l_age:.0f}",
            "chaser_tyre": f"{c_comp} {c_age:.0f}",
            **actuals,
            "pred_window": call.get('window_open_lap') is not None,
            "pred_window_lap": call.get('window_open_lap'),
            "pred_pass": pred_pass_lap is not None,
            "pred_pass_lap": pred_pass_lap,
            "verdict": call.get('verdict'),
            "cum_probability": call.get('cumulative_probability'),
            "gap_mae_s": round(mae_s / n, 3) if n else None,
            "gap_bias_s": round(bias_s / n, 3) if n else None,
            "n_gap_laps": n,
        })
        n_checked += 1
        if limit_checkpoints and n_checked >= limit_checkpoints:
            break
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def rate(n, d):
    return f"{n}/{d}" if d else "0/0"


def print_report(checkpoints, segments_used):
    print("\n" + "=" * 90)
    print("RACE CALL BACKTEST — per race / pair / checkpoint")
    print("=" * 90)
    print(f"{'TRACK':<24}{'PAIR':<9}{'L0':>4}{'->':>3}{'L':>4}{'GAP':>6}  "
          f"{'W!':>3}{'W?':>3}  {'P!':>3}{'P?':>3}  {'MAE':>6}  PRED")
    print("-" * 90)
    for c in checkpoints:
        mae = f"{c['gap_mae_s']:.2f}" if c.get('gap_mae_s') is not None \
            else "  -"
        pred_txt = (f"PASS L{c['pred_pass_lap']}" if c['pred_pass']
                    else (f"~L{c['pred_window_lap']}" if c['pred_window']
                          else "no window"))
        print(f"{c['race_track'][:24]:<24}"
              f"{c['leader'] + '/' + c['chaser']:<9}"
              f"{c['checkpoint_lap']:>4}->{c['horizon_lap']:>3}"
              f"{c['gap0_s']:>6.1f}  "
              f"{'Y' if c['actual_window'] else '.':>3}"
              f"{'Y' if c['pred_window'] else '.':>3}  "
              f"{'Y' if c['actual_pass'] else '.':>3}"
              f"{'Y' if c['pred_pass'] else '.':>3}  {mae:>6}  {pred_txt}")

    def counts(pred_key, act_key):
        tp = sum(1 for c in checkpoints if c[pred_key] and c[act_key])
        fp = sum(1 for c in checkpoints if c[pred_key] and not c[act_key])
        fn = sum(1 for c in checkpoints if not c[pred_key] and c[act_key])
        tn = sum(1 for c in checkpoints if not c[pred_key] and not c[act_key])
        return tp, fp, fn, tn

    w_tp, w_fp, w_fn, w_tn = counts('pred_window', 'actual_window')
    p_tp, p_fp, p_fn, p_tn = counts('pred_pass', 'actual_pass')

    print("-" * 90)
    print(f"Pair-segments evaluated: {segments_used}   "
          f"checkpoints: {len(checkpoints)}")
    print()
    print("WINDOW (real gap <= 1.2s before the next stop)")
    print(f"  TP {w_tp}  FP {w_fp}  FN {w_fn}  TN {w_tn}   "
          f"precision {rate(w_tp, w_tp + w_fp)}   "
          f"recall {rate(w_tp, w_tp + w_fn)}")
    print("PASS (real on-track order swap before the next stop)")
    print(f"  TP {p_tp}  FP {p_fp}  FN {p_fn}  TN {p_tn}   "
          f"precision {rate(p_tp, p_tp + p_fp)}   "
          f"recall {rate(p_tp, p_tp + p_fn)}")

    scored = [c for c in checkpoints if c.get('gap_mae_s') is not None]
    if scored:
        mae = sum(c['gap_mae_s'] for c in scored) / len(scored)
        bias = sum(c.get('gap_bias_s') or 0.0 for c in scored) / len(scored)
        print(f"\nGAP TRAJECTORY: MAE {mae:.3f}s/lap over {len(scored)}"
              f" checkpoints  (bias {bias:+.3f}s/lap = "
              f"model {'over' if bias > 0 else 'under'}closes)")

    print("\nCALIBRATION (model cumulative P vs real pass rate)")
    buckets = [(0.0, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 1.01)]
    for lo, hi in buckets:
        grp = [c for c in checkpoints
               if c.get('cum_probability') is not None
               and lo <= c['cum_probability'] < hi]
        if not grp:
            continue
        real = sum(1 for c in grp if c['actual_pass'])
        lab = f"[{lo:.1f},{min(hi, 1.0):.1f})"
        print(f"  P {lab:<9} n={len(grp):>4}  real pass {real} "
              f"({100.0 * real / len(grp):5.1f}%)")
    return {"window": (w_tp, w_fp, w_fn, w_tn),
            "pass": (p_tp, p_fp, p_fn, p_tn)}


def main():
    ap = argparse.ArgumentParser(
        description="Replay stored races and score Race Call predictions "
                    "against what actually happened.")
    ap.add_argument("--year", type=int, default=None, help="season filter")
    ap.add_argument("--track", default=None, help="track substring filter")
    ap.add_argument("--pair", default=None,
                    help="comma-separated driver codes, e.g. HAM,VER")
    ap.add_argument("--every", type=int, default=5,
                    help="checkpoint spacing in laps (default 5)")
    ap.add_argument("--min-gap", type=float, default=0.4)
    ap.add_argument("--max-gap", type=float, default=8.0,
                    help="skip checkpoints further behind than this")
    ap.add_argument("--min-span", type=int, default=5,
                    help="minimum racing laps left in the segment")
    ap.add_argument("--limit-checkpoints", type=int, default=0)
    ap.add_argument("--pass-cum", type=float, default=None,
                    help="cumulative P(overtake) target for an attack call "
                         "(default: overtake_inference.LIVE_PASS_CUM)")
    ap.add_argument("--max-segments", type=int, default=0,
                    help="stop after this many pair-segments")
    ap.add_argument("--out", default=None, help="write JSON results here")
    args = ap.parse_args()

    pair_filter = None
    if args.pair:
        pair_filter = tuple(c.strip().upper() for c in args.pair.split(","))
        if len(pair_filter) != 2:
            sys.exit("--pair needs two codes, e.g. HAM,VER")

    conn = get_db_connection()
    races = find_race_sessions(conn, year=args.year, track_sub=args.track)
    print(f"Found {len(races)} race weekends with >= 2 timed driver sessions.")

    checkpoints = []
    segments_used = 0
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
                                 args.min_span, args.limit_checkpoints,
                                 pass_cum=args.pass_cum)
            for c in got:
                if 'skip' not in c:
                    checkpoints.append(c)
            if got:
                segments_used += 1
            if args.max_segments and segments_used >= args.max_segments:
                break
        if args.max_segments and segments_used >= args.max_segments:
            break

    conn.close()

    if not checkpoints:
        print("No checkpoints produced — loosen --max-gap/--every or check"
              " model coverage.")
        return

    summary = print_report(checkpoints, segments_used)
    if args.out:
        out = {
            "meta": vars(args),
            "segments_used": segments_used,
            "checkpoints": checkpoints,
            "summary": {
                "window": dict(zip(("tp", "fp", "fn", "tn"),
                                   summary["window"])),
                "pass": dict(zip(("tp", "fp", "fn", "tn"),
                                 summary["pass"])),
            },
        }
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"\n[SAVED] {args.out}")


if __name__ == "__main__":
    main()
