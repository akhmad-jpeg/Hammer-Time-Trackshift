"""Fleet-wide energy-strategy benchmark — every stored race, one pass.

Runs the exact projection of scripts/benchmark_energy_strategy.py (AI
strategy vs 'Flat-Out' baseline: per-lap pace credits = measured s/MJ x
deployed MJ, fastest feasible mode wins, closing phase = final --closing-laps
laps) across EVERY stored Race session with >= 25 timed laps, and emits:

  * backtests/energy_fleet_summary.json  — one aggregate row per race
    (track/date, driver codes, pace + basis, closing-phase / full-race
    improvement means, winning mode), sorted biggest-strategy-gain first so
    "where the model's gains are largest" is the first line.  Small,
    deterministic, committed.
  * backtests/energy/<track>__<date>.json — FULL per-race artifacts with the
    per-lap audit trail (the chart data the Validation panel draws).
    Deterministic and regenerable, so they are gitignored (see .gitignore).

The track-name resolution is alias/case aware (energy_simulator.
resolve_track_profile), so every stored race uses its measured per-track
s/MJ — no silent flat-constant fallbacks.

Usage:
    python scripts/benchmark_energy_fleet.py                      # all 115 races
    python scripts/benchmark_energy_fleet.py --track Monaco       # one circuit
    python scripts/benchmark_energy_fleet.py --year 2026 --limit 4   # smoke run
    python scripts/benchmark_energy_fleet.py --no-artifacts       # summary only
"""

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_energy_strategy import (
    FLAT_OUT_MODE,
    PROJECT_ROOT,
    find_race_sessions,
    load_energy_pace,
    session_payload,
)
from energy_simulator import normalize_track_name

# The race the deck's page-8 claim describes; its per-race artifact carries
# the deck-claim metadata so the Validation panel can show the "REPRODUCED"
# badge exactly there and nowhere else.  CLAIM is the magnitude (1.8 s) --
# the signed headline the deck shows is -1.8 s.
DECK_CLAIM_RACE = ('2023-05-28', 1.8)   # (race date, claim magnitude s)

DEFAULT_SUMMARY = PROJECT_ROOT / 'backtests' / 'energy_fleet_summary.json'
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / 'backtests' / 'energy'


def race_key(track, date):
    """Sanitised artifact stem: '<track>__<date>' (filesystem-safe)."""
    s = unicodedata.normalize('NFKD', str(track))
    s = ''.join(c for c in s if c.isalnum() or c in ' -_')
    s = re.sub(r'[^A-Za-z0-9_-]+', '_', s.strip()).strip('_')
    return f"{s or 'track'}__{date}"


def build_fleet(conn, energy_pace, args):
    """Run every race's projection; return (races, artifacts_dir)."""
    races = find_race_sessions(conn, year=args.year, track_sub=args.track)
    keys = sorted(races)   # (track, date) — deterministic order
    if args.limit:
        keys = keys[:args.limit]

    summary_rows = []
    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    for i, (track, date) in enumerate(keys, 1):
        codes = sorted(races[(track, date)])
        payloads = []
        for code in codes:
            sid = races[(track, date)][code]
            payloads.append(session_payload(
                conn, sid, energy_pace, closing_laps=args.closing_laps))
        if not payloads:
            continue

        close = [p['closing_phase_improvement_s'] for p in payloads]
        full = [p['full_race_improvement_s'] for p in payloads]
        p0 = payloads[0]
        strategies = Counter(p['strategy_mode'] for p in payloads)

        row = {
            "key": race_key(track, date),
            "track": track,
            "date": date,
            "year": str(date)[:4],
            "drivers": codes,
            "sessions": len(payloads),
            "laps": p0['laps'],
            "pace_s_per_mj": p0['pace_s_per_mj'],
            "pace_basis": p0['pace_basis'],
            "strategy_mode": strategies.most_common(1)[0][0],
            "strategy_ok": all(p['strategy_ok'] for p in payloads),
            "closing_phase_improvement_s_mean": round(sum(close) / len(close), 3),
            "full_race_improvement_s_mean": round(sum(full) / len(full), 3),
        }
        summary_rows.append(row)

        # Full per-race artifact: same schema as the deck anchor output, so
        # the Validation panel renders any race with the existing renderer.
        if args.artifacts:
            per_race = {
                "meta": {
                    "script": "scripts/benchmark_energy_fleet.py",
                    "generated_at": datetime.now().isoformat(timespec='seconds'),
                    "race": f"{track} | {date}",
                    "note": "Per-race artifact of the fleet sweep — same "
                            "projection as scripts/benchmark_energy_strategy.py",
                    "flat_out_mode": FLAT_OUT_MODE,
                },
                "config": {
                    "era": p0['spec'],
                    "era_label": p0['spec_label'],
                    "pace_s_per_mj": p0['pace_s_per_mj'],
                    "pace_basis": p0['pace_basis'],
                    "closing_laps": args.closing_laps,
                },
                "sessions": payloads,
                "headline": {
                    "closing_phase_improvement_s_mean": row['closing_phase_improvement_s_mean'],
                    "full_race_improvement_s_mean": row['full_race_improvement_s_mean'],
                },
            }
            # The deck claim is about THIS race (Monaco 2023): attach the
            # same metadata the anchor artifact carries, so the panel shows
            # the reproduction badge only for the race it actually describes.
            if (normalize_track_name(track) == 'monaco'
                    and str(date) == DECK_CLAIM_RACE[0]):
                claim = DECK_CLAIM_RACE[1]
                per_race['meta']['deck_claim'] = (
                    "Trackshift 'Hammer Time' deck p.8: '-1.8s race time "
                    "improvement' of AI strategy vs 'Flat-Out' baseline, "
                    "2023 Monaco GP")
                per_race['headline']['deck_claim_s'] = -claim   # signed, as shown
                per_race['headline']['deck_claim_reproduced'] = (
                    abs(row['closing_phase_improvement_s_mean'] + claim) < 0.25)
            (artifacts_dir / f"{row['key']}.json").write_text(
                json.dumps(per_race, indent=1), encoding='utf-8')

        sys.stdout.write(f"\r[{i}/{len(keys)}] {track:<40} {len(payloads)} drivers   ")
        sys.stdout.flush()
    print()

    # Biggest strategy gains first (most negative closing-phase delta).
    summary_rows.sort(key=lambda r: r['closing_phase_improvement_s_mean'])
    return summary_rows, artifacts_dir


def print_report(rows):
    print("=" * 100)
    print(f"FLEET ENERGY BENCHMARK — {len(rows)} races "
          f"(AI strategy vs '{FLAT_OUT_MODE}' baseline)")
    print("=" * 100)
    print(f"{'TRACK':<38}{'DATE':<12}{'N':>3}{'LAPS':>6}{'PACE':>8}  "
          f"{'CLOSING':>9}{'FULL':>9}  MODE")
    for r in rows[:15]:
        print(f"{r['track']:<38}{r['date']:<12}{r['sessions']:>3}{r['laps']:>6}"
              f"{r['pace_s_per_mj']:>8.4f}  {r['closing_phase_improvement_s_mean']:>+8.2f}s"
              f"{r['full_race_improvement_s_mean']:>+8.2f}s  {r['strategy_mode']}")
    if len(rows) > 20:
        print("  ...")
        for r in rows[-5:]:
            print(f"{r['track']:<38}{r['date']:<12}{r['sessions']:>3}{r['laps']:>6}"
                  f"{r['pace_s_per_mj']:>8.4f}  {r['closing_phase_improvement_s_mean']:>+8.2f}s"
                  f"{r['full_race_improvement_s_mean']:>+8.2f}s  {r['strategy_mode']}")
    close = [r['closing_phase_improvement_s_mean'] for r in rows]
    full = [r['full_race_improvement_s_mean'] for r in rows]
    print("-" * 100)
    print(f"RACES: {len(rows)}   closing-phase mean {sum(close)/len(close):+.2f}s   "
          f"full-race mean {sum(full)/len(full):+.2f}s")
    print("=" * 100)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(
        description="Fleet-wide energy-strategy benchmark over every stored race.")
    ap.add_argument("--year", type=int, default=None)
    ap.add_argument("--track", default=None)
    ap.add_argument("--closing-laps", type=int, default=15)
    ap.add_argument("--limit", type=int, default=0,
                    help="cap number of races (smoke runs)")
    ap.add_argument("--out", default=str(DEFAULT_SUMMARY))
    ap.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR))
    ap.add_argument("--no-artifacts", action="store_true",
                    help="skip writing full per-race artifacts")
    args = ap.parse_args()
    args.artifacts = not args.no_artifacts

    from config import get_db_connection
    conn = get_db_connection()
    energy_pace = load_energy_pace()
    rows, _ = build_fleet(conn, energy_pace, args)
    conn.close()

    print_report(rows)

    out = {
        "meta": {
            "script": "scripts/benchmark_energy_fleet.py",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "command": " ".join(sys.argv),
            "method": "same projection as scripts/benchmark_energy_strategy.py — "
                      "per-lap credits = measured s/MJ x deployed MJ; strategy = "
                      "fastest feasible mode; closing phase = final "
                      "--closing-laps laps",
            "track_name_resolution": "casefold + short/full alias aware "
                                     "(energy_simulator.resolve_track_profile)",
        },
        "config": {
            "closing_laps": args.closing_laps,
            "flat_out_mode": FLAT_OUT_MODE,
        },
        "races": rows,
        "headline": {
            "races": len(rows),
            "closing_phase_improvement_s_mean": round(
                sum(r['closing_phase_improvement_s_mean'] for r in rows) / len(rows), 3)
            if rows else None,
            "full_race_improvement_s_mean": round(
                sum(r['full_race_improvement_s_mean'] for r in rows) / len(rows), 3)
            if rows else None,
            "biggest_gain": (rows[0]['key'], rows[0]['closing_phase_improvement_s_mean'])
            if rows else None,
            "deck_anchor": "backtests/monaco_2023_energy.json (closing -1.88s "
                           "reproduces the deck's -1.8s)",
        },
    }
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"[SAVED] {args.out}")
    if args.artifacts:
        print(f"[SAVED] per-race artifacts -> {args.artifacts_dir}/")


if __name__ == "__main__":
    main()