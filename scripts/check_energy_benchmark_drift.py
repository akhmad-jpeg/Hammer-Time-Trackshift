"""Scheduled drift check for the energy benchmarks.

Re-runs the benchmark projections FRESH (same DB -> same numbers by design)
and fails LOUDLY if any headline number has moved beyond epsilon, or the
deck-claim reproduction flag flipped.  A moved number means the energy
simulator, the pace models, or the DB changed since the artifact was
committed -- the validation no longer says what the deck claims, and a
re-run of scripts/benchmark_energy_strategy.py / benchmark_energy_fleet.py
would silently rewrite the story.

Exit codes:  0 = no drift  1 = drift (report on stderr).

Scheduling (run daily):
    # cron
    0 6 * * *  cd /path/to/project && python scripts/check_energy_benchmark_drift.py
    # Windows Task Scheduler (schtasks)
    schtasks /Create /TN "F1 EnergyBenchmarkDrift" ^
      /TR "python C:\\path\\to\\project\\scripts\\check_energy_benchmark_drift.py" ^
      /SC DAILY /ST 06:00

Usage:
    python scripts/check_energy_benchmark_drift.py            # anchor + fleet (~5s)
    python scripts/check_energy_benchmark_drift.py --scope anchor
    python scripts/check_energy_benchmark_drift.py --eps 0.02 --refresh
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_energy_strategy import (
    PROJECT_ROOT,
    find_race_sessions,
    load_energy_pace,
    session_payload,
)
from benchmark_energy_fleet import build_fleet

ANCHOR_ARTIFACT = PROJECT_ROOT / 'backtests' / 'monaco_2023_energy.json'
FLEET_SUMMARY = PROJECT_ROOT / 'backtests' / 'energy_fleet_summary.json'

_DRIFT = []


def _check(name, field, old, new, eps):
    if old is None or new is None:
        if old != new:
            _DRIFT.append(f"  {name}.{field}: committed={old}  fresh={new}")
        return
    if abs(float(old) - float(new)) > eps:
        _DRIFT.append(f"  {name}.{field}: committed={old}  fresh={new}  "
                      f"(|diff| {abs(float(old) - float(new)):.3f}s > eps {eps}s)")


def check_anchor(conn, energy_pace, eps):
    races = find_race_sessions(conn, year=2023, track_sub='Monaco')
    payloads = []
    for (track, date) in sorted(races):
        for code in sorted(races[(track, date)]):
            payloads.append(session_payload(conn, races[(track, date)][code], energy_pace))
    if not payloads:
        _DRIFT.append("  anchor: no Monaco 2023 sessions found in DB")
        return
    close = [p['closing_phase_improvement_s'] for p in payloads]
    full = [p['full_race_improvement_s'] for p in payloads]
    fresh = {
        'closing_phase_improvement_s_mean': round(sum(close) / len(close), 3),
        'full_race_improvement_s_mean': round(sum(full) / len(full), 3),
        'deck_claim_reproduced': abs(sum(close) / len(close) + 1.8) < 0.25,
    }
    committed = json.loads(ANCHOR_ARTIFACT.read_text(encoding='utf-8'))['headline']
    print(f"  anchor (Monaco 2023, {len(payloads)} sessions):")
    for f, v in fresh.items():
        _check('anchor', f, committed.get(f), v, eps)
        print(f"    {f:<40} committed={committed.get(f)}  fresh={v}"
              + ("   <-- DRIFT" if committed.get(f) != v else ""))


def check_fleet(conn, energy_pace, eps):
    import types
    ns = types.SimpleNamespace(year=None, track=None, closing_laps=15, limit=0,
                               artifacts=False, artifacts_dir=str(PROJECT_ROOT))
    fresh_rows, _ = build_fleet(conn, energy_pace, ns)
    fresh_by_key = {r['key']: r for r in fresh_rows}
    committed = json.loads(FLEET_SUMMARY.read_text(encoding='utf-8'))
    committed_by_key = {r['key']: r for r in committed.get('races', [])}

    for key in sorted(set(fresh_by_key) | set(committed_by_key)):
        fr = fresh_by_key.get(key)
        cr = committed_by_key.get(key)
        if cr is None:
            _DRIFT.append(f"  fleet {key}: present fresh but NOT in committed summary")
            continue
        if fr is None:
            _DRIFT.append(f"  fleet {key}: in committed summary but NOT produced fresh")
            continue
        for f in ('closing_phase_improvement_s_mean',
                  'full_race_improvement_s_mean',
                  'pace_s_per_mj'):
            _check(f"fleet {key}", f, cr.get(f), fr.get(f), eps)
        for f in ('strategy_mode', 'strategy_ok'):
            if cr.get(f) != fr.get(f):
                _DRIFT.append(f"  fleet {key}.{f}: committed={cr.get(f)}  fresh={fr.get(f)}")
    print(f"  fleet: {len(committed_by_key)} committed races vs {len(fresh_by_key)} fresh")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(
        description="Fail loudly if the energy benchmarks drifted from their "
                    "committed artifacts.")
    ap.add_argument("--scope", choices=['anchor', 'fleet', 'both'], default='both')
    ap.add_argument("--eps", type=float, default=0.02,
                    help="headline tolerance in seconds (default 0.02)")
    ap.add_argument("--refresh", action="store_true",
                    help="on a PASS, rewrite the committed artifacts so their "
                         "timestamps stay current")
    args = ap.parse_args()

    from config import get_db_connection
    conn = get_db_connection()
    energy_pace = load_energy_pace()

    print("ENERGY BENCHMARK DRIFT CHECK (eps %.3fs)" % args.eps)
    if args.scope in ('anchor', 'both'):
        if not ANCHOR_ARTIFACT.exists():
            _DRIFT.append("  anchor artifact missing: backtests/monaco_2023_energy.json")
        else:
            check_anchor(conn, energy_pace, args.eps)
    if args.scope in ('fleet', 'both'):
        if not FLEET_SUMMARY.exists():
            _DRIFT.append("  fleet summary missing: backtests/energy_fleet_summary.json")
        else:
            check_fleet(conn, energy_pace, args.eps)
    conn.close()

    if _DRIFT:
        print("\nDRIFT DETECTED — the committed validation no longer matches a "
              "fresh run:", file=sys.stderr)
        for line in _DRIFT:
            print(line, file=sys.stderr)
        print("\nEither the simulator / models / DB changed (re-run "
              "scripts/benchmark_energy_fleet.py and review the new numbers), "
              "or the change is intended (re-commit the refreshed artifacts).",
              file=sys.stderr)
        return 1

    print("\nNO DRIFT — fresh numbers match the committed artifacts "
          "(deterministic, as designed).")
    if args.refresh:
        import subprocess
        cmds = [[sys.executable, str(PROJECT_ROOT / 'scripts' / 'benchmark_energy_strategy.py')],
                [sys.executable, str(PROJECT_ROOT / 'scripts' / 'benchmark_energy_fleet.py')]]
        for cmd in cmds:
            subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True,
                           capture_output=True, text=True)
        print("[REFRESHED] committed artifacts rewritten.")
    return 0


if __name__ == "__main__":
    sys.exit(main())