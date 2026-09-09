"""Lightweight DB schema smoke test for the f1_strategy platform.

Verifies the LIVE database matches what the code expects: every table the
application reads/writes exists, and the tables dropped in the Sept 2026
cleanup (driver_metrics, overtake_events, race_predictions,
strategy_recommendations, tyre_stints) are ABSENT — a restore from a stale
export would silently recreate them, and a missing required table would
crash the dashboard/importers at runtime.

Intentionally dependency-light (config + mysql.connector only, no pandas /
sklearn imports) so it runs fast and can be scheduled alongside
check_energy_benchmark_drift.py.

Exit codes:  0 = schema OK   1 = schema drift / connection problem.

Usage:
    python scripts/check_db_schema.py            # tables + key columns
    python scripts/check_db_schema.py --tables   # table presence only
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DB_CONFIG, get_db_connection  # noqa: E402

# The 11 live tables (matches database/f1_strategy.sql after the Sept 2026
# cleanup).  Everything the code touches is listed here.
REQUIRED_TABLES = {
    'data_sources',
    'drivers',
    'laps',
    'race_state',
    'regulations',
    'seasons',
    'sessions',
    'strategy_events',
    'telemetry',
    'track_aliases',
    'tracks',
}

# Dropped in the Sept 2026 cleanup — must NOT exist.  A restore from a
# pre-cleanup export would recreate them (empty and unused, but confusing).
FORBIDDEN_TABLES = {
    'driver_metrics',
    'overtake_events',
    'race_predictions',
    'strategy_recommendations',
    'tyre_stints',
}

# Minimal column contract for the hot tables: if one of these is missing the
# code WILL break (importers INSERT them, models SELECT them).  Checked with
# --full (default); skip with --tables for a presence-only check.
REQUIRED_COLUMNS = {
    'laps': {'lap_id', 'session_id', 'driver_id', 'lap_number',
             'lap_time_ms', 'tyre_compound', 'tyre_age', 'is_valid'},
    'sessions': {'session_id', 'driver_id', 'track_name', 'session_type',
                 'date', 'source_id'},
    'race_state': {'session_id', 'driver_id', 'lap_number',
                   'energy_start_mj', 'energy_deployed_mj',
                   'energy_harvested_mj', 'energy_end_mj'},
    'strategy_events': {'event_id', 'lap_id', 'event_type'},
}


def main():
    ap = argparse.ArgumentParser(
        description='Schema smoke test: required tables present, dropped '
                    'tables absent, key columns intact.')
    ap.add_argument('--tables', action='store_true',
                    help='check table presence only (skip column checks)')
    args = ap.parse_args()

    problems = []
    try:
        conn = get_db_connection()
    except Exception as exc:
        print(f"[FAIL] Cannot connect to '{DB_CONFIG['database']}' at "
              f"{DB_CONFIG['host']}:{DB_CONFIG['port']}: {exc}")
        return 1

    cur = conn.cursor()
    try:
        cur.execute("SHOW TABLES")
        existing = {row[0] for row in cur.fetchall()}

        missing = sorted(REQUIRED_TABLES - existing)
        if missing:
            problems.append(f"missing required tables: {', '.join(missing)}")

        present_forbidden = sorted(FORBIDDEN_TABLES & existing)
        if present_forbidden:
            problems.append(
                "dropped tables still exist (restore from a stale export?): "
                + ', '.join(present_forbidden))

        if not args.tables:
            for table, cols in REQUIRED_COLUMNS.items():
                if table not in existing:
                    continue  # already reported as missing
                cur.execute(f"SHOW COLUMNS FROM `{table}`")  # noqa: S608
                have = {row[0] for row in cur.fetchall()}
                absent = sorted(cols - have)
                if absent:
                    problems.append(
                        f"`{table}` missing columns: {', '.join(absent)}")
    finally:
        cur.close()
        conn.close()

    if problems:
        print("[FAIL] Schema drift detected:")
        for p in problems:
            print(f"  - {p}")
        return 1

    scope = ('table presence' if args.tables
             else f"tables + columns on {len(REQUIRED_COLUMNS)} hot tables")
    print(f"[OK] Schema matches expectations ({scope}): "
          f"{len(REQUIRED_TABLES)} required tables present, "
          f"{len(FORBIDDEN_TABLES)} dropped tables absent.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
