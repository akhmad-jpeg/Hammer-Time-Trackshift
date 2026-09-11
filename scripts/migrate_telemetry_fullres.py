"""Schema migration: full-resolution telemetry (time_s, distance_m).

The FastF1 importer now stores every ~0.2-0.3 s sample with its intra-lap
time (time_s, seconds from the lap start) and distance (distance_m, metres
along the lap).  Older databases lack these columns; this script adds them
idempotently.  Both columns stay NULL on rows written by older importers or
by the live game-capture client (which has no intra-lap clock yet), so all
existing code paths keep working unchanged.

Idempotency: checks information_schema before each ALTER, so re-running on
a fully migrated database is a no-op (exit 0).

Optional purge (--purge-telemetry): empties the telemetry table before
migration so a fresh re-import starts clean.  FK-SAFE ORDER: telemetry rows
are deleted FIRST (they reference laps), then laps, strategy_events,
race_state, and finally sessions.  drivers / seasons / tracks / regulations
/ data_sources reference tables are KEPT so driver ids and track ids stay
stable across the re-import.  Never touches anything outside the project's
database.

Usage:
    python scripts/migrate_telemetry_fullres.py                  # migrate only
    python scripts/migrate_telemetry_fullres.py --purge-telemetry
    python scripts/migrate_telemetry_fullres.py --purge-telemetry --yes
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import get_db_connection

# Column definitions added by this migration.
COLUMNS = {
    "time_s": ("DOUBLE NULL", "intra-lap time (s from the lap's first sample)"),
    "distance_m": ("DOUBLE NULL", "distance along the lap (m)"),
}

# FK-safe child-first deletion order.  Everything here references sessions
# (or laps, which reference sessions); drivers/seasons/tracks/regulations/
# data_sources are deliberately absent (kept).
PURGE_ORDER = ("telemetry", "strategy_events", "race_state", "laps", "sessions")


def column_exists(cur, table: str, column: str) -> bool:
    cur.execute(
        """
        SELECT COUNT(*) FROM information_schema.COLUMNS
        WHERE TABLE_SCHEMA = DATABASE()
          AND TABLE_NAME = %s AND COLUMN_NAME = %s
        """,
        (table, column),
    )
    return bool(cur.fetchone()[0])


def table_exists(cur, table: str) -> bool:
    cur.execute(
        """
        SELECT COUNT(*) FROM information_schema.TABLES
        WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s
        """,
        (table,),
    )
    return bool(cur.fetchone()[0])


def row_counts(cur) -> dict:
    out = {}
    for t in ("sessions", "laps", "telemetry", "race_state", "strategy_events"):
        if table_exists(cur, t):
            cur.execute(f"SELECT COUNT(*) FROM `{t}`")
            out[t] = int(cur.fetchone()[0])
        else:
            out[t] = None
    return out


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(
        description="Add full-resolution telemetry columns (time_s, distance_m) "
                    "and optionally purge the transactional tables for re-import.")
    parser.add_argument("--purge-telemetry", action="store_true",
                        help="DELETE telemetry rows (child-first FK-safe order: "
                             "telemetry, strategy_events, race_state, laps, "
                             "sessions) before migrating.  Reference tables "
                             "(drivers, seasons, tracks, ...) are kept.")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="Skip the interactive confirmation.")
    args = parser.parse_args()

    conn = get_db_connection()
    cur = conn.cursor()

    before = row_counts(cur)
    print("=" * 62)
    print("FULL-RESOLUTION TELEMETRY MIGRATION")
    print("=" * 62)
    for t, n in before.items():
        print(f"  {t:<16} {n if n is not None else '(table missing)'}")

    if args.purge_telemetry:
        if not args.yes:
            answer = input(
                f"\nThis will DELETE {before.get('telemetry', 0)} telemetry rows "
                f"(and their parent laps/sessions/strategy_events/race_state "
                f"rows).  Reference tables are kept.  Type PURGE to continue: "
            ).strip()
            if answer != "PURGE":
                print("Aborted — nothing was modified.")
                cur.close()
                conn.close()
                return 1

        print("\n[PURGE] deleting child-first ...")
        try:
            for t in PURGE_ORDER:
                if table_exists(cur, t):
                    cur.execute(f"DELETE FROM `{t}`")
                    print(f"  deleted {cur.rowcount} row(s) from {t}")
            conn.commit()
        except Exception as exc:
            conn.rollback()
            print(f"[ERROR] purge failed and was rolled back: {exc}")
            cur.close()
            conn.close()
            return 1

    # ------------------------------------------------------------------
    # Migration: add the columns when missing (idempotent).
    # ------------------------------------------------------------------
    print("\n[SCHEMA] telemetry columns ...")
    changed = False
    for col, (ddl, _why) in COLUMNS.items():
        if column_exists(cur, "telemetry", col):
            print(f"  {col} already present — nothing to do")
            continue
        # NOTE: no COMMENT clause — f-string interpolation of arbitrary
        # comment text is an SQL-injection-shaped footgun, and the column
        # meaning is documented in this script's docstring.
        cur.execute(f"ALTER TABLE `telemetry` ADD COLUMN `{col}` {ddl} AFTER `drs`")
        changed = True
        print(f"  added {col} ({_why})")

    if changed:
        conn.commit()
        print("\n[OK] migration committed.")
    else:
        print("\n[OK] nothing to migrate — database already at full-resolution schema.")

    after = row_counts(cur)
    if before != after:
        print("\nRow counts after:")
        for t, n in after.items():
            print(f"  {t:<16} {n if n is not None else '(table missing)'}")

    cur.close()
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
