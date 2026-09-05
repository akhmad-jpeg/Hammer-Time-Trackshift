"""
Backfill measured FastF1 sector times (Sector1/2/3Time) into the laps table.

Historical imports (before sector1_ms..sector3_ms existed) dropped the
measured sector split that FastF1 carries on every lap row.  This script
re-fetches each "Real World" session from the FastF1 archive and writes the
measured Sector1/2/3Time values back onto the matching laps.

Once this has run, the lap models train on *measured* sector splits instead
of the proxy (equal-time thirds) derived from the ~6 stored telemetry
samples — the models re-read the DB, so no other step is needed after a
successful backfill.

Network requirement: the FastF1 archive (api.fastf1.dev / ergast) must be
reachable.  When it is not (as in offline environments), the script reports
the failure and exits non-zero; the proxy-based pipeline remains fully
functional in the meantime.

Usage:
    python scripts/backfill_sector_times.py            # all missing sessions
    python scripts/backfill_sector_times.py --session 78
    python scripts/backfill_sector_times.py --limit 5
"""
import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

try:
    import fastf1
except ImportError:
    fastf1 = None

from config import get_db_connection

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

_CACHE_DIR = str(Path(__file__).resolve().parent.parent / "f1_cache")
if fastf1 is not None:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    fastf1.Cache.enable_cache(_CACHE_DIR)

# session_type enum -> FastF1 identifier code
_SESSION_CODE = {"Race": "R", "Qualifying": "Q", "Practice": "FP1"}


def sessions_missing_sectors(conn, only_session=None, limit=None):
    """Sessions whose laps all lack measured sector data (Real World only)."""
    cur = conn.cursor(dictionary=True)
    cur.execute(
        """
        SELECT s.session_id, s.driver_id, s.track_name, s.session_type,
               s.date, se.year, t.canonical_name
        FROM sessions s
        JOIN data_sources ds ON ds.source_id = s.source_id
        JOIN seasons se ON se.season_id = s.season_id
        LEFT JOIN tracks t ON t.track_id = s.track_id
        WHERE ds.source_type = 'Real World'
          AND EXISTS (
              SELECT 1 FROM laps l
              WHERE l.session_id = s.session_id AND l.sector1_ms IS NULL
          )
        ORDER BY s.session_id
        """
    )
    rows = cur.fetchall()
    cur.close()
    if only_session is not None:
        rows = [r for r in rows if r["session_id"] == only_session]
    if limit is not None:
        rows = rows[:limit]
    return rows


def _date_of(ev):
    d = ev.get("EventDate") or ev.get("Session1Date")
    if hasattr(d, "date"):
        d = d.date()
    return pd.to_datetime(d).date() if pd.notna(d) else None


def find_event_name(year: int, location: str, session_date=None):
    """Resolve the FastF1 event for a stored session via the schedule.

    The DB track names were recorded from an older FastF1 release's
    ``session.event["Location"]`` (full circuit names like "Bahrain
    International Circuit"), while current schedules carry short locations
    ("Sakhir") — name matching is therefore unreliable.  Each season has at
    most one event on any given date, and the stored session date is the
    race weekend's EventDate, so an exact EventDate match uniquely picks the
    weekend (double-headers at one circuit are always on different dates).

    Returns (EventName, event_row) or None.  FastF1 silently "corrects"
    unresolvable event names to the season's last Grand Prix, which would
    write one race's sectors onto another's laps — the caller additionally
    refuses to write when the loaded session's own date disagrees.
    """
    try:
        schedule = fastf1.get_event_schedule(year)
    except Exception as exc:  # network / archive errors
        logging.warning(f"{year} schedule unavailable: {exc}")
        return None
    if schedule is None or getattr(schedule, "empty", True):
        return None
    if session_date is not None:
        wanted = pd.to_datetime(session_date).date()
        day = pd.Timedelta(days=1)
        cand = [ev for _, ev in schedule.iterrows()
                if _date_of(ev) is not None and abs(_date_of(ev) - wanted) <= day]
        if len(cand) == 1:
            return str(cand[0]["EventName"]), cand[0]
        if cand:
            logging.warning(f"{year}: {len(cand)} events near {session_date} — "
                            f"refusing ambiguous match")
            return None
    # Last resort: exact Location equality (works for tracks whose short and
    # full names coincide, e.g. 'Circuit de Monaco').
    loc = str(location).strip().lower()
    exact = [ev for _, ev in schedule.iterrows()
             if str(ev.get("Location", "")).strip().lower() == loc]
    if len(exact) == 1:
        return str(exact[0]["EventName"]), exact[0]
    return None


def _sector_ms(td):
    """Timedelta -> whole ms, or None when the cell is NaN."""
    if td is None or not pd.notna(td):
        return None
    return int(round(td.total_seconds() * 1000.0))


def backfill_session(conn, info: dict) -> dict:
    """Fetch one FastF1 session and write sector times onto its laps."""
    sid = info["session_id"]
    year = int(info["year"])
    location = info["canonical_name"] or info["track_name"]
    session_type = info["session_type"]
    code = _SESSION_CODE.get(session_type)
    if code is None:
        return {"session_id": sid, "ok": False,
                "reason": f"unsupported session type {session_type!r}"}

    resolved = find_event_name(year, location, info.get("date"))
    if resolved is None:
        return {"session_id": sid, "ok": False,
                "reason": f"no exact schedule match for {location} {year}"}
    event_name, _ev = resolved

    try:
        session = fastf1.get_session(year, event_name, code)
        session.load(laps=True, telemetry=False, weather=False, messages=False)
    except Exception as exc:
        return {"session_id": sid, "ok": False,
                "reason": f"FastF1 load failed: {exc}"}

    # FastF1 can silently substitute a different event when the name does
    # not resolve (e.g. a testing weekend that has no race).  Refuse to
    # write anything unless the loaded session is the event we asked for.
    if info.get("date") is not None:
        loaded_date = _date_of(getattr(session, "event", {}))
        wanted = pd.to_datetime(info["date"]).date()
        if loaded_date is not None and abs(loaded_date - wanted) > pd.Timedelta(days=1):
            return {"session_id": sid, "ok": False,
                    "reason": f"loaded {session.event.get('EventName')} "
                              f"({loaded_date}) but DB says {wanted} — refused"}

    laps = session.laps
    if laps is None or laps.empty:
        return {"session_id": sid, "ok": False, "reason": "no FastF1 laps"}
    driver_laps = laps[laps["DriverNumber"].astype(str) == str(info["driver_id"])]
    if driver_laps.empty:
        return {"session_id": sid, "ok": False,
                "reason": f"driver #{info['driver_id']} not in session"}

    cur = conn.cursor()
    updated = 0
    unchanged = 0
    for _, lap in driver_laps.iterrows():
        if not pd.notna(lap.get("LapNumber")):
            continue
        lap_num = int(lap["LapNumber"])
        s1 = _sector_ms(lap.get("Sector1Time"))
        s2 = _sector_ms(lap.get("Sector2Time"))
        s3 = _sector_ms(lap.get("Sector3Time"))
        if s1 is None and s2 is None and s3 is None:
            unchanged += 1
            continue
        cur.execute(
            """
            UPDATE laps
               SET sector1_ms = COALESCE(%s, sector1_ms),
                   sector2_ms = COALESCE(%s, sector2_ms),
                   sector3_ms = COALESCE(%s, sector3_ms)
             WHERE session_id = %s AND lap_number = %s
            """,
            (s1, s2, s3, sid, lap_num),
        )
        updated += cur.rowcount
    conn.commit()
    cur.close()
    return {"session_id": sid, "ok": True, "updated": updated,
            "unchanged": unchanged, "event": event_name}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", type=int, default=None,
                    help="backfill only this session_id")
    ap.add_argument("--limit", type=int, default=None,
                    help="backfill at most N sessions (oldest first)")
    args = ap.parse_args()

    if fastf1 is None:
        print("[ERROR] fastf1 is not installed — run `pip install fastf1`.")
        sys.exit(1)

    conn = get_db_connection()
    targets = sessions_missing_sectors(conn, only_session=args.session,
                                       limit=args.limit)
    if not targets:
        print("[INFO] No sessions missing sector data — nothing to do.")
        conn.close()
        return
    print(f"[INFO] {len(targets)} session(s) to backfill from FastF1.\n")

    ok = fail = 0
    for t in targets:
        started = datetime.now()
        res = backfill_session(conn, t)
        ms = int((datetime.now() - started).total_seconds() * 1000)
        if res["ok"]:
            ok += 1
            print(f"[ OK ] session {t['session_id']} {t['year']} {t['track_name']} "
                  f"({res['event']}) — updated {res['updated']} lap(s), "
                  f"{res['unchanged']} without sector times [{ms} ms]")
        else:
            fail += 1
            print(f"[FAIL] session {t['session_id']} {t['year']} {t['track_name']} "
                  f"— {res['reason']}")
    conn.close()
    print(f"\n[INFO] Done: {ok} ok, {fail} failed.")
    if fail:
        sys.exit(2)


if __name__ == "__main__":
    main()
