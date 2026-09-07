"""
Batch ingestion utility.

Imports one or more drivers across multiple races/seasons by calling
import_race() from import_f1_race.  All driver and track selection goes
through the same DB-backed helpers used by the single-race importer,
ensuring identical resolution behaviour in both scripts.

Driver discovery is self-service: a driver code that is not yet in the
drivers table (e.g. a rookie) triggers a single lightweight warm-up pass
that reads the reference session's full grid from FastF1 and seeds the
drivers table before the import matrix runs — all in one process.
"""

import argparse
import sys
import logging
from pathlib import Path

# Ensure scripts directory is on sys.path for sibling imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from import_f1_race import (
    import_race,
    fetch_drivers_from_db,
    fetch_grid_from_fastf1,
    seed_drivers_from_grid,
    resolve_driver_input,
    resolve_race_input,
    print_driver_menu,
    print_track_menu,
    RACE_CALENDAR,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s - %(message)s")


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def import_dataset(
    seasons: list[int],
    races: list[str] | None,
    drivers: list[tuple[int, str]],
    session_type: str = "R",
    allow_duplicate: bool = False,
) -> None:
    """
    Import multiple drivers across multiple seasons and races.

    Parameters
    ----------
    seasons        : List of championship years, e.g. [2023, 2024].
    races          : Explicit race-name list, or None to use RACE_CALENDAR.
    drivers        : List of (driver_id, driver_code) pairs.  driver_id is the
                     permanent F1 driver id (= drivers.driver_id in the DB);
                     driver_code is the stable 3-letter code (e.g. "VER") passed
                     to import_race so the driver is matched inside each session
                     by code — race numbers change across seasons (champion
                     takes #1, etc.) and can collide between years.
    session_type   : "R", "Q", "FP1", "FP2", or "FP3".
    allow_duplicate: If False, skip sessions already in the DB.
    """
    logging.info("=" * 60)
    logging.info("BATCH F1 DATASET INGESTION")
    logging.info("=" * 60)

    # Pre-flight plan: (year, race_list) per season, then a matrix summary.
    plan: list[tuple[int, list[str]]] = []
    for year in seasons:
        race_list = races if races else RACE_CALENDAR.get(year, ["Bahrain", "Monaco"])
        plan.append((year, race_list))

    race_slots = sum(len(rl) for _, rl in plan)
    total_cells = len(drivers) * race_slots
    logging.info(
        f"Matrix: {len(drivers)} driver(s) x {race_slots} race-slot(s) "
        f"across {len(seasons)} season(s) = {total_cells} session import(s)"
    )

    successful = 0
    failed     = 0
    per_driver = {code: {"ok": 0, "fail": 0} for _, code in drivers}

    for driver_id, driver_code in drivers:
        label = f"#{driver_id} ({driver_code})"
        for year, race_list in plan:
            logging.info(f"\n--- Season {year}: {len(race_list)} race(s), driver {label} ---")

            for race_name in race_list:
                try:
                    session_id = import_race(
                        year=year,
                        race_name=race_name,
                        driver_id=driver_id,
                        session_type=session_type,
                        allow_duplicate=allow_duplicate,
                        driver_code=driver_code,
                    )
                    if session_id is not None:
                        successful += 1
                        per_driver[driver_code]["ok"] += 1
                    else:
                        failed += 1
                        per_driver[driver_code]["fail"] += 1
                except Exception as exc:
                    logging.error(f"  Batch error for {year} {race_name} {label}: {exc}")
                    failed += 1
                    per_driver[driver_code]["fail"] += 1

    print("\n" + "=" * 60)
    print("BATCH COMPLETE")
    print("=" * 60)
    for driver_id, driver_code in drivers:
        stats = per_driver[driver_code]
        print(f"  #{driver_id} {driver_code:<4}: {stats['ok']} ok, {stats['fail']} failed")
    print(f"  Total successful / already-existed : {successful}")
    print(f"  Total failed                       : {failed}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Driver resolution + warm-up pass
# ---------------------------------------------------------------------------

def _resolve_driver_tokens(tokens: list[str], db_drivers: list[dict]) -> tuple[list[dict], list[str]]:
    """Resolve raw driver tokens (F1 numbers or 3-letter codes) against the DB.

    Returns (resolved driver rows, unresolved raw tokens).
    """
    resolved: list[dict] = []
    unresolved: list[str] = []
    for token in tokens:
        try:
            resolved.append(resolve_driver_input(token, db_drivers))
        except ValueError:
            unresolved.append(token)
    return resolved, unresolved


def _warmup_seed(seasons: list[int], races: list[str] | None) -> list[dict]:
    """Single reference-row warm-up: seed the drivers table from one session.

    Reads the full grid of the first requested (year, race) directly from
    FastF1 (no laps/telemetry downloaded) and inserts any drivers that are
    not in the drivers table yet.  Returns the refreshed driver list.
    """
    year = seasons[0]
    race_list = races if races else RACE_CALENDAR.get(year)
    race = race_list[0] if race_list else "Bahrain"

    logging.info(
        f"Warm-up pass: reading {year} {race} grid to seed the drivers table..."
    )
    grid = fetch_grid_from_fastf1(year, race)
    if not grid:
        raise RuntimeError(
            f"Warm-up pass found no drivers for {year} {race} — check the year "
            "and race names."
        )
    inserted = seed_drivers_from_grid(grid)
    logging.info(
        f"Warm-up pass seeded {inserted} new driver row(s) from {year} {race}."
    )
    return fetch_drivers_from_db()


def _resolve_or_warmup(tokens: list[str], db_drivers: list[dict],
                       seasons: list[int], races: list[str] | None) -> list[dict]:
    """Resolve every requested driver, running the warm-up pass if needed.

    Runs the warm-up only when the drivers table is empty or some requested
    driver is not in it.  Prints an error and exits when a token still cannot
    be resolved after the warm-up (the driver did not race in the reference
    session — e.g. a rookie from a later year than the first requested year).
    """
    resolved, unresolved = _resolve_driver_tokens(tokens, db_drivers)
    if unresolved:
        if not db_drivers:
            logging.info("The drivers table is empty — running the warm-up pass first.")
        else:
            logging.info(
                f"Driver(s) not in the drivers table ({', '.join(unresolved)}) — "
                "running the warm-up pass to discover the grid."
            )
        db_drivers = _warmup_seed(seasons, races)
        resolved, unresolved = _resolve_driver_tokens(tokens, db_drivers)
        if unresolved:
            year0 = seasons[0]
            race0 = races[0] if races else RACE_CALENDAR.get(year0, ["?"])[0]
            valid = sorted({d["driver_code"] for d in db_drivers})
            print(
                f"ERROR: Could not resolve driver(s): {', '.join(unresolved)}.\n"
                f"They are not in the drivers table or in the reference grid "
                f"({year0} {race0}).\n"
                f"Valid codes: {', '.join(valid)}\n"
                "Tip: pick a first year/race that includes these drivers."
            )
            sys.exit(1)
    return resolved


# ---------------------------------------------------------------------------
# Interactive prompt (identical resolution logic to import_f1_race)
# ---------------------------------------------------------------------------

def _interactive_prompt() -> dict:
    print("=" * 60)
    print("BATCH F1 DATASET INGESTION — INTERACTIVE MODE")
    print("=" * 60)

    # Drivers (one or more).  Codes are the stable identity across seasons;
    # numbers resolve against the DB ids.  Codes not yet in the DB are
    # discovered automatically via the warm-up pass.
    db_drivers = fetch_drivers_from_db()
    if db_drivers:
        print_driver_menu(db_drivers)
        drivers_raw = input(
            "\nEnter 3-letter codes separated by spaces (e.g. VER LEC HAM), "
            "or numbers.  Codes not yet in the DB are discovered automatically: "
        ).strip()
    else:
        print("\n  (The drivers table is empty — no menu to show; the grid is")
        print("   discovered automatically from the first race.)")
        drivers_raw = input(
            "\nEnter 3-letter codes separated by spaces (e.g. VER LEC HAM): "
        ).strip()
    if not drivers_raw:
        print("ERROR: At least one driver selection is required.")
        sys.exit(1)
    driver_tokens = drivers_raw.split()

    # Years
    years_raw = input("\nEnter year(s) separated by spaces [default: 2024]: ").strip()
    if years_raw:
        seasons = [int(y) for y in years_raw.split() if y.isdigit()]
        if not seasons:
            print("ERROR: No valid years entered.")
            sys.exit(1)
    else:
        seasons = [2024]

    # Races (optional — blank → full calendar)
    races: list[str] | None = None
    use_calendar = input("\nImport the full season calendar? (Y/n) [default: Y]: ").strip().lower()
    if use_calendar in ("n", "no"):
        races = []
        for year in seasons:
            calendar = print_track_menu(year)
            while True:
                raw = input(
                    "Enter race numbers separated by spaces (or race names), "
                    "blank line to finish: "
                ).strip()
                if not raw:
                    break
                for token in raw.split():
                    try:
                        races.append(resolve_race_input(token, calendar))
                    except ValueError as exc:
                        print(f"  Warning: {exc}")
        if not races:
            print("ERROR: No valid races selected.")
            sys.exit(1)

    # Session type
    session_raw = input("\nSession type  R=Race  Q=Qualifying  FP1/FP2/FP3  [default: R]: ").strip()
    session_type = session_raw if session_raw else "R"

    # Duplicates
    dup_raw = input("Allow duplicate import if session already exists? (y/N): ").strip().lower()
    allow_duplicate = dup_raw in ("y", "yes")

    # Resolve drivers now — the warm-up pass runs if any code is unknown.
    resolved = _resolve_or_warmup(driver_tokens, db_drivers, seasons, races)

    print("\n  Selected drivers:")
    for d in resolved:
        print(f"    #{d['driver_id']} - {d['driver_name']} ({d['driver_code']})")

    return dict(
        seasons=seasons,
        races=races,
        drivers=[(d["driver_id"], d["driver_code"]) for d in resolved],
        session_type=session_type,
        allow_duplicate=allow_duplicate,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch F1 Dataset Ingestion (multiple drivers, multiple races/years)"
    )
    parser.add_argument("--years",    type=int, nargs="+", default=None, help="Season years (e.g. 2023 2024)")
    parser.add_argument("--races",    type=str, nargs="+", default=None, help="Race names (e.g. Bahrain Monaco)")
    parser.add_argument("--driver",   type=str, default=None,            help="Driver F1 number or 3-letter code (single)")
    parser.add_argument("--drivers",  type=str, nargs="+", default=None, help="Driver F1 numbers or 3-letter codes (e.g. VER LEC 44)")
    parser.add_argument("--session-type", type=str, default=None,        help="Session type: R, Q, FP1, FP2, FP3")
    parser.add_argument("--allow-duplicate", action="store_true",        help="Re-import even if session exists")
    parser.add_argument("--interactive", "-i", action="store_true",      help="Force interactive prompt")
    args = parser.parse_args()

    use_interactive = (
        len(sys.argv) == 1
        or args.interactive
        or args.years is None
        or (args.driver is None and args.drivers is None)
    )

    if use_interactive:
        params = _interactive_prompt()
    else:
        tokens = list(args.drivers or [])
        if args.driver:
            tokens.append(args.driver)
        db_drivers = fetch_drivers_from_db()
        resolved = _resolve_or_warmup(tokens, db_drivers, args.years, args.races)
        params = dict(
            seasons=args.years,
            races=args.races,
            drivers=[(d["driver_id"], d["driver_code"]) for d in resolved],
            session_type=args.session_type or "R",
            allow_duplicate=args.allow_duplicate,
        )

    import_dataset(**params)