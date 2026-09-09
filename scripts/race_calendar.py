"""Official FIA race calendars 2020-2026 + DB coverage annotation.

Single source of truth for the dashboard's calendar widget.  The
/api/calendar endpoint serves these rounds annotated with what is
actually in the database (race session counts, the DB circuit name(s)
matching each round, drivers present, best lap, max lap), so the
calendar doubles as a data-coverage map for the P0/P1 overtake tooling
and a launcher that loads a round's sessions straight into the
dashboard.

Round fields:
  round   — calendar round number
  country — display name (duplicate venues keep their GP name, e.g. the
            Styria/Austria double header at the Red Bull Ring)
  code    — flag PNG name under scripts/dashboard/static/flags/
  dates   — weekend dates for display
  track   — |-separated substrings matched against the DB's track_name,
            so a round highlights when a session's track matches it
            (e.g. 'imola|enzo' matches 'Autodromo Internazionale Enzo e
            Dino Ferrari')
"""

CALENDARS = {
2020: [
    {"round": 1, "country": "Austria", "code": "at", "dates": "03 - 05 Jul", "track": "red bull"},
    {"round": 2, "country": "Styria", "code": "at", "dates": "10 - 12 Jul", "track": "red bull"},
    {"round": 3, "country": "Hungary", "code": "hu", "dates": "17 - 19 Jul", "track": "hungaroring"},
    {"round": 4, "country": "Great Britain", "code": "gb", "dates": "31 Jul - 02 Aug", "track": "silverstone"},
    {"round": 5, "country": "70th Anniversary", "code": "gb", "dates": "07 - 09 Aug", "track": "silverstone"},
    {"round": 6, "country": "Spain", "code": "es", "dates": "14 - 16 Aug", "track": "barcelona"},
    {"round": 7, "country": "Belgium", "code": "be", "dates": "28 - 30 Aug", "track": "spa"},
    {"round": 8, "country": "Italy", "code": "it", "dates": "04 - 06 Sep", "track": "monza"},
    {"round": 9, "country": "Tuscany", "code": "it", "dates": "11 - 13 Sep", "track": "mugello"},
    {"round": 10, "country": "Russia", "code": "ru", "dates": "25 - 27 Sep", "track": "sochi"},
    {"round": 11, "country": "Eifel", "code": "de", "dates": "09 - 11 Oct", "track": "nurburg|nürburg"},
    {"round": 12, "country": "Portugal", "code": "pt", "dates": "23 - 25 Oct", "track": "algarve"},
    {"round": 13, "country": "Emilia-Romagna", "code": "it", "dates": "31 Oct - 01 Nov", "track": "imola|enzo"},
    {"round": 14, "country": "Turkey", "code": "tr", "dates": "13 - 15 Nov", "track": "istanbul"},
    {"round": 15, "country": "Bahrain", "code": "bh", "dates": "27 - 29 Nov", "track": "bahrain"},
    {"round": 16, "country": "Sakhir", "code": "bh", "dates": "04 - 06 Dec", "track": "bahrain"},
    {"round": 17, "country": "Abu Dhabi", "code": "ae", "dates": "11 - 13 Dec", "track": "yas"},
],
2021: [
    {"round": 1, "country": "Bahrain", "code": "bh", "dates": "26 - 28 Mar", "track": "bahrain"},
    {"round": 2, "country": "Emilia-Romagna", "code": "it", "dates": "16 - 18 Apr", "track": "imola|enzo"},
    {"round": 3, "country": "Portugal", "code": "pt", "dates": "30 Apr - 02 May", "track": "algarve"},
    {"round": 4, "country": "Spain", "code": "es", "dates": "07 - 09 May", "track": "barcelona"},
    {"round": 5, "country": "Monaco", "code": "mc", "dates": "20 - 23 May", "track": "monaco"},
    {"round": 6, "country": "Azerbaijan", "code": "az", "dates": "04 - 06 Jun", "track": "baku"},
    {"round": 7, "country": "France", "code": "fr", "dates": "18 - 20 Jun", "track": "castellet"},
    {"round": 8, "country": "Styria", "code": "at", "dates": "25 - 27 Jun", "track": "red bull"},
    {"round": 9, "country": "Austria", "code": "at", "dates": "02 - 04 Jul", "track": "red bull"},
    {"round": 10, "country": "Great Britain", "code": "gb", "dates": "16 - 18 Jul", "track": "silverstone"},
    {"round": 11, "country": "Hungary", "code": "hu", "dates": "30 Jul - 01 Aug", "track": "hungaroring"},
    {"round": 12, "country": "Belgium", "code": "be", "dates": "27 - 29 Aug", "track": "spa"},
    {"round": 13, "country": "Netherlands", "code": "nl", "dates": "03 - 05 Sep", "track": "zandvoort"},
    {"round": 14, "country": "Italy", "code": "it", "dates": "10 - 12 Sep", "track": "monza"},
    {"round": 15, "country": "Russia", "code": "ru", "dates": "24 - 26 Sep", "track": "sochi"},
    {"round": 16, "country": "Turkey", "code": "tr", "dates": "08 - 10 Oct", "track": "istanbul"},
    {"round": 17, "country": "United States", "code": "us", "dates": "22 - 24 Oct", "track": "austin"},
    {"round": 18, "country": "Mexico", "code": "mx", "dates": "05 - 07 Nov", "track": "hermanos|rodriguez"},
    {"round": 19, "country": "Brazil", "code": "br", "dates": "12 - 14 Nov", "track": "interlagos|sao paulo|são paulo"},
    {"round": 20, "country": "Qatar", "code": "qa", "dates": "19 - 21 Nov", "track": "lusail"},
    {"round": 21, "country": "Saudi Arabia", "code": "sa", "dates": "03 - 05 Dec", "track": "jeddah"},
    {"round": 22, "country": "Abu Dhabi", "code": "ae", "dates": "10 - 12 Dec", "track": "yas"},
],
2022: [
    {"round": 1, "country": "Bahrain", "code": "bh", "dates": "18 - 20 Mar", "track": "bahrain"},
    {"round": 2, "country": "Saudi Arabia", "code": "sa", "dates": "25 - 27 Mar", "track": "jeddah"},
    {"round": 3, "country": "Australia", "code": "au", "dates": "08 - 10 Apr", "track": "albert park"},
    {"round": 4, "country": "Emilia-Romagna", "code": "it", "dates": "22 - 24 Apr", "track": "imola|enzo"},
    {"round": 5, "country": "Miami", "code": "us", "dates": "06 - 08 May", "track": "miami"},
    {"round": 6, "country": "Spain", "code": "es", "dates": "20 - 22 May", "track": "barcelona"},
    {"round": 7, "country": "Monaco", "code": "mc", "dates": "27 - 29 May", "track": "monaco"},
    {"round": 8, "country": "Azerbaijan", "code": "az", "dates": "10 - 12 Jun", "track": "baku"},
    {"round": 9, "country": "Canada", "code": "ca", "dates": "17 - 19 Jun", "track": "gilles villeneuve"},
    {"round": 10, "country": "Great Britain", "code": "gb", "dates": "01 - 03 Jul", "track": "silverstone"},
    {"round": 11, "country": "Austria", "code": "at", "dates": "08 - 10 Jul", "track": "red bull"},
    {"round": 12, "country": "France", "code": "fr", "dates": "22 - 24 Jul", "track": "castellet"},
    {"round": 13, "country": "Hungary", "code": "hu", "dates": "29 - 31 Jul", "track": "hungaroring"},
    {"round": 14, "country": "Belgium", "code": "be", "dates": "26 - 28 Aug", "track": "spa"},
    {"round": 15, "country": "Netherlands", "code": "nl", "dates": "02 - 04 Sep", "track": "zandvoort"},
    {"round": 16, "country": "Italy", "code": "it", "dates": "09 - 11 Sep", "track": "monza"},
    {"round": 17, "country": "Singapore", "code": "sg", "dates": "30 Sep - 02 Oct", "track": "marina bay"},
    {"round": 18, "country": "Japan", "code": "jp", "dates": "07 - 09 Oct", "track": "suzuka"},
    {"round": 19, "country": "United States", "code": "us", "dates": "21 - 23 Oct", "track": "austin"},
    {"round": 20, "country": "Mexico", "code": "mx", "dates": "28 - 30 Oct", "track": "hermanos|rodriguez"},
    {"round": 21, "country": "Brazil", "code": "br", "dates": "11 - 13 Nov", "track": "interlagos|sao paulo|são paulo"},
    {"round": 22, "country": "Abu Dhabi", "code": "ae", "dates": "18 - 20 Nov", "track": "yas"},
],
2023: [
    {"round": 1, "country": "Bahrain", "code": "bh", "dates": "03 - 05 Mar", "track": "bahrain"},
    {"round": 2, "country": "Saudi Arabia", "code": "sa", "dates": "17 - 19 Mar", "track": "jeddah"},
    {"round": 3, "country": "Australia", "code": "au", "dates": "31 Mar - 02 Apr", "track": "albert park"},
    {"round": 4, "country": "Azerbaijan", "code": "az", "dates": "28 - 30 Apr", "track": "baku"},
    {"round": 5, "country": "Miami", "code": "us", "dates": "05 - 07 May", "track": "miami"},
    {"round": 6, "country": "Monaco", "code": "mc", "dates": "26 - 28 May", "track": "monaco"},
    {"round": 7, "country": "Spain", "code": "es", "dates": "02 - 04 Jun", "track": "barcelona"},
    {"round": 8, "country": "Canada", "code": "ca", "dates": "16 - 18 Jun", "track": "gilles villeneuve"},
    {"round": 9, "country": "Austria", "code": "at", "dates": "30 Jun - 02 Jul", "track": "red bull"},
    {"round": 10, "country": "Great Britain", "code": "gb", "dates": "07 - 09 Jul", "track": "silverstone"},
    {"round": 11, "country": "Hungary", "code": "hu", "dates": "21 - 23 Jul", "track": "hungaroring"},
    {"round": 12, "country": "Belgium", "code": "be", "dates": "28 - 30 Jul", "track": "spa"},
    {"round": 13, "country": "Netherlands", "code": "nl", "dates": "25 - 27 Aug", "track": "zandvoort"},
    {"round": 14, "country": "Italy", "code": "it", "dates": "01 - 03 Sep", "track": "monza"},
    {"round": 15, "country": "Singapore", "code": "sg", "dates": "15 - 17 Sep", "track": "marina bay"},
    {"round": 16, "country": "Japan", "code": "jp", "dates": "22 - 24 Sep", "track": "suzuka"},
    {"round": 17, "country": "Qatar", "code": "qa", "dates": "06 - 08 Oct", "track": "lusail"},
    {"round": 18, "country": "United States", "code": "us", "dates": "20 - 22 Oct", "track": "austin"},
    {"round": 19, "country": "Mexico", "code": "mx", "dates": "27 - 29 Oct", "track": "hermanos|rodriguez"},
    {"round": 20, "country": "Brazil", "code": "br", "dates": "03 - 05 Nov", "track": "interlagos|sao paulo|são paulo"},
    {"round": 21, "country": "Las Vegas", "code": "us", "dates": "16 - 18 Nov", "track": "las vegas"},
    {"round": 22, "country": "Abu Dhabi", "code": "ae", "dates": "24 - 26 Nov", "track": "yas"},
],
2024: [
    {"round": 1, "country": "Bahrain", "code": "bh", "dates": "29 Feb - 02 Mar", "track": "bahrain"},
    {"round": 2, "country": "Saudi Arabia", "code": "sa", "dates": "07 - 09 Mar", "track": "jeddah"},
    {"round": 3, "country": "Australia", "code": "au", "dates": "22 - 24 Mar", "track": "albert park"},
    {"round": 4, "country": "Japan", "code": "jp", "dates": "05 - 07 Apr", "track": "suzuka"},
    {"round": 5, "country": "China", "code": "cn", "dates": "19 - 21 Apr", "track": "shanghai"},
    {"round": 6, "country": "Miami", "code": "us", "dates": "03 - 05 May", "track": "miami"},
    {"round": 7, "country": "Emilia-Romagna", "code": "it", "dates": "17 - 19 May", "track": "imola|enzo"},
    {"round": 8, "country": "Monaco", "code": "mc", "dates": "24 - 26 May", "track": "monaco"},
    {"round": 9, "country": "Canada", "code": "ca", "dates": "07 - 09 Jun", "track": "gilles villeneuve"},
    {"round": 10, "country": "Spain", "code": "es", "dates": "21 - 23 Jun", "track": "barcelona"},
    {"round": 11, "country": "Austria", "code": "at", "dates": "28 - 30 Jun", "track": "red bull"},
    {"round": 12, "country": "Great Britain", "code": "gb", "dates": "05 - 07 Jul", "track": "silverstone"},
    {"round": 13, "country": "Hungary", "code": "hu", "dates": "19 - 21 Jul", "track": "hungaroring"},
    {"round": 14, "country": "Belgium", "code": "be", "dates": "26 - 28 Jul", "track": "spa"},
    {"round": 15, "country": "Netherlands", "code": "nl", "dates": "23 - 25 Aug", "track": "zandvoort"},
    {"round": 16, "country": "Italy", "code": "it", "dates": "30 Aug - 01 Sep", "track": "monza"},
    {"round": 17, "country": "Azerbaijan", "code": "az", "dates": "13 - 15 Sep", "track": "baku"},
    {"round": 18, "country": "Singapore", "code": "sg", "dates": "20 - 22 Sep", "track": "marina bay"},
    {"round": 19, "country": "United States", "code": "us", "dates": "18 - 20 Oct", "track": "austin"},
    {"round": 20, "country": "Mexico", "code": "mx", "dates": "25 - 27 Oct", "track": "hermanos|rodriguez"},
    {"round": 21, "country": "Brazil", "code": "br", "dates": "01 - 03 Nov", "track": "interlagos|sao paulo|são paulo"},
    {"round": 22, "country": "Las Vegas", "code": "us", "dates": "21 - 23 Nov", "track": "las vegas"},
    {"round": 23, "country": "Qatar", "code": "qa", "dates": "29 Nov - 01 Dec", "track": "lusail"},
    {"round": 24, "country": "Abu Dhabi", "code": "ae", "dates": "06 - 08 Dec", "track": "yas"},
],
2025: [
    {"round": 1, "country": "Australia", "code": "au", "dates": "14 - 16 Mar", "track": "albert park"},
    {"round": 2, "country": "China", "code": "cn", "dates": "21 - 23 Mar", "track": "shanghai"},
    {"round": 3, "country": "Japan", "code": "jp", "dates": "04 - 06 Apr", "track": "suzuka"},
    {"round": 4, "country": "Bahrain", "code": "bh", "dates": "11 - 13 Apr", "track": "bahrain"},
    {"round": 5, "country": "Saudi Arabia", "code": "sa", "dates": "18 - 20 Apr", "track": "jeddah"},
    {"round": 6, "country": "Miami", "code": "us", "dates": "02 - 04 May", "track": "miami"},
    {"round": 7, "country": "Emilia-Romagna", "code": "it", "dates": "16 - 18 May", "track": "imola|enzo"},
    {"round": 8, "country": "Monaco", "code": "mc", "dates": "23 - 25 May", "track": "monaco"},
    {"round": 9, "country": "Spain", "code": "es", "dates": "30 May - 01 Jun", "track": "barcelona"},
    {"round": 10, "country": "Canada", "code": "ca", "dates": "13 - 15 Jun", "track": "gilles villeneuve"},
    {"round": 11, "country": "Austria", "code": "at", "dates": "27 - 29 Jun", "track": "red bull"},
    {"round": 12, "country": "Great Britain", "code": "gb", "dates": "04 - 06 Jul", "track": "silverstone"},
    {"round": 13, "country": "Belgium", "code": "be", "dates": "25 - 27 Jul", "track": "spa"},
    {"round": 14, "country": "Hungary", "code": "hu", "dates": "01 - 03 Aug", "track": "hungaroring"},
    {"round": 15, "country": "Netherlands", "code": "nl", "dates": "29 - 31 Aug", "track": "zandvoort"},
    {"round": 16, "country": "Italy", "code": "it", "dates": "05 - 07 Sep", "track": "monza"},
    {"round": 17, "country": "Azerbaijan", "code": "az", "dates": "19 - 21 Sep", "track": "baku"},
    {"round": 18, "country": "Singapore", "code": "sg", "dates": "03 - 05 Oct", "track": "marina bay"},
    {"round": 19, "country": "United States", "code": "us", "dates": "17 - 19 Oct", "track": "austin"},
    {"round": 20, "country": "Mexico", "code": "mx", "dates": "24 - 26 Oct", "track": "hermanos|rodriguez"},
    {"round": 21, "country": "Brazil", "code": "br", "dates": "07 - 09 Nov", "track": "interlagos|sao paulo|são paulo"},
    {"round": 22, "country": "Las Vegas", "code": "us", "dates": "20 - 22 Nov", "track": "las vegas"},
    {"round": 23, "country": "Qatar", "code": "qa", "dates": "28 - 30 Nov", "track": "lusail"},
    {"round": 24, "country": "Abu Dhabi", "code": "ae", "dates": "05 - 07 Dec", "track": "yas"},
],
2026: [
    {"round": 1, "country": "Australia", "code": "au", "dates": "06 - 08 Mar", "track": "albert park"},
    {"round": 2, "country": "China", "code": "cn", "dates": "13 - 15 Mar", "track": "shanghai"},
    {"round": 3, "country": "Japan", "code": "jp", "dates": "27 - 29 Mar", "track": "suzuka"},
    {"round": 4, "country": "Miami", "code": "us", "dates": "01 - 03 May", "track": "miami"},
    {"round": 5, "country": "Canada", "code": "ca", "dates": "22 - 24 May", "track": "gilles villeneuve"},
    {"round": 6, "country": "Monaco", "code": "mc", "dates": "05 - 07 Jun", "track": "monaco"},
    {"round": 7, "country": "Barcelona-Catalunya", "code": "es", "dates": "12 - 14 Jun", "track": "barcelona|barcelona-catalunya"},
    {"round": 8, "country": "Austria", "code": "at", "dates": "26 - 28 Jun", "track": "red bull"},
    {"round": 9, "country": "Great Britain", "code": "gb", "dates": "03 - 05 Jul", "track": "silverstone"},
    {"round": 10, "country": "Belgium", "code": "be", "dates": "17 - 19 Jul", "track": "spa"},
    {"round": 11, "country": "Hungary", "code": "hu", "dates": "24 - 26 Jul", "track": "hungaroring"},
    {"round": 12, "country": "Netherlands", "code": "nl", "dates": "21 - 23 Aug", "track": "zandvoort"},
    {"round": 13, "country": "Italy", "code": "it", "dates": "04 - 06 Sep", "track": "monza"},
    {"round": 14, "country": "Spain", "code": "es", "dates": "11 - 13 Sep", "track": "madrid"},
    {"round": 15, "country": "Azerbaijan", "code": "az", "dates": "24 - 26 Sep", "track": "baku"},
    {"round": 16, "country": "Bahrain", "code": "bh", "dates": "02 - 04 Oct", "track": "bahrain"},
    {"round": 17, "country": "Singapore", "code": "sg", "dates": "09 - 11 Oct", "track": "marina bay"},
    {"round": 18, "country": "United States", "code": "us", "dates": "23 - 25 Oct", "track": "austin"},
    {"round": 19, "country": "Mexico", "code": "mx", "dates": "30 Oct - 01 Nov", "track": "hermanos|rodriguez"},
    {"round": 20, "country": "Brazil", "code": "br", "dates": "06 - 08 Nov", "track": "interlagos|sao paulo|são paulo"},
    {"round": 21, "country": "Las Vegas", "code": "us", "dates": "19 - 21 Nov", "track": "las vegas"},
    {"round": 22, "country": "Qatar", "code": "qa", "dates": "27 - 29 Nov", "track": "lusail"},
    {"round": 23, "country": "Abu Dhabi", "code": "ae", "dates": "04 - 06 Dec", "track": "yas"},
],
}

YEARS = sorted(CALENDARS)


def rounds_for_year(year):
    """Calendar rounds for a year (empty list when the year is absent)."""
    return CALENDARS.get(int(year), [])


def match_track(track_name, track_key):
    """True when a DB track_name matches a round's |-separated track key.

    Mirrors the frontend highlight logic: either side may be a substring
    of the other ('imola|enzo' matches 'Autodromo Internazionale Enzo e
    Dino Ferrari' via 'enzo').
    """
    t = str(track_name).strip().lower()
    return any(p and (p in t or t in p)
               for p in str(track_key).lower().split("|"))


def annotate_with_db(year, conn=None):
    """Merge DB coverage into one season's calendar rounds.

    For each round, every DB track that matches its track key is pulled
    in, then aggregated into per-round coverage: db_tracks (the distinct
    DB circuit names that matched), race_sessions (distinct full-length
    Race sessions), drivers present, best lap, max lap and covered flag.
    """
    if conn is None:
        from config import get_db_connection
        conn = get_db_connection()
        close_after = True
    else:
        close_after = False
    cur = conn.cursor(dictionary=True)
    try:
        cur.execute("""
            SELECT s.track_name, s.session_id, s.session_type,
                   d.driver_code, MIN(l.lap_time_ms) AS best_lap_ms,
                   MAX(l.lap_number) AS max_lap
            FROM sessions s
            JOIN drivers d ON s.driver_id = d.driver_id
            JOIN laps l ON l.session_id = s.session_id AND l.lap_time_ms > 0
            WHERE YEAR(s.date) = %s
            GROUP BY s.track_name, s.session_id, s.session_type, d.driver_code
        """, (int(year),))
        by_track = {}
        for r in cur.fetchall():
            by_track.setdefault(str(r['track_name']).strip().lower(),
                                []).append(r)

        out = []
        for rnd in CALENDARS.get(int(year), []):
            rows = []
            for tname, trows in by_track.items():
                if match_track(tname, rnd['track']):
                    rows.extend(trows)
            race_sessions = len({r['session_id'] for r in rows
                                 if (r['session_type'] or '').lower() == 'race'})
            db_tracks = sorted({r['track_name'] for r in rows})
            drivers = sorted({r['driver_code'] for r in rows})
            best_ms = min((r['best_lap_ms'] for r in rows), default=None)
            out.append({
                "round": rnd['round'],
                "country": rnd['country'],
                "code": rnd['code'],
                "dates": rnd['dates'],
                "track": rnd['track'],
                "covered": len(rows) > 0,
                "race_sessions": race_sessions,
                "db_tracks": db_tracks,
                "drivers": drivers,
                "best_lap_s": (round(best_ms / 1000.0, 3)
                               if best_ms else None),
                "max_lap": max((r['max_lap'] or 0 for r in rows), default=0),
            })
        return out
    finally:
        cur.close()
        if close_after:
            conn.close()