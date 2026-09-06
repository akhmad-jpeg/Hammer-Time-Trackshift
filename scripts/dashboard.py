from flask import Flask, render_template, jsonify, request
import os
import sys
import joblib
import json
import traceback
import datetime
import time
from pathlib import Path
from fuel_estimation import estimate_fuel_load
from config import get_db_connection
from stint_analysis import detrend_laps
from feature_pipeline import (
    construct_prediction_input,
    covered_tracks,
    covered_tyres,
    validate_model_inputs,
)
import driver_comparison
import overtake_inference  # import-safe; models load lazily below
import race_calendar  # official calendars 2020-2026 (single source of truth)

# Battery capacity/floor used to express the synthetic ERS trace as a
# percentage and to judge feasibility.  Single source of truth lives in the
# simulator that produces the rows (module constants mirror the default 2026
# PU spec; era-specific numbers are resolved per session via spec_for_year).
from energy_simulator import (
    BATTERY_CAPACITY_MJ,
    BATTERY_MIN_MJ,
    DEFAULT_START_SOC_MJ,
    RECOVER_FLOW_CAP_MJ,
    DEPLOY_RATE_MJ_S,
    RECOVER_RATE_MJ_S,
    PU_SPECS,
    spec_for_year,
    MODES,
    DEPLOY_PACE_S_PER_MJ,
    LIMITED_LAP_PACE_EFFECTIVENESS,
    project_energy_trace,
    intra_lap_battery_curve,
    _speed_drop_regen as regen_from_speed_samples,
)

app = Flask(__name__)

# ── ML model (loaded once at startup) ───────────────────────
# Model artifacts always live in <project root>/ml_models so the dashboard
# finds them regardless of the working directory it is launched from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / 'ml_models' / 'best_model.pkl'
FEATURES_PATH = PROJECT_ROOT / 'ml_models' / 'feature_names.pkl'

model = None
feature_names = []

# Fuel-burn rate: the pace effect the model bakes into tyre_age that is fuel
# (and track evolution), not tyre wear.  The strategy advisor detrends every
# scenario's predictions by this rate so stay-out vs pit comparisons are made
# on equal fuel footing — both scenarios burn the same fuel over the
# remaining laps, so crediting the stay-out scenario's higher ages with it
# would bias the comparison by |rate| * laps_rem * cur_age in its favour
# (the old advisor therefore ALWAYS said "Stay Out", regardless of tyre
# age).  Written at training time as min(0, tyre_age coefficient): the
# cleaned dataset shows no net wear after warm-up laps are excluded, so the
# whole negative age slope is treated as fuel; a positive slope (wear
# dominating) would be clamped to 0 and left fully in the comparison.
MODEL_INFO_PATH = PROJECT_ROOT / 'ml_models' / 'model_info.json'
fuel_burn_rate = 0.0

# Per-track energy-to-time profile (energy_pace.json, written at training
# time): measured full-throttle share -> pace_s_per_mj per track, and the
# measured mean S1:S2:S3 split of lap time.  When the file is absent (models
# never retrained) the endpoints fall back to the flat DEPLOY_PACE_S_PER_MJ
# constant and equal sector thirds.
ENERGY_PACE_PATH = PROJECT_ROOT / 'ml_models' / 'energy_pace.json'
energy_pace = {}
if ENERGY_PACE_PATH.exists():
    try:
        energy_pace = json.loads(ENERGY_PACE_PATH.read_text(encoding='utf-8'))
        n_tracks = len(energy_pace.get('per_track', {}))
        print(f"[INFO] Energy pace profile: {n_tracks} tracks "
              f"(measured s/MJ + sector splits)")
    except Exception:
        traceback.print_exc()
        energy_pace = {}


def track_pace_s_per_mj(track_name):
    """Measured per-track energy-to-time conversion (s/MJ) for a track.

    Falls back to the flat calibrated constant when the track is not in the
    training profile.
    """
    try:
        pt = energy_pace['per_track'].get(str(track_name))
        if pt and pt.get('pace_s_per_mj'):
            return float(pt['pace_s_per_mj'])
    except Exception:
        pass
    return DEPLOY_PACE_S_PER_MJ


def track_sector_profile(track_name):
    """Measured sector structure for a track from the training artifact.

    Returns (shares, source) where shares sums to 1.  source is
    'measured_fastf1' when real sector times were available at training,
    else 'proxy_thirds'.
    """
    try:
        pt = energy_pace['per_track'].get(str(track_name))
        sh = pt.get('sector_time_share') if pt else None
        src = pt.get('sector_share_source') if pt else 'proxy_thirds'
        if sh and len(sh) == 3 and abs(sum(sh) - 1.0) < 1e-3:
            return [float(x) for x in sh], src
    except Exception:
        pass
    return [1.0 / 3.0] * 3, 'proxy_thirds'


def track_sector_pace(track_name):
    """Per-sector s/MJ values for a track (falls back to the track pace)."""
    try:
        pt = energy_pace['per_track'].get(str(track_name))
        if pt and pt.get('sector_pace_s_per_mj'):
            return [float(x) for x in pt['sector_pace_s_per_mj']]
    except Exception:
        pass
    p = track_pace_s_per_mj(track_name)
    return [p, p, p]

if MODEL_PATH.exists() and FEATURES_PATH.exists():
    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    print(f"[INFO] Model loaded: {type(model).__name__} | {len(feature_names)} features")
    print(f"[INFO] Covers {len(covered_tracks(feature_names))} tracks, "
          f"{len(covered_tyres(feature_names))} tyres")
    if MODEL_INFO_PATH.exists():
        try:
            info = json.loads(MODEL_INFO_PATH.read_text(encoding='utf-8'))
            fuel_burn_rate = min(0.0, float(info.get('fuel_burn_rate', 0.0)))
            print(f"[INFO] Fuel burn rate (detrended from advisor comparisons): "
                  f"{fuel_burn_rate:+.4f} s/lap")
        except Exception:
            traceback.print_exc()
else:
    print("[WARNING] Model not found — run scripts/ml_lap_predictions.py first")

# Measured per-compound wear slopes exported by the training run
# (model_info.json -> tyre_wear_slopes_per_compound): the fitted tyre_age x
# compound slopes, which mix fuel burn (a negative age effect) with real
# wear.  For the advisor's RELATIVE comparisons the fuel part cancels — every
# scenario burns the same fuel over the same remaining laps — so the raw
# slopes are the right magnitudes to drive the wear term, floored per
# compound (tyres never improve with age) and capped.  Per-track
# differentiation comes from the researched abrasion map: the model's raw
# PER-TRACK age slopes were measured and rejected as too noisy — a few
# seasons of two drivers make Bahrain's Soft slope come out negative, which
# would make the most abrasive circuit wear tyres slowest.
MEASURED_COMPOUND_SLOPE = {}
if MODEL_INFO_PATH.exists():
    try:
        _info = json.loads(MODEL_INFO_PATH.read_text(encoding='utf-8'))
        MEASURED_COMPOUND_SLOPE = _info.get('tyre_wear_slopes_per_compound', {}) or {}
        print(f"[INFO] Measured per-compound wear: "
              + ", ".join(f"{k}={v:+.4f}" for k, v in sorted(MEASURED_COMPOUND_SLOPE.items())))
    except Exception:
        traceback.print_exc()
        MEASURED_COMPOUND_SLOPE = {}

# Physical floors for the measured slopes: the regression can fit ~zero or
# negative slopes for compounds whose fuel masking is strong (Medium comes
# out ~0.001 s/lap), but no tyre wears nothing.  Floors keep a fresh Medium
# stint realistically expensive instead of nearly free.
MEASURED_SLOPE_FLOOR = {'Soft': 0.05, 'Medium': 0.030, 'Hard': 0.030,
                        'Intermediate': 0.025, 'Wet': 0.020}
MEASURED_SLOPE_CAP = 0.25       # s/lap — anything above is an artefact


# PAGE ROUTES
@app.route('/')
def index():
    return render_template('dashboard.html')


# SESSION / TELEMETRY API
@app.route('/api/sessions')
def get_sessions():
    conn = None
    cursor = None
    try:
        # Pagination: ?limit=&offset= (defaults keep the historical 50-row cap;
        # limit is clamped to [1, 500]).  Optional ?driver=<CODE> filters to
        # one driver's sessions (used by the dashboard driver selector);
        # ?track=<calendar track key>&year=YYYY returns a calendar round's
        # sessions (the track key is matched against the DB the same way the
        # calendar coverage is built, so 'imola|enzo' finds any Imola race).
        limit = max(1, min(request.args.get('limit', default=50, type=int), 500))
        offset = max(0, request.args.get('offset', default=0, type=int))
        driver = request.args.get('driver', '').strip().upper()
        track = request.args.get('track', '').strip()
        year = request.args.get('year', type=int)
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        base_sql = """
            SELECT
                s.session_id,
                s.track_name,
                s.session_type,
                s.weather,
                s.date,
                d.driver_code,
                COUNT(l.lap_id) AS total_laps,
                MIN(CASE WHEN l.is_valid = 1 AND l.lap_time_ms > 0
                         THEN l.lap_time_ms END) / 1000 AS fastest_lap
            FROM sessions s
            LEFT JOIN drivers d ON s.driver_id = d.driver_id
            LEFT JOIN laps l ON s.session_id = l.session_id
        """
        where = []
        params = []
        if driver:
            where.append('d.driver_code = %s')
            params.append(driver)
        if year:
            where.append('YEAR(s.date) = %s')
            params.append(year)
        group_order = """
                GROUP BY s.session_id, s.track_name, s.session_type, s.weather,
                         s.date, d.driver_code
                ORDER BY s.date DESC, s.session_id DESC
        """
        where_sql = (' WHERE ' + ' AND '.join(where)) if where else ''
        if track:
            # A calendar track key rarely equals the stored track_name, so the
            # round's rows are pulled in full and matched in Python using the
            # same substring rule race_calendar uses to draw the calendar.
            cursor.execute(base_sql + where_sql + group_order, params)
            matched = [s for s in cursor.fetchall()
                       if race_calendar.match_track(str(s['track_name'] or ''),
                                                    track)]
            sessions = matched[offset:offset + limit]
        else:
            cursor.execute(base_sql + where_sql + group_order +
                           ' LIMIT %s OFFSET %s', params + [limit, offset])
            sessions = cursor.fetchall()

        for s in sessions:
            if s['date']:
                s['date'] = s['date'].strftime('%Y-%m-%d')
            s['fastest_lap'] = float(s['fastest_lap']) if s['fastest_lap'] else None

        return jsonify(sessions)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if cursor:
            try: cursor.close()
            except: pass
        if conn:
            try: conn.close()
            except: pass


@app.route('/api/session/<int:session_id>/laps')
def get_session_laps(session_id):
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT
                l.lap_number,
                l.lap_time_ms / 1000.0 AS lap_time,
                l.tyre_compound,
                l.tyre_age,
                l.fuel_load,
                l.is_valid,
                MAX(CASE WHEN se.event_type = 'PitStop' THEN 1 ELSE 0 END) AS has_pit_stop
            FROM laps l
            LEFT JOIN strategy_events se ON l.lap_id = se.lap_id
            WHERE l.session_id = %s AND l.lap_time_ms > 0
            GROUP BY l.lap_id, l.lap_number, l.lap_time_ms, l.tyre_compound, l.tyre_age, l.fuel_load, l.is_valid
            ORDER BY l.lap_number
        """, (session_id,))
        laps = cursor.fetchall()

        for lap in laps:
            lap['lap_time'] = float(lap['lap_time']) if lap['lap_time'] is not None else None
            lap['fuel_load'] = float(lap['fuel_load']) if lap['fuel_load'] is not None else 0.0

        return jsonify(laps)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if cursor:
            try: cursor.close()
            except: pass
        if conn:
            try: conn.close()
            except: pass


def _energy_trace_points(cursor, session_id):
    """Synthetic ERS battery-% tracking-line points for a session.

    Reads the per-lap SOC rows in race_state (written by the energy
    simulator) and reconstructs the battery *inside* each lap at telemetry
    resolution, so a single line drops as energy is deployed (full-throttle
    stretches) and climbs as it is harvested (deceleration), anchored to the
    stored per-lap start/end SOC.

    Returns one point per telemetry sample: {lap_number, fraction (0..1
    through the lap), x (lap + fraction, for a linear axis), battery_mj,
    battery_pct}.  Empty list = no trace exists yet for the session.
    """
    cursor.execute("""
        SELECT r.lap_number,
               r.energy_start_mj,
               r.energy_deployed_mj,
               r.energy_harvested_mj,
               r.energy_end_mj,
               l.lap_time_ms,
               t.speed,
               t.throttle,
               t.telemetry_id
        FROM race_state r
        JOIN laps l
          ON l.session_id = r.session_id AND l.lap_number = r.lap_number
        LEFT JOIN telemetry t ON t.lap_id = l.lap_id
        WHERE r.session_id = %s
        ORDER BY r.lap_number, t.telemetry_id
    """, (session_id,))
    rows = cursor.fetchall()
    capacity = BATTERY_CAPACITY_MJ if BATTERY_CAPACITY_MJ > 0 else 1.0

    # Group samples per lap, keeping per-lap policy totals and the lap
    # time (used to bound intra-lap SOC movement at the 120 kW rate).
    laps = {}
    for r in rows:
        lap = laps.setdefault(int(r['lap_number']), {
            "start": float(r['energy_start_mj'] or 0.0),
            "deployed": float(r['energy_deployed_mj'] or 0.0),
            "harvested": float(r['energy_harvested_mj'] or 0.0),
            "end": float(r['energy_end_mj'] or 0.0),
            "lap_time_s": float(r['lap_time_ms'] or 0.0) / 1000.0,
            "samples": [],
        })
        if r['telemetry_id'] is not None:
            lap['samples'].append(r)

    trace = []
    for lap_no in sorted(laps):
        lap = laps[lap_no]
        curve = intra_lap_battery_curve(
            lap['samples'], lap['start'], lap['deployed'],
            lap['harvested'], lap['end'],
            lap_time_s=lap['lap_time_s'])
        for p in curve:
            soc = max(0.0, min(capacity, float(p['soc_mj'])))
            trace.append({
                "lap_number": lap_no,
                "fraction": float(p['fraction']),
                "x": round(lap_no + float(p['fraction']), 4),
                "battery_mj": round(soc, 4),
                "battery_pct": round(soc / capacity * 100.0, 2),
            })
    return trace


@app.route('/api/session/<int:session_id>/energy')
def get_session_energy(session_id):
    """Synthetic ERS battery-% tracking line for one session (GET).

    An empty list means no trace exists yet -- run the simulator from the UI
    (POST /api/session/<id>/energy-simulate) or via:
        python scripts/energy_simulator.py --session <id> --mode balanced
    """
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        return jsonify(_energy_trace_points(cursor, session_id))
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if cursor:
            try: cursor.close()
            except: pass
        if conn:
            try: conn.close()
            except: pass


@app.route('/api/session/<int:session_id>/energy-simulate', methods=['POST'])
def simulate_session_energy_api(session_id):
    """Run the synthetic ERS simulator for a session under one mode (live).

    Same engine and persistence as the CLI backfill
    (energy_simulator.simulate_session_energy): the battery starts the race
    FULL, drains through the opening laps, then cycles in the soft 30-80%
    band (or the mode's own band for push / lift & coast).  The session's
    race_state rows are replaced, and the freshly written tracking-line
    points are returned so the caller can redraw the chart without a reload.
    """
    conn = None
    cursor = None
    t0 = time.perf_counter()
    try:
        body = request.get_json() or {}
        mode = str(body.get('mode', 'balanced')).lower()
        if mode not in MODES:
            return jsonify({
                "error": f"Unknown mode '{mode}' — use one of {sorted(MODES)}"
            }), 400

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute("""
            SELECT session_id, driver_id, track_name, date
            FROM sessions WHERE session_id = %s
        """, (session_id,))
        session = cursor.fetchone()
        if session is None:
            return jsonify({"error": f"Session {session_id} not found"}), 404
        if session['date'] is None:
            return jsonify({"error": "Session has no date — cannot pick a PU spec"}), 400

        spec_key = spec_for_year(session['date'].year)

        cursor.execute("""
            SELECT lap_id, lap_number, lap_time_ms
            FROM laps
            WHERE session_id = %s AND lap_time_ms > 0
            ORDER BY lap_number
        """, (session_id,))
        laps = cursor.fetchall()
        if not laps:
            return jsonify({"error": "Session has no timed laps to simulate"}), 400

        ids = [l['lap_id'] for l in laps]
        ph = ','.join(['%s'] * len(ids))
        cursor.execute(f"SELECT lap_id, speed FROM telemetry "
                       f"WHERE lap_id IN ({ph}) ORDER BY telemetry_id", ids)
        telem = {}
        for row in cursor.fetchall():
            telem.setdefault(row['lap_id'], []).append(row)

        # Per-lap regeneration from the speed traces + pace coupling, exactly
        # as the CLI backfill does it (see energy_simulator.py): expected
        # pace = median of each lap's ~+/-5 neighbours, so a lap faster than
        # its neighbours spends stored energy and a slower lap banks it.
        regen_list = [regen_from_speed_samples(telem.get(l['lap_id'], []), spec_key)
                      for l in laps]
        lap_times = [float(l['lap_time_ms']) for l in laps]
        n_lap = len(lap_times)
        half = 5
        baseline = []
        for i in range(n_lap):
            window = sorted(lap_times[max(0, i - half): min(n_lap, i + half + 1)])
            baseline.append(window[len(window) // 2])
        pace_dev = [(base - t) / base for base, t in zip(baseline, lap_times)]

        trace = project_energy_trace(mode, DEFAULT_START_SOC_MJ, regen_list,
                                     spec_key, pace_dev_per_lap=pace_dev)

        rows = []
        for lap, rec in zip(laps, trace['laps']):
            rows.append((session_id, session['driver_id'], lap['lap_number'],
                         round(rec['start_mj'], 4), round(rec['deployed_mj'], 4),
                         round(rec['harvested_mj'], 4), round(rec['end_mj'], 4)))
        cursor.execute("DELETE FROM race_state WHERE session_id = %s", (session_id,))
        cursor.executemany(
            """INSERT INTO race_state
               (session_id, driver_id, lap_number,
                energy_start_mj, energy_deployed_mj, energy_harvested_mj,
                energy_end_mj)
               VALUES (%s, %s, %s, %s, %s, %s, %s)""",
            rows,
        )
        conn.commit()

        return jsonify({
            "session_id": session_id,
            "track": session['track_name'],
            "mode": mode,
            "spec": PU_SPECS[spec_key]['label'],
            "laps": len(laps),
            "written": len(rows),
            "summary": trace['summary'],
            "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 1),
            "trace": _energy_trace_points(cursor, session_id),
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if cursor:
            try: cursor.close()
            except: pass
        if conn:
            try: conn.close()
            except: pass


@app.route('/api/session/<int:session_id>/tyre-degradation')
def get_tyre_degradation(session_id):
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT 
                l.lap_number,
                l.lap_time_ms / 1000.0 AS lap_time,
                l.lap_time_ms / 1000.0 AS avg_lap_time,
                l.tyre_compound,
                l.tyre_age,
                l.is_valid,
                MAX(CASE WHEN se.event_type = 'PitStop' THEN 1 ELSE 0 END) AS has_pit_stop
            FROM laps l
            LEFT JOIN strategy_events se ON l.lap_id = se.lap_id
                AND se.event_type = 'PitStop'
                AND (se.duration_sec IS NULL OR se.duration_sec >= 15)
            WHERE l.session_id = %s AND l.lap_time_ms > 0
            GROUP BY l.lap_id, l.lap_number, l.lap_time_ms, l.tyre_compound, l.tyre_age, l.is_valid
            ORDER BY l.lap_number
        """, (session_id,))
        deg = cursor.fetchall()
        for d in deg:
            d['lap_time'] = float(d['lap_time']) if d.get('lap_time') is not None else None
            d['avg_lap_time'] = float(d['avg_lap_time']) if d.get('avg_lap_time') is not None else d['lap_time']
        # Fuel-adjusted degradation: stint_delta is each lap's time relative
        # to its own stint's pace line (fuel burn removed).
        deg = detrend_laps(deg)
        return jsonify(deg)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if cursor:
            try: cursor.close()
            except: pass
        if conn:
            try: conn.close()
            except: pass




# How recent a captured lap must be to count as "live".  The top stat
# cards show genuine live telemetry only — game UDP capture, a live
# feed, or a same-day real-race import (import_f1_race stamps captured_at
# when the race ran today).  Older historical imports keep captured_at
# NULL and never qualify.
# Overridable via LIVE_WINDOW_MINUTES (e.g. 30 for long pauses).
def _live_window():
    try:
        minutes = float(os.environ.get('LIVE_WINDOW_MINUTES', '10'))
        if minutes <= 0:
            raise ValueError('must be positive')
        return datetime.timedelta(minutes=minutes)
    except (TypeError, ValueError):
        print(f"[WARNING] Invalid LIVE_WINDOW_MINUTES="
              f"{os.environ.get('LIVE_WINDOW_MINUTES')!r} — using default 10 minutes")
        return datetime.timedelta(minutes=10)


LIVE_WINDOW = _live_window()


@app.route('/api/latest-lap')
def get_latest_lap():
    # Optional ?driver=<CODE> shows that driver's latest live lap instead of
    # the most recent live lap in the whole database (dashboard selector).
    driver = request.args.get('driver', '').strip().upper()
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cutoff = (datetime.datetime.now() - LIVE_WINDOW).strftime('%Y-%m-%d %H:%M:%S')
        base_sql = """
            SELECT 
                l.lap_time_ms / 1000.0 AS lap_time,
                l.lap_number,
                l.tyre_compound,
                l.tyre_age,
                l.session_id,
                s.track_name
            FROM laps l
            JOIN sessions s ON l.session_id = s.session_id
        """
        if driver:
            cursor.execute(base_sql + """
                JOIN drivers d ON l.driver_id = d.driver_id
                WHERE d.driver_code = %s AND l.lap_time_ms > 0
                  AND l.captured_at >= %s
                ORDER BY l.lap_id DESC
                LIMIT 1
            """, (driver, cutoff))
        else:
            cursor.execute(base_sql + """
                WHERE l.lap_time_ms > 0 AND l.captured_at >= %s
                ORDER BY l.lap_id DESC
                LIMIT 1
            """, (cutoff,))
        lap = cursor.fetchone()
        if lap:
            lap['lap_time'] = float(lap['lap_time']) if lap['lap_time'] else None
            lap['live'] = True
            return jsonify(lap)
        # No lap captured in the live window: the frontend shows dashes.
        return jsonify({})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if cursor:
            try: cursor.close()
            except: pass
        if conn:
            try: conn.close()
            except: pass


# DASHBOARD DRIVER LIST
@app.route('/api/drivers/list')
def get_drivers_list():
    """Distinct drivers that have sessions in the database.

    Feeds the dashboard driver selector; returns per-driver session/lap
    counts and the most recent session date.
    """
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT
                d.driver_id,
                d.driver_code,
                d.driver_name,
                COUNT(DISTINCT s.session_id) AS sessions,
                COUNT(l.lap_id) AS laps,
                MAX(s.date) AS last_seen
            FROM drivers d
            JOIN sessions s ON s.driver_id = d.driver_id
            LEFT JOIN laps l ON l.session_id = s.session_id
            GROUP BY d.driver_id, d.driver_code, d.driver_name
            ORDER BY d.driver_code
        """)
        drivers = []
        for r in cursor.fetchall():
            drivers.append({
                "driver_id": r['driver_id'],
                "code": r['driver_code'],
                "name": r['driver_name'],
                "sessions": r['sessions'],
                "laps": r['laps'],
                "last_seen": r['last_seen'].strftime('%Y-%m-%d') if r['last_seen'] else None,
            })
        resp = jsonify({"drivers": drivers})
        resp.headers['Cache-Control'] = 'no-store'
        return resp
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if cursor:
            try: cursor.close()
            except: pass
        if conn:
            try: conn.close()
            except: pass


# PREDICTOR API
@app.route('/api/predict/options')
def get_predict_options():
    if not feature_names:
        return jsonify({"error": "Model not loaded"}), 500

    tyres  = []
    tracks = []

    for feature in feature_names:
        if feature.startswith('tyre_'):
            # tyre_age and the tyre_age x compound interactions (named
            # 'tyre_age:tyre_<c>') are not compounds.
            if feature in ('tyre_age', 'tyre_load') or ':' in feature:
                continue
            name = feature.replace('tyre_', '')
            tyre_type = 'INTERMEDIATE' if name == 'Intermediate' else ('WET' if name == 'Wet' else 'DRY')
            tyres.append({"name": name, "type": tyre_type})
        elif feature.startswith('track_'):
            tracks.append(feature.replace('track_', ''))

    return jsonify({"tyres": tyres, "tracks": tracks})


@app.route('/api/predict', methods=['POST'])
def predict_lap():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    try:
        body          = request.get_json()
        tyre_age      = float(body['tyre_age'])
        lap_number    = int(body['lap_number'])
        fuel_load     = estimate_fuel_load(lap_number)
        tyre_compound = body['tyre_compound']
        track_name    = body['track_name']
        # Optional season year -> era bucket (defaults to the most
        # representative era when omitted).
        year_raw = body.get('year')
        year = None
        if year_raw not in (None, ''):
            try:
                year = int(year_raw)
            except (TypeError, ValueError):
                year = None

        # Reject unseen tracks/tyres with a clear message instead of
        # silently predicting on an all-zero feature row.
        try:
            validate_model_inputs(tyre_compound, track_name, feature_names)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        input_data = construct_prediction_input(
            tyre_age=tyre_age,
            lap_number=lap_number,
            tyre_compound=tyre_compound,
            track_name=track_name,
            feature_names=feature_names,
            year=year
        )

        predicted_time = float(model.predict(input_data)[0])
        minutes = int(predicted_time // 60)
        seconds = predicted_time % 60

        return jsonify({
            "predicted_time": predicted_time,
            "formatted":      f"{minutes}:{seconds:06.3f}",
            "track":          track_name,
            "tyre_compound":  tyre_compound,
            "tyre_age":       tyre_age,
            "lap_number":     lap_number,
            "fuel_load":      fuel_load
        })
    except KeyError as e:
        return jsonify({"error": f"Missing field: {e}"}), 400
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# STRATEGY ADVISOR API
#
# WHY TYRE AGE USED TO BE IGNORED (and the advisor always said "Stay Out"):
# the trained model's only time-varying feature is tyre_age and its fitted
# slope (small and negative, and treated as the fuel-burn rate) is mostly
# FUEL burn — a car laps get faster as the tank lightens.  `_stint_time`
# detrends every prediction by that slope so stay-out and pit scenarios are
# compared on equal fuel footing.  The side effect: after detrending the
# model is age-FLAT, so a 25-lap-old Soft is never slower than a fresh Soft
# and a pit stop only ever *adds* the pit-loss.  Reproduced on the live
# endpoint: "Stay Out" for tyre ages 2, 25 and 40 under VSC, Safety Car,
# rain and crashes alike.
#
# Fix: keep the ML as the fuel-neutral ABSOLUTE pace baseline (fresh tyres,
# real seconds per lap) and layer an explicit, researched tyre-DEGRADATION
# model on top (per-track abrasion scaled).  Decisions come from a
# multi-stop stint optimizer (how many stops, on which lap, onto which
# compound) over that tyre model; tyre age matters because wear costs real
# seconds per lap and grows past each compound's knee.

# ── Researched tyre model ──────────────────────────────────────────────
# Numbers below are based on published F1 tyre behaviour:
#   * Usable stint life — Soft ~15-22 laps, Medium ~25-30, Hard ~35-45
#     ("How Long Do F1 Tires Last?", flowracers.com/blog/how-long-do-f1-tires-last).
#   * Median measured degradation across 2026 stints ~0.065-0.071 s/lap,
#     growing sharply once a compound passes its window (F1 Chronicle
#     analysis of official timing data, July 2026).
#   * Green-flag pit stop costs ~21.9 s of race time, ~18.1 s under a VSC
#     and ~14.2 s under the Safety Car — the whole field slows, so the stop
#     is effectively subsidised (Formula Dream, measured across 2025 races:
#     formuladream.app/blog/f1-pit-stop-cost-measured).
#   * In rain, slicks lose several seconds per lap to Inters; full Wets only
#     become faster once standing water appears (standard crossover
#     guidance from race-day coverage).

# Wear model: since the Sept-2026 retrain the advisor's degradation comes
# from the model's MEASURED per-compound tyre-age slopes (see
# MEASURED_COMPOUND_SLOPE above), floored per compound and scaled per track
# by TRACK_ABRASION.  TYRE_WEAR below is the pre-measurement researched
# piecewise curve — extra seconds per lap at age a: s1*a up to `knee`, then
# s1*knee + s2*(a-knee) — kept as the fallback for models trained before the
# age-interaction features existed.
TYRE_WEAR = {
    'Soft':         {'knee': 12, 's1': 0.030, 's2': 0.120},
    'Medium':       {'knee': 20, 's1': 0.025, 's2': 0.090},
    'Hard':         {'knee': 28, 's1': 0.020, 's2': 0.065},
    'Intermediate': {'knee': 16, 's1': 0.025, 's2': 0.080},
    'Wet':          {'knee': 20, 's1': 0.020, 's2': 0.060},
}

# Usable life (laps, medium-abrasion track): the age at which the wear
# penalty reaches ~1.5 s/lap and the set is realistically finished.  On
# abrasive tracks the effective life shortens (see _life_laps).
TYRE_LIFE = {
    'Soft': 22,
    'Medium': 31,
    'Hard': 42,
    'Intermediate': 28,
    'Wet': 36,
}

# Fresh-compound pace offsets (s/lap slower than the fastest dry slick).
# The ML model's compound intercepts are NOT reliably ordered (see the
# retrained model note), so compound switches are scored on nominal
# Pirelli-style gaps (~0.3-0.5 s/lap between compounds) instead of the ML
# intercepts.
TYRE_PACE_DELTA = {'Soft': 0.0, 'Medium': 0.30, 'Hard': 0.60}

# Wet-condition pace offsets (s/lap) vs Inters — applied only when the race
# event is Rain.
RAIN_PACE_DELTA = {'Intermediate': 0.0, 'Wet': 1.6,
                   'Soft': 3.0, 'Medium': 3.0, 'Hard': 3.0}

# Race-time cost of a stop taken under the CURRENT neutralisation.  A VSC
# / Safety Car bunches the field and subsidises the stop; a red flag makes
# the tyre change FREE (0 s).  Stops taken later under green running are
# charged DEFAULT_PIT_LOSS.
PIT_LOSS_BY_EVENT = {'VSC': 18.0, 'SafetyCar': 14.0, 'RedFlag': 0.0}
DEFAULT_PIT_LOSS = 22.0

# ── Fresh-set inventory ────────────────────────────────────────────────
# How many new sets of each compound a driver starts a race weekend with
# (standard dry allocation: 2 Hard / 3 Medium / 4 Soft; wet compounds 3
# Intermediate / 2 Wet).  Each stint a driver completes consumes one set of
# that compound, so fresh-left = allocation - stints-used.  When the strategy
# advisor has a real session + driver it counts actual stints from the laps
# table; without one it assumes the full allocation.
SET_ALLOCATION = {'Soft': 4, 'Medium': 3, 'Hard': 2,
                  'Intermediate': 3, 'Wet': 2}

# Rejoin-traffic multiplier on pit-loss: rejoining into clear air is the
# cheapest stop; light traffic adds a bit; a gaggle of cars costs real
# seconds on the out-lap and first laps behind slower traffic.
TRAFFIC_PIT_FACTOR = {'Clear': 0.95, 'Light': 1.0, 'Heavy': 1.25}

# Gap-to-ahead thresholds for the undercut/overcut call:
#   * within UNDERCUT_GAP_S of the car ahead -> an undercut window is open
#     (stop a lap early and use fresh-tyre pace to jump it in the pits);
#   * OVCUT_GAP_S or more clear air ahead   -> overcut territory (let rivals
#     pit, keep the track yours, then stop later).
UNDERCUT_GAP_S = 5.0
OVERCUT_GAP_S = 8.0


def _driver_inventory(session_id, driver_id, conn):
    """Fresh sets left per compound for (session, driver) from actual stint
    history in the laps table: each run of consecutive laps on a compound is
    one set consumed.  Falls back to the full allocation when no session /
    driver is given or the DB has no laps for them.
    """
    inv = {c: {'allocation': SET_ALLOCATION.get(c, 0), 'used': 0,
               'fresh': SET_ALLOCATION.get(c, 0)}
           for c in SET_ALLOCATION}
    if not session_id or not driver_id or conn is None:
        return inv
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute("""
            SELECT lap_number, tyre_compound, tyre_age
            FROM laps
            WHERE session_id = %s AND driver_id = %s
              AND tyre_compound IS NOT NULL
            ORDER BY lap_number
        """, (int(session_id), int(driver_id)))
        prev_compound, prev_age, used = None, None, {}
        for row in cur.fetchall():
            c = str(row['tyre_compound']).strip()
            age = row.get('tyre_age')
            # New set: compound change, OR the age counter reset (a same-
            # compound stop fits a fresh set and the age drops back to ~0).
            if (c != prev_compound
                    or (age is not None and prev_age is not None
                        and int(age) < int(prev_age))):
                used[c] = used.get(c, 0) + 1
            prev_compound, prev_age = c, age
        cur.close()
        for c, n in used.items():
            if c in inv:
                inv[c]['used'] = n
                inv[c]['fresh'] = max(0, inv[c]['allocation'] - n)
    except Exception:
        traceback.print_exc()
        # Leave the full allocation — never fail the advisor on inventory.
    return inv

# ── Track abrasiveness ────────────────────────────────────────────────
# Multiplies the nominal wear curves above.  1.0 = a "medium" circuit;
# abrasive / high-degradation venues (Bahrain, Jeddah, Singapore-style
# street races) wear tyres faster, smooth low-degradation venues
# (Silverstone, Sochi, Monza, Red Bull Ring) slower.  Magnitudes follow
# Pirelli's per-race compound selection and race-day stint analysis; any
# circuit not listed defaults to 1.0.
TRACK_ABRASION = {
    'Bahrain International Circuit': 1.25,
    'Jeddah Corniche Circuit': 1.20,
    'Marina Bay': 1.30,
    'Miami Gardens': 1.15,
    'Suzuka International Racing Course': 1.15,
    'Circuit De Barcelona-Catalunya': 1.15,
    'Autodromo Internazionale Enzo E Dino Ferrari': 1.20,
    'Hungaroring': 1.10,
    'Yas Island': 1.05,
    'Baku City Circuit': 1.05,
    'Autódromo Hermanos Rodríguez': 1.10,
    'Circuit De Monaco': 1.15,
    'Silverstone Circuit': 0.90,
    'Sochi Autodrom': 0.90,
    'Autodromo Nazionale Di Monza': 0.95,
    'Spa-Francorchamps': 0.95,
    'Circuit Zandvoort': 0.95,
    'Shanghai International Circuit': 1.00,
    'Red Bull Ring': 0.90,
}


def _track_abrasion(track):
    """Wear multiplier for a circuit (1.0 when unlisted / unknown).  With a
    retrained model it scales the measured per-compound slopes per track
    (_measured_slope); it only scales the researched fallback curve when no
    measured slopes exist (pre-retrain model)."""
    return TRACK_ABRASION.get(str(track).strip().title(), 1.0)


# Cap on the per-lap wear penalty (s/lap) — beyond it a set is simply dead;
# letting the penalty grow unbounded would make "nurse a clapped set home"
# absurdly expensive instead of merely very expensive.
MAX_WEAR_PENALTY = 1.8


def _measured_slope(tyre, track=None):
    """Per-lap wear slope (s/lap) for (tyre, track): the model's measured
    per-compound age slope, floored per compound and scaled by the track's
    researched abrasion.  None when the model has no measured slope for the
    compound (pre-retrain model) — callers then use the researched curves."""
    if tyre not in MEASURED_COMPOUND_SLOPE:
        return None
    s = max(MEASURED_SLOPE_FLOOR.get(tyre, 0.03),
            min(MEASURED_SLOPE_CAP, float(MEASURED_COMPOUND_SLOPE[tyre])))
    return s * (_track_abrasion(track) if track else 1.0)


def _wear_per_lap(tyre, age, track=None):
    """Seconds/lap lost to wear at a given tyre age (0 on a fresh set).

    With a retrained model this is the measured per-compound slope times age
    (capped at MAX_WEAR_PENALTY).  Without one it falls back to the
    researched piecewise curve — managed phase then the post-knee cliff.
    """
    slope = _measured_slope(tyre, track)
    if slope is not None:
        return min(slope * age, MAX_WEAR_PENALTY)
    spec = TYRE_WEAR.get(tyre)
    if not spec:
        return 0.0
    f = _track_abrasion(track) if track else 1.0
    knee, s1, s2 = spec['knee'], spec['s1'], spec['s2']
    s1f, s2f = f * s1, f * s2
    return s1f * age if age <= knee else s1f * knee + s2f * (age - knee)


def _life_laps(tyre, track=None):
    """Age (laps) at which a set is realistically finished on this circuit
    — where the wear penalty reaches ~1.5 s/lap (clamped to the researched
    nominal life so measurement noise cannot extend a set forever)."""
    slope = _measured_slope(tyre, track)
    if slope is not None and slope > 0:
        return max(3, min(int(1.5 / slope), TYRE_LIFE.get(tyre, 30)))
    spec = TYRE_WEAR.get(tyre)
    if not spec:
        return TYRE_LIFE.get(tyre, 30)
    f = _track_abrasion(track) if track else 1.0
    knee, s1, s2 = spec['knee'], spec['s1'], spec['s2']
    s1f, s2f = f * s1, f * s2
    knee_pen = s1f * knee
    if knee_pen >= 1.5:
        return max(1, int(1.5 / s1f))
    return max(1, int(knee + (1.5 - knee_pen) / s2f))


def _wear_cost(tyre, start_age, laps, track=None):
    """Total wear time (s) across a stint of `laps` starting at `start_age`
    (track-aware)."""
    if laps <= 0:
        return 0.0
    return sum(_wear_per_lap(tyre, start_age + i, track) for i in range(laps))


def _stint_time(tyre, start_age, laps, track, start_lap_number, year=None):
    """Fuel-neutral ML lap-time total over `laps` on `tyre` from `start_age`.

    Each prediction is detrended by the fuel-burn rate so every scenario is
    evaluated on equal fuel footing (stay-out and pit burn identical fuel
    over the same remaining laps).  For the deployed linear model this makes
    the total effectively independent of `start_age` — its tyre_age slope is
    fuel, not wear — so tyre age is deliberately NOT consulted here.  Wear is
    layered on separately by the advisor via `_wear_cost()`; the ML only
    supplies the absolute lap-time scale.  `year` selects the model's era
    bucket (the season's pace level).
    """
    tf = f'tyre_{tyre}'
    tkf = f'track_{track}'
    if tf not in feature_names or tkf not in feature_names or laps <= 0:
        return 0.0
    total = 0.0
    for i in range(laps):
        age = start_age + i
        row = construct_prediction_input(
            tyre_age=age,
            lap_number=start_lap_number + i,
            tyre_compound=tyre,
            track_name=track,
            feature_names=feature_names,
            year=year
        )
        predicted = float(model.predict(row)[0])
        # Remove the fuel (non-wear) component of the age effect.
        total += predicted - fuel_burn_rate * age
    return total


def _pace_abs(tyre, event):
    """Per-lap pace offset of `tyre` vs the fastest tyre for the current
    conditions (dry compounds vs Soft; wet compounds vs Intermediate)."""
    if event == 'Rain':
        return RAIN_PACE_DELTA.get(tyre, 3.0)
    return TYRE_PACE_DELTA.get(tyre, 0.0)


def _stint_cost(tyre, laps, event, track):
    """Pace-offset + wear cost of `laps` on a FRESH `tyre`.

    The shared ML baseline and pit losses are added separately, so this is
    a relative cost between otherwise-identical options.
    """
    return laps * _pace_abs(tyre, event) + _wear_cost(tyre, 0, laps, track)


# ---------------------------------------------------------------------------
# Multi-stop stint optimizer
#
# Finds the fastest way to cover the remaining race laps: how many stops,
# on which lap, and onto which compound.  A stop taken NOW is charged the
# current neutralisation's loss (0 under a red flag, ~14 s under SC, ~18 s
# under VSC); if the driver stays out, the neutralisation will have ended
# by the time they stop, so later stops are charged the full green-flag
# loss.  Pit losses are scaled by rejoin traffic (clear air is cheapest;
# rejoining into a gaggle of cars costs seconds on the out-lap).
#
# The optimizer only plans stints with COMPOUNDS THE DRIVER ACTUALLY HAS
# fresh sets of (inventory passed in; default = full allocation): each new
# stint consumes a fresh set, so a plan is feasible only if its fresh-stint
# compound count does not exceed the sets left.  Wear and pace use the
# measured-blended model above; the ML only supplies the shared absolute
# baseline for display.
# ---------------------------------------------------------------------------
def _reconstruct_stints(dp, k, r):
    """Expand dp[k][r]'s stored first choice into the fresh-stint sequence
    [(compound, start_age=0, laps)]."""
    segs = []
    while r > 0 and k >= 0 and dp[k][r] and dp[k][r][1]:
        c, m = dp[k][r][1]
        segs.append((c, 0, m))
        r -= m
        if r > 0:
            k -= 1
    return segs


def _optimize_plans(cur_tyre, cur_age, laps_rem, event, track, pit_loss_now,
                    cur_lap, inventory=None, traffic_factor=1.0, max_stops=3):
    """Return one candidate plan per total-stop count, cheapest handling of
    the remaining laps first-ish (caller sorts).

    `inventory` maps compound -> {'fresh': n, ...} (fresh sets left); when
    None every compound is assumed available.  `traffic_factor` scales the
    green-flag pit loss (the caller pre-scales the current-stop loss for
    display).

    Plan dict keys: q (stops), cost (relative research-model cost: pace +
    wear + pit losses, ML baseline excluded), m1 (laps run on the current
    set before the first stop; 0 = box now), segs ([(compound, start_age,
    laps), ...] including the current-tyre leg), box_laps (lap numbers of
    each stop) and risk.
    """
    L = laps_rem
    if L <= 0:
        return []
    green = DEFAULT_PIT_LOSS * traffic_factor

    def _fresh_available(compound):
        if not inventory:
            return True
        return inventory.get(compound, {}).get('fresh', 0) > 0

    fresh_pool = (['Intermediate', 'Wet'] if event == 'Rain'
                  else ['Soft', 'Medium', 'Hard'])
    fresh_pool = [c for c in fresh_pool
                  if f'tyre_{c}' in feature_names and _fresh_available(c)]
    if not fresh_pool:
        # No usable compound left to pit onto (nothing in the model's set OR
        # every compound's fresh sets are exhausted) — finish on the current
        # set.
        return [{'q': 0,
                 'cost': (L * _pace_abs(cur_tyre, event)
                          + _wear_cost(cur_tyre, cur_age, L, track)),
                 'm1': L, 'segs': [(cur_tyre, cur_age, L)],
                 'box_laps': [],
                 'risk': _stay_risk(cur_tyre, cur_age, L, event, track)}]

    K = min(max_stops, L - 1) if L > 1 else 0
    pace_cur = _pace_abs(cur_tyre, event)

    # dp[k][r] = (cost, first_choice=(compound, laps_then_stop)) to cover r
    # laps from a FRESH start with exactly k stops.
    dp = [[None] * (L + 1) for _ in range(K + 1)]
    for r in range(1, L + 1):
        best = None
        for c in fresh_pool:
            cost = _stint_cost(c, r, event, track)
            if best is None or cost < best[0]:
                best = (cost, (c, r))
        dp[0][r] = best
    for k in range(1, K + 1):
        for r in range(1, L + 1):
            best = None
            for c in fresh_pool:
                pc = _pace_abs(c, event)
                for m in range(1, r):
                    tail = dp[k - 1][r - m]
                    if tail is None:
                        continue
                    cost = (m * pc + _wear_cost(c, 0, m, track)
                            + green + tail[0])
                    if best is None or cost < best[0]:
                        best = (cost, (c, m))
            dp[k][r] = best

    def _inventory_ok(segs):
        """A plan is feasible only if each FRESH stint's compound still has a
        set available (the first leg rides the set already on the car, which
        is consumed by definition)."""
        if not inventory:
            return True
        used = {}
        for i, (c, s, m) in enumerate(segs):
            if i == 0 or m <= 0:
                continue
            used[c] = used.get(c, 0) + 1
        return all(used.get(c, 0) <= inventory.get(c, {}).get('fresh', 0)
                   for c in used)

    def _candidate(q, m1):
        """Cost & shape for: run m1 more laps on the current set, then take
        q stops in total (m1 == 0 means box now).  Returns None when the
        plan needs a fresh set the driver no longer has."""
        cur_cost = (m1 * pace_cur
                    + _wear_cost(cur_tyre, cur_age, m1, track))
        if q == 0:
            if m1 != L:
                return None
            return cur_cost, [(cur_tyre, cur_age, m1)], []
        rem = L - m1
        if rem <= 0:
            return None
        tail = dp[q - 1][rem]
        if tail is None:
            return None
        stop_cost = pit_loss_now if m1 == 0 else green
        cost = cur_cost + stop_cost + tail[0]
        segs = [(cur_tyre, cur_age, m1)] if m1 > 0 else [(cur_tyre, cur_age, 0)]
        segs += _reconstruct_stints(dp, q - 1, rem)
        if not _inventory_ok(segs):
            return None
        box = []
        acc = 0
        for i in range(len(segs) - 1):
            acc += segs[i][2]
            box.append(cur_lap + acc if acc > 0 else cur_lap)
        return cost, segs, box

    plans = []
    for q in range(0, K + 1):
        best = None
        for m1 in ([L] if q == 0 else range(0, L)):
            cand = _candidate(q, m1)
            if cand is None:
                continue
            cost, segs, box = cand
            if best is None or cost < best[0]:
                best = (cost, m1, segs, box)
        if best is None:
            continue
        cost, m1, segs, box = best
        risk = 'Low'
        for (c, s, m) in segs:
            if m <= 0:
                continue
            life = _life_laps(c, track)
            if event == 'Rain' and c not in ('Intermediate', 'Wet'):
                risk = 'High'
            if s + m > life + 2:
                risk = 'High'
            elif s + m > int(life * 0.85) and risk != 'High':
                risk = 'Medium'
        plans.append({'q': q, 'cost': cost, 'm1': m1, 'segs': segs,
                      'box_laps': box, 'risk': risk})
    return plans


def _format_plan(cur_tyre, cur_age, cur_lap, plan, pit_loss_now):
    """Human summary of an optimised plan -> (headline, description)."""
    q, m1 = plan['q'], plan['m1']
    segs = plan['segs']
    box = plan['box_laps']
    nonzero = [(c, s, m) for (c, s, m) in segs if m > 0]

    if q == 0:
        c, s, m = segs[0]
        return ("Stay Out",
                f"{c} (age {s}) {m} laps to the flag — no more stops "
                f"(would end age {s + m} on this track)")
    if not nonzero:
        return (f"{q}-stop plan", "No laps left to plan around.")

    if m1 == 0:
        head = (f"Pit now — {nonzero[0][0]} to flag" if q == 1
                else f"Pit now ({q}-stop plan)")
    else:
        head = f"Pit lap {cur_lap + m1} ({q}-stop plan)"

    timeline = []
    if m1 == 0:
        timeline.append("BOX NOW (free change)" if pit_loss_now <= 0
                        else f"BOX NOW (~{pit_loss_now:.0f}s)")
    for i, (c, s, m) in enumerate(nonzero):
        if i > 0:
            b = box[i] if m1 == 0 else box[i - 1]
            timeline.append(f"box lap {b}")
        label = f"{c} (age {s})" if s > 0 else f"fresh {c}"
        timeline.append(f"{label} {m} laps")
    timeline.append("flag")
    return head, " · ".join(timeline)


def _stay_risk(tyre, age, laps_rem, event, track=None):
    """Risk of simply continuing on the current set to the flag."""
    life = _life_laps(tyre, track)
    if age >= life or age + laps_rem > life + 3:
        return 'High'                     # cannot realistically finish
    if event == 'Rain' and tyre not in ('Intermediate', 'Wet'):
        return 'High'                     # slicks on a wet track
    # Past ~60% of usable life the set is degrading fast enough to warrant
    # a Medium-risk flag.
    return 'Medium' if age >= int(life * 0.6) else 'Low'


def _reason(event, best, cur_tyre, cur_age, laps_rem, pit_loss_now, track):
    """Explain the winning plan using the researched tyre model."""
    q = best.get('pit_stops', 0)
    desc = best.get('description', '')
    first_new = best.get('first_new')
    wear_now = _wear_per_lap(cur_tyre, cur_age, track)
    life = _life_laps(cur_tyre, track)

    if event == 'RedFlag':
        if q > 0:
            return ("Red flag = free tyre change with no time lost — put on "
                    f"fresh tyres for the restart. {desc}")
        return (f"Red flag lets you change tyres for free, but the best model "
                f"path keeps the {cur_tyre}s (age {cur_age}). {desc}")

    if event == 'Rain':
        if first_new in ('Intermediate', 'Wet'):
            return (f"Rain is here — the {cur_tyre}s (age {cur_age}) are ~3 s/lap "
                    f"off the pace on a wet track. {desc}")
        if cur_tyre in ('Intermediate', 'Wet'):
            if q > 0:
                return (f"Still wet — the {cur_tyre}s (age {cur_age}) are losing "
                        f"{wear_now:.2f} s/lap now; a fresh set sees out the rain. "
                        f"{desc}")
            return (f"Already on a wet-weather tyre — monitor intensity; only stop "
                    f"again if it becomes full standing water. {desc}")
        return (f"Only {laps_rem} lap(s) left — the {pit_loss_now:.0f}s pit loss "
                f"buys less than staying out would cost. Nurse the {cur_tyre}s "
                f"to the flag. {desc}")

    if event in ('VSC', 'SafetyCar'):
        tag = 'Safety Car' if event == 'SafetyCar' else 'VSC'
        if q > 0:
            return (f"{tag} has bunched the field — a stop now costs only "
                    f"~{pit_loss_now:.0f}s instead of the ~{DEFAULT_PIT_LOSS:.0f}s "
                    f"green-flag loss. {desc}")
        return (f"The {cur_tyre}s (age {cur_age}) cover the remaining {laps_rem} "
                f"laps without a stop — hold track position under the {tag}. "
                f"{desc}")

    if event == 'Crash':
        if q > 0:
            return (f"Incident up ahead — a VSC/SC window is likely and a stop "
                    f"under it is cheap. {desc}")
        return (f"No confirmed neutralisation yet — hold position. {desc}")

    if q > 0:
        return (f"Wear on the {cur_tyre}s (age {cur_age}) is costing {wear_now:.2f} "
                f"s/lap now and rising — the extra stop pays for itself. {desc}")
    return (f"Remaining wear on the {cur_tyre}s (age {cur_age} of ~{life} laps of "
            f"life on this track) costs less than a stop. {desc}")


def _position_notes(gap_to_ahead, best, cur_tyre, cur_age, track):
    """Track-position assessment: undercut / overcut windows and whether the
    recommended plan risks losing track position.

    gap_to_ahead = seconds behind the car directly ahead (0 = nose-to-tail;
    negative = you are ahead of it; None = unknown / not supplied).
    """
    if gap_to_ahead is None or not best:
        return {'note': None, 'undercut_open': False, 'overcut_open': False}
    gap = float(gap_to_ahead)
    q = best.get('pit_stops', 0)

    undercut_open = overcut_open = False
    if q >= 1:
        # Undercut: close enough behind that stopping a lap EARLY (while the
        # car ahead stays out) lets fresh-tyre pace do the overtake in the
        # pit window.
        if 0 <= gap <= UNDERCUT_GAP_S:
            undercut_open = True
        # Overcut: leading with clear air (or the car ahead is so far back it
        # cannot cover) — stay out while rivals pit, then stop later.
        if gap < 0 and -gap >= OVERCUT_GAP_S:
            overcut_open = True

    if undercut_open:
        note = (f"UNDERCUT WINDOW — {gap:.1f}s to the car ahead and it's on "
                f"older tyres: box a lap early and use the fresh-set delta to "
                f"jump it in the pit window.")
    elif overcut_open:
        note = (f"OVERCUT WINDOW — you lead by {-gap:.1f}s in clear air: let "
                f"rivals pit first and keep the track to yourself, then stop "
                f"later; the time you bank on clear-air laps beats covering now.")
    elif q >= 1 and 0 <= gap <= DEFAULT_PIT_LOSS:
        note = (f"Position watch: {gap:.1f}s to the car ahead — the {cur_tyre}s "
                f"(age {cur_age}) stop puts the stop-and-hold into the pit "
                f"window; cover if they box, don't gift the place.")
    elif q == 0 and 0 <= gap <= UNDERCUT_GAP_S:
        note = (f"Position watch: {gap:.1f}s behind — holding track position "
                f"matters more than the marginal stop right now.")
    elif gap < 0:
        note = (f"You lead the car behind by {-gap:.1f}s — the plan's stops are "
                f"safe from an immediate undercut.")
    else:
        note = (f"{gap:.1f}s to the car ahead — comfortably outside the pit "
                f"delta, so the stop does not cost track position.")
    return {'note': note, 'undercut_open': undercut_open,
            'overcut_open': overcut_open}


@app.route('/api/strategy/analyze', methods=['POST'])
def analyze_strategy():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    conn = None
    try:
        body         = request.get_json()
        cur_lap      = int(body['current_lap'])
        total_laps   = int(body['total_laps'])
        cur_tyre     = body['current_tyre']
        cur_age      = int(body['current_age'])
        track        = body['track']
        event        = body['event_type']

        # Optional context: real session + driver (for set inventory and the
        # season's era), gap to the car ahead and rejoin traffic (for the
        # undercut/overcut call and traffic-scaled pit loss).
        session_id = body.get('session_id')
        driver_in  = body.get('driver')
        gap_raw    = body.get('gap_to_ahead')
        traffic    = str(body.get('traffic', 'Light')).strip().title()
        year       = body.get('year')

        # Reject unseen tracks/tyres explicitly — otherwise every strategy
        # totals 0.0 and the advisor silently answers "Stay Out".
        try:
            validate_model_inputs(cur_tyre, track, feature_names)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        laps_rem = max(1, total_laps - cur_lap)

        # Resolve the driver (code like HAM / VER, or a numeric id) and the
        # season year from the DB when a session is supplied, so the set
        # inventory and era are the real ones.
        driver_id = None
        conn = get_db_connection()
        cur = conn.cursor(dictionary=True)
        if session_id:
            cur.execute("SELECT date FROM sessions WHERE session_id = %s",
                        (int(session_id),))
            sess = cur.fetchone()
            if sess and sess.get('date') and not year:
                try:
                    d = sess['date']
                    year = int(d.year if hasattr(d, 'year') else str(d)[:4])
                except Exception:
                    pass
        if driver_in:
            d = str(driver_in).strip()
            if d.isdigit():
                driver_id = int(d)
            else:
                cur.execute("SELECT driver_id FROM drivers "
                            "WHERE UPPER(driver_code) = %s LIMIT 1",
                            (d.upper(),))
                row = cur.fetchone()
                if row:
                    driver_id = int(row['driver_id'])
        cur.close()

        inventory = _driver_inventory(session_id, driver_id, conn)

        # A stop taken under the CURRENT neutralisation is discounted: free
        # under a red flag, ~14 s under SC, ~18 s under VSC (bunched field);
        # stops taken later under green cost the full ~22 s.  Both scale with
        # rejoin traffic — emerging into a gaggle costs seconds, clear air is
        # the cheapest stop.
        traffic_factor = TRAFFIC_PIT_FACTOR.get(traffic, 1.0)
        pit_loss_now = (PIT_LOSS_BY_EVENT.get(event, DEFAULT_PIT_LOSS)
                        * traffic_factor)

        # Multi-stop stint optimisation: best plan per total-stop count,
        # planned only with compounds the driver still has fresh sets of.
        plans = _optimize_plans(cur_tyre, cur_age, laps_rem, event, track,
                                pit_loss_now, cur_lap,
                                inventory=inventory,
                                traffic_factor=traffic_factor)

        base = _stint_time(cur_tyre, 0, laps_rem, track, cur_lap, year=year)
        strategies = []
        if base > 0:
            cur_pace_ref = laps_rem * _pace_abs(cur_tyre, event)
            for p in plans:
                head, desc = _format_plan(cur_tyre, cur_age, cur_lap, p,
                                          pit_loss_now)
                # Displayed total = shared ML baseline + plan's relative cost
                # (researched pace + wear + losses) vs the current tyre.
                total = base + p['cost'] - cur_pace_ref
                nz = [(c, s, m) for (c, s, m) in p['segs'] if m > 0]
                if p['m1'] == 0 and nz:
                    first_new = nz[0][0]
                elif len(nz) > 1:
                    first_new = nz[1][0]
                else:
                    first_new = None
                strategies.append({
                    "option":      head,
                    "description": desc,
                    "total_time":  total,
                    "pit_stops":   p['q'],
                    "risk":        p['risk'],
                    "box_laps":    p['box_laps'],
                    "first_new":   first_new,
                })
            strategies.sort(key=lambda x: x['total_time'])

        best = strategies[0] if strategies else None
        recommendation = {
            "action": best['option'] if best else "No data",
            "reason": (_reason(event, best, cur_tyre, cur_age, laps_rem,
                                pit_loss_now, track)
                        if best else "Not enough model data.")
        }

        gap_to_ahead = None
        if gap_raw not in (None, ''):
            try:
                gap_to_ahead = float(gap_raw)
            except (TypeError, ValueError):
                gap_to_ahead = None

        return jsonify({
            "event":             event,
            "current_situation": {"lap": cur_lap, "laps_remaining": laps_rem,
                                  "tyre": cur_tyre, "tyre_age": cur_age,
                                  "traffic": traffic,
                                  "gap_to_ahead": gap_to_ahead},
            "strategies":        strategies,
            "recommendation":    recommendation,
            "inventory":         inventory,
            "position":          _position_notes(gap_to_ahead, best, cur_tyre,
                                                  cur_age, track),
            "era":               year,
            "fuel_burn_rate":    fuel_burn_rate
        })

    except KeyError as e:
        return jsonify({"error": f"Missing field: {e}"}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass

# ENERGY MODE WHAT-IF API
#
# Compares push / balanced / lift-and-coast deployment over the laps remaining
# in a session and returns which modes finish with the battery above the floor.
# Pace is estimated from the same lap-time model as the pit strategy advisor;
# the energy trace itself is projected by energy_simulator.project_energy_trace
# from per-lap regeneration estimated on the session's own telemetry.

@app.route('/api/strategy/energy-analyze', methods=['POST'])
def analyze_energy_modes():
    if model is None:
        return jsonify({"error": "Model not loaded — run scripts/ml_lap_predictions.py first"}), 500
    conn = None
    cursor = None
    try:
        body = request.get_json() or {}
        session_id = int(body.get('session_id', 0))
        if session_id <= 0:
            return jsonify({"error": "Missing field: session_id"}), 400
        current_lap = int(body.get('current_lap', 1))
        battery_mj = body.get('battery_mj')

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute("""
            SELECT s.session_id, s.track_name, s.date, s.driver_id, d.driver_name
            FROM sessions s
            JOIN drivers d ON s.driver_id = d.driver_id
            WHERE s.session_id = %s
        """, (session_id,))
        session = cursor.fetchone()
        if session is None:
            return jsonify({"error": f"Session {session_id} not found"}), 404

        # Regulation era of this session's car (2026 PU vs legacy 120 kW PU)
        # decides the energy-store and per-lap caps the projection uses.
        spec_key = spec_for_year(session['date'].year if session['date'] else 0)
        spec = PU_SPECS[spec_key]

        cursor.execute("""
            SELECT lap_id, lap_number, tyre_compound, tyre_age
            FROM laps
            WHERE session_id = %s AND lap_time_ms > 0
            ORDER BY lap_number
        """, (session_id,))
        all_laps = cursor.fetchall()
        if not all_laps:
            return jsonify({"error": "Session has no timed laps to simulate"}), 400

        # Context lap: the last real lap at or before current_lap (fall back to
        # the first lap when current_lap precedes all stored laps).
        ctx = None
        for l in all_laps:
            if l['lap_number'] <= current_lap:
                ctx = l
        if ctx is None:
            ctx = all_laps[0]
            current_lap = ctx['lap_number']
        future = [l for l in all_laps if l['lap_number'] > current_lap]
        if not future:
            return jsonify({"error": f"No laps remaining after lap {current_lap}"}), 400

        # Track/tyre context for pace estimates (model feature alignment).
        track = session['track_name']
        tyre = ctx.get('tyre_compound')
        if not tyre:
            # NULL compound context lap: use the most common remaining tyre.
            counts = {}
            for l in future:
                if l.get('tyre_compound'):
                    counts[l['tyre_compound']] = counts.get(l['tyre_compound'], 0) + 1
            tyre = max(counts, key=counts.get) if counts else None
        if not tyre:
            return jsonify({"error": "No tyre compound known for this session"}), 400
        try:
            validate_model_inputs(tyre, track, feature_names)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        # Per-lap regeneration for the remaining laps (speed-drop estimate).
        regen_list = []
        future_ids = [l['lap_id'] for l in future]
        if future_ids:
            ph = ','.join(['%s'] * len(future_ids))
            cursor.execute(f"SELECT lap_id, speed FROM telemetry WHERE lap_id IN ({ph}) ORDER BY telemetry_id", future_ids)
            telem = {}
            for row in cursor.fetchall():
                telem.setdefault(row['lap_id'], []).append(row)
            regen_list = [regen_from_speed_samples(telem.get(l['lap_id'], []), spec_key)
                          for l in future]

        # Starting battery: explicit override, else the stored trace's state at
        # the end of the context lap, else a full store.
        if battery_mj is not None:
            start = min(max(float(battery_mj), BATTERY_MIN_MJ), BATTERY_CAPACITY_MJ)
        else:
            cursor.execute("""
                SELECT energy_end_mj FROM race_state
                WHERE session_id = %s AND lap_number = %s
            """, (session_id, current_lap))
            row = cursor.fetchone()
            # No stored trace: assume a full store, as a car leaving the
            # grid -- races start with the Energy Store fully charged.
            start = (float(row['energy_end_mj']) if row
                     else DEFAULT_START_SOC_MJ)

        # Baseline per-lap pace for the remaining laps (fuel-detrended model
        # predictions, age +1 per lap from the context lap's tyre age).
        cur_age = int(ctx['tyre_age'] or 1)
        base_times = []
        for i, l in enumerate(future):
            age = cur_age + 1 + i
            row_vec = construct_prediction_input(
                tyre_age=age, lap_number=int(l['lap_number']),
                tyre_compound=tyre, track_name=track,
                feature_names=feature_names)
            base_times.append(float(model.predict(row_vec)[0]) - fuel_burn_rate * age)

        # Energy-to-time conversion: measured per-track value (full-throttle
        # share of this track's telemetry, trained into energy_pace.json)
        # instead of the flat global constant.
        pace_track = track_pace_s_per_mj(track)
        pace_basis = 'measured per-track (full-throttle share of telemetry)'

        results = []
        for mode in sorted(MODES):
            trace = project_energy_trace(mode, start, regen_list, spec_key)
            s = trace['summary']
            laps_out = trace['laps']
            # Pace credit per lap: pace_track per deployed MJ, scaled
            # down on energy-limited laps (energy that arrives around re-charge
            # windows is worth less than deployment at the optimum power points).
            total_time = sum(
                base - pace_track * r['deployed_mj'] *
                (LIMITED_LAP_PACE_EFFECTIVENESS if r['limited'] else 1.0)
                for base, r in zip(base_times, laps_out))
            feasible = s['final_battery_mj'] > BATTERY_MIN_MJ + 0.05
            results.append({
                "mode": mode,
                "final_battery_mj": s['final_battery_mj'],
                "final_battery_pct": s['final_battery_pct'],
                "min_battery_mj": s['min_battery_mj'],
                "limited_laps": s['limited_laps'],
                "deployed_total_mj": s['deployed_total'],
                "total_time_s": round(total_time, 2),
                "feasible": feasible,
            })

        feasible = [r for r in results if r['feasible']]
        if feasible:
            # Fastest of the modes that finish above the floor.
            best = min(feasible, key=lambda r: r['total_time_s'])
            reason = (f"{best['mode'].title()} finishes with the battery above the floor "
                      f"({best['final_battery_pct']:.1f}%) with the best projected remaining "
                      f"race time ({best['total_time_s']:.1f}s).")
        else:
            # No mode survives: pick the one that preserves the most battery.
            best = max(results, key=lambda r: r['final_battery_mj'])
            reason = (f"No mode finishes above the floor from {start:.1f} MJ with "
                      f"{len(future)} laps to go — {best['mode'].title()} preserves "
                      f"the most battery ({best['final_battery_pct']:.1f}%).")

        payload = {
            "session": {"id": session_id, "track": track,
                        "date": str(session['date']),
                        "driver": session['driver_name']},
            "current": {"lap": current_lap, "tyre": tyre, "tyre_age": cur_age,
                        "battery_mj": round(start, 3),
                        "battery_pct": round(start / BATTERY_CAPACITY_MJ * 100.0, 1)},
            "laps_remaining": len(future),
            "config": {
                "era": spec["label"],
                "mgu_k_kw": spec["mgu_k_kw"],
                "capacity_mj": spec["capacity_mj"],
                "reserve_mj": BATTERY_MIN_MJ,
                # Per-lap recharge/deploy caps the projection actually
                # applies (the era spec's own cap; the 2026 PU has no fixed
                # per-lap deploy quota -- deployment is store-limited).
                "per_lap_recharge_mj": min(spec["harvest_limit_mj"], RECOVER_FLOW_CAP_MJ),
                "deploy_allowance_mj": spec["deploy_ceiling_mj"],
                "rate_mj_s": 0.12,  # 120 kW = 0.12 MJ/s (deploy & recover)
                "pace_s_per_mj": pace_track,
                "pace_basis": pace_basis,
                "limited_effectiveness": LIMITED_LAP_PACE_EFFECTIVENESS,
                "mode_notes": {
                    "push": "asks the full deploy ceiling every lap; drains the store fast -> energy-limited laps (2026: no fixed per-lap deploy quota — store-limited bursts)",
                    "balanced": "spends ~ the lap's own harvest; SOC floats in a soft 30-80% band, draining below ~30% on faster-than-average laps and banking above ~80% on slower ones",
                    "liftcoast": "early lift-off recovers more (+20% harvest); banks the battery toward ~90% as insurance",
                },
            },
            "modes": results,
            "recommendation": {"mode": best['mode'], "feasible": bool(feasible),
                               "reason": reason},
        }
        return jsonify(payload)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if cursor:
            try: cursor.close()
            except: pass
        if conn:
            try: conn.close()
            except: pass


# ENERGY SANDBOX API (PPT module 02)
#
# Per-sector deployment reallocation inside a fixed per-lap budget: three
# sliders shift the lap's deploy between S1/S2/S3 (the budget itself stays as
# the chosen mode planned it -- deltas are normalised to zero-sum), and the
# backend re-simulates the lap at sector resolution.  Returns per-sector
# deploy / harvest / SOC movement plus warnings when a sector hits a
# regulation or physical limit: FIA per-lap caps, the 120 kW deploy/recover
# flow over the sector's time, store headroom (regen surplus wasted) and the
# management reserve (battery floor -> energy-limited lap).  Pure energy
# model -- the ML lap-time model is not required.

@app.route('/api/strategy/energy-sandbox', methods=['POST'])
def energy_sandbox():
    conn = None
    cursor = None
    t0 = time.perf_counter()
    try:
        body = request.get_json() or {}
        session_id = int(body.get('session_id', 0))
        if session_id <= 0:
            return jsonify({"error": "Missing field: session_id"}), 400
        current_lap = int(body.get('current_lap', 1))
        mode = str(body.get('mode', 'balanced')).lower()
        if mode not in MODES:
            return jsonify({"error": f"Unknown mode '{mode}' — use one of {sorted(MODES)}"}), 400
        deltas_raw = body.get('deltas_mj') or {}
        battery_mj = body.get('battery_mj')

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute("""
            SELECT s.session_id, s.track_name, s.date, s.driver_id, d.driver_name
            FROM sessions s
            JOIN drivers d ON s.driver_id = d.driver_id
            WHERE s.session_id = %s
        """, (session_id,))
        session = cursor.fetchone()
        if session is None:
            return jsonify({"error": f"Session {session_id} not found"}), 404

        spec_key = spec_for_year(session['date'].year if session['date'] else 0)
        spec = PU_SPECS[spec_key]
        capacity = BATTERY_CAPACITY_MJ if BATTERY_CAPACITY_MJ > 0 else 1.0
        reserve = BATTERY_MIN_MJ

        cursor.execute("""
            SELECT lap_id, lap_number, lap_time_ms,
                   sector1_ms, sector2_ms, sector3_ms
            FROM laps
            WHERE session_id = %s AND lap_time_ms > 0
            ORDER BY lap_number
        """, (session_id,))
        all_laps = cursor.fetchall()
        if not all_laps:
            return jsonify({"error": "Session has no timed laps"}), 400

        # Context lap: the last real lap at or before the requested one.
        ctx = None
        for l in all_laps:
            if l['lap_number'] <= current_lap:
                ctx = l
        if ctx is None:
            ctx = all_laps[0]
        lap = ctx['lap_number']
        lap_time_s = (float(ctx['lap_time_ms'] or 0.0) / 1000.0) or 1.0

        cursor.execute("""
            SELECT speed, throttle, brake FROM telemetry
            WHERE lap_id = %s ORDER BY telemetry_id
        """, (ctx['lap_id'],))
        samples = cursor.fetchall()
        if not samples or len(samples) < 2:
            return jsonify({"error": f"No telemetry samples for lap {lap} — pick a lap that has samples"}), 400

        # Starting battery: explicit override, else the stored trace's state
        # at this lap, else a full store (as at lights-out).
        if battery_mj is not None:
            start = min(max(float(battery_mj), reserve), capacity)
        else:
            cursor.execute("""
                SELECT energy_end_mj FROM race_state
                WHERE session_id = %s AND lap_number = %s
            """, (session_id, lap))
            row = cursor.fetchone()
            start = (float(row['energy_end_mj']) if row else DEFAULT_START_SOC_MJ)

        # Baseline lap under the mode: the same single-lap projection the
        # mode what-if uses, so the sandbox reallocates the exact budget that
        # mode planned for this lap.
        regen = regen_from_speed_samples(samples, spec_key)
        trace = project_energy_trace(mode, start, [regen], spec_key, pace_dev_per_lap=[0.0])
        base = trace['laps'][0]
        budget = float(base['deployed_mj'])
        harvest_total = float(base['harvested_mj'])

        # Sector classification: thirds of the sample stream.  The importer
        # stores only ~6 samples/lap and no sector markers, so each sample is
        # assigned to a third of the stream.  A speed drop is credited to the
        # sector that contains its later sample (so a braking zone straddling
        # a sector boundary is not lost), and the raw per-sector shares are
        # blended toward a uniform prior because six samples cannot resolve
        # three sectors reliably (SAMPLING_PRIOR below).
        n = len(samples)
        bounds = [0, round(n / 3.0), round(2.0 * n / 3.0), n]
        if bounds[1] == 0 or bounds[2] == bounds[1]:
            bounds = [0, max(1, n // 3), max(2, 2 * n // 3), n]
        counts = [bounds[k + 1] - bounds[k] for k in range(3)]

        # Sector DURATIONS: measured where this lap carries real FastF1
        # sector times (backfilled laps); otherwise the track's measured mean
        # S1:S2:S3 split of lap time from the training profile (energy_pace.json),
        # and only equal thirds when the track is unknown to that profile.
        real_sec = [ctx.get('sector1_ms'), ctx.get('sector2_ms'),
                    ctx.get('sector3_ms')]
        if all(v is not None for v in real_sec) and sum(real_sec) > 0:
            times = [v / 1000.0 for v in real_sec]
            sector_time_source = 'measured_lap'
        else:
            sh, src = track_sector_profile(session['track_name'])
            times = [sh[k] * lap_time_s for k in range(3)]
            sector_time_source = src

        bin_of = []
        for j in range(n):
            k = 0
            while k < 2 and j >= bounds[k + 1]:
                k += 1
            bin_of.append(k)

        mass = spec['car_mass_kg']
        SAMPLING_PRIOR = 0.60       # weight given to an even split (see above)
        h_loss = [0.0, 0.0, 0.0]
        thr_sum = [0.0, 0.0, 0.0]
        for j in range(1, n):       # speed drops: credit to the later sample's sector
            a, b = samples[j - 1], samples[j]
            va = float(a['speed'] or 0.0) / 3.6
            vb = float(b['speed'] or 0.0) / 3.6
            if vb < va:
                h_loss[bin_of[j]] += 0.5 * mass * (va * va - vb * vb) * 0.8 / 1e6
        for j in range(n):          # throttle presence per sector (all samples)
            thr_sum[bin_of[j]] += float(samples[j].get('throttle') or 0.0)

        def _shares(raw):
            tot = sum(raw)
            norm = [x / tot if tot > 0 else 1.0 / 3.0 for x in raw]
            blend = [SAMPLING_PRIOR / 3.0 + (1.0 - SAMPLING_PRIOR) * w for w in norm]
            s = sum(blend)
            return [w / s for w in blend]

        h_w = _shares(h_loss)
        d_w = _shares(thr_sum)

        # Measured per-sector energy-to-time value: how much each sector's
        # deployment is worth per MJ (training artifact scales the 0.35 s/MJ
        # anchor by each sector's measured full-throttle share).
        sec_pace = track_sector_pace(session['track_name'])
        track_pace_now = track_pace_s_per_mj(session['track_name'])

        base_deploy = [budget * d_w[k] for k in range(3)]
        base_harvest = [harvest_total * h_w[k] for k in range(3)]

        # Sliders: per-sector deploy deltas, reallocated within the lap
        # budget (zero-sum after normalisation -- negative requests are
        # clamped away and the remainder is redistributed by weight).
        delta = [0.0, 0.0, 0.0]
        raw_sum = 0.0
        for k in range(3):
            d = float(deltas_raw.get(f's{k + 1}', 0.0) or 0.0)
            delta[k] = d
            raw_sum += d
        req = [max(0.0, base_deploy[k] + delta[k]) for k in range(3)]
        tot_req = sum(req)
        if tot_req <= 1e-9:
            req = [budget / 3.0] * 3
        deploy_shifted = [budget * req[k] / tot_req for k in range(3)]
        eff_delta = [deploy_shifted[k] - base_deploy[k] for k in range(3)]

        def _sim_sectors(deploy_list, harvest_list):
            # Per-sector guardrails (deploy first, then harvest within the
            # sector -- conservative, because the sector's regen arrives after
            # its straights):
            #   * deploy is hard-capped at what the store holds above the
            #     management reserve (you cannot spend energy that the later
            #     braking has not produced yet) and at the 120 kW flow over
            #     the sector's time (an FIA-era power cap);
            #   * harvest is capped by the 120 kW recovery flow and by store
            #     headroom (surplus regen would be wasted to the friction
            #     brakes).
            # A request that hits a cap is reported as a warning and the
            # shortfall shows up as UNSPENT lap budget.
            soc = start
            min_soc = soc
            delivered_d = 0.0
            delivered_h = 0.0
            warns = []
            rows = []
            for k in range(3):
                t_k = times[k]
                soc0 = soc
                d_req = deploy_list[k]
                d_max = min(d_req, DEPLOY_RATE_MJ_S * t_k, max(0.0, soc - reserve))
                if d_max < d_req - 1e-9:
                    if d_req - DEPLOY_RATE_MJ_S * t_k > 1e-9:
                        warns.append(("warn", "sector_deploy_power",
                                      f"S{k + 1}: {d_req:.2f} MJ deploy exceeds 120 kW over the {t_k:.1f}s sector — capped at {d_max:.2f} MJ."))
                    if d_req - max(0.0, soc - reserve) > 1e-9:
                        warns.append(("warn", "battery_floor",
                                      f"S{k + 1}: only {max(0.0, soc - reserve):.2f} MJ is available above the {reserve:.2f} MJ reserve right now — deploy capped at {d_max:.2f} MJ (energy-limited; reallocate later in the lap or accept the cut)."))
                soc -= d_max
                min_soc = min(min_soc, soc)
                delivered_d += d_max
                h_req = harvest_list[k]
                h_max = min(h_req, RECOVER_RATE_MJ_S * t_k, max(0.0, capacity - soc))
                if h_req - RECOVER_RATE_MJ_S * t_k > 1e-9:
                    warns.append(("warn", "sector_recover_power",
                                  f"S{k + 1}: {h_req:.2f} MJ regen exceeds 120 kW over the {t_k:.1f}s sector — capped at {h_max:.2f} MJ."))
                if h_req - max(0.0, capacity - soc) > 1e-9:
                    warns.append(("info", "regen_surplus",
                                  f"S{k + 1}: store full — {h_req - h_max:.2f} MJ of regen would be wasted to the friction brakes."))
                soc += h_max
                delivered_h += h_max
                rows.append({
                    "sector": k + 1,
                    "time_s": round(t_k, 1),
                    "share": round(times[k] / lap_time_s, 3),
                    "pace_s_per_mj": round(sec_pace[k], 3),
                    "time_credit_s": round(d_max * sec_pace[k], 3),
                    "requested_deploy_mj": round(d_req, 4),
                    "deploy_mj": round(d_max, 4),
                    "harvest_mj": round(h_max, 4),
                    "net_mj": round(d_max - h_max, 4),
                    "start_soc_pct": round(soc0 / capacity * 100.0, 2),
                    "min_soc_pct": round((soc0 - d_max) / capacity * 100.0, 2),
                    "end_soc_pct": round(soc / capacity * 100.0, 2),
                })
            return rows, soc, min_soc, delivered_d, delivered_h, warns

        base_rows, base_end, base_min, _, _, base_warns = _sim_sectors(base_deploy, base_harvest)
        res_rows, res_end, res_min, res_d, res_h, res_warns = _sim_sectors(deploy_shifted, base_harvest)

        # FIA per-lap cap checks (era-dependent).
        lap_warns = []
        dep_cap = spec['deploy_ceiling_mj']
        rec_cap = min(spec['harvest_limit_mj'], RECOVER_FLOW_CAP_MJ)
        if spec_key.startswith('legacy'):
            if budget > dep_cap + 1e-9:
                lap_warns.append(("warn", "fia_deploy_quota",
                                  f"Lap deploy budget {budget:.2f} MJ exceeds the 2014-2025 4 MJ/lap quota — not regulation-compliant."))
        else:
            lap_warns.append(("info", "fia_deploy_quota",
                              "2026 PU: no fixed per-lap deploy quota — deployment is store-limited bursts up to "
                              f"{dep_cap:.1f} MJ/lap."))
        if harvest_total > rec_cap + 1e-9:
            lap_warns.append(("warn", "fia_recharge_cap",
                              f"Lap regen {harvest_total:.2f} MJ exceeds the {rec_cap:.1f} MJ/lap recharge cap — capped."))

        def _warn_json(warns):
            return [{"level": lvl, "code": code, "message": msg} for lvl, code, msg in warns]

        return jsonify({
            "session": {"id": session_id, "track": session['track_name'],
                         "date": str(session['date']), "driver": session['driver_name']},
            "lap": {"lap_number": lap, "lap_time_s": round(lap_time_s, 1),
                     "sector_time_s": [round(x, 1) for x in times],
                     "telemetry_samples": n, "mode": mode},
            "budget": {
                "lap_budget_mj": round(budget, 4),
                "raw_deltas_mj": {"s1": round(delta[0], 3), "s2": round(delta[1], 3), "s3": round(delta[2], 3)},
                "raw_sum_mj": round(raw_sum, 3),
                "zero_sum_after_normalise": True,
            },
            "config": {
                "era": spec['label'],
                "capacity_mj": capacity,
                "reserve_mj": reserve,
                "battery_start_mj": round(start, 3),
                "battery_start_pct": round(start / capacity * 100.0, 1),
                "deploy_rate_mj_s": DEPLOY_RATE_MJ_S,
                "recover_rate_mj_s": RECOVER_RATE_MJ_S,
                "per_lap_recharge_mj": rec_cap,
                "deploy_allowance_mj": dep_cap,
                "pace_s_per_mj": round(track_pace_now, 4),
                "sector_pace_s_per_mj": [round(x, 4) for x in sec_pace],
                "sector_time_source": sector_time_source,
            },
            "baseline": {
                "deploy_mj": round(sum(base_deploy), 4),
                "harvest_mj": round(sum(base_harvest), 4),
                "end_soc_pct": round(base_end / capacity * 100.0, 2),
                "min_soc_pct": round(base_min / capacity * 100.0, 2),
                "sectors": base_rows,
                "warnings": _warn_json(lap_warns + base_warns),
            },
            "result": {
                "deploy_mj": round(res_d, 4),
                "harvest_mj": round(res_h, 4),
                "end_soc_pct": round(res_end / capacity * 100.0, 2),
                "min_soc_pct": round(res_min / capacity * 100.0, 2),
                "unspent_mj": round(max(0.0, budget - res_d), 4),
                "effective_deltas_mj": {"s1": round(eff_delta[0], 3),
                                        "s2": round(eff_delta[1], 3),
                                        "s3": round(eff_delta[2], 3)},
                "sectors": res_rows,
                "warnings": _warn_json(lap_warns + res_warns),
            },
            "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 1),
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if cursor:
            try: cursor.close()
            except: pass
        if conn:
            try: conn.close()
            except: pass


# DRIVER COMPARISON API (race data)
#
# The user flow: pick a season -> pick a track raced that season -> pick
# two or more drivers -> overlay their actual race laps on one chart.

def _sessions_on_track(cursor, year, track):
    """Best session per driver for (year, track) from the database.

    A driver can have several sessions on the same track in one season
    (re-imports create duplicate sessions).  For each driver we keep the
    session with the most timed laps, tie-broken by latest date, so the
    comparison always uses one well-defined race per driver.
    """
    cursor.execute("""
        SELECT d.driver_code, d.driver_name, s.session_id,
               s.session_type, s.date,
               COUNT(l.lap_id) AS laps
        FROM sessions s
        JOIN drivers d ON s.driver_id = d.driver_id
        LEFT JOIN laps l ON l.session_id = s.session_id AND l.lap_time_ms > 0
        WHERE YEAR(s.date) = %s AND s.track_name = %s
        GROUP BY d.driver_code, d.driver_name, s.session_id, s.session_type, s.date
        ORDER BY d.driver_code, s.date DESC
    """, (year, track))
    best = {}
    for row in cursor.fetchall():
        code = row['driver_code']
        key = (row['laps'], row['date'] or datetime.date.min, row['session_id'])
        cur = best.get(code)
        if cur is None or key > cur[0]:
            best[code] = (key, row)
    return {code: row for code, (_, row) in best.items()}


@app.route('/api/comparison/years')
def comparison_years():
    """Seasons that have timed laps in the database (newest first)."""
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT DISTINCT YEAR(s.date) AS year
            FROM sessions s
            JOIN laps l ON l.session_id = s.session_id
            WHERE s.date IS NOT NULL AND l.lap_time_ms > 0
            ORDER BY year DESC
        """)
        years = [r['year'] for r in cursor.fetchall()]
        return jsonify({"years": years})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if cursor:
            try: cursor.close()
            except: pass
        if conn:
            try: conn.close()
            except: pass


@app.route('/api/comparison/tracks')
def comparison_tracks():
    """Tracks with timed laps in a given season."""
    year = request.args.get('year', type=int)
    if not year:
        return jsonify({"error": "year query param required"}), 400
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT DISTINCT s.track_name
            FROM sessions s
            JOIN laps l ON l.session_id = s.session_id
            WHERE YEAR(s.date) = %s AND s.track_name IS NOT NULL
              AND l.lap_time_ms > 0
            ORDER BY s.track_name
        """, (year,))
        tracks = [r['track_name'] for r in cursor.fetchall()]
        return jsonify({"tracks": tracks})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if cursor:
            try: cursor.close()
            except: pass
        if conn:
            try: conn.close()
            except: pass


@app.route('/api/comparison/drivers')
def comparison_drivers():
    """Drivers (best session each) that raced a track in a season."""
    year = request.args.get('year', type=int)
    track = request.args.get('track', '').strip()
    if not year or not track:
        return jsonify({"error": "year and track query params required"}), 400
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        best = _sessions_on_track(cursor, year, track)
        drivers = []
        for code in sorted(best):
            row = best[code]
            drivers.append({
                "code": code,
                "name": row['driver_name'],
                "session_id": row['session_id'],
                "session_type": row['session_type'],
                "date": row['date'].strftime('%Y-%m-%d') if row['date'] else None,
                "laps": row['laps'],
            })
        return jsonify({"drivers": drivers})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if cursor:
            try: cursor.close()
            except: pass
        if conn:
            try: conn.close()
            except: pass


@app.route('/api/comparison/race')
def comparison_race():
    """Actual race laps for two or more drivers on a track in a season.

    drivers is a comma-separated list of driver codes (e.g. VER,LEC).
    Each driver's laps come from their best session on that track/season.
    """
    year = request.args.get('year', type=int)
    track = request.args.get('track', '').strip()
    codes = [c.strip().upper() for c in request.args.get('drivers', '').split(',') if c.strip()]
    if not year or not track or not codes:
        return jsonify({"error": "year, track and drivers query params required"}), 400
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        best = _sessions_on_track(cursor, year, track)
        result = {}
        missing = []
        for code in codes:
            if code not in best:
                missing.append(code)
                continue
            row = best[code]
            cursor.execute("""
                SELECT
                    l.lap_id,
                    l.lap_number,
                    l.lap_time_ms / 1000.0 AS lap_time,
                    l.tyre_compound,
                    l.tyre_age,
                    l.is_valid,
                    MAX(CASE WHEN se.event_type = 'PitStop' THEN 1 ELSE 0 END) AS has_pit_stop
                FROM laps l
                LEFT JOIN strategy_events se ON l.lap_id = se.lap_id
                WHERE l.session_id = %s AND l.lap_time_ms > 0
                GROUP BY l.lap_id, l.lap_number, l.lap_time_ms, l.tyre_compound,
                         l.tyre_age, l.is_valid
                ORDER BY l.lap_number
            """, (row['session_id'],))
            laps = []
            lap_ids = []
            for lap in cursor.fetchall():
                lap_ids.append(lap['lap_id'])
                laps.append({
                    "lap_id": lap['lap_id'],
                    "lap_number": int(lap['lap_number']),
                    "lap_time": float(lap['lap_time']),
                    "tyre_compound": lap['tyre_compound'],
                    "tyre_age": int(lap['tyre_age']) if lap['tyre_age'] is not None else None,
                    "is_valid": bool(lap['is_valid']),
                    "has_pit_stop": bool(lap['has_pit_stop']),
                })

            # Per-lap telemetry aggregates (speed / gear / RPM) for the
            # chart tooltip.  Both live capture and FastF1 imports (sampled
            # telemetry) populate the table; laps without rows simply carry
            # no 'telemetry' key.
            telemetry = {}
            if lap_ids:
                placeholders = ','.join(['%s'] * len(lap_ids))
                cursor.execute(f"""
                    SELECT t.lap_id,
                           AVG(t.speed) AS avg_speed,
                           MAX(t.speed) AS top_speed,
                           AVG(t.gear) AS avg_gear,
                           AVG(t.rpm) AS avg_rpm
                    FROM telemetry t
                    WHERE t.lap_id IN ({placeholders})
                    GROUP BY t.lap_id
                """, lap_ids)
                for t in cursor.fetchall():
                    telemetry[t['lap_id']] = t

            for lap in laps:
                lap_id = lap.pop('lap_id')
                agg = telemetry.get(lap_id)
                if agg is not None:
                    lap['telemetry'] = {
                        "avg_speed": int(round(float(agg['avg_speed']))) if agg['avg_speed'] is not None else None,
                        "top_speed": int(round(float(agg['top_speed']))) if agg['top_speed'] is not None else None,
                        "avg_gear": round(float(agg['avg_gear']), 1) if agg['avg_gear'] is not None else None,
                        "avg_rpm": int(round(float(agg['avg_rpm']))) if agg['avg_rpm'] is not None else None,
                    }
            result[code] = {
                "name": row['driver_name'],
                "session_id": row['session_id'],
                "session_type": row['session_type'],
                "date": row['date'].strftime('%Y-%m-%d') if row['date'] else None,
                "laps": laps,
            }
        payload = {"drivers": result, "year": year, "track": track}
        if missing:
            payload["missing"] = missing
        return jsonify(payload)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if cursor:
            try: cursor.close()
            except: pass
        if conn:
            try: conn.close()
            except: pass


# DRIVER COMPARISON API (models)
@app.route('/api/drivers')
def get_drivers():
    """List per-driver models available for head-to-head comparison."""
    try:
        # no-store: a retrain adds year models, and a browser must never
        # reuse a stale model list across page loads.
        resp = jsonify(driver_comparison.list_driver_models())
        resp.headers['Cache-Control'] = 'no-store'
        return resp
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/drivers/compare')
def compare_drivers_api():
    """Head-to-head: same (track, tyre, age) predicted by each driver's model.

    Optional ?year=<SEASON> compares the same-season per-driver-per-year
    models (apples-to-apples); when a driver has no model for that season
    the code falls back to their aggregate all-seasons model and the
    payload's summary.year_used is None.
    """
    try:
        code_a = request.args.get('driver_a', '').strip()
        code_b = request.args.get('driver_b', '').strip()
        if not code_a or not code_b:
            return jsonify({"error": "driver_a and driver_b query params required"}), 400
        year = request.args.get('year', type=int)
        if 'year' in request.args and year is None:
            return jsonify({"error": "year must be an integer season (e.g. 2021)"}), 400
        resp = jsonify(driver_comparison.compare_drivers(code_a, code_b, year=year))
        resp.headers['Cache-Control'] = 'no-store'
        return resp
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ─────────────────────────────────────────────────────────────
# P0 DUAL-AGENT OVERTAKE API (What If tab)
#
# The overtake models are loaded lazily on first request so the dashboard
# starts fine even before scripts/ml_overtake_predictions.py has run; the
# endpoints then return a clear "run the trainer" error (same pattern as the
# lap predictor's "Model not loaded").
# ─────────────────────────────────────────────────────────────
_OVERTAKE_STATE = {"models": None, "info": None, "error": None}


def _load_overtake():
    """Lazily load (closing, overtake, feature_names), info, error."""
    if _OVERTAKE_STATE["models"] is None and _OVERTAKE_STATE["error"] is None:
        try:
            closing, overtake, fnames, info = \
                overtake_inference.load_overtake_models()
            _OVERTAKE_STATE["models"] = (closing, overtake, fnames)
            _OVERTAKE_STATE["info"] = info
        except FileNotFoundError as exc:
            _OVERTAKE_STATE["error"] = str(exc)
    return (_OVERTAKE_STATE["models"], _OVERTAKE_STATE["info"],
            _OVERTAKE_STATE["error"])


@app.route('/api/overtake/options')
def overtake_options():
    """Drivers with pace models + tracks/tyres the overtake model covers."""
    _models, info, err = _load_overtake()
    drivers = []
    for d in driver_comparison.list_driver_models():
        drivers.append({
            "code": d["code"], "name": d["name"],
            "years": d.get("years", []), "tracks": d.get("tracks", []),
            "mae": d.get("mae"),
        })
    tracks = (overtake_inference.covered_tracks(_models[2])
              if _models else [])
    payload = {
        "drivers": drivers,
        "tracks": tracks,
        "tyres": ["Soft", "Medium", "Hard", "Intermediate", "Wet"],
        "model_loaded": _models is not None,
        "model_error": err,
        "summary": None,
    }
    if info:
        pc = info.get("pair_construction", {})
        m = info.get("metrics", {})
        payload["summary"] = {
            "battle_laps": pc.get("battle_laps"),
            "overtake_labels": pc.get("overtake_labels"),
            "overtake_rate": pc.get("overtake_rate"),
            "closing_mae": m.get("closing_rate", {}).get("mae"),
            "overtake_auc": m.get("overtake", {}).get("auc"),
            "trained_at": info.get("trained_at"),
        }
    resp = jsonify(payload)
    resp.headers['Cache-Control'] = 'no-store'
    return resp


@app.route('/api/calendar')
def api_calendar():
    """Race calendars 2020-2026 annotated with DB coverage.

    No query params: returns every season's rounds (from
    race_calendar.py, the single source of truth), each annotated with
    what the database actually holds for that round — race session
    counts, the DB circuit name(s) it matches, drivers present, best lap
    and max lap — so
    the dashboard calendar doubles as a data-coverage map for the
    overtake tooling.  ?year=YYYY narrows the response to one season.
    """
    year = request.args.get('year', type=int)
    conn = None
    try:
        conn = get_db_connection()
        try:
            if year:
                calendars = {year: race_calendar.annotate_with_db(year, conn=conn)}
            else:
                calendars = {y: race_calendar.annotate_with_db(y, conn=conn)
                             for y in race_calendar.YEARS}
        finally:
            conn.close()
        payload = {"years": race_calendar.YEARS, "calendars": calendars}
        resp = jsonify(payload)
        resp.headers['Cache-Control'] = 'no-store'
        return resp
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/overtake/sessions')
def overtake_sessions():
    """Resolve the two drivers' race sessions for a track/season.

    ?leader=CODE&chaser=CODE&track=&year= returns the best (most timed
    laps, latest date) per-driver race session on that track/season — the
    pair the P1 full-race simulator runs on.  Same selection rule as the
    driver-comparison API (_sessions_on_track).
    """
    leader = request.args.get('leader', '').strip().upper()
    chaser = request.args.get('chaser', '').strip().upper()
    track = request.args.get('track', '').strip()
    year = request.args.get('year', type=int)
    if not leader or not chaser or not track or not year:
        return jsonify({"error": "leader, chaser, track and year query params required"}), 400
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        best = _sessions_on_track(cursor, year, track)
        out = {}
        missing = []
        for code in (leader, chaser):
            row = best.get(code)
            if row is None:
                missing.append(code)
                continue
            cursor.execute("""
                SELECT MAX(l.lap_number) AS max_lap, COUNT(l.lap_id) AS laps
                FROM laps l
                WHERE l.session_id = %s AND l.lap_time_ms > 0
            """, (row['session_id'],))
            span = cursor.fetchone()
            out[code] = {
                "code": code,
                "name": row['driver_name'],
                "session_id": row['session_id'],
                "date": row['date'].strftime('%Y-%m-%d') if row['date'] else None,
                "laps": row['laps'],
                "max_lap": span['max_lap'],
            }
        payload = {"track": track, "year": year, "sessions": out}
        if missing:
            payload["missing"] = missing
        resp = jsonify(payload)
        resp.headers['Cache-Control'] = 'no-store'
        return resp
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if cursor:
            try: cursor.close()
            except: pass
        if conn:
            try: conn.close()
            except: pass


# Season-scan cache: (leader_code, chaser_code, year) -> calibration payload.
# A full scan is ~2-3s per race (light sim runs, models cached in-process),
# so cache per pair/season and only rebuild when the model or data changes.
_CALIBRATION_CACHE = {}


@app.route('/api/overtake/calibration')
def overtake_calibration():
    """Season-wide model-vs-reality calibration for a driver pair.

    ?leader=CODE&chaser=CODE&year=YYYY scans every track where BOTH drivers
    have a race session that season.  For each race, the P0 closing-rate
    model's per-lap predictions are regressed against the two drivers'
    REAL lap-time deltas (calibration_stats in overtake_inference):
    agreement %, sign-flip laps, OLS slope/R^2 and a trust tier — so the
    projected leaderboard can say which races it is trustworthy on.  Races
    where the model's NET direction contradicts the real deltas are
    flagged.  Results are cached per (leader, chaser, year).
    """
    leader = request.args.get('leader', '').strip().upper()
    chaser = request.args.get('chaser', '').strip().upper()
    year = request.args.get('year', type=int)
    if not leader or not chaser or not year:
        return jsonify({"error": "leader, chaser and year query params required"}), 400
    if leader == chaser:
        return jsonify({"error": "leader and chaser must differ"}), 400
    key = (leader, chaser, year)
    cached = _CALIBRATION_CACHE.get(key)
    if cached is not None:
        return jsonify(cached)
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT DISTINCT s.track_name
            FROM sessions s
            JOIN drivers d ON s.driver_id = d.driver_id
            JOIN laps l ON l.session_id = s.session_id AND l.lap_time_ms > 0
            WHERE YEAR(s.date) = %s AND d.driver_code IN (%s, %s)
              AND s.track_name IS NOT NULL
            ORDER BY s.track_name
        """, (year, leader, chaser))
        tracks = [r['track_name'] for r in cursor.fetchall()]
        races = []
        for track in tracks:
            best = _sessions_on_track(cursor, year, track)
            l_row = best.get(leader)
            c_row = best.get(chaser)
            if l_row is None or c_row is None:
                continue
            try:
                cal = overtake_inference.calibrate_race(
                    leader_session_id=l_row['session_id'],
                    chaser_session_id=c_row['session_id'],
                )
            except Exception as exc:
                # Expected for rounds where a driver has no timed laps (e.g.
                # DNS / early DNF sessions are kept in the DB): report the
                # round as an error row rather than a stack-trace flood.
                print(f"[calibration] {track} {year}: {leader}/{chaser} skipped "
                      f"- {exc}")
                cal = {"error": str(exc)}
            races.append({
                "track": track,
                "date": (l_row['date'].strftime('%Y-%m-%d')
                          if l_row['date'] else None),
                "leader_session_id": l_row['session_id'],
                "chaser_session_id": c_row['session_id'],
                **cal,
            })
        tiers = {}
        flagged = []
        for race in races:
            t = race.get('trust', 'insufficient')
            tiers[t] = tiers.get(t, 0) + 1
            if t == 'low' or (t == 'medium' and not race.get('net_direction_ok', True)):
                flagged.append({
                    "track": race['track'],
                    "trust": t,
                    "flips": race.get('sign_flips', 0),
                    "net_direction_ok": race.get('net_direction_ok', True),
                })
        payload = {
            "leader": leader,
            "chaser": chaser,
            "year": year,
            "races": races,
            "trust_counts": tiers,
            "flagged_races": flagged,
        }
        _CALIBRATION_CACHE[key] = payload
        resp = jsonify(payload)
        resp.headers['Cache-Control'] = 'no-store'
        return resp
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if cursor:
            try: cursor.close()
            except: pass
        if conn:
            try: conn.close()
            except: pass


@app.route('/api/overtake/sim', methods=['POST'])
def overtake_sim():
    """Predict one leader/chaser lap (or a short battle) with the P0 model.

    Body fields: leader_code, chaser_code, track_name, lap_number,
    gap_before_s, leader_tyre_compound, chaser_tyre_compound,
    leader_tyre_age, chaser_tyre_age, fuel_diff_kg, energy_diff_mj,
    year (optional), sim_laps (optional, >1 runs a lap-by-lap battle).

    With full_race=true the P1 simulator runs instead: leader_session_id /
    chaser_session_id, start_lap, gap_before_s, end_lap (optional),
    max_laps (optional) — a whole-race, speed-trace-aligned progression
    with sector-level Overtake % (see overtake_inference.simulate_full_race).
    """
    _models, _info, err = _load_overtake()
    if _models is None:
        return jsonify({"error": err or "Overtake model not loaded"}), 500
    try:
        body = request.get_json()
        if body.get('full_race'):
            return _overtake_full_race(body)
        required = ["leader_code", "chaser_code", "track_name",
                    "lap_number", "gap_before_s", "leader_tyre_compound",
                    "chaser_tyre_compound", "leader_tyre_age",
                    "chaser_tyre_age"]
        for f in required:
            if body.get(f) in (None, ""):
                return jsonify({"error": f"Missing field: {f}"}), 400

        leader_code = str(body["leader_code"]).strip().upper()
        chaser_code = str(body["chaser_code"]).strip().upper()
        if leader_code == chaser_code:
            return jsonify({"error": "Pick two different drivers."}), 400
        track_name = str(body["track_name"]).strip()
        lap_number = int(body["lap_number"])
        gap_before = float(body["gap_before_s"])
        leader_tyre = str(body["leader_tyre_compound"]).strip()
        chaser_tyre = str(body["chaser_tyre_compound"]).strip()
        leader_age = float(body["leader_tyre_age"])
        chaser_age = float(body["chaser_tyre_age"])
        fuel_diff = float(body.get("fuel_diff_kg") or 0.0)
        energy_diff = float(body.get("energy_diff_mj") or 0.0)
        year_raw = body.get("year")
        year = int(year_raw) if year_raw not in (None, "") else None
        sim_laps = int(body.get("sim_laps") or 1)

        if gap_before <= 0:
            return jsonify({"error": "gap_before_s must be > 0"}), 400
        if sim_laps < 1 or sim_laps > 60:
            return jsonify({"error": "sim_laps must be 1..60"}), 400

        pace_gap, pace_detail = overtake_inference.compute_pace_gap(
            leader_code=leader_code, chaser_code=chaser_code,
            track_name=track_name, lap_number=lap_number,
            leader_tyre_compound=leader_tyre,
            chaser_tyre_compound=chaser_tyre,
            leader_tyre_age=leader_age, chaser_tyre_age=chaser_age,
            year=year,
        )
        if pace_gap is None:
            return jsonify({"error": str(pace_detail)}), 400

        result = overtake_inference.predict_overtake(
            gap_before_s=gap_before, pace_gap_s=pace_gap,
            chaser_tyre_age=chaser_age, leader_tyre_age=leader_age,
            chaser_tyre_compound=chaser_tyre,
            leader_tyre_compound=leader_tyre,
            fuel_diff_kg=fuel_diff, energy_diff_mj=energy_diff,
            lap_number=lap_number, track_name=track_name, year=year,
        )

        # Short battle simulation: gap progression + pass detection.
        sim = None
        if sim_laps > 1:
            sim = _simulate_battle(
                leader_code=leader_code, chaser_code=chaser_code,
                track_name=track_name, lap_number=lap_number,
                leader_tyre=leader_tyre, chaser_tyre=chaser_tyre,
                leader_age=leader_age, chaser_age=chaser_age,
                gap_before=gap_before, fuel_diff=fuel_diff,
                energy_diff=energy_diff, year=year, sim_laps=sim_laps,
            )

        resp = jsonify({
            "pace_gap_s": pace_gap,
            "pace_gap_detail": pace_detail,
            "single": result,
            "sim": sim,
        })
        resp.headers['Cache-Control'] = 'no-store'
        return resp
    except KeyError as e:
        return jsonify({"error": f"Missing field: {e}"}), 400
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


OVERTAKE_TRIGGER_PROB = 0.5   # pass fires when per-lap probability crosses this
BATTLE_END_GAP_S = 10.0       # battle considered over beyond this gap


def _overtake_full_race(body):
    """P1 full-race branch of /api/overtake/sim.

    Runs overtake_inference.simulate_full_race over the two drivers' real
    race sessions and returns the per-lap, sector-level payload.  Reads
    every P0 feature from the database (tyre / fuel / energy from
    race_state), so the energy_diff term is real whenever the energy
    simulator has run for those sessions.
    """
    try:
        leader_session_id = int(body.get('leader_session_id') or 0)
        chaser_session_id = int(body.get('chaser_session_id') or 0)
        if leader_session_id <= 0 or chaser_session_id <= 0:
            return jsonify({"error": "full_race needs leader_session_id and "
                                      "chaser_session_id (per-driver race "
                                      "sessions on the same track)"}), 400
        if leader_session_id == chaser_session_id:
            return jsonify({"error": "Leader and chaser must be different sessions."}), 400

        start_lap = int(body.get('start_lap') or 1)
        gap = float(body.get('gap_before_s') or 0.0)
        if gap <= 0:
            return jsonify({"error": "gap_before_s must be > 0"}), 400
        end_lap = body.get('end_lap')
        end_lap = int(end_lap) if end_lap not in (None, '') else None
        max_laps = min(120, max(1, int(body.get('max_laps') or 80)))

        # Optional ERS-mode head-to-head: each requested mode re-scores the
        # race with that mode's projected energy trace (only energy_diff_mj
        # changes), so the dashboard can show how the deployment choice
        # shifts the corner pass probabilities.
        modes_raw = body.get('modes')
        modes = None
        if modes_raw:
            modes = [str(m).lower() for m in modes_raw]
            unknown = [m for m in modes if m not in MODES]
            if unknown:
                return jsonify({"error": f"Unknown energy mode(s) "
                                          f"{unknown} — use one of {sorted(MODES)}"}), 400

        # Optional ASYMMETRIC ERS duels: a (leader_spec, chaser_spec) list
        # where each spec is an energy-simulator mode or 'stored' (that
        # driver's real race_state trace).  Runs the race once more per pair
        # with the two cars on different deployments, so the output answers
        # e.g. "when does the overtake happen if the leader defends on
        # Balanced while the chaser attacks on Push".
        ERS_SPECS = set(MODES) | {'stored'}
        mode_pairs_raw = body.get('mode_pairs')
        mode_pairs = None
        if mode_pairs_raw:
            mode_pairs = []
            for mp in mode_pairs_raw:
                if isinstance(mp, dict):
                    lm, cm = mp.get('leader_mode'), mp.get('chaser_mode')
                else:
                    lm, cm = mp[0], mp[1]
                lm, cm = str(lm).lower(), str(cm).lower()
                if lm not in ERS_SPECS or cm not in ERS_SPECS:
                    return jsonify({"error": f"Unknown ERS spec in mode pair "
                                              f"({lm}, {cm}) — each side must be "
                                              f"one of {sorted(ERS_SPECS)}"}), 400
                mode_pairs.append((lm, cm))

        result = overtake_inference.simulate_full_race(
            leader_session_id=leader_session_id,
            chaser_session_id=chaser_session_id,
            start_lap=max(1, start_lap),
            gap_before_s=gap,
            end_lap=end_lap,
            max_laps=max_laps,
            trigger_prob=OVERTAKE_TRIGGER_PROB,
            modes=modes,
            mode_pairs=mode_pairs,
        )
        resp = jsonify({"full_race": result})
        resp.headers['Cache-Control'] = 'no-store'
        return resp
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def _simulate_battle(leader_code, chaser_code, track_name, lap_number,
                     leader_tyre, chaser_tyre, leader_age, chaser_age,
                     gap_before, fuel_diff, energy_diff, year, sim_laps):
    """Lap-by-lap gap progression: closing rate shrinks the gap, a per-lap
    probability crossing OVERTAKE_TRIGGER_PROB swaps the roles.

    After a swap the (new) chaser's and leader's tyre contexts are exchanged
    and the gap resets small, keeping the progression physical (no negative
    gaps, no pass-and-teleport).
    """
    gap = float(gap_before)
    l_code, c_code = leader_code, chaser_code
    l_tyre, c_tyre = leader_tyre, chaser_tyre
    l_age, c_age = float(leader_age), float(chaser_age)
    laps_out = []
    pass_lap = None
    i = 0
    while i < sim_laps and gap <= BATTLE_END_GAP_S:
        i += 1
        L = lap_number + i - 1
        pg, _d = overtake_inference.compute_pace_gap(
            leader_code=l_code, chaser_code=c_code, track_name=track_name,
            lap_number=L, leader_tyre_compound=l_tyre,
            chaser_tyre_compound=c_tyre, leader_tyre_age=l_age,
            chaser_tyre_age=c_age, year=year,
        )
        if pg is None:
            break
        res = overtake_inference.predict_overtake(
            gap_before_s=gap, pace_gap_s=pg,
            chaser_tyre_age=c_age, leader_tyre_age=l_age,
            chaser_tyre_compound=c_tyre, leader_tyre_compound=l_tyre,
            fuel_diff_kg=fuel_diff, energy_diff_mj=energy_diff,
            lap_number=L, track_name=track_name, year=year,
        )
        prob = res["overtake_probability"]
        closing = res["closing_rate_s"]
        passed = prob >= OVERTAKE_TRIGGER_PROB
        laps_out.append({
            "lap": L, "gap_before_s": round(gap, 2),
            "closing_rate_s": round(closing, 3),
            "overtake_probability": prob, "passed": passed,
        })
        if passed:
            pass_lap = L
            # Roles swap: the chaser is now ahead by a small margin.
            gap = max(0.3, gap * 0.35)
            l_code, c_code = c_code, l_code
            l_tyre, c_tyre = c_tyre, l_tyre
            l_age, c_age = c_age, l_age
        else:
            gap = max(0.05, gap - closing)
        l_age += 1
        c_age += 1
    return {
        "laps": laps_out, "pass_lap": pass_lap,
        "simulated_laps": len(laps_out),
        "ended": "pass" if pass_lap else ("blown_open" if gap > BATTLE_END_GAP_S else "ran_out"),
        "final_gap_s": round(gap, 2),
    }


def clickable(url, text=None):
    if text is None:
        text = url
    return f"\033]8;;{url}\033\\{text}\033]8;;\033\\"


if __name__ == '__main__':
    # Windows consoles default to cp1252/cp437 which cannot encode the 🌐
    # banner emoji — force UTF-8 like predict_lap_times.py does.
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    print("=" * 60)
    print("F1 DIGITAL PIT WALL — Starting")
    print("=" * 60)
    print(f"\n🌐  {clickable('http://localhost:5000')}")
    print("\nPress Ctrl+C to stop\n")
    # Loopback only: the Werkzeug debug console is remote code execution if
    # this dev server is reachable off-box.  Serve on a network interface
    # via run_server.py (Waitress, debug off) instead.
    app.run(debug=True, host='127.0.0.1', port=5000)
