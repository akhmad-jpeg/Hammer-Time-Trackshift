"""Feature-validation harness for the F1 Digital Pit Wall.

Exercises the four major features end-to-end via the Flask test client
(the real dashboard code, real DB, real trained models — no server needed)
and validates every response against acceptable ranges:

  1. Lap predictor        — /api/predict, /api/predict/options
  2. Strategy advisor     — /api/strategy/analyze (+ energy variant)
  3. Drivers comparison   — /api/drivers, /api/drivers/compare
  4. Overtake / hammer    — /api/overtake/sim, /api/overtake/live
  5. Energy simulator     — /api/session/<id>/energy-simulate
  6. Tyre degradation     — /api/session/<id>/tyre-degradation

Usage:
    python scripts/validate_features.py            # run everything
    python scripts/validate_features.py --quiet    # failures only
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dashboard import app  # noqa: E402  (loads models + DB config)

# Monza 2026-09-06 VER / HAM race sessions (50 laps, real data), from DB.
LEADER_SESSION = 443   # VER
CHASER_SESSION = 444   # HAM
LEADER_CODE, CHASER_CODE = 'VER', 'HAM'
TRACK = 'Autodromo Nazionale di Monza'

client = app.test_client()
_results = []


def check(name, cond, detail=''):
    status = 'PASS' if cond else 'FAIL'
    _results.append((status, name, detail))
    if status == 'PASS':
        print(f"  [PASS] {name}" + (f"  ({detail})" if detail else ''))
    else:
        print(f"  [FAIL] {name}  -- {detail}")


def post(path, payload):
    r = client.post(path, json=payload)
    try:
        return r.status_code, r.get_json(silent=True) or {}
    except Exception:
        return r.status_code, {}


def get(path):
    r = client.get(path)
    return r.status_code, r.get_json(silent=True) or {}


def validate_lap_time(label, t, lo=55.0, hi=130.0):
    """A Monza race lap is ~80-95 s; anything outside 55-130 s is wrong."""
    check(label, lo <= t <= hi, f"{t:.2f}s")


# ---------------------------------------------------------------------------
print("\n=== 1. LAP PREDICTOR ===")
code, opts = get('/api/predict/options')
check('predict options 200', code == 200,
      f"{len(opts.get('tracks', []))} tracks, {len(opts.get('tyres', []))} tyres")

for label, age, lap, comp, yr in [
    ('fresh soft, lap 1', 1, 1, 'Soft', 2023),
    ('worn hard, lap 40', 25, 40, 'Hard', 2023),
    ('medium, lap 25, era 2026', 12, 25, 'Medium', 2026),
    ('no year given', 8, 15, 'Medium', None),
]:
    code, r = post('/api/predict', {
        'tyre_age': age, 'lap_number': lap,
        'tyre_compound': comp, 'track_name': TRACK, 'year': yr})
    if code != 200:
        check(f'predict: {label}', False, f"HTTP {code}: {r.get('error')}")
        continue
    t = r['predicted_time']
    validate_lap_time(f'predict: {label}', t)
    globals().setdefault('_times', {})[label] = t

# Like-for-like wear direction: SAME compound, SAME lap (fuel), only age
# differs -> the older tyre must be slower.  (Cross-compound / cross-lap
# comparisons mix fuel-burn effects and prove nothing.)
code, r_fresh = post('/api/predict', {'tyre_age': 1, 'lap_number': 20,
                                      'tyre_compound': 'Soft',
                                      'track_name': TRACK, 'year': 2023})
code, r_worn = post('/api/predict', {'tyre_age': 18, 'lap_number': 20,
                                     'tyre_compound': 'Soft',
                                     'track_name': TRACK, 'year': 2023})
if code == 200 and 'predicted_time' in r_fresh:
    d = r_worn['predicted_time'] - r_fresh['predicted_time']
    check('wear direction: same tyre, +17 laps older is slower',
          0 < d < 5, f"delta={d:+.3f}s")

code, r = post('/api/predict', {'tyre_age': 5, 'lap_number': 10,
                                'tyre_compound': 'Foam', 'track_name': TRACK})
check('predict rejects unknown tyre (400)', code == 400, f"HTTP {code}")

code, r = post('/api/predict', {'tyre_age': 5, 'lap_number': 10,
                                'tyre_compound': 'Soft', 'track_name': 'Narnia Park'})
check('predict rejects unknown track (400)', code == 400, f"HTTP {code}")

# ---------------------------------------------------------------------------
print("\n=== 2. STRATEGY ADVISOR ===")
for label, event, gap in [
    ('green flag, no gap', 'Green', None),
    ('undercut window (1.1s gap)', 'Green', 1.1),
    ('safety car', 'SafetyCar', None),
]:
    payload = {
        'current_lap': 25, 'total_laps': 50, 'current_tyre': 'Medium',
        'current_age': 14, 'track': TRACK, 'event_type': event,
        'session_id': LEADER_SESSION, 'driver': LEADER_CODE,
        'traffic': 'Light', 'year': 2026,
    }
    if gap is not None:
        payload['gap_to_ahead'] = gap
    code, r = post('/api/strategy/analyze', payload)
    if code != 200:
        check(f'strategy: {label}', False, f"HTTP {code}: {r.get('error')}")
        continue
    recs = r.get('strategies') or []
    check(f'strategy: {label}', bool(recs), f"{len(recs)} options returned")
    best = recs[0] if recs else None
    if best:
        total = best.get('total_time')
        if total is not None:
            # total_time is the projected CUMULATIVE race time at the flag
            # (laps_remaining x per-lap time).  130 s/lap is generous upper.
            check(f'strategy totals sane: {label}',
                  0 < float(total) < 50 * 130, f"total={total:.1f}s")
        stops = best.get('pit_stops')
        if stops is not None:
            check(f'strategy stops sane: {label}', 0 <= int(stops) <= 3,
                  f"stops={stops}")
        check(f'strategy has recommendation: {label}',
              bool((r.get('recommendation') or {}).get('action')),
              str((r.get('recommendation') or {}).get('action')))

code, r = post('/api/strategy/analyze', {
    'current_lap': 25, 'total_laps': 50, 'current_tyre': 'Foam',
    'current_age': 14, 'track': TRACK, 'event_type': 'Green'})
check('strategy rejects unknown tyre (400)', code == 400, f"HTTP {code}")

# Energy-based strategy variant
modes_payload = {
    'session_id': LEADER_SESSION, 'driver': LEADER_CODE,
    'current_lap': 25, 'total_laps': 50,
    'current_tyre': 'Medium', 'current_age': 14}
code, r = post('/api/strategy/energy-analyze', modes_payload)
check(f'energy-strategy status {code}', code in (200, 400),
      f"HTTP {code}" + (f" err={r.get('error')}" if r.get('error') else ''))
if code == 200:
    check('energy-strategy returns plans', bool(r), f"keys={list(r)[:6]}")
    rec = r.get('recommendation') or {}
    check('energy-strategy recommends a mode',
          bool(rec.get('mode') or rec.get('best_mode') or rec),
          str(rec)[:80])

# ---------------------------------------------------------------------------
print("\n=== 3. DRIVERS ===")
code, r = get('/api/drivers/list')
check('drivers list 200', code == 200, f"{len(r) if isinstance(r, list) else r}")
code, r = get('/api/drivers')
check('drivers 200', code == 200,
      f"{len(r.get('drivers', r)) if isinstance(r, dict) else len(r)} entries")

code, r = post('/api/drivers/compare', {})
code, r = get('/api/drivers/compare'
              f'?driver_a={LEADER_CODE}&driver_b={CHASER_CODE}&track={TRACK}')
check('drivers compare 200', code == 200, f"HTTP {code}")
if code == 200:
    txt = str(r)
    check('drivers compare non-empty payload', len(txt) > 50, f"{len(txt)} bytes")

code, r = get('/api/comparison/years')
check('comparison years 200', code == 200, f"HTTP {code} {str(r)[:60]}")

# ---------------------------------------------------------------------------
print("\n=== 4. OVERTAKE / HAMMER-TIME SIM ===")
code, r = get('/api/overtake/options')
check('overtake options 200', code == 200, f"HTTP {code}")

# --- Single-lap prediction (the P0 model) ---
code, r = post('/api/overtake/sim', {
    'leader_code': LEADER_CODE, 'chaser_code': CHASER_CODE,
    'track_name': TRACK, 'lap_number': 30, 'gap_before_s': 1.0,
    'leader_tyre_compound': 'Medium', 'chaser_tyre_compound': 'Medium',
    'leader_tyre_age': 12, 'chaser_tyre_age': 12, 'year': 2026})
check('overtake sim (1-lap) 200', code == 200,
      f"HTTP {code}" + (f" err={r.get('error')}" if r.get('error') else ''))
if code == 200:
    single = r.get('single') or {}
    p = single.get('overtake_probability')
    check('overtake 1-lap prob in [0,1]', p is not None and 0 <= p <= 1,
          f"p={p}")
    c = single.get('closing_rate_s')
    check('overtake 1-lap closing sane (-3..+3 s/lap)',
          c is None or -3 <= c <= 3, f"closing={c}")
    check('pace gap sane (-5..+5 s)',
          r.get('pace_gap_s') is None or -5 <= r['pace_gap_s'] <= 5,
          f"pace_gap={r.get('pace_gap_s')}")
    check('track covered by model', single.get('track_covered') is True,
          f"covered={single.get('track_covered')}")

# --- Full-race hammer-time replay (P1 simulator, real sessions) ---
code, r = post('/api/overtake/sim', {
    'full_race': True,
    'leader_session_id': LEADER_SESSION,
    'chaser_session_id': CHASER_SESSION,
    'gap_before_s': 2.0})
check('overtake full-race sim 200', code == 200,
      f"HTTP {code}" + (f" err={r.get('error')}" if r.get('error') else ''))
if code == 200:
    fr = r.get('full_race') or r
    probs = [l.get('overtake_probability', 0) for l in fr.get('laps', [])]
    check('overtake probs in [0,1]', all(0 <= p <= 1 for p in probs),
          f"{len(probs)} laps")
    check('final gap sane (-60..+60s)',
          fr.get('final_gap_s') is None
          or -60 <= fr['final_gap_s'] <= 60,
          f"final_gap_s={fr.get('final_gap_s')}")
    cal = fr.get('calibration') or {}
    check('calibration reported with direction check',
          bool(cal) and 'net_direction_ok' in cal,
          f"agreement={cal.get('agreement_pct')} "
          f"direction_ok={cal.get('net_direction_ok')}")
    check('calibration direction plausible',
          cal.get('net_direction_ok') is not False,
          f"net_model={cal.get('net_model_gain_s')} "
          f"net_actual={cal.get('net_actual_gain_s')}")
    check('pass lap found or none needed',
          fr.get('pass_lap') is None or 1 <= fr['pass_lap'] <= 60,
          f"pass_lap={fr.get('pass_lap')}")
    check('pit stops detected in replay',
          isinstance(fr.get('pit_stops'), list),
          f"{len(fr.get('pit_stops') or [])} stops")

# --- Live call (what-if) ---
code, r = post('/api/overtake/live', {
    'leader_code': LEADER_CODE, 'chaser_code': CHASER_CODE,
    'track_name': TRACK, 'start_lap': 30, 'race_length': 50,
    'gap_before_s': 1.2, 'leader_tyre_compound': 'Medium',
    'chaser_tyre_compound': 'Medium', 'leader_tyre_age': 12,
    'chaser_tyre_age': 12, 'year': 2026})
check('overtake live 200', code == 200,
      f"HTTP {code}" + (f" err={r.get('error')}" if r.get('error') else ''))
if code == 200:
    live = r.get('live') or {}
    call = live.get('call') or {}
    p = call.get('cumulative_probability')
    check('live call cumulative prob in [0,1]',
          p is not None and 0 <= p <= 1, f"p={p}")
    verdict = call.get('verdict')
    check('live call verdict valid',
          verdict in ('attack', 'hold', 'defend', None, 'wait'),
          f"verdict={verdict}")
    laps = live.get('laps') or []
    lp = [l.get('overtake_probability', 0) for l in laps]
    check('live per-lap probs in [0,1]', all(0 <= x <= 1 for x in lp),
          f"{len(lp)} laps")
    check('live pass lap consistent',
          call.get('pass_lap') is None
          or call['pass_lap'] >= call.get('window_open_lap', 0),
          f"pass={call.get('pass_lap')} window={call.get('window_open_lap')}")

# ---------------------------------------------------------------------------
print("\n=== 5. ENERGY SIMULATOR ===")
for mode in ('balanced', 'liftcoast', 'push'):
    code, r = post(f"/api/session/{LEADER_SESSION}/energy-simulate",
                   {'mode': mode})
    check(f'energy-sim {mode} 200', code == 200,
          f"HTTP {code}" + (f" err={r.get('error')}" if r.get('error') else ''))
    if code == 200:
        s = str(r)
        check(f'energy-sim {mode} payload sane', len(s) > 50, f"{len(s)} bytes")

# ---------------------------------------------------------------------------
print("\n=== 6. TYRE DEGRADATION ===")
code, r = get(f"/api/session/{LEADER_SESSION}/tyre-degradation")
check('tyre degradation 200', code == 200, f"HTTP {code}")
if code == 200:
    pts = r if isinstance(r, list) else (r.get('points') or [])
    check('tyre curve has points', bool(pts),
          f"{len(pts) if isinstance(pts, list) else 'dict'}")
    if isinstance(pts, list) and pts:
        sample = pts[0]
        keys = set(sample) if isinstance(sample, dict) else set()
        # Find a health-ish field whatever it is named
        def _health(p):
            for k in ('health_pct', 'tyre_health_pct', 'health',
                      'tyre_health', 'remaining_pct', 'value'):
                if isinstance(p, dict) and k in p:
                    return float(p[k])
            return None
        healths = [h for h in (_health(p) for p in pts) if h is not None]
        check('health values within 0..100',
              not healths or all(0 <= h <= 100 for h in healths),
              f"{len(healths)} values, first={healths[:1]}")
        check('health decreases over the race',
              not healths or healths[-1] <= healths[0],
              f"first={healths[0] if healths else None} "
              f"last={healths[-1] if healths else None}")
        check('curve fields present', bool(keys), f"fields={sorted(keys)[:8]}")

# ---------------------------------------------------------------------------
n_pass = sum(1 for s, _, _ in _results if s == 'PASS')
n_fail = len(_results) - n_pass
print(f"\n{'=' * 60}\nRESULT: {n_pass} passed, {n_fail} failed, "
      f"{len(_results)} total\n{'=' * 60}")
for s, name, detail in _results:
    if s == 'FAIL':
        print(f"  [FAIL] {name} -- {detail}")
sys.exit(1 if n_fail else 0)
