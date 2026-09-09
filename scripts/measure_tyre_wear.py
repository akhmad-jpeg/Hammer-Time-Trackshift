"""Measure per-track tyre degradation multipliers from stored race laps.

Within a stint, lap times drift for two reasons that are both LINEAR in lap
number — fuel burn (car gets faster, negative slope) and tyre wear (car gets
slower, positive slope) — so a single stint cannot separate them, and the DB's
fuel load is SYNTHETIC (fuel_estimation.py: max(0, 110 - 2*lap), identical for
every track), so the fuel correction is a FLAT +|fuel_burn_rate| s/lap added
back to every stint's raw slope (the ML's fleet tyre_age coefficient, treated
by the codebase as fuel).

The DB therefore cannot set ABSOLUTE wear per track — real stints also carry
traffic, safety cars and track evolution.  What it CAN do is RANK tracks: the
median fuel-corrected within-stint slope per track x compound, relative to the
fleet median for that compound, is the per-track wear multiplier.  That
multiplier (with Bayesian shrinkage toward 1.0 for thin cells, and clamped to
a sane band) is applied to the Pirelli-grounded compound rates from
tyre_degradation.py (LOSS_PER_LAP), so the chart shows per-track degradation
that is measured where the data is strong and Pirelli-grounded everywhere.

Output: ml_models/tyre_wear_per_track.json
    { "<track>": { "<compound>": {
          "wear_s_per_lap": float,      # median fuel-corrected raw slope
          "loss_pct_per_lap": float,    # fleet rate * multiplier
          "multiplier": float,          # measured / fleet median (shrunk)
          "stints": int, "laps": int,
          "measured": bool,             # False -> caller falls back to 1.0
    } } }

Run:  python scripts/measure_tyre_wear.py [--min-stints 2] [--min-laps 12]
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import get_db_connection
from tyre_degradation import LOSS_PER_LAP, FALLBACK_PCT, MAX_LOSS_PER_LAP

# The ML's fitted fleet fuel-burn rate (negative: the car gets faster as fuel
# burns).  The DB fuel ramp is synthetic and track-independent, so this is
# added back FLAT to every stint's raw slope.
FUEL_BURN_RATE_S_PER_LAP = -0.0703

# Laps further than this from the stint's median are SC / red-flag / pit-in
# outliers and are dropped before the slope fit.
MAX_LAP_DEVIATION_S = 2.5

# Minimum clean laps a stint needs to contribute a slope.
MIN_STINT_LAPS = 5

# Multiplier shrinkage: mult = (n/(n+K))*raw_ratio + (K/(n+K))*1.0.  K stints
# is the "prior strength" — below it the measured ratio is pulled to 1.0.
SHRINK_K = 6.0

# Clamp the shipped multiplier to this band (0.5x = half fleet wear, 2.0x =
# double) so single noisy stints cannot make a track absurd.
MIN_MULT = 0.5
MAX_MULT = 2.0


def _stints(session_id, cursor):
    """Split one session's laps into stints (compound change or age reset)."""
    cursor.execute("""
        SELECT lap_number, lap_time_ms, tyre_compound, tyre_age, is_valid
        FROM laps
        WHERE session_id = %s AND lap_time_ms > 0
        ORDER BY lap_number
    """, (session_id,))
    laps = cursor.fetchall()
    if not laps:
        return
    cur = None
    prev_age = None
    prev_comp = None
    for l in laps:
        comp = l['tyre_compound']
        age = l['tyre_age']
        new_stint = (cur is None or comp != prev_comp
                     or (age is not None and prev_age is not None
                         and age < prev_age))
        if new_stint:
            if cur is not None and len(cur['laps']) >= MIN_STINT_LAPS:
                yield cur
            cur = {'compound': comp, 'laps': []}
        if l['is_valid'] == 1 and age is not None and age > 1:
            cur['laps'].append((age, float(l['lap_time_ms']) / 1000.0))
        prev_age, prev_comp = age, comp
    if cur is not None and len(cur['laps']) >= MIN_STINT_LAPS:
        yield cur


def _stint_wear_slope(stint_laps):
    """Robust within-stint slope (s/lap) via OLS on the clean laps."""
    times = [t for _, t in stint_laps]
    med = sorted(times)[len(times) // 2]
    clean = [(a, t) for a, t in stint_laps
             if abs(t - med) <= MAX_LAP_DEVIATION_S]
    if len(clean) < MIN_STINT_LAPS:
        return None
    xs = [a for a, _ in clean]
    ys = [t for _, t in clean]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    var = sum((x - mx) ** 2 for x in xs)
    if var == 0:
        return None
    return cov / var


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--min-stints', type=int, default=2)
    ap.add_argument('--min-laps', type=int, default=12)
    args = ap.parse_args()

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT session_id, track_name FROM sessions")
    sessions = cursor.fetchall()

    # per compound: per-track {stints, laps, slopes}
    cells = defaultdict(lambda: defaultdict(
        lambda: {'stints': 0, 'laps': 0, 'slopes': []}))
    # Only dry compounds are measured: Intermediate/Wet stint slopes are
    # dominated by the track Drying (cars get faster), not wear — the fleet
    # median comes out negative.  They keep the base Pirelli rates.
    MEASURED_COMPOUNDS = ('Soft', 'Medium', 'Hard')
    for s in sessions:
        for st in _stints(s['session_id'], cursor):
            comp = st['compound']
            if comp not in MEASURED_COMPOUNDS:
                continue
            slope = _stint_wear_slope(st['laps'])
            if slope is None:
                continue
            cell = cells[s['track_name']][comp]
            cell['stints'] += 1
            cell['laps'] += len(st['laps'])
            # flat fuel correction (synthetic ramp is track-independent)
            cell['slopes'].append(slope - FUEL_BURN_RATE_S_PER_LAP)

    # Fleet median per compound (measured, flat-corrected).
    fleet = {}
    for comp in MEASURED_COMPOUNDS:
        slopes = [sl for tc in cells.values()
                  for sl in (tc.get(comp) or {}).get('slopes', [])]
        if slopes:
            slopes.sort()
            fleet[comp] = slopes[len(slopes) // 2]

    out = {}
    rows = []
    for track, comps in sorted(cells.items()):
        out[track] = {}
        for comp, c in sorted(comps.items()):
            base = float(LOSS_PER_LAP.get(comp, FALLBACK_PCT))
            measured = (comp in MEASURED_COMPOUNDS
                        and c['stints'] >= args.min_stints
                        and c['laps'] >= args.min_laps
                        and comp in fleet and fleet[comp] > 0)
            if measured:
                slopes = sorted(c['slopes'])
                med = slopes[len(slopes) // 2]
                raw_ratio = med / fleet[comp]
                n = float(c['stints'])
                mult = (n / (n + SHRINK_K)) * raw_ratio \
                    + (SHRINK_K / (n + SHRINK_K)) * 1.0
                mult = max(MIN_MULT, min(MAX_MULT, mult))
                # Clamp the shipped rate to the published per-lap band (the
                # same clamp tyre_degradation applies on load) so the
                # artifact itself never stores an "impossible" wear rate.
                loss = round(min(base * mult,
                                 MAX_LOSS_PER_LAP.get(comp, base * mult)), 2)
            else:
                med, mult, loss = None, None, None
            out[track][comp] = {
                'wear_s_per_lap': round(med, 4) if med is not None else None,
                'loss_pct_per_lap': loss,
                'multiplier': round(mult, 3) if mult is not None else None,
                'stints': c['stints'],
                'laps': c['laps'],
                'measured': measured,
            }
            if measured:
                rows.append((track, comp, med, loss, mult, c['stints']))

    rows.sort(key=lambda r: -r[3])
    print(f"Fleet medians (flat fuel-corrected, s/lap): "
          + ", ".join(f"{c}={v:.4f}" for c, v in sorted(fleet.items())))
    print(f"MEASURED cells ({args.min_stints}+ stints, {args.min_laps}+ laps): "
          f"{len(rows)}")
    print(f"{'TRACK':32s} {'COMP':4s} {'S/LAP':>7s} {'%/LAP':>6s} "
          f"{'MULT':>5s} {'ST':>3s}")
    for t, comp, med, loss, mult, st in rows[:36]:
        print(f"{t[:31]:32s} {comp[:4]:4s} {med:7.4f} {loss:6.2f} "
              f"{mult:5.2f} {st:3d}")

    path = Path(__file__).resolve().parent.parent / 'ml_models' \
        / 'tyre_wear_per_track.json'
    path.write_text(json.dumps(out, indent=1, sort_keys=True),
                    encoding='utf-8')
    print(f"\nWrote {path} ({len(out)} tracks; {len(rows)} measured cells)")


if __name__ == '__main__':
    main()