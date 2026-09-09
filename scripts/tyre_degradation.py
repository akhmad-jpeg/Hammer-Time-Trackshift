"""Predictable tyre degradation model — Pirelli-style tyre health.

A deterministic, monotonic, closed-form model of tyre *health* (%) as a
function of (compound, tyre age, track).  Same inputs -> same curve, every
time — deliberately not a fitted black box, so the dashboard chart is
explainable ("the Softs were at 34% health on lap 18") and the deck's
"physics guardrails / monotonic degradation" pitch holds.

HEALTH = max(FLOOR, 100 - loss_per_lap(compound, track) * age)

  * loss_per_lap is the health %/lap for that track & compound.  For
    MEASURED cells it is the Pirelli base rate x the per-track multiplier
    measured from stored race stints (scripts/measure_tyre_wear.py); for
    unmeasured cells it is the Pirelli base x the researched abrasion map.
    The Pirelli base is the "usable life lost per lap" (% of the set's
    usable life consumed each lap on a medium-abrasion track) — the
    number teams actually talk about: Soft 2.8-5.0%/lap, Medium
    1.6-2.8%/lap, Hard 1.1-1.6%/lap.  Mid-range values reproduce the
    measured health after 20 race laps: Soft expired (cliff ~lap 15),
    Medium ~45-65%, Hard ~70-80%.  So health declines LINEARLY, not on a
    slow-then-crash power curve — a Soft that has run 15 laps really is
    nearly done.
  * The PIT WINDOW (40-50% health) is where teams stop: they pit 1-2
    laps before the performance cliff, while the set still has ~40-50%
    of its life.  The cliff itself is the PACE penalty below 40% health
    (tyre_cliff_penalty, 1.5-2.5 s/lap) — the strategy pits before it.
  * HEALTH_FLOOR (10%) is the extreme-gamble / safety-car zone — health
    never displays below it, so the dashboard never shows a tyre at 0%.

The per-lap pace-loss magnitude the health curve implies is grounded in
measured 2026 stint data (F1 Chronicle, via Yahoo Sports, Jul 2026: median
0.063 s/lap Soft / 0.065 Medium / 0.071 Hard across 156 measured stints);
the codebase already cites that band for the advisor's wear layer.

Usage:
    python scripts/tyre_degradation.py                 # table for all compounds
    python scripts/tyre_degradation.py --compound Soft --track Monaco
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from energy_simulator import canonical_track_name

# Usable life LOST PER LAP (%/lap, medium-abrasion track) — measured race
# data: Soft 2.8-5.0, Medium 1.6-2.8, Hard 1.1-1.6.  Mid-range values,
# tuned to reproduce the reported health after 20 race laps:
#   Soft  -> expired / cliff ~lap 15   (4.5%/lap: 10% at 20)
#   Medium-> ~45-65%                   (2.25%/lap: 55% at 20)
#   Hard  -> ~70-80%                   (1.35%/lap: 73% at 20)
# Intermediate/Wet use rain-tyre data (~30-50 lap life).
LOSS_PER_LAP = {
    'Soft': 4.5,
    'Medium': 2.25,
    'Hard': 1.35,
    'Intermediate': 2.0,
    'Wet': 1.8,
}

# Derived: usable life in laps on a MEDIUM-abrasion track = 100 / rate
# (health reaches 0% there; display clamps at HEALTH_FLOOR before it).
TYRE_LIFE = {c: round(100.0 / r) for c, r in LOSS_PER_LAP.items()}

# Published per-lap degradation MAXIMA (the top of the researched range,
# from the same race-data table the base rates are tuned to): Soft 2.8-5.0,
# Medium 1.6-2.8, Hard 1.1-1.6 %/lap.  No track may wear faster than this
# band — measured multipliers (measure_tyre_wear.py) and the researched
# abrasion map are both clamped here, so a Bahrain Soft can never hit the
# display floor twice as fast as real rubber, and the chart never shows a
# long flatline pinned at the floor for an "impossible" stint.
# Intermediate/Wet have no published dry band and keep their base rates.
MAX_LOSS_PER_LAP = {
    'Soft': 5.0,
    'Medium': 2.8,
    'Hard': 1.6,
}

# Per-track measured multipliers (scripts/measure_tyre_wear.py): each
# track x compound's median fuel-corrected within-stint wear slope relative
# to the fleet median, shrunk toward 1.0 for thin cells and clamped to
# [0.5, 2.0].  Applied to LOSS_PER_LAP so degradation is measured where the
# data is strong and Pirelli-grounded everywhere.  Keys use the raw DB
# track names; lookup is alias/case aware like TRACK_ABRASION.
_MEASURED_WEAR = {}
_MEASURE_ARTIFACT = (Path(__file__).resolve().parent.parent
                     / 'ml_models' / 'tyre_wear_per_track.json')
if _MEASURE_ARTIFACT.exists():
    try:
        _MEASURED_WEAR = json.loads(
            _MEASURE_ARTIFACT.read_text(encoding='utf-8'))
    except Exception:
        _MEASURED_WEAR = {}


def _measured_loss(compound, track):
    """The artifact's measured loss %/lap for (compound, track), or None.
    Alias/case aware via canonical_track_name."""
    if not track or not _MEASURED_WEAR:
        return None
    n = canonical_track_name(track)
    for key, cell in _MEASURED_WEAR.items():
        if canonical_track_name(key) == n and compound in cell:
            loss = cell[compound].get('loss_pct_per_lap')
            return float(loss) if loss is not None else None
    return None


def loss_per_lap(compound, track=None):
    """Health %/lap for (compound, track).

    MEASURED cells use the data-derived rate (Pirelli base x per-track
    multiplier from measure_tyre_wear.py) — the measured multiplier already
    IS the track's abrasiveness, so the researched TRACK_ABRASION is NOT
    applied again.  Unmeasured cells (thin data, or Intermediate/Wet) fall
    back to the Pirelli base scaled by the researched abrasion map.
    """
    if track is not None:
        m = _measured_loss(compound, track)
        if m is not None:
            return min(m, MAX_LOSS_PER_LAP.get(compound, m))
    base = float(LOSS_PER_LAP.get(compound, FALLBACK_PCT)) \
        * track_abrasion(track)
    return min(base, MAX_LOSS_PER_LAP.get(compound, base))

# Display/engineering floor (%).  Real F1 tyres are only run into single
# digits during extreme safety-car disruptions, unexpected rain or
# end-of-race gambles — so health is clamped here and NEVER prints 0%.
HEALTH_FLOOR = 10.0

# The performance cliff: below 40% health lap times "suddenly plummet by
# 1.5 to 2.5 seconds per lap" (widely-reported Pirelli behaviour) — that is
# exactly why teams pit 1-2 laps before this threshold (the 40-50% band).
# tyre_cliff_penalty ramps 0 at 40% health up to CLIFF_PENALTY_MAX s/lap at 0%.
CLIFF_HEALTH = 40.0
CLIFF_PENALTY_MAX = 2.5

# Medium-like fallback rate for unknown compounds.
FALLBACK_PCT = 2.25

# Pirelli race-note abrasiveness: >1 wears faster (life shortens), <1 slower.
# Same characterisation as the Strategy Advisor (dashboard.py TRACK_ABRASION),
# resolved alias/case aware so 'Monaco' and 'Circuit de Monaco' agree.
TRACK_ABRASION = {
    'Bahrain International Circuit': 1.25,
    'Jeddah Corniche Circuit': 1.20,
    'Marina Bay': 1.30,
    'Miami Gardens': 1.15,
    'Suzuka International Racing Course': 1.15,
    'Circuit de Barcelona-Catalunya': 1.15,
    'Autodromo Internazionale Enzo e Dino Ferrari': 1.20,
    'Hungaroring': 1.10,
    'Yas Island': 1.05,
    'Baku City Circuit': 1.05,
    'Autódromo Hermanos Rodríguez': 1.10,
    'Circuit de Monaco': 1.15,
    'Silverstone Circuit': 0.90,
    'Sochi Autodrom': 0.90,
    'Autodromo Nazionale di Monza': 0.95,
    'Spa-Francorchamps': 0.95,
    'Circuit Zandvoort': 0.95,
    'Shanghai International Circuit': 1.00,
    'Red Bull Ring': 0.90,
}


def track_abrasion(track):
    """Wear multiplier for a circuit (1.0 when unlisted).

    Alias/case aware via energy_simulator.canonical_track_name, so 'Monaco'
    and 'Circuit de Monaco' resolve to the same 1.15 multiplier.
    """
    if not track:
        return 1.0
    n = canonical_track_name(track)
    for key, val in TRACK_ABRASION.items():
        if canonical_track_name(key) == n:
            return val
    return 1.0


def usable_life(compound, track=None):
    """Laps until a fresh set of `compound` reaches 0% on `track`
    (100 / loss_per_lap)."""
    rate = loss_per_lap(compound, track)
    return 100.0 / rate if rate > 0 else 50.0


def _raw_health(compound, age, track=None):
    """Unfloored linear health 0-100 (the depletion meter before clamping)."""
    age = max(0.0, float(age))
    rate = loss_per_lap(compound, track)
    return 100.0 - rate * age


def tyre_health(compound, age, track=None):
    """Tyre health 0-100 for `age` laps on `compound` at `track`.

    Deterministic and monotonic: health(0)=100, declines LINEARLY at the
    compound's measured loss rate (usable life lost per lap), and clamps
    at HEALTH_FLOOR (10%) — a tyre is never displayed at 0% (teams pit at
    40-50% health; only extreme gambles reach single digits).
    Missing/unknown compound falls back to a Medium-like rate.
    """
    if age is None:
        return None
    return round(max(HEALTH_FLOOR, _raw_health(compound, age, track)), 1)


def tyre_cliff_penalty(compound, age, track=None):
    """Extra s/lap lost when a set falls below the performance cliff.

    Below CLIFF_HEALTH (40%) lap times plummet by 1.5-2.5 s/lap; the penalty
    ramps linearly from 0 at 40% health to CLIFF_PENALTY_MAX at 0%, so the
    advisor's pace layer can make "stay out on a clapped set" genuinely
    expensive and push the strategy to pit before the cliff.  Uses the RAW
    (unfloored) health so the full 0-2.5 s/lap band is expressible.
    """
    if compound is None or age is None:
        return 0.0
    raw_health = _raw_health(compound, age, track)
    if raw_health >= CLIFF_HEALTH:
        return 0.0
    frac = (CLIFF_HEALTH - raw_health) / CLIFF_HEALTH
    return round(CLIFF_PENALTY_MAX * min(1.0, frac), 3)


def health_curve(compound, track=None, laps=60):
    """Full curve (age -> health) up to `laps` for charting / tables."""
    return [tyre_health(compound, age, track) for age in range(laps + 1)]


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--compound", default=None, help="one compound (default: all)")
    ap.add_argument("--track", default=None, help="track name (abrasion scaling)")
    ap.add_argument("--max-age", type=int, default=50)
    args = ap.parse_args()

    compounds = [args.compound] if args.compound else sorted(TYRE_LIFE)
    print(f"Tyre degradation model (Pirelli-style)"
          + (f" @ {args.track}" if args.track else " — fleet base")
          + f" · measured cells use data rates; the researched abrasion map "
            f"applies only as the unmeasured fallback")
    print(f"{'AGE':>4} " + " ".join(f"{c[:6]:>7}" for c in compounds))
    for age in range(0, args.max_age + 1, 2):
        row = [f"{tyre_health(c, age, args.track):>6.1f}%" for c in compounds]
        print(f"{age:>4} " + " ".join(row))
    print("\nLoss per lap (usable-life %/lap): "
          + ", ".join(f"{c} {loss_per_lap(c, args.track):.2f}%"
                      for c in compounds))
    print("Usable life to 0% (display floor " + f"{HEALTH_FLOOR:.0f}%): "
          + ", ".join(f"{c} {usable_life(c, args.track):.0f} laps"
                      for c in compounds))
    print(f"Pit window: teams stop at 40-50% health (the gold band); below "
          f"{CLIFF_HEALTH:.0f}% the cliff penalty ramps to "
          f"{CLIFF_PENALTY_MAX} s/lap (strategy pits before it).")


if __name__ == "__main__":
    main()