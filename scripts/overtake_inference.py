"""Dual-agent overtake model — inference API (P0).

Loads the artifacts trained by scripts/ml_overtake_predictions.py and turns
a leader/chaser head-to-head context into:

  * closing_rate_s      — predicted gap closure during the lap, in seconds
                          (positive = the chaser gains on the leader).
  * overtake_probability — predicted probability the chaser gets past the
                          leader during the lap.

The training set is built from PAIRED laps: two drivers racing the same race,
reconstructed into a per-lap head-to-head via their cumulative race clocks
(see ml_overtake_predictions.py for the full construction).  Both drivers'
pace / tyre / fuel / energy features enter the model, with the per-driver
lap-time machinery (driver_comparison.load_driver_model +
feature_pipeline.construct_prediction_input) supplying the pace-gap term —
this is the "reuse the existing per-driver model machinery" requirement.

Deployment contract (what the P1 simulator and the dashboard use):

    from overtake_inference import load_overtake_models, predict_overtake

    closing, overtake = load_overtake_models()          # (model, model, fnames, info)
    result = predict_overtake(gap_before_s=1.4, pace_gap_s=-0.35, ...)

When the overtake models are missing, load_overtake_models raises a clear
FileNotFoundError pointing at the trainer.  When a track was never seen in
training, predict_overtake still runs (the one-hot track row is all zeros)
but returns track_covered=False so callers can fall back to the documented
pace-gap-only heuristic instead of trusting a structure-less prediction.

NOTE: this module is import-safe (no side effects at import time), the same
convention as driver_comparison.py.
"""

from bisect import bisect_right
from pathlib import Path

import joblib
import pandas as pd

from driver_comparison import load_driver_model
from feature_pipeline import (
    construct_prediction_input,
    covered_tracks as lap_covered_tracks,
    covered_tyres as lap_covered_tyres,
    race_phase_index,
    era_bucket,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OVERTAKE_MODEL_DIR = PROJECT_ROOT / "ml_models" / "overtake"

# Fixed numeric features at the FRONT of the overtake feature vector (in this
# order).  Phase / era / track one-hots follow; sklearn only needs matching
# column NAMES, so construct_pair_row aligns by name against feature_names.
OVERTAKE_BASE_FEATURES = [
    "gap_before_s",          # gap entering the lap (seconds, chaser behind)
    "pace_gap_s",            # leader predicted lap - chaser predicted lap (<0 => chaser faster)
    "chaser_age_minus_leader",  # chaser tyre age - leader tyre age (negative => fresher tyres)
    "chaser_tyre_advantage",    # >0 => chaser on a softer compound than the leader
    "fuel_diff_kg",          # chaser fuel - leader fuel at lap start (negative => lighter chaser)
    "energy_diff_mj",        # chaser SOC - leader SOC entering the lap (negative => less ERS left)
]

# Compound speed order, fastest -> slowest.  "Advantage" = leader index -
# chaser index, so a chaser on a softer tyre than the leader scores > 0.
TYRE_ORDER = [
    "Hypersoft", "Ultrasoft", "Supersoft", "Soft",
    "Medium", "Hard", "Superhard", "Intermediate", "Wet",
]
_TYRE_RANK = {c: i for i, c in enumerate(TYRE_ORDER)}


def tyre_advantage(chaser_compound, leader_compound):
    """Compound-speed advantage of the chaser over the leader (>0 softer)."""
    c = _TYRE_RANK.get(str(chaser_compound).strip(), TYRE_ORDER.index("Medium"))
    l = _TYRE_RANK.get(str(leader_compound).strip(), TYRE_ORDER.index("Medium"))
    return l - c


def _normalise_track_name(track_name):
    """Same strip + title-case normalisation the lap pipeline uses."""
    return str(track_name).strip().title()


def covered_tracks(feature_names):
    """Sorted track names the overtake model can predict for (title-cased)."""
    return sorted(f.replace("track_", "") for f in feature_names
                  if f.startswith("track_"))


def covered_phases(feature_names):
    """Sorted race-phase buckets present in the feature set."""
    return sorted(f.replace("phase_", "") for f in feature_names
                  if f.startswith("phase_"))


def covered_eras(feature_names):
    """Sorted era buckets present in the feature set."""
    return sorted(f.replace("era_", "") for f in feature_names
                  if f.startswith("era_"))


def construct_pair_row(gap_before_s, pace_gap_s,
                       chaser_tyre_age, leader_tyre_age,
                       chaser_tyre_compound, leader_tyre_compound,
                       fuel_diff_kg, energy_diff_mj,
                       lap_number, track_name, feature_names, year=None):
    """Build a 1-row feature DataFrame aligned to the trained feature_names.

    Mirrors feature_pipeline.construct_prediction_input: start from a
    zero-frame with the trained columns and fill what this row knows, so any
    compound / phase / era / track value absent from training stays a clean
    all-zero term instead of a missing column.
    """
    track_name = _normalise_track_name(track_name)

    row = pd.DataFrame(0, index=[0], columns=feature_names)

    base = {
        "gap_before_s": float(gap_before_s or 0.0),
        "pace_gap_s": float(pace_gap_s or 0.0),
        "chaser_age_minus_leader": float(chaser_tyre_age - leader_tyre_age),
        "chaser_tyre_advantage": float(
            tyre_advantage(chaser_tyre_compound, leader_tyre_compound)),
        "fuel_diff_kg": float(fuel_diff_kg or 0.0),
        "energy_diff_mj": float(energy_diff_mj or 0.0),
    }
    for name, value in base.items():
        if name in row.columns:
            row[name] = value

    phase_col = f"phase_{race_phase_index(lap_number)}"
    if phase_col in row.columns:
        row[phase_col] = 1

    era_col = f"era_{era_bucket(year)}"
    if era_col in row.columns:
        row[era_col] = 1

    track_col = f"track_{track_name}"
    if track_col in row.columns:
        row[track_col] = 1

    return row


def _pin_in_process(model):
    """Keep a loaded tree model's predict() single-threaded.

    The forest models are TRAINED with n_jobs=-1.  That is fine at
    training time, but sklearn re-reads n_jobs on every predict() — so a
    whole-race sim that scores ~60 laps would spawn and tear down a
    multiprocessing pool PER LAP (roughly 40-50 ms of process overhead on
    Windows alone).  Pinning n_jobs=1 here makes each predict() run
    in-process (no pool), which cuts the P1 race simulator and the season
    calibration scan from tens of seconds to a few.
    """
    if hasattr(model, "n_jobs"):
        try:
            model.n_jobs = 1
        except Exception:
            pass
    return model


def load_overtake_models(models_dir=None):
    """Return (closing_model, overtake_model, feature_names, info_dict).

    Raises FileNotFoundError with a clear message when the artifacts are
    missing (train them with scripts/ml_overtake_predictions.py).
    """
    base = Path(models_dir) if models_dir else OVERTAKE_MODEL_DIR
    closing_path = base / "closing_model.pkl"
    overtake_path = base / "overtake_model.pkl"
    fnames_path = base / "feature_names.pkl"
    if not (closing_path.exists() and overtake_path.exists()
            and fnames_path.exists()):
        raise FileNotFoundError(
            f"Overtake models not found in {base} — run "
            f"scripts/ml_overtake_predictions.py first (it trains the "
            f"closing-rate regressor and the overtake classifier from "
            f"paired leader/chaser race laps)."
        )
    closing_model = _pin_in_process(joblib.load(closing_path))
    overtake_model = _pin_in_process(joblib.load(overtake_path))
    feature_names = joblib.load(fnames_path)
    info = {}
    info_path = base / "model_info.json"
    if info_path.exists():
        try:
            info = json_load(info_path)
        except Exception:
            info = {}
    return closing_model, overtake_model, feature_names, info


def json_load(path):
    import json
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# P1 FULL-RACE SIMULATOR (speed-trace-aligned)
#
# The P0 what-if above predicts ONE lap from hand-entered context.  This
# section is the P1 layer: a whole-race simulator that replays a real
# leader/chaser pair's actual race laps (the importer stores one session per
# driver, so a head-to-head is two sessions on the same track).  Every P0
# feature is then READ from the database instead of typed in:
#
#   * tyre compound / age and fuel load come from each lap's own row;
#   * energy_diff_mj comes from race_state (written by the energy
#     simulator) — races without a synthetic ERS trace impute 0.0 and are
#     counted in `energy_imputed_laps`;
#   * the gap progression INSIDE each lap is aligned to the two cars' speed
#     traces: both traces are resampled onto a common segment grid (by lap
#     fraction, since the importer stores ~6 timestamp-less samples/lap),
#     the chaser's per-segment time gain follows the speed difference at
#     that fraction of the circuit (corners where the chaser is slower open
#     the gap, straights where they are faster close it), and the segment
#     path is scaled so the lap's total equals the P0 closing-rate
#     prediction;
#   * the per-lap overtake probability is split into sector-level Overtake %
#     (S1/S2/S3) weighted by where the chaser is gaining AND where the
#     braking evidence sits — passes happen at braking zones, so corners
#     dominate the share.
# ---------------------------------------------------------------------------

# Pass fires when a lap's overtake probability crosses this (same default as
# the dashboard battle sim; callers may override).
RACE_TRIGGER_PROB = 0.5

# Speed-trace alignment grid: how many lap-fraction segments each lap is
# split into for the intra-lap gap path and the corner heatmap (~2x the
# importer's ~6 samples/lap, so braking zones land on 1-2 segments).
RACE_SEGMENTS = 12

# Sane domain for the energy_diff_mj feature fed to the P0 models.  The
# closing-rate regressor was trained on real race_state deltas (p90 ~0.64 MJ,
# observed range about -2.04 .. +1.62 MJ) and its response is only physically
# sane inside roughly [-2.0, +0.6] MJ: beyond that it extrapolates to
# impossible +/-several-second closings from a handful of anomalous tail rows
# (verified by probing the model across dE).  Clipping keeps the asymmetric
# leader>chaser ERS duels meaningful (a >0.6 MJ chaser advantage still reads
# as an attack window) without the tail artefact.  Values that hit the clip
# are counted and surfaced so the sim never silently over-claims.
ENERGY_DIFF_CLIP_MIN = -2.0
ENERGY_DIFF_CLIP_MAX = 0.6


def load_session_race_laps(session_id, conn=None):
    """Per-lap race context for one driver session.

    Returns a list of dicts (lap_number, lap_time_s, sector_times [s1,s2,s3]
    or None, tyre_compound, tyre_age, fuel_load, is_valid, energy_start_mj,
    energy_end_mj, samples) ordered by lap number.  `samples` is the lap's
    telemetry rows (speed / throttle / brake), empty when the importer
    stored none.  Energy columns come from race_state and are None when the
    synthetic ERS simulator has not run for this session.
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
            SELECT l.lap_id, l.lap_number, l.lap_time_ms,
                   l.sector1_ms, l.sector2_ms, l.sector3_ms,
                   l.tyre_compound, l.tyre_age, l.fuel_load, l.is_valid,
                   r.energy_start_mj, r.energy_end_mj
            FROM laps l
            LEFT JOIN race_state r
                   ON r.session_id = l.session_id
                  AND r.lap_number = l.lap_number
            WHERE l.session_id = %s AND l.lap_time_ms > 0
            ORDER BY l.lap_number
        """, (session_id,))
        laps = cur.fetchall()
        if not laps:
            return []

        ids = [l['lap_id'] for l in laps]
        ph = ','.join(['%s'] * len(ids))
        cur.execute(
            f"SELECT lap_id, speed, throttle, brake FROM telemetry "
            f"WHERE lap_id IN ({ph}) ORDER BY telemetry_id", ids)
        telem = {}
        for row in cur.fetchall():
            telem.setdefault(row['lap_id'], []).append(row)

        out = []
        for l in laps:
            st = None
            if all(l.get(f'sector{k}_ms') is not None for k in (1, 2, 3)):
                st = [float(l[f'sector{k}_ms']) for k in (1, 2, 3)]
            out.append({
                "lap_id": l['lap_id'],
                "lap_number": int(l['lap_number']),
                "lap_time_s": float(l['lap_time_ms'] or 0.0) / 1000.0,
                "sector_times": st,
                "tyre_compound": (str(l['tyre_compound']).strip()
                                   if l['tyre_compound'] else None),
                "tyre_age": float(l['tyre_age'] or 0.0),
                "fuel_load": float(l['fuel_load'] or 0.0),
                "is_valid": bool(l['is_valid']),
                "energy_start_mj": (float(l['energy_start_mj'])
                                     if l['energy_start_mj'] is not None
                                     else None),
                "energy_end_mj": (float(l['energy_end_mj'])
                                   if l['energy_end_mj'] is not None
                                   else None),
                "samples": telem.get(l['lap_id'], []),
            })
        return out
    finally:
        cur.close()
        if close_after:
            conn.close()


def _interp_speed(v, fracs, f):
    """Linear-interpolated speed at lap fraction f (fracs = sample fractions)."""
    if not v or f <= fracs[0]:
        return v[0] if v else 0.0
    if f >= fracs[-1]:
        return v[-1]
    i = bisect_right(fracs, f) - 1
    f0, f1 = fracs[i], fracs[i + 1]
    t = (f - f0) / (f1 - f0) if f1 > f0 else 0.0
    return v[i] + (v[i + 1] - v[i]) * t


def _resample_speeds(samples, n_segments):
    """Average speed per lap-fraction segment of a sampled trace.

    The importer stores ~6 timestamp-less samples/lap, so sample INDEX is
    treated as lap fraction (sample i of n sits at fraction i/(n-1)).  Each
    segment's speed is the mean of 3-point quadrature over the interpolated
    trace, which lets a 5-sample and a 6-sample trace align on the same grid.
    """
    v = [float(s.get('speed') or 0.0) for s in samples]
    n = len(v)
    if n == 0:
        return [0.0] * n_segments
    if n == 1:
        return [v[0]] * n_segments
    fracs = [i / (n - 1) for i in range(n)]
    seg = []
    for j in range(n_segments):
        f0, f1 = j / n_segments, (j + 1) / n_segments
        acc = 0.0
        for q in (0.25, 0.5, 0.75):
            acc += _interp_speed(v, fracs, f0 + (f1 - f0) * q)
        seg.append(acc / 3.0)
    return seg


def _brake_evidence(samples, n_segments):
    """Per-segment braking evidence: km/h speed drops from the raw trace,
    credited to the segment containing the later sample."""
    v = [float(s.get('speed') or 0.0) for s in samples]
    n = len(v)
    ev = [0.0] * n_segments
    if n < 2:
        return ev
    for i in range(n - 1):
        if v[i + 1] < v[i]:
            f = (i + 1) / (n - 1)
            j = min(n_segments - 1, int(f * n_segments))
            ev[j] += (v[i] - v[i + 1]) / 3.6
    return ev


def _sector_bounds(lap):
    """Cumulative lap-fraction boundaries [0, b1, b2, 1] for S1/S2/S3.

    Uses the lap's real FastF1 sector times when present; equal thirds
    otherwise (same fallback the energy sandbox uses)."""
    st = lap.get('sector_times') if lap else None
    if st and all(x and x > 0 for x in st):
        tot = float(sum(st))
        cum, acc = [0.0], 0.0
        for t in st:
            acc += float(t) / tot
            cum.append(min(1.0, max(0.0, acc)))
        return cum
    return [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]


def _aligned_segments(lead_samples, chase_samples, lap_time_s, closing_rate,
                      n_segments=RACE_SEGMENTS):
    """Per-segment gap movement aligned to the two speed traces.

    Returns (segments, sector_gains) where segments is a list of dicts
    {fraction, sector, gain_s} and sector_gains is the chaser's time gained
    in each of S1/S2/S3 (negative = lost there).  The raw trace shape —
    corner losses / straight gains — is scaled so the segments sum exactly
    to the P0 closing rate for the lap; when the trace's net direction
    disagrees with the model (or there is no telemetry) the closing rate is
    spread flat across the lap.
    """
    dt = (float(lap_time_s) or 1.0) / n_segments
    v_lead = _resample_speeds(lead_samples, n_segments)
    v_chase = _resample_speeds(chase_samples, n_segments)
    raw = []
    for j in range(n_segments):
        vl, vc = v_lead[j], v_chase[j]
        g = dt * (1.0 - vl / vc) if (vl > 1.0 and vc > 1.0) else 0.0
        raw.append(g)
    net = sum(raw)
    if abs(net) > 1e-6 and net * closing_rate >= 0.0:
        scale = closing_rate / net
        gains = [g * scale for g in raw]
    else:
        gains = [closing_rate / n_segments] * n_segments

    segments = []
    sec_gain = [0.0, 0.0, 0.0]
    for j in range(n_segments):
        f0, f1 = j / n_segments, (j + 1) / n_segments
        mid = (f0 + f1) / 2.0
        sec = 2 if mid >= 2.0 / 3.0 else (1 if mid >= 1.0 / 3.0 else 0)
        sec_gain[sec] += gains[j]
        segments.append({"fraction": round(f1, 4), "sector": sec + 1,
                         "gain_s": round(gains[j], 4)})
    return segments, sec_gain


def _corner_pass_mass(prob, segments, brake_ev, n_segments=RACE_SEGMENTS):
    """Per-segment pass-probability mass for the corner heatmap.

    Overtakes happen at braking zones, so a lap's overtake probability is
    concentrated on the segments that show braking evidence (speed drops in
    the chaser's trace), boosted where the chaser is also GAINING time there
    (arriving with extra speed into the corner).  Straights carry ~0 mass;
    the per-segment probabilities sum back to the lap's overtake
    probability, so the heatmap reads "where the pass mass sits this lap".
    Falls back to a flat distribution when a lap has no detectable braking.
    """
    gains = [seg.get('gain_s') or 0.0 for seg in segments]
    if len(gains) < n_segments:
        gains += [0.0] * (n_segments - len(gains))
    w = []
    for j in range(n_segments):
        b = brake_ev[j] if j < len(brake_ev) else 0.0
        w.append(b * (1.0 + 2.0 * max(0.0, gains[j])))
    tot = sum(w)
    if tot <= 1e-9:
        return [round(prob / n_segments, 6)] * n_segments
    return [round(prob * x / tot, 6) for x in w]


def _sector_split_from_segments(prob, sec_gain, brake_ev, bounds, n_segments):
    """Split a lap's overtake probability into S1/S2/S3 shares.

    Weight_k = max(0, gain_k) + BRAKE_WEIGHT * brake_k — where the chaser is
    GAINING this lap plus braking evidence (passes happen at braking zones).
    Shares sum to 1; the per-sector probability is share x lap probability,
    so the three sector probabilities sum back to the P0 per-lap value.
    Falls back to equal thirds when the lap has no telemetry.
    """
    BRAKE_WEIGHT = 0.5
    sec_gain_3 = list(sec_gain) + [0.0] * (3 - len(sec_gain))
    brake_3 = [0.0, 0.0, 0.0]
    for j in range(n_segments):
        mid = (j + 0.5) / n_segments
        sec = 2 if mid >= bounds[2] else (1 if mid >= bounds[1] else 0)
        brake_3[sec] += brake_ev[j]
    w = [max(0.0, sec_gain_3[k]) + BRAKE_WEIGHT * brake_3[k] for k in range(3)]
    tot = sum(w)
    if tot <= 1e-9:
        shares = [1.0 / 3.0] * 3
    else:
        shares = [x / tot for x in w]
    probs = [shares[k] * float(prob) for k in range(3)]
    return shares, probs


def project_driver_energy(laps_by_lap, spec_key):
    """Per-mode per-lap ERS projection for ONE driver's session.

    Uses the SAME engine as the energy simulator CLI/API: per-lap regen from
    the speed-drop traces, pace coupling from each lap's deviation vs the
    median of its ~+/-5 neighbours, and project_energy_trace per mode
    starting from a FULL store at lights-out.  Returns (laps_by_mode,
    summary) where laps_by_mode maps mode -> {lap_number: {"start_mj",
    "deployed_mj", "limited", ...}} and summary[mode] is the trace summary
    (deployed totals, final battery %).  Lazy-imports the energy simulator
    so this module stays import-safe.
    """
    from energy_simulator import (project_energy_trace, _speed_drop_regen,
                                  DEFAULT_START_SOC_MJ, MODES)
    ordered = [laps_by_lap[L] for L in sorted(laps_by_lap)]
    regen = [_speed_drop_regen(lap.get('samples', []), spec_key)
             for lap in ordered]
    times = [lap['lap_time_s'] or 0.0 for lap in ordered]
    n = len(times)
    half = 5
    baseline = []
    for i in range(n):
        window = sorted(times[max(0, i - half): min(n, i + half + 1)])
        baseline.append(window[len(window) // 2])
    devs = [(base - t) / base if base > 0 else 0.0
            for base, t in zip(baseline, times)]
    laps_by_mode = {}
    summary = {}
    for mode in sorted(MODES):
        trace = project_energy_trace(mode, DEFAULT_START_SOC_MJ, regen,
                                     spec_key, pace_dev_per_lap=devs)
        laps_by_mode[mode] = {
            lap['lap_number']: rec
            for lap, rec in zip(ordered, trace['laps'])}
        summary[mode] = trace['summary']
    return laps_by_mode, summary


def _mode_energy_projections(lead_laps, chase_laps, spec_key):
    """Per-mode per-lap ERS state for BOTH drivers' sessions.

    Two calls to project_driver_energy (kept for the P1 race sim); the
    leaderboard uses project_driver_energy directly per driver.  Returns
    ((lead_laps_by_mode, lead_summary), (chase_laps_by_mode, chase_summary)).
    """
    return (project_driver_energy(lead_laps, spec_key),
            project_driver_energy(chase_laps, spec_key))


def calibration_stats(model_closings, actual_deltas, laps=None):
    """Regress the P0 closing-rate model against reality for one race.

    model_closings: per-lap PREDICTED closing rate (s; positive = the
        model says the chaser gains on the leader that lap).
    actual_deltas:  per-lap REAL leader-lap - chaser-lap time (s; positive
        = the chaser was actually faster that lap) over the same scored
        laps.

    Returns the OLS slope/intercept/Pearson r/R^2 of actual ~ model, the
    sign-agreement and sign-FLIP counts (with the flip lap numbers — a lap
    where the model says the chaser gains but the real deltas show the
    chaser losing time, or vice versa), the NET modeled vs NET actual
    chaser gain over the window, and a trust tier (high/medium/low /
    insufficient when there are too few scored laps for stable stats).

    The dashboard uses this to say whether a projected leaderboard for a
    given race can be trusted: agreement_pct is how often the model's sign
    matches reality, slope ~1 means the model's scale is right (< 1 =
    overstatement, < 0 = inverted), and net_direction_ok is False when the
    model's overall verdict contradicts the real lap evidence.
    """
    n = len(model_closings)
    base = {"n": n, "trust": "insufficient"}
    if n < 5:
        return base
    mm = sum(model_closings) / n
    ma = sum(actual_deltas) / n
    cov = sum((m - mm) * (a - ma)
              for m, a in zip(model_closings, actual_deltas))
    var_m = sum((m - mm) ** 2 for m in model_closings)
    var_a = sum((a - ma) ** 2 for a in actual_deltas)
    slope = cov / var_m if var_m > 0 else 0.0
    intercept = ma - slope * mm
    r = (cov / (var_m * var_a) ** 0.5) if var_m > 0 and var_a > 0 else 0.0

    # Sign agreement: laps where the model's direction matches reality.
    # Real deltas under AGREE_EPS are lap-time measurement noise, so they
    # count as neutral rather than agreement or flip.
    AGREE_EPS = 0.02
    agree = flip = neutral = 0
    flip_laps = []
    laps = laps or list(range(1, n + 1))
    for m, a, lap in zip(model_closings, actual_deltas, laps):
        if abs(a) < AGREE_EPS:
            neutral += 1
        elif m * a > 0:
            agree += 1
        else:
            flip += 1
            flip_laps.append(lap)

    net_model = sum(model_closings)
    net_actual = sum(actual_deltas)
    # Direction agreement on the NET verdict; when the real net is tiny the
    # direction is a coin flip, so don't penalise the model for it.
    net_ok = (net_model * net_actual > 0) or abs(net_actual) < 1.0
    pct = agree / n if n else 0.0
    if n >= 10 and pct >= 0.65 and r >= 0.25 and net_ok:
        trust = "high"
    elif n >= 10 and pct >= 0.45:
        trust = "medium"
    else:
        trust = "low"
    return {
        "n": n,
        "trust": trust,
        "pearson": round(r, 3),
        "r2": round(r * r, 3),
        "slope": round(slope, 3),
        "intercept": round(intercept, 3),
        "sign_agreements": agree,
        "sign_flips": flip,
        "sign_neutral": neutral,
        "agreement_pct": round(pct, 3),
        "flip_laps": flip_laps,
        "net_model_gain_s": round(net_model, 3),
        "net_actual_gain_s": round(net_actual, 3),
        "net_direction_ok": bool(net_ok),
    }


def calibrate_race(leader_session_id, chaser_session_id, start_lap=1,
                   gap_before_s=1.0, end_lap=None, max_laps=120,
                   models_dir=None, conn=None):
    """Model-vs-reality calibration for ONE race pair (lightweight).

    Runs the same per-lap loop as simulate_full_race with a light payload
    (no segments / corner mass / per-lap rows) and returns ONLY the
    calibration block, so a season-wide scan across every shared track is
    cheap.  See calibration_stats for what the block contains.
    """
    result = simulate_full_race(
        leader_session_id=leader_session_id,
        chaser_session_id=chaser_session_id,
        start_lap=start_lap, gap_before_s=gap_before_s,
        end_lap=end_lap, max_laps=max_laps,
        models_dir=models_dir, conn=conn, light=True,
    )
    return result.get("calibration") or {}


def simulate_full_race(leader_session_id, chaser_session_id, start_lap=1,
                       gap_before_s=1.0, end_lap=None, max_laps=80,
                       trigger_prob=RACE_TRIGGER_PROB,
                       modes=None, mode_pairs=None,
                       models_dir=None, conn=None, light=False):
    """Whole-race leader/chaser simulation, speed-trace aligned.

    Replays the two drivers' real race laps (per-driver sessions on the same
    track): for every lap both have, the P0 model scores the head-to-head
    from the REAL lap context (tyre compound/age, fuel load, ERS state from
    race_state), the intra-lap gap movement follows the aligned speed
    traces, and the lap's overtake probability is split into sector-level
    Overtake %.  A pass fires when the per-lap probability crosses
    ``trigger_prob``; the roles then swap and the race continues with the
    other driver ahead.

    ``modes`` (optional list of energy-simulator mode keys) runs the race
    once MORE per mode with that mode's projected ERS trace (the energy
    simulator's project_energy_trace engine, full store at lights-out) —
    the ONLY feature that changes is energy_diff_mj, so the mode comparison
    isolates "how the ERS deployment choice shifts the overtake
    probabilities".  The primary result uses the STORED race_state trace;
    ``result['modes']`` maps each source key to a lighter per-lap record set
    and ``result['mode_compare']`` summarises them (deployed MJ, final
    battery, mean/max P(overtake), pass lap, per-mode corner heat for the
    circuit overlay).

    ``mode_pairs`` (optional list of (leader_spec, chaser_spec) tuples)
    runs the race once MORE per pair with the two cars on DIFFERENT ERS
    specs — each spec is a projected mode key or the special ``'stored'``
    (that driver's real race_state trace).  Each driver keeps their own
    spec even when a pass swaps the roles (lookups key by driver code).
    The pair's source key is ``"<leader_spec>><chaser_spec>"``; compare
    entries carry ``leader_mode``/``chaser_mode`` and per-driver deployment
    summaries so the table can show both batteries.  This answers "when
    does the overtake happen when the leader defends on one mode while the
    chaser attacks on another".

    ``light=True`` (used by calibrate_race for season scans) drops the
    heavy per-lap rows (segments / corner mass / tyre context) — the
    calibration, energy counters, leaderboard accumulators and mode
    summaries are still computed.

    Returns a dict:
      meta      — sessions, drivers, track, date, year, start/end lap
      laps      — per-lap records (gap before/after, closing rate, overtake
                  probability, pace gap, sector shares + probabilities,
                  intra-lap segment path, tyre/fuel/energy context)
      pass_lap / pass_sector — where the pass happened (None = no pass)
      final_gap_s, laps_simulated, laps_skipped
      sector_totals — aggregate Overtake % by sector across the race
      corner_heat — per-corner (braking-zone) pass mass across the race
      energy     — {real_laps, imputed_laps} for the energy_diff feature
      calibration — model-vs-reality stats (closing rate regressed against
                  the real lap-time deltas, sign-flip laps, trust tier)
      leaderboard — P4 projected finishing order per energy source
      modes / mode_compare — present when ``modes`` was given
    """
    from config import get_db_connection
    own_conn = conn is None
    if own_conn:
        conn = get_db_connection()
    cur = conn.cursor(dictionary=True)
    try:
        cur.execute("""
            SELECT s.session_id, s.driver_id, s.track_name, s.date,
                   d.driver_code, d.driver_name
            FROM sessions s
            JOIN drivers d ON s.driver_id = d.driver_id
            WHERE s.session_id IN (%s, %s)
        """, (int(leader_session_id), int(chaser_session_id)))
        meta = {}
        for row in cur.fetchall():
            meta[int(row['session_id'])] = row
        if len(meta) < 2:
            raise ValueError(
                f"Sessions {leader_session_id} / {chaser_session_id} not found "
                f"(need two per-driver race sessions on the same track).")
        l_meta = meta[int(leader_session_id)]
        c_meta = meta[int(chaser_session_id)]
        if str(l_meta['track_name']).strip().lower() != str(c_meta['track_name']).strip().lower():
            raise ValueError(
                f"Sessions are on different tracks "
                f"({l_meta['track_name']} vs {c_meta['track_name']}) — pick "
                f"the two drivers' sessions for the same race.")

        leader_laps = {lap['lap_number']: lap
                       for lap in load_session_race_laps(leader_session_id, conn=conn)}
        chaser_laps = {lap['lap_number']: lap
                       for lap in load_session_race_laps(chaser_session_id, conn=conn)}
        if not leader_laps or not chaser_laps:
            raise ValueError("One of the sessions has no timed laps.")

        track = str(l_meta['track_name']).strip()
        year = int(l_meta['date'].year) if l_meta['date'] else None
        l_code = str(l_meta['driver_code']).strip().upper()
        c_code = str(c_meta['driver_code']).strip().upper()

        race_end = min(max(leader_laps), max(chaser_laps))
        if end_lap:
            race_end = min(race_end, int(end_lap))
        start = max(1, int(start_lap))
        if start > race_end:
            raise ValueError(
                f"start_lap {start} is beyond the race's last shared lap "
                f"({race_end}).")

        closing_model, overtake_model, feature_names, _info = \
            load_overtake_models(models_dir=models_dir)

        gap0 = float(gap_before_s)
        if gap0 <= 0.0:
            raise ValueError("gap_before_s must be > 0")

        # Energy sources: 'stored' reads the real race_state (written by the
        # energy simulator); each requested mode re-scores the race with that
        # mode's projected ERS trace (project_energy_trace, full store at
        # lights-out), so the ONLY P0 feature that changes is energy_diff_mj.
        from energy_simulator import spec_for_year, MODES as ENERGY_MODES
        spec_key = spec_for_year(year or 0)
        # Symmetric sources: one mode key applied to BOTH cars.
        mode_list = sorted(set(m for m in (modes or [])
                               if m in ENERGY_MODES))
        # Asymmetric sources: (leader_spec, chaser_spec); each spec is a
        # projected mode or 'stored' (that driver's real race_state trace).
        ENERGY_SPECS = set(ENERGY_MODES) | {'stored'}
        pair_list = []
        for item in (mode_pairs or []):
            try:
                lm, cm = str(item[0]).lower(), str(item[1]).lower()
            except (TypeError, IndexError):
                continue
            if lm not in ENERGY_SPECS or cm not in ENERGY_SPECS:
                continue
            if (lm == cm and lm in mode_list) or (lm == 'stored' and cm == 'stored'):
                continue  # already covered by the symmetric list / primary run
            if (lm, cm) not in pair_list:
                pair_list.append((lm, cm))
        proj_modes = sorted(set(mode_list)
                            | {lm for lm, _ in pair_list if lm != 'stored'}
                            | {cm for _, cm in pair_list if cm != 'stored'})
        proj_by_code = {}
        lead_sum = chase_sum = None
        mode_meta = {}
        if proj_modes:
            (lead_proj, lead_sum), (chase_proj, chase_sum) = \
                _mode_energy_projections(leader_laps, chaser_laps, spec_key)
            proj_by_code[l_code] = lead_proj
            proj_by_code[c_code] = chase_proj
            mode_meta = {m: {"deployed_total_mj": lead_sum[m]['deployed_total'],
                             "final_battery_pct": lead_sum[m]['final_battery_pct']}
                         for m in proj_modes}

        def _stored_provider(L, lead_code, chase_code, llap, clap):
            e_lead = llap.get('energy_start_mj')
            if e_lead is None:
                e_lead = llap.get('energy_end_mj')
            e_chase = clap.get('energy_start_mj')
            if e_chase is None:
                e_chase = clap.get('energy_end_mj')
            if e_lead is not None and e_chase is not None:
                return float(e_chase) - float(e_lead), 0.0, False
            return 0.0, 0.0, True

        def _mode_provider(m):
            def prov(L, lead_code, chase_code, llap, clap):
                rec_l = proj_by_code[lead_code][m].get(L)
                rec_c = proj_by_code[chase_code][m].get(L)
                if rec_l is None or rec_c is None:
                    return 0.0, 0.0, True
                diff = float(rec_c['start_mj']) - float(rec_l['start_mj'])
                return diff, float(rec_c['deployed_mj']), False
            return prov

        def _pair_provider(spec_of):
            """Energy provider for an ASYMMETRIC spec map {driver_code: spec}.

            Each driver keeps their own spec even after a pass swaps the
            roles, because the lookup keys on the driver code (which is what
            swaps).  'stored' reads that driver's real race_state trace;
            any other spec reads their projected trace for that mode.
            """
            def side(spec, code, lap_rec, L):
                if spec == 'stored':
                    e = lap_rec.get('energy_start_mj')
                    if e is None:
                        e = lap_rec.get('energy_end_mj')
                    return e, 0.0
                rec = proj_by_code[code][spec].get(L)
                if rec is None:
                    return None, 0.0
                return rec['start_mj'], rec['deployed_mj']

            def prov(L, lead_code, chase_code, llap, clap):
                e_lead, _ = side(spec_of.get(lead_code, 'stored'),
                                 lead_code, llap, L)
                e_chase, dep_c = side(spec_of.get(chase_code, 'stored'),
                                      chase_code, clap, L)
                if e_lead is None or e_chase is None:
                    return 0.0, 0.0, True
                return float(e_chase) - float(e_lead), float(dep_c), False
            return prov

        def _run(energy_provider, light):
            """One full-race pass under one energy source."""
            # Roles: leader/chaser maps swap when a pass happens.  The cars
            # stay fixed — the tyre/fuel/energy context follows the DRIVER's
            # lap, so the side maps ARE the two sessions' lap maps.
            lead_by_lap, chase_by_lap = leader_laps, chaser_laps
            lead_code, chase_code = l_code, c_code
            gap = gap0

            laps_out = []
            skipped = []
            pass_lap = pass_sector = None
            energy_real = energy_imputed = 0
            energy_clipped_laps = 0
            sector_share_acc = [0.0, 0.0, 0.0]
            sector_prob_acc = [0.0, 0.0, 0.0]
            n_scored = 0
            # Calibration pairs: (model closing rate, REAL leader-lap minus
            # chaser-lap time, lap) per scored lap — reality is the actual
            # gap movement, the model's closing rate is the projection.
            calib = []
            # P4 leaderboard accumulators: per-driver modeled time GAINED
            # while chasing (sum of that driver's closing rates, credited
            # each lap to whoever is the chaser that lap — the code keys
            # stay fixed so role swaps keep landing in the right bucket)
            # and per-driver REAL lap time over the scored window.
            gain_acc = {l_code: 0.0, c_code: 0.0}
            time_acc = {l_code: 0.0, c_code: 0.0}
            heat_acc = [{"total": 0.0, "hot_count": 0, "max": 0.0}
                        for _ in range(RACE_SEGMENTS)]

            for L in range(start, race_end + 1):
                if len(laps_out) >= max_laps:
                    break
                llap = lead_by_lap.get(L)
                clap = chase_by_lap.get(L)
                if llap is None or clap is None:
                    skipped.append(L)
                    continue
                if not llap.get('tyre_compound') or not clap.get('tyre_compound'):
                    skipped.append(L)
                    continue

                pace_gap, _pd = compute_pace_gap(
                    leader_code=lead_code, chaser_code=chase_code,
                    track_name=track, lap_number=L,
                    leader_tyre_compound=llap['tyre_compound'],
                    chaser_tyre_compound=clap['tyre_compound'],
                    leader_tyre_age=llap['tyre_age'],
                    chaser_tyre_age=clap['tyre_age'],
                    year=year,
                )
                if pace_gap is None:
                    skipped.append(L)
                    continue

                fuel_diff = float(clap['fuel_load'] or 0.0) \
                    - float(llap['fuel_load'] or 0.0)
                energy_diff, deployed_mj, imputed = energy_provider(
                    L, lead_code, chase_code, llap, clap)
                if imputed:
                    energy_imputed += 1
                else:
                    energy_real += 1

                res = _predict_pair_with(
                    closing_model, overtake_model, feature_names,
                    gap_before_s=gap, pace_gap_s=pace_gap,
                    chaser_tyre_age=clap['tyre_age'],
                    leader_tyre_age=llap['tyre_age'],
                    chaser_tyre_compound=clap['tyre_compound'],
                    leader_tyre_compound=llap['tyre_compound'],
                    fuel_diff_kg=fuel_diff, energy_diff_mj=energy_diff,
                    lap_number=L, track_name=track, year=year,
                )
                closing = res['closing_rate_s']
                prob = res['overtake_probability']
                lap_energy_clipped = bool(res.get('energy_clipped'))
                if lap_energy_clipped:
                    energy_clipped_laps += 1

                gain_acc[chase_code] += closing
                time_acc[lead_code] += llap.get('lap_time_s') or 0.0
                time_acc[chase_code] += clap.get('lap_time_s') or 0.0
                calib.append((closing,
                              (llap.get('lap_time_s') or 0.0)
                              - (clap.get('lap_time_s') or 0.0), L))

                lap_time_s = llap.get('lap_time_s') or 0.0
                segments, sec_gain = _aligned_segments(
                    llap.get('samples', []), clap.get('samples', []),
                    lap_time_s, closing)
                bounds = _sector_bounds(llap)
                brake_ev = _brake_evidence(clap.get('samples', []),
                                           RACE_SEGMENTS)
                shares, sec_probs = _sector_split_from_segments(
                    prob, sec_gain, brake_ev, bounds, RACE_SEGMENTS)

                # Corner heatmap: the lap's overtake probability concentrated
                # on braking-zone segments, plus the lap's "hot corner".
                corner_mass = _corner_pass_mass(prob, segments, brake_ev)
                hot_idx = (max(range(len(corner_mass)),
                               key=lambda j: corner_mass[j])
                           if any(m > 0 for m in corner_mass) else None)
                hot_zone = None
                if hot_idx is not None:
                    hot_zone = {
                        "segment": hot_idx + 1,
                        "fraction": round((hot_idx + 0.5) / len(corner_mass), 3),
                        "sector": segments[hot_idx]["sector"],
                        "probability": corner_mass[hot_idx],
                    }

                passed = prob >= trigger_prob
                if passed:
                    pass_lap = L
                    pass_sector = (int(max(range(3), key=lambda k: sec_probs[k]))
                                   + 1)

                for j, m in enumerate(corner_mass):
                    heat_acc[j]["total"] += m
                    heat_acc[j]["max"] = max(heat_acc[j]["max"], m)
                    if hot_idx == j:
                        heat_acc[j]["hot_count"] += 1

                for k in range(3):
                    sector_share_acc[k] += shares[k]
                    sector_prob_acc[k] += sec_probs[k]
                n_scored += 1

                rec = {
                    "lap": L,
                    "energy_clipped": lap_energy_clipped,
                    "gap_before_s": round(gap, 3),
                    "gap_after_s": None,
                    "closing_rate_s": closing,
                    "overtake_probability": prob,
                    "pace_gap_s": round(pace_gap, 4),
                    "energy_diff_mj": round(energy_diff, 4),
                    "deployed_mj": round(deployed_mj, 4),
                    "sectors": [{"sector": k + 1,
                                  "share": round(shares[k], 4),
                                  "probability": round(sec_probs[k], 4)}
                                 for k in range(3)],
                    "corner_mass": corner_mass,
                    "hot_zone": hot_zone,
                    "passed": passed,
                    "pass_sector": pass_sector if passed else None,
                }
                if not light:
                    rec.update({
                        "fuel_diff_kg": round(fuel_diff, 3),
                        "leader": {"code": lead_code,
                                    "tyre": llap['tyre_compound'],
                                    "tyre_age": llap['tyre_age']},
                        "chaser": {"code": chase_code,
                                    "tyre": clap['tyre_compound'],
                                    "tyre_age": clap['tyre_age']},
                    })
                    # Intra-lap gap path at segment resolution (dashboard's
                    # in-lap chart): running gap after each segment.
                    path = []
                    g = gap
                    for seg in segments:
                        g = max(0.02, g - seg['gain_s'])
                        path.append({"fraction": seg['fraction'],
                                     "sector": seg['sector'],
                                     "gap_s": round(g, 3)})
                    rec["segments"] = path
                laps_out.append(rec)

                if passed:
                    # Roles swap; the (new) leader is the car that just got
                    # past, so from the NEXT lap its session's context feeds
                    # the leader side.  Gap resets small, as in the battle sim.
                    lead_by_lap, chase_by_lap = chase_by_lap, lead_by_lap
                    lead_code, chase_code = chase_code, lead_code
                    gap = max(0.3, gap * 0.35)
                else:
                    gap = max(0.05, gap - closing)
                laps_out[-1]["gap_after_s"] = round(gap, 3)

            totals = []
            if n_scored:
                for k in range(3):
                    totals.append({
                        "sector": k + 1,
                        "share": round(sector_share_acc[k] / n_scored, 4),
                        "mean_probability": round(sector_prob_acc[k] / n_scored, 4),
                        "total_probability": round(sector_prob_acc[k], 4),
                    })

            corner_heat = [{
                "segment": j + 1,
                "fraction_start": round(j / RACE_SEGMENTS, 3),
                "fraction_end": round((j + 1) / RACE_SEGMENTS, 3),
                "fraction_mid": round((j + 0.5) / RACE_SEGMENTS, 3),
                "total_probability": round(heat_acc[j]["total"], 6),
                "mean_probability": round(heat_acc[j]["total"] / max(1, n_scored), 6),
                "max_probability": round(heat_acc[j]["max"], 6),
                "hot_count": heat_acc[j]["hot_count"],
            } for j in range(RACE_SEGMENTS)]

            return {
                "laps": laps_out,
                "pass_lap": pass_lap,
                "pass_sector": pass_sector,
                "final_gap_s": round(gap, 3),
                "laps_simulated": len(laps_out),
                "laps_skipped": skipped,
                "sector_totals": totals,
                "corner_heat": corner_heat,
                "energy": {"real_laps": energy_real,
                            "imputed_laps": energy_imputed},
                "energy_clipped_laps": energy_clipped_laps,
                "mean_probability": (round(sum(l['overtake_probability']
                                               for l in laps_out) / max(1, n_scored), 6)
                                      if laps_out else 0.0),
                "max_probability": (max(l['overtake_probability']
                                         for l in laps_out)
                                     if laps_out else 0.0),
                # Driver ahead at the flag (roles may have swapped on a pass).
                "final_leader": lead_code,
                "calibration": calibration_stats(
                    [c for c, _, _ in calib],
                    [a for _, a, _ in calib],
                    [l for _, _, l in calib]),
            }, gain_acc, time_acc

        def _finishing_order(run, gain_acc, time_acc, codes):
            """P4: projected finishing order for one energy source.

            Projected times are anchored on the REAL race total of the driver
            ahead at the flag (the sim's final_leader), with the other
            driver at that anchor PLUS the sim's final modeled gap — the
            standard "leader's race time + gaps" projection, so the P1/P2
            margin always equals the gap chart's final value.  real_s and
            gained_s are kept as evidence columns (raw lap totals over the
            scored window, and the seconds the model had that driver gain
            while chasing).  The source's calibration block rides along so
            the card can badge how much the projection can be trusted.
            """
            ahead = run.get("final_leader")
            if ahead not in codes:
                ahead = codes[0]
            behind = codes[1] if codes[1] != ahead else codes[0]
            base = time_acc.get(ahead, 0.0)
            margin = abs(float(run.get("final_gap_s") or 0.0))
            rows = [
                {
                    "code": ahead,
                    "position": 1,
                    "projected_s": round(base, 3),
                    "real_s": round(time_acc.get(ahead, 0.0), 3),
                    "gained_s": round(gain_acc.get(ahead, 0.0), 3),
                },
                {
                    "code": behind,
                    "position": 2,
                    "projected_s": round(base + margin, 3),
                    "real_s": round(time_acc.get(behind, 0.0), 3),
                    "gained_s": round(gain_acc.get(behind, 0.0), 3),
                },
            ]
            cal = run.get("calibration") or {}
            return {
                "order": rows,
                "margin_s": round(margin, 3),
                "laps_scored": run.get("laps_simulated", 0),
                "pass_lap": run.get("pass_lap"),
                "trust": cal.get("trust", "insufficient"),
                "calibration": cal,
            }

        primary, p_gain, p_time = _run(_stored_provider, light=light)
        result = {
            "meta": {
                "leader": {"code": l_code,
                            "name": l_meta['driver_name'],
                            "session_id": int(leader_session_id)},
                "chaser": {"code": c_code,
                            "name": c_meta['driver_name'],
                            "session_id": int(chaser_session_id)},
                "track": track,
                "date": str(l_meta['date']),
                "year": year,
                "start_lap": start,
                "end_lap": race_end,
            },
            **{k: primary[k] for k in
               ("laps", "pass_lap", "pass_sector", "final_gap_s",
                "laps_simulated", "laps_skipped", "sector_totals",
                "corner_heat", "energy", "calibration")},
        }

        # Every requested extra energy source, symmetric or not, is one
        # labelled pass: a symmetric mode's label is the mode key; an
        # asymmetric pair's label is "<leader_spec>><<chaser_spec>" (e.g.
        # push>balanced = leader attacks while the chaser holds balanced).
        energy_runs = [(m, m, _mode_provider(m)) for m in mode_list]
        energy_runs += [(lm, cm, _pair_provider({l_code: lm, c_code: cm}))
                        for (lm, cm) in pair_list]
        if energy_runs:
            modes_out = {}
            compare = []
            mode_gain = {}
            mode_time = {}
            for lm, cm, prov in energy_runs:
                label = lm if lm == cm else f"{lm}>{cm}"
                run, g_acc, t_acc = _run(prov, light=True)
                modes_out[label] = run
                mode_gain[label] = g_acc
                mode_time[label] = t_acc
                entry = {
                    "mode": label,
                    "symmetric": lm == cm,
                    **{k: run[k] for k in
                       ("pass_lap", "pass_sector", "final_gap_s",
                        "laps_simulated", "mean_probability",
                        "max_probability", "energy_clipped_laps")},
                    "deployed_total_mj": round(
                        sum(l['deployed_mj'] for l in run['laps']), 3),
                    "energy_diff_mean_mj": round(
                        sum(l['energy_diff_mj'] for l in run['laps'])
                        / max(1, len(run['laps'])), 4),
                    "corner_heat": run["corner_heat"],
                    "calibration": run.get("calibration") or {},
                }
                if lm == cm:
                    entry["final_battery_pct"] = mode_meta[lm]["final_battery_pct"]
                else:
                    entry["leader_mode"] = lm
                    entry["chaser_mode"] = cm
                    if lead_sum is not None and lm != 'stored':
                        entry["leader_deployed_mj"] = round(
                            lead_sum[lm]['deployed_total'], 3)
                        entry["leader_final_battery_pct"] = round(
                            lead_sum[lm]['final_battery_pct'], 1)
                    if chase_sum is not None and cm != 'stored':
                        entry["chaser_deployed_mj"] = round(
                            chase_sum[cm]['deployed_total'], 3)
                        entry["chaser_final_battery_pct"] = round(
                            chase_sum[cm]['final_battery_pct'], 1)
                compare.append(entry)
            result["modes"] = modes_out
            result["mode_compare"] = compare

        # P4 projected leaderboard: finishing order per energy source
        # (stored race_state trace + each requested ERS mode / asymmetric
        # pair).  The projected times are the drivers' real laps over the
        # scored window MINUS the modeled seconds gained while chasing, so a
        # position swap means the source's deployment actually flips the
        # classification.
        leaderboard = {
            "window": {"start_lap": start, "end_lap": race_end},
            "initial_leader": l_code,
            "initial_chaser": c_code,
            "sources": {"stored": _finishing_order(
                primary, p_gain, p_time, (l_code, c_code))},
            "order": ["stored"],
        }
        for label, run, g_acc, t_acc in (
                [(m, modes_out[m], mode_gain[m], mode_time[m])
                 for m in mode_list]
                + [(f"{lm}>{cm}", modes_out[f"{lm}>{cm}"],
                    mode_gain[f"{lm}>{cm}"], mode_time[f"{lm}>{cm}"])
                   for (lm, cm) in pair_list]):
            leaderboard["sources"][label] = _finishing_order(
                run, g_acc, t_acc, (l_code, c_code))
            leaderboard["order"].append(label)
        result["leaderboard"] = leaderboard
        return result
    finally:
        cur.close()
        if own_conn:
            conn.close()


# Per-process cache for per-driver pace models, keyed by (code, year).  The
# P1 race simulator scores every lap of a whole race (dozens of laps), and
# loading a model artifact per lap would dominate the runtime; the cache
# makes each (driver, season) model load exactly once per process.
_PACE_MODEL_CACHE = {}


def _load_driver_model_cached(code, year=None):
    """load_driver_model with a per-process cache (same fallback semantics)."""
    key = (str(code).strip().upper(), year)
    if key not in _PACE_MODEL_CACHE:
        model, fnames, info, used_year = load_driver_model(code, year=year)
        # Keep per-driver tree models single-threaded too (see _pin_in_process).
        _PACE_MODEL_CACHE[key] = (_pin_in_process(model), fnames, info,
                                  used_year)
    return _PACE_MODEL_CACHE[key]


def compute_pace_gap(leader_code, chaser_code, track_name, lap_number,
                     leader_tyre_compound, chaser_tyre_compound,
                     leader_tyre_age, chaser_tyre_age, year=None):
    """Predicted leader lap - predicted chaser lap (s) from per-driver models.

    Reuses the exact per-driver machinery used by driver_comparison.py: each
    side loads their per-driver(-per-year) model and predicts their lap time
    on the shared (track, tyre compound, tyre age, lap) context.  The
    year-specific model is preferred; when it exists but does not cover the
    track/tyre (e.g. the 2026 VER model skips Monaco), the driver's aggregate
    model is tried as a fallback — the same fallback driver_comparison uses.

    Returns (pace_gap_s, detail_dict) on success, or (None, reason_str) when
    either driver cannot be scored on this context (no model, or the track /
    compound was never in that driver's training data).
    """
    track_norm = _normalise_track_name(track_name)

    def _side_time(code, tyre_compound, tyre_age, note):
        model, fnames, info, used_year = None, None, {}, None
        try:
            model, fnames, info, used_year = _load_driver_model_cached(code, year)
        except FileNotFoundError as exc:
            return None, str(exc)
        # Year model loaded but lacks the track/tyre -> aggregate fallback.
        if (track_norm not in lap_covered_tracks(fnames)
                or str(tyre_compound).strip() not in lap_covered_tyres(fnames)):
            try:
                model, fnames, info, used_year = _load_driver_model_cached(code)
            except FileNotFoundError:
                return None, f"{code}: no model covers {track_norm}"
            if (track_norm not in lap_covered_tracks(fnames)
                    or str(tyre_compound).strip() not in lap_covered_tyres(fnames)):
                return None, f"{code}: no model covers {track_norm} / {tyre_compound}"
        row = construct_prediction_input(
            tyre_age=tyre_age, lap_number=lap_number,
            tyre_compound=tyre_compound, track_name=track_norm,
            feature_names=fnames, year=year,
        )
        pred = float(model.predict(row)[0])
        return pred, (code, used_year)

    leader_t, l_note = _side_time(leader_code, leader_tyre_compound,
                                  leader_tyre_age, "leader")
    chaser_t, c_note = _side_time(chaser_code, chaser_tyre_compound,
                                  chaser_tyre_age, "chaser")
    if leader_t is None:
        return None, l_note
    if chaser_t is None:
        return None, c_note
    detail = {
        "leader_model": l_note,
        "chaser_model": c_note,
        "leader_predicted_s": round(leader_t, 3),
        "chaser_predicted_s": round(chaser_t, 3),
    }
    return round(leader_t - chaser_t, 4), detail


def _predict_pair_with(closing_model, overtake_model, feature_names,
                       gap_before_s, pace_gap_s,
                       chaser_tyre_age, leader_tyre_age,
                       chaser_tyre_compound, leader_tyre_compound,
                       fuel_diff_kg, energy_diff_mj,
                       lap_number, track_name, year):
    """Score one head-to-head lap with ALREADY-LOADED models.

    predict_overtake loads the artifacts then delegates here; the P1 race
    simulator loads them once for the whole race and calls this directly,
    so a multi-lap what-if does not re-read the model files per lap.

    Returns {closing_rate_s, overtake_probability, track_covered}.
    """
    row = construct_pair_row(
        gap_before_s=gap_before_s, pace_gap_s=pace_gap_s,
        chaser_tyre_age=chaser_tyre_age, leader_tyre_age=leader_tyre_age,
        chaser_tyre_compound=chaser_tyre_compound,
        leader_tyre_compound=leader_tyre_compound,
        fuel_diff_kg=fuel_diff_kg, energy_diff_mj=energy_diff_mj,
        lap_number=lap_number, track_name=track_name,
        feature_names=feature_names, year=year,
    )
    # Keep energy_diff inside the domain where the models were trained
    # (see ENERGY_DIFF_CLIP_*); asymmetric ERS what-ifs can otherwise push
    # it to +/-2 MJ where the closing regressor extrapolates to absurd
    # closings.  Flag every clipped lap so callers can report it.
    energy_clipped = (energy_diff_mj < ENERGY_DIFF_CLIP_MIN
                      or energy_diff_mj > ENERGY_DIFF_CLIP_MAX)
    energy_used = max(ENERGY_DIFF_CLIP_MIN,
                      min(ENERGY_DIFF_CLIP_MAX, float(energy_diff_mj or 0.0)))
    row = construct_pair_row(
        gap_before_s=gap_before_s, pace_gap_s=pace_gap_s,
        chaser_tyre_age=chaser_tyre_age, leader_tyre_age=leader_tyre_age,
        chaser_tyre_compound=chaser_tyre_compound,
        leader_tyre_compound=leader_tyre_compound,
        fuel_diff_kg=fuel_diff_kg, energy_diff_mj=energy_used,
        lap_number=lap_number, track_name=track_name,
        feature_names=feature_names, year=year,
    )
    closing = float(closing_model.predict(row)[0])
    if hasattr(overtake_model, "predict_proba"):
        prob = float(overtake_model.predict_proba(row)[0][1])
    else:
        prob = float(overtake_model.predict(row)[0])
    return {
        "closing_rate_s": round(closing, 4),
        "overtake_probability": round(prob, 4),
        "energy_clipped": energy_clipped,
        "track_covered": _normalise_track_name(track_name) in covered_tracks(feature_names),
    }


def predict_overtake(gap_before_s, pace_gap_s,
                     chaser_tyre_age, leader_tyre_age,
                     chaser_tyre_compound, leader_tyre_compound,
                     fuel_diff_kg=0.0, energy_diff_mj=0.0,
                     lap_number=1, track_name="", year=None,
                     models_dir=None):
    """Predict one head-to-head lap.

    Returns a dict:
      closing_rate_s       — predicted gap closure this lap (s, +ve = chaser gains)
      overtake_probability — predicted pass probability (0..1)
      track_covered        — False when track_name was unseen in training
    """
    closing_model, overtake_model, feature_names, _info = \
        load_overtake_models(models_dir=models_dir)
    return _predict_pair_with(
        closing_model, overtake_model, feature_names,
        gap_before_s=gap_before_s, pace_gap_s=pace_gap_s,
        chaser_tyre_age=chaser_tyre_age, leader_tyre_age=leader_tyre_age,
        chaser_tyre_compound=chaser_tyre_compound,
        leader_tyre_compound=leader_tyre_compound,
        fuel_diff_kg=fuel_diff_kg, energy_diff_mj=energy_diff_mj,
        lap_number=lap_number, track_name=track_name, year=year,
    )