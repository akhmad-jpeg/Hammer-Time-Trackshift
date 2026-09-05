import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
import json
from datetime import datetime
from pathlib import Path

from config import get_db_connection
from feature_pipeline import (
    add_tyre_age_interactions,
    race_phase_index,
    era_bucket,
)

# ---------------------------------------------------------------------------
# Model artifacts always live in <project root>/ml_models — never relative to
# the current working directory.  (Running this script from scripts/ vs the
# root used to produce two divergent model directories.)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "ml_models"

# ---------------------------------------------------------------------------
# Training data
#
# Only representative steady-state racing laps are used.  Excluded:
#   * invalid laps (track-limit deletions etc.)
#   * laps outside the 60–180 s window (red-flag laps, formation laps)
#   * pit in-laps  — the lap that carries a real PitStop event
#   * pit out-laps — the lap immediately after a real pit in-lap
#   * the first two laps of every stint (cold-tyre / traffic / race-start
#     laps that are not representative of steady-state pace)
#   * SC / VSC / red-flag laps — laps > 130% of their session's median lap
#     time (red-flag periods are stored session-wide with lap_id = NULL, so
#     they cannot be excluded by lap id; the session-relative time ratio is
#     the reliable signal).
#
# A PitStop event only excludes the laps around it when the stop duration is
# plausible (>= 15 s) or unknown (NULL -- recorded by the importer when the
# box time could not be resolved; the stop still happened).  Implausibly
# short "stops" (data glitches that attach a 2.3 s pit event to a normal
# lap) are ignored so they do not drop valid racing laps.
# ---------------------------------------------------------------------------
TRAINING_QUERY = """
SELECT
    l.lap_time_ms / 1000.0 AS lap_time,
    l.lap_number,
    l.tyre_age,
    l.tyre_compound,
    l.lap_id,
    l.sector1_ms,
    l.sector2_ms,
    l.sector3_ms,
    l.session_id,
    s.track_name,
    s.date AS session_date,
    l.driver_id,
    d.driver_code,
    COALESCE(d.driver_name, d.driver_code) AS driver_name
FROM laps l
JOIN sessions s ON l.session_id = s.session_id
LEFT JOIN drivers d ON l.driver_id = d.driver_id
WHERE l.is_valid = 1
  AND l.lap_time_ms BETWEEN 60000 AND 180000
  -- Pit in-lap: the lap carrying a real PitStop event
  AND l.lap_id NOT IN (
      SELECT lap_id FROM strategy_events
      WHERE event_type = 'PitStop'
        AND (duration_sec IS NULL OR duration_sec >= 15.0)
  )
  -- Pit out-lap: the lap immediately after a real pit in-lap
  AND l.lap_id NOT IN (
      SELECT l2.lap_id
      FROM laps l2
      JOIN strategy_events se ON se.lap_id = l2.lap_id
      WHERE se.event_type = 'PitStop'
        AND (se.duration_sec IS NULL OR se.duration_sec >= 15.0)
        AND l2.session_id = l.session_id
        AND l.lap_number = l2.lap_number + 1
  )
"""

# Laps slower than this multiple of their session's median lap time are
# treated as SC / VSC / red-flag / formation laps and excluded.
SC_VSC_REDFLAG_RATIO = 1.30

# Cold-tyre / traffic / race-start laps dropped from the start of each stint.
STINT_WARMUP_LAPS = 2

# A driver needs at least this many CLEAN training laps to get their own
# model.  Below that the model is too noisy to be a meaningful signature.
MIN_DRIVER_LAPS = 60


def mean_sector_shares(df_laps):
    """Per-track mean S1:S2:S3 time share from rows carrying real sectors.

    Only rows with all three FastF1 sector times present contribute.  Each
    row's share is its own sector ms / lap total, then averaged per track.
    Returns {track_name: [s1_share, s2_share, s3_share]} (sums to 1).
    """
    cols = ['sector1_ms', 'sector2_ms', 'sector3_ms']
    has = df_laps[df_laps[cols].notna().all(axis=1)].copy()
    out = {}
    for trk, g in has.groupby('track_name'):
        g = g[g[cols].sum(axis=1) > 0]
        if g.empty:
            continue
        row_share = g[cols].div(g[cols].sum(axis=1), axis=0)
        out[trk] = row_share.mean().tolist()
    return out


SECTOR_NAMES = ['s1', 's2', 's3']


def encode_features(df, track_terms=True):
    """One-hot tyre / track / race-phase / era and add age interactions.

    Shared by the global model, the sector models and every per-driver /
    per-driver-per-year model so their feature vectors always agree with
    feature_pipeline.construct_prediction_input().  Per-track age terms
    (tyre_age:track_<t>) are only trained by the global model (they feed the
    measured per-track wear export); per-driver models keep just the
    compound terms — one driver's per-track age slopes would be pure noise.
    """
    df = df.copy()
    # Era bucket from the session's date (all row-builders at prediction
    # time go through construct_prediction_input, which sets the same column
    # from the caller's year).
    if 'session_date' in df.columns:
        df['era'] = (pd.to_datetime(df['session_date'], errors='coerce')
                     .dt.year.map(era_bucket))
        df = df.drop(columns=['session_date'])
    elif '_year' in df.columns:
        df['era'] = df['_year'].map(era_bucket)
        df = df.drop(columns=['_year'])
    df_enc = pd.get_dummies(
        df,
        columns=['tyre_compound', 'track_name', 'race_phase', 'era'],
        prefix={'tyre_compound': 'tyre', 'track_name': 'track',
                'race_phase': 'phase', 'era': 'era'},
    )
    return add_tyre_age_interactions(df_enc, track_terms=track_terms)


def clean_training_data(df, verbose=True, label="Training data"):
    """Return the cleaned training DataFrame for a driver subset (or all).

    Applies the shared cleaning used by the global model and every
    per-driver model:

      * normalises track/tyre casing
      * drops the first STINT_WARMUP_LAPS laps of each stint
      * drops SC/VSC/red-flag/formation laps (> SC_VSC_REDFLAG_RATIO x the
        session's median lap time)

    Stints are detected per (session, driver): one driver's pit stop must
    never split another driver's stint, which is why the global model also
    groups by driver_id here.
    """
    df = df.copy()
    df['track_name']    = df['track_name'].str.strip().str.title()
    df['tyre_compound'] = df['tyre_compound'].str.strip()

    # -------------------------------------------------------------------
    # Drop the first STINT_WARMUP_LAPS laps of every stint.  A stint is a
    # run of consecutive laps on the same compound within a session (and
    # driver).  These opening laps are distorted by cold tyres, race-start
    # traffic and tyre warm-up, and were the main driver of the old model's
    # bogus "laps get faster as tyres age" coefficient.
    # -------------------------------------------------------------------
    df = df.sort_values(['session_id', 'driver_id', 'lap_number'])
    # Race-phase bucket (from the absolute lap number) survives the cleaning:
    # it anchors the fuel/race-phase LEVEL of each stint so that late-race
    # Hard stints (light fuel) are not credited to the Hard intercept.  See
    # feature_pipeline.py for the investigation behind this.
    df['race_phase'] = df['lap_number'].map(race_phase_index)
    df['_stint'] = (
        df['tyre_compound'] != df.groupby(['session_id', 'driver_id'])['tyre_compound'].shift()
    ).groupby([df['session_id'], df['driver_id']]).cumsum()
    df['_lap_in_stint'] = df.groupby(['session_id', 'driver_id', '_stint']).cumcount()
    warmup_mask = df['_lap_in_stint'] < STINT_WARMUP_LAPS
    df = df[~warmup_mask].drop(columns=['_stint', '_lap_in_stint', 'lap_number'])
    if verbose:
        print(f"[INFO] Excluded {int(warmup_mask.sum())} stint warm-up laps "
              f"(first {STINT_WARMUP_LAPS} laps of each stint)")

    # -------------------------------------------------------------------
    # SC / VSC / red-flag lap exclusion (session-relative outlier filter)
    # -------------------------------------------------------------------
    session_medians = df.groupby('session_id')['lap_time'].transform('median')
    outlier_mask = df['lap_time'] <= session_medians * SC_VSC_REDFLAG_RATIO
    dropped_outliers = int((~outlier_mask).sum())
    df = df[outlier_mask]
    if verbose:
        print(f"[INFO] Excluded {dropped_outliers} SC/VSC/red-flag/formation laps "
              f"(> {SC_VSC_REDFLAG_RATIO:.2f}x session median)")
    return df


print("=" * 60)
print("F1 LAP TIME PREDICTION - MODEL TRAINING")
print("=" * 60)

conn = get_db_connection()

print("\n[DATABASE] Loading training data...")
df = pd.read_sql(TRAINING_QUERY, conn)
conn.close()

if df.empty:
    print("[ERROR] No valid laps found in database. Train data is empty!")
    sys.exit(1)

print(f"[INFO] Raw candidate laps: {len(df)}")

# Keep a pre-cleaning copy so each driver's model re-runs the stint warm-up
# and outlier filters on their OWN laps (grouped by session + driver).
raw_df = df.copy()

df = clean_training_data(df)
print(f"[INFO] Training laps after cleaning: {len(df)}")
print(f"[INFO] Tracks covered: {df['track_name'].nunique()} — {sorted(df['track_name'].unique())}")
print(f"[INFO] Tyres:  {df['tyre_compound'].nunique()}")

# ---------------------------------------------------------------------------
# Sector-time targets
#
# The sector models below predict S1/S2/S3 times from the SAME tyre / track /
# tyre-age features as the lap-time model, so an energy-to-time conversion
# can work on measured per-track sector structure instead of a flat global
# constant.  Targets are the lap's real FastF1 sector times where the DB has
# them; laps imported before sector capture existed fall back to the track's
# mean MEASURED sector split x lap time (equal thirds only when a track has
# no measured sectors at all, e.g. simulator sessions).  The share source is
# recorded in model_info / energy_pace.json.
# ---------------------------------------------------------------------------
per_track_sector_shares = mean_sector_shares(df)
real_sector_laps = int(df[['sector1_ms', 'sector2_ms', 'sector3_ms']]
                       .notna().all(axis=1).sum())
proxy_laps = int((len(df) - real_sector_laps))
sector_share_source = ('measured_fastf1' if real_sector_laps > 0
                       else 'proxy_thirds')

for k in range(3):
    col = f'sector{k + 1}_ms'
    m = df[col].isna()
    if m.any():
        shares = df.loc[m, 'track_name'].map(
            lambda t: per_track_sector_shares.get(t) or [1.0 / 3.0] * 3
        )
        fill_s = (pd.Series([sh[k] for sh in shares], index=df.index[m])
                  * df.loc[m, 'lap_time'] * 1000.0)
        df.loc[m, col] = fill_s
    df[f'{SECTOR_NAMES[k]}_s'] = df[col] / 1000.0
df = df.drop(columns=['sector1_ms', 'sector2_ms', 'sector3_ms'])

print(f"[INFO] Sector targets: {real_sector_laps} laps with real FastF1 sectors, "
      f"{proxy_laps} proxied from per-track measured splits")

# ---------------------------------------------------------------------------
# Feature engineering
#
# Feature set (details + the investigation behind it in feature_pipeline.py):
#   * tyre_age           — numeric, laps since the set was fitted.
#   * phase_<0..3>       — race-phase bucket derived from lap_number.  The
#                          fuel/race-phase LEVEL of each stint used to leak
#                          into the compound intercepts: Hard is raced almost
#                          exclusively as a light-fuel late-race second stint
#                          (mean race fraction ~0.64 in the DB) while Soft
#                          spans heavy-fuel race starts (~0.40) — which is
#                          why the old model "learned" Hard is faster than
#                          Soft.  A coarse bucket is ~constant within a
#                          stint, so it is NOT collinear with tyre_age (a
#                          continuous fuel term like 110 - 2*lap WAS: it is
#                          a deterministic function of lap_number and hence
#                          perfectly collinear with tyre_age inside a stint).
#   * tyre_age:tyre_<c>  — tyre_age x compound interactions so each compound
#                          gets its own wear slope instead of one global one.
#   * era_<bucket>         — season-era one-hot (from the session's year).
#                          Cross-season pace drift used to leak into the
#                          compound intercepts (Hard skews to recent seasons);
#                          the era term absorbs it.
#   * tyre_age:track_<t>   — tyre_age x track interactions (global model only)
#                          so measured per-track wear can be exported.
# ---------------------------------------------------------------------------
df_encoded = encode_features(df)

y = df_encoded['lap_time']
groups = df_encoded['session_id']
X = df_encoded.drop(columns=['lap_time', 'session_id', 'driver_id', 'driver_code',
                              'driver_name', 'lap_id', 's1_s', 's2_s', 's3_s'])
feature_names = list(X.columns)
# Sector targets share the lap model's feature vector and (random-state 42)
# the same 80/20 split, so their metrics are directly comparable.
sector_ys = [df_encoded[f'{name}_s'] for name in SECTOR_NAMES]
print(f"[INFO] Feature Count: {len(feature_names)}")

# ---------------------------------------------------------------------------
# Evaluation
#
# Two metrics are reported:
#   * Within-track accuracy (random 80/20 split) — this is the deployment
#     scenario: the API/CLI only predict on tracks the model has seen
#     (unseen tracks are rejected with a clear error), so the useful
#     question is "how well does the model predict remaining laps of a
#     known track?".
#   * Unseen-track generalization (GroupShuffleSplit on session_id) — held
#     out sessions are whole tracks the model never saw, which a one-hot
#     track model structurally cannot predict.  Its R2 is expected to be
#     poor; it is reported for transparency and is NOT the deployment
#     metric.
# Model selection uses the within-track MAE.
# ---------------------------------------------------------------------------
def train_and_eval(X_tr, y_tr, X_te, y_te):
    results = {}
    lr = LinearRegression()
    lr.fit(X_tr, y_tr)
    yp = lr.predict(X_te)
    results['LinearRegression'] = (lr, {
        'mae': mean_absolute_error(y_te, yp),
        'rmse': np.sqrt(mean_squared_error(y_te, yp)),
        'r2': r2_score(y_te, yp),
        'predictions': yp,
        'y_test': y_te,
    })

    rf = RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_split=4, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    yp = rf.predict(X_te)
    results['RandomForest'] = (rf, {
        'mae': mean_absolute_error(y_te, yp),
        'rmse': np.sqrt(mean_squared_error(y_te, yp)),
        'r2': r2_score(y_te, yp),
        'predictions': yp,
        'y_test': y_te,
    })
    return results

print("=" * 60)
print("EVALUATION 1/2 — WITHIN-TRACK (random 80/20 split)")
print("=" * 60)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"  Train: {len(X_train)} | Test: {len(X_test)}")
within = train_and_eval(X_train, y_train, X_test, y_test)
for name, (_, m) in within.items():
    print(f"  {name:<16} MAE {m['mae']:.3f}s  RMSE {m['rmse']:.3f}s  R2 {m['r2']:.3f}")

lr_within = within['LinearRegression'][0]
lr_coefs = dict(zip(feature_names, lr_within.coef_))
print(f"\n  tyre_age coefficient (within-track): "
      f"{lr_coefs.get('tyre_age', 0.0):+.4f} s/lap  "
      f"[positive = tyres slow you down as they age]")

# Effective per-lap age slope per compound = global tyre_age slope + that
# compound's tyre_age:tyre_<c> interaction.  A value near or above zero for
# a SOFT compound is the physically expected signature of wear once the
# race-phase level is carried by the phase buckets (fuel burn still pushes
# the global slope negative — within a single stint a linear fuel trend and
# a linear wear trend are indistinguishable).
model_compounds = sorted(f.replace('tyre_', '') for f in feature_names
                         if f.startswith('tyre_')
                         and f not in ('tyre_age', 'tyre_load') and ':' not in f)
compound_wear = {}
for _c in model_compounds:
    compound_wear[_c] = (lr_coefs.get('tyre_age', 0.0)
                         + lr_coefs.get(f'tyre_age:tyre_{_c}', 0.0))
print("  per-compound tyre_age slope (s/lap): " +
      ", ".join(f"{k}={v:+.4f}" for k, v in sorted(compound_wear.items())))

# Fuel-burn rate: the pace effect of fuel burn (and other non-wear age
# effects) that remains baked into the fitted age slopes — within a single
# stint a linear fuel trend and a linear wear trend cannot be separated.
# The strategy advisor detrends predictions by this rate so that stay-out vs
# pit comparisons are made on equal fuel footing: fuel is burned identically
# by both scenarios over the same remaining laps, so crediting the stay-out
# scenario's higher ages with it would bias the comparison by
# |rate| * laps_rem * cur_age in its favour.  A negative slope is treated as
# fuel and removed; a positive slope (wear dominating) would be clamped to
# zero and left in the comparison.
fuel_burn_rate = min(0.0, lr_coefs.get('tyre_age', 0.0))

print("=" * 60)
print("EVALUATION 2/2 — UNSEEN-TRACK (GroupShuffleSplit by session)")
print("=" * 60)
print("  Held-out sessions are whole tracks the model has never seen.")
print("  One-hot track models cannot predict them; the API rejects such")
print("  requests by design.  Reported for transparency only.")
if groups.nunique() > 1:
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(gss.split(X, y, groups))
    unseen = train_and_eval(X.iloc[tr_idx], y.iloc[tr_idx], X.iloc[te_idx], y.iloc[te_idx])
    for name, (_, m) in unseen.items():
        print(f"  {name:<16} MAE {m['mae']:.3f}s  RMSE {m['rmse']:.3f}s  R2 {m['r2']:.3f}")
else:
    unseen = None
    print("  Only one session — skipping.")

# Select best model on the within-track (deployment) metric
best_name  = min(within, key=lambda k: within[k][1]['mae'])
best_model = within[best_name][0]
best_metrics = within[best_name][1]

print("\n" + "=" * 60)
print(f"[BEST MODEL] {best_name} (selected on within-track MAE)")
print(f"   Within-track MAE:  {best_metrics['mae']:.3f}s")
print(f"   Within-track RMSE: {best_metrics['rmse']:.3f}s")
print(f"   Within-track R2:   {best_metrics['r2']:.3f}")
print("=" * 60)

# Save artifacts
MODEL_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(best_model,   MODEL_DIR / 'best_model.pkl')
joblib.dump(feature_names, MODEL_DIR / 'feature_names.pkl')
print(f"\n[INFO] Saved: {MODEL_DIR / 'best_model.pkl'}")
print(f"[INFO] Saved: {MODEL_DIR / 'feature_names.pkl'}")

# Track / tyre coverage (used by the predictors to reject unseen inputs)
covered_tracks = sorted(f.replace('track_', '') for f in feature_names if f.startswith('track_'))
covered_tyres  = sorted(f.replace('tyre_', '') for f in feature_names
                        if f.startswith('tyre_') and f not in ('tyre_age', 'tyre_load')
                        and ':' not in f)
per_track_laps = df.groupby('track_name').size().sort_values(ascending=False)

# ---------------------------------------------------------------------------
# Measured per-track / per-compound wear export
#
# The tyre_age:track_<t> interactions give each circuit its own age slope.
# Effective wear for (track t, compound c) =
#     global_tyre_age + tyre_age:track_t + tyre_age:tyre_c
# minus the fuel-burn rate (the age effect that is fuel, not wear — removed
# so the advisor can layer the result on top of fuel-neutral predictions).
#
# The raw per-track numbers are NOISY (a few seasons of two drivers): an
# unregularised table would say Bahrain wears tyres *less* than Silverstone.
# So the export records the lap counts per (track, compound) cell and a
# physically ordered slope (Soft >= Medium >= Hard); the strategy advisor
# blends measured values with its researched curves, weighting measurement
# only where the cell has data.
# ---------------------------------------------------------------------------
dry_order = ['Soft', 'Medium', 'Hard']
measured_wear = {}
for _t in covered_tracks:
    t_adj = lr_coefs.get(f'tyre_age:track_{_t}', 0.0)
    cell = {}
    for _c in covered_tyres:
        c_adj = lr_coefs.get(f'tyre_age:tyre_{_c}', 0.0)
        raw = lr_coefs.get('tyre_age', 0.0) + t_adj + c_adj - fuel_burn_rate
        n = int(((df['track_name'] == _t) & (df['tyre_compound'] == _c)).sum())
        cell[_c] = {'slope': round(float(raw), 4), 'laps': n}
    dry_vals = {c: cell[c]['slope'] for c in dry_order if c in cell}
    names = [c for c in dry_order if c in cell]
    # Project onto non-increasing (Soft >= Medium >= Hard): reversed cummax,
    # then un-reverse — each earlier compound's slope becomes at least the
    # maximum of all later ones, guaranteeing the physical ordering.
    vals = np.maximum.accumulate([dry_vals[c] for c in names][::-1])[::-1]
    for c, v in zip(names, vals):
        cell[c]['slope'] = round(float(v), 4)
    measured_wear[_t] = cell

print("  measured per-track wear (s/lap, detrended, ordered): "
      + ", ".join(
          f"{t}={measured_wear[t].get('Soft', {}).get('slope', 0.0):+.3f}"
          for t in covered_tracks[:6]) + " ...")

def fmt_metrics(m):
    return (f"MAE: {m['mae']:.3f}s\n"
            f"RMSE: {m['rmse']:.3f}s\n"
            f"R2: {m['r2']:.3f}")

# Text metadata
with open(MODEL_DIR / 'model_info.txt', 'w', encoding='utf-8') as f:
    f.write("F1 LAP TIME PREDICTION MODEL\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Best Model:       {best_name}\n")
    f.write(f"Within-track:     {fmt_metrics(best_metrics)}\n")
    if unseen:
        f.write(f"Unseen-track:     {fmt_metrics(unseen[best_name][1])} "
                f"(informational — unseen tracks are rejected)\n")
    f.write(f"Training samples: {len(X_train)}\n")
    f.write(f"Test samples:     {len(X_test)}\n")
    f.write(f"Features:         {len(feature_names)}\n")
    f.write(f"Trained on:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Tracks covered:   {len(covered_tracks)}\n")
    f.write(f"Tyres covered:    {len(covered_tyres)}\n")
    f.write(f"Fuel burn rate:   {fuel_burn_rate:+.4f} s/lap "
            f"(detrended out of advisor comparisons)\n\n")
    f.write("Measured per-track wear (s/lap, detrended, Soft/Medium/Hard ordered):\n")
    for _t in covered_tracks:
        cell = measured_wear.get(_t, {})
        parts = [f"{_c}={cell.get(_c, {}).get('slope', 0.0):+.4f}"
                 f"(n={cell.get(_c, {}).get('laps', 0)})" for _c in covered_tyres]
        f.write(f"  {_t:<45} {' '.join(parts)}\n")
    f.write("\nLaps per track:\n")
    for track, n in per_track_laps.items():
        f.write(f"  {track:<50} {n}\n")
    f.write("\nFeatures used:\n")
    for fn in feature_names:
        f.write(f"  {fn}\n")

print(f"[INFO] Saved: {MODEL_DIR / 'model_info.txt'}")

# JSON metadata
metadata = {
    "best_model": best_name,
    "metrics": {
        "within_track": {k: round(v, 4) for k, v in best_metrics.items() if k not in ('predictions', 'y_test')},
    },
    "unseen_track": (
        {k: round(v, 4) for k, v in unseen[best_name][1].items() if k not in ('predictions', 'y_test')}
        if unseen else None
    ),
    "training_samples": len(X_train),
    "test_samples": len(X_test),
    "features": feature_names,
    "coverage": {
        "tracks": covered_tracks,
        "tyres": covered_tyres,
        "laps_per_track": {str(k): int(v) for k, v in per_track_laps.items()},
    },
    "tyre_age_coefficient": round(lr_coefs.get('tyre_age', 0.0), 4),
    "tyre_wear_slopes_per_compound": {k: round(v, 4) for k, v in compound_wear.items()},
    "tyre_wear_per_track_compound": {
        str(t): {c: {"slope": cell[c]["slope"], "laps": cell[c]["laps"]}
                 for c in cell}
        for t, cell in measured_wear.items()
    },
    "fuel_burn_rate": round(fuel_burn_rate, 4),
    "trained_at": datetime.now().isoformat()
}
with open(MODEL_DIR / 'model_info.json', 'w', encoding='utf-8') as f_json:
    json.dump(metadata, f_json, indent=2, ensure_ascii=False)
print(f"[INFO] Saved: {MODEL_DIR / 'model_info.json'}")

# ===========================================================================
# SECTOR MODELS (S1 / S2 / S3 times)
#
# One model per sector predicts that sector's time (seconds) from the same
# tyre / track / age features as the lap-time model.  Together with the
# throttle profile below these give the energy-to-time conversion a measured
# per-track structure: the S1:S2:S3 split of a predicted lap, and how much of
# each sector is spent at full throttle (where the MGU-K's extra power
# actually reaches the road).
# ===========================================================================
print("\n" + "=" * 60)
print("SECTOR MODELS (S1 / S2 / S3 times)")
print("=" * 60)
sector_models = {}
sector_metrics = {}
sector_details = []
for k in range(3):
    yk = sector_ys[k]
    X_tr, X_te, y_tr, y_te = train_test_split(X, yk, test_size=0.2, random_state=42)
    ev = train_and_eval(X_tr, y_tr, X_te, y_te)
    best_k = min(ev, key=lambda kk: ev[kk][1]['mae'])
    joblib.dump(ev[best_k][0], MODEL_DIR / f'sector_model_{k + 1}.pkl')
    sector_models[str(k + 1)] = best_k
    sector_metrics[str(k + 1)] = {kk: round(v, 4) for kk, v in ev[best_k][1].items()
                                  if kk not in ('predictions', 'y_test')}
    sector_details.append((k + 1, best_k, ev[best_k][1]['mae'],
                           ev[best_k][1]['rmse'], ev[best_k][1]['r2']))
    print(f"  S{k + 1}: model={best_k:<16} MAE {ev[best_k][1]['mae']:.3f}s  "
          f"RMSE {ev[best_k][1]['rmse']:.3f}s  R2 {ev[best_k][1]['r2']:.3f}")
print(f"[INFO] Saved: ml_models/sector_model_1..3.pkl "
      f"(share the lap model's feature_names.pkl)")

# ---------------------------------------------------------------------------
# Measured per-track energy profile (energy_pace.json)
#
# The dashboard's energy what-if converts deployed MJ into seconds with a
# flat global constant.  The per-track numbers below replace it with
# measured values from this training set:
#   * sector_time_share   — measured mean S1:S2:S3 split of lap time (FastF1
#     sector times where available, equal thirds before a FastF1 backfill)
#   * full_throttle_share — measured fraction of stored telemetry samples at
#     >= 90% throttle, per track and per sector-third of the sample stream.
#     Deployment's time value scales with it: the MGU-K can only add speed
#     where the engine is already at full throttle, otherwise the driver
#     simply eases off the pedal and the extra power is wasted.
#   * pace_s_per_mj       — the 0.35 s/MJ anchor (calibrated 2026-PU
#     order-of-magnitude) scaled by the track's measured full-throttle share
#     relative to the fleet mean, so power tracks (Monza, Red Bull Ring)
#     rate higher and stop-go tracks (Monaco, Mexico) lower, matching where
#     real-world ERS deployment pays.
# ---------------------------------------------------------------------------
print("\n[DATA] Measuring per-track energy profile from stored telemetry...")
conn = get_db_connection()
clean_ids = df['lap_id'].tolist()
lap_data = {}   # lap_id -> [track_name, [throttle, ...]]
cur = conn.cursor(dictionary=True)
for i in range(0, len(clean_ids), 500):
    ids = clean_ids[i:i + 500]
    ph = ','.join(['%s'] * len(ids))
    cur.execute(f"""
        SELECT s.track_name, t.lap_id, t.throttle
        FROM telemetry t
        JOIN laps l ON l.lap_id = t.lap_id
        JOIN sessions s ON s.session_id = l.session_id
        WHERE t.lap_id IN ({ph})
        ORDER BY t.lap_id, t.telemetry_id
    """, ids)
    for row in cur.fetchall():
        # Normalise the track key exactly like clean_training_data does
        # (strip + title), or multi-word names such as 'Circuit de Monaco'
        # would never match the per_track_laps keys below.
        trk_norm = str(row['track_name']).strip().title()
        rec = lap_data.setdefault(row['lap_id'], [trk_norm, []])
        rec[1].append(float(row['throttle'] or 0.0))
cur.close()
conn.close()

track_prof = {}
for _lid, (trk, thrs) in lap_data.items():
    n = len(thrs)
    if n < 3:
        continue
    p = track_prof.setdefault(trk, {'samples': [0, 0, 0], 'ft': [0, 0, 0], 'laps': 0})
    p['laps'] += 1
    for j, t in enumerate(thrs):
        k = min(2, j * 3 // n)
        p['samples'][k] += 1
        p['ft'][k] += 1 if t >= 0.9 else 0

tot_s = sum(sum(p['samples']) for p in track_prof.values())
tot_f = sum(sum(p['ft']) for p in track_prof.values())
fleet_ft = (tot_f / tot_s) if tot_s > 0 else 0.65

# Lazy import: keeps this training script runnable even if the simulator's
# heavy deps are missing.
from energy_simulator import DEPLOY_PACE_S_PER_MJ as PACE_ANCHOR
PACE_MIN, PACE_MAX = 0.12, 0.75

per_track_pace = {}
for trk in per_track_laps.index:
    prof = track_prof.get(trk)
    if prof is None or sum(prof['samples']) == 0:
        ft_overall = None
        ft_sec = [None, None, None]
    else:
        ft_overall = sum(prof['ft']) / float(sum(prof['samples']))
        ft_sec = [prof['ft'][k] / float(prof['samples'][k])
                  if prof['samples'][k] else None for k in range(3)]
    pace = (PACE_ANCHOR * ft_overall / fleet_ft) if ft_overall else PACE_ANCHOR
    pace = max(PACE_MIN, min(PACE_MAX, pace))
    sec_pace = [pace * (ft_sec[k] / ft_overall)
                if ft_overall and ft_sec[k] is not None else pace
                for k in range(3)]
    sec_pace = [max(PACE_MIN * 0.9, min(PACE_MAX, v)) for v in sec_pace]
    shares = per_track_sector_shares.get(trk)
    per_track_pace[str(trk)] = {
        "sector_time_share": ([round(x, 4) for x in shares] if shares
                              else [round(1.0 / 3.0, 4)] * 3),
        "sector_share_source": ('measured_fastf1' if shares else 'proxy_thirds'),
        "full_throttle_share": (round(ft_overall, 4) if ft_overall else None),
        "sector_full_throttle_share": [round(x, 4) if x is not None else None
                                       for x in ft_sec],
        "pace_s_per_mj": round(pace, 4),
        "sector_pace_s_per_mj": [round(x, 4) for x in sec_pace],
        "laps": prof['laps'] if prof else 0,
    }

pace_artifact = {
    "trained_at": datetime.now().isoformat(),
    "anchor_pace_s_per_mj": PACE_ANCHOR,
    "basis": "pace scaled by measured full-throttle share of stored telemetry "
             "(relative to fleet mean); sector split measured from FastF1 "
             "sector times when available",
    "sector_share_source": sector_share_source,
    "real_sector_laps": int(real_sector_laps),
    "fleet_full_throttle_share": round(fleet_ft, 4),
    "per_track": per_track_pace,
}
with open(MODEL_DIR / 'energy_pace.json', 'w', encoding='utf-8') as f:
    json.dump(pace_artifact, f, indent=2, ensure_ascii=False)
print(f"[INFO] Saved: {MODEL_DIR / 'energy_pace.json'} "
      f"({len(per_track_pace)} tracks, fleet FT share {fleet_ft:.3f})")

# Extend model_info.txt / model_info.json with the sector models
with open(MODEL_DIR / 'model_info.txt', 'a', encoding='utf-8') as f:
    f.write("\nSector models (same features; within-track):\n")
    for (k, model_name, mae, rmse, r2) in sector_details:
        f.write(f"  S{k}: {model_name}  MAE {mae:.3f}s  RMSE {rmse:.3f}s  R2 {r2:.3f}\n")
    f.write(f"Sector share source: {sector_share_source} "
            f"({real_sector_laps} real laps)\n")
    f.write(f"Fleet full-throttle share: {fleet_ft:.4f} "
            f"(energy pace anchor: {PACE_ANCHOR} s/MJ)\n")

with open(MODEL_DIR / 'model_info.json', encoding='utf-8') as fh:
    md = json.load(fh)
md['sector_models'] = {k: {"model": sector_models[k],
                           "within_track": sector_metrics[k]}
                       for k in sector_models}
md['sector_share_source'] = sector_share_source
md['real_sector_laps'] = int(real_sector_laps)
md['fleet_full_throttle_share'] = round(fleet_ft, 4)
with open(MODEL_DIR / 'model_info.json', 'w', encoding='utf-8') as fh:
    json.dump(md, fh, indent=2, ensure_ascii=False)

# ---------------------------------------------------------------------------
# Per-driver models
#
# One model per driver, trained on the driver's OWN race laps with the same
# cleaning + feature pipeline as the global model.  Each driver model covers
# only the tracks/tyres that driver has raced, which is exactly what makes
# a head-to-head comparison meaningful: on a shared (track, tyre) the
# difference between two drivers' predicted times is their pace gap.
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("PER-DRIVER MODELS")
print("=" * 60)
print(f"  Training one model per driver with >= {MIN_DRIVER_LAPS} clean laps.")
print("  Artifacts: ml_models\\drivers\\<driver_code>\\")

drivers_dir = MODEL_DIR / 'drivers'
drivers_dir.mkdir(parents=True, exist_ok=True)

driver_summary = []
# NULL driver ids (telemetry-only laps) are dropped by groupby's dropna.
for (driver_code, driver_name), grp in raw_df.groupby(['driver_code', 'driver_name']):
    grp_clean = clean_training_data(grp, verbose=False)
    if len(grp_clean) < MIN_DRIVER_LAPS:
        print(f"  [SKIP ] {str(driver_code):<4} {str(driver_name):<24} "
              f"only {len(grp_clean)} clean laps (< {MIN_DRIVER_LAPS})")
        continue

    df_enc = encode_features(grp_clean, track_terms=False)
    y_drv = df_enc['lap_time']
    drop_cols = [c for c in ('lap_time', 'session_id', 'driver_id', 'driver_code',
                             'driver_name', 'session_date', 'lap_id',
                             'sector1_ms', 'sector2_ms', 'sector3_ms',
                             's1_s', 's2_s', 's3_s')
                 if c in df_enc.columns]
    X_drv = df_enc.drop(columns=drop_cols)
    drv_features = list(X_drv.columns)

    X_tr, X_te, y_tr, y_te = train_test_split(X_drv, y_drv, test_size=0.2, random_state=42)
    within_drv = train_and_eval(X_tr, y_tr, X_te, y_te)
    drv_best = min(within_drv, key=lambda k: within_drv[k][1]['mae'])
    drv_model = within_drv[drv_best][0]
    drv_metrics = within_drv[drv_best][1]

    drv_lr = within_drv['LinearRegression'][0]
    drv_coefs = dict(zip(drv_features, drv_lr.coef_))
    drv_fuel_burn = min(0.0, drv_coefs.get('tyre_age', 0.0))

    drv_tracks = sorted(f.replace('track_', '') for f in drv_features if f.startswith('track_'))
    drv_tyres = sorted(f.replace('tyre_', '') for f in drv_features
                       if f.startswith('tyre_') and f not in ('tyre_age', 'tyre_load')
                       and ':' not in f)

    ddir = drivers_dir / str(driver_code)
    ddir.mkdir(parents=True, exist_ok=True)
    joblib.dump(drv_model, ddir / 'best_model.pkl')
    joblib.dump(drv_features, ddir / 'feature_names.pkl')

    drv_info = {
        "driver": {"code": str(driver_code), "name": str(driver_name)},
        "best_model": drv_best,
        "metrics": {"within_track": {k: round(v, 4) for k, v in drv_metrics.items()
                                     if k not in ('predictions', 'y_test')}},
        "training_samples": int(len(X_tr)),
        "test_samples": int(len(X_te)),
        "clean_laps": int(len(grp_clean)),
        "features": drv_features,
        "coverage": {"tracks": drv_tracks, "tyres": drv_tyres},
        "tyre_age_coefficient": round(drv_coefs.get('tyre_age', 0.0), 4),
        "fuel_burn_rate": round(drv_fuel_burn, 4),
        "trained_at": datetime.now().isoformat()
    }
    with open(ddir / 'model_info.json', 'w', encoding='utf-8') as f:
        json.dump(drv_info, f, indent=2, ensure_ascii=False)

    driver_summary.append((str(driver_code), str(driver_name), len(grp_clean),
                           len(drv_tracks), drv_best, drv_metrics['mae']))
    print(f"  [TRAIN] {str(driver_code):<4} {str(driver_name):<24} "
          f"laps={len(grp_clean):>5} tracks={len(drv_tracks):>2} "
          f"model={drv_best:<16} MAE={drv_metrics['mae']:.3f}s")

# -----------------------------------------------------------------------
# Per-driver-per-year models
#
# One model per (driver, season) pair, trained on that driver's own race
# laps from that specific year.  Same pipeline as the global model, but
# the intercept now absorbs the driver + car + regulations of that exact
# season, so cross-year comparisons mix car changes with driver pace.
# These models are stored in:
#   ml_models/drivers/<code>/<year>/
# and are the primary comparison unit in the dashboard (same-year =
# apples-to-apples).  The per-driver aggregate model in
# ml_models/drivers/<code>/ is kept as a fallback for drivers with only
# one year of data, and for multi-year 'career shape' queries.
# -----------------------------------------------------------------------
MIN_YEAR_LAPS = 60

print("\n" + "=" * 60)
print("PER-DRIVER-PER-YEAR MODELS")
print("=" * 60)
print(f"  Training one model per (driver, year) with >= {MIN_YEAR_LAPS} clean laps.")
print("  Artifacts: ml_models\\drivers\\<driver_code>\\<year>\\")

year_summary = []
for (driver_code, driver_name), grp in raw_df.groupby(['driver_code', 'driver_name']):
    if pd.isna(driver_code):
        continue
    # session_date is in the query result; extract year for grouping
    grp = grp.copy()
    grp['_year'] = pd.to_datetime(grp['session_date'], errors='coerce').dt.year
    for year, year_grp in grp.groupby('_year'):
        if pd.isna(year):
            continue
        year_int = int(year)
        year_clean = clean_training_data(year_grp, verbose=False)
        if len(year_clean) < MIN_YEAR_LAPS:
            continue

        df_enc = encode_features(year_clean, track_terms=False)
        y_drv = df_enc['lap_time']
        drop_cols = [c for c in ('lap_time', 'session_id', 'driver_id', 'driver_code',
                                 'driver_name', 'session_date', '_year', 'lap_id',
                                 'sector1_ms', 'sector2_ms', 'sector3_ms',
                                 's1_s', 's2_s', 's3_s')
                     if c in df_enc.columns]
        X_drv = df_enc.drop(columns=drop_cols)
        drv_features = list(X_drv.columns)

        X_tr, X_te, y_tr, y_te = train_test_split(X_drv, y_drv, test_size=0.2, random_state=42)
        within_drv = train_and_eval(X_tr, y_tr, X_te, y_te)
        drv_best = min(within_drv, key=lambda k: within_drv[k][1]['mae'])
        drv_model = within_drv[drv_best][0]
        drv_metrics = within_drv[drv_best][1]

        drv_lr = within_drv['LinearRegression'][0]
        drv_coefs = dict(zip(drv_features, drv_lr.coef_))
        drv_fuel_burn = min(0.0, drv_coefs.get('tyre_age', 0.0))

        drv_tracks = sorted(f.replace('track_', '') for f in drv_features if f.startswith('track_'))
        drv_tyres = sorted(f.replace('tyre_', '') for f in drv_features
                           if f.startswith('tyre_') and f not in ('tyre_age', 'tyre_load')
                           and ':' not in f)

        ydir = drivers_dir / str(driver_code) / str(year_int)
        ydir.mkdir(parents=True, exist_ok=True)
        joblib.dump(drv_model, ydir / 'best_model.pkl')
        joblib.dump(drv_features, ydir / 'feature_names.pkl')

        yr_info = {
            "driver": {"code": str(driver_code), "name": str(driver_name)},
            "year": year_int,
            "best_model": drv_best,
            "metrics": {"within_track": {k: round(v, 4) for k, v in drv_metrics.items()
                                         if k not in ('predictions', 'y_test')}},
            "training_samples": int(len(X_tr)),
            "test_samples": int(len(X_te)),
            "clean_laps": int(len(year_clean)),
            "features": drv_features,
            "coverage": {"tracks": drv_tracks, "tyres": drv_tyres},
            "tyre_age_coefficient": round(drv_coefs.get('tyre_age', 0.0), 4),
            "fuel_burn_rate": round(drv_fuel_burn, 4),
            "trained_at": datetime.now().isoformat()
        }
        with open(ydir / 'model_info.json', 'w', encoding='utf-8') as f:
            json.dump(yr_info, f, indent=2, ensure_ascii=False)

        year_summary.append((str(driver_code), str(driver_name), year_int, len(year_clean),
                               len(drv_tracks), drv_best, drv_metrics['mae']))
        print(f"  [TRAIN] {str(driver_code):<4} {year_int} {str(driver_name):<20} "
              f"laps={len(year_clean):>5} tracks={len(drv_tracks):>2} "
              f"model={drv_best:<16} MAE={drv_metrics['mae']:.3f}s")

if not year_summary:
    print("  No (driver, year) pair met the minimum-lap threshold.")
else:
    print(f"\n  Trained {len(year_summary)} per-driver-per-year models.")

if not driver_summary:
    print("  No driver met the minimum-lap threshold — only the global model was saved.")
else:
    print(f"\n  Trained {len(driver_summary)} per-driver models in {drivers_dir}.")
    print("  Compare two drivers head-to-head with: python scripts/driver_comparison.py")

# Visualizations
if best_name == 'RandomForest':
    importances = best_model.feature_importances_
    feat_imp = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feat_imp = feat_imp.sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 6))
    top15 = feat_imp.head(15)
    plt.barh(range(len(top15)), top15['importance'])
    plt.yticks(range(len(top15)), top15['feature'])
    plt.xlabel('Importance')
    plt.title('Top 15 Feature Importances — Random Forest')
    plt.tight_layout()
    plt.savefig(MODEL_DIR / 'feature_importance.png', dpi=300)
    plt.close()
    print(f"[INFO] Saved: {MODEL_DIR / 'feature_importance.png'}")

plt.figure(figsize=(10, 6))
plt.scatter(y_test, best_metrics['predictions'], alpha=0.5, label='Predictions')
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--', lw=2, label='Perfect Prediction'
)
plt.xlabel('Actual Lap Time (seconds)')
plt.ylabel('Predicted Lap Time (seconds)')
plt.title(f'Actual vs Predicted Lap Times ({best_name}, within-track)')
plt.legend()
plt.tight_layout()
plt.savefig(MODEL_DIR / 'predictions_vs_actual.png', dpi=300)
plt.close()
print(f"[INFO] Saved: {MODEL_DIR / 'predictions_vs_actual.png'}")
