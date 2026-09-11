"""Dual-agent overtake model — training (P0).

Trains TWO heads from PAIRED leader/chaser race laps, the differentiator the
deck leads with:

  * closing_model   — predicts the gap-delta closing rate (s gained by the
                      chaser during the lap) from BOTH drivers' features.
  * overtake_model  — predicts the per-lap overtake probability (binary:
                      did the chaser get past the leader this lap?).

Pair construction
-----------------
The laps table has no on-track position, so the head-to-head is rebuilt from
each driver's cumulative race clock: for a race with two (or more) sessions,
each lap number where both drivers have a lap gives a sample, with the
driver ahead on cumulative time cast as the LEADER and the one behind as the
CHASER.  Gap before = |cum(L-1) diff|, gap after = |cum(L) diff|, closing
rate = before - after.  A position swap across the lap (leader at L-1 is no
longer the leader at L) labels an overtake.

Samples only count as battles (gap before <= GAP_WINDOW_S), and overtake
labels are only trusted from inside a short window (OVERTAKE_GAP_MAX_S) —
a "pass" from 8 s back in one lap is a pit/incident artifact, not racing.
Pit in/out laps, the first two laps of each stint (cold tyres / race start),
and SC/VSC/red-flag laps (session-relative outlier filter, same rule as
ml_lap_predictions.py) are excluded, as are race-start lap 1 samples.

Features — both agents, reused machinery
-----------------------------------------
  * pace_gap_s            — leader predicted - chaser predicted lap time from
                            the per-driver(-per-year) lap-time models
                            (driver_comparison.load_driver_model), the exact
                            machinery driver_comparison.py uses for its
                            "pace gap only" comparison.  This is what lets
                            the overtake model decompose a battle into
                            "expected pace advantage" vs "everything else".
  * gap_before_s          — the gap entering the lap (context).
  * chaser_age_minus_leader — tyre freshness advantage.
  * chaser_tyre_advantage — compound-speed advantage (softer = faster).
  * fuel_diff_kg          — fuel-load advantage at lap start.
  * energy_diff_mj        — ERS state-of-charge advantage (from race_state,
                            imputed 0 when the synthetic ERS sim has not run
                            for that race; coverage is recorded).
  * phase_<0..3> / era_<b> / track_<t> — same buckets as the lap model.

Training mirrors ml_lap_predictions.py: LinearRegression vs RandomForest
for the closing rate (selection on within-race MAE), LogisticRegression vs
RandomForest for the overtake probability (selection on within-race ROC-AUC,
with class_weight='balanced' because overtakes are ~4% of battle laps).

The overtake head is trained on a TIME-ORDERED split: seasons before the
cutoff train the model, later seasons are held out entirely for model
selection, isotonic calibration and testing.  A random split lets
future-regime races leak into the fit, so the reliability table would
describe interpolation instead of the live use case (predicting forward).
It falls back to a random 20% split — recorded in the artifacts — only when
the held-out years cannot support a classifier/calibrator.  A group-held-out
race split is still reported for transparency.

Artifacts (ml_models/overtake/): closing_model.pkl, overtake_model.pkl,
feature_names.pkl, model_info.json, model_info.txt, training_pairs.csv.
"""

import sys
import json
import joblib
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import get_db_connection
from feature_pipeline import race_phase_index, era_bucket
from overtake_inference import (
    OVERTAKE_MODEL_DIR,
    compute_pace_gap,
    tyre_advantage,
    covered_tracks,
    _canonical_track_name,
)

from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    roc_auc_score, accuracy_score, precision_score, recall_score,
    brier_score_loss,
)

# ---------------------------------------------------------------------------
# Pair-construction constants
# ---------------------------------------------------------------------------
# A battle lap is one where the chaser starts within this gap of the leader.
GAP_WINDOW_S = 10.0
# An overtake label is only trusted when the swap happens from inside this
# gap; swaps from further back are pit/incident artifacts and dropped.
OVERTAKE_GAP_MAX_S = 5.0
# Same SC/VSC/red-flag outlier rule as the lap-time trainer.
SC_VSC_REDFLAG_RATIO = 1.30
# First N laps of each stint dropped (cold tyres / traffic / race start).
STINT_WARMUP_LAPS = 2
MIN_PAIR_LAPS = 40  # need at least this many battle laps before training

# Time-ordered split for the overtake head: every season BEFORE this one
# trains the model; this season and later are held out entirely (selection,
# isotonic calibration, testing).  Training on the future is leakage — a
# random split lets later-regime races into the fit and flatters every
# reported metric.  Only a fallback when the held-out block lacks positives.
TEMPORAL_SPLIT_CUTOFF = 2024

LAPS_QUERY = """
SELECT
    l.lap_id, l.driver_id, d.driver_code, l.session_id, l.lap_number,
    l.lap_time_ms / 1000.0 AS lap_time, l.is_valid,
    l.tyre_compound, l.tyre_age, l.fuel_load,
    s.track_name, s.date, s.weather,
    r.energy_end_mj
FROM laps l
JOIN sessions s ON l.session_id = s.session_id
LEFT JOIN drivers d ON l.driver_id = d.driver_id
LEFT JOIN race_state r
       ON r.session_id = l.session_id AND r.driver_id = l.driver_id
      AND r.lap_number = l.lap_number
WHERE s.session_type = 'Race'
"""

PIT_QUERY = """
SELECT l.session_id, l.driver_id, l.lap_number
FROM strategy_events se
JOIN laps l ON l.lap_id = se.lap_id
WHERE se.event_type = 'PitStop'
  AND (se.duration_sec IS NULL OR se.duration_sec >= 15.0)
"""


def load_race_laps(conn):
    """Raw race laps + per-(session, driver, lap) pit in/out flags."""
    laps = pd.read_sql(LAPS_QUERY, conn)
    if laps.empty:
        return laps, set()
    laps = laps.sort_values(["session_id", "lap_number"]).reset_index(drop=True)

    cur = conn.cursor()
    cur.execute(PIT_QUERY)
    pit_in = {(row[0], row[1], row[2]) for row in cur.fetchall()}
    cur.close()

    # Cumulative race clock per (session, driver) — sessions are per driver,
    # so grouping by session_id alone is exact.  Pit/SC laps stay in the
    # cumulative time (they are real track time); the SAMPLE filters drop
    # them from the training set afterwards.
    laps["cum"] = laps.groupby("session_id")["lap_time"].cumsum()
    laps["is_pit_in"] = [
        (sid, did, lap) in pit_in
        for sid, did, lap in zip(laps["session_id"], laps["driver_id"],
                                 laps["lap_number"])
    ]
    laps["is_pit_out"] = [
        (sid, did, lap + 1) in pit_in
        for sid, did, lap in zip(laps["session_id"], laps["driver_id"],
                                 laps["lap_number"])
    ]
    return laps, pit_in


def build_pairs(laps):
    """Rebuild the per-lap leader/chaser head-to-head from race clocks.

    Returns a wide DataFrame with one row per battle lap (gap before <=
    GAP_WINDOW_S), leader/chaser side columns, and the two targets
    (closing_rate_s, overtake).
    """
    rows = []
    # Per-session median for the SC/VSC filter (valid, sane laps only).
    sane = laps[(laps["is_valid"] == 1) &
                laps["lap_time"].between(60.0, 180.0)]
    medians = sane.groupby("session_id")["lap_time"].median()
    laps = laps.copy()
    # A session with no sane laps gets no median -> treat its laps as racing
    # laps (median := own lap time, so the SC ratio filter never fires).
    laps["_session_median"] = (laps["session_id"].map(medians)
                                .fillna(laps["lap_time"]))
    laps["_is_sc"] = laps["lap_time"] > laps["_session_median"] * SC_VSC_REDFLAG_RATIO

    # Stint index per session: compound change starts a new stint.
    laps = laps.sort_values(["session_id", "lap_number"])
    laps["_stint"] = (
        laps["tyre_compound"] != laps.groupby("session_id")["tyre_compound"].shift()
    ).groupby(laps["session_id"]).cumsum()
    laps["_lap_in_stint"] = laps.groupby(["session_id", "_stint"]).cumcount()

    for (track_name, date), race in laps.groupby(["track_name", "date"]):
        sessions = {sid: g for sid, g in race.groupby("session_id")}
        if len(sessions) < 2:
            continue
        sids = sorted(sessions)
        for i in range(len(sids)):
            for j in range(i + 1, len(sids)):
                a = sessions[sids[i]].set_index("lap_number")
                b = sessions[sids[j]].set_index("lap_number")
                # Same driver imported twice (duplicated session) — skip.
                if a["driver_code"].iloc[0] == b["driver_code"].iloc[0]:
                    continue
                _collect_pair(rows, a, b, track_name, date)

    if not rows:
        return pd.DataFrame()
    pairs = pd.DataFrame(rows)
    # Order columns sensibly for the CSV artifact.
    front = ["track_name", "date", "lap_number", "year",
             "leader_code", "chaser_code", "gap_before_s", "gap_after_s",
             "closing_rate_s", "overtake"]
    pairs = pairs[[c for c in front if c in pairs.columns]
                  + [c for c in pairs.columns if c not in front]]
    return pairs


def _collect_pair(rows, a, b, track_name, date):
    """Emit battle-lap samples for one session pair (a and b are per-lap dfs)."""
    common = sorted(set(a.index) & set(b.index))
    for L in common:
        if L <= 1:
            continue  # lap 1 has no meaningful gap-before (race start)
        prev = L - 1
        if prev not in a.index or prev not in b.index:
            continue
        pa, pb = a.loc[prev, "cum"], b.loc[prev, "cum"]
        ca, cb = a.loc[L, "cum"], b.loc[L, "cum"]
        if any(pd.isna(x) for x in (pa, pb, ca, cb)):
            continue
        gap_before = abs(pa - pb)
        if gap_before > GAP_WINDOW_S:
            continue
        gap_after = abs(ca - cb)

        # Leader / chaser by cumulative position entering the lap.
        if pa < pb:
            lead, chase = a.loc[L], b.loc[L]
            lead_p, chase_p = a.loc[prev], b.loc[prev]
            lead_code, chase_code = a.loc[L, "driver_code"], b.loc[L, "driver_code"]
        else:
            lead, chase = b.loc[L], a.loc[L]
            lead_p, chase_p = b.loc[prev], a.loc[prev]
            lead_code, chase_code = b.loc[L, "driver_code"], a.loc[L, "driver_code"]

        # Role swap across the lap => overtake candidate.
        swapped = (ca < cb) != (pa < pb)

        # --- Sample exclusions -------------------------------------------
        # Invalid laps, pit in/out laps, SC/VSC laps, stint warm-up laps.
        if lead["is_valid"] != 1 or chase["is_valid"] != 1:
            continue
        if lead["is_pit_in"] or chase["is_pit_in"]:
            continue
        if lead["is_pit_out"] or chase["is_pit_out"]:
            continue
        if lead_p["is_pit_in"]:  # swap caused by the leader pitting last lap
            continue
        if lead["_is_sc"] or chase["_is_sc"]:
            continue
        if lead["_lap_in_stint"] < STINT_WARMUP_LAPS:
            continue
        if chase["_lap_in_stint"] < STINT_WARMUP_LAPS:
            continue

        # Overtake labels only trusted from a short gap; swaps from further
        # back are pit/incident artifacts (pit laps already excluded, so
        # these are rare) — drop the row rather than mislabel it.
        if swapped and gap_before > OVERTAKE_GAP_MAX_S:
            continue
        overtake = 1 if swapped else 0

        year = pd.to_datetime(date).year
        rows.append({
            "track_name": track_name, "date": date, "lap_number": int(L),
            "year": int(year),
            "leader_code": str(lead_code), "chaser_code": str(chase_code),
            "leader_tyre_compound": lead["tyre_compound"],
            "chaser_tyre_compound": chase["tyre_compound"],
            "leader_tyre_age": float(lead["tyre_age"] or 0.0),
            "chaser_tyre_age": float(chase["tyre_age"] or 0.0),
            "leader_fuel_load": lead["fuel_load"],
            "chaser_fuel_load": chase["fuel_load"],
            "leader_energy_end_mj": lead_p["energy_end_mj"],
            "chaser_energy_end_mj": chase_p["energy_end_mj"],
            "leader_lap_time": lead["lap_time"],
            "chaser_lap_time": chase["lap_time"],
            "gap_before_s": round(gap_before, 3),
            "gap_after_s": round(gap_after, 3),
            "closing_rate_s": round(gap_before - gap_after, 3),
            "overtake": overtake,
        })


def build_features(pairs):
    """Add the engineered head-to-head features (pace gap, diffs, dummies).

    Returns (X, y_close, y_overtake, feature_names, stats) where stats logs
    how many samples were dropped and why (missing pace models, imputed
    energy, etc.).
    """
    stats = {"pace_ok": 0, "pace_missing": 0, "pace_missing_reasons": {},
             "energy_imputed": 0, "energy_total": 0}
    feats = []
    for _, r in pairs.iterrows():
        pace, detail = compute_pace_gap(
            leader_code=r["leader_code"], chaser_code=r["chaser_code"],
            track_name=r["track_name"], lap_number=int(r["lap_number"]),
            leader_tyre_compound=r["leader_tyre_compound"],
            chaser_tyre_compound=r["chaser_tyre_compound"],
            leader_tyre_age=r["leader_tyre_age"],
            chaser_tyre_age=r["chaser_tyre_age"],
            year=int(r["year"]),
        )
        if pace is None:
            stats["pace_missing"] += 1
            reason = str(detail).split(":")[0][:40]
            stats["pace_missing_reasons"][reason] = \
                stats["pace_missing_reasons"].get(reason, 0) + 1
            continue
        stats["pace_ok"] += 1

        fuel_diff = (float(r["chaser_fuel_load"] or 0.0)
                     - float(r["leader_fuel_load"] or 0.0))
        e_lead = r["leader_energy_end_mj"]
        e_chase = r["chaser_energy_end_mj"]
        if pd.isna(e_lead) or pd.isna(e_chase):
            energy_diff = 0.0
            stats["energy_imputed"] += 1
        else:
            energy_diff = float(e_chase) - float(e_lead)
        stats["energy_total"] += 1

        feats.append({
            "_pair_idx": r.name,
            "gap_before_s": r["gap_before_s"],
            "pace_gap_s": pace,
            "chaser_age_minus_leader": r["chaser_tyre_age"] - r["leader_tyre_age"],
            "chaser_tyre_advantage": tyre_advantage(r["chaser_tyre_compound"],
                                                     r["leader_tyre_compound"]),
            "fuel_diff_kg": round(fuel_diff, 3),
            "energy_diff_mj": round(energy_diff, 3),
            "phase": race_phase_index(int(r["lap_number"])),
            "era": era_bucket(int(r["year"])),
            "track_name": _canonical_track_name(r["track_name"]),
            "closing_rate_s": r["closing_rate_s"],
            "overtake": r["overtake"],
        })

    if not feats:
        return None, None, None, [], stats, None, None
    feat_df = pd.DataFrame(feats)
    # Keep the ORIGINAL pair row index so race-holdout grouping stays aligned
    # with X (feat_df gets a fresh RangeIndex, pairs does not).
    pair_idx = feat_df["_pair_idx"]
    y_close = feat_df["closing_rate_s"].astype(float)
    y_overtake = feat_df["overtake"].astype(int)
    X = pd.get_dummies(
        feat_df.drop(columns=["closing_rate_s", "overtake", "_pair_idx"]),
        columns=["phase", "era", "track_name"],
        prefix={"phase": "phase", "era": "era", "track_name": "track"},
    )
    feature_names = list(X.columns)
    groups = pairs.loc[pair_idx, "track_name"]  # aligned with X rows
    years = pairs.loc[pair_idx, "year"].astype(int).to_numpy()
    return X, y_close, y_overtake, feature_names, stats, groups, years


def make_temporal_split(X, y, years, cutoff=TEMPORAL_SPLIT_CUTOFF):
    """Time-ordered train/test split for the overtake head.

    Train on every season BEFORE ``cutoff``; hold out ``cutoff`` and later
    entirely — model selection, isotonic calibration and testing all happen
    out-of-time.  A random split would let future-regime races leak into the
    fit and flatter every metric; the reliability table would then describe
    interpolation, not the live use case (predicting forward).

    Returns (train_idx, test_idx, info) with POSITIONAL indices into X/y, or
    (None, None, reason_dict) when the temporal split cannot support training
    + calibration (either side short on samples or positives) — the caller
    falls back to a random 20% split and records why.
    """
    yrs = pd.Series(years).reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    train_mask = yrs < cutoff
    test_mask = ~train_mask
    if not train_mask.any() or not test_mask.any():
        return None, None, {"type": "random_20pct_fallback",
                            "reason": f"no samples on one side of the "
                                      f"{cutoff} cutoff"}
    if int(y[train_mask].sum()) < 3 or int(y[test_mask].sum()) < 3:
        return None, None, {"type": "random_20pct_fallback",
                            "reason": "fewer than 3 positive labels on one "
                                      f"side of the {cutoff} cutoff"}
    info = {
        "type": "time_ordered",
        "cutoff_year": int(cutoff),
        "train_years": sorted(int(v) for v in yrs[train_mask].unique()),
        "test_years": sorted(int(v) for v in yrs[test_mask].unique()),
        "train_samples": int(train_mask.sum()),
        "test_samples": int(test_mask.sum()),
        "train_positives": int(y[train_mask].sum()),
        "test_positives": int(y[test_mask].sum()),
    }
    return np.where(train_mask)[0], np.where(test_mask)[0], info


def train_and_eval_reg(X_tr, y_tr, X_te, y_te):
    results = {}
    lr = LinearRegression()
    lr.fit(X_tr, y_tr)
    yp = lr.predict(X_te)
    results["LinearRegression"] = (lr, {
        "mae": mean_absolute_error(y_te, yp),
        "rmse": np.sqrt(mean_squared_error(y_te, yp)),
        "r2": r2_score(y_te, yp),
        "predictions": yp, "y_test": y_te,
    })
    rf = RandomForestRegressor(n_estimators=200, max_depth=10,
                               min_samples_split=4, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    yp = rf.predict(X_te)
    results["RandomForest"] = (rf, {
        "mae": mean_absolute_error(y_te, yp),
        "rmse": np.sqrt(mean_squared_error(y_te, yp)),
        "r2": r2_score(y_te, yp),
        "predictions": yp, "y_test": y_te,
    })
    return results


def train_and_eval_clf(X_tr, y_tr, X_te, y_te):
    results = {}
    def _score(y_true, y_pred, y_prob):
        if len(set(y_true)) < 2:
            # Single-class test split: AUC undefined; -1 so selection skips it.
            auc = -1.0
        else:
            auc = roc_auc_score(y_true, y_prob)
        return {
            "auc": auc,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "predictions": y_pred, "probabilities": y_prob, "y_test": y_true,
        }

    lr = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)
    lr.fit(X_tr, y_tr)
    results["LogisticRegression"] = (lr, _score(
        y_te, lr.predict(X_te), lr.predict_proba(X_te)[:, 1]))

    rf = RandomForestClassifier(n_estimators=200, max_depth=8,
                                min_samples_split=4, class_weight="balanced",
                                random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    results["RandomForest"] = (rf, _score(
        y_te, rf.predict(X_te), rf.predict_proba(X_te)[:, 1]))
    return results


def fmt_clf(m):
    return (f"AUC {m['auc']:.3f}  acc {m['accuracy']:.3f}  "
            f"prec {m['precision']:.3f}  rec {m['recall']:.3f}")


def fit_isotonic_calibrator(clf, X_te, y_te, n_bins=10,
                            fitted_on="held-out test split (20% of training pairs)",
                            framing=None):
    """Fit an IsotonicRegression calibrator on the held-out test split.

    Uses the raw predict_proba scores from the already-selected best
    classifier (so the calibrator never touches training data) and returns
    (calibrator, calibration_info) where calibration_info contains:

      * brier_raw        — Brier score of the uncalibrated model on the test set
      * brier_calibrated — Brier score after isotonic mapping
      * ece_raw          — Expected Calibration Error before calibration
      * ece_calibrated   — Expected Calibration Error after calibration
      * reliability_bins — list of {bin_lo, bin_hi, bin_center, mean_predicted,
                           fraction_positive, n_samples, delta, thin} dicts,
                           one per fixed-width probability bin (width=1/n_bins).
                           ``thin`` is True when n_samples < 5.

    When the test set has fewer than 10 positive samples (too few to fit
    a meaningful calibration curve) the function raises RuntimeError so the
    caller can skip saving the calibrator rather than silently storing garbage.
    """
    y_arr = np.array(y_te)
    if y_arr.sum() < 3:
        raise RuntimeError(
            f"Only {int(y_arr.sum())} positive samples on the test split — "
            "too few to fit a meaningful isotonic calibrator.  Rerun with "
            "more data or a larger training set.")

    raw_probs = clf.predict_proba(X_te)[:, 1]

    # Fit isotonic regression: maps raw scores monotonically to [0,1].
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_probs, y_arr)
    cal_probs = calibrator.predict(raw_probs)

    # Brier scores (lower = better).
    brier_raw = float(brier_score_loss(y_arr, raw_probs))
    brier_cal = float(brier_score_loss(y_arr, cal_probs))

    # Fixed-width reliability bins.
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    for i in range(n_bins):
        lo, hi = float(bin_edges[i]), float(bin_edges[i + 1])
        mask = (raw_probs >= lo) & (raw_probs < hi if i < n_bins - 1 else raw_probs <= hi)
        n = int(mask.sum())
        if n > 0:
            mean_pred = float(raw_probs[mask].mean())
            frac_pos = float(y_arr[mask].mean())
        else:
            mean_pred = (lo + hi) / 2.0
            frac_pos = 0.0
        bins.append({
            "bin_lo": round(lo, 3),
            "bin_hi": round(hi, 3),
            "bin_center": round((lo + hi) / 2.0, 3),
            "mean_predicted": round(mean_pred, 4),
            "fraction_positive": round(frac_pos, 4),
            "n_samples": n,
            "delta": round(frac_pos - mean_pred, 4),
            "thin": n < 5,
        })

    # ECE: weighted mean |predicted - actual| across non-empty bins.
    def _ece(probs):
        total = len(probs)
        err = 0.0
        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            mask = (probs >= lo) & (probs < hi if i < n_bins - 1 else probs <= hi)
            n = mask.sum()
            if n > 0:
                err += (n / total) * abs(probs[mask].mean() - y_arr[mask].mean())
        return float(err)

    ece_raw = _ece(raw_probs)
    ece_cal = _ece(cal_probs)

    calibration_info = {
        "n_test_samples": int(len(y_arr)),
        "n_test_positives": int(y_arr.sum()),
        "brier_raw": round(brier_raw, 5),
        "brier_calibrated": round(brier_cal, 5),
        "brier_improvement": round(brier_raw - brier_cal, 5),
        "ece_raw": round(ece_raw, 5),
        "ece_calibrated": round(ece_cal, 5),
        "ece_improvement": round(ece_raw - ece_cal, 5),
        "n_bins": n_bins,
        "reliability_bins": bins,
        "fitted_on": fitted_on,
        "framing": (framing or "Raw scores mapped monotonically to true "
                    "empirical frequencies on held-out races."),
        "method": "IsotonicRegression(out_of_bounds='clip')",
    }
    return calibrator, calibration_info


def main():
    global GAP_WINDOW_S, OVERTAKE_GAP_MAX_S
    parser = argparse.ArgumentParser(
        description="Train the P0 dual-agent overtake model "
                    "(closing-rate regressor + per-lap overtake classifier).")
    parser.add_argument("--gap-window", type=float, default=GAP_WINDOW_S,
                        help="max gap before (s) for a battle lap sample")
    parser.add_argument("--overtake-max-gap", type=float,
                        default=OVERTAKE_GAP_MAX_S,
                        help="max gap before (s) for a trusted overtake label")
    parser.add_argument("--split-cutoff", type=int,
                        default=TEMPORAL_SPLIT_CUTOFF,
                        help="overtake head trains on seasons BEFORE this "
                             "year; this year and later are held out for "
                             "selection/calibration/testing (out-of-time)")
    args = parser.parse_args()
    GAP_WINDOW_S = args.gap_window
    OVERTAKE_GAP_MAX_S = args.overtake_max_gap

    print("=" * 60)
    print("P0 DUAL-AGENT OVERTAKE MODEL — TRAINING")
    print("=" * 60)

    conn = get_db_connection()
    print("\n[DATABASE] Loading race laps...")
    laps, _pit = load_race_laps(conn)
    conn.close()
    if laps.empty:
        print("[ERROR] No race laps found in database.")
        sys.exit(1)

    print(f"[INFO] Raw race laps: {len(laps)} across "
          f"{laps['track_name'].nunique()} tracks")

    print("\n[PAIRS] Rebuilding leader/chaser head-to-head from race clocks...")
    pairs = build_pairs(laps)
    if pairs.empty or len(pairs) < MIN_PAIR_LAPS:
        print(f"[ERROR] Only {len(pairs)} battle laps built "
              f"(need >= {MIN_PAIR_LAPS}). Nothing to train.")
        sys.exit(1)
    print(f"[INFO] Battle laps (gap <= {GAP_WINDOW_S:.1f}s): {len(pairs)}")
    print(f"[INFO] Overtake labels: {int(pairs['overtake'].sum())} "
          f"({100 * pairs['overtake'].mean():.1f}% of battle laps)")
    print("  per season (time-ordering reference):")
    for yr, g in (pairs.groupby("year")["overtake"]
                  .agg(["size", "sum"]).iterrows()):
        print(f"    {yr}: {int(g['size'])} battle laps, {int(g['sum'])} overtakes")
    by_race = pairs.groupby(["track_name", "date"]).size().sort_values(ascending=False)
    print("  per race (top 10):")
    for (t, d), n in by_race.head(10).items():
        print(f"    {str(d)[:10]} {str(t):<42} {n} battle laps")

    print("\n[FEATURES] Engineering head-to-head features...")
    X, y_close, y_overtake, feature_names, stats, groups, years = \
        build_features(pairs)
    if X is None:
        print("[ERROR] No sample could be scored by the per-driver pace models.")
        sys.exit(1)
    print(f"[INFO] Usable samples: {len(X)}  "
          f"(pace-gap unavailable: {stats['pace_missing']} "
          f"[{stats['pace_missing_reasons']}])")
    if stats["energy_total"]:
        print(f"[INFO] Energy diff imputed to 0 on "
              f"{stats['energy_imputed']}/{stats['energy_total']} samples "
              f"(race_state not written for those races)")
    print(f"[INFO] Feature count: {len(feature_names)}")

    # ------------------------------------------------------------------
    # CLOSING-RATE MODEL (regression)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("HEAD 1/2 — GAP-DELTA CLOSING RATE (regression)")
    print("=" * 60)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_close, test_size=0.2, random_state=42)
    within = train_and_eval_reg(X_tr, y_tr, X_te, y_te)
    for name, (_, m) in within.items():
        print(f"  {name:<16} MAE {m['mae']:.3f}s  RMSE {m['rmse']:.3f}s  "
              f"R2 {m['r2']:.3f}")
    close_best = min(within, key=lambda k: within[k][1]["mae"])
    close_model = within[close_best][0]
    close_metrics = within[close_best][1]
    print(f"  [BEST] {close_best} (within-race MAE {close_metrics['mae']:.3f}s)")

    # Unseen-race holdout, for transparency (mirrors the lap trainer).
    if groups.nunique() > 1:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        tr_i, te_i = next(gss.split(X, y_close, groups=groups))
        unseen = train_and_eval_reg(X.iloc[tr_i], y_close.iloc[tr_i],
                                    X.iloc[te_i], y_close.iloc[te_i])
        print("  unseen-race (transparency): "
              + "  ".join(f"{k} MAE {m['mae']:.3f}"
                          for k, (_, m) in unseen.items()))

    # ------------------------------------------------------------------
    # OVERTAKE PROBABILITY MODEL (classification)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("HEAD 2/2 — PER-LAP OVERTAKE PROBABILITY (classification)")
    print("=" * 60)
    print(f"  Positive rate: {y_overtake.mean():.1%}  "
          f"(baseline always-0 accuracy {1 - y_overtake.mean():.1%})")
    # TIME-ORDERED split: train on seasons before the cutoff, hold out the
    # later seasons for selection/calibration/testing.  Falls back to a
    # random 20% split only when the held-out years lack positives (recorded
    # in the artifacts either way).
    tr_idx, te_idx, split_info = make_temporal_split(
        X, y_overtake, years, cutoff=args.split_cutoff)
    if tr_idx is not None:
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y_overtake.iloc[tr_idx], y_overtake.iloc[te_idx]
        print(f"  [SPLIT] TIME-ORDERED: train {split_info['train_years']} "
              f"({split_info['train_samples']} samples / "
              f"{split_info['train_positives']} positives)  ->  "
              f"hold out {split_info['test_years']} "
              f"({split_info['test_samples']} samples / "
              f"{split_info['test_positives']} positives)")
    else:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y_overtake, test_size=0.2, random_state=42)
        print(f"  [SPLIT] random 20% fallback — {split_info.get('reason')}")
    clf_ev = train_and_eval_clf(X_tr, y_tr, X_te, y_te)
    for name, (_, m) in clf_ev.items():
        print(f"  {name:<18} {fmt_clf(m)}")
    ov_best = max(clf_ev, key=lambda k: clf_ev[k][1]["auc"])
    ov_model = clf_ev[ov_best][0]
    ov_metrics = clf_ev[ov_best][1]
    print(f"  [BEST] {ov_best} (within-race ROC-AUC {ov_metrics['auc']:.3f})")

    if groups.nunique() > 1:
        tr_i, te_i = next(gss.split(X, y_overtake, groups=groups))
        unseen_c = train_and_eval_clf(X.iloc[tr_i], y_overtake.iloc[tr_i],
                                      X.iloc[te_i], y_overtake.iloc[te_i])
        print("  unseen-race (transparency): "
              + "  ".join(f"{k} AUC {m['auc']:.3f}"
                          for k, (_, m) in unseen_c.items()))

    # ------------------------------------------------------------------
    # ISOTONIC CALIBRATION
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ISOTONIC CALIBRATION -- mapping raw scores to true probabilities")
    print("=" * 60)
    isotonic_calibrator = None
    isotonic_info = None
    if split_info.get("type") == "time_ordered":
        test_years = split_info["test_years"]
        iso_fitted_on = (f"held-out later seasons ({test_years[0]}-"
                         f"{test_years[-1]}) — time-ordered split")
        iso_framing = ("Out-of-time calibration: raw scores mapped to observed "
                       "pass rates on LATER seasons the model never saw in "
                       "training — the reliability table is honest, not "
                       "flattering.")
    else:
        iso_fitted_on = "held-out test split (20% of training pairs)"
        iso_framing = None
    try:
        isotonic_calibrator, isotonic_info = fit_isotonic_calibrator(
            ov_model, X_te, y_te, fitted_on=iso_fitted_on, framing=iso_framing)
        print(f"  Brier score:  raw {isotonic_info['brier_raw']:.5f}  "
              f"->  calibrated {isotonic_info['brier_calibrated']:.5f}  "
              f"(improvement {isotonic_info['brier_improvement']:+.5f})")
        print(f"  ECE:          raw {isotonic_info['ece_raw']:.5f}  "
              f"->  calibrated {isotonic_info['ece_calibrated']:.5f}  "
              f"(improvement {isotonic_info['ece_improvement']:+.5f})")
        print(f"  Test set:     {isotonic_info['n_test_samples']} samples  "
              f"({isotonic_info['n_test_positives']} positives)")
        print("  Reliability bins (raw predicted -> actual pass rate):")
        print(f"  {'Bin':<12} {'Mean pred':>9} {'Actual':>8} {'N':>5}  "
              f"{'Delta':>7}  note")
        for b in isotonic_info["reliability_bins"]:
            if b["n_samples"] == 0:
                continue
            note = "thin" if b["thin"] else ""
            delta_txt = f"{b['delta']:+.4f}"
            print(f"  [{b['bin_lo']:.2f},{b['bin_hi']:.2f})  "
                  f"{b['mean_predicted']:>9.4f} "
                  f"{b['fraction_positive']:>8.4f} "
                  f"{b['n_samples']:>5}  "
                  f"{delta_txt:>7}  {note}")
    except RuntimeError as exc:
        print(f"  [SKIP] Isotonic calibration skipped: {exc}")

    # ------------------------------------------------------------------
    # Artifacts
    # ------------------------------------------------------------------
    out_dir = OVERTAKE_MODEL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(close_model, out_dir / "closing_model.pkl")
    joblib.dump(ov_model, out_dir / "overtake_model.pkl")
    joblib.dump(feature_names, out_dir / "feature_names.pkl")
    if isotonic_calibrator is not None:
        joblib.dump(isotonic_calibrator, out_dir / "isotonic_calibrator.pkl")
    pairs.to_csv(out_dir / "training_pairs.csv", index=False)

    metadata = {
        "p0_dual_agent_overtake": True,
        "closing_model": close_best,
        "overtake_model": ov_best,
        "metrics": {
            "closing_rate": {k: round(v, 4) for k, v in close_metrics.items()
                             if k not in ("predictions", "y_test")},
            "overtake": {k: round(v, 4) for k, v in ov_metrics.items()
                         if k not in ("predictions", "probabilities", "y_test")},
        },
        "pair_construction": {
            "gap_window_s": GAP_WINDOW_S,
            "overtake_gap_max_s": OVERTAKE_GAP_MAX_S,
            "battle_laps": int(len(pairs)),
            "overtake_labels": int(pairs["overtake"].sum()),
            "overtake_rate": round(float(pairs["overtake"].mean()), 4),
            "pace_gap_unavailable": stats["pace_missing"],
            "races": {f"{d} {t}": int(n)
                      for (t, d), n in pairs.groupby(
                          ["track_name", "date"]).size().items()},
        },
        "energy_diff": {
            "imputed_to_zero": stats["energy_imputed"],
            "samples": stats["energy_total"],
            "coverage": (round(1 - stats["energy_imputed"] / stats["energy_total"], 4)
                         if stats["energy_total"] else 0.0),
        },
        "training_samples": int(len(X_tr)),
        "test_samples": int(len(X_te)),
        "split": split_info,
        "features": feature_names,
        "coverage": {"tracks": covered_tracks(feature_names)},
        "isotonic_calibration": isotonic_info,  # None when skipped
        "trained_at": datetime.now().isoformat(),
    }
    with open(out_dir / "model_info.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    with open(out_dir / "model_info.txt", "w", encoding="utf-8") as f:
        f.write("P0 DUAL-AGENT OVERTAKE MODEL\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Closing-rate model:  {close_best}  "
                f"(within-race MAE {close_metrics['mae']:.3f}s)\n")
        f.write(f"Overtake model:      {ov_best}  "
                f"(within-race ROC-AUC {ov_metrics['auc']:.3f})\n")
        f.write(f"Battle laps:         {len(pairs)}  "
                f"(gap <= {GAP_WINDOW_S:.1f}s), "
                f"{int(pairs['overtake'].sum())} overtake labels "
                f"({100 * pairs['overtake'].mean():.1f}%)\n")
        f.write(f"Features:            {len(feature_names)}\n")
        if split_info.get("type") == "time_ordered":
            f.write(f"Split:               time-ordered — train "
                    f"{split_info['train_years']}, hold out "
                    f"{split_info['test_years']} (out-of-time)\n")
        else:
            f.write(f"Split:               random 20% fallback "
                    f"({split_info.get('reason')})\n")
        f.write(f"Energy coverage:     {metadata['energy_diff']['coverage']:.1%} "
                f"(race_state rows present)\n")
        if isotonic_info:
            f.write(f"Isotonic calibration: fitted  "
                    f"(Brier {isotonic_info['brier_raw']:.5f} → "
                    f"{isotonic_info['brier_calibrated']:.5f},  "
                    f"ECE {isotonic_info['ece_raw']:.5f} → "
                    f"{isotonic_info['ece_calibrated']:.5f})\n")
        else:
            f.write("Isotonic calibration: skipped (too few positives on test split)\n")
        f.write(f"Trained at:          {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Tracks covered:\n")
        for t in covered_tracks(feature_names):
            f.write(f"  {t}\n")
        f.write("\nFeatures:\n")
        for fn in feature_names:
            f.write(f"  {fn}\n")

    print("\n" + "=" * 60)
    print(f"[SAVED] {out_dir}/closing_model.pkl")
    print(f"[SAVED] {out_dir}/overtake_model.pkl")
    print(f"[SAVED] {out_dir}/feature_names.pkl")
    if isotonic_calibrator is not None:
        print(f"[SAVED] {out_dir}/isotonic_calibrator.pkl")
    else:
        print("[SKIP]  isotonic_calibrator.pkl (skipped — see reason above)")
    print(f"[SAVED] {out_dir}/model_info.json")
    print(f"[SAVED] {out_dir}/model_info.txt")
    print(f"[SAVED] {out_dir}/training_pairs.csv")
    print("=" * 60)
    print("Next: P1 can consume via overtake_inference.predict_overtake(...)")


if __name__ == "__main__":
    main()