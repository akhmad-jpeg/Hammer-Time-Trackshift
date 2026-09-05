"""Shared feature engineering and vector alignment pipeline for F1 Telemetry Platform.

Feature set (training AND prediction must build the same columns):

  * tyre_age                      — numeric, laps since the set was fitted.
  * tyre_<compound>               — one-hot dry/wet compound.
  * track_<name>                  — one-hot track.
  * phase_<0..3>                  — one-hot race-phase bucket derived from
                                    lap_number (see race_phase_index).  The
                                    phase bucket is deliberately NOT a
                                    continuous lap term: within a stint a
                                    continuous lap term is perfectly
                                    collinear with tyre_age (both advance one
                                    per lap), which is what made the old
                                    fuel-load feature unstable.  A coarse
                                    bucket is ~constant inside a stint, so it
                                    can absorb the fuel/race-phase LEVEL that
                                    used to leak into the compound
                                    intercepts.
  * tyre_age:tyre_<compound>      — tyre_age x compound interactions, so each
                                    compound gets its OWN wear slope instead
                                    of sharing one global tyre_age slope.
  * era_<bucket>                   — one-hot season era bucket derived from the
                                    session's year (18_20 / 21 / 22_23 / 24_26).
                                    Cars (and fields) get faster over seasons;
                                    without an era term the compound intercepts
                                    absorb cross-season pace drift.  Adding it
                                    cut the residual "Hard faster than Medium"
                                    inversion from ~0.42 s to ~0.23 s.
  * tyre_age:track_<name>          — tyre_age x track interactions, so each
                                    circuit gets its own age slope.  These let
                                    the training run export MEASURED per-track
                                    wear for the strategy advisor.

Investigation note (why these were added, Sept 2026): the old model predicted
Hard FASTER than Soft and a tyre_age slope that was pure fuel burn.  DB
analysis of the training stints showed Hard is used almost exclusively as a
late-race second stint (mean race fraction ~0.64, light fuel, rubbered track)
while Soft spans race starts (~0.40, heavy fuel).  With no race-phase feature
the compound one-hot intercepts absorbed that phase difference (Hard
"inherited" late-race pace) and the single tyre_age slope absorbed fuel burn.
The phase buckets remove the level confound from the compound intercepts; the
per-compound interactions let wear differ by compound.  A coarse *global*
phase (absolute lap-number buckets rather than a fraction of each race's
length) is used so prediction-time rows can be built from just lap_number,
with no race-length input required.
"""

import pandas as pd

# Absolute lap-number edges of the race-phase buckets.  These approximate
# "heavy fuel race start / early race / mid race / late race" for a typical
# ~60-70 lap grand prix without needing each session's exact race length.
PHASE_EDGES = (13, 26, 42)
PHASE_NAMES = ('early', 'mid1', 'mid2', 'late')

# Season-era buckets (year -> era).  Cars/fields get faster over seasons; the
# era one-hot absorbs that drift so compound intercepts are not biased toward
# whichever seasons a compound happened to be raced in (Hard skews to recent
# seasons, which is part of why the old model ranked it above Medium).
ERA_EDGES = (2021, 2022, 2024)          # era = 0..3 for years <, ==, ..., >
ERA_NAMES = ('18_20', '21', '22_23', '24_26')

# Era used at prediction time when the caller does not know the session's
# year (e.g. the generic /api/predict endpoint): the era with the most
# training laps, so unknown-year rows land on the most representative level.
DEFAULT_ERA = '22_23'


def era_bucket(year):
    """Era bucket index (0..3) for a season year (None-safe -> DEFAULT_ERA)."""
    if year is None:
        return DEFAULT_ERA
    try:
        y = int(year)
    except (TypeError, ValueError):
        return DEFAULT_ERA
    if y < ERA_EDGES[0]:
        return ERA_NAMES[0]
    if y < ERA_EDGES[1]:
        return ERA_NAMES[1]
    if y < ERA_EDGES[2]:
        return ERA_NAMES[2]
    return ERA_NAMES[3]


def race_phase_index(lap_number):
    """Bucket index (0..3) for an absolute lap number."""
    lap = int(lap_number)
    if lap < PHASE_EDGES[0]:
        return 0
    if lap < PHASE_EDGES[1]:
        return 1
    if lap < PHASE_EDGES[2]:
        return 2
    return 3


def add_race_phase(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the race_phase bucket column from lap_number (if present)."""
    if 'lap_number' in df.columns:
        df = df.copy()
        df['race_phase'] = df['lap_number'].map(race_phase_index)
    return df


def add_tyre_age_interactions(df: pd.DataFrame, track_terms: bool = False) -> pd.DataFrame:
    """Add tyre_age x compound (and optionally x track) interaction columns.

    Must run AFTER get_dummies has created the tyre_<c> / track_<t> flags.
    Each flag is 0/1 so the interaction column equals tyre_age on that
    compound/track and 0 elsewhere, letting a linear model fit a
    per-compound (and per-track) wear slope.

    Per-track age terms are only used by the training script (they let it
    export measured per-track wear); prediction-time row builders only need
    the compound terms.
    """
    df = df.copy()
    if 'tyre_age' not in df.columns:
        return df
    for col in df.columns:
        if col.startswith('tyre_') and col not in ('tyre_age', 'tyre_load') \
                and ':' not in col:
            df[f'tyre_age:{col}'] = df['tyre_age'] * df[col]
        if track_terms and col.startswith('track_') and ':' not in col:
            df[f'tyre_age:{col}'] = df['tyre_age'] * df[col]
    return df


def preprocess_laps_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess raw database laps dataframe for ML training/prediction.

    1. Normalizes string casing for track_name and tyre_compound.
    2. Derives the race-phase bucket from lap_number (lap_number itself is
       then dropped — within a stint it is collinear with tyre_age).
    3. Derives the era bucket from session_date/_year (session_date is then
       dropped — era is the only time level the model needs).
    4. One-hot encodes tyre_compound, track_name, race_phase and era.
    5. Adds tyre_age x compound interactions.
    """
    df = df.copy()
    if 'track_name' in df.columns:
        df['track_name'] = df['track_name'].str.strip().str.title()
    if 'tyre_compound' in df.columns:
        df['tyre_compound'] = df['tyre_compound'].str.strip()

    df = add_race_phase(df)

    year = None
    if 'session_date' in df.columns:
        year = pd.to_datetime(df['session_date'], errors='coerce').dt.year
        df = df.drop(columns=['session_date'])
    elif '_year' in df.columns:
        year = df['_year']
        df = df.drop(columns=['_year'])
    if year is not None:
        df['era'] = year.map(era_bucket)

    if 'lap_number' in df.columns:
        # Not a model feature — within a stint it is collinear with tyre_age.
        df = df.drop(columns=['lap_number'])

    cats = ['tyre_compound', 'track_name']
    if 'race_phase' in df.columns:
        cats.append('race_phase')
    if 'era' in df.columns:
        cats.append('era')
    df_encoded = pd.get_dummies(df, columns=cats)
    df_encoded = add_tyre_age_interactions(df_encoded)
    return df_encoded


def covered_tracks(feature_names: list) -> list:
    """Sorted list of track names the trained model can predict for."""
    return sorted(f.replace('track_', '') for f in feature_names if f.startswith('track_'))


def covered_tyres(feature_names: list) -> list:
    """Sorted list of tyre compounds the trained model can predict for.

    Interaction features are named 'tyre_age:tyre_<compound>'; they start with
    'tyre_' but are not compounds, so they are excluded via the ':' guard.
    """
    return sorted(f.replace('tyre_', '') for f in feature_names
                  if f.startswith('tyre_') and f not in ('tyre_age', 'tyre_load')
                  and ':' not in f)


def _normalise_track_name(track_name: str) -> str:
    """Apply the same normalisation used during training (strip + title-case).

    Track names from the sessions API keep the raw FastF1 casing (e.g.
    "Circuit de Barcelona-Catalunya") while the model features are trained
    on the title-cased form ("Circuit De Barcelona-Catalunya") — both must
    resolve to the same feature.
    """
    return str(track_name).strip().title()


def validate_model_inputs(tyre_compound: str, track_name: str, feature_names: list) -> None:
    """Raise ValueError with a clear message when a tyre/track is not covered.

    The one-hot features only exist for values seen during training, so any
    unseen track or compound would otherwise produce an all-zero row and a
    silently meaningless prediction.
    """
    tyre_compound = str(tyre_compound).strip()
    track_name    = _normalise_track_name(track_name)

    tyre_feature  = f'tyre_{tyre_compound}'
    track_feature = f'track_{track_name}'

    if tyre_feature not in feature_names:
        available = covered_tyres(feature_names)
        raise ValueError(
            f"Unknown tyre compound '{tyre_compound}'. "
            f"Model was trained on: {', '.join(available) or '(none)'}."
        )
    if track_feature not in feature_names:
        available = covered_tracks(feature_names)
        raise ValueError(
            f"Unknown track '{track_name}'. "
            f"Model was trained on: {', '.join(available) or '(none)'}."
        )


def construct_prediction_input(tyre_age: float, lap_number: int, tyre_compound: str, track_name: str, feature_names: list, year=None) -> pd.DataFrame:
    """Construct a 1-row feature DataFrame aligned with expected feature_names order.

    lap_number is used to derive the race-phase bucket; the fuel/phase level
    enters through that bucket (a continuous lap term would be collinear with
    tyre_age inside a stint).  `year` (the session's season) selects the era
    bucket; when unknown it defaults to the most-representative era so the
    row stays well-defined.
    """
    tyre_compound = str(tyre_compound).strip()
    track_name    = _normalise_track_name(track_name)

    tyre_feature = f'tyre_{tyre_compound}'
    track_feature = f'track_{track_name}'

    input_data = pd.DataFrame(0, index=[0], columns=feature_names)
    if 'tyre_age' in input_data.columns:
        input_data['tyre_age'] = tyre_age
    if tyre_feature in input_data.columns:
        input_data[tyre_feature] = 1
    if track_feature in input_data.columns:
        input_data[track_feature] = 1

    # Race-phase bucket (phase_<0..3>) derived from the absolute lap number.
    phase_col = f'phase_{race_phase_index(lap_number)}'
    if phase_col in input_data.columns:
        input_data[phase_col] = 1

    # Season-era bucket derived from the session's year.
    era_col = f'era_{era_bucket(year)}'
    if era_col in input_data.columns:
        input_data[era_col] = 1

    # tyre_age x compound interactions present in the trained feature set.
    interaction = f'tyre_age:{tyre_feature}'
    if interaction in input_data.columns:
        input_data[interaction] = tyre_age

    return input_data
