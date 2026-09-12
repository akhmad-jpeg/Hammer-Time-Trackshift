import os
import sys
from pathlib import Path
# pyrefly: ignore [missing-import]
import fastf1

cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "f1_cache"))
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

print(f"Loading 2026 Australia session with cache at {cache_dir}...", flush=True)
session = fastf1.get_session(2026, "Australia", "R")
session.load(telemetry=False, laps=True, weather=False)

lec_laps = session.laps.pick_driver('LEC')
rus_laps = session.laps.pick_driver('RUS')

print(f"LEC laps: {len(lec_laps)}, RUS laps: {len(rus_laps)}", flush=True)

merged = lec_laps[['LapNumber', 'Position', 'LapTime', 'PitInTime', 'PitOutTime']].merge(
    rus_laps[['LapNumber', 'Position', 'LapTime', 'PitInTime', 'PitOutTime']],
    on='LapNumber', suffixes=('_LEC', '_RUS')
)

# Sort by LapNumber
merged = merged.sort_values('LapNumber')

for idx, row in merged.iterrows():
    lap = int(row['LapNumber'])
    p_lec = row['Position_LEC']
    p_rus = row['Position_RUS']
    t_lec = row['LapTime_LEC']
    t_rus = row['LapTime_RUS']
    pit_lec = "PIT" if row['PitInTime_LEC'] is not None and str(row['PitInTime_LEC']) != 'NaT' else ""
    pit_rus = "PIT" if row['PitInTime_RUS'] is not None and str(row['PitInTime_RUS']) != 'NaT' else ""
    order = "LEC ahead" if p_lec < p_rus else ("RUS ahead" if p_rus < p_lec else "TIED")
    print(f"Lap {lap:02d} | P_LEC: {p_lec} {pit_lec:3s} | P_RUS: {p_rus} {pit_rus:3s} | {order}", flush=True)
