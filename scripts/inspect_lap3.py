import os
# pyrefly: ignore [missing-import]
import fastf1
import numpy as np

cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "f1_cache"))
fastf1.Cache.enable_cache(cache_dir)

session = fastf1.get_session(2026, "Australia", "R")
session.load(telemetry=True, laps=True, weather=False)

lap_lec = session.laps.pick_drivers('LEC').pick_lap(3)
lap_rus = session.laps.pick_drivers('RUS').pick_lap(3)

tel_lec = lap_lec.get_telemetry()
tel_rus = lap_rus.get_telemetry()

print("LEC Lap 3 telemetry shape:", tel_lec.shape, "Columns:", list(tel_lec.columns))
print("RUS Lap 3 telemetry shape:", tel_rus.shape, "Columns:", list(tel_rus.columns))
print("LEC Distance range:", tel_lec['Distance'].min(), "->", tel_lec['Distance'].max())
print("RUS Distance range:", tel_rus['Distance'].min(), "->", tel_rus['Distance'].max())
print("LEC X/Y sample:", tel_lec[['Distance', 'X', 'Y', 'Speed', 'Time']].head(2))
print("RUS X/Y sample:", tel_rus[['Distance', 'X', 'Y', 'Speed', 'Time']].head(2))
