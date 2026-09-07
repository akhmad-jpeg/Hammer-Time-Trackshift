# 🖱️ One-Click Launchers

Double-click any file below to run that step of the F1 strategy platform.
Each launcher works from anywhere (it changes to the project root itself)
and prefers the project virtualenv (`.venv\Scripts\python.exe`), falling
back to the `python` on your PATH.

> Run them **in numbered order** the first time. Steps 2–4 need your
> MySQL database to exist and be configured (see the root README:
> `database/schema.sql` + `scripts/config.py` / `DB_*` environment vars).

| # | File | What it does | Notes |
|---|------|--------------|-------|
| 01 | `01_setup_dependencies.bat` | Creates `.venv` if missing and installs `requirements.txt` | Run once; needs internet |
| 02 | `02_import_race.bat` | Imports **one** race (year / race / session / driver) from FastF1 into MySQL | Interactive; re-downloads `f1_cache\` on first run |
| 03 | `03_import_dataset.bat` | Batch-imports one or more drivers across many races/seasons | More data → better model |
| 04 | `04_audit_pit_events.bat` | Checks pit-event consistency (dry run), asks before applying fixes | Optional but recommended after imports |
| 05 | `05_train_model.bat` | Trains LinearRegression vs RandomForest, saves the winner to `ml_models\` **plus one model per driver** in `ml_models\drivers\` | **The training step** |
| 06 | `06_run_server.bat` | Starts the Flask/Waitress dashboard at http://localhost:5000 | Ctrl+C to stop |
| 07 | `07_capture_telemetry.bat` | Live F1 2017/2018 UDP telemetry capture (asks track / tyre / weather / heartbeat interval) | Heartbeat defaults to 5 s — see `scripts\capture_telemetry.py` `--heartbeat` |
| 08 | `08_predict_lap_times.bat` | Interactive lap-time predictor CLI | Needs a trained model (step 05) |
| 09 | `09_analyze_performance.bat` | CLI performance charts + summary report for a picked session | Saved under `analysis\` at the project root |
| 10 | `10_benchmark_models.bat` | Reproducible model-selection benchmark | Informational; does not touch `ml_models\` |
| 11 | `11_run_tests.bat` | Runs the full unit-test suite (no DB needed) | `python -m unittest discover -s tests` · prints a notice and skips if the `tests\` folder isn't there yet |
| 12 | `12_compare_drivers.bat` | Head-to-head driver comparison: same track/tyre/age predicted by each driver's own model | Needs step 05 (per-driver models); chart → `ml_models\comparisons\` |
| 13 | `13_energy_simulate.bat` | Regenerates the synthetic ERS energy trace for a session (balanced / push / lift & coast) | Asks session id (blank = latest 2026 race) + mode; writes `race_state` |

## Typical first-time flow

1. `01_setup_dependencies.bat`
2. Set up MySQL (`database/schema.sql`) and DB credentials (`scripts/config.py`)
3. `02_import_race.bat` and/or `03_import_dataset.bat`
4. `04_audit_pit_events.bat`
5. `05_train_model.bat`
6. `06_run_server.bat` → open http://localhost:5000
7. `12_compare_drivers.bat` (once models exist) → pick two drivers
   (or use the 🏁 Driver Comparison tab in the dashboard)

## Troubleshooting

- **`'python' is not recognized`** → install Python 3.13 and check "Add to PATH".
- **MySQL connection errors** → verify the DB exists and `DB_*` env vars /
  defaults in `scripts/config.py` match your server.
- **`Model not found` / predictor refuses to start** → run `05_train_model.bat` first.
- **Launchers only ever print "not found" errors** → the `.bat` files were moved out of the `launchers\` folder at the project root.
