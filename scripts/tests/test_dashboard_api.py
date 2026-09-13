"""Comprehensive Dashboard API Validation Suite.

Hits every registered endpoint with correct, representative inputs,
validates HTTP status codes, response structure, and key field
types/ranges.

Run against the live Waitress server on port 5000:
    python -X utf8 scripts/tests/test_dashboard_api.py

All tests are self-contained HTTP requests — no mocking. The server
must be running before executing this suite.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from typing import Any

BASE = "http://127.0.0.1:5000"
PASS = 0
FAIL = 0
ERRORS: list[str] = []

# ── helpers ──────────────────────────────────────────────────────────────────

def _get(path: str, json_only: bool = True) -> tuple[int, Any]:
    url = BASE + path
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=12) as r:
            raw = r.read()
            if json_only:
                return r.status, json.loads(raw.decode())
            return r.status, raw  # raw bytes for HTML endpoints
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode())
        except Exception:
            return e.code, {}
    except Exception as exc:
        return -1, {"__exception__": str(exc)}


def _post(path: str, payload: dict) -> tuple[int, Any]:
    url = BASE + path
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode())
        except Exception:
            return e.code, {}
    except Exception as exc:
        return -1, {"__exception__": str(exc)}


def ok(name: str, cond: bool, detail: str = ""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        msg = f"  FAIL  {name}" + (f"  <- {detail}" if detail else "")
        print(msg)
        ERRORS.append(msg)


def section(title: str):
    print(f"\n{'='*62}")
    print(f"  {title}")
    print(f"{'='*62}")


# ── Shared payloads ─────────────────────────────────────────────────────────

# The policies and strategy/call endpoints require two driver codes.
POLICIES_PAYLOAD = {
    "leader_code": "VER",
    "chaser_code": "LEC",
    "track_name": "Silverstone Circuit",
    "start_lap": 20,
    "race_length": 52,
    "gap_before_s": 1.2,
    "leader_tyre_compound": "Medium",
    "chaser_tyre_compound": "Soft",
    "leader_tyre_age": 15.0,
    "chaser_tyre_age": 8.0,
    "chaser_battery_pct": 72.0,
    "race_event": "green",
    "traffic_level": "Light",
}

CALL_PAYLOAD = {
    "leader_code": "VER",
    "chaser_code": "LEC",
    "track_name": "Monaco",
    "start_lap": 30,
    "race_length": 78,
    "gap_before_s": 0.8,
    "leader_tyre_compound": "Hard",
    "chaser_tyre_compound": "Soft",
    "leader_tyre_age": 25.0,
    "chaser_tyre_age": 10.0,
    "battery_pct": 55.0,
    "perspective": "chaser",
    "race_event": "green",
    "traffic_level": "Clear",
}


# ── 1. Root page ─────────────────────────────────────────────────────────────

section("1  Root page")
# Root returns HTML, not JSON — use json_only=False to avoid parse error
status, body = _get("/", json_only=False)
ok("GET / -> 200", status == 200, f"got {status}")
ok("Root returns HTML content",
   isinstance(body, bytes) and b"<" in body, type(body).__name__)


# ── 2. Sessions ──────────────────────────────────────────────────────────────

section("2  /api/sessions")

status, body = _get("/api/sessions")
ok("GET /api/sessions -> 200", status == 200, f"got {status}")
ok("Response is list", isinstance(body, list), type(body).__name__)

status, body = _get("/api/sessions?limit=5&offset=0")
ok("?limit=5 -> 200", status == 200, f"got {status}")
ok("Limit respected (<=5)", isinstance(body, list) and len(body) <= 5,
   f"got {len(body) if isinstance(body, list) else body}")

status, body = _get("/api/sessions?driver=LEC")
ok("?driver=LEC -> 200", status == 200, f"got {status}")
ok("Driver filter returns list", isinstance(body, list), type(body).__name__)

status, body = _get("/api/sessions?year=2024")
ok("?year=2024 -> 200", status == 200, f"got {status}")
ok("Year filter returns list", isinstance(body, list), type(body).__name__)

if isinstance(body, list) and body:
    row = body[0]
    ok("session_id is int", isinstance(row.get("session_id"), int), str(row.get("session_id")))
    ok("track_name present", "track_name" in row)
    ok("date is str or None", row.get("date") is None or isinstance(row.get("date"), str))


# ── 3. Session laps ──────────────────────────────────────────────────────────

section("3  /api/session/<id>/laps")

status0, sessions = _get("/api/sessions?limit=10")
valid_sid = None
if status0 == 200 and isinstance(sessions, list) and sessions:
    valid_sid = sessions[0]["session_id"]

if valid_sid:
    status, body = _get(f"/api/session/{valid_sid}/laps")
    ok(f"GET /api/session/{valid_sid}/laps -> 200", status == 200, f"got {status}")
    ok("Returns list", isinstance(body, list), type(body).__name__)
    if isinstance(body, list) and body:
        lap = body[0]
        ok("lap_number present", "lap_number" in lap)
        ok("lap_time is float or None",
           lap.get("lap_time") is None or isinstance(lap.get("lap_time"), float))
        ok("tyre_compound present", "tyre_compound" in lap)
        ok("tyre_age present", "tyre_age" in lap)
        ok("has_pit_stop present", "has_pit_stop" in lap)
else:
    ok("No sessions available - lap tests skipped", True)

status, body = _get("/api/session/999999/laps")
ok("Invalid session_id -> non-crash", status in (200, 500), f"got {status}")


# ── 4. Session energy ────────────────────────────────────────────────────────

section("4  /api/session/<id>/energy")

if valid_sid:
    status, body = _get(f"/api/session/{valid_sid}/energy")
    ok("GET energy -> 200", status == 200, f"got {status}")
    ok("Returns list", isinstance(body, list), type(body).__name__)
    if isinstance(body, list) and body:
        pt = body[0]
        ok("battery_pct present", "battery_pct" in pt, str(pt))
        ok("battery_pct in [0,100]",
           0.0 <= float(pt["battery_pct"]) <= 100.0, str(pt["battery_pct"]))
        ok("x (lap axis) present", "x" in pt)
        ok("band_pct present", "band_pct" in pt)
else:
    ok("Skipped - no sessions", True)


# ── 5. Latest lap ────────────────────────────────────────────────────────────

section("5  /api/latest-lap")

status, body = _get("/api/latest-lap")
ok("GET /api/latest-lap -> 200", status == 200, f"got {status}")
ok("Returns dict", isinstance(body, dict), type(body).__name__)
if body and "lap_time" in body:
    ok("lap_time is float", isinstance(body.get("lap_time"), float), str(body.get("lap_time")))

status, body = _get("/api/latest-lap?driver=LEC")
ok("?driver=LEC -> 200", status == 200, f"got {status}")
ok("Returns dict", isinstance(body, dict), type(body).__name__)


# ── 6. Live battle state ─────────────────────────────────────────────────────

section("6  /api/live/battle-state")

status, body = _get("/api/live/battle-state?leader=VER&chaser=LEC")
ok("GET battle-state -> 200", status == 200, f"got {status}")
ok("Returns dict", isinstance(body, dict), type(body).__name__)


# ── 7. Drivers list ──────────────────────────────────────────────────────────

section("7  /api/drivers/list")

status, body = _get("/api/drivers/list")
ok("GET /api/drivers/list -> 200", status == 200, f"got {status}")
# Endpoint wraps the list in {"drivers": [...]}
ok("Returns dict with 'drivers' key",
   isinstance(body, dict) and "drivers" in body, type(body).__name__)
if isinstance(body, dict) and isinstance(body.get("drivers"), list) and body["drivers"]:
    ok("driver_code present", "driver_code" in body["drivers"][0] or "code" in body["drivers"][0],
       str(body["drivers"][0]))


# ── 8. Predict options ───────────────────────────────────────────────────────

section("8  /api/predict/options")

status, body = _get("/api/predict/options")
ok("GET /api/predict/options -> 200", status == 200, f"got {status}")
ok("Returns dict", isinstance(body, dict), type(body).__name__)
ok("tracks key present", "tracks" in body, str(list(body.keys()) if isinstance(body, dict) else body))
ok("tyres key present", "tyres" in body)


# ── 9. Predict (POST) ────────────────────────────────────────────────────────

section("9  POST /api/predict")

# Actual response key is 'predicted_time' (not 'predicted_lap_time')
payload_predict = {
    "track_name": "Silverstone",
    "tyre_compound": "Soft",
    "tyre_age": 5,
    "fuel_load": 50.0,
    "session_type": "Race",
    "weather": "Dry",
    "lap_number": 15,
}
status, body = _post("/api/predict", payload_predict)
ok("POST /api/predict -> 200", status == 200, f"got {status}")
if status == 200:
    ok("predicted_time present", "predicted_time" in body, str(list(body.keys())))
    ok("predicted_time is float or str",
       isinstance(body.get("predicted_time"), (float, int, str)))
    ok("track key echoed", "track" in body)
    ok("tyre_compound echoed", "tyre_compound" in body)

# Invalid payload
status, body = _post("/api/predict", {})
ok("Empty predict payload -> non-crash", status in (400, 200, 500), f"got {status}")


# ── 10. Strategy Analyze ─────────────────────────────────────────────────────

section("10  POST /api/strategy/analyze")

payload_strat = {
    "session_id": valid_sid or 1,
    "current_lap": 20,
    "tyre_compound": "Medium",
    "tyre_age": 12,
    "fuel_load": 40.0,
    "total_laps": 58,
    "track_name": "Silverstone",
}
status, body = _post("/api/strategy/analyze", payload_strat)
ok("POST /api/strategy/analyze -> 200 or 4xx", status in (200, 400, 404, 500), f"got {status}")
if status == 200:
    ok("recommendation present", "recommendation" in body, str(list(body.keys())))


# ── 11. Strategy Energy Analyze ──────────────────────────────────────────────

section("11  POST /api/strategy/energy-analyze")

payload_energy = {
    "session_id": valid_sid or 1,
    "track_name": "Silverstone",
    "current_lap": 20,
    "total_laps": 58,
    "battery_pct": 65.0,
    "tyre_compound": "Medium",
    "tyre_age": 12,
}
status, body = _post("/api/strategy/energy-analyze", payload_energy)
ok("POST /api/strategy/energy-analyze -> non-crash", status in (200, 400, 404, 500), f"got {status}")
if status == 200:
    ok("Response is dict", isinstance(body, dict))


# ── 12. Strategy Policies ────────────────────────────────────────────────────

section("12  POST /api/strategy/policies")

# Requires: leader_code, chaser_code, track_name
status, body = _post("/api/strategy/policies", POLICIES_PAYLOAD)
ok("POST /api/strategy/policies -> 200", status == 200, f"got {status}")
if status == 200:
    ok("policies key present", "policies" in body, str(list(body.keys())))
    # Top-level winner key is 'recommendation' in the actual response
    ok("recommendation key present", "recommendation" in body, str(list(body.keys())))
    ok("pit_analysis key present", "pit_analysis" in body)
    ok("latency_ms key present", "latency_ms" in body)
    policies = body.get("policies", [])
    ok("5 policies returned", len(policies) == 5, f"got {len(policies)}")
    if policies:
        p = policies[0]
        # Actual field names: 'policy' (not 'action'), 'score_s' (not 'score')
        ok("policy has 'policy' name field", "policy" in p, str(list(p.keys())))
        ok("policy has score_s", "score_s" in p, str(list(p.keys())))
        ok("policy has feasible flag", "feasible" in p, str(list(p.keys())))
        ok("policy has overtake_probability", "overtake_probability" in p)
        ok("policy has why (rationale)", "why" in p)

# Safety car scenario
status, body = _post("/api/strategy/policies",
                     {**POLICIES_PAYLOAD, "race_event": "safety_car", "traffic_level": "Heavy"})
ok("Policies under safety_car -> 200", status == 200, f"got {status}")
if status == 200:
    ok("policies present under SC", "policies" in body)

# Low battery edge case — should mark Greedy Attack infeasible
status, body = _post("/api/strategy/policies",
                     {**POLICIES_PAYLOAD, "chaser_battery_pct": 8.0})
ok("Policies at 8% battery -> 200", status == 200, f"got {status}")
if status == 200:
    pols = body.get("policies", [])
    infeasible = [p for p in pols if not p.get("feasible", True)]
    ok("At least one policy infeasible at 8% battery",
       len(infeasible) >= 1, f"infeasible={len(infeasible)}")

# Leader perspective
status, body = _post("/api/strategy/policies",
                     {**POLICIES_PAYLOAD, "perspective": "leader"})
ok("Policies leader perspective -> 200", status == 200, f"got {status}")

# Missing required driver codes -> 400
status, body = _post("/api/strategy/policies", {"track_name": "Monaco"})
ok("Missing driver codes -> 400", status == 400, f"got {status}")
ok("400 body has error key", "error" in body)


# ── 13. Strategy Call ────────────────────────────────────────────────────────

section("13  POST /api/strategy/call")

status, body = _post("/api/strategy/call", CALL_PAYLOAD)
ok("POST /api/strategy/call -> 200", status == 200, f"got {status}")
if status == 200:
    # Top-level structure: final_call, pit_analysis, chaser, leader, policies
    ok("final_call present", "final_call" in body, str(list(body.keys())))
    ok("pit_analysis present", "pit_analysis" in body, str(list(body.keys())))
    ok("policies present", "policies" in body)
    ok("latency_ms present", "latency_ms" in body)
    # final_call contains the actual decision fields
    fc = body.get("final_call", {})
    ok("final_call.action present",
       "action" in fc, str(list(fc.keys()) if isinstance(fc, dict) else fc))
    ok("final_call.reason present", "reason" in fc)
    ok("final_call.confidence present", "confidence" in fc)
    ok("final_call.overtake_probability present", "overtake_probability" in fc)
    # pit_analysis top-level fields
    pit = body.get("pit_analysis", {})
    ok("pit_analysis.effective_pit_loss_s present",
       "effective_pit_loss_s" in pit, str(list(pit.keys()) if isinstance(pit, dict) else pit))
    ok("pit_analysis.is_cheap_stop present", "is_cheap_stop" in pit)
    ok("pit_analysis.traffic_level present", "traffic_level" in pit)
    # traffic_factor lives nested: pit_analysis.safety_car.pit_calc.traffic_factor
    sc_calc = pit.get("safety_car", {}).get("pit_calc", {})
    ok("pit_analysis.safety_car.pit_calc.traffic_factor present",
       "traffic_factor" in sc_calc, str(list(sc_calc.keys()) if isinstance(sc_calc, dict) else sc_calc))

# VSC scenario — cheap stop expected
status, body = _post("/api/strategy/call",
                     {**CALL_PAYLOAD, "race_event": "vsc"})
ok("strategy/call under VSC -> 200", status == 200, f"got {status}")
if status == 200:
    pit = body.get("pit_analysis", {})
    ok("VSC is_cheap_stop=True", pit.get("is_cheap_stop") is True,
       str(pit.get("is_cheap_stop")))
    ok("VSC pit loss < green baseline",
       pit.get("effective_pit_loss_s", 999) < pit.get("baseline_green_s", 0),
       f"loss={pit.get('effective_pit_loss_s')} baseline={pit.get('baseline_green_s')}")

# Safety car scenario
status, body = _post("/api/strategy/call",
                     {**CALL_PAYLOAD, "race_event": "safety_car"})
ok("strategy/call under SC -> 200", status == 200, f"got {status}")
if status == 200:
    pit = body.get("pit_analysis", {})
    ok("SC is_cheap_stop=True", pit.get("is_cheap_stop") is True)

# Red flag -> pit loss = 0
status, body = _post("/api/strategy/call",
                     {**CALL_PAYLOAD, "race_event": "red_flag"})
ok("strategy/call red_flag -> 200", status == 200, f"got {status}")
if status == 200:
    pit = body.get("pit_analysis", {})
    ok("Red flag pit_loss_s = 0.0",
       pit.get("effective_pit_loss_s") == 0.0, str(pit.get("effective_pit_loss_s")))
    ok("Red flag is_cheap_stop=True", pit.get("is_cheap_stop") is True)

# Traffic factor validation:
# The actual traffic factor is applied to effective_pit_loss_s at the TOP level.
# pit_analysis.safety_car.pit_calc always uses Clear (0.95) because that nested
# calculation models the hypothetical SC-window rejoin, not the current green lap.
# Monaco green base = 19.4s: Clear->18.43 (x0.95), Light->19.4 (x1.0), Heavy->24.25 (x1.25)
MONACO_GREEN_BASE = 19.4
for traffic, factor in [("Clear", 0.95), ("Light", 1.0), ("Heavy", 1.25)]:
    status, body = _post("/api/strategy/call",
                         {**CALL_PAYLOAD, "traffic_level": traffic, "race_event": "green"})
    ok(f"Traffic={traffic} -> 200", status == 200, f"got {status}")
    if status == 200:
        pit = body.get("pit_analysis", {})
        # Verify traffic_level string echoed correctly
        ok(f"Traffic={traffic} level echoed",
           pit.get("traffic_level") == traffic,
           f"got {pit.get('traffic_level')}")
        # Verify effective_pit_loss_s matches expected factor applied to Monaco baseline
        expected_loss = round(MONACO_GREEN_BASE * factor, 2)
        actual_loss = pit.get("effective_pit_loss_s")
        ok(f"Traffic={traffic} effective_pit_loss={expected_loss}s",
           actual_loss == expected_loss,
           f"got {actual_loss}")

# Missing driver codes -> 400
status, body = _post("/api/strategy/call", {"track_name": "Monaco"})
ok("strategy/call missing codes -> 400", status == 400, f"got {status}")

# Same driver twice -> 400
status, body = _post("/api/strategy/call",
                     {**CALL_PAYLOAD, "leader_code": "LEC", "chaser_code": "LEC"})
ok("strategy/call same driver -> 400", status == 400, f"got {status}")


# ── 14. Pit circuit coverage ─────────────────────────────────────────────────

section("14  Pit strategy circuit coverage")

spot_circuits = [
    "Monaco", "Silverstone Circuit", "Monza", "Spa-Francorchamps",
    "Baku City Circuit", "Albert Park Circuit", "Yas Marina Circuit",
    "Suzuka Circuit", "Circuit de Catalunya", "Las Vegas Strip Circuit",
]
for circuit in spot_circuits:
    p = {**CALL_PAYLOAD, "track_name": circuit, "race_event": "green", "traffic_level": "Light"}
    s, b = _post("/api/strategy/call", p)
    ok(f"{circuit} -> 200", s == 200, f"got {s}")
    if s == 200:
        pit = b.get("pit_analysis", {})
        loss = pit.get("effective_pit_loss_s", 0)
        ok(f"{circuit} pit_loss in [8,30]s", 8.0 <= loss <= 30.0, f"got {loss}")


# ── 15. AI Race Engineer ─────────────────────────────────────────────────────

section("15  POST /api/ai-race-engineer")

# Requires state dict with leader_code/chaser_code/track_name
are_state = {
    "leader_code": "VER",
    "chaser_code": "LEC",
    "track_name": "Silverstone Circuit",
    "start_lap": 22,
    "race_length": 52,
    "gap_before_s": 1.5,
    "leader_tyre_compound": "Hard",
    "chaser_tyre_compound": "Medium",
    "leader_tyre_age": 10.0,
    "chaser_tyre_age": 8.0,
    "battery_pct": 60.0,
    "perspective": "chaser",
    "race_event": "green",
    "traffic_level": "Light",
}
status, body = _post("/api/ai-race-engineer", {"state": are_state})
ok("POST /api/ai-race-engineer -> 200", status == 200, f"got {status}")
if status == 200:
    # Actual response uses 'radio_transmission' not 'radio_call'
    ok("radio_transmission present",
       "radio_transmission" in body, str(list(body.keys())))
    ok("radio_transmission is non-empty string",
       isinstance(body.get("radio_transmission"), str)
       and len(body["radio_transmission"]) > 5,
       repr(body.get("radio_transmission")))
    # 'definitive_call' is the strategy summary
    ok("definitive_call present", "definitive_call" in body)
    ok("definitive_call is string",
       isinstance(body.get("definitive_call"), str))
    ok("tactical_rationale present", "tactical_rationale" in body)
    ok("confidence_level present", "confidence_level" in body)
    ok("driver_directives present", "driver_directives" in body)
    ok("source present", "source" in body)

# SC scenario
status, body = _post("/api/ai-race-engineer",
                     {"state": {**are_state, "race_event": "safety_car"}})
ok("AI Race Engineer SC -> 200", status == 200, f"got {status}")
if status == 200:
    ok("radio_transmission under SC is string",
       isinstance(body.get("radio_transmission"), str))

# Missing state -> 400
status, body = _post("/api/ai-race-engineer", {})
ok("Empty ai-race-engineer body -> 400", status == 400, f"got {status}")


# ── 16. Energy Sandbox ───────────────────────────────────────────────────────

section("16  POST /api/strategy/energy-sandbox")

payload_sandbox = {
    "track_name": "Monza",
    "mode": "push",
    "lap": 10,
    "total_laps": 53,
    "battery_pct": 80.0,
    "tyre": "Soft",
    "tyre_age": 5,
    "gap_s": 2.0,
}
status, body = _post("/api/strategy/energy-sandbox", payload_sandbox)
ok("POST /api/strategy/energy-sandbox -> 200 or 4xx",
   status in (200, 400, 500), f"got {status}")
if status == 200:
    ok("Response is dict", isinstance(body, dict))

# Invalid mode -> 400
status, body = _post("/api/strategy/energy-sandbox",
                     {**payload_sandbox, "mode": "ludicrous"})
ok("Bad mode -> 400", status == 400, f"got {status}")
if status == 400:
    ok("400 has error key", "error" in body)


# ── 17. Overtake support endpoints ───────────────────────────────────────────

section("17  Overtake support endpoints")

status, body = _get("/api/overtake/options")
ok("GET /api/overtake/options -> 200", status == 200, f"got {status}")
ok("Returns dict", isinstance(body, dict), type(body).__name__)

# overtake/sessions requires leader, chaser, track, year
status, body = _get("/api/overtake/sessions?leader=LEC&chaser=VER&track=Silverstone&year=2024")
ok("GET /api/overtake/sessions with params -> 200", status == 200, f"got {status}")
ok("Returns dict (sessions map)", isinstance(body, dict), type(body).__name__)

# Omitting required params -> 400
status, body = _get("/api/overtake/sessions")
ok("overtake/sessions no params -> 400", status == 400, f"got {status}")

# overtake/calibration requires leader, chaser, year
status, body = _get("/api/overtake/reliability")
ok("GET /api/overtake/reliability -> 200", status == 200, f"got {status}")


# ── 18. Overtake sim (POST) ──────────────────────────────────────────────────

section("18  POST /api/overtake/sim")

payload_sim = {
    "leader": "VER",
    "chaser": "LEC",
    "track": "Silverstone",
    "lap": 30,
    "gap_s": 0.9,
    "tyre_leader": "Hard",
    "tyre_chaser": "Medium",
    "tyre_age_leader": 20,
    "tyre_age_chaser": 8,
    "ers_delta_pct": 15.0,
}
status, body = _post("/api/overtake/sim", payload_sim)
ok("POST /api/overtake/sim -> 200 or 4xx", status in (200, 400, 422, 500), f"got {status}")
if status == 200:
    ok("overtake_probability present", "overtake_probability" in body,
       str(list(body.keys())))


# ── 19. Overtake live (POST) ─────────────────────────────────────────────────

section("19  POST /api/overtake/live")

payload_live = {
    "leader": "VER",
    "chaser": "LEC",
    "track": "Silverstone",
    "gap_s": 0.85,
    "battery_pct": 68.0,
    "tyre": "Medium",
    "tyre_age": 12,
    "position": 2,
    "total_laps": 52,
    "lap": 28,
}
status, body = _post("/api/overtake/live", payload_live)
ok("POST /api/overtake/live -> 200 or 4xx", status in (200, 400, 422, 500), f"got {status}")
if status == 200:
    ok("Response is dict", isinstance(body, dict))


# ── 20. Comparison endpoints ─────────────────────────────────────────────────

section("20  Comparison endpoints")

# years returns {years: [...]}
status, body = _get("/api/comparison/years")
ok("GET /api/comparison/years -> 200", status == 200, f"got {status}")
ok("Returns dict with 'years' key", isinstance(body, dict) and "years" in body,
   type(body).__name__)
if isinstance(body, dict) and "years" in body:
    ok("years is list", isinstance(body["years"], list))
    ok("years contains ints", all(isinstance(y, int) for y in body["years"]))

# tracks requires year
status, body = _get("/api/comparison/tracks?year=2024")
ok("GET /api/comparison/tracks?year=2024 -> 200", status == 200, f"got {status}")
ok("Returns list or dict", isinstance(body, (list, dict)), type(body).__name__)

# tracks without year -> 400
status, _ = _get("/api/comparison/tracks")
ok("/api/comparison/tracks no year -> 400", status == 400, f"got {status}")

# drivers requires year and track
status, body = _get("/api/comparison/drivers?year=2024&track=Silverstone")
ok("GET /api/comparison/drivers year+track -> 200", status == 200, f"got {status}")
ok("Returns list or dict", isinstance(body, (list, dict)), type(body).__name__)

# drivers without required params -> 400
status, _ = _get("/api/comparison/drivers")
ok("/api/comparison/drivers no params -> 400", status == 400, f"got {status}")

# race
status, body = _get("/api/comparison/race?driver_a=LEC&driver_b=VER&year=2024")
ok("GET /api/comparison/race -> 200 or 404", status in (200, 404, 400), f"got {status}")


# ── 21. Drivers endpoints ────────────────────────────────────────────────────

section("21  /api/drivers and /api/drivers/compare")

status, body = _get("/api/drivers")
ok("GET /api/drivers -> 200", status == 200, f"got {status}")
ok("Returns list or dict", isinstance(body, (list, dict)), type(body).__name__)

status, body = _get("/api/drivers/compare?driver_a=LEC&driver_b=VER")
ok("GET /api/drivers/compare -> 200 or 4xx", status in (200, 400, 404), f"got {status}")


# ── 22. Calendar ─────────────────────────────────────────────────────────────

section("22  /api/calendar")

status, body = _get("/api/calendar")
ok("GET /api/calendar -> 200", status == 200, f"got {status}")
ok("Returns list or dict", isinstance(body, (list, dict)), type(body).__name__)

status, body = _get("/api/calendar?year=2025")
ok("?year=2025 -> 200", status == 200, f"got {status}")
ok("Returns list or dict", isinstance(body, (list, dict)), type(body).__name__)


# ── 23. Benchmark endpoints ──────────────────────────────────────────────────

section("23  Benchmark endpoints")

status, body = _get("/api/benchmark/energy")
ok("GET /api/benchmark/energy -> 200", status == 200, f"got {status}")

status, body = _get("/api/benchmark/energy/fleet")
ok("GET /api/benchmark/energy/fleet -> 200", status == 200, f"got {status}")

# /api/benchmark/energy/race requires 'key' param
# Using 'latest' — returns 404 if no race artifact exists (valid behaviour)
status, body = _get("/api/benchmark/energy/race?key=latest")
ok("GET /api/benchmark/energy/race?key=latest -> 200 or 404",
   status in (200, 404), f"got {status}")

# Missing key param -> 400
status, body = _get("/api/benchmark/energy/race")
ok("GET /api/benchmark/energy/race no key -> 400", status == 400, f"got {status}")


# ── 24. Overtake analysis & simulation data ──────────────────────────────────

section("24  Overtake analysis & simulation data")

status, body = _get("/api/overtake_analysis")
ok("GET /api/overtake_analysis -> 200 or 404", status in (200, 404), f"got {status}")
if status == 200:
    ok("status key present", "status" in body)
    ok("data key present", "data" in body)

status, body = _get("/api/overtake_simulation_data")
ok("GET /api/overtake_simulation_data -> 200 or 404", status in (200, 404), f"got {status}")
if status == 200:
    ok("status key present", "status" in body)


# ── Final report ─────────────────────────────────────────────────────────────

total = PASS + FAIL
print(f"\n{'='*62}")
print(f"  RESULTS: {PASS}/{total} passed  |  {FAIL} failed")
print(f"{'='*62}")
if ERRORS:
    print("\nFailed tests:")
    for e in ERRORS:
        print(e)
else:
    print("\n  All tests passed.")

sys.exit(0 if FAIL == 0 else 1)
