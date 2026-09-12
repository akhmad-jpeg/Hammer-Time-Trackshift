"""AI Race Engineer — Real-time Pit Wall Strategy Transceiver.

Synthesizes race state, tyre degradation, battery State-of-Charge (SOC),
opponents' defensive posture, and candidate policy evaluations from
the Multi-Policy Decision Engine, formatting a comprehensive tactical
briefing for an LLM (OpenAI, Anthropic Claude, Google Gemini, OpenRouter,
or local endpoint) via user-provided API key.

When an API key is absent or on hold, provides a high-fidelity deterministic
tactical briefing matching the exact decision engine outputs so the UI is
always fully functional.
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import urllib.request
import urllib.error

# Project roots and model metadata
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "ml_models"


def load_model_telemetry_context(track_name: str = "") -> Dict[str, Any]:
    """Extract grounding telemetry and calibration data from trained artifacts."""
    ctx: Dict[str, Any] = {
        "fuel_burn_rate_s_per_lap": -0.1355,
        "compound_wear_rates": {
            "Soft": 0.0403,
            "Medium": 0.0190,
            "Hard": 0.0309,
            "Intermediate": 0.3460,
            "Wet": 0.4382,
        },
        "overtake_model_info": {},
        "track_energy_profile": {},
    }

    # Overtake model info & isotonic calibration
    ot_info_path = MODELS_DIR / "overtake" / "model_info.json"
    if ot_info_path.exists():
        try:
            with open(ot_info_path, "r", encoding="utf-8") as f:
                info = json.load(f)
                ctx["overtake_model_info"] = {
                    "classifier": info.get("best_overtake_classifier", "LogisticRegression"),
                    "roc_auc": info.get("roc_auc", 0.74),
                    "brier_raw": info.get("isotonic_calibration", {}).get("brier_raw"),
                    "brier_calibrated": info.get("isotonic_calibration", {}).get("brier_calibrated"),
                    "ece_calibrated": info.get("isotonic_calibration", {}).get("ece_calibrated"),
                    "calibrator_status": "Isotonic Monotonic Regressor (Held-out Test Split)",
                }
        except Exception:
            pass

    # Track energy sensitivity
    ep_path = MODELS_DIR / "energy_pace.json"
    if ep_path.exists():
        try:
            with open(ep_path, "r", encoding="utf-8") as f:
                ep = json.load(f)
                tracks = ep.get("per_track", {})
                for k, v in tracks.items():
                    if k.lower() in track_name.lower() or track_name.lower() in k.lower():
                        ctx["track_energy_profile"] = {
                            "track_name": k,
                            "measured_s_per_mj": v.get("s_per_mj", 0.35),
                            "full_throttle_share": v.get("full_throttle_share", 0.65),
                            "sector_pace_s_per_mj": v.get("sector_s_per_mj", [0.4, 0.3, 0.4]),
                        }
                        break
        except Exception:
            pass

    return ctx


def build_system_prompt() -> str:
    """Build the F1 Chief Race Strategist system persona."""
    return (
        "You are the Senior Chief Race Strategist and Race Engineer on an F1 pit wall "
        "(analogous to Peter Bonnington or Gianpiero Lambiase). You speak directly to the "
        "driver over pit-to-car radio and provide definitive, authoritative tactical commands.\n\n"
        "RULES OF ENGAGEMENT:\n"
        "1. Voice & Tone: Decisive, concise, urgent yet calm, professional motorsport radio style. "
        "Use authentic terminology: 'Mode Attack', 'Strat 5', 'Clipping', 'Lift and Coast', "
        "'Tyre cliff', 'Dirty air penalty', 'Undercut window', 'Delta negative'.\n"
        "2. Ground Truth Adherence: The mathematical Decision Engine recommendation and policy scores "
        "are ground truth. You must never invent numbers or contradict the policy feasibility. If a policy "
        "is INFEASIBLE (e.g., drains battery to the 10% management reserve floor), explain why it is rejected.\n"
        "3. Honesty & Uncertainty: The battery State of Charge is a synthesized estimate with a ±2-8% confidence "
        "envelope. Account for battery opportunity cost: deploying electrical boost now may win a position "
        "in Turn 3 but leaves the car defenseless 2 laps later if the leader counters.\n"
        "4. Output format: You MUST return a valid JSON object matching the requested schema exactly, with NO markdown ticks or preamble."
    )


def build_user_prompt(
    state: Dict[str, Any],
    call_result: Dict[str, Any],
    perspective: str = "chaser",
    online_context: Optional[str] = None,
) -> str:
    """Package the full race telemetry, ML models, and policy evaluations into an LLM prompt."""
    track_name = state.get("track_name", "Circuit")
    grounding = load_model_telemetry_context(track_name)

    rec = call_result.get("final_call", {}).get("action_card", {})
    perspective_label = "CHASER (WE ARE ATTACKING)" if perspective == "chaser" else "LEADER (WE ARE DEFENDING)"

    # Compile candidate policies table
    engine_data = call_result.get(perspective, {})
    policies = engine_data.get("policies", [])
    policy_summary = []
    for p in policies:
        policy_summary.append({
            "action": p.get("action"),
            "feasible": p.get("feasible"),
            "infeasible_reason": p.get("infeasible_reason"),
            "score_s": p.get("score_s"),
            "expected_finish_delta_s": p.get("projected_finish_delta_s"),
            "overtake_prob_calibrated": p.get("calibrated_probability") or p.get("overtake_probability"),
            "raw_probability": p.get("raw_probability"),
            "projected_pass_lap": p.get("projected_pass_lap"),
            "battery_margin_worst_pct": p.get("battery_margin_worst_pct"),
            "soc_band_pct": p.get("soc_band_pct"),
            "wear_penalty_s": p.get("wear_penalty_s"),
        })

    prompt_data = {
        "perspective": perspective_label,
        "race_context": {
            "track": track_name,
            "season": state.get("year", 2026),
            "current_lap": state.get("start_lap", 20),
            "total_laps": state.get("race_length", 57),
            "laps_remaining": max(0, int(state.get("race_length", 57)) - int(state.get("start_lap", 20))),
            "gap_to_car_ahead_s": state.get("gap_before_s", 0.8),
        },
        "drivers_and_tyres": {
            "leader": {
                "code": state.get("leader_code", "RUS"),
                "compound": state.get("leader_tyre_compound", "Medium"),
                "tyre_age_laps": state.get("leader_tyre_age", 12),
            },
            "chaser": {
                "code": state.get("chaser_code", "LEC"),
                "compound": state.get("chaser_tyre_compound", "Medium"),
                "tyre_age_laps": state.get("chaser_tyre_age", 8),
            },
        },
        "energy_and_battery": {
            "current_battery_pct": state.get("battery_pct", 62.5),
            "battery_uncertainty_band_pct": state.get("battery_band_pct", 4.0),
            "management_reserve_floor_pct": 10.0,
            "management_reserve_floor_mj": 0.4,
            "reserve_target_pct": state.get("reserve_target_pct", 40.0),
            "opponent_defense_posture": call_result.get("coupling", {}).get("chaser_posture", "balanced"),
        },
        "safety_car_and_pit_strategy": {
            "race_event": call_result.get("pit_analysis", {}).get("race_event", state.get("race_event", "green")),
            "traffic_level": call_result.get("pit_analysis", {}).get("traffic_level", state.get("traffic_level", "Clear")),
            "effective_pit_loss_s": call_result.get("pit_analysis", {}).get("effective_pit_loss_s"),
            "baseline_green_s": call_result.get("pit_analysis", {}).get("baseline_green_s"),
            "time_saved_s": call_result.get("pit_analysis", {}).get("time_saved_s"),
            "is_cheap_stop": call_result.get("pit_analysis", {}).get("is_cheap_stop", False),
            "sc_probability": call_result.get("pit_analysis", {}).get("sc_probability"),
            "sc_risk_tier": call_result.get("pit_analysis", {}).get("sc_risk_tier"),
            "undercut": call_result.get("pit_analysis", {}).get("undercut"),
            "safety_car_eval": call_result.get("pit_analysis", {}).get("safety_car"),
        },
        "ml_grounding_evidence": grounding,
        "evaluated_tactical_policies": policy_summary,
        "mathematical_optimizer_recommendation": rec,
    }

    if online_context:
        prompt_data["online_and_track_intelligence"] = online_context

    return (
        f"ANALYZE THIS LIVE RACE SITUATION AND ISSUE THE DEFINITIVE CALL:\n\n"
        f"{json.dumps(prompt_data, indent=2)}\n\n"
        "Respond in strict JSON with the following structure:\n"
        "{\n"
        '  "definitive_call": "Short punchy command (e.g. TACTICAL STALK — DEPLOY LAP 21)",\n'
        '  "radio_transmission": "Direct quote from race engineer to driver over team radio",\n'
        '  "confidence_level": "HIGH / MEDIUM / MARGINAL",\n'
        '  "confidence_score_pct": 85,\n'
        '  "tactical_rationale": "2-3 paragraphs detailing why this policy wins, energy opportunity cost, opponent defense response, and tyre delta.",\n'
        '  "telemetry_proof": [\n'
        '    "Bullet 1: Battery state and reserve safety margin",\n'
        '    "Bullet 2: Tyre compound speed delta and degradation cliff",\n'
        '    "Bullet 3: Overtake ML probability and isotonic calibration honesty",\n'
        '    "Bullet 4: Expected race time delta in seconds"\n'
        '  ],\n'
        '  "driver_directives": [\n'
        '    "Turn-by-turn instruction 1",\n'
        '    "Turn-by-turn instruction 2"\n'
        '  ],\n'
        '  "contingency_protocol": "What to do if leader defends with boost or safety car deploys"\n'
        "}"
    )


def generate_deterministic_call(
    state: Dict[str, Any],
    call_result: Dict[str, Any],
    perspective: str = "chaser",
) -> Dict[str, Any]:
    """High-fidelity deterministic fallback when no LLM API key is provided."""
    rec = call_result.get("final_call", {}).get("action_card", {})
    action = rec.get("action", "BALANCED HOLD")
    leader = state.get("leader_code", "RUS")
    chaser = state.get("chaser_code", "LEC")
    driver = chaser if perspective == "chaser" else leader
    opponent = leader if perspective == "chaser" else chaser
    gap = state.get("gap_before_s", 0.8)
    delta_s = rec.get("projected_finish_delta_s", -1.2)
    delta_str = f"{delta_s:+.2f}s" if delta_s is not None else "-0.8s"
    rec_lap = rec.get("recommended_lap") or (int(state.get("start_lap", 20)) + 1)
    batt_margin = rec.get("battery_margin_pct", 24.5)
    conf = rec.get("confidence_pct", 82)

    pit_analysis = call_result.get("pit_analysis") or {}
    race_event = str(pit_analysis.get("race_event") or state.get("race_event") or "green").lower()
    time_saved = float(pit_analysis.get("time_saved_s") or 0.0)
    is_cheap_stop = pit_analysis.get("is_cheap_stop", False) or race_event in ("vsc", "safety_car")
    undercut_info = pit_analysis.get("undercut") or {}

    if is_cheap_stop:
        event_label = "VSC" if race_event == "vsc" else "SAFETY CAR"
        radio = (
            f"Box, box, box under {event_label}, {driver}! Exploiting cheap stop saving ~{time_saved:.1f}s "
            f"on transit time. Box now, confirm tyres."
        )
        rationale = (
            f"Under {event_label}, on-track delta speeds are capped while pit transit remains at speed limit, "
            f"slashing net pit loss from {pit_analysis.get('baseline_green_s', 21.0):.1f}s down to "
            f"{pit_analysis.get('effective_pit_loss_s', 11.5):.1f}s (saving ~{time_saved:.1f}s of race time). "
            f"Pitting now secures cheap track position without racing on degrading rubber."
        )
    elif "UNDERCUT" in action or undercut_info.get("status") in ("OPEN_FAVORABLE", "MARGINAL"):
        outlap_gain = undercut_info.get("fresh_tyre_outlap_gain_s", 1.8)
        exit_margin = undercut_info.get("net_exit_margin_s", 0.5)
        radio = (
            f"{driver}, prepare for the undercut. Gap is {gap:.1f}s. Fresh tyre out-lap delta gives "
            f"+{outlap_gain:.1f}s advantage. Push on in-lap, box next lap."
        )
        rationale = (
            f"The undercut window is {undercut_info.get('status', 'OPEN')}. A fresh set of rubber provides a "
            f"+{outlap_gain:.1f}s out-lap pace delta against {opponent}'s degrading tyres, yielding a projected "
            f"+{exit_margin:.1f}s buffer on pit exit to jump track position."
        )
    elif perspective == "chaser":
        if "ATTACK" in action:
            radio = (
                f"Okay {driver}, Hammer Time. Mode Attack now. Gap is {gap:.1f}s. "
                f"We have the tyre advantage and deploy delta. Strike into Turn 3 on Lap {rec_lap}."
            )
            rationale = (
                f"The Multi-Policy Decision Engine evaluates an immediate push as mathematically optimal. "
                f"{driver} holds fresh rubber against {opponent}, producing a closing rate sufficient to "
                f"convert the pass on Lap {rec_lap} while preserving a +{batt_margin:.1f}% battery reserve margin "
                f"above the 10% management safety floor."
            )
        elif "STALK" in action:
            radio = (
                f"Okay {driver}, settle in. Mode Stalk. Lift and coast through Sector 2, bank electrical energy. "
                f"{opponent} is burning rear tyres in dirty air. We deploy full boost on Lap {rec_lap}."
            )
            rationale = (
                f"Immediate greedy attack is rejected due to energy opportunity cost: deploying now exhausts the battery "
                f"to its management floor, creating defensive vulnerability later. Tactical Stalk banks energy in "
                f"low-overtake sectors, waiting for {opponent}'s tyres to enter the thermal wear window, executing on Lap {rec_lap} "
                f"with an expected finish delta of {delta_str}."
            )
        elif "SAVE" in action or "DEFEND" in action:
            radio = (
                f"{driver}, manage tyres and recharge. Battery reserve is constrained. Protect the store, "
                f"do not lunge unless {opponent} makes an unforced error."
            )
            rationale = (
                f"Energy stores are near the critical 10% reserve constraint. Attacking now would cause an energy-limited "
                f"derate on the subsequent straight. Holding preserves optimal race finish time ({delta_str})."
            )
        else:
            radio = f"Hold delta to {opponent} at {gap:.1f}s, {driver}. Maintain pace window and monitor front tyre temps."
            rationale = f"Pace gap is stable. Maintaining balanced deployment yields the lowest race time degradation."
    else:
        # Leader perspective
        if "DEFEND" in action or "PUSH" in action or "COUNTER" in action:
            radio = (
                f"{driver}, {opponent} is within {gap:.1f}s entering the detection zone. "
                f"Mode Defend on the main straight, deploy K-boost out of the final corner to break DRS."
            )
            rationale = (
                f"Chaser {opponent} is closing. Deploying targeted defensive electrical burst on the exit straight "
                f"suppresses closing velocity while preserving sufficient battery above the reserve target."
            )
        else:
            radio = f"Clean air ahead, {driver}. Keep tyres in the operating window, manage energy through Sector 2."
            rationale = f"Delta to trailing car is sustainable. Conserving battery protects against an undercut threat."

    proofs = [
        f"Battery State: {state.get('battery_pct', 62.5):.1f}% ±{state.get('battery_band_pct', 4.0):.1f}% (Post-Pass Margin: +{batt_margin:.1f}%)",
        f"Tyre Delta: {driver} ({state.get('chaser_tyre_compound' if perspective=='chaser' else 'leader_tyre_compound', 'Medium')}) vs {opponent} ({state.get('leader_tyre_compound' if perspective=='chaser' else 'chaser_tyre_compound', 'Medium')})",
        f"Overtake Probability: {rec.get('calibrated_probability', 0.76)*100:.1f}% (Isotonic Monotonic Calibration)",
        f"Projected Finish Time Delta: {delta_str} vs neutral baseline",
    ]
    if pit_analysis.get("effective_pit_loss_s") is not None:
        proofs.append(
            f"Pit Delta: {pit_analysis.get('effective_pit_loss_s'):.1f}s ({race_event.upper()}, "
            f"Saved: {time_saved:.1f}s, SC Rate: {pit_analysis.get('sc_probability', 0.5)*100:.0f}%)"
        )

    directives = [
        f"Maintain disciplined gap of {gap:.1f}s through Sector 1 chicane.",
        f"Activate high-harvest engine braking in Sector 2 hairpins.",
        f"Unleash maximum deploy boost in primary DRS acceleration zone.",
    ]
    if is_cheap_stop or "UNDERCUT" in action:
        directives.insert(0, "BOX THIS LAP: Pit speed limit 80 km/h; fresh compound standing by.")

    return {
        "definitive_call": f"{action} — {driver} (LAP {rec_lap})",
        "radio_transmission": radio,
        "confidence_level": "HIGH" if conf >= 75 else "MEDIUM",
        "confidence_score_pct": conf,
        "tactical_rationale": rationale,
        "telemetry_proof": proofs,
        "driver_directives": directives,
        "contingency_protocol": (
            f"If {opponent} counters with sudden defensive deployment, immediately abort lunge, "
            f"slot into slipstream, bank 1.0 MJ, and re-arm for the next lap."
        ),
        "source": "deterministic_engine_fallback",
        "note": "Deterministic Racecraft Engine output. Connect your LLM API Key in Settings for live neural pit-wall synthesis.",
    }


def call_openai_compatible(
    api_key: str,
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> Dict[str, Any]:
    """Call OpenAI or OpenAI-compatible API (OpenRouter, Groq, DeepSeek, vLLM)."""
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.25,
        "response_format": {"type": "json_object"},
    }

    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=28) as resp:
        res_json = json.loads(resp.read().decode("utf-8"))
        content = res_json["choices"][0]["message"]["content"]
        return json.loads(content)


def call_anthropic(
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> Dict[str, Any]:
    """Call Anthropic Claude API."""
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    payload = {
        "model": model or "claude-3-5-sonnet-20241022",
        "system": system_prompt,
        "messages": [
            {"role": "user", "content": user_prompt + "\n\nOutput only raw JSON matching the requested structure."},
        ],
        "max_tokens": 1500,
        "temperature": 0.25,
    }

    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=28) as resp:
        res_json = json.loads(resp.read().decode("utf-8"))
        text = res_json["content"][0]["text"]
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return json.loads(text)


def call_gemini(
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> Dict[str, Any]:
    """Call Google Gemini API."""
    model_name = model or "gemini-1.5-pro"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"parts": [{"text": user_prompt + "\nOutput strictly valid JSON."}]}],
        "generationConfig": {
            "temperature": 0.25,
            "response_mime_type": "application/json",
        },
    }

    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=28) as resp:
        res_json = json.loads(resp.read().decode("utf-8"))
        text = res_json["candidates"][0]["content"]["parts"][0]["text"]
        return json.loads(text)


def consult_ai_race_engineer(
    state: Dict[str, Any],
    call_result: Dict[str, Any],
    api_key: Optional[str] = None,
    provider: str = "auto",
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    online_context: Optional[str] = None,
) -> Dict[str, Any]:
    """Main entry point: generates Chief Race Strategist assessment."""
    perspective = str(state.get("perspective") or "chaser").lower()

    # Resolve API Key from argument or environment variables
    key = api_key or os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("GEMINI_API_KEY")

    if not key or str(key).strip().lower() in ("none", "", "null", "undefined"):
        fallback = generate_deterministic_call(state, call_result, perspective=perspective)
        fallback["mode"] = "deterministic_offline"
        fallback["api_key_configured"] = False
        return fallback

    key = str(key).strip()
    sys_prompt = build_system_prompt()
    user_prompt = build_user_prompt(state, call_result, perspective=perspective, online_context=online_context)

    # Auto-detect provider if needed
    p = provider.lower() if provider else "auto"
    if p == "auto":
        if key.startswith("sk-ant-"):
            p = "anthropic"
        elif key.startswith("AIza"):
            p = "gemini"
        else:
            p = "openai"

    t0 = time.perf_counter()
    try:
        if p == "anthropic":
            m = model or "claude-3-5-sonnet-20241022"
            result = call_anthropic(key, m, sys_prompt, user_prompt)
        elif p == "gemini":
            m = model or "gemini-1.5-pro"
            result = call_gemini(key, m, sys_prompt, user_prompt)
        else:  # openai, openrouter, custom
            b_url = base_url or ("https://openrouter.ai/api/v1" if "openrouter" in p else "https://api.openai.com/v1")
            m = model or ("meta-llama/llama-3.3-70b-instruct" if "openrouter" in p else "gpt-4o")
            result = call_openai_compatible(key, b_url, m, sys_prompt, user_prompt)

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        result["latency_ms"] = elapsed_ms
        result["provider"] = p
        result["model"] = m
        result["source"] = "llm_live_telemetry_synthesis"
        result["api_key_configured"] = True
        return result

    except Exception as e:
        fallback = generate_deterministic_call(state, call_result, perspective=perspective)
        fallback["mode"] = "fallback_on_api_error"
        fallback["api_key_configured"] = True
        fallback["api_error"] = str(e)
        fallback["note"] = f"LLM API call failed ({str(e)}). Displaying verified deterministic strategy call."
        return fallback
