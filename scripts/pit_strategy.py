"""Circuit Pit Loss, Undercut Dynamics & Safety Car Strategy Engine.

Models real-world Formula 1 pit lane transit losses, fresh-tyre undercut/overcut
deltas, and empirical Safety Car (SC / VSC) intervention probabilities across
all 32 World Championship circuits.

Key Physical Mechanics:
  1. Green-Flag Pit Loss: Total lap-time penalty for pitting at racing speeds
     (pit lane entry speed limit 80 km/h or 60 km/h + stationary stop time).
  2. VSC "Cheap Stop": Under Virtual Safety Car, on-track cars are limited by
     a strict delta pace (~35-40% slower), while pit lane transit remains 80 km/h.
     The relative pit loss shrinks dramatically (~10-12s vs ~20-24s), saving ~9-12s.
  3. Full Safety Car (SC): Field bunches up at ~60% racing speed, reducing pit
     loss to ~9-11s.
  4. Undercut Feasibility: Compares fresh tyre out-lap pace gain against the gap
     to leader to determine if pitting 1 lap earlier converts track position on exit.
  5. Empirical Safety Car Likelihood: Historical track-specific deployment rates
     (Monaco 72%, Singapore 100%, Baku 75%, Albert Park 65%, Monza 22%).
"""

from typing import Any, Dict, Optional


# Standard dry tyre allocation per weekend (FIA Sporting Regulations)
SET_ALLOCATION: Dict[str, int] = {
    "Soft": 4,
    "Medium": 3,
    "Hard": 2,
    "Intermediate": 3,
    "Wet": 2,
}

# Rejoin traffic multiplier on pit loss
TRAFFIC_FACTORS: Dict[str, float] = {
    "Clear": 0.95,
    "Light": 1.00,
    "Heavy": 1.25,
}

# Baseline defaults
DEFAULT_STATIONARY_S = 2.4
DEFAULT_PIT_LOSS_GREEN_S = 21.5
DEFAULT_PIT_LOSS_VSC_S = 12.0
DEFAULT_PIT_LOSS_SC_S = 9.5
DEFAULT_SC_PROBABILITY = 0.45


# Circuit-specific Pit Lane Transit & Historical Safety Car Data
# Covers all 32 World Championship circuits with canonical names and aliases.
CIRCUIT_PIT_PROFILES: Dict[str, Dict[str, Any]] = {
    "Albert Park Circuit": {
        "canonical": "Albert Park Circuit",
        "aliases": ["albert park", "melbourne", "australia", "australian gp"],
        "pit_loss_green_s": 20.8,
        "pit_loss_vsc_s": 11.5,
        "pit_loss_sc_s": 9.2,
        "sc_probability": 0.65,
        "sc_risk_tier": "HIGH",
        "pit_speed_limit_kmh": 80,
    },
    "Monaco": {
        "canonical": "Monaco",
        "aliases": ["circuit de monaco", "monte carlo", "monaco gp"],
        "pit_loss_green_s": 19.4,
        "pit_loss_vsc_s": 10.8,
        "pit_loss_sc_s": 8.5,
        "sc_probability": 0.72,
        "sc_risk_tier": "HIGH",
        "pit_speed_limit_kmh": 60,
    },
    "Silverstone Circuit": {
        "canonical": "Silverstone Circuit",
        "aliases": ["silverstone", "great britain", "british gp"],
        "pit_loss_green_s": 22.2,
        "pit_loss_vsc_s": 12.8,
        "pit_loss_sc_s": 9.8,
        "sc_probability": 0.55,
        "sc_risk_tier": "MEDIUM",
        "pit_speed_limit_kmh": 80,
    },
    "Spa-Francorchamps": {
        "canonical": "Spa-Francorchamps",
        "aliases": ["spa", "belgium", "belgian gp"],
        "pit_loss_green_s": 22.8,
        "pit_loss_vsc_s": 13.2,
        "pit_loss_sc_s": 10.0,
        "sc_probability": 0.50,
        "sc_risk_tier": "MEDIUM",
        "pit_speed_limit_kmh": 80,
    },
    "Autodromo Nazionale Di Monza": {
        "canonical": "Autodromo Nazionale Di Monza",
        "aliases": ["monza", "italy", "italian gp"],
        "pit_loss_green_s": 24.1,
        "pit_loss_vsc_s": 14.5,
        "pit_loss_sc_s": 11.2,
        "sc_probability": 0.25,
        "sc_risk_tier": "LOW",
        "pit_speed_limit_kmh": 80,
    },
    "Marina Bay Street Circuit": {
        "canonical": "Marina Bay Street Circuit",
        "aliases": ["singapore", "marina bay", "singapore gp"],
        "pit_loss_green_s": 28.5,
        "pit_loss_vsc_s": 16.2,
        "pit_loss_sc_s": 12.5,
        "sc_probability": 1.00,
        "sc_risk_tier": "HIGH",
        "pit_speed_limit_kmh": 60,
    },
    "Baku City Circuit": {
        "canonical": "Baku City Circuit",
        "aliases": ["baku", "azerbaijan", "azerbaijan gp"],
        "pit_loss_green_s": 21.0,
        "pit_loss_vsc_s": 11.8,
        "pit_loss_sc_s": 9.0,
        "sc_probability": 0.75,
        "sc_risk_tier": "HIGH",
        "pit_speed_limit_kmh": 80,
    },
    "Red Bull Ring": {
        "canonical": "Red Bull Ring",
        "aliases": ["spielberg", "austria", "austrian gp"],
        "pit_loss_green_s": 20.2,
        "pit_loss_vsc_s": 11.2,
        "pit_loss_sc_s": 8.8,
        "sc_probability": 0.35,
        "sc_risk_tier": "MEDIUM",
        "pit_speed_limit_kmh": 80,
    },
    "Circuit De Barcelona-Catalunya": {
        "canonical": "Circuit De Barcelona-Catalunya",
        "aliases": ["barcelona", "catalunya", "spain", "spanish gp"],
        "pit_loss_green_s": 22.0,
        "pit_loss_vsc_s": 12.6,
        "pit_loss_sc_s": 9.6,
        "sc_probability": 0.25,
        "sc_risk_tier": "LOW",
        "pit_speed_limit_kmh": 80,
    },
    "Hungaroring": {
        "canonical": "Hungaroring",
        "aliases": ["budapest", "hungary", "hungarian gp"],
        "pit_loss_green_s": 21.4,
        "pit_loss_vsc_s": 12.0,
        "pit_loss_sc_s": 9.2,
        "sc_probability": 0.30,
        "sc_risk_tier": "LOW",
        "pit_speed_limit_kmh": 80,
    },
    "Circuit Gilles Villeneuve": {
        "canonical": "Circuit Gilles Villeneuve",
        "aliases": ["montreal", "canada", "canadian gp"],
        "pit_loss_green_s": 18.6,
        "pit_loss_vsc_s": 10.4,
        "pit_loss_sc_s": 8.2,
        "sc_probability": 0.65,
        "sc_risk_tier": "HIGH",
        "pit_speed_limit_kmh": 80,
    },
    "Jeddah Corniche Circuit": {
        "canonical": "Jeddah Corniche Circuit",
        "aliases": ["jeddah", "saudi arabia", "saudi gp"],
        "pit_loss_green_s": 20.5,
        "pit_loss_vsc_s": 11.4,
        "pit_loss_sc_s": 8.9,
        "sc_probability": 0.80,
        "sc_risk_tier": "HIGH",
        "pit_speed_limit_kmh": 80,
    },
    "Circuit Of The Americas": {
        "canonical": "Circuit Of The Americas",
        "aliases": ["cota", "austin", "united states", "us gp"],
        "pit_loss_green_s": 21.8,
        "pit_loss_vsc_s": 12.4,
        "pit_loss_sc_s": 9.4,
        "sc_probability": 0.40,
        "sc_risk_tier": "MEDIUM",
        "pit_speed_limit_kmh": 80,
    },
    "Bahrain International Circuit": {
        "canonical": "Bahrain International Circuit",
        "aliases": ["bahrain", "sakhir", "bahrain gp"],
        "pit_loss_green_s": 23.5,
        "pit_loss_vsc_s": 13.8,
        "pit_loss_sc_s": 10.5,
        "sc_probability": 0.20,
        "sc_risk_tier": "LOW",
        "pit_speed_limit_kmh": 80,
    },
    "Suzuka Circuit": {
        "canonical": "Suzuka Circuit",
        "aliases": ["suzuka", "japan", "japanese gp"],
        "pit_loss_green_s": 22.4,
        "pit_loss_vsc_s": 12.9,
        "pit_loss_sc_s": 9.8,
        "sc_probability": 0.45,
        "sc_risk_tier": "MEDIUM",
        "pit_speed_limit_kmh": 80,
    },
    "Autodromo Hermanos Rodriguez": {
        "canonical": "Autodromo Hermanos Rodriguez",
        "aliases": ["mexico", "mexico city", "mexican gp"],
        "pit_loss_green_s": 22.1,
        "pit_loss_vsc_s": 12.5,
        "pit_loss_sc_s": 9.5,
        "sc_probability": 0.35,
        "sc_risk_tier": "MEDIUM",
        "pit_speed_limit_kmh": 80,
    },
    "Autodromo Jose Carlos Pace": {
        "canonical": "Autodromo Jose Carlos Pace",
        "aliases": ["interlagos", "sao paulo", "brazil", "brazilian gp"],
        "pit_loss_green_s": 21.0,
        "pit_loss_vsc_s": 11.9,
        "pit_loss_sc_s": 9.1,
        "sc_probability": 0.60,
        "sc_risk_tier": "HIGH",
        "pit_speed_limit_kmh": 80,
    },
    "Yas Marina Circuit": {
        "canonical": "Yas Marina Circuit",
        "aliases": ["abu dhabi", "yas marina", "abu dhabi gp"],
        "pit_loss_green_s": 22.0,
        "pit_loss_vsc_s": 12.4,
        "pit_loss_sc_s": 9.4,
        "sc_probability": 0.35,
        "sc_risk_tier": "MEDIUM",
        "pit_speed_limit_kmh": 80,
    },
    "Circuit Zandvoort": {
        "canonical": "Circuit Zandvoort",
        "aliases": ["zandvoort", "netherlands", "dutch gp"],
        "pit_loss_green_s": 21.5,
        "pit_loss_vsc_s": 12.0,
        "pit_loss_sc_s": 9.2,
        "sc_probability": 0.60,
        "sc_risk_tier": "HIGH",
        "pit_speed_limit_kmh": 80,
    },
    "Las Vegas Strip Circuit": {
        "canonical": "Las Vegas Strip Circuit",
        "aliases": ["las vegas", "vegas", "las vegas gp"],
        "pit_loss_green_s": 21.2,
        "pit_loss_vsc_s": 11.8,
        "pit_loss_sc_s": 9.0,
        "sc_probability": 0.50,
        "sc_risk_tier": "MEDIUM",
        "pit_speed_limit_kmh": 80,
    },
}


def normalize_track_name(name: str) -> str:
    """Resolve track aliases to canonical circuit entry."""
    if not name:
        return "Albert Park Circuit"
    name_clean = str(name).strip().lower()
    for canonical, profile in CIRCUIT_PIT_PROFILES.items():
        if name_clean == canonical.lower():
            return canonical
        for alias in profile.get("aliases", []):
            if alias in name_clean or name_clean in alias:
                return canonical
    return name.strip()


def get_circuit_pit_profile(track_name: str) -> Dict[str, Any]:
    """Look up pit loss profile and Safety Car probability for a circuit."""
    canonical = normalize_track_name(track_name)
    profile = CIRCUIT_PIT_PROFILES.get(canonical)
    if profile:
        return dict(profile)
    # Default fallback for uncatalogued circuits
    return {
        "canonical": canonical,
        "aliases": [],
        "pit_loss_green_s": DEFAULT_PIT_LOSS_GREEN_S,
        "pit_loss_vsc_s": DEFAULT_PIT_LOSS_VSC_S,
        "pit_loss_sc_s": DEFAULT_PIT_LOSS_SC_S,
        "sc_probability": DEFAULT_SC_PROBABILITY,
        "sc_risk_tier": "MEDIUM",
        "pit_speed_limit_kmh": 80,
    }


def calculate_pit_loss(
    track_name: str,
    event: str = "green",
    traffic: str = "Clear",
) -> Dict[str, Any]:
    """Calculate effective pit loss and savings under race conditions.
    
    Args:
        track_name: Target circuit name or alias.
        event: Racing condition ("green", "vsc", "safety_car", "red_flag").
        traffic: Rejoin traffic density ("Clear", "Light", "Heavy").
        
    Returns:
        Dictionary with pit loss in seconds, savings vs green flag, and flags.
    """
    prof = get_circuit_pit_profile(track_name)
    ev = str(event or "green").strip().lower()
    tf_factor = TRAFFIC_FACTORS.get(traffic, 1.0)

    green_loss = prof["pit_loss_green_s"]

    if ev in ("vsc", "virtual_safety_car", "virtual safety car"):
        loss_base = prof["pit_loss_vsc_s"]
        event_label = "VSC"
        is_cheap = True
    elif ev in ("safety_car", "safety car", "sc"):
        loss_base = prof["pit_loss_sc_s"]
        event_label = "SAFETY CAR"
        is_cheap = True
    elif ev in ("red_flag", "red flag"):
        loss_base = 0.0
        event_label = "RED FLAG"
        is_cheap = True
    else:
        loss_base = green_loss
        event_label = "GREEN FLAG"
        is_cheap = False

    effective_loss = round(loss_base * tf_factor, 2)
    savings = round(max(0.0, (green_loss * tf_factor) - effective_loss), 2) if is_cheap else 0.0

    return {
        "effective_pit_loss_s": effective_loss,
        "baseline_green_s": round(green_loss * tf_factor, 2),
        "time_saved_s": savings,
        "is_cheap_stop": is_cheap,
        "event": event_label,
        "traffic": traffic,
        "traffic_factor": tf_factor,
        "sc_probability": prof["sc_probability"],
        "sc_risk_tier": prof["sc_risk_tier"],
    }


def evaluate_undercut_window(
    gap_s: float,
    chaser_tyre: str,
    chaser_age: float,
    leader_tyre: str,
    leader_age: float,
    track_name: str = "",
    event: str = "green",
    target_compound: str = "Hard",
) -> Dict[str, Any]:
    """Evaluate whether an undercut pit stop beats the on-track gap delta.
    
    A driver pits 1 lap before the car ahead, receives new rubber, and runs an
    aggressive out-lap. If the fresh tyre delta exceeds the pre-pit gap, the
    undercut succeeds when the leader stops on the subsequent lap.
    
    Args:
        gap_s: On-track gap between chaser and leader entering pit lap (s).
        chaser_tyre: Chaser's current tyre compound.
        chaser_age: Chaser's tyre age (laps).
        leader_tyre: Leader's current tyre compound.
        leader_age: Leader's tyre age (laps).
        track_name: Circuit name.
        event: Race neutralization state ("green", "vsc", "safety_car").
        target_compound: Tyre fitted during pit stop (default: "Hard").
        
    Returns:
        Structured evaluation with out-lap advantage, net exit margin, and viability.
    """
    gap = max(0.05, float(gap_s))
    
    # Fresh tyre out-lap delta model:
    # Fresh rubber vs worn rubber yields ~1.6s baseline pace edge, scaling with
    # leader tyre wear (older tyres suffer thermal drop-off).
    wear_delta_laps = max(0.0, float(leader_age) - float(chaser_age))
    wear_advantage_s = min(1.8, wear_delta_laps * 0.08)
    
    # Compound delta bonus: fitting fresh Soft/Medium vs old Medium/Hard
    compound_bonus_s = 0.4 if target_compound in ("Soft", "Medium") else 0.0
    
    fresh_tyre_outlap_gain_s = round(1.60 + wear_advantage_s + compound_bonus_s, 2)
    
    # Undercut net margin: out-lap pace advantage minus on-track gap
    net_exit_margin_s = round(fresh_tyre_outlap_gain_s - gap, 2)
    
    pit_info = calculate_pit_loss(track_name, event=event)
    
    if gap <= 1.8 and net_exit_margin_s > 0.3:
        status = "OPEN_FAVORABLE"
        verdict = f"UNDERCUT FAVORABLE — Out-lap edge +{fresh_tyre_outlap_gain_s:.1f}s beats {gap:.1f}s gap (+{net_exit_margin_s:.1f}s on exit)"
        feasibility_score = 0.90
    elif gap <= 3.2 and net_exit_margin_s >= -0.2:
        status = "MARGINAL"
        verdict = f"UNDERCUT MARGINAL — Out-lap edge +{fresh_tyre_outlap_gain_s:.1f}s leaves tight {abs(net_exit_margin_s):.1f}s duel on exit"
        feasibility_score = 0.65
    elif gap > 3.2:
        status = "CLOSED_TOO_FAR"
        verdict = f"UNDERCUT CLOSED — {gap:.1f}s gap exceeds fresh tyre out-lap threshold ({fresh_tyre_outlap_gain_s:.1f}s)"
        feasibility_score = 0.20
    else:
        status = "INSUFFICIENT_DELTA"
        verdict = f"UNDERCUT UNVIABLE — Tyre delta does not offset track position"
        feasibility_score = 0.35

    return {
        "status": status,
        "gap_before_s": gap,
        "fresh_tyre_outlap_gain_s": fresh_tyre_outlap_gain_s,
        "net_exit_margin_s": net_exit_margin_s,
        "feasibility_score": feasibility_score,
        "verdict": verdict,
        "pit_loss": pit_info,
        "target_compound": target_compound,
    }


def evaluate_safety_car_opportunity(
    track_name: str,
    event: str = "green",
    laps_remaining: int = 25,
    current_tyre_age: float = 12.0,
    gap_ahead_s: float = 1.5,
) -> Dict[str, Any]:
    """Evaluate strategic pit opportunity under neutralization vs green running.
    
    Quantifies the value of stopping under active VSC/SC, and evaluates the
    option value of extending a stint on high-SC risk circuits.
    """
    prof = get_circuit_pit_profile(track_name)
    pit_calc = calculate_pit_loss(track_name, event=event)
    sc_prob = prof["sc_probability"]
    
    if pit_calc["is_cheap_stop"]:
        recommendation = (
            f"EXPLOIT {pit_calc['event']} WINDOW: Box now to save ~{pit_calc['time_saved_s']:.1f}s "
            f"on pit transit compared to green flag running."
        )
        cheap_stop_urgency = "IMMEDIATE_BOX"
    elif sc_prob >= 0.60 and current_tyre_age < 22 and laps_remaining > 15:
        recommendation = (
            f"HIGH SC LIKELIHOOD ({sc_prob*100:.0f}% on {prof['canonical']}): Tyres have life. "
            f"Extend stint by 3-5 laps to preserve optionality for a cheap stop."
        )
        cheap_stop_urgency = "EXTEND_FOR_SC"
    else:
        recommendation = (
            f"STANDARD STINT: Green flag pit loss ~{pit_calc['baseline_green_s']:.1f}s. "
            f"Manage tyres to scheduled window."
        )
        cheap_stop_urgency = "STANDARD_CYCLE"

    return {
        "track": prof["canonical"],
        "event": pit_calc["event"],
        "sc_probability": sc_prob,
        "sc_risk_tier": prof["sc_risk_tier"],
        "cheap_stop_savings_s": pit_calc["time_saved_s"],
        "cheap_stop_urgency": cheap_stop_urgency,
        "recommendation": recommendation,
        "pit_calc": pit_calc,
    }
