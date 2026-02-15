import os
import json
from datetime import datetime, timedelta

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "wave_activity_log.json")

def load_wave_activity_log():
    try:
        if os.path.exists(DATA_PATH):
            with open(DATA_PATH, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return []

def get_wave_events(wave_name, event_type_filter=None, time_horizon_days=None, trigger_filter=None):
    all_events = load_wave_activity_log()
    filtered = [e for e in all_events if e.get("wave") == wave_name]

    if event_type_filter and event_type_filter != "All":
        filtered = [e for e in filtered if e.get("event_type") == event_type_filter]

    if time_horizon_days and time_horizon_days != "All":
        try:
            days = int(time_horizon_days)
            cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            filtered = [e for e in filtered if e.get("date", "") >= cutoff]
        except (ValueError, TypeError):
            pass

    if trigger_filter and trigger_filter != "All":
        trigger_lower = trigger_filter.lower()
        filtered = [e for e in filtered if trigger_lower in e.get("trigger", "").lower()]

    filtered.sort(key=lambda x: x.get("date", ""), reverse=True)
    return filtered

def get_wave_summary(wave_name, events):
    summary = {
        "current_regime": "Normal",
        "last_rebalance_date": "—",
        "last_overlay_change": "—",
        "wave_status": "Active",
    }

    try:
        adaptive_path = os.path.join(os.path.dirname(__file__), "..", "data", "adaptive_state.json")
        if os.path.exists(adaptive_path):
            with open(adaptive_path, "r") as f:
                adaptive = json.load(f)
            summary["current_regime"] = adaptive.get("regime_state", "Normal").title()
    except Exception:
        pass

    rebalances = [e for e in events if e.get("event_type") == "Rebalance"]
    if rebalances:
        summary["last_rebalance_date"] = rebalances[0].get("date", "—")

    overlays = [e for e in events if e.get("event_type") == "Overlay Adjustment"]
    if overlays:
        summary["last_overlay_change"] = overlays[0].get("date", "—")

    regime_shifts = [e for e in events if e.get("event_type") == "Regime Shift"]
    param_triggers = [e for e in events if e.get("event_type") == "Parameter Trigger"]

    if regime_shifts:
        summary["wave_status"] = "Under Review"
    elif param_triggers:
        summary["wave_status"] = "Monitoring"
    elif rebalances or overlays:
        summary["wave_status"] = "Active"
    else:
        summary["wave_status"] = "Stable"

    return summary
