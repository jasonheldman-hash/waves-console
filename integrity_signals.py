import pandas as pd
import numpy as np


def compute_selection_integrity(snapshot_df, attrib_df):
    result = {
        "aggregate": {
            "holdings_compliant_pct": 0,
            "total_holdings": 0,
            "drift_status": "Low",
            "median_decay_score": 1,
        },
        "holdings": []
    }

    if snapshot_df is None or snapshot_df.empty:
        return result

    name_col = "display_name" if "display_name" in snapshot_df.columns else "wave_name"
    if name_col not in snapshot_df.columns:
        return result

    holdings = []
    compliant_count = 0
    decay_scores = []

    for _, row in snapshot_df.iterrows():
        wave_name = row.get(name_col, "Unknown")

        alpha_30d = pd.to_numeric(row.get("alpha_30d", 0), errors="coerce")
        if pd.isna(alpha_30d):
            alpha_30d = 0.0

        selection_alpha = 0.0
        if attrib_df is not None and not attrib_df.empty and "selection_alpha" in attrib_df.columns:
            wave_attrib = attrib_df[attrib_df.get("wave", pd.Series()) == wave_name] if "wave" in attrib_df.columns else pd.DataFrame()
            if not wave_attrib.empty:
                sel_vals = pd.to_numeric(wave_attrib["selection_alpha"], errors="coerce").dropna()
                if len(sel_vals) > 0:
                    selection_alpha = float(sel_vals.mean())

        criteria_met = alpha_30d > -0.02 and selection_alpha > -0.01

        if alpha_30d < -0.05:
            decay_risk = "High"
            decay_score = 3
        elif alpha_30d < -0.02:
            decay_risk = "Medium"
            decay_score = 2
        else:
            decay_risk = "Low"
            decay_score = 1

        if criteria_met:
            compliant_count += 1

        decay_scores.append(decay_score)
        holdings.append({
            "wave": wave_name,
            "current_criteria_met": criteria_met,
            "decay_risk": decay_risk,
            "alpha_30d": round(float(alpha_30d) * 100, 2),
            "selection_alpha": round(float(selection_alpha) * 100, 2),
        })

    total = len(holdings)
    result["aggregate"]["total_holdings"] = total
    result["aggregate"]["holdings_compliant_pct"] = round(compliant_count / total * 100, 1) if total > 0 else 0

    if decay_scores:
        result["aggregate"]["median_decay_score"] = int(np.median(decay_scores))

    high_decay = sum(1 for s in decay_scores if s >= 3)
    if high_decay > total * 0.3:
        result["aggregate"]["drift_status"] = "High"
    elif high_decay > total * 0.1:
        result["aggregate"]["drift_status"] = "Moderate"
    else:
        result["aggregate"]["drift_status"] = "Low"

    result["holdings"] = holdings
    return result
