import pandas as pd
import numpy as np


def compute_review_signals(snapshot_df, attrib_df):
    signals = []
    if snapshot_df is None or snapshot_df.empty:
        return signals

    if "alpha_30d" in snapshot_df.columns:
        alpha_vals = pd.to_numeric(snapshot_df["alpha_30d"], errors="coerce").dropna()
        if len(alpha_vals) > 0:
            negative_pct = (alpha_vals < 0).sum() / len(alpha_vals) * 100
            if negative_pct > 50:
                signals.append({
                    "signal": "Broad Alpha Deterioration",
                    "severity": "High",
                    "detail": f"{negative_pct:.0f}% of waves showing negative 30D alpha",
                    "recommendation": "Review portfolio composition"
                })

    if "drawdown_30d" in snapshot_df.columns:
        dd_vals = pd.to_numeric(snapshot_df["drawdown_30d"], errors="coerce").dropna()
        if len(dd_vals) > 0:
            high_dd = (dd_vals > 0.05).sum()
            if high_dd > 0:
                signals.append({
                    "signal": "Elevated Drawdown",
                    "severity": "Medium",
                    "detail": f"{high_dd} wave(s) exceeding 5% drawdown threshold",
                    "recommendation": "Monitor for continued deterioration"
                })

    return signals
