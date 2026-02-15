import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


def load_decision_log(path=None):
    log_path = path or Path("data/decision_log.json")
    if Path(log_path).exists():
        try:
            with open(log_path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return []


def get_canonical_wave_names(snapshot_df):
    if snapshot_df is None or snapshot_df.empty:
        return []
    if "display_name" in snapshot_df.columns:
        return snapshot_df["display_name"].dropna().unique().tolist()
    if "wave_name" in snapshot_df.columns:
        return snapshot_df["wave_name"].dropna().unique().tolist()
    return []


def _filter_decisions(decisions, scope, selected_wave):
    if scope == "wave" and selected_wave:
        return [d for d in decisions if d.get("wave") == selected_wave]
    return decisions


def compute_volatility_stress(snapshot_df):
    if snapshot_df is None or snapshot_df.empty:
        return {"level": "Low", "score": 0.2}
    if "benchmark_volatility_30d" in snapshot_df.columns:
        vol_vals = pd.to_numeric(snapshot_df["benchmark_volatility_30d"], errors="coerce").dropna()
        if len(vol_vals) > 0:
            avg_vol = vol_vals.mean()
            if avg_vol > 0.30:
                return {"level": "Elevated", "score": 0.8}
            elif avg_vol > 0.20:
                return {"level": "Moderate", "score": 0.5}
    return {"level": "Low", "score": 0.2}


def compute_cross_horizon_alignment(snapshot_df):
    if snapshot_df is None or snapshot_df.empty:
        return "Insufficient Data", "Not enough data to assess cross-horizon alignment."

    horizons = ["alpha_1d", "alpha_30d", "alpha_365d"]
    available = [h for h in horizons if h in snapshot_df.columns]
    if len(available) < 2:
        return "Insufficient Data", "Fewer than 2 alpha horizons available."

    positive_counts = []
    for h in available:
        vals = pd.to_numeric(snapshot_df[h], errors="coerce").dropna()
        if len(vals) > 0:
            pct_positive = (vals > 0).sum() / len(vals)
            positive_counts.append(pct_positive)

    if not positive_counts:
        return "Insufficient Data", "No valid alpha data."

    avg_positive = np.mean(positive_counts)
    if avg_positive > 0.7:
        return "Strong", f"Strong positive alignment across {len(available)} horizons ({avg_positive:.0%} positive)"
    elif avg_positive > 0.4:
        return "Moderate", f"Mixed alignment across horizons ({avg_positive:.0%} positive)"
    else:
        return "Weak", f"Weak alignment — majority of horizons show negative alpha ({avg_positive:.0%} positive)"


def compute_attribution_drag(attrib_df, scope, selected_wave):
    drags = []
    if attrib_df is None or attrib_df.empty:
        return drags

    filtered = attrib_df.copy()
    if scope == "wave" and selected_wave and "wave" in filtered.columns:
        filtered = filtered[filtered["wave"] == selected_wave]

    component_cols = ["selection_alpha", "momentum_alpha", "volatility_alpha", "regime_alpha", "exposure_alpha", "residual_alpha"]
    for col in component_cols:
        if col in filtered.columns:
            vals = pd.to_numeric(filtered[col], errors="coerce").dropna()
            if len(vals) > 0 and vals.mean() < -0.002:
                drags.append({
                    "component": col.replace("_alpha", "").replace("_", " ").title(),
                    "value": round(vals.mean(), 4),
                    "severity": "High" if vals.mean() < -0.01 else "Moderate"
                })
    return drags


def generate_orientation_sentence(vol_stress, alignment_level, alignment_desc, drags, snapshot_df):
    vol_part = f"Volatility stress is {vol_stress['level'].lower()}"
    align_part = f"cross-horizon alignment is {alignment_level.lower()}"

    if drags:
        drag_names = [d["component"] for d in drags[:2]]
        drag_part = f"with attribution drag observed in {', '.join(drag_names)}"
    else:
        drag_part = "with no significant attribution drags detected"

    wave_count = len(snapshot_df) if snapshot_df is not None else 0
    return f"Across {wave_count} monitored waves, {vol_part}, {align_part}, {drag_part}. All signals are observational and non-executing."


def compute_stage_counts(decisions, snapshot_df, attrib_df, scope, selected_wave):
    filtered = _filter_decisions(decisions, scope, selected_wave)

    awaiting = [d for d in filtered if d.get("status") == "Awaiting Approval"]
    under_review = [d for d in filtered if d.get("status") == "Under Review"]
    recorded = [d for d in filtered if d.get("status") == "Recorded"]
    with_outcomes = [d for d in recorded if d.get("outcome_30d") and d["outcome_30d"] != "Pending"]

    has_signals = snapshot_df is not None and not snapshot_df.empty
    has_attrib = attrib_df is not None and not attrib_df.empty

    approved = [d for d in filtered if d.get("approval_status") == "Approved"]
    pending_impl = [d for d in approved if d.get("implementation_state") in ["Pending Execution", "Scheduled", "In Progress"]]

    stages = [
        {"stage": "Signal Context", "status": "Active" if has_signals else "Insufficient Data", "label": f"{len(snapshot_df) if has_signals else 0} waves"},
        {"stage": "Issue / Opportunity", "status": "Active" if has_attrib else "Quiet", "label": f"{len(attrib_df) if has_attrib else 0} records"},
        {"stage": "Decision Formation", "status": "Active" if awaiting or under_review else "Quiet", "label": f"{len(awaiting) + len(under_review)} pending"},
        {"stage": "Approval & Governance", "status": "Active" if awaiting else "Quiet", "label": f"{len(awaiting)} awaiting"},
        {"stage": "Implementation", "status": "Active" if recorded else "Quiet", "label": f"{len(recorded)} recorded"},
        {"stage": "Outcome Observation", "status": "Active" if with_outcomes else "Quiet", "label": f"{len(with_outcomes)} observed"},
        {"stage": "Learning & Adaptation", "status": "Active" if len(with_outcomes) >= 2 else "Quiet", "label": f"{len(with_outcomes)} inputs"},
        {"stage": "Finalization", "status": "Active" if approved else "Quiet", "label": f"{len(pending_impl)} pending"},
    ]
    return stages


def _make_stage_response(what_this_means, bullets, why_it_matters, review_prompts, sources, **extra):
    response = {
        "what_this_means": what_this_means,
        "bullets": bullets,
        "why_it_matters": why_it_matters,
        "review_prompts": review_prompts,
        "sources": sources,
    }
    response.update(extra)
    return response


def compute_stage_1(snapshot_df, attrib_df, scope, selected_wave):
    bullets = []
    if snapshot_df is not None and not snapshot_df.empty:
        filtered = snapshot_df.copy()
        if scope == "wave" and selected_wave and "display_name" in filtered.columns:
            filtered = filtered[filtered["display_name"] == selected_wave]

        if "alpha_30d" in filtered.columns:
            alpha_vals = pd.to_numeric(filtered["alpha_30d"], errors="coerce").dropna()
            if len(alpha_vals) > 0:
                mean_a = alpha_vals.mean()
                bullets.append(f"Mean 30D alpha: {mean_a:.4f} ({'positive' if mean_a > 0 else 'negative'})")
                positive_pct = (alpha_vals > 0).sum() / len(alpha_vals) * 100
                bullets.append(f"{positive_pct:.0f}% of waves showing positive 30D alpha")

        if "return_30d" in filtered.columns:
            ret_vals = pd.to_numeric(filtered["return_30d"], errors="coerce").dropna()
            if len(ret_vals) > 0:
                bullets.append(f"30D return range: {ret_vals.min():.2%} to {ret_vals.max():.2%}")

        if "drawdown_30d" in filtered.columns:
            dd_vals = pd.to_numeric(filtered["drawdown_30d"], errors="coerce").dropna()
            if len(dd_vals) > 0:
                max_dd = dd_vals.max()
                bullets.append(f"Maximum 30D drawdown: {max_dd:.2%}")

    what_this_means = "The system is observing current signal data across monitored waves to provide situational awareness."
    if not bullets:
        what_this_means = "No signal data currently available for the selected scope."

    return _make_stage_response(
        what_this_means=what_this_means,
        bullets=bullets,
        why_it_matters="Signal context provides the foundation for identifying whether conditions warrant further attention or are within normal operating ranges.",
        review_prompts=[
            "Are return and alpha signals consistent with expectations?",
            "Are any drawdown levels approaching policy thresholds?",
            "Is cross-horizon alignment supporting or conflicting?"
        ],
        sources=["live_snapshot.csv", "alpha_attribution_summary.csv"]
    )


def compute_stage_2(attrib_df, scope, selected_wave):
    signals_list = []
    if attrib_df is not None and not attrib_df.empty:
        filtered = attrib_df.copy()
        if scope == "wave" and selected_wave and "wave" in filtered.columns:
            filtered = filtered[filtered["wave"] == selected_wave]

        component_cols = ["selection_alpha", "momentum_alpha", "volatility_alpha", "regime_alpha", "exposure_alpha", "residual_alpha"]
        for col in component_cols:
            if col in filtered.columns:
                vals = pd.to_numeric(filtered[col], errors="coerce").dropna()
                if len(vals) > 0:
                    mean_val = vals.mean()
                    if abs(mean_val) > 0.003:
                        component_name = col.replace("_alpha", "").replace("_", " ").title()
                        is_drag = mean_val < 0
                        signals_list.append({
                            "component": component_name,
                            "value": round(mean_val, 4),
                            "status": "Review Recommended" if is_drag else "Contributing",
                            "scope": scope if scope == "portfolio" else selected_wave or "portfolio",
                            "description": f"{'Drag' if is_drag else 'Contribution'} of {mean_val:.4f} detected",
                            "explanation": f"This component is {'detracting from' if is_drag else 'adding to'} portfolio alpha across the observed horizons.",
                            "typical_cause": f"{'Adverse' if is_drag else 'Favorable'} {component_name.lower()} conditions relative to benchmark"
                        })

    return _make_stage_response(
        what_this_means="Attribution analysis identifies which components are contributing to or detracting from alpha generation." if signals_list else "No significant attribution signals detected at current thresholds.",
        bullets=[f"{s['component']}: {s['value']:.4f} ({s['status']})" for s in signals_list[:5]],
        why_it_matters="Understanding attribution drivers helps distinguish between structural alpha and transient effects, informing whether action may be warranted.",
        review_prompts=[
            "Are attribution drags structural or transient?",
            "Do contributing components align with the investment thesis?",
            "Should any component trigger a deeper review?"
        ],
        sources=["alpha_attribution_summary.csv"],
        has_signals=len(signals_list) > 0,
        signals=signals_list
    )


def compute_stage_3(decisions, scope, selected_wave):
    filtered = _filter_decisions(decisions, scope, selected_wave)
    in_formation = [d for d in filtered if d.get("status") in ("Awaiting Approval", "Under Review")]

    items = []
    for d in in_formation:
        items.append({
            "id": d.get("id", "N/A"),
            "status": d.get("status", "Unknown"),
            "scope": d.get("scope", "portfolio"),
            "type": d.get("decision_type", d.get("event_type", "Other")),
            "actor": d.get("actor", "Unknown"),
            "date": d.get("date", "N/A"),
            "about": d.get("context_notes", "No context provided"),
            "context_note": d.get("rationale", "")
        })

    return _make_stage_response(
        what_this_means=f"{len(items)} decision(s) currently in formation." if items else "No decisions are currently in the formation stage.",
        bullets=[f"{item['id']}: {item['type']} ({item['status']})" for item in items],
        why_it_matters="Decisions in formation represent active governance engagement. Tracking them ensures nothing falls through the process without review.",
        review_prompts=[
            "Are all in-formation decisions properly scoped?",
            "Do context notes adequately capture the rationale?",
            "Is the governance pathway clear for each decision?"
        ],
        sources=["decision_log.json"],
        items=items
    )


def compute_stage_4(decisions, scope, selected_wave):
    filtered = _filter_decisions(decisions, scope, selected_wave)

    counts = {
        "Awaiting Approval": 0,
        "Under Review": 0,
        "Recorded": 0,
        "Deferred": 0,
        "Modified": 0
    }
    for d in filtered:
        status = d.get("status", "")
        if status in counts:
            counts[status] += 1

    awaiting_details = [d for d in filtered if d.get("status") == "Awaiting Approval"]

    total = sum(counts.values())
    if counts["Awaiting Approval"] > 2:
        posture = "Elevated governance attention required — multiple decisions awaiting approval."
    elif counts["Awaiting Approval"] > 0:
        posture = "Normal governance posture — decisions are progressing through approval."
    else:
        posture = "Clear governance posture — no decisions pending approval."

    return _make_stage_response(
        what_this_means=f"{total} decisions tracked in governance pipeline. {counts['Awaiting Approval']} awaiting approval.",
        bullets=[f"{status}: {count}" for status, count in counts.items() if count > 0],
        why_it_matters="The governance stage ensures decisions receive proper review and authorization before implementation. Bottlenecks here may delay necessary actions.",
        review_prompts=[
            "Are any decisions stalled in the approval queue?",
            "Is the approval cadence appropriate for current conditions?",
            "Do deferred decisions need revisiting?"
        ],
        sources=["decision_log.json"],
        counts=counts,
        governance_posture=posture,
        awaiting_details=awaiting_details,
        status_explanations={
            "Awaiting Approval": "Decision has been proposed and is waiting for authorized approval.",
            "Under Review": "Decision is being actively evaluated by the designated reviewer.",
            "Recorded": "Decision has been approved and formally recorded.",
            "Deferred": "Decision has been postponed for future consideration.",
            "Modified": "Original decision was adjusted based on review feedback."
        }
    )


def compute_stage_5(decisions, snapshot_df, attrib_df, scope, selected_wave):
    filtered = _filter_decisions(decisions, scope, selected_wave)
    recorded = [d for d in filtered if d.get("status") == "Recorded"]

    items = []
    for d in recorded:
        decision_date = d.get("date", "")
        try:
            d_date = datetime.strptime(decision_date, "%Y-%m-%d")
            days_since = (datetime.now() - d_date).days
            if days_since <= 90:
                window = f"{days_since}D post-decision"
            else:
                continue
        except (ValueError, TypeError):
            window = "Unknown"
            days_since = 0

        watching = [
            f"Alpha trajectory since decision ({decision_date})",
            f"Regime consistency with conditions at decision time ({d.get('regime_at_decision', 'N/A')})",
            "Attribution component stability"
        ]

        items.append({
            "id": d.get("id", "N/A"),
            "scope": d.get("scope", "portfolio"),
            "type": d.get("decision_type", d.get("event_type", "Other")),
            "monitoring_window": window,
            "watching": watching
        })

    summary = f"{len(items)} decision(s) currently being monitored post-implementation." if items else "No decisions in active monitoring window."

    return _make_stage_response(
        what_this_means=summary,
        bullets=[f"{item['id']}: {item['type']} — {item['monitoring_window']}" for item in items],
        why_it_matters="Post-decision monitoring validates whether the conditions that informed the decision remain intact and whether outcomes are tracking as expected.",
        review_prompts=[
            "Are monitored decisions tracking within expected ranges?",
            "Have market conditions changed materially since implementation?",
            "Should any monitoring windows be extended or shortened?"
        ],
        sources=["decision_log.json", "live_snapshot.csv"],
        items=items,
        monitoring_summary=summary
    )


def compute_stage_6(decisions, scope, selected_wave):
    filtered = _filter_decisions(decisions, scope, selected_wave)

    outcomes = []
    for d in filtered:
        o30 = d.get("outcome_30d", "Pending")
        o90 = d.get("outcome_90d", "Pending")
        if o30 == "Pending" and o90 == "Pending":
            classification = "Pending"
        elif d.get("regime_at_decision") == "Elevated Volatility":
            classification = "Regime-Influenced"
        else:
            try:
                val = float(str(o30).replace("%", "").replace("+", "")) if o30 != "Pending" else None
                if val is not None:
                    classification = "Observed" if val > 0 else "Mixed"
                else:
                    classification = "Pending"
            except (ValueError, TypeError):
                classification = "Pending"

        outcomes.append({
            "id": d.get("id", "N/A"),
            "scope": d.get("scope", "portfolio"),
            "type": d.get("decision_type", d.get("event_type", "Other")),
            "outcome_30d": o30 if o30 else "Pending",
            "outcome_90d": o90 if o90 else "Pending",
            "classification": classification,
            "regime_context": d.get("regime_at_decision", "N/A"),
            "explanation": f"Decision {d.get('id', 'N/A')} ({d.get('decision_type', 'Other')}) made on {d.get('date', 'N/A')}. 30D outcome: {o30}. Classification: {classification}."
        })

    observed = [o for o in outcomes if o["classification"] not in ("Pending",)]
    positive = [o for o in observed if o["classification"] == "Observed"]

    if observed:
        pattern = f"{len(positive)} of {len(observed)} observed decisions showing positive outcomes."
    else:
        pattern = "No outcomes available for pattern analysis yet."

    return _make_stage_response(
        what_this_means=f"{len(outcomes)} decisions tracked. {len(observed)} have observable outcomes.",
        bullets=[f"{o['id']}: {o['outcome_30d']} (30D), {o['classification']}" for o in outcomes[:5]],
        why_it_matters="Outcome observation completes the decision lifecycle by connecting initial signals and governance actions to actual results, enabling evidence-based learning.",
        review_prompts=[
            "Are outcome patterns consistent with the decision thesis?",
            "Do regime-influenced outcomes require separate analysis?",
            "Are there systematic biases in decision outcomes?"
        ],
        sources=["decision_log.json"],
        outcomes=outcomes,
        outcome_pattern=pattern
    )


def compute_stage_7(decisions, attrib_df, scope, selected_wave):
    filtered = _filter_decisions(decisions, scope, selected_wave)
    with_outcomes = [d for d in filtered if d.get("outcome_30d") and d["outcome_30d"] != "Pending"]

    learnings = []
    if len(with_outcomes) >= 2:
        positive_outcomes = []
        negative_outcomes = []
        for d in with_outcomes:
            try:
                val = float(str(d["outcome_30d"]).replace("%", "").replace("+", ""))
                if val > 0:
                    positive_outcomes.append(d)
                else:
                    negative_outcomes.append(d)
            except (ValueError, TypeError):
                pass

        if positive_outcomes:
            types = set(d.get("decision_type", "Other") for d in positive_outcomes)
            learnings.append(f"Positive outcomes observed in {', '.join(types)} decisions — these decision types may have stronger process alignment.")

        if negative_outcomes:
            types = set(d.get("decision_type", "Other") for d in negative_outcomes)
            learnings.append(f"Mixed or negative outcomes in {', '.join(types)} — consider whether timing or regime conditions played a role.")

        regime_decisions = [d for d in with_outcomes if d.get("regime_at_decision") == "Elevated Volatility"]
        if regime_decisions:
            learnings.append(f"{len(regime_decisions)} decision(s) made during elevated volatility — outcomes may be regime-influenced rather than process-driven.")

    if not learnings:
        learnings.append("Insufficient outcome data to generate learning signals. More completed decision cycles are needed.")

    return _make_stage_response(
        what_this_means="Emerging patterns from completed decision cycles are summarized below for institutional review." if len(with_outcomes) >= 2 else "Not enough completed decision cycles to generate learning patterns.",
        bullets=[],
        why_it_matters="Learning and adaptation ensure that the decision process improves over time by incorporating evidence from completed cycles.",
        review_prompts=[
            "Do learning signals suggest process improvements?",
            "Are there decision types that consistently underperform?",
            "Should regime-awareness be incorporated into future decisions?"
        ],
        sources=["decision_log.json", "alpha_attribution_summary.csv"],
        learnings=learnings
    )


def get_wave_decision_history(decisions, wave_name):
    wave_decisions = [d for d in decisions if d.get("wave") == wave_name]
    wave_decisions.sort(key=lambda d: d.get("date", ""), reverse=False)
    return wave_decisions
