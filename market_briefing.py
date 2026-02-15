import numpy as np
import pandas as pd
from helpers.market_data import (
    compute_returns, compute_slope, compute_realized_vol, compute_vol_of_vol,
    compute_drawdown, compute_pct_up_days, compute_above_ma, compute_relative_strength,
    load_yesterday_snapshot, save_yesterday_snapshot,
    BENCHMARK_TICKERS, SECTOR_TICKERS, RATE_TICKERS
)


def compute_direction_label(ret):
    if ret is None:
        return "—", "Data unavailable"
    if ret > 0.02:
        return "Up", f"+{ret*100:.1f}%"
    elif ret < -0.02:
        return "Down", f"{ret*100:.1f}%"
    else:
        return "Flat", f"{ret*100:.1f}%"


def compute_strength_score(prices, days):
    if prices is None or len(prices) < days:
        return 0
    ret = compute_returns(prices, days)
    pct_up = compute_pct_up_days(prices, days)
    dd = compute_drawdown(prices, days)
    slope = compute_slope(prices, days)

    score = 50
    if ret is not None:
        if ret > 0.10:
            score += 20
        elif ret > 0.05:
            score += 15
        elif ret > 0.02:
            score += 10
        elif ret > 0:
            score += 5
        elif ret > -0.02:
            score -= 5
        elif ret > -0.05:
            score -= 10
        else:
            score -= 20

    if pct_up is not None:
        if pct_up > 60:
            score += 10
        elif pct_up > 55:
            score += 5
        elif pct_up < 40:
            score -= 10
        elif pct_up < 45:
            score -= 5

    if dd is not None:
        if dd > -0.03:
            score += 10
        elif dd > -0.05:
            score += 5
        elif dd < -0.10:
            score -= 10
        elif dd < -0.07:
            score -= 5

    if slope is not None:
        if slope > 0.1:
            score += 10
        elif slope > 0.05:
            score += 5
        elif slope < -0.1:
            score -= 10

    return max(0, min(100, score))


def compute_horizon_explanation(prices_dict, days, label):
    spy = prices_dict.get("SPY")
    qqq = prices_dict.get("QQQ")
    iwm = prices_dict.get("IWM")
    efa = prices_dict.get("EFA")
    hyg = prices_dict.get("HYG")

    parts = []
    spy_ret = compute_returns(spy, days) if spy is not None else None
    qqq_ret = compute_returns(qqq, days) if qqq is not None else None
    iwm_ret = compute_returns(iwm, days) if iwm is not None else None
    efa_ret = compute_returns(efa, days) if efa is not None else None

    if spy_ret is not None:
        if spy_ret > 0.03:
            parts.append(f"The S&P 500 has advanced {spy_ret*100:.1f}% over {label}, indicating sustained buying interest")
        elif spy_ret < -0.02:
            parts.append(f"The S&P 500 has declined {spy_ret*100:.1f}% over {label}, reflecting broad selling pressure")
        else:
            parts.append(f"The S&P 500 is roughly flat ({spy_ret*100:+.1f}%) over {label}, suggesting range-bound conditions")

    if qqq_ret is not None and spy_ret is not None:
        diff = qqq_ret - spy_ret
        if diff > 0.02:
            parts.append("Growth/tech is outperforming, suggesting risk appetite favors high-beta areas")
        elif diff < -0.02:
            parts.append("Growth/tech is lagging broad markets, which may signal rotation into value or defensive sectors")

    if iwm_ret is not None and spy_ret is not None:
        diff = iwm_ret - spy_ret
        if diff > 0.02:
            parts.append("Small caps are participating, which historically signals broad-based confidence")
        elif diff < -0.03:
            parts.append("Small caps are notably underperforming, suggesting narrow leadership concentrated in large caps")

    spy_vol = compute_realized_vol(spy, min(days, 21)) if spy is not None else None
    if spy_vol is not None:
        if spy_vol > 0.25:
            parts.append("Realized volatility is elevated, which may compress risk-adjusted returns")
        elif spy_vol < 0.10:
            parts.append("Volatility is compressed, creating a generally supportive backdrop for trend-following approaches")

    spy_pct_up = compute_pct_up_days(spy, min(days, 60)) if spy is not None else None
    if spy_pct_up is not None:
        if spy_pct_up > 60:
            parts.append(f"Market breadth is constructive with {spy_pct_up:.0f}% positive sessions")
        elif spy_pct_up < 40:
            parts.append(f"Session-level breadth is weak with only {spy_pct_up:.0f}% positive days")

    hyg_ret = compute_returns(hyg, days) if hyg is not None else None
    if hyg_ret is not None:
        if hyg_ret > 0.01:
            parts.append("Credit markets are supportive, with high-yield spreads tightening")
        elif hyg_ret < -0.02:
            parts.append("Credit markets show signs of stress, with high-yield underperformance")

    if not parts:
        return "Market data is being established for this horizon."

    return ". ".join(parts) + "."


def compute_volatility_stress_assessment(prices_dict):
    spy = prices_dict.get("SPY")
    qqq = prices_dict.get("QQQ")
    iwm = prices_dict.get("IWM")

    vol_readings = []
    vol_of_vol_readings = []
    dd_readings = []
    for t in ["SPY", "QQQ", "IWM", "EFA", "HYG"]:
        p = prices_dict.get(t)
        if p is not None:
            v = compute_realized_vol(p, 21)
            if v is not None:
                vol_readings.append(v)
            vov = compute_vol_of_vol(p, 63)
            if vov is not None:
                vol_of_vol_readings.append(vov)
            dd = compute_drawdown(p, 30)
            if dd is not None:
                dd_readings.append(dd)

    avg_vol = np.mean(vol_readings) if vol_readings else 0.15
    avg_vov = np.mean(vol_of_vol_readings) if vol_of_vol_readings else 0.03
    worst_dd = min(dd_readings) if dd_readings else -0.03

    regime = "Neutral"
    if avg_vol < 0.10:
        regime = "Compression"
    elif avg_vol > 0.28:
        regime = "Expansion"
    elif avg_vol > 0.20 and avg_vov > 0.04:
        regime = "Exhaustion"

    opp_context = "Neutral"
    if regime == "Compression":
        opp_context = "Tailwind"
    elif regime in ["Expansion", "Exhaustion"]:
        opp_context = "Headwind"

    stress_level = "Low"
    if avg_vol > 0.25 or worst_dd < -0.10 or avg_vov > 0.06:
        stress_level = "Elevated"
    elif avg_vol > 0.16 or worst_dd < -0.05 or avg_vov > 0.04:
        stress_level = "Moderate"

    spy_vol_short = compute_realized_vol(spy, 10) if spy is not None and len(spy) > 15 else None
    spy_vol_long = compute_realized_vol(spy, 42) if spy is not None and len(spy) > 50 else None
    trend = "Stable"
    if spy_vol_short is not None and spy_vol_long is not None:
        if spy_vol_short > spy_vol_long * 1.2:
            trend = "Rising"
        elif spy_vol_short < spy_vol_long * 0.8:
            trend = "Subsiding"

    vol_agreement = True
    if len(vol_readings) >= 3:
        vol_std = np.std(vol_readings)
        if vol_std > 0.08:
            vol_agreement = False

    return {
        "regime": regime,
        "opportunity_context": opp_context,
        "stress_level": stress_level,
        "trend": trend,
        "avg_vol": avg_vol,
        "avg_vov": avg_vov,
        "worst_dd": worst_dd,
        "cross_asset_agreement": vol_agreement,
    }


def compute_breadth_assessment(prices_dict):
    equity_tickers = ["SPY", "QQQ", "IWM", "EFA"]
    above_50 = 0
    above_200 = 0
    total = 0
    for t in equity_tickers:
        p = prices_dict.get(t)
        if p is not None and len(p) > 200:
            total += 1
            a50 = compute_above_ma(p, 50)
            a200 = compute_above_ma(p, 200)
            if a50:
                above_50 += 1
            if a200:
                above_200 += 1

    pct_above_50 = (above_50 / total * 100) if total > 0 else 0
    pct_above_200 = (above_200 / total * 100) if total > 0 else 0

    classification = "Mixed"
    if pct_above_50 >= 75 and pct_above_200 >= 75:
        classification = "Broad"
    elif pct_above_50 <= 25 or pct_above_200 <= 25:
        classification = "Narrow"

    cross_asset = {}
    asset_groups = {
        "Equities": ["SPY", "QQQ", "IWM"],
        "Credit": ["HYG", "LQD"],
        "Rates": ["TLT", "IEF"],
        "Commodities": ["GLD"],
        "Crypto": ["BTC-USD"],
    }
    for group, tickers in asset_groups.items():
        trends = []
        vols = []
        for t in tickers:
            p = prices_dict.get(t)
            if p is not None:
                ret = compute_returns(p, 30)
                vol = compute_realized_vol(p, 21)
                if ret is not None:
                    trends.append(ret)
                if vol is not None:
                    vols.append(vol)
        avg_trend = np.mean(trends) if trends else None
        avg_vol_val = np.mean(vols) if vols else None

        if avg_trend is not None:
            if avg_trend > 0.02:
                trend_label = "Up"
            elif avg_trend < -0.02:
                trend_label = "Down"
            else:
                trend_label = "Flat"
        else:
            trend_label = "—"

        if avg_vol_val is not None:
            if avg_vol_val > 0.25:
                vol_label = "Stress"
            elif avg_vol_val > 0.15:
                vol_label = "Choppy"
            else:
                vol_label = "Stable"
        else:
            vol_label = "—"

        interp = _cross_asset_interpretation(group, trend_label, vol_label)
        cross_asset[group] = {"trend": trend_label, "volatility": vol_label, "interpretation": interp}

    return {
        "pct_above_50dma": pct_above_50,
        "pct_above_200dma": pct_above_200,
        "total_tracked": total,
        "above_50": above_50,
        "above_200": above_200,
        "classification": classification,
        "cross_asset": cross_asset,
    }


def _cross_asset_interpretation(group, trend, vol):
    if trend == "Up" and vol == "Stable":
        return f"{group} showing constructive trend with contained volatility."
    elif trend == "Up" and vol in ["Choppy", "Stress"]:
        return f"{group} trending higher but with elevated volatility — less reliable signal."
    elif trend == "Down" and vol == "Stress":
        return f"{group} under pressure with stress-level volatility — defensiveness warranted."
    elif trend == "Down" and vol == "Stable":
        return f"{group} drifting lower in orderly fashion — not yet signaling acute stress."
    elif trend == "Flat":
        return f"{group} range-bound — no clear directional signal at this time."
    return f"{group} data limited — interpretation unavailable."


def compute_rates_credit_assessment(prices_dict):
    tlt = prices_dict.get("TLT")
    ief = prices_dict.get("IEF")
    shy = prices_dict.get("SHY")
    hyg = prices_dict.get("HYG")
    lqd = prices_dict.get("LQD")
    uup = prices_dict.get("UUP")

    rates_trend = "—"
    if tlt is not None:
        tlt_ret = compute_returns(tlt, 30)
        if tlt_ret is not None:
            if tlt_ret > 0.02:
                rates_trend = "Falling"
            elif tlt_ret < -0.02:
                rates_trend = "Rising"
            else:
                rates_trend = "Flat"

    curve_proxy = "—"
    if tlt is not None and shy is not None:
        tlt_ret30 = compute_returns(tlt, 30)
        shy_ret30 = compute_returns(shy, 30)
        if tlt_ret30 is not None and shy_ret30 is not None:
            spread_change = tlt_ret30 - shy_ret30
            if spread_change > 0.01:
                curve_proxy = "Steepening"
            elif spread_change < -0.01:
                curve_proxy = "Flattening"
            else:
                curve_proxy = "Stable"

    credit_condition = "—"
    if hyg is not None and lqd is not None:
        hyg_ret = compute_returns(hyg, 30)
        lqd_ret = compute_returns(lqd, 30)
        if hyg_ret is not None and lqd_ret is not None:
            credit_spread_proxy = hyg_ret - lqd_ret
            if credit_spread_proxy > 0.01:
                credit_condition = "Tightening"
            elif credit_spread_proxy < -0.01:
                credit_condition = "Widening"
            else:
                credit_condition = "Stable"
    elif hyg is not None:
        hyg_ret = compute_returns(hyg, 30)
        if hyg_ret is not None:
            credit_condition = "Stable" if hyg_ret > -0.01 else "Widening"

    liquidity_proxy = "—"
    spy = prices_dict.get("SPY")
    qqq = prices_dict.get("QQQ")
    if qqq is not None and spy is not None:
        qqq_ret = compute_returns(qqq, 30)
        spy_ret = compute_returns(spy, 30)
        if qqq_ret is not None and spy_ret is not None:
            risk_appetite = qqq_ret - spy_ret
            if risk_appetite > 0.02:
                liquidity_proxy = "Risk-seeking"
            elif risk_appetite < -0.02:
                liquidity_proxy = "Risk-averse"
            else:
                liquidity_proxy = "Neutral"

    dollar_trend = "—"
    if uup is not None:
        uup_ret = compute_returns(uup, 30)
        if uup_ret is not None:
            if uup_ret > 0.01:
                dollar_trend = "Strengthening"
            elif uup_ret < -0.01:
                dollar_trend = "Weakening"
            else:
                dollar_trend = "Stable"

    return {
        "rates_trend": rates_trend,
        "curve_proxy": curve_proxy,
        "credit_condition": credit_condition,
        "liquidity_proxy": liquidity_proxy,
        "dollar_trend": dollar_trend,
    }


def compute_sector_assessment(prices_dict):
    spy = prices_dict.get("SPY")
    results = []
    for ticker, name in SECTOR_TICKERS.items():
        p = prices_dict.get(ticker)
        if p is None:
            continue
        rs_30 = compute_relative_strength(p, spy, 30)
        rs_90 = compute_relative_strength(p, spy, 90)
        ret_30 = compute_returns(p, 30)
        results.append({
            "ticker": ticker,
            "name": name,
            "rs_30d": rs_30,
            "rs_90d": rs_90,
            "return_30d": ret_30,
        })

    results.sort(key=lambda x: (x["rs_30d"] or 0), reverse=True)
    top2 = results[:2] if len(results) >= 2 else results
    bottom2 = results[-2:] if len(results) >= 2 else []

    return {
        "sectors": results,
        "top2": top2,
        "bottom2": bottom2,
    }


def compute_regime_assessment(prices_dict):
    spy = prices_dict.get("SPY")
    qqq = prices_dict.get("QQQ")
    hyg = prices_dict.get("HYG")

    spy_ret = compute_returns(spy, 30) if spy is not None else None
    spy_above_50 = compute_above_ma(spy, 50) if spy is not None else None
    spy_above_200 = compute_above_ma(spy, 200) if spy is not None else None
    hyg_ret = compute_returns(hyg, 30) if hyg is not None else None

    risk_on_signals = 0
    risk_off_signals = 0
    total_signals = 0

    if spy_ret is not None:
        total_signals += 1
        if spy_ret > 0.02:
            risk_on_signals += 1
        elif spy_ret < -0.02:
            risk_off_signals += 1

    if spy_above_50 is not None:
        total_signals += 1
        if spy_above_50:
            risk_on_signals += 1
        else:
            risk_off_signals += 1

    if spy_above_200 is not None:
        total_signals += 1
        if spy_above_200:
            risk_on_signals += 1
        else:
            risk_off_signals += 1

    if hyg_ret is not None:
        total_signals += 1
        if hyg_ret > 0.005:
            risk_on_signals += 1
        elif hyg_ret < -0.01:
            risk_off_signals += 1

    breadth = compute_breadth_assessment(prices_dict)
    if breadth["classification"] == "Broad":
        risk_on_signals += 1
        total_signals += 1
    elif breadth["classification"] == "Narrow":
        risk_off_signals += 1
        total_signals += 1
    else:
        total_signals += 1

    if total_signals == 0:
        return "Transitional"

    on_pct = risk_on_signals / total_signals
    off_pct = risk_off_signals / total_signals

    if on_pct >= 0.7:
        return "Risk-On"
    elif off_pct >= 0.7:
        return "Risk-Off"
    else:
        return "Transitional"


def compute_executive_chips(prices_dict, vol_assessment, breadth_assessment, rates_assessment, regime):
    chips = []

    regime_color = "#48BB78" if regime == "Risk-On" else ("#EF4444" if regime == "Risk-Off" else "#F59E0B")
    chips.append(("Market Regime", regime, regime_color))

    spy = prices_dict.get("SPY")
    for days, label in [(30, "30D"), (90, "90D"), (365, "365D")]:
        ret = compute_returns(spy, days) if spy is not None else None
        dir_label, desc = compute_direction_label(ret)
        color = "#48BB78" if dir_label == "Up" else ("#EF4444" if dir_label == "Down" else "#9EA3AE")
        chips.append((f"Direction ({label})", f"{dir_label} ({desc})", color))

    sl = vol_assessment["stress_level"]
    tr = vol_assessment["trend"]
    sc = "#48BB78" if sl == "Low" else ("#F59E0B" if sl == "Moderate" else "#EF4444")
    chips.append(("Vol Stress", f"{sl} · {tr}", sc))

    bc = breadth_assessment["classification"]
    bcolor = "#48BB78" if bc == "Broad" else ("#F59E0B" if bc == "Mixed" else "#EF4444")
    chips.append(("Breadth", bc, bcolor))

    rt = rates_assessment["rates_trend"]
    rcolor = "#60A5FA"
    chips.append(("Rates Regime", rt, rcolor))

    cc = rates_assessment["credit_condition"]
    ccolor = "#48BB78" if cc in ["Stable", "Tightening"] else ("#F59E0B" if cc == "—" else "#EF4444")
    chips.append(("Credit Conditions", cc, ccolor))

    return chips


def compute_orientation_sentence(regime, vol_assessment, breadth_assessment, rates_assessment):
    regime_word = regime.lower().replace("-", " ")
    vol_word = vol_assessment["stress_level"].lower()
    vol_trend = vol_assessment["trend"].lower()
    breadth_word = breadth_assessment["classification"].lower()
    credit_word = rates_assessment["credit_condition"].lower() if rates_assessment["credit_condition"] != "—" else "undetermined"
    rates_word = rates_assessment["rates_trend"].lower() if rates_assessment["rates_trend"] != "—" else "neutral"

    return (
        f"Market conditions are currently {regime_word}, with {breadth_word} cross-asset participation, "
        f"{vol_word} volatility stress ({vol_trend}), {rates_word} rate trajectory, "
        f"and {credit_word} credit conditions."
    )


def compute_what_changed(current_snapshot):
    yesterday = load_yesterday_snapshot()

    save_yesterday_snapshot(current_snapshot)

    if yesterday is None:
        return {"available": False, "message": "First run — establishing baseline. Changes will be tracked from the next refresh."}

    saved_date = yesterday.get("saved_date", "")
    changes = []

    y_regime = yesterday.get("regime", "")
    c_regime = current_snapshot.get("regime", "")
    if y_regime and c_regime and y_regime != c_regime:
        changes.append(f"Regime shifted from {y_regime} to {c_regime}")

    y_stress = yesterday.get("stress_level", "")
    c_stress = current_snapshot.get("stress_level", "")
    if y_stress and c_stress and y_stress != c_stress:
        changes.append(f"Volatility stress probability moved from {y_stress} to {c_stress}")

    y_breadth = yesterday.get("breadth", "")
    c_breadth = current_snapshot.get("breadth", "")
    if y_breadth and c_breadth and y_breadth != c_breadth:
        changes.append(f"Market breadth changed from {y_breadth} to {c_breadth}")

    y_credit = yesterday.get("credit", "")
    c_credit = current_snapshot.get("credit", "")
    if y_credit and c_credit and y_credit != c_credit:
        changes.append(f"Credit conditions changed from {y_credit} to {c_credit}")

    y_rates = yesterday.get("rates", "")
    c_rates = current_snapshot.get("rates", "")
    if y_rates and c_rates and y_rates != c_rates:
        changes.append(f"Rates regime changed from {y_rates} to {c_rates}")

    if not changes:
        changes.append("No material regime or stress probability changes detected since last snapshot.")

    return {"available": True, "saved_date": saved_date, "changes": changes}


def compute_regime_structure(prices, regime_label, vol_assess, breadth_assess, rates_assess):
    spy = prices.get("SPY")
    qqq = prices.get("QQQ")
    iwm = prices.get("IWM")

    regime_stability = "Stable"
    regime_stability_note = "Current market regime has been consistent across timeframes."
    if spy is not None:
        r30 = compute_returns(spy, 30)
        r90 = compute_returns(spy, 90)
        r365 = compute_returns(spy, 365)
        if r30 is not None and r90 is not None and r365 is not None:
            signs = [1 if r > 0.02 else (-1 if r < -0.02 else 0) for r in [r30, r90, r365]]
            unique_signs = len(set(signs))
            if unique_signs == 1:
                regime_stability = "Stable"
                regime_stability_note = "Directional signals are consistent across all measured horizons."
            elif unique_signs == 2:
                regime_stability = "Transitional"
                regime_stability_note = "Mixed directional signals suggest a regime transition may be underway."
            else:
                regime_stability = "Unstable"
                regime_stability_note = "Conflicting signals across horizons indicate regime instability."

    trend_persistence = "Moderate"
    trend_persistence_note = "Trend strength is at moderate levels."
    if spy is not None:
        slope_30 = compute_slope(spy, 30)
        slope_90 = compute_slope(spy, 90)
        if slope_30 is not None and slope_90 is not None:
            avg_slope = (abs(slope_30) + abs(slope_90)) / 2
            if avg_slope > 0.0015:
                trend_persistence = "Strong"
                trend_persistence_note = "Price slopes show strong directional persistence across horizons."
            elif avg_slope > 0.0005:
                trend_persistence = "Moderate"
                trend_persistence_note = "Modest trend persistence detected; directional conviction is not extreme."
            else:
                trend_persistence = "Weak"
                trend_persistence_note = "Flat or choppy slopes suggest weak trend persistence."

    vol_regime = vol_assess.get("regime", "Neutral")
    vol_regime_note = {
        "Compression": "Realized volatility is compressed, often preceding directional moves.",
        "Expansion": "Volatility is expanding, consistent with active repricing or stress.",
        "Neutral": "Volatility levels are within normal historical bands.",
    }.get(vol_regime, "Volatility regime data unavailable.")

    macro_alignment = "Mixed"
    macro_alignment_note = "Macro signal alignment is partial."
    eq_up = spy is not None and compute_returns(spy, 30) is not None and compute_returns(spy, 30) > 0
    rates_tightening = rates_assess.get("rates_trend", "").lower() in ["rising", "tightening"]
    credit_ok = rates_assess.get("credit_condition", "").lower() in ["normal", "tight", "stable"]
    breadth_broad = breadth_assess.get("classification", "") == "Broad"
    alignment_count = sum([eq_up, not rates_tightening, credit_ok, breadth_broad])
    if alignment_count >= 3:
        macro_alignment = "Aligned"
        macro_alignment_note = "Equity, rates, credit, and breadth signals are broadly aligned."
    elif alignment_count >= 2:
        macro_alignment = "Mixed"
        macro_alignment_note = "Some macro signals are in agreement while others diverge."
    else:
        macro_alignment = "Diverging"
        macro_alignment_note = "Key macro indicators are sending conflicting signals."

    return {
        "regime_stability": regime_stability,
        "regime_stability_note": regime_stability_note,
        "trend_persistence": trend_persistence,
        "trend_persistence_note": trend_persistence_note,
        "volatility_regime": vol_regime,
        "volatility_regime_note": vol_regime_note,
        "macro_alignment": macro_alignment,
        "macro_alignment_note": macro_alignment_note,
    }


def compute_directional_agreement(prices):
    spy = prices.get("SPY")
    if spy is None:
        return {
            "agreement": "Data unavailable",
            "orientation": "Directional agreement data is currently unavailable.",
        }

    r30 = compute_returns(spy, 30)
    r90 = compute_returns(spy, 90)
    r365 = compute_returns(spy, 365)

    if r30 is None or r90 is None or r365 is None:
        return {
            "agreement": "Data unavailable",
            "orientation": "Insufficient price history for cross-horizon agreement assessment.",
        }

    dirs = []
    for r in [r30, r90, r365]:
        if r > 0.02:
            dirs.append("up")
        elif r < -0.02:
            dirs.append("down")
        else:
            dirs.append("flat")

    unique = set(dirs)
    if len(unique) == 1:
        if dirs[0] == "up":
            agreement = "Strong"
            orientation = "Short-, intermediate-, and long-term directional trends are aligned to the upside."
        elif dirs[0] == "down":
            agreement = "Strong"
            orientation = "All measured horizons show downward directional alignment."
        else:
            agreement = "Moderate"
            orientation = "Directional signals are flat across all horizons, indicating a range-bound market."
    elif len(unique) == 2:
        agreement = "Moderate"
        labels = {"up": "positive", "down": "negative", "flat": "neutral"}
        orientation = f"Short-term trend is {labels[dirs[0]]}, intermediate-term is {labels[dirs[1]]}, while longer-term direction is {labels[dirs[2]]}."
    else:
        agreement = "Conflicted"
        orientation = "Directional signals are conflicted across horizons, with each timeframe showing a different trend."

    return {
        "agreement": agreement,
        "orientation": orientation,
    }


def compute_decision_implications(regime_label, vol_assess, breadth_assess, rates_assess):
    portfolio_implications = []
    strategy_implications = []
    decision_implications = []

    stress = vol_assess.get("stress_level", "Low")
    vol_regime = vol_assess.get("regime", "Neutral")
    breadth = breadth_assess.get("classification", "Mixed")
    rates = rates_assess.get("rates_trend", "Stable")
    credit = rates_assess.get("credit_condition", "Normal")

    if regime_label in ["Recovery", "Bull"]:
        portfolio_implications.append("Current regime historically favors equity participation and trend-following strategies.")
    elif regime_label in ["Correction", "Bear"]:
        portfolio_implications.append("Current regime is typically associated with defensive positioning and reduced risk appetite.")
    else:
        portfolio_implications.append("The regime environment is transitional, calling for balanced exposure and monitoring.")

    if breadth == "Broad":
        portfolio_implications.append("Broad market participation supports diversified equity exposure.")
    elif breadth == "Narrow":
        portfolio_implications.append("Narrow participation suggests concentration risk in leadership names.")
    else:
        portfolio_implications.append("Mixed breadth indicates uneven participation across market segments.")

    if vol_regime == "Compression":
        strategy_implications.append("Compressed volatility environments have historically preceded directional moves.")
    elif vol_regime == "Expansion":
        strategy_implications.append("Expanding volatility may challenge momentum and trend-following approaches.")
    else:
        strategy_implications.append("Volatility is within normal ranges, supporting standard strategy execution.")

    if stress == "High":
        strategy_implications.append("Elevated stress levels warrant heightened awareness across active strategies.")
    elif stress == "Low":
        strategy_implications.append("Low stress conditions provide a stable backdrop for strategy implementation.")
    else:
        strategy_implications.append("Moderate stress levels suggest a watchful but not alarmed posture.")

    if rates in ["Rising", "Tightening"]:
        decision_implications.append("Rising rate environment may influence duration-sensitive and growth-oriented decisions.")
    elif rates in ["Falling", "Easing"]:
        decision_implications.append("Falling rates environment historically supports risk asset revaluation.")
    else:
        decision_implications.append("Rates environment is stable, with no immediate directional pressure on decisions.")

    if credit in ["Stress", "Widening"]:
        decision_implications.append("Credit conditions show signs of stress, which may inform risk management timing.")
    else:
        decision_implications.append("Credit conditions are within normal ranges and do not currently signal elevated risk.")

    return {
        "portfolio": portfolio_implications,
        "strategy": strategy_implications,
        "decision": decision_implications,
    }


def compute_structural_signals(prices, vol_assess, rates_assess):
    signals = []

    spy = prices.get("SPY")
    qqq = prices.get("QQQ")
    iwm = prices.get("IWM")

    eq_concentration = "Neutral"
    eq_concentration_note = "Leadership concentration data unavailable."
    if spy is not None and qqq is not None and iwm is not None:
        r_spy = compute_returns(spy, 90)
        r_qqq = compute_returns(qqq, 90)
        r_iwm = compute_returns(iwm, 90)
        if r_spy is not None and r_qqq is not None and r_iwm is not None:
            spread = abs(r_qqq - r_iwm)
            if spread > 0.12:
                eq_concentration = "High"
                eq_concentration_note = f"Large-cap tech is outperforming small-caps by {spread*100:.0f}pp over 90 days, indicating concentrated leadership."
            elif spread > 0.05:
                eq_concentration = "Moderate"
                eq_concentration_note = f"Some divergence between large-cap tech and small-caps ({spread*100:.0f}pp), but not extreme."
            else:
                eq_concentration = "Low"
                eq_concentration_note = "Equity returns are relatively dispersed across market cap segments."
    signals.append(("Equity Leadership Concentration", eq_concentration, eq_concentration_note))

    from helpers.market_data import SECTOR_TICKERS
    sector_rets = []
    for ticker in SECTOR_TICKERS:
        p = prices.get(ticker)
        if p is not None:
            r = compute_returns(p, 30)
            if r is not None:
                sector_rets.append(r)
    if len(sector_rets) >= 4:
        dispersion = max(sector_rets) - min(sector_rets)
        if dispersion > 0.10:
            sec_disp = "High"
            sec_disp_note = f"Sector return dispersion is {dispersion*100:.0f}pp over 30 days, indicating differentiated sector performance."
        elif dispersion > 0.04:
            sec_disp = "Moderate"
            sec_disp_note = f"Sector returns show moderate dispersion ({dispersion*100:.0f}pp) over 30 days."
        else:
            sec_disp = "Low"
            sec_disp_note = "Sector performance is tightly clustered, indicating broad market consensus."
    else:
        sec_disp = "Data unavailable"
        sec_disp_note = "Insufficient sector data for dispersion analysis."
    signals.append(("Sector Dispersion", sec_disp, sec_disp_note))

    credit_cond = rates_assess.get("credit_condition", "Normal")
    credit_note = rates_assess.get("credit_note", "")
    if credit_cond in ["Stress", "Widening"]:
        cs_label = "Widening"
        cs_note = credit_note if credit_note else "Credit spreads are widening, indicating increased risk perception."
    elif credit_cond in ["Tight", "Narrowing"]:
        cs_label = "Tightening"
        cs_note = credit_note if credit_note else "Credit spreads are compressing, reflecting risk appetite."
    else:
        cs_label = "Stable"
        cs_note = credit_note if credit_note else "Credit spread direction is stable with no material movement."
    signals.append(("Credit Spread Direction", cs_label, cs_note))

    vol_regime = vol_assess.get("regime", "Neutral")
    vol_trend = vol_assess.get("trend", "Stable")
    if vol_regime == "Compression" and vol_trend in ["Subsiding", "Stable"]:
        vts_label = "Contango"
        vts_note = "Volatility structure is in contango — near-term vol is below longer-term, typical of calm markets."
    elif vol_regime == "Expansion" and vol_trend == "Rising":
        vts_label = "Backwardation"
        vts_note = "Volatility structure suggests backwardation — near-term stress exceeds longer-term expectations."
    else:
        vts_label = "Neutral"
        vts_note = "Volatility term structure is not showing a clear directional skew."
    signals.append(("Volatility Term Structure", vts_label, vts_note))

    tlt = prices.get("TLT")
    hyg = prices.get("HYG")
    gld = prices.get("GLD")
    divergence_count = 0
    if spy is not None and tlt is not None:
        r_spy30 = compute_returns(spy, 30)
        r_tlt30 = compute_returns(tlt, 30)
        if r_spy30 is not None and r_tlt30 is not None:
            if (r_spy30 > 0.01 and r_tlt30 > 0.01) or (r_spy30 < -0.01 and r_tlt30 < -0.01):
                divergence_count += 1
    if spy is not None and gld is not None:
        r_spy30g = compute_returns(spy, 30)
        r_gld30 = compute_returns(gld, 30)
        if r_spy30g is not None and r_gld30 is not None:
            if (r_spy30g > 0.02 and r_gld30 > 0.02):
                divergence_count += 1
    if divergence_count >= 2:
        ca_div = "Elevated"
        ca_div_note = "Multiple asset classes are moving in the same direction, which may indicate macro-driven flows."
    elif divergence_count == 1:
        ca_div = "Moderate"
        ca_div_note = "Some cross-asset correlation is present but not pervasive."
    else:
        ca_div = "Low"
        ca_div_note = "Asset classes are behaving independently, suggesting differentiated drivers."
    signals.append(("Cross-Asset Divergence", ca_div, ca_div_note))

    return signals
