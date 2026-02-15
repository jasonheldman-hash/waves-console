import pandas as pd
import numpy as np
import os
from pathlib import Path
import streamlit as st


@st.cache_data(ttl=600)
def _load_prices():
    prices_path = Path("data/prices.csv")
    if not prices_path.exists():
        return None
    try:
        pdf = pd.read_csv(prices_path, parse_dates=["date"])
        pdf = pdf.sort_values(["ticker", "date"])
        return pdf
    except Exception:
        return None


def _compute_ticker_metrics(prices_df, tickers):
    metrics = {}
    for ticker in tickers:
        t_prices = prices_df[prices_df["ticker"] == ticker].sort_values("date")
        if len(t_prices) < 22:
            metrics[ticker] = {
                "ret_30d": np.nan,
                "momentum": np.nan,
                "vol_30d": np.nan,
            }
            continue

        closes = t_prices["close"].values
        log_rets = np.log(closes[1:] / closes[:-1])

        p_now = closes[-1]
        p_30 = closes[-min(22, len(closes))]
        ret_30d = (p_now / p_30) - 1

        recent_rets = log_rets[-22:] if len(log_rets) >= 22 else log_rets
        vol_30d = float(np.std(recent_rets) * np.sqrt(252)) if len(recent_rets) > 5 else np.nan

        if len(closes) >= 44:
            p_60 = closes[-min(44, len(closes))]
            ret_prior_30d = (p_30 / p_60) - 1
        else:
            ret_prior_30d = np.nan

        momentum = ret_30d - ret_prior_30d if not np.isnan(ret_prior_30d) else ret_30d

        metrics[ticker] = {
            "ret_30d": ret_30d,
            "momentum": momentum,
            "vol_30d": vol_30d,
        }
    return metrics


def _classify_structural_state(ret_30d, momentum, vol_30d):
    if np.isnan(ret_30d):
        return "Neutral"

    mom = momentum if not np.isnan(momentum) else 0.0
    vol = vol_30d if not np.isnan(vol_30d) else 0.0

    high_vol_threshold = 0.45

    if vol > high_vol_threshold and ret_30d < -0.08:
        return "Liquidity Stress"

    if vol > high_vol_threshold:
        return "Volatility Spike"

    if ret_30d < -0.05 and mom > 0:
        return "Failed Breakout"

    if ret_30d > 0.08 and mom > 0 and vol < high_vol_threshold:
        return "Breakout"

    if ret_30d > 0 and mom > 0:
        return "Structural Leader"

    if ret_30d < 0 and mom < 0:
        return "Deteriorating"

    return "Neutral"


def _resolve_wave_dir(wave_name):
    import re
    base = Path("data/waves")
    if not base.exists():
        return None
    direct = base / wave_name / "weights.csv"
    if direct.exists():
        return base / wave_name
    slug = re.sub(r"[^a-z0-9]+", "_", wave_name.lower()).strip("_")
    for d in base.iterdir():
        if d.is_dir():
            d_slug = re.sub(r"[^a-z0-9]+", "_", d.name.lower()).strip("_")
            if d_slug == slug or d_slug == slug + "_wave" or slug == d_slug + "_wave":
                if (d / "weights.csv").exists():
                    return d
    for d in base.iterdir():
        if d.is_dir() and (d / "weights.csv").exists():
            d_slug = re.sub(r"[^a-z0-9]+", "_", d.name.lower()).strip("_")
            if slug in d_slug or d_slug in slug:
                return d
    return None


def get_wave_diagnostics(wave_name, prices_df=None):
    try:
        wave_dir = _resolve_wave_dir(wave_name)
        if wave_dir is None:
            return None
        wave_path = wave_dir / "weights.csv"

        weights_df = pd.read_csv(wave_path)
        if weights_df.empty or "ticker" not in weights_df.columns or "weight" not in weights_df.columns:
            return None

        top_10 = weights_df.sort_values("weight", ascending=False).head(10).copy()
        top_10["External Link"] = top_10["ticker"].apply(
            lambda t: f"https://www.google.com/finance/quote/{t}"
        )

        if prices_df is None:
            prices_df = _load_prices()

        if prices_df is None or prices_df.empty:
            return {"top_10": top_10, "abnormal": "Insufficient data"}

        tickers = weights_df["ticker"].unique()
        ticker_metrics = _compute_ticker_metrics(prices_df, tickers)

        weights_df["return_30d"] = weights_df["ticker"].map(lambda t: ticker_metrics.get(t, {}).get("ret_30d", np.nan))
        weights_df["momentum"] = weights_df["ticker"].map(lambda t: ticker_metrics.get(t, {}).get("momentum", np.nan))
        weights_df["vol_30d"] = weights_df["ticker"].map(lambda t: ticker_metrics.get(t, {}).get("vol_30d", np.nan))

        valid_returns = weights_df.dropna(subset=["return_30d"]).copy()

        if valid_returns.empty:
            return {"top_10": top_10, "abnormal": "Insufficient data"}

        wave_avg_ret = valid_returns["return_30d"].mean()
        valid_returns["deviation"] = valid_returns["return_30d"] - wave_avg_ret

        valid_returns["Structural State"] = valid_returns.apply(
            lambda row: _classify_structural_state(
                row["return_30d"], row["momentum"], row["vol_30d"]
            ),
            axis=1,
        )

        valid_returns["External Link"] = valid_returns["ticker"].apply(
            lambda t: f"https://www.google.com/finance/quote/{t}"
        )

        outliers = valid_returns.sort_values("deviation", ascending=False)
        pos_outliers = outliers.head(5).copy()
        neg_outliers = outliers.tail(5).copy()

        return {
            "top_10": top_10,
            "pos_outliers": pos_outliers,
            "neg_outliers": neg_outliers,
            "wave_avg_ret": wave_avg_ret,
        }
    except Exception as e:
        print(f"Error in portfolio state diagnostics: {e}")
        return None
