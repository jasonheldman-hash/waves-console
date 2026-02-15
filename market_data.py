import os
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "cache")
MARKET_DATA_CACHE = os.path.join(CACHE_DIR, "market_data_cache.json")
YESTERDAY_SNAPSHOT_FILE = os.path.join(CACHE_DIR, "mi_yesterday_snapshot.json")
CACHE_TTL = 900

BENCHMARK_TICKERS = {
    "SPY": "S&P 500",
    "QQQ": "Nasdaq 100",
    "IWM": "Small Caps",
    "EFA": "Intl Developed",
    "HYG": "High Yield",
    "LQD": "Investment Grade",
    "TLT": "Long Duration",
    "GLD": "Gold",
    "UUP": "US Dollar",
}

SECTOR_TICKERS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLV": "Healthcare",
    "XLI": "Industrials",
    "XLY": "Consumer Disc.",
    "XLP": "Consumer Staples",
    "XLU": "Utilities",
}

RATE_TICKERS = {
    "TLT": "20Y+ Treasury",
    "IEF": "7-10Y Treasury",
    "SHY": "1-3Y Treasury",
}

INDEX_TICKERS = {
    "^GSPC": "S&P 500",
    "^IXIC": "Nasdaq Composite",
    "^DJI": "Dow Jones Industrial Average",
    "^RUT": "Russell 2000",
}

YIELD_TICKERS = {
    "^TNX": "10-Year Treasury Yield",
}

ALL_TICKERS = list(set(list(BENCHMARK_TICKERS.keys()) + list(SECTOR_TICKERS.keys()) + list(RATE_TICKERS.keys()) + list(INDEX_TICKERS.keys()) + list(YIELD_TICKERS.keys()) + ["IEF", "SHY", "BTC-USD"]))


def _load_price_cache():
    try:
        if os.path.exists(MARKET_DATA_CACHE):
            with open(MARKET_DATA_CACHE, "r") as f:
                cache = json.load(f)
            if time.time() - cache.get("timestamp", 0) < CACHE_TTL:
                return cache.get("data", {})
    except Exception:
        pass
    return None


def _save_price_cache(data_dict):
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        serializable = {}
        for ticker, series_data in data_dict.items():
            if isinstance(series_data, dict):
                serializable[ticker] = series_data
            else:
                serializable[ticker] = {"dates": list(series_data.index.astype(str)), "values": list(series_data.values)}
        with open(MARKET_DATA_CACHE, "w") as f:
            json.dump({"timestamp": time.time(), "data": serializable}, f)
    except Exception:
        pass


def fetch_all_prices(tickers=None, lookback_days=400):
    if tickers is None:
        tickers = ALL_TICKERS

    cached = _load_price_cache()
    if cached is not None:
        result = {}
        for t in tickers:
            if t in cached:
                try:
                    d = cached[t]
                    result[t] = pd.Series(d["values"], index=pd.to_datetime(d["dates"]), name=t).sort_index()
                except Exception:
                    pass
        if len(result) >= len(tickers) * 0.6:
            return result

    result = {}
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days + 30)

    try:
        import yfinance as yf
        data = yf.download(tickers, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"), progress=False, threads=True)
        if data is not None and len(data) > 0:
            close = data["Close"] if "Close" in data.columns or (hasattr(data.columns, 'get_level_values') and "Close" in data.columns.get_level_values(0)) else data
            if isinstance(close, pd.Series):
                close = close.to_frame(name=tickers[0] if len(tickers) == 1 else "Close")
            for t in tickers:
                try:
                    if t in close.columns:
                        s = close[t].dropna()
                        if len(s) >= 20:
                            result[t] = s
                except Exception:
                    pass
    except Exception:
        pass

    if len(result) < len(tickers) * 0.3:
        for t in tickers:
            if t in result:
                continue
            try:
                import yfinance as yf
                single = yf.download(t, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"), progress=False)
                if single is not None and len(single) > 0:
                    close_raw = single["Close"]
                    if hasattr(close_raw, 'squeeze'):
                        s = close_raw.squeeze().dropna()
                    else:
                        s = close_raw.dropna()
                    if len(s) >= 20:
                        result[t] = s
            except Exception:
                continue

    if result:
        _save_price_cache(result)

    return result


def compute_returns(prices, days):
    if len(prices) < days + 1:
        return None
    recent = prices.iloc[-1]
    past = prices.iloc[-min(days + 1, len(prices))]
    if past == 0:
        return None
    return (recent - past) / past


def compute_slope(prices, days):
    if len(prices) < max(days, 20):
        return None
    subset = prices.tail(days)
    x = np.arange(len(subset))
    y = subset.values.astype(float)
    if np.std(y) == 0:
        return 0
    coeffs = np.polyfit(x, y, 1)
    return coeffs[0] / np.mean(y) * 100


def compute_realized_vol(prices, window=21):
    if len(prices) < window + 5:
        return None
    returns = prices.pct_change().dropna()
    vol = returns.tail(window).std() * np.sqrt(252)
    return vol


def compute_vol_of_vol(prices, window=63):
    if len(prices) < window + 10:
        return None
    returns = prices.pct_change().dropna()
    rolling_vol = returns.rolling(21).std() * np.sqrt(252)
    rolling_vol = rolling_vol.dropna()
    if len(rolling_vol) < 10:
        return None
    return rolling_vol.tail(min(window, len(rolling_vol))).std()


def compute_drawdown(prices, days=30):
    if len(prices) < days:
        return None
    subset = prices.tail(days)
    peak = subset.cummax()
    dd = (subset - peak) / peak
    return dd.min()


def compute_pct_up_days(prices, days=30):
    if len(prices) < days + 1:
        return None
    returns = prices.pct_change().dropna().tail(days)
    if len(returns) == 0:
        return None
    return (returns > 0).sum() / len(returns) * 100


def compute_above_ma(prices, ma_period=50):
    if len(prices) < ma_period + 5:
        return None
    ma = prices.rolling(ma_period).mean()
    current = prices.iloc[-1]
    ma_val = ma.iloc[-1]
    if pd.isna(ma_val) or ma_val == 0:
        return None
    return current > ma_val


def compute_relative_strength(prices_ticker, prices_benchmark, days=30):
    if prices_ticker is None or prices_benchmark is None:
        return None
    if len(prices_ticker) < days + 1 or len(prices_benchmark) < days + 1:
        return None
    ret_t = compute_returns(prices_ticker, days)
    ret_b = compute_returns(prices_benchmark, days)
    if ret_t is None or ret_b is None:
        return None
    return ret_t - ret_b


def load_yesterday_snapshot():
    try:
        if os.path.exists(YESTERDAY_SNAPSHOT_FILE):
            with open(YESTERDAY_SNAPSHOT_FILE, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return None


def save_yesterday_snapshot(snapshot_data):
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        snapshot_data["saved_date"] = datetime.now().strftime("%Y-%m-%d")
        with open(YESTERDAY_SNAPSHOT_FILE, "w") as f:
            json.dump(snapshot_data, f, indent=2)
    except Exception:
        pass
