import feedparser
import json
import os
import re
import time
from datetime import datetime, timezone

CACHE_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "cache", "news_cache.json")
CACHE_TTL = 600

RSS_FEEDS = [
    {"name": "Yahoo Finance", "url": "https://finance.yahoo.com/news/rssindex"},
    {"name": "CNBC Markets", "url": "https://www.cnbc.com/id/100003114/device/rss/rss.html"},
    {"name": "Federal Reserve", "url": "https://www.federalreserve.gov/feeds/press_all.xml"},
    {"name": "U.S. Treasury", "url": "https://home.treasury.gov/system/files/136/TreasuryDeptPressReleases.xml"},
    {"name": "SEC Press", "url": "https://www.sec.gov/news/pressreleases.rss"},
    {"name": "Reuters Business", "url": "https://www.reutersagency.com/feed/?taxonomy=best-sectors&post_type=best"},
    {"name": "MarketWatch", "url": "https://feeds.marketwatch.com/marketwatch/topstories/"},
]

CATEGORY_KEYWORDS = {
    "Macro / Fed / Inflation": [
        "fed", "federal reserve", "fomc", "inflation", "cpi", "pce", "gdp", "jobs", "employment",
        "unemployment", "nonfarm", "payroll", "recession", "fiscal", "deficit", "interest rate",
        "monetary policy", "powell", "treasury", "economic", "macro", "stimulus",
    ],
    "Equity / Tech": [
        "stock", "equity", "nasdaq", "s&p", "dow", "earnings", "ipo", "tech", "apple", "google",
        "microsoft", "amazon", "nvidia", "ai", "semiconductor", "chip", "software", "growth stock",
        "mega cap", "market cap", "rally", "correction", "bull", "bear",
    ],
    "Credit / Rates": [
        "bond", "yield", "treasury", "credit", "spread", "high yield", "investment grade",
        "corporate bond", "fixed income", "duration", "curve", "2-year", "10-year", "rate cut",
        "rate hike", "basis point", "coupon",
    ],
    "Energy / Commodities": [
        "oil", "crude", "brent", "wti", "natural gas", "energy", "opec", "commodity", "gold",
        "silver", "copper", "metal", "mining", "coal", "lng", "refinery", "barrel",
    ],
    "Crypto": [
        "bitcoin", "btc", "ethereum", "eth", "crypto", "blockchain", "defi", "token", "coin",
        "stablecoin", "mining", "web3", "nft", "binance", "coinbase",
    ],
}


def _categorize_article(title, summary):
    text = (title + " " + summary).lower()
    scores = {}
    for category, keywords in CATEGORY_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in text)
        if count > 0:
            scores[category] = count
    if scores:
        return max(scores, key=scores.get)
    return "Macro / Fed / Inflation"


def _load_cache():
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "r") as f:
                cache = json.load(f)
            if time.time() - cache.get("timestamp", 0) < CACHE_TTL:
                return cache.get("articles", []), cache.get("timestamp", 0)
    except Exception:
        pass
    return None, None


def _save_cache(articles):
    try:
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        with open(CACHE_FILE, "w") as f:
            json.dump({"timestamp": time.time(), "articles": articles}, f)
    except Exception:
        pass


def _time_ago(published_parsed):
    try:
        if published_parsed:
            pub_time = datetime(*published_parsed[:6], tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            diff = now - pub_time
            seconds = int(diff.total_seconds())
            if seconds < 60:
                return "just now"
            elif seconds < 3600:
                mins = seconds // 60
                return f"{mins} min{'s' if mins != 1 else ''} ago"
            elif seconds < 86400:
                hours = seconds // 3600
                return f"{hours} hour{'s' if hours != 1 else ''} ago"
            else:
                days = seconds // 86400
                return f"{days} day{'s' if days != 1 else ''} ago"
    except Exception:
        pass
    return ""


def fetch_market_news(max_items=15):
    cached, cache_ts = _load_cache()
    if cached is not None:
        return cached[:max_items], cache_ts

    articles = []
    for feed_info in RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_info["url"])
            if feed.bozo and not feed.entries:
                continue
            for entry in feed.entries[:10]:
                title = entry.get("title", "").strip()
                link = entry.get("link", "")
                summary = entry.get("summary", "")
                if summary:
                    summary = re.sub(r'<[^>]+>', '', summary).strip()
                    if len(summary) > 200:
                        summary = summary[:197] + "..."

                published_parsed = entry.get("published_parsed")
                time_ago_str = _time_ago(published_parsed)

                pub_ts = 0
                if published_parsed:
                    try:
                        pub_ts = int(datetime(*published_parsed[:6], tzinfo=timezone.utc).timestamp())
                    except Exception:
                        pass

                if title:
                    category = _categorize_article(title, summary or "")
                    articles.append({
                        "title": title,
                        "source": feed_info["name"],
                        "link": link,
                        "time_ago": time_ago_str,
                        "summary": summary if summary and summary != title else "",
                        "pub_ts": pub_ts,
                        "category": category,
                    })
        except Exception:
            continue

    articles.sort(key=lambda x: x.get("pub_ts", 0), reverse=True)
    result = articles[:max_items * 2]

    if result:
        _save_cache(result)

    return result[:max_items], time.time()
