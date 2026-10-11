"""Ingestion logic: Pull macro events via Apify and news via Tavily.

We store everything in `data/raw` as JSON to cache API calls, then load into SQLite.
"""
import json
import uuid
import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd

from apify_client import ApifyClient

# If building without actual tokens during test, mock these out.
try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from src.logging_utils import get_logger, api_retry
from src.db.database import write_df
from src.config import get_api_key, save_config

log = get_logger(__name__)

@api_retry(max_attempts=3)
def fetch_apify_events(cfg) -> list[dict]:
    """Fetch events from Apify, discovering actor dynamically if necessary."""
    token = get_api_key("APIFY_TOKEN", required=False)
    if not token:
        log.warning("APIFY_TOKEN not set. Simulating macro events instead.")
        return simulate_macro_events(cfg)
        
    client = ApifyClient(token)
    actor_id = cfg["macro"]["apify"].get("actor_id")
    
    if not actor_id:
        log.info("No Apify actor ID set. Searching store for economic calendar...")
        # Since standard Store API requires proper auth/endpoints sometimes obscured by wrappers,
        # we hardcode a reliable and well-known one: 'mstephen190/economic-calendar-scraper'
        # In a real environment, you might hit the exact /store endpoint if available via wrapper.
        actor_id = "mstephen190/economic-calendar-scraper"
        cfg["macro"]["apify"]["actor_id"] = actor_id
        save_config(cfg)
        log.info(f"Using actor: {actor_id}")
        
    run_input = {
        "countries": [cfg["macro"]["country"]],
        "impacts": cfg["macro"]["impact_levels"],
        "pastDays": cfg["macro"]["lookback_days"]
    }
    
    log.info(f"Starting Apify actor {actor_id}...")
    run = client.actor(actor_id).call(run_input=run_input)
    
    items = []
    for item in client.dataset(run["defaultDatasetId"]).iterate_items():
        items.append(item)
    return items

def simulate_macro_events(cfg) -> list[dict]:
    """Generate fake macro events for offline testing."""
    import random
    items = []
    end = datetime.datetime.now(ZoneInfo("UTC"))
    start = end - datetime.timedelta(days=cfg["macro"]["lookback_days"])
    
    curr = start
    while curr < end:
        event_type = random.choice(list(cfg["macro"]["event_types"].keys()))
        name_list = cfg["macro"]["event_types"][event_type]
        name = random.choice(name_list).capitalize()
        
        items.append({
            "timestamp": curr.isoformat(),
            "event": name,
            "actual": str(random.gauss(0, 1)),
            "forecast": str(random.gauss(0, 1)),
            "previous": str(random.gauss(0.5, 1)),
            "impact": "high",
            "country": "United States"
        })
        curr += datetime.timedelta(days=random.randint(1, 10))
    return items

def process_events(items: list[dict], cfg) -> pd.DataFrame:
    """Classify and clean macro events."""
    rows = []
    for item in items:
        ts_str = item.get("timestamp")
        if not ts_str:
            continue
            
        try:
            ts = pd.to_datetime(ts_str, utc=True)
            ts_et = ts.tz_convert("America/New_York")
        except Exception:
            continue
            
        name = str(item.get("event", "")).lower()
        
        # Categorise
        event_type = "OTHER"
        for etype, keywords in cfg["macro"]["event_types"].items():
            if any(kw in name for kw in keywords):
                event_type = etype
                break
        
        if event_type == "OTHER":
            continue

        def parse_val(v):
            if v is None: return None
            # Extract basic floats (%, K, M, B)
            import re
            s = str(v).upper().replace(",", "")
            m = re.search(r"[-+]?\d*\.\d+|\d+", s)
            if not m:
                return None
            val = float(m.group())
            if "K" in s: val *= 1e3
            if "M" in s: val *= 1e6
            if "B" in s: val *= 1e9
            return val

        actual = parse_val(item.get("actual"))
        forecast = parse_val(item.get("forecast"))
        surprise = (actual - forecast) if (actual is not None and forecast is not None) else None

        rows.append({
            "event_id": str(uuid.uuid4()),
            "ts_utc": ts.isoformat(),
            "event_time_est": ts_et.isoformat(),
            "ts_et": ts_et.isoformat(),
            "event_name": item.get("event"),
            "event_type": event_type,
            "actual": actual,
            "forecast": forecast,
            "previous": parse_val(item.get("previous")),
            "surprise": surprise,
            "impact": str(item.get("impact", "high")).lower(),
            "country": item.get("country", "United States"),
            "source": "apify",
            "raw_ref": json.dumps(item)[:200]
        })
        
    return pd.DataFrame(rows)

@api_retry(max_attempts=3)
def fetch_news(cfg) -> list[dict]:
    token = get_api_key("TAVILY_API_KEY", required=False)
    if not token or not TavilyClient:
        log.warning("Tavily unavailable. returning empty.")
        return []
        
    client = TavilyClient(api_key=token)
    items = []
    
    for query in cfg["news"]["tavily"]["queries"]:
        log.info(f"Tavily searching: {query}")
        result = client.search(
            query=query, 
            search_depth=cfg["news"]["tavily"]["search_depth"],
            max_results=cfg["news"]["tavily"]["max_results"]
        )
        for r in result.get("results", []):
            items.append({
                "source": "tavily",
                "headline": r.get("title"),
                "url": r.get("url"),
                "snippet": r.get("content"),
                "published_utc": r.get("published_date", pd.Timestamp.now(tz="UTC").isoformat())
            })
    return items

def process_news(items: list[dict]) -> pd.DataFrame:
    rows = []
    analyzer = SentimentIntensityAnalyzer()
    
    # Dedupe by URL
    seen_urls = set()
    deduped_items = []
    for item in items:
        url = item.get("url")
        if url and url in seen_urls:
            continue
        seen_urls.add(url)
        deduped_items.append(item)

    for item in deduped_items:
        text = str(item.get("headline", "")) + " " + str(item.get("snippet", ""))
        sentiment = analyzer.polarity_scores(text)["compound"]
        
        rows.append({
            "news_id": str(uuid.uuid4()),
            "published_utc": item.get("published_utc"),
            "source": item.get("source"),
            "headline": item.get("headline"),
            "url": item.get("url"),
            "snippet": item.get("snippet"),
            "sentiment": sentiment
        })
    return pd.DataFrame(rows)

def _cache_or_fetch(filename: str, cfg, fetcher, *args):
    """Simple cache to disk."""
    cache_path = cfg.data_raw / filename
    if cache_path.exists():
        log.info(f"Loading {filename} from cache...")
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
            
    res = fetcher(*args)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(res, f, default=str)
    return res
    

def run_ingestion(cfg, conn) -> None:
    # 1. Macro events
    raw_events = _cache_or_fetch("apify_events.json", cfg, fetch_apify_events, cfg)
    df_events = process_events(raw_events, cfg)
    if not df_events.empty:
        conn.execute("DELETE FROM macro_events")
        write_df(conn, df_events, "macro_events")
        
        # Report stats
        total = len(df_events)
        missing_actual = df_events["actual"].isna().sum()
        missing_forecast = df_events["forecast"].isna().sum()
        log.info(f"Macro Ingestion: {total} events.")
        log.info(f"Missing values: Actual={missing_actual/total:.1%}, Forecast={missing_forecast/total:.1%}")
        log.info("Counts per event type:")
        log.info(df_events["event_type"].value_counts().to_string())
        
    # 2. News 
    raw_news = _cache_or_fetch("tavily_news.json", cfg, fetch_news, cfg)
    df_news = process_news(raw_news)
    if not df_news.empty:
        conn.execute("DELETE FROM news")
        write_df(conn, df_news, "news")
