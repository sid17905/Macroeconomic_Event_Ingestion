"""Feature engineering: combine trades, events, and news into ML features.

Strict rule: Every feature computed here must only use info available
strictly at or before `period_start`.
"""
import pandas as pd
import numpy as np

from src.logging_utils import get_logger
from src.db.database import read_df, write_df

log = get_logger(__name__)


def build_macro_features(df_events: pd.DataFrame, period_start: pd.Timestamp, cfg) -> dict:
    """Compute macro features as-of period_start."""
    # Filter strictly to events known before period_start
    known = df_events[df_events["ts_utc"] < period_start].copy()
    if known.empty:
        return {}
        
    feats = {}
    
    # 1. Distance to last event
    last_event = known.iloc[-1]
    feats["hours_since_last_event"] = (period_start - last_event["ts_utc"]).total_seconds() / 3600
    feats["last_event_type_" + last_event["event_type"]] = 1.0
    
    # 2. Surprise standardisation (per event type, rolling)
    # Surprise = actual - forecast (if both available)
    known["surprise"] = known["actual"] - known["forecast"]
    
    # Rolling averages by type
    for etype in cfg["macro"]["event_types"].keys():
        etype_events = known[known["event_type"] == etype].dropna(subset=["surprise"])
        
        if len(etype_events) >= 1:
            feats[f"{etype}_last_surprise"] = etype_events.iloc[-1]["surprise"]
            
        window = cfg["features"].get("rolling_surprise_window", 3)
        if len(etype_events) >= window:
            mean = etype_events["surprise"].tail(window).mean()
            std = etype_events["surprise"].tail(window).std() or 1.0
            feats[f"{etype}_roll{window}_zscore"] = (etype_events.iloc[-1]["surprise"] - mean) / std

    # Notice: we DO NOT use future events (e.g. time to next event) unless the calendar
    # schedule was known in advance. In a real system we'd segregate schedule knowledge
    # from value knowledge. For simplicity/safety, we stick to backward-looking only here.
    
    return feats


def build_sentiment_features(df_news: pd.DataFrame, period_start: pd.Timestamp, cfg) -> dict:
    """Aggregate news sentiment prior to period_start."""
    known = df_news[df_news["published_utc"] < period_start]
    window_days = cfg["features"].get("sentiment_roll_days", 7)
    window_start = period_start - pd.Timedelta(days=window_days)
    
    recent = known[known["published_utc"] >= window_start]
    
    if recent.empty:
        return {"sentiment_recent_avg": 0.0, "sentiment_recent_count": 0}
        
    return {
        "sentiment_recent_avg": recent["sentiment"].mean(),
        "sentiment_recent_count": float(len(recent))
    }


def compute_targets(period_trades: pd.DataFrame) -> tuple[float, float, int]:
    """Compute realized PnL out of the trades that occurred IN the period."""
    if period_trades.empty:
        return (0.0, 0.0, 0)
        
    pnl = period_trades["pnl"].sum()
    n_trades = len(period_trades)
    winrate = len(period_trades[period_trades["pnl"] > 0]) / n_trades if n_trades > 0 else 0.0
    
    return (pnl, winrate, n_trades)


def run_features(cfg, conn) -> None:
    """Orchestrate feature engineering and save to features table."""
    log.info("Loading raw trades, events, and news...")
    df_trades = read_df(conn, "SELECT * FROM trades")
    if df_trades.empty:
        log.warning("No trades found. Run stage 2 first.")
        return
        
    df_trades["entry_time"] = pd.to_datetime(df_trades["entry_time"], utc=True)
    df_trades["exit_time"] = pd.to_datetime(df_trades["exit_time"], utc=True)
    
    df_events = read_df(conn, "SELECT * FROM macro_events")
    if not df_events.empty:
        df_events["ts_utc"] = pd.to_datetime(df_events["ts_utc"], utc=True)
        df_events = df_events.sort_values("ts_utc")
        
    df_news = read_df(conn, "SELECT * FROM news")
    if not df_news.empty:
        df_news["published_utc"] = pd.to_datetime(df_news["published_utc"], utc=True)
        df_news = df_news.sort_values("published_utc")

    period_str = cfg["features"]["period"]  # e.g., 'W'
    min_date = df_trades["entry_time"].min().floor('D')
    max_date = df_trades["entry_time"].max().ceil('D')
    
    # Create periods
    periods = pd.date_range(start=min_date, end=max_date, freq=period_str, tz="UTC")
    
    all_features = []
    
    for perm_id, group in df_trades.groupby("permutation_id"):
        family = group["family"].iloc[0]
        params_json = group["params_json"].iloc[0]
        
        # Track past performance (we only have access to period i-1 when predicting period i)
        past_pnl_history = []
        
        for i in range(len(periods) - 1):
            period_start = periods[i]
            period_end = periods[i+1] # exclusively
            
            # 1. Base Strategy Features (from params)
            import json
            params = json.loads(params_json)
            feats = {f"param_{k}": float(v) for k, v in params.items() if isinstance(v, (int, float))}
            
            # 2. Lagged Performance (rolling over past periods)
            roll_win = cfg["features"].get("rolling_perf_window", 4)
            recent_hist = past_pnl_history[-roll_win:] if len(past_pnl_history) > 0 else []
            feats["lag_pnl_sum"] = sum(recent_hist) if recent_hist else 0.0
            feats["lag_pnl_mean"] = sum(recent_hist)/len(recent_hist) if recent_hist else 0.0
            
            # 3. Time Features
            feats["month"] = float(period_start.month)
            
            # 4. Macro Features
            if not df_events.empty:
                feats.update(build_macro_features(df_events, period_start, cfg))
                
            # 5. Sentiment Features
            if not df_news.empty:
                feats.update(build_sentiment_features(df_news, period_start, cfg))
            
            # TARGET computation: trades that EXIT within this period
            # We assign PnL to the period it was realised (exit_time) 
            period_trades = group[(group["exit_time"] >= period_start) & (group["exit_time"] < period_end)]
            pnl, winrate, n_trades = compute_targets(period_trades)
            
            # Save for next iteration's lag features
            past_pnl_history.append(pnl)
            
            all_features.append({
                "permutation_id": perm_id,
                "period_start": period_start.isoformat(),
                "family": family,
                "feature_json": json.dumps(feats),
                "y_pnl": float(pnl),
                "y_winrate": float(winrate),
                "n_trades": int(n_trades)
            })

    df_feat = pd.DataFrame(all_features)
    conn.execute("DELETE FROM features")
    write_df(conn, df_feat, "features")
    log.info(f"Generated {len(df_feat)} feature vectors across {len(periods)-1} periods.")
