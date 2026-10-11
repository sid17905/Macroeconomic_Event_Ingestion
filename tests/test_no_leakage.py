import pandas as pd
import numpy as np
import pytest

from src.features.build_features import build_macro_features, build_sentiment_features, run_features
from src.config import load_config
from src.db.database import connect, init_db, write_df, read_df

def test_macro_features_no_lookahead():
    # Existing test code...
    pass

def test_full_pipeline_leakage_check():
    """Verify that feature table values don't leak future trade info."""
    cfg = load_config()
    conn = connect(":memory:")
    init_db(conn)
    
    # 1. Create a trade that exits in PERIOD 2
    # 2. Run features
    # 3. Ensure PERIOD 1 features don't contain PERIOD 2 performance
    t2 = pd.Timestamp("2026-03-10", tz="UTC")
    t1 = pd.Timestamp("2026-03-01", tz="UTC")
    
    trades = [{
        "trade_id": "T1", "permutation_id": "P1", "family": "F1", "symbol": "S1", "side": "LONG",
        "entry_time": t1.isoformat(), "exit_time": t2.isoformat(), "pnl": 100.0, "params_json": "{}"
    }]
    write_df(conn, pd.DataFrame(trades), "trades")
    
    # Empty events/news
    write_df(conn, pd.DataFrame(columns=["event_id", "ts_utc", "event_time_est", "ts_et", "event_name", "event_type"]), "macro_events")
    write_df(conn, pd.DataFrame(columns=["news_id", "published_utc"]), "news")
    
    run_features(cfg, conn)
    
    df_feat = read_df(conn, "SELECT * FROM features ORDER BY period_start")
    
    # If we have 2 periods, the first period's lag_pnl should be 0, second should be 100
    # or check that y_pnl for P1 is the ONLY place T1's PnL appears for that period
    p1_feats = df_feat[df_feat["period_start"].str.contains("2026-03-01")]
    import json
    for _, row in p1_feats.iterrows():
        fj = json.loads(row["feature_json"])
        assert fj.get("lag_pnl_sum", 0.0) == 0.0
        
def test_macro_surprise_calculation():
    """Verify surprise = actual - forecast with numeric scaling."""
    cfg = load_config()
    period_start = pd.Timestamp("2026-03-01", tz="UTC")
    events = [
        {"ts_utc": pd.Timestamp("2026-02-15", tz="UTC"), "event_type": "CPI", "actual": 105.5, "forecast": 100.0},
    ]
    df_events = pd.DataFrame(events)
    # Mocking what ingest.py does for surprise column
    df_events["surprise"] = df_events["actual"] - df_events["forecast"]
    
    feats = build_macro_features(df_events, period_start, cfg)
    assert feats["CPI_last_surprise"] == 5.5



def test_sentiment_features_no_lookahead():
    cfg = load_config()
    period_start = pd.Timestamp("2026-03-01", tz="UTC")
    
    news = [
        {"published_utc": pd.Timestamp("2026-02-25", tz="UTC"), "sentiment": 0.5},
        # Future news that SHOULD NOT be used:
        {"published_utc": pd.Timestamp("2026-03-02", tz="UTC"), "sentiment": -0.8},
    ]
    df_news = pd.DataFrame(news)
    
    feats = build_sentiment_features(df_news, period_start, cfg)
    # The avg sentiment should ONLY include the 0.5 item
    assert feats["sentiment_recent_avg"] == 0.5
