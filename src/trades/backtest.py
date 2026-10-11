"""Trade generation via simple moving average crossover and breakout strategies.

This module acts as a stand-in for historical trading logs since none were provided.
It backtests parameterized strategies to generate a universe of trades for the model.
"""
import itertools
import json
import uuid
import pandas as pd
import yfinance as yf

from src.logging_utils import get_logger
from src.db.database import write_df

log = get_logger(__name__)

def fetch_data(symbols: list[str], lookback_days: int) -> pd.DataFrame:
    """Fetch daily price data for backtesting."""
    end = pd.Timestamp.now(tz="UTC")
    start = end - pd.Timedelta(days=lookback_days)
    
    dfs = []
    for sym in symbols:
        log.info(f"Downloading {sym}...")
        df = yf.download(sym, start=start, end=end, progress=False)
        if df.empty:
            continue
            
        df = df.reset_index()
        if df['Date'].dt.tz is None:
            df['Date'] = df['Date'].dt.tz_localize('UTC')
        else:
            df['Date'] = df['Date'].dt.tz_convert('UTC')
            
        df = df.rename(columns={"Date": "ts"})
        df['symbol'] = sym
        df = df[['ts', 'symbol', 'Open', 'High', 'Low', 'Close']]
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            
        dfs.append(df)
        
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True).sort_values(['symbol', 'ts'])

def compute_ma_crossover(df: pd.DataFrame, fast: int, slow: int) -> pd.DataFrame:
    """Generate trade signals for an MA crossover."""
    res = pd.DataFrame(index=df.index)
    res['fast_ma'] = df['Close'].rolling(fast).mean()
    res['slow_ma'] = df['Close'].rolling(slow).mean()
    res['signal'] = 0
    res.loc[res['fast_ma'] > res['slow_ma'], 'signal'] = 1
    res.loc[res['fast_ma'] < res['slow_ma'], 'signal'] = -1
    return res

def run_trades(cfg, conn) -> None:
    """Run backtests or load external trades."""
    raw_path = cfg.data_raw / "trades.csv"
    if raw_path.exists():
        log.info(f"Loading external trades from {raw_path}...")
        df_trades = pd.read_csv(raw_path)
        conn.execute("DELETE FROM trades")
        conn.commit()
        write_df(conn, df_trades, "trades")
        log.info(f"Ingested {len(df_trades)} external trades.")
        return

    tc = cfg["trades"]
    # Existing simulation logic...

    
    df_px = fetch_data(tc["symbols"], tc["price_history_days"])
    if df_px.empty:
        log.error("Failed to fetch price data.")
        return
        
    all_trades = []
    
    # 1. MA Crossover Family
    ma_cfg = tc["families"]["ma_crossover"]
    ma_grid = list(itertools.product(ma_cfg["fast"], ma_cfg["slow"]))
    
    for fast_window, slow_window in ma_grid:
        if fast_window >= slow_window:
            continue
            
        perm_id = f"MA_{fast_window}_{slow_window}"
        params_json = json.dumps({"fast": fast_window, "slow": slow_window, "type": "ma_crossover"})
        
        for sym, group in df_px.groupby("symbol"):
            signals = compute_ma_crossover(group, fast_window, slow_window)
            
            pos = 0
            entry_px = 0.0
            entry_time = None
            bars_held = 0
            
            for i in range(1, len(group)):
                prev_sig = signals['signal'].iloc[i-1]
                curr_sig = signals['signal'].iloc[i]
                
                # Check exit
                if pos != 0 and prev_sig != pos:
                    exit_px = float(group['Close'].iloc[i])
                    exit_time = group['ts'].iloc[i].isoformat()
                    pnl_pct = (exit_px - entry_px) / entry_px if pos > 0 else (entry_px - exit_px) / entry_px
                    pnl_abs = pnl_pct * (tc["initial_capital"] * tc["risk_per_trade"])
                    
                    all_trades.append({
                        "trade_id": str(uuid.uuid4()),
                        "permutation_id": perm_id,
                        "family": "ma_crossover",
                        "symbol": sym,
                        "side": "LONG" if pos > 0 else "SHORT",
                        "entry_time": entry_time,
                        "exit_time": exit_time,
                        "entry_px": entry_px,
                        "exit_px": exit_px,
                        "qty": tc["initial_capital"] * tc["risk_per_trade"] / entry_px,
                        "pnl": pnl_abs,
                        "pnl_pct": pnl_pct,
                        "bars_held": bars_held,
                        "exit_reason": "signal_flip",
                        "params_json": params_json
                    })
                    
                    pos = 0
                    bars_held = 0
                
                # Check entry
                if pos == 0 and curr_sig != 0:
                    pos = curr_sig
                    entry_px = float(group['Open'].iloc[i])
                    entry_time = group['ts'].iloc[i].isoformat()
                    bars_held = 1
                elif pos != 0:
                    bars_held += 1

    # 2. Breakout / Donchian Family
    bo_cfg = tc["families"].get("breakout", {})
    if bo_cfg:
        lookbacks = bo_cfg.get("lookback", [20, 55])
        stop_pcts = bo_cfg.get("stop_pct", [0.01, 0.02])
        tp_pcts = bo_cfg.get("take_profit_pct", [0.02, 0.04])
        bo_grid = list(itertools.product(lookbacks, stop_pcts, tp_pcts))
        
        for lb, stop_pct, tp_pct in bo_grid:
            perm_id = f"BO_{lb}_stop{int(stop_pct*100)}_tp{int(tp_pct*100)}"
            params_json = json.dumps({"lookback": lb, "stop_pct": stop_pct, "take_profit_pct": tp_pct, "type": "breakout"})
            
            for sym, group in df_px.groupby("symbol"):
                high_roll = group['High'].shift(1).rolling(lb).max()
                low_roll = group['Low'].shift(1).rolling(lb).min()
                
                pos = 0
                entry_px = 0.0
                entry_time = None
                bars_held = 0
                
                for i in range(lb + 1, len(group)):
                    curr_open = float(group['Open'].iloc[i])
                    curr_high = float(group['High'].iloc[i])
                    curr_low = float(group['Low'].iloc[i])
                    curr_close = float(group['Close'].iloc[i])
                    curr_ts = group['ts'].iloc[i].isoformat()
                    
                    # Manage open position
                    if pos != 0:
                        bars_held += 1
                        unrealized_return = (curr_close - entry_px) / entry_px if pos > 0 else (entry_px - curr_close) / entry_px
                        exit_reason = None
                        
                        if unrealized_return <= -stop_pct:
                            exit_reason = "stop_loss"
                        elif unrealized_return >= tp_pct:
                            exit_reason = "take_profit"
                        elif bars_held >= lb:
                            exit_reason = "time_exit"
                            
                        if exit_reason:
                            exit_px = curr_close
                            pnl_pct = (exit_px - entry_px) / entry_px if pos > 0 else (entry_px - exit_px) / entry_px
                            pnl_abs = pnl_pct * (tc["initial_capital"] * tc["risk_per_trade"])
                            
                            all_trades.append({
                                "trade_id": str(uuid.uuid4()),
                                "permutation_id": perm_id,
                                "family": "breakout",
                                "symbol": sym,
                                "side": "LONG" if pos > 0 else "SHORT",
                                "entry_time": entry_time,
                                "exit_time": curr_ts,
                                "entry_px": entry_px,
                                "exit_px": exit_px,
                                "qty": tc["initial_capital"] * tc["risk_per_trade"] / entry_px,
                                "pnl": pnl_abs,
                                "pnl_pct": pnl_pct,
                                "bars_held": bars_held,
                                "exit_reason": exit_reason,
                                "params_json": params_json
                            })
                            pos = 0
                            bars_held = 0
                            continue

                    # Signal entries if flat
                    if pos == 0:
                        h_prev = high_roll.iloc[i]
                        l_prev = low_roll.iloc[i]
                        if pd.notna(h_prev) and curr_close > h_prev:
                            pos = 1
                            entry_px = curr_open
                            entry_time = curr_ts
                            bars_held = 1
                        elif pd.notna(l_prev) and curr_close < l_prev:
                            pos = -1
                            entry_px = curr_open
                            entry_time = curr_ts
                            bars_held = 1

    df_trades = pd.DataFrame(all_trades)
    conn.execute("DELETE FROM trades")
    conn.commit()
    write_df(conn, df_trades, "trades")
    log.info(f"Generated {len(df_trades)} trades across MA & Breakout permutations.")
