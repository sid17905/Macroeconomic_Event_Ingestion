"""Evaluate out-of-sample execution on the predictions."""

import pandas as pd
import numpy as np

from src.logging_utils import get_logger
from src.db.database import read_df, write_df

log = get_logger(__name__)

def calc_drawdown(eq_curve: pd.Series) -> pd.Series:
    roll_max = eq_curve.cummax()
    return (eq_curve - roll_max) / roll_max


def run_evaluation(cfg, conn) -> None:
    """Compute backtest results per period and full-run metrics."""
    df_preds = read_df(conn, "SELECT * FROM predictions")
    if df_preds.empty:
        log.warning("No predictions. Run stage 4 first.")
        return
        
    df_preds["period_start"] = pd.to_datetime(df_preds["period_start"], utc=True)
    df_feats = read_df(conn, "SELECT permutation_id, period_start, y_pnl, n_trades FROM features")
    df_feats["period_start"] = pd.to_datetime(df_feats["period_start"], utc=True)
    
    # Merge exact realised trades metadata back just in case
    df = pd.merge(df_preds, df_feats, on=["permutation_id", "period_start"])
    
    # We only care about the rows we selected to trade (selected == 1)
    trades_taken = df[df["selected"] == 1]
    
    results = []
    run_metrics = []
    
    capital = cfg["trades"]["initial_capital"]
    ppy = cfg["evaluation"]["periods_per_year"]
    rfr = cfg["evaluation"]["risk_free_rate"]
    
    for (run_id, model), g in trades_taken.groupby(["run_id", "model"]):
        # Sort by period
        period_stats = g.groupby("period_start").agg(
            realized_pnl=("y_true", "sum"),
            n_trades=("n_trades", "sum")
        ).reset_index().sort_values("period_start")
        
        period_stats["realized_ret"] = period_stats["realized_pnl"] / capital
        
        # Save per-period backtest result
        for _, row in period_stats.iterrows():
            results.append({
                "run_id": run_id,
                "strategy": model,
                "period_start": row["period_start"].isoformat(),
                "realized_pnl": row["realized_pnl"],
                "realized_ret": row["realized_ret"],
                "n_trades": row["n_trades"]
            })
            
        # Calc overall metrics
        rets = period_stats["realized_ret"].values
        cum_pnl = period_stats["realized_pnl"].sum()
        
        if len(rets) < 2 or rets.std() == 0:
            sharpe = 0.0
            sortino = 0.0
        else:
            mean_ret = rets.mean() - (rfr / ppy)
            sharpe = (mean_ret / rets.std()) * np.sqrt(ppy)
            
            downside_rets = rets[rets < 0]
            if len(downside_rets) > 0 and downside_rets.std() > 0:
                sortino = (mean_ret / downside_rets.std()) * np.sqrt(ppy)
            else:
                sortino = sharpe
        
        # Bootstrap CI for Sharpe
        sharpe_ci_low, sharpe_ci_high = 0.0, 0.0
        if len(rets) > 10:
            boot_sharpes = []
            for _ in range(500):
                boot_rets = np.random.choice(rets, size=len(rets), replace=True)
                if boot_rets.std() > 0:
                    boot_sharpe = (boot_rets.mean() / boot_rets.std()) * np.sqrt(ppy)
                    boot_sharpes.append(boot_sharpe)
            if boot_sharpes:
                sharpe_ci_low = np.percentile(boot_sharpes, 2.5)
                sharpe_ci_high = np.percentile(boot_sharpes, 97.5)

        eq_curve = 1.0 + pd.Series(rets).cumsum()
        max_dd = calc_drawdown(eq_curve).min() * 100
        
        win_rate = len(rets[rets > 0]) / len(rets) if len(rets) > 0 else 0
        pf_up = g[g["y_true"] > 0]["y_true"].sum() 
        pf_dn = abs(g[g["y_true"] < 0]["y_true"].sum())
        pf = pf_up / pf_dn if pf_dn > 0 else float('inf')
        
        metrics = {
            "Total_PnL": cum_pnl,
            "Sharpe": sharpe,
            "Sharpe_CI_Low": sharpe_ci_low,
            "Sharpe_CI_High": sharpe_ci_high,
            "Sortino": sortino,
            "Max_DD_Pct": max_dd,
            "Win_Rate": win_rate,
            "Profit_Factor": pf,
            "Total_Trades": g["n_trades"].sum(),
            "Samples": len(rets)
        }
        
        for m_name, m_val in metrics.items():
            run_metrics.append({
                "run_id": run_id,
                "strategy": model,
                "metric_name": m_name,
                "metric_value": float(m_val)
            })
            
    if results:
        conn.execute("DELETE FROM backtest_results")
        write_df(conn, pd.DataFrame(results), "backtest_results")
        
        conn.execute("DELETE FROM run_metrics")
        write_df(conn, pd.DataFrame(run_metrics), "run_metrics")
        
        # Log a quick summary of the primary top_1 run
        summary_query = """
            SELECT strategy, metric_name, metric_value 
            FROM run_metrics 
            WHERE run_id LIKE '%top1' 
            ORDER BY strategy, metric_name
        """
        df_sum = read_df(conn, summary_query)
        if not df_sum.empty:
            log.info("\n=== Evaluation Results (Top 1) ===")
            for strategy, sub in df_sum.groupby("strategy"):
                log.info(f"Model: {strategy}")
                for _, r in sub.iterrows():
                    log.info(f"  {r['metric_name']:>15}: {r['metric_value']:.4f}")
                    
        log.info(f"Evaluated {len(run_metrics)} strategy variations.")
