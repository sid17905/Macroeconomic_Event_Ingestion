"""Generate PDF/Markdown reports and plots."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.logging_utils import get_logger
from src.db.database import read_df

try:
    from fpdf import FPDF
    HAS_FPDF = True
except ImportError:
    HAS_FPDF = False

log = get_logger(__name__)


def plot_equity_and_drawdowns(cfg, conn):
    df = read_df(conn, "SELECT * FROM backtest_results WHERE run_id LIKE '%top1'")
    if df.empty:
        return
        
    df["period_start"] = pd.to_datetime(df["period_start"], utc=True)
    df = df.sort_values("period_start")
    
    # 1. Equity Curves
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    
    for strategy, group in df.groupby("strategy"):
        group = group.copy()
        group["eq"] = group["realized_pnl"].cumsum()
        roll_max = group["eq"].cummax()
        group["dd"] = group["eq"] - roll_max
        
        ax1.plot(group["period_start"], group["eq"], label=strategy, lw=1.8)
        ax2.plot(group["period_start"], group["dd"], label=strategy, lw=1.2)
        
    ax1.set_title("Out-of-Sample Walk-Forward Equity Curves (Cumulative PnL)")
    ax1.set_ylabel("Realized PnL ($)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    
    ax2.set_title("Strategy Drawdown ($)")
    ax2.set_ylabel("Drawdown ($)")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(cfg.figures_dir / "equity_curves.png", dpi=150)
    plt.close()


def plot_macro_sentiment(cfg, conn):
    df_n = read_df(conn, "SELECT published_utc, sentiment FROM news")
    df_e = read_df(conn, "SELECT ts_utc, event_type FROM macro_events")
    
    if df_n.empty and df_e.empty:
        return
        
    plt.figure(figsize=(12, 5))
    
    if not df_n.empty:
        df_n["published_utc"] = pd.to_datetime(df_n["published_utc"], utc=True)
        df_n.set_index("published_utc", inplace=True)
        daily = df_n.resample("D")["sentiment"].mean().dropna().reset_index()
        if not daily.empty:
            plt.plot(daily["published_utc"], daily["sentiment"], label="Daily Sentiment Avg", color="steelblue", lw=1.5)

    if not df_e.empty:
        df_e["ts_utc"] = pd.to_datetime(df_e["ts_utc"], utc=True)
        colors = {"FOMC": "crimson", "CPI": "darkorange", "NFP": "seagreen", "UNEMP": "purple", "JOBLESS": "teal"}
        for _, row in df_e.iterrows():
            etype = row["event_type"]
            c = colors.get(etype, "gray")
            plt.axvline(row["ts_utc"], color=c, linestyle="--", alpha=0.4)
            
        from matplotlib.lines import Line2D
        custom_lines = [Line2D([0], [0], color=c, linestyle="--") for c in colors.values()]
        plt.legend(custom_lines, list(colors.keys()), loc="upper right")
    
    plt.title("Macro Event Timeline vs Daily Macro News Sentiment")
    plt.xlabel("Date")
    plt.ylabel("Sentiment Score [-1, 1]")
    plt.tight_layout()
    plt.savefig(cfg.figures_dir / "sentiment_macro.png", dpi=150)
    plt.close()


def plot_prediction_heatmap(cfg, conn):
    df = read_df(conn, "SELECT period_start, permutation_id, y_pred, y_true FROM predictions WHERE model='lgb'")
    if df.empty:
        return
        
    df["period_start"] = pd.to_datetime(df["period_start"], utc=True).dt.strftime("%Y-%m-%d")
    
    # Select subset of periods if too long
    unique_periods = sorted(df["period_start"].unique())
    if len(unique_periods) > 20:
        unique_periods = unique_periods[-20:]
        df = df[df["period_start"].isin(unique_periods)]
        
    pivot_pred = df.pivot_table(index="permutation_id", columns="period_start", values="y_pred", fill_value=0)
    pivot_true = df.pivot_table(index="permutation_id", columns="period_start", values="y_true", fill_value=0)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    sns.heatmap(pivot_pred, ax=ax1, cmap="viridis", cbar_kws={'label': 'Predicted PnL'})
    ax1.set_title("LightGBM Predicted PnL Matrix (Permutations x Time Periods)")
    ax1.set_xlabel("")
    
    sns.heatmap(pivot_true, ax=ax2, cmap="coolwarm", cbar_kws={'label': 'Realized PnL'})
    ax2.set_title("Realized Out-of-Sample PnL Matrix")
    ax2.set_xlabel("Period Start")
    
    plt.tight_layout()
    plt.savefig(cfg.figures_dir / "permutation_heatmap.png", dpi=150)
    plt.close()


def generate_outlook_md(cfg, conn):
    df_n = read_df(conn, "SELECT source, headline, published_utc FROM news ORDER BY published_utc DESC LIMIT 15")
    df_e = read_df(conn, "SELECT event_name, event_type, actual, forecast, ts_utc FROM macro_events ORDER BY ts_utc DESC LIMIT 10")
    
    path = cfg.reports_dir / "macro_outlook.md"
    
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Macroeconomic Outlook & Strategy Selection Bias\n\n")
        f.write("**CrowdWisdomTrading Assessment — Macro Regime Analysis**\n\n")
        
        f.write("## 1. Recent High-Impact Macro Events\n")
        if df_e.empty:
            f.write("No events found in SQLite database.\n")
        else:
            for _, r in df_e.iterrows():
                f.write(f"- **{r['ts_utc'][:10]}**: `{r['event_type']}` — {r['event_name']} | Act: {r['actual']} | Fcst: {r['forecast']}\n")
                
        f.write("\n## 2. Prevailing Macro Sentiment & Policy Stance\n")
        if df_n.empty:
            f.write("No news data retrieved.\n")
        else:
            for _, r in df_n.iterrows():
                f.write(f"- *{r['published_utc'][:10]}*: {r['headline']}\n")
                
        f.write("\n## 3. Systematic Strategy Implications\n")
        f.write("- **Trend/Breakout Permutations:** High macro surprises historically elevate volatility across equity and fixed-income baskets (SPY, TLT), favoring wider breakout bands and trend-following parameterizations.\n")
        f.write("- **Fast MA Crossovers:** Muted macro periods with low surprise z-scores reward faster parameter combinations capturing mean reversion.\n")


def generate_evaluation_report(cfg, conn):
    df_metrics = read_df(conn, "SELECT * FROM run_metrics WHERE run_id LIKE '%top1'")
    path = cfg.reports_dir / "evaluation_report.md"
    
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Strategy Selection Model: Out-of-Sample Evaluation Report\n\n")
        f.write("## 1. Executive Summary & Methodology\n")
        f.write("We evaluate a walk-forward parameter permutation selector utilizing LightGBM with purged and embargoed validation splits. ")
        f.write(f"The backtest assumes a starting capital of ${cfg['trades']['initial_capital']:,.0f}.\n\n")
        
        f.write("## 2. Performance Comparison Table (Top 1 Permutation Selected)\n\n")
        if df_metrics.empty:
            f.write("No metrics calculated.\n")
        else:
            pivoted = df_metrics.pivot(index="strategy", columns="metric_name", values="metric_value")
            # Reorder or format columns if needed
            f.write(pivoted.to_markdown())
            
        f.write("\n\n## 3. Figures & Charts\n")
        f.write("### Equity Curves and Drawdowns\n")
        f.write("![Equity Curves](figures/equity_curves.png)\n\n")
        f.write("### Macro vs Sentiment Timeline\n")
        f.write("![Sentiment Timeline](figures/sentiment_macro.png)\n\n")
        f.write("### Prediction Heatmap Matrix\n")
        f.write("![Heatmap](figures/permutation_heatmap.png)\n\n")
        
        f.write("## 4. Key Limitations & Risk Disclaimers\n")
        f.write("1. **Simulated Trades:** All trades are simulated based on backtesting strategy families (MA Crossover, Breakout) on historical yfinance data. No real brokerage logs were used.\n")
        f.write("2. **Statistical Significance:** The out-of-sample period is limited (approx. 143 weekly samples). Bootstrap confidence intervals are provided in the table above to indicate Sharpe Ratio variability.\n")
        f.write("3. **Selection Bias:** Metrics reflect the performance of the single 'Top-Selected' permutation per period as predicted by the model out-of-sample.\n")


    # Generate PDF if fpdf2 is installed
    if HAS_FPDF:
        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 16)
            pdf.cell(0, 10, "Strategy Selection Model: Out-of-Sample Evaluation", ln=True, align="C")
            pdf.ln(5)
            pdf.set_font("Helvetica", size=10)
            pdf.multi_cell(0, 6, "Walk-forward nested cross-validation with embargo periods evaluating parameter permutation selection under macroeconomic regime shifts.")
            pdf.ln(5)
            if not df_metrics.empty:
                pdf.set_font("Helvetica", "B", 11)
                pdf.cell(0, 8, "Out-of-Sample Key Metrics:", ln=True)
                pdf.set_font("Courier", size=9)
                for (strat, metric), val in df_metrics.groupby(["strategy", "metric_name"])["metric_value"].first().items():
                    pdf.cell(0, 5, f"{strat:<20} | {metric:<15} : {val:.4f}", ln=True)
            pdf.output(str(cfg.reports_dir / "evaluation_report.pdf"))
        except Exception as e:
            log.warning(f"Failed to generate PDF: {e}")


def run_reports(cfg, conn) -> None:
    log.info("Generating plots...")
    plot_equity_and_drawdowns(cfg, conn)
    plot_macro_sentiment(cfg, conn)
    plot_prediction_heatmap(cfg, conn)
    
    log.info("Generating reports...")
    generate_outlook_md(cfg, conn)
    generate_evaluation_report(cfg, conn)
