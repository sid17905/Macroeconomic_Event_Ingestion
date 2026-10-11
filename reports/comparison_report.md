# Deliverables vs. Requirements Comparison Report

This document compares the requirements specified in the CrowdWisdomTrading Quant Data Scientist intern assessment with the implemented solution.

---

## 1. Requirements vs. Deliverables

| Requirement | Status | Implementation Detail |
| :--- | :--- | :--- |
| **Macro Ingestion (Apify)** | ✅ Done | Implemented via `src/ingestion/ingest.py`, using `mstephen190/economic-calendar-scraper`. Cached in `data/raw/`. |
| **Macro News (Tavily)** | ✅ Done | Implemented via `src/ingestion/ingest.py` + VADER sentiment analysis. |
| **Database (SQLite)** | ✅ Done | Full schema (`schema.sql`) for trades, macro, news, features, predictions, and metrics. |
| **Trade Data** | ✅ Done | Simulated backtest ("stand-in") on yfinance data (SPY/QQQ/TLT/GLD) via `src/trades/backtest.py`. |
| **Feature Engineering** | ✅ Done | Strict as-of pipeline (`src/features/build_features.py`) ensuring no look-ahead. |
| **Model + Walk-Forward** | ✅ Done | LightGBM regression, expanding window + purge/embargo (`src/models/train.py`). |
| **Evaluation Metrics** | ✅ Done | Sharpe, Sortino, MaxDD, Profit Factor, Win Rate, calculated in `src/evaluation/evaluate.py`. |
| **Macro Outlook** | ✅ Done | 1-page report (`reports/macro_outlook.md`) grounded in scraped data. |
| **Visual Matrix Output** | ✅ Done | Figures: SentimentTimeline, Drawdowns, Prediction Heatmaps (`src/viz/report.py`). |
| **180-day data scope** | ✅ Done | Configured in `config.yaml` as requested. |

---

## 2. Key Design & Implementation Decisions

1. **No-Look-Ahead Guarantee:** Implemented strict as-of feature construction; no future information leaks into training sets. Validated with `tests/test_no_leakage.py`.
2. **Reproducibility:** All Random seeds are fixed (config.yaml). API calls are cached, and the pipeline is fully idempotent (`run_pipeline.py`).
3. **Pipeline Orchestration:** A single entry point (`run_pipeline.py`) manages all 8 stages, with a `--stage` flag for granular control.
4. **Walk-Forward Validation:** Utilized expanding windows with a mandatory **purge** of trade-overlapping periods and an **embargo** gap to eliminate train/test contamination.

---

## 3. Performance Summary (Top-Selected Strategy)

| strategy | Max_DD_Pct | Profit_Factor | Sharpe | Sortino | Total_PnL |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **LGB (Model)** | -0.56% | 1.80 | 1.37 | 2.50 | +1697.23 |
| **Baseline** | -1.01% | 0.89 | -0.18 | -0.24 | -372.44 |

*Note: The model outperformed the baseline strategy consistently across the tested macro regime.*

---

## 4. Submission Notes
- **Repo Structure:** Well-organized structure under `src/`, with all data cached in `data/raw`.
- **Reproducibility:** See `README.md` for full instructions to re-run the pipeline with the cached data, including the recorded Apify actor ID used.
- **Limitations:** Documented in `reports/evaluation_report.md` regarding sample size and stand-in trade generation.
