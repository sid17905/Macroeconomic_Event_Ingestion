# Feature Documentation & Point-in-Time Construction Rules

All features in the CrowdWisdomTrading Quant Data Scientist ML pipeline are engineered strictly under a point-in-time (**as-of**) policy at the start of each weekly prediction period (`period_start`). No future data (events, news, or trade outcomes occurring after `period_start`) is ever incorporated into feature vectors.

---

## 1. Feature Groups Overview

| Feature Group | Feature Name | Type | Description / Construction Formula |
| :--- | :--- | :--- | :--- |
| **Strategy Hyperparameters** | `param_fast` | float | Fast Moving Average period length (e.g. 10, 20) |
| | `param_slow` | float | Slow Moving Average period length (e.g. 50, 100, 200) |
| | `param_lookback` | float | Donchian Channel breakout lookback window in days (e.g. 20, 55) |
| | `param_stop_pct` | float | Stop loss percentage threshold (e.g. 0.01 = 1%) |
| | `param_take_profit_pct` | float | Take profit percentage threshold (e.g. 0.02 = 2%) |
| **Lagged Performance** | `lag_pnl_sum` | float | Sum of realized trade PnL ($) across the rolling `rolling_perf_window` (4 weeks) prior to `period_start`. |
| | `lag_pnl_mean` | float | Average realized weekly PnL ($) across past 4 periods prior to `period_start`. |
| **Time & Calendar** | `month` | float | Calendar month (1–12) of `period_start`. |
| | `day_of_week` | float | Day of week (0=Monday, 6=Sunday) of `period_start`. |
| | `hour_of_day` | float | Hour of day (0–23 UTC) of `period_start`. |
| **Macro Event Timings** | `hours_since_last_event` | float | Elapsed time in hours from the most recent high-impact macro event prior to `period_start`. |
| | `last_event_type_<TYPE>` | binary | One-hot indicator (1.0 or 0.0) for event type of last known event (`CPI`, `FOMC`, `NFP`, `UNEMP`, `JOBLESS`). |
| **Macro Surprise Metrics** | `<TYPE>_last_surprise` | float | Raw surprise (`actual - forecast`) from the most recent event of type `<TYPE>` prior to `period_start`. |
| | `<TYPE>_roll3_zscore` | float | Rolling 3-event z-score: `(surprise - mean(past_3)) / std(past_3)` computed strictly on historical releases. |
| **News Sentiment** | `sentiment_recent_avg` | float | Average polarity score (VADER/financial compound) of news published within `sentiment_roll_days` (7 days) before `period_start`. |
| | `sentiment_recent_count` | float | Number of macro news articles published within the 7-day lookback window before `period_start`. |

---

## 2. No-Lookahead / Point-in-Time Leakage Prevention

1. **Macro Events (`build_macro_features`)**:
   Filtered via `known = df_events[df_events["ts_utc"] < period_start]`. Events released after `period_start` are omitted from timing and surprise calculation.
2. **News Articles (`build_sentiment_features`)**:
   Filtered via `known = df_news[df_news["published_utc"] < period_start]`. Articles published after `period_start` are excluded from sentiment aggregation.
3. **Trade Performance Targets & Lags**:
   Realized PnL is attributed to periods based on `exit_time`. Only trades exiting before `period_start` feed into `lag_pnl_*`. Target values (`y_pnl`, `y_winrate`) are calculated exclusively from trades exiting in `[period_start, period_end)`.
