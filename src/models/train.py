"""Walk-forward nested cross-validation and LightGBM model training."""
import json
import uuid
import pandas as pd
import numpy as np
try:
    import lightgbm as lgb
except ImportError:
    from sklearn.ensemble import HistGradientBoostingRegressor as lgb  # fallback
    
from src.logging_utils import get_logger
from src.db.database import read_df, write_df

log = get_logger(__name__)


def flat_df(df_feat: pd.DataFrame) -> pd.DataFrame:
    """Expand feature_json into columns."""
    if df_feat.empty:
        return df_feat
        
    records = []
    for _, row in df_feat.iterrows():
        d = row.to_dict()
        feats = json.loads(d.pop("feature_json"))
        d.update(feats)
        records.append(d)
        
    res = pd.DataFrame(records)
    res["period_start"] = pd.to_datetime(res["period_start"], utc=True)
    return res


def get_walk_forward_splits(periods: list[pd.Timestamp], cfg):
    """Generate train/test split indices obeying embargo rules."""
    min_train = cfg["walkforward"]["min_train_periods"]
    embargo = cfg["walkforward"]["embargo_periods"]
    
    splits = []
    
    for i in range(min_train, len(periods)):
        train_end_idx = i - embargo
        if train_end_idx < 1:
            continue
            
        train_periods = periods[:train_end_idx]
        test_period = periods[i] # Just one period forward
        
        splits.append((train_periods, test_period))
        
    return splits


def train_and_predict(X_train, y_train, X_test, cfg) -> np.ndarray:
    """Train LightGBM regression on historical trades."""
    params = cfg["model"].get("lightgbm", {})
    if "HistGradientBoosting" in str(lgb):
        # Fallback handling
        model = lgb(random_state=cfg.get("seed", 42))
    else:
        # Proper lightgbm
        model = lgb.LGBMRegressor(
            n_estimators=params.get("n_estimators", 100),
            learning_rate=params.get("learning_rate", 0.05),
            num_leaves=params.get("num_leaves", 31),
            random_state=cfg.get("seed", 42),
            verbose=-1
        )
        
    model.fit(X_train, y_train)
    return model.predict(X_test)


def run_walkforward(cfg, conn) -> None:
    """Execute walk-forward validation and store predictions."""
    df_raw = read_df(conn, "SELECT * FROM features")
    if df_raw.empty:
        log.warning("No features found, skipping model.")
        return
        
    df = flat_df(df_raw)
    df = df.sort_values(["period_start", "permutation_id"]).reset_index(drop=True)
    
    periods = sorted(df["period_start"].unique())
    splits = get_walk_forward_splits(periods, cfg)
    
    # Identify feature cols (everything except meta logic and targets)
    exclude = {"permutation_id", "period_start", "family", "y_pnl", "y_winrate", "n_trades"}
    feat_cols = [c for c in df.columns if c not in exclude]
    
    # Fill NAs in features with 0 (since they represent missing events largely)
    df[feat_cols] = df[feat_cols].fillna(0)
    
    target_col = f"y_{cfg['model']['target']}"
    run_id = str(uuid.uuid4())
    top_k_list = cfg["walkforward"]["top_k"]
    predictions = []
    
    log.info(f"Starting walk-forward validation over {len(splits)} splits...")
    
    for train_periods, test_period in splits:
        train_mask = df["period_start"].isin(train_periods)
        test_mask = df["period_start"] == test_period
        
        train_df = df[train_mask]
        test_df = df[test_mask]
        
        if train_df.empty or test_df.empty:
            continue
            
        X_train, y_train = train_df[feat_cols], train_df[target_col]
        X_test, y_test = test_df[feat_cols], test_df[target_col]
        
        # We need something to learn, handle edge case if all targets are 0
        if y_train.var() < 1e-6:
            preds = np.zeros(len(X_test))
        else:
            preds = train_and_predict(X_train, y_train, X_test, cfg)
            
        # Baseline model: just use past naive average (lag_pnl_mean)
        if "lag_pnl_mean" in test_df.columns:
            baseline_preds = test_df["lag_pnl_mean"].values
        else:
            baseline_preds = np.zeros(len(X_test))
            
        # Ranks per group
        test_res = test_df[["permutation_id"]].copy()
        test_res["y_pred_lgbm"] = preds
        test_res["y_pred_base"] = baseline_preds
        test_res["y_true"] = y_test.values
        
        for top_k in top_k_list:
            for model_name, pred_col in [("lgb", "y_pred_lgbm"), ("baseline_momentum", "y_pred_base")]:
                ordered = test_res.sort_values(pred_col, ascending=False).reset_index()
                ordered["rank"] = ordered.index + 1
                ordered["selected"] = (ordered["rank"] <= top_k).astype(int)
                
                for _, row in ordered.iterrows():
                    predictions.append({
                        "run_id": f"{run_id}_top{top_k}",
                        "period_start": test_period.isoformat(),
                        "permutation_id": row["permutation_id"],
                        "model": model_name,
                        "y_pred": row[pred_col],
                        "rank": int(row["rank"]),
                        "selected": int(row["selected"]),
                        "y_true": row["y_true"]
                    })

    df_preds = pd.DataFrame(predictions)
    
    if not df_preds.empty:
        conn.execute("DELETE FROM predictions")
        write_df(conn, df_preds, "predictions")
        log.info(f"Saved {len(df_preds)} predictions. Run ID: {run_id}")
    else:
        log.warning("No predictions generated (needs more periods).")
