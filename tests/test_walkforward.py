import pandas as pd
import pytest
from src.models.train import get_walk_forward_splits

def test_walk_forward_splits_embargo():
    cfg = {
        "walkforward": {
            "min_train_periods": 5,
            "embargo_periods": 1
        }
    }
    periods = [pd.Timestamp(f"2026-01-{i:02d}", tz="UTC") for i in range(1, 15)]
    splits = get_walk_forward_splits(periods, cfg)
    
    assert len(splits) > 0
    for train_periods, test_period in splits:
        # Train periods must be strictly before test_period - embargo
        assert max(train_periods) < test_period
        # With embargo = 1, the gap must be at least 1 period
        assert len(train_periods) >= 4
        # Assert no overlap
        assert test_period not in train_periods
