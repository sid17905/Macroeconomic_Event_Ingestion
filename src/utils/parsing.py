from __future__ import annotations

import math
import re
from typing import Any


def parse_numeric_value(v: Any) -> float | None:
    """Parse macroeconomic calendar values into clean numeric floats.

    Supports:
      - Percentages: '3.2%', '-0.5%' -> 3.2, -0.5 (expressed in percentage points)
      - Multipliers: '250K' / '250k' -> 250000.0, '1.5M' -> 1500000.0, '2.1B' -> 2100000000.0
      - Currency symbols & commas: '$1,250K', '150,000'
      - Signs: '+0.4', '-1.2'
      - Missing markers: None, '', '-', 'N/A', 'nan', 'null' -> None
    """
    if v is None:
        return None

    if isinstance(v, (int, float)):
        if math.isnan(v):
            return None
        return float(v)

    s = str(v).strip()
    if not s or s.lower() in {"-", "--", "n/a", "na", "null", "none", "nan", "."}:
        return None

    cleaned = s.replace(",", "").replace("$", "").replace(" ", "")

    multiplier = 1.0
    if cleaned.endswith("%"):
        cleaned = cleaned[:-1]
    elif cleaned.endswith(("k", "K")):
        multiplier = 1e3
        cleaned = cleaned[:-1]
    elif cleaned.endswith(("m", "M")):
        multiplier = 1e6
        cleaned = cleaned[:-1]
    elif cleaned.endswith(("b", "B")):
        multiplier = 1e9
        cleaned = cleaned[:-1]
    elif cleaned.endswith(("t", "T")):
        multiplier = 1e12
        cleaned = cleaned[:-1]

    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", cleaned)
    if not m:
        return None

    try:
        val = float(m.group()) * multiplier
        return val
    except (ValueError, OverflowError):
        return None


def compute_surprise(actual: float | None, forecast: float | None) -> float | None:
    """Compute surprise = actual - forecast. Returns None if either is missing."""
    if actual is None or forecast is None:
        return None
    if math.isnan(actual) or math.isnan(forecast):
        return None
    return float(actual - forecast)
