from __future__ import annotations

import datetime
from zoneinfo import ZoneInfo
import pandas as pd

NY_TZ = ZoneInfo("America/New_York")
UTC_TZ = ZoneInfo("UTC")


def to_utc(ts: str | datetime.datetime | pd.Timestamp) -> pd.Timestamp:
    """Parse and convert any timestamp into a UTC-localized pandas Timestamp."""
    t = pd.to_datetime(ts)
    if t.tz is None:
        t = t.tz_localize(UTC_TZ)
    else:
        t = t.tz_convert(UTC_TZ)
    return t


def utc_to_new_york(ts: str | datetime.datetime | pd.Timestamp) -> pd.Timestamp:
    """Convert a UTC timestamp to America/New_York (DST-aware Eastern Time)."""
    t = to_utc(ts)
    return t.tz_convert(NY_TZ)


def new_york_to_utc(
    ts: str | datetime.datetime | pd.Timestamp,
    is_dst: bool | None = None,
) -> pd.Timestamp:
    """Convert an America/New_York timestamp to UTC, handling DST transitions."""
    t = pd.to_datetime(ts)
    if t.tz is None:
        if is_dst is not None:
            t = t.tz_localize(NY_TZ, ambiguous=is_dst)
        else:
            try:
                t = t.tz_localize(NY_TZ, ambiguous="infer")
            except Exception:
                t = t.tz_localize(NY_TZ, ambiguous=False)
    else:
        t = t.tz_convert(NY_TZ)
    return t.tz_convert(UTC_TZ)


def format_iso_utc(ts: str | datetime.datetime | pd.Timestamp) -> str:
    return to_utc(ts).isoformat()


def format_iso_new_york(ts: str | datetime.datetime | pd.Timestamp) -> str:
    return utc_to_new_york(ts).isoformat()
