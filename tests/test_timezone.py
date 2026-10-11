import datetime
import pandas as pd
import pytest
from zoneinfo import ZoneInfo

from src.utils.timezone import (
    to_utc,
    utc_to_new_york,
    new_york_to_utc,
    format_iso_utc,
    format_iso_new_york,
)


def test_utc_to_new_york_winter_est():
    # Winter date: January 15, 2026 -> EST (UTC - 5 hours)
    utc_ts = pd.Timestamp("2026-01-15 13:30:00", tz="UTC")
    ny_ts = utc_to_new_york(utc_ts)
    assert ny_ts.hour == 8
    assert ny_ts.minute == 30
    assert ny_ts.utcoffset() == datetime.timedelta(hours=-5)
    assert "-05:00" in format_iso_new_york(utc_ts)


def test_utc_to_new_york_summer_edt():
    # Summer date: July 15, 2026 -> EDT (UTC - 4 hours)
    utc_ts = pd.Timestamp("2026-07-15 12:30:00", tz="UTC")
    ny_ts = utc_to_new_york(utc_ts)
    assert ny_ts.hour == 8
    assert ny_ts.minute == 30
    assert ny_ts.utcoffset() == datetime.timedelta(hours=-4)
    assert "-04:00" in format_iso_new_york(utc_ts)


def test_spring_forward_dst_boundary():
    # US DST spring forward in 2026: Sunday March 8, 2026.
    # At 02:00 local, clocks jump to 03:00 EDT.
    # 06:59:00 UTC is 01:59:00 EST (UTC-5)
    ts_pre = pd.Timestamp("2026-03-08 06:59:00", tz="UTC")
    ny_pre = utc_to_new_york(ts_pre)
    assert ny_pre.hour == 1
    assert ny_pre.minute == 59
    assert ny_pre.utcoffset() == datetime.timedelta(hours=-5)

    # 07:01:00 UTC is 03:01:00 EDT (UTC-4)
    ts_post = pd.Timestamp("2026-03-08 07:01:00", tz="UTC")
    ny_post = utc_to_new_york(ts_post)
    assert ny_post.hour == 3
    assert ny_post.minute == 1
    assert ny_post.utcoffset() == datetime.timedelta(hours=-4)


def test_fall_back_dst_boundary():
    # US DST fall back in 2026: Sunday November 1, 2026.
    # At 02:00 local, clocks turn back to 01:00 EST.
    # 05:59:00 UTC is 01:59:00 EDT (UTC-4)
    ts_pre = pd.Timestamp("2026-11-01 05:59:00", tz="UTC")
    ny_pre = utc_to_new_york(ts_pre)
    assert ny_pre.utcoffset() == datetime.timedelta(hours=-4)

    # 06:01:00 UTC is 01:01:00 EST (UTC-5)
    ts_post = pd.Timestamp("2026-11-01 06:01:00", tz="UTC")
    ny_post = utc_to_new_york(ts_post)
    assert ny_post.utcoffset() == datetime.timedelta(hours=-5)


def test_round_trip_conversion():
    ts_utc = pd.Timestamp("2026-05-13 12:30:00", tz="UTC")
    ts_ny = utc_to_new_york(ts_utc)
    ts_roundtrip = new_york_to_utc(ts_ny)
    assert ts_roundtrip == ts_utc


def test_naive_string_handling():
    # Naive strings without tz info default to UTC in to_utc
    ts = to_utc("2026-08-01 10:00:00")
    assert ts.tz == ZoneInfo("UTC")
    assert ts.year == 2026
    assert ts.month == 8
