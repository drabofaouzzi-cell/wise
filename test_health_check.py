"""Tests for tv_health_check()."""

import pandas as pd
import numpy as np
import pytest
from dow_theory_analyzer import tv_health_check, HealthCheckResult


def _make_df(n=100, seed=0):
    """Return a clean OHLCV DataFrame."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")
    close = 100.0 + np.cumsum(rng.normal(0, 1, n))
    close = np.abs(close) + 1  # keep positive
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.01, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.01, n))
    volume = rng.lognormal(10, 0.5, n)
    return pd.DataFrame({"date": dates, "open": open_, "high": high, "low": low, "close": close, "volume": volume})


# ── Passing cases ──────────────────────────────────────────────────────────────

def test_clean_data_passes():
    result = tv_health_check(_make_df(200))
    assert result.passed is True
    assert result.errors == []


def test_stats_populated():
    result = tv_health_check(_make_df(50))
    assert result.stats["row_count"] == 50
    assert "date_start" in result.stats
    assert "date_end" in result.stats


# ── Row count ──────────────────────────────────────────────────────────────────

def test_too_few_rows_errors():
    result = tv_health_check(_make_df(5))
    assert result.passed is False
    assert any("Insufficient" in e for e in result.errors)


def test_few_rows_warns():
    result = tv_health_check(_make_df(15))
    assert result.passed is True
    assert any("30" in w for w in result.warnings)


def test_moderate_rows_warns():
    result = tv_health_check(_make_df(50))
    assert result.passed is True
    assert any("reliability" in w for w in result.warnings)


# ── Missing columns ────────────────────────────────────────────────────────────

def test_missing_column_errors():
    df = _make_df().drop(columns=["close"])
    result = tv_health_check(df)
    assert result.passed is False
    assert any("close" in e for e in result.errors)


# ── NaN values ────────────────────────────────────────────────────────────────

def test_few_nans_warns():
    df = _make_df(100)
    df.loc[0, "close"] = np.nan
    result = tv_health_check(df)
    assert result.passed is True
    assert any("close" in w for w in result.warnings)


def test_many_nans_errors():
    df = _make_df(100)
    df.loc[:10, "close"] = np.nan  # 11% → error
    result = tv_health_check(df)
    assert result.passed is False
    assert any("close" in e for e in result.errors)


# ── Non-positive prices ────────────────────────────────────────────────────────

def test_zero_price_errors():
    df = _make_df(50)
    df.loc[5, "open"] = 0.0
    result = tv_health_check(df)
    assert result.passed is False
    assert any("non-positive" in e for e in result.errors)


def test_negative_price_errors():
    df = _make_df(50)
    df.loc[3, "low"] = -1.5
    result = tv_health_check(df)
    assert result.passed is False


# ── OHLC consistency ───────────────────────────────────────────────────────────

def test_high_less_than_low_errors():
    df = _make_df(50)
    df.loc[10, "high"] = df.loc[10, "low"] - 1
    result = tv_health_check(df)
    assert result.passed is False
    assert any("high < low" in e for e in result.errors)


def test_high_less_than_close_errors():
    df = _make_df(50)
    idx = 10
    df.loc[idx, "close"] = df.loc[idx, "high"] + 5
    result = tv_health_check(df)
    assert result.passed is False
    assert any("high < max" in e for e in result.errors)


def test_low_greater_than_open_errors():
    df = _make_df(50)
    idx = 10
    df.loc[idx, "open"] = df.loc[idx, "low"] - 5
    result = tv_health_check(df)
    assert result.passed is False
    assert any("low > min" in e for e in result.errors)


# ── Duplicate timestamps ───────────────────────────────────────────────────────

def test_duplicate_dates_warns():
    df = _make_df(50)
    df.loc[5, "date"] = df.loc[4, "date"]
    result = tv_health_check(df)
    assert any("duplicate" in w.lower() for w in result.warnings)


# ── Time gaps ─────────────────────────────────────────────────────────────────

def test_large_gap_warns():
    df = _make_df(50)
    # Insert a 60-day gap after row 20
    df.loc[20:, "date"] = df.loc[20:, "date"] + pd.Timedelta(days=60)
    result = tv_health_check(df)
    assert any("gap" in w.lower() for w in result.warnings)


# ── Volume ────────────────────────────────────────────────────────────────────

def test_missing_volume_warns():
    df = _make_df(50).drop(columns=["volume"])
    result = tv_health_check(df)
    assert any("volume" in w.lower() for w in result.warnings)


def test_all_zero_volume_warns():
    df = _make_df(50)
    df["volume"] = 0
    result = tv_health_check(df)
    assert any("volume" in w.lower() for w in result.warnings)


def test_mostly_zero_volume_warns():
    df = _make_df(50)
    df.loc[:40, "volume"] = 0  # 82% zeros
    result = tv_health_check(df)
    assert any("zero volume" in w for w in result.warnings)


# ── Extreme moves ─────────────────────────────────────────────────────────────

def test_extreme_candle_warns():
    df = _make_df(50)
    df.loc[25, "close"] = df.loc[25, "open"] * 2.0  # 100% body move
    # Fix high so OHLC stays consistent
    df.loc[25, "high"] = df.loc[25, "close"] * 1.001
    result = tv_health_check(df)
    assert any("30%" in w for w in result.warnings)
