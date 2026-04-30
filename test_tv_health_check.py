"""Tests for tv_health_check."""

import pandas as pd
import numpy as np
import pytest
from dow_theory_analyzer import tv_health_check


def _make_df(n=100, seed=42) -> pd.DataFrame:
    """Return a clean synthetic OHLCV DataFrame."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    close = 100 * np.cumprod(1 + rng.normal(0, 0.01, n))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) * (1 + rng.uniform(0.001, 0.005, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.001, 0.005, n))
    volume = rng.lognormal(10, 0.5, n)
    return pd.DataFrame({"date": dates, "open": open_, "high": high, "low": low, "close": close, "volume": volume})


# ── Passing cases ──────────────────────────────────────────────────────────────

def test_clean_data_passes():
    df = _make_df()
    r = tv_health_check(df)
    assert r.passed
    assert r.errors == []


def test_stats_populated():
    df = _make_df(n=50)
    r = tv_health_check(df)
    assert r.stats["bars"] == 50
    assert "start" in r.stats
    assert "end" in r.stats
    assert "median_interval" in r.stats


# ── Error cases ────────────────────────────────────────────────────────────────

def test_too_few_bars():
    df = _make_df(n=10)
    r = tv_health_check(df, min_bars=30)
    assert not r.passed
    assert any("Trop peu" in e for e in r.errors)


def test_duplicate_timestamps():
    df = _make_df()
    df = pd.concat([df, df.iloc[[5]]], ignore_index=True)
    r = tv_health_check(df)
    assert not r.passed
    assert any("dupliqué" in e for e in r.errors)


def test_invalid_high():
    df = _make_df()
    df.loc[10, "high"] = df.loc[10, "close"] * 0.5  # high < close
    r = tv_health_check(df)
    assert not r.passed
    assert any("high < max" in e for e in r.errors)


def test_invalid_low():
    df = _make_df()
    df.loc[10, "low"] = df.loc[10, "close"] * 1.5  # low > close
    r = tv_health_check(df)
    assert not r.passed
    assert any("low > min" in e for e in r.errors)


def test_negative_prices():
    df = _make_df()
    df.loc[5, "close"] = -1.0
    r = tv_health_check(df)
    assert not r.passed
    assert any("<= 0" in e for e in r.errors)


def test_excessive_nan():
    df = _make_df()
    df.loc[df.index[:10], "close"] = np.nan  # 10% NaN
    r = tv_health_check(df)
    assert not r.passed
    assert any("NaN" in e for e in r.errors)


def test_negative_volume():
    df = _make_df()
    df.loc[3, "volume"] = -500
    r = tv_health_check(df)
    assert not r.passed
    assert any("volume négative" in e for e in r.errors)


# ── Warning cases ──────────────────────────────────────────────────────────────

def test_no_volume_warns():
    df = _make_df()
    df["volume"] = 0
    r = tv_health_check(df)
    assert r.passed  # warnings don't block
    assert any("volume" in w for w in r.warnings)


def test_temporal_gap_warns():
    df = _make_df(n=60)
    # Insert a 10-day gap between row 29 and 30
    df.loc[30:, "date"] = df.loc[30:, "date"] + pd.Timedelta(days=10)
    r = tv_health_check(df)
    assert any("gap" in w for w in r.warnings)


def test_extreme_move_warns():
    df = _make_df()
    df.loc[50, "close"] = df.loc[49, "close"] * 3.0  # 200% move
    r = tv_health_check(df)
    assert any("extrême" in w for w in r.warnings)


def test_small_nan_is_warning_not_error():
    df = _make_df(n=100)
    # 3 NaN = 3% < 5% threshold → warning only
    df.loc[1:3, "close"] = np.nan
    r = tv_health_check(df)
    assert r.passed
    assert any("NaN" in w for w in r.warnings)
