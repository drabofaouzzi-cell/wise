"""Tests pour tv_health_check()."""

import pandas as pd
import numpy as np
import pytest
from dow_theory_analyzer import tv_health_check, HealthCheckResult


def _make_df(n=100, seed=0):
    """Retourne un DataFrame OHLCV propre."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")
    close = 100.0 + np.cumsum(rng.normal(0, 1, n))
    close = np.abs(close) + 1  # garder positif
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.01, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.01, n))
    volume = rng.lognormal(10, 0.5, n)
    return pd.DataFrame({"date": dates, "open": open_, "high": high, "low": low, "close": close, "volume": volume})


# ── Cas valides ────────────────────────────────────────────────────────────────

def test_donnees_propres_validees():
    result = tv_health_check(_make_df(200))
    assert result.passed is True
    assert result.errors == []


def test_statistiques_renseignees():
    result = tv_health_check(_make_df(50))
    assert result.stats["nb_lignes"] == 50
    assert "date_debut" in result.stats
    assert "date_fin" in result.stats


# ── Nombre de lignes ───────────────────────────────────────────────────────────

def test_trop_peu_de_lignes_erreur():
    result = tv_health_check(_make_df(5))
    assert result.passed is False
    assert any("insuffisantes" in e.lower() for e in result.errors)


def test_peu_de_lignes_avertissement():
    result = tv_health_check(_make_df(15))
    assert result.passed is True
    assert any("30" in w for w in result.warnings)


def test_lignes_moderees_avertissement():
    result = tv_health_check(_make_df(50))
    assert result.passed is True
    assert any("fiabilité" in w for w in result.warnings)


# ── Colonnes manquantes ────────────────────────────────────────────────────────

def test_colonne_manquante_erreur():
    df = _make_df().drop(columns=["close"])
    result = tv_health_check(df)
    assert result.passed is False
    assert any("close" in e for e in result.errors)


# ── Valeurs NaN ────────────────────────────────────────────────────────────────

def test_peu_de_nan_avertissement():
    df = _make_df(100)
    df.loc[0, "close"] = np.nan
    result = tv_health_check(df)
    assert result.passed is True
    assert any("close" in w for w in result.warnings)


def test_beaucoup_de_nan_erreur():
    df = _make_df(100)
    df.loc[:10, "close"] = np.nan  # 11% → erreur
    result = tv_health_check(df)
    assert result.passed is False
    assert any("close" in e for e in result.errors)


# ── Prix nuls ou négatifs ──────────────────────────────────────────────────────

def test_prix_nul_erreur():
    df = _make_df(50)
    df.loc[5, "open"] = 0.0
    result = tv_health_check(df)
    assert result.passed is False
    assert any("non positive" in e.lower() for e in result.errors)


def test_prix_negatif_erreur():
    df = _make_df(50)
    df.loc[3, "low"] = -1.5
    result = tv_health_check(df)
    assert result.passed is False


# ── Cohérence OHLC ────────────────────────────────────────────────────────────

def test_high_inferieur_low_erreur():
    df = _make_df(50)
    df.loc[10, "high"] = df.loc[10, "low"] - 1
    result = tv_health_check(df)
    assert result.passed is False
    assert any("high < low" in e for e in result.errors)


def test_high_inferieur_close_erreur():
    df = _make_df(50)
    idx = 10
    df.loc[idx, "close"] = df.loc[idx, "high"] + 5
    result = tv_health_check(df)
    assert result.passed is False
    assert any("high < max" in e for e in result.errors)


def test_low_superieur_open_erreur():
    df = _make_df(50)
    idx = 10
    df.loc[idx, "open"] = df.loc[idx, "low"] - 5
    result = tv_health_check(df)
    assert result.passed is False
    assert any("low > min" in e for e in result.errors)


# ── Horodatages dupliqués ─────────────────────────────────────────────────────

def test_dates_dupliquees_avertissement():
    df = _make_df(50)
    df.loc[5, "date"] = df.loc[4, "date"]
    result = tv_health_check(df)
    assert any("dupliqué" in w.lower() for w in result.warnings)


# ── Lacunes temporelles ───────────────────────────────────────────────────────

def test_grande_lacune_avertissement():
    df = _make_df(50)
    # Insère une lacune de 60 jours après la ligne 20
    df.loc[20:, "date"] = df.loc[20:, "date"] + pd.Timedelta(days=60)
    result = tv_health_check(df)
    assert any("lacune" in w.lower() for w in result.warnings)


# ── Volume ────────────────────────────────────────────────────────────────────

def test_volume_absent_avertissement():
    df = _make_df(50).drop(columns=["volume"])
    result = tv_health_check(df)
    assert any("volume" in w.lower() for w in result.warnings)


def test_volume_nul_avertissement():
    df = _make_df(50)
    df["volume"] = 0
    result = tv_health_check(df)
    assert any("volume" in w.lower() for w in result.warnings)


def test_volume_majoritairement_nul_avertissement():
    df = _make_df(50)
    df.loc[:40, "volume"] = 0  # 82% de zéros
    result = tv_health_check(df)
    assert any("volume nul" in w for w in result.warnings)


# ── Mouvements extrêmes ───────────────────────────────────────────────────────

def test_bougie_extreme_avertissement():
    df = _make_df(50)
    df.loc[25, "close"] = df.loc[25, "open"] * 2.0  # corps à +100%
    # Corriger high pour rester cohérent
    df.loc[25, "high"] = df.loc[25, "close"] * 1.001
    result = tv_health_check(df)
    assert any("30%" in w for w in result.warnings)
