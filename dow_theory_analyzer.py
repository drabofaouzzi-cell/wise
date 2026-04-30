"""
Dow Theory Analyzer
Analyse les données de prix exportées depuis TradingView selon les 6 principes de la théorie de Dow.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyArrowPatch
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# Contrôle qualité des données TV
# ─────────────────────────────────────────────

@dataclass
class HealthCheckResult:
    passed: bool = True
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def tv_health_check(df: pd.DataFrame) -> HealthCheckResult:
    """
    Vérifie la qualité des données OHLCV TradingView avant l'analyse.
    Les erreurs sont bloquantes (l'analyse ne doit pas continuer) ; les avertissements sont indicatifs.
    """
    result = HealthCheckResult()
    n = len(df)

    # ── Statistiques de base ──
    result.stats["nb_lignes"] = n
    if n > 0 and "date" in df.columns:
        result.stats["date_debut"] = str(df["date"].iloc[0])[:10]
        result.stats["date_fin"] = str(df["date"].iloc[-1])[:10]

    # ── Nombre de lignes minimum ──
    if n < 10:
        result.errors.append(f"Données insuffisantes : {n} ligne(s) (minimum 10 requis pour toute analyse)")
        result.passed = False
        return result  # inutile de continuer
    if n < 30:
        result.warnings.append(f"Seulement {n} lignes — la détection de phase requiert ≥30 bougies")
    elif n < 100:
        result.warnings.append(f"Seulement {n} lignes — la fiabilité de l'analyse augmente avec plus de données")

    # ── Colonnes obligatoires ──
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        result.errors.append(f"Colonnes manquantes : {sorted(missing)}")
        result.passed = False
        return result

    # ── Valeurs NaN dans les colonnes de prix ──
    price_cols = ["open", "high", "low", "close"]
    for col in price_cols:
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            pct = nan_count / n * 100
            if pct > 5:
                result.errors.append(f"Colonne '{col}' : {nan_count} valeur(s) NaN ({pct:.1f}%) — trop de prix manquants")
                result.passed = False
            else:
                result.warnings.append(f"Colonne '{col}' : {nan_count} valeur(s) NaN ({pct:.1f}%)")

    if not result.passed:
        return result

    # ── Prix nuls ou négatifs ──
    for col in price_cols:
        non_pos = (df[col] <= 0).sum()
        if non_pos > 0:
            result.errors.append(f"Colonne '{col}' : {non_pos} valeur(s) non positive(s) (prix nul ou négatif)")
            result.passed = False

    # ── Cohérence OHLC : high >= low ──
    invalid_hl = (df["high"] < df["low"]).sum()
    if invalid_hl > 0:
        result.errors.append(f"{invalid_hl} bougie(s) avec high < low (données OHLC corrompues)")
        result.passed = False

    # ── Cohérence OHLC : high >= max(open, close) et low <= min(open, close) ──
    tol = 1e-8
    invalid_high = ((df["high"] + tol) < df[["open", "close"]].max(axis=1)).sum()
    invalid_low = ((df["low"] - tol) > df[["open", "close"]].min(axis=1)).sum()
    if invalid_high > 0:
        result.errors.append(f"{invalid_high} bougie(s) avec high < max(open, close)")
        result.passed = False
    if invalid_low > 0:
        result.errors.append(f"{invalid_low} bougie(s) avec low > min(open, close)")
        result.passed = False

    # ── Horodatages dupliqués ──
    if "date" in df.columns:
        dupes = df["date"].duplicated().sum()
        if dupes > 0:
            result.warnings.append(f"{dupes} horodatage(s) dupliqué(s) détecté(s)")

    # ── Lacunes temporelles ──
    if "date" in df.columns and n >= 2:
        deltas = df["date"].diff().dropna()
        median_delta = deltas.median()
        if median_delta.total_seconds() > 0:
            gap_threshold = median_delta * 5
            large_gaps = (deltas > gap_threshold).sum()
            if large_gaps > 0:
                result.warnings.append(
                    f"{large_gaps} lacune(s) importante(s) dans la série temporelle (>{int(gap_threshold.total_seconds() / 3600)}h entre deux bougies)"
                )

    # ── Volume ──
    if "volume" not in df.columns or df["volume"].sum() == 0:
        result.warnings.append("Aucune donnée de volume — les signaux basés sur le volume seront indisponibles")
    else:
        zero_vol = (df["volume"] == 0).sum()
        if zero_vol / n > 0.2:
            result.warnings.append(f"{zero_vol} bougie(s) ({zero_vol/n*100:.0f}%) avec volume nul")

    # ── Mouvements de bougie extrêmes (corps > 30% du prix) ──
    body_pct = (df["close"] - df["open"]).abs() / df["open"] * 100
    extreme = (body_pct > 30).sum()
    if extreme > 0:
        result.warnings.append(f"{extreme} bougie(s) avec un corps >30% — anomalie possible ou ajustement de cours")

    return result


def print_health_report(result: HealthCheckResult) -> None:
    sep = "═" * 60
    status = "✅ VALIDÉ" if result.passed else "❌ ÉCHEC"
    print(f"\n{sep}")
    print(f"  CONTRÔLE QUALITÉ DONNÉES TV — {status}")
    print(sep)

    stats = result.stats
    if "nb_lignes" in stats:
        periode = f"{stats.get('date_debut', '?')} → {stats.get('date_fin', '?')}"
        print(f"  Lignes : {stats['nb_lignes']}   |   Période : {periode}")

    if result.errors:
        print("\n  ERREURS (analyse bloquée) :")
        for e in result.errors:
            print(f"    ✗  {e}")

    if result.warnings:
        print("\n  AVERTISSEMENTS :")
        for w in result.warnings:
            print(f"    ⚠  {w}")

    if not result.errors and not result.warnings:
        print("\n  Aucun problème détecté — données propres.")

    print(f"{sep}\n")


# ─────────────────────────────────────────────
# Types / Enums
# ─────────────────────────────────────────────

class Trend(Enum):
    BULL = "Haussier"
    BEAR = "Baissier"
    NEUTRAL = "Neutre / Indéfini"


class Phase(Enum):
    ACCUMULATION = "Accumulation"
    PARTICIPATION = "Participation du public"
    DISTRIBUTION = "Distribution / Excès"
    CAPITULATION = "Capitulation / Panique"
    UNKNOWN = "Indéterminée"


@dataclass
class SwingPoint:
    index: int
    date: pd.Timestamp
    price: float
    kind: str  # "high" | "low"


@dataclass
class TrendAnalysis:
    primary: Trend = Trend.NEUTRAL
    secondary: Trend = Trend.NEUTRAL
    phase: Phase = Phase.UNKNOWN
    swing_highs: list = field(default_factory=list)
    swing_lows: list = field(default_factory=list)
    volume_confirms: bool = False
    last_signal: str = ""
    support: float = 0.0
    resistance: float = 0.0
    score: int = 0          # score haussier : +1 par critère validé, -1 baissier
    signals: list = field(default_factory=list)


# ─────────────────────────────────────────────
# Chargement des données TradingView
# ─────────────────────────────────────────────

def load_tradingview_csv(path: str) -> pd.DataFrame:
    """
    Lit un CSV exporté depuis TradingView.
    Colonnes attendues : time, open, high, low, close, volume
    """
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]

    # Détection colonne date
    time_col = next((c for c in df.columns if c in ("time", "date", "datetime", "timestamp")), None)
    if time_col is None:
        raise ValueError("Aucune colonne de date trouvée. Colonnes disponibles : " + str(df.columns.tolist()))
    df.rename(columns={time_col: "date"}, inplace=True)

    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    else:
        df["volume"] = 0

    return df


# ─────────────────────────────────────────────
# Détection des swing points (pivots)
# ─────────────────────────────────────────────

def find_swing_points(df: pd.DataFrame, window: int = 5) -> tuple[list[SwingPoint], list[SwingPoint]]:
    """
    Détecte les pivots hauts et bas locaux avec une fenêtre glissante.
    """
    highs, lows = [], []
    n = len(df)
    for i in range(window, n - window):
        hi = df["high"].iloc[i]
        lo = df["low"].iloc[i]
        if hi == df["high"].iloc[i - window: i + window + 1].max():
            highs.append(SwingPoint(i, df["date"].iloc[i], hi, "high"))
        if lo == df["low"].iloc[i - window: i + window + 1].min():
            lows.append(SwingPoint(i, df["date"].iloc[i], lo, "low"))
    return highs, lows


def _trend_from_swings(points: list[SwingPoint]) -> Trend:
    """Détermine la tendance à partir d'une série de pivots (3 derniers minimum)."""
    if len(points) < 3:
        return Trend.NEUTRAL
    recent = points[-4:]
    prices = [p.price for p in recent]
    # Vérification HH/HL (haussier) ou LH/LL (baissier)
    bull_count = sum(prices[i] > prices[i - 1] for i in range(1, len(prices)))
    bear_count = sum(prices[i] < prices[i - 1] for i in range(1, len(prices)))
    if bull_count >= len(prices) - 1:
        return Trend.BULL
    if bear_count >= len(prices) - 1:
        return Trend.BEAR
    return Trend.NEUTRAL


# ─────────────────────────────────────────────
# Analyse du volume
# ─────────────────────────────────────────────

def analyze_volume(df: pd.DataFrame, primary_trend: Trend) -> bool:
    """
    Principe 4 : Le volume doit confirmer la tendance.
    - Hausse : volume plus fort sur les rallyes que sur les corrections.
    - Baisse : volume plus fort sur les baisses que sur les rebonds.
    """
    if df["volume"].sum() == 0:
        return False  # Pas de données volume

    up_days = df[df["close"] > df["open"]]
    down_days = df[df["close"] <= df["open"]]
    avg_vol_up = up_days["volume"].mean() if len(up_days) else 0
    avg_vol_down = down_days["volume"].mean() if len(down_days) else 0

    if primary_trend == Trend.BULL:
        return avg_vol_up > avg_vol_down
    elif primary_trend == Trend.BEAR:
        return avg_vol_down > avg_vol_up
    return False


# ─────────────────────────────────────────────
# Détection de phase (marché bull)
# ─────────────────────────────────────────────

def detect_phase(df: pd.DataFrame, primary_trend: Trend) -> Phase:
    """
    Principe 2 : Les tendances primaires ont 3 phases.
    Approche : analyse de la volatilité, du volume et de la performance sur sous-périodes.
    """
    n = len(df)
    if n < 30:
        return Phase.UNKNOWN

    third = n // 3
    p1 = df.iloc[:third]
    p2 = df.iloc[third: 2 * third]
    p3 = df.iloc[2 * third:]

    perf = [
        (s.iloc[-1]["close"] - s.iloc[0]["close"]) / s.iloc[0]["close"]
        for s in [p1, p2, p3]
    ]
    vol_ratio = [s["volume"].mean() / (df["volume"].mean() + 1e-9) for s in [p1, p2, p3]]

    if primary_trend == Trend.BULL:
        # Accumulation : perf modeste, vol faible → Participation : perf forte, vol croissant → Distribution : vol élevé, perf ralentit
        if perf[0] < perf[1] and vol_ratio[1] > vol_ratio[0]:
            if vol_ratio[2] > vol_ratio[1] and perf[2] < perf[1]:
                return Phase.DISTRIBUTION
            return Phase.PARTICIPATION
        return Phase.ACCUMULATION
    elif primary_trend == Trend.BEAR:
        if perf[0] < 0 and perf[1] < perf[0]:
            return Phase.CAPITULATION
        return Phase.DISTRIBUTION
    return Phase.UNKNOWN


# ─────────────────────────────────────────────
# Niveaux support / résistance
# ─────────────────────────────────────────────

def compute_sr_levels(highs: list[SwingPoint], lows: list[SwingPoint]) -> tuple[float, float]:
    if not lows:
        support = 0.0
    else:
        support = min(p.price for p in lows[-3:])
    if not highs:
        resistance = 0.0
    else:
        resistance = max(p.price for p in highs[-3:])
    return support, resistance


# ─────────────────────────────────────────────
# Moteur principal
# ─────────────────────────────────────────────

def analyze(df: pd.DataFrame, swing_window: int = 5) -> TrendAnalysis:
    result = TrendAnalysis()

    # Swing points
    highs, lows = find_swing_points(df, window=swing_window)
    result.swing_highs = highs
    result.swing_lows = lows

    # Tendance primaire (données complètes)
    primary_h = _trend_from_swings(highs)
    primary_l = _trend_from_swings(lows)
    if primary_h == primary_l:
        result.primary = primary_h
    elif Trend.NEUTRAL in (primary_h, primary_l):
        result.primary = primary_h if primary_l == Trend.NEUTRAL else primary_l
    else:
        result.primary = Trend.NEUTRAL

    # Tendance secondaire (dernier 20% des données)
    cutoff = int(len(df) * 0.80)
    df_sec = df.iloc[cutoff:].reset_index(drop=True)
    h2, l2 = find_swing_points(df_sec, window=max(3, swing_window // 2))
    result.secondary = _trend_from_swings(h2) if len(h2) >= 3 else _trend_from_swings(l2)

    # Volume
    result.volume_confirms = analyze_volume(df, result.primary)

    # Phase
    result.phase = detect_phase(df, result.primary)

    # Support / Résistance
    result.support, result.resistance = compute_sr_levels(highs, lows)

    # Score global Dow (sur 6 principes)
    score = 0
    signals = []

    # 1. Tendance primaire définie
    if result.primary != Trend.NEUTRAL:
        score += 1 if result.primary == Trend.BULL else -1
        signals.append(f"✅ Tendance primaire {result.primary.value}")
    else:
        signals.append("⚠️  Tendance primaire indéfinie (marché sans direction claire)")

    # 2. Phase identifiée
    signals.append(f"📊 Phase : {result.phase.value}")

    # 3. Tendance secondaire confirme / contredit
    if result.secondary == result.primary:
        score += 1 if result.primary == Trend.BULL else -1
        signals.append(f"✅ Tendance secondaire confirme ({result.secondary.value})")
    elif result.secondary != Trend.NEUTRAL:
        score -= 1 if result.primary == Trend.BULL else -1
        signals.append(f"⚠️  Tendance secondaire {result.secondary.value} — correction possible")

    # 4. Volume
    if result.volume_confirms:
        score += 1 if result.primary == Trend.BULL else -1
        signals.append("✅ Volume confirme la tendance principale")
    else:
        signals.append("⚠️  Volume ne confirme pas la tendance (signal de faiblesse)")

    # 5. HH/HL ou LH/LL cohérents
    if len(highs) >= 2 and len(lows) >= 2:
        hh = highs[-1].price > highs[-2].price
        hl = lows[-1].price > lows[-2].price
        ll = lows[-1].price < lows[-2].price
        lh = highs[-1].price < highs[-2].price
        if result.primary == Trend.BULL and hh and hl:
            score += 1
            signals.append("✅ Structure HH/HL intacte (bull)")
        elif result.primary == Trend.BEAR and ll and lh:
            score += 1  # confirmation bear
            score -= 2  # mais mauvais pour le portefeuille
            signals.append("✅ Structure LH/LL intacte (bear)")
        else:
            signals.append("⚠️  Structure de pivots mixte — prudence")

    # 6. Dernier prix vs support/résistance
    last_close = df["close"].iloc[-1]
    if result.resistance > 0 and last_close >= result.resistance * 0.98:
        signals.append(f"⚠️  Prix proche de la résistance ({result.resistance:.4f})")
        score -= 1
    elif result.support > 0 and last_close <= result.support * 1.02:
        signals.append(f"⚠️  Prix proche du support ({result.support:.4f})")
    else:
        signals.append(f"✅ Prix dans la zone intermédiaire ({last_close:.4f})")

    result.score = score
    result.signals = signals

    # Signal de synthèse
    if score >= 3:
        result.last_signal = "ACHAT / LONG — Confluence haussière forte"
    elif score <= -3:
        result.last_signal = "VENTE / SHORT — Confluence baissière forte"
    elif score > 0:
        result.last_signal = "BIAIS HAUSSIER — Attendre confirmation"
    elif score < 0:
        result.last_signal = "BIAIS BAISSIER — Attendre confirmation"
    else:
        result.last_signal = "NEUTRE — Aucune direction claire"

    return result


# ─────────────────────────────────────────────
# Rapport texte
# ─────────────────────────────────────────────

def print_report(df: pd.DataFrame, result: TrendAnalysis, symbol: str = "Actif"):
    start = df["date"].iloc[0].strftime("%Y-%m-%d")
    end = df["date"].iloc[-1].strftime("%Y-%m-%d")
    last = df["close"].iloc[-1]
    change = (last - df["close"].iloc[0]) / df["close"].iloc[0] * 100

    sep = "═" * 60
    print(f"\n{sep}")
    print(f"  ANALYSE DOW THEORY — {symbol}")
    print(f"  Période : {start} → {end}  ({len(df)} bougies)")
    print(f"  Dernier cours : {last:.4f}  ({change:+.2f}% sur la période)")
    print(sep)

    print("\n📌 TENDANCES")
    print(f"  Primaire  : {result.primary.value}")
    print(f"  Secondaire: {result.secondary.value}")
    print(f"  Phase     : {result.phase.value}")

    print("\n📌 NIVEAUX CLÉS")
    print(f"  Support   : {result.support:.4f}")
    print(f"  Résistance: {result.resistance:.4f}")

    print("\n📌 SIGNAUX DOW (6 principes)")
    for s in result.signals:
        print(f"  {s}")

    print(f"\n{'─'*60}")
    bar = ("█" * max(0, result.score + 6)).ljust(12)
    print(f"  Score Dow : {result.score:+d}/6   [{bar}]")
    print(f"\n  🎯 SIGNAL : {result.last_signal}")
    print(f"{sep}\n")


# ─────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────

def plot_analysis(df: pd.DataFrame, result: TrendAnalysis, symbol: str = "Actif", save_path: Optional[str] = None):
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), gridspec_kw={"height_ratios": [4, 1.5, 1]})
    fig.patch.set_facecolor("#0d1117")
    for ax in axes:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#30363d")

    dates = df["date"]
    ax1, ax2, ax3 = axes

    # ── Prix + chandeliers simplifiés ──
    colors = ["#26a641" if c >= o else "#f85149" for c, o in zip(df["close"], df["open"])]
    ax1.bar(dates, df["high"] - df["low"], bottom=df["low"], width=0.6, color="#555", alpha=0.4)
    ax1.bar(dates, (df["close"] - df["open"]).abs(), bottom=df[["open", "close"]].min(axis=1),
            width=0.6, color=colors, alpha=0.9)

    # ── Swing points ──
    for sp in result.swing_highs[-20:]:
        ax1.plot(sp.date, sp.price, "v", color="#f0a500", markersize=7, alpha=0.9)
    for sp in result.swing_lows[-20:]:
        ax1.plot(sp.date, sp.price, "^", color="#58a6ff", markersize=7, alpha=0.9)

    # ── Support / Résistance ──
    if result.support:
        ax1.axhline(result.support, color="#58a6ff", linestyle="--", linewidth=1.2, alpha=0.7, label=f"Support {result.support:.4f}")
    if result.resistance:
        ax1.axhline(result.resistance, color="#f0a500", linestyle="--", linewidth=1.2, alpha=0.7, label=f"Résistance {result.resistance:.4f}")

    # ── Moyennes mobiles 50 / 200 ──
    for period, color in [(50, "#a371f7"), (200, "#ff7b72")]:
        if len(df) > period:
            ma = df["close"].rolling(period).mean()
            ax1.plot(dates, ma, color=color, linewidth=1, alpha=0.8, label=f"MA{period}")

    trend_color = "#26a641" if result.primary == Trend.BULL else ("#f85149" if result.primary == Trend.BEAR else "#8b949e")
    ax1.set_title(
        f"Théorie de Dow — {symbol}  |  Tendance: {result.primary.value}  |  Phase: {result.phase.value}",
        color="white", fontsize=13, pad=10
    )
    ax1.legend(loc="upper left", facecolor="#21262d", edgecolor="#30363d", labelcolor="white", fontsize=8)
    ax1.yaxis.label.set_color("white")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    # ── Volume ──
    vol_colors = ["#26a641" if c >= o else "#f85149" for c, o in zip(df["close"], df["open"])]
    ax2.bar(dates, df["volume"], color=vol_colors, alpha=0.7)
    if len(df) > 20:
        ax2.plot(dates, df["volume"].rolling(20).mean(), color="#f0a500", linewidth=1.2, label="Vol MA20")
    vol_label = "✅ Volume confirme" if result.volume_confirms else "⚠️ Volume diverge"
    ax2.set_title(f"Volume  —  {vol_label}", color="white", fontsize=10)
    ax2.legend(loc="upper left", facecolor="#21262d", edgecolor="#30363d", labelcolor="white", fontsize=8)

    # ── Score Dow ──
    score_colors = ["#26a641" if i < max(0, result.score + 3) else "#f85149" if i >= max(0, result.score + 3) else "#555"
                    for i in range(6)]
    bars = ["P1\nTendance", "P2\nPhase", "P3\nConf.", "P4\nVolume", "P5\nPivots", "P6\nNiveaux"]
    bar_vals = [1] * 6
    bar_clrs = []
    for i, sig in enumerate(result.signals[:6]):
        if sig.startswith("✅"):
            bar_clrs.append("#26a641")
        elif sig.startswith("⚠️"):
            bar_clrs.append("#f0a500")
        else:
            bar_clrs.append("#8b949e")

    ax3.bar(bars[:len(bar_clrs)], bar_vals[:len(bar_clrs)], color=bar_clrs, alpha=0.85)
    ax3.set_yticks([])
    ax3.set_title(
        f"Principes Dow  |  Score: {result.score:+d}/6  |  Signal: {result.last_signal}",
        color="white", fontsize=10
    )
    for spine in ax3.spines.values():
        spine.set_visible(False)
    ax3.tick_params(axis="x", colors="white", labelsize=8)

    plt.tight_layout(pad=2)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  📁 Graphique sauvegardé : {save_path}")
    else:
        plt.show()
    plt.close()
