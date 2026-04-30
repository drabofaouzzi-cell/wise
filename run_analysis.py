"""
Point d'entrée — Analyse Dow Theory sur données TradingView.

Usage :
    python run_analysis.py --file data.csv --symbol BTCUSDT
    python run_analysis.py --file data.csv --symbol EURUSD --window 8 --save chart.png
    python run_analysis.py --demo   # génère des données synthétiques pour tester

Export TradingView :
    Ouvre le graphique → clic droit sur la bougie → "Download chart data"
    Le CSV contient : time, open, high, low, close, volume
"""

import argparse
import numpy as np
import pandas as pd
from dow_theory_analyzer import (
    load_tradingview_csv,
    tv_health_check,
    print_health_report,
    analyze,
    print_report,
    plot_analysis,
)


# ─────────────────────────────────────────────
# Données de démo (synthétiques)
# ─────────────────────────────────────────────

def generate_demo_data(n: int = 300) -> pd.DataFrame:
    """Génère un marché bull avec correction intermédiaire."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")
    price = 100.0
    prices = []
    for i in range(n):
        drift = 0.0004 if i < n * 0.65 else -0.0002  # correction finale
        ret = drift + np.random.normal(0, 0.015)
        price *= (1 + ret)
        prices.append(price)

    closes = np.array(prices)
    opens = np.roll(closes, 1)
    opens[0] = closes[0] * 0.999
    highs = np.maximum(opens, closes) * (1 + np.abs(np.random.normal(0, 0.005, n)))
    lows = np.minimum(opens, closes) * (1 - np.abs(np.random.normal(0, 0.005, n)))
    volumes = np.random.lognormal(10, 0.5, n) * (1 + 0.3 * (closes > opens).astype(float))

    return pd.DataFrame({
        "date": dates,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Dow Theory Analyzer — TradingView CSV")
    parser.add_argument("--file", type=str, help="Chemin vers le CSV TradingView")
    parser.add_argument("--symbol", type=str, default="Actif", help="Nom de l'actif (ex: BTCUSDT)")
    parser.add_argument("--window", type=int, default=5, help="Fenêtre swing points (défaut: 5)")
    parser.add_argument("--save", type=str, default=None, help="Sauvegarder le graphique (ex: chart.png)")
    parser.add_argument("--demo", action="store_true", help="Lancer avec données synthétiques de démo")
    parser.add_argument("--no-health-check", action="store_true", help="Ignorer le contrôle qualité des données")
    args = parser.parse_args()

    if args.demo:
        print("\n[MODE DÉMO] Génération de données synthétiques…")
        df = generate_demo_data()
        symbol = "DEMO_ASSET"
    elif args.file:
        print(f"\nChargement : {args.file}")
        df = load_tradingview_csv(args.file)
        symbol = args.symbol
    else:
        parser.print_help()
        print("\n⚠️  Fournissez --file <csv> ou --demo pour tester.\n")
        return

    if not args.no_health_check:
        health = tv_health_check(df)
        print_health_report(health)
        if not health.passed:
            print("❌ Contrôle qualité échoué — analyse annulée. Corrigez les erreurs ou utilisez --no-health-check.\n")
            return

    result = analyze(df, swing_window=args.window)
    print_report(df, result, symbol=symbol)
    plot_analysis(df, result, symbol=symbol, save_path=args.save)


if __name__ == "__main__":
    main()
