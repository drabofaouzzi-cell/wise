"""
Point d'entrée — Analyse Dow Theory sur données TradingView.

Usage :
    python run_analysis.py --file data.csv --symbol BTCUSDT
    python run_analysis.py --file data.csv --symbol EURUSD --window 8 --save chart.png
    python run_analysis.py --demo          # génère des données synthétiques pour tester
    python run_analysis.py --demo --check  # contrôle qualité des données uniquement

Export TradingView :
    Ouvre le graphique → clic droit sur la bougie → "Télécharger les données du graphique"
    Le CSV contient : time, open, high, low, close, volume
"""

import argparse
import numpy as np
import pandas as pd
from dow_theory_analyzer import (
    load_tradingview_csv,
    analyze,
    print_report,
    plot_analysis,
    tv_health_check,
    print_health_report,
)


# ─────────────────────────────────────────────
# Données de démo (synthétiques)
# ─────────────────────────────────────────────

def generate_demo_data(n: int = 300) -> pd.DataFrame:
    """Génère un marché EUR/USD synthétique avec tendance haussière et correction."""
    np.random.seed(42)
    # Utilise uniquement les jours ouvrés (lundi–vendredi), comme le forex
    dates = pd.bdate_range("2023-01-01", periods=n, tz="UTC")
    price = 1.0800  # cours de départ réaliste EUR/USD
    prices = []
    for i in range(n):
        drift = 0.00003 if i < n * 0.65 else -0.00002  # mouvement typique forex
        ret = drift + np.random.normal(0, 0.003)        # volatilité ~0.3%/jour
        price *= (1 + ret)
        prices.append(price)

    closes = np.array(prices)
    opens = np.roll(closes, 1)
    opens[0] = closes[0] * 0.9999
    # Mèches : ~0.05% au-dessus/en-dessous du corps, cohérent avec le forex
    highs = np.maximum(opens, closes) * (1 + np.abs(np.random.normal(0, 0.0005, n)))
    lows = np.minimum(opens, closes) * (1 - np.abs(np.random.normal(0, 0.0005, n)))
    # Le volume forex sur TradingView est un volume tick, pas un volume réel
    volumes = np.random.lognormal(8, 0.4, n) * (1 + 0.2 * (closes > opens).astype(float))

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
    parser.add_argument("--check", action="store_true", help="Afficher uniquement le rapport de santé des données (sans analyse)")
    parser.add_argument("--forex", action="store_true", help="Mode forex : ajuste les seuils pour les paires de devises (ex: EURUSD)")
    args = parser.parse_args()

    if args.demo:
        print("\n[MODE DÉMO] Génération de données synthétiques EUR/USD…")
        df = generate_demo_data()
        symbol = args.symbol if args.symbol != "Actif" else "EURUSD"
    elif args.file:
        print(f"\nChargement : {args.file}")
        df = load_tradingview_csv(args.file)
        symbol = args.symbol
    else:
        parser.print_help()
        print("\n⚠️  Fournissez --file <csv> ou --demo pour tester.\n")
        return

    seuil_corps = 2.0 if args.forex else 30.0
    health = tv_health_check(df, seuil_corps_pct=seuil_corps)
    print_health_report(health)

    if args.check:
        return

    if not health.passed:
        print("⛔  Analyse interrompue : corrigez les erreurs ci-dessus avant de continuer.\n")
        return

    result = analyze(df, swing_window=args.window)
    print_report(df, result, symbol=symbol)
    plot_analysis(df, result, symbol=symbol, save_path=args.save)


if __name__ == "__main__":
    main()
