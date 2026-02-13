"""
T06 – Baseline strategies + Backtest
Compare 3 stratégies de référence sur GBP/USD M15 :
  1. Stratégie aléatoire (Random)
  2. Buy & Hold
  3. Règles fixes (EMA Cross + RSI)

Usage :
  python evaluation/baseline_strategies.py
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Setup path ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.backtester import Backtester

# ── Config ──────────────────────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data" / "m15"
OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "baseline_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

YEARS = {2022: "Train", 2023: "Validation", 2024: "Test", 2025: "External Validation"}
TRANSACTION_COST = 0.0002  # ~2 pips
SEED = 42
FIGSIZE = (16, 10)
DPI = 150


# ── Chargement données ─────────────────────────────────────────────────────
def load_m15(year: int) -> pd.DataFrame:
    """Charge un CSV M15."""
    path = DATA_DIR / f"GBPUSD_M15_{year}.csv"
    df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les indicateurs techniques nécessaires aux stratégies."""
    df = df.copy()
    # EMA
    df["ema_20"] = df["close_15m"].ewm(span=20, adjust=False).mean()
    df["ema_50"] = df["close_15m"].ewm(span=50, adjust=False).mean()

    # RSI 14
    delta = df["close_15m"].diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=14).mean()
    rs = gain / loss
    df["rsi_14"] = 100 - (100 / (1 + rs))

    return df


# ── Stratégies ──────────────────────────────────────────────────────────────
def strategy_random(n: int, seed: int = SEED) -> pd.Series:
    """Stratégie aléatoire : BUY/SELL/HOLD avec probabilités égales."""
    rng = np.random.RandomState(seed)
    signals = rng.choice([1, -1, 0], size=n)
    return pd.Series(signals, name="random")


def strategy_buy_and_hold(n: int) -> pd.Series:
    """Buy & Hold : BUY au premier pas, HOLD ensuite."""
    signals = np.zeros(n, dtype=int)
    signals[0] = 1  # BUY
    return pd.Series(signals, name="buy_hold")


def strategy_ema_cross_rsi(df: pd.DataFrame) -> pd.Series:
    """
    Stratégie à règles fixes :
    - BUY  si EMA20 > EMA50 et RSI < 70 (tendance haussière, pas suracheté)
    - SELL si EMA20 < EMA50 et RSI > 30 (tendance baissière, pas survendu)
    - HOLD sinon
    """
    signals = np.zeros(len(df), dtype=int)

    for i in range(1, len(df)):
        ema20 = df["ema_20"].iloc[i]
        ema50 = df["ema_50"].iloc[i]
        rsi = df["rsi_14"].iloc[i]

        if pd.isna(ema20) or pd.isna(ema50) or pd.isna(rsi):
            signals[i] = 0
            continue

        if ema20 > ema50 and rsi < 70:
            signals[i] = 1  # BUY
        elif ema20 < ema50 and rsi > 30:
            signals[i] = -1  # SELL
        else:
            signals[i] = 0  # HOLD

    return pd.Series(signals, index=df.index, name="ema_rsi")


# ── Exécution ───────────────────────────────────────────────────────────────
def run_all_baselines():
    """Exécute toutes les stratégies sur toutes les périodes."""
    bt = Backtester(transaction_cost=TRANSACTION_COST)
    all_results = {}

    strategies = {
        "Random": lambda df: strategy_random(len(df)),
        "Buy & Hold": lambda df: strategy_buy_and_hold(len(df)),
        "EMA Cross + RSI": lambda df: strategy_ema_cross_rsi(df),
    }

    for year, label in YEARS.items():
        print(f"\n{'═' * 50}")
        print(f"  Année {year} ({label})")
        print(f"{'═' * 50}")

        df = load_m15(year)
        df = add_indicators(df)
        prices = df["close_15m"]

        year_results = {}

        for strat_name, strat_fn in strategies.items():
            signals = strat_fn(df)
            if isinstance(signals, pd.Series) and signals.index is not df.index:
                signals.index = df.index

            result = bt.run(prices, signals)
            year_results[strat_name] = result

            m = result["metrics"]
            print(f"\n  📊 {strat_name}:")
            print(f"     Profit cumulé  : {m['profit_cumule_pct']:+.2f}%  ({m['profit_cumule_abs']:+.2f})")
            print(f"     Max Drawdown   : {m['max_drawdown_pct']:.2f}%")
            print(f"     Sharpe         : {m['sharpe']:.3f}")
            print(f"     Profit Factor  : {m['profit_factor']:.3f}")
            print(f"     Win Rate       : {m['win_rate']:.1f}%")
            print(f"     Nb Trades      : {m['nb_trades']}")

        all_results[year] = year_results

    return all_results


# ── Visualisation ───────────────────────────────────────────────────────────
def plot_equity_curves(all_results: dict):
    """Trace les courbes d'equity pour chaque année."""
    fig, axes = plt.subplots(1, len(YEARS), figsize=(5*len(YEARS), 6), sharey=False)
    colors = {"Random": "#e74c3c", "Buy & Hold": "#3498db", "EMA Cross + RSI": "#2ecc71"}

    for idx, (year, label) in enumerate(YEARS.items()):
        ax = axes[idx]
        year_results = all_results[year]

        for strat_name, result in year_results.items():
            equity = result["equity_curve"]
            # Normaliser à 100 pour comparaison
            equity_norm = equity / equity[0] * 100
            ax.plot(equity_norm, linewidth=1.2, label=strat_name,
                    color=colors.get(strat_name, "gray"))

        ax.axhline(y=100, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
        ax.set_title(f"{year} ({label})", fontsize=13, fontweight="bold")
        ax.set_xlabel("Bougie M15")
        ax.set_ylabel("Equity (base 100)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Comparaison des stratégies baseline – GBP/USD M15",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "equity_curves.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"\n  ✓ equity_curves.png")


def plot_metrics_comparison(all_results: dict):
    """Tableau de comparaison des métriques."""
    metrics_names = ["profit_cumule_pct", "max_drawdown_pct", "sharpe", "profit_factor"]
    metric_labels = ["Profit (%)", "Max DD (%)", "Sharpe", "Profit Factor"]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    strat_names = list(list(all_results.values())[0].keys())
    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    x = np.arange(len(YEARS))
    width = 0.25

    for m_idx, (metric, label) in enumerate(zip(metrics_names, metric_labels)):
        ax = axes[m_idx]
        for s_idx, strat in enumerate(strat_names):
            values = [all_results[y][strat]["metrics"][metric] for y in YEARS]
            bars = ax.bar(x + s_idx * width, values, width, label=strat,
                         color=colors[s_idx], alpha=0.8)

        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xticks(x + width)
        ax.set_xticklabels([f"{y}\n({l})" for y, l in YEARS.items()], fontsize=9)
        ax.axhline(y=0, color="black", linewidth=0.5)
        if m_idx == 0:
            ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

    plt.suptitle("Métriques de performance par stratégie et par année",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "metrics_comparison.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  ✓ metrics_comparison.png")


def generate_summary_table(all_results: dict):
    """Génère un tableau récapitulatif sauvegardé en CSV et affiché."""
    rows = []
    for year, label in YEARS.items():
        for strat, result in all_results[year].items():
            m = result["metrics"]
            rows.append({
                "Année": year,
                "Période": label,
                "Stratégie": strat,
                "Profit (%)": round(m["profit_cumule_pct"], 2),
                "Max DD (%)": round(m["max_drawdown_pct"], 2),
                "Sharpe": round(m["sharpe"], 3),
                "Profit Factor": round(m["profit_factor"], 3),
                "Win Rate (%)": round(m["win_rate"], 1),
                "Nb Trades": m["nb_trades"],
            })

    df_summary = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / "baseline_summary.csv"
    df_summary.to_csv(csv_path, index=False)

    print(f"\n{'═' * 80}")
    print("TABLEAU RÉCAPITULATIF")
    print(f"{'═' * 80}")
    print(df_summary.to_string(index=False))
    print(f"\n  ✓ Sauvegardé : {csv_path}")

    return df_summary


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("T06 – Baseline strategies + Backtest")
    print(f"     Transaction cost: {TRANSACTION_COST * 10000:.0f} pips")
    print("=" * 60)

    # Exécution
    all_results = run_all_baselines()

    # Visualisation
    print(f"\n{'─' * 40}")
    print("  Génération des graphiques...")
    plot_equity_curves(all_results)
    plot_metrics_comparison(all_results)

    # Résumé
    generate_summary_table(all_results)

    print(f"\n{'=' * 60}")
    print("✅ Baselines terminées!")
    print(f"   Résultats dans : {OUTPUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
