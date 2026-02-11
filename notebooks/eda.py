"""
T04 – Analyse exploratoire + ADF/ACF
Analyse complète des données M15 GBP/USD (2022-2024).

Génère :
  - Distribution des rendements
  - Volatilité dans le temps
  - Analyse horaire
  - Autocorrélation (ACF/PACF)
  - Test ADF (stationnarité)

Usage :
  python notebooks/eda.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as sp_stats
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "m15"
OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "eda"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

YEARS = [2022, 2023, 2024]
LABELS = {2022: "Train", 2023: "Validation", 2024: "Test"}

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("deep")
FIGSIZE = (14, 7)
DPI = 150


# ── Chargement données ─────────────────────────────────────────────────────
def load_all_m15() -> pd.DataFrame:
    """Charge et concatène tous les CSV M15."""
    frames = []
    for year in YEARS:
        path = DATA_DIR / f"GBPUSD_M15_{year}.csv"
        if not path.exists():
            print(f"  ⚠ Fichier introuvable : {path}")
            continue
        df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
        df["year"] = year
        frames.append(df)
        print(f"  ✓ {year}: {len(df):,} lignes")

    df_all = pd.concat(frames)
    df_all = df_all.sort_index()

    # Calcul des rendements
    df_all["return_15m"] = df_all["close_15m"].pct_change()
    df_all["log_return"] = np.log(df_all["close_15m"] / df_all["close_15m"].shift(1))

    return df_all


# ── 1. Distribution des rendements ─────────────────────────────────────────
def plot_return_distribution(df: pd.DataFrame):
    """Histogramme des rendements + QQ plot."""
    returns = df["return_15m"].dropna()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Histogramme global
    axes[0].hist(returns, bins=150, density=True, alpha=0.7, color="steelblue",
                 edgecolor="white", linewidth=0.3)
    x = np.linspace(returns.min(), returns.max(), 200)
    axes[0].plot(x, sp_stats.norm.pdf(x, returns.mean(), returns.std()),
                 "r-", linewidth=2, label="Loi normale")
    axes[0].set_title("Distribution des rendements M15", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Rendement")
    axes[0].set_ylabel("Densité")
    axes[0].legend()

    # Histogramme par année
    for year in YEARS:
        r = df[df["year"] == year]["return_15m"].dropna()
        axes[1].hist(r, bins=100, density=True, alpha=0.5,
                     label=f"{year} ({LABELS[year]})")
    axes[1].set_title("Distribution par année", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Rendement")
    axes[1].legend()

    # QQ Plot
    sp_stats.probplot(returns, dist="norm", plot=axes[2])
    axes[2].set_title("QQ Plot vs Normale", fontsize=13, fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_distribution_rendements.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("  ✓ 01_distribution_rendements.png")

    # Statistiques descriptives
    print("\n  📊 Statistiques des rendements:")
    print(f"     Moyenne     : {returns.mean():.8f}")
    print(f"     Écart-type  : {returns.std():.6f}")
    print(f"     Skewness    : {returns.skew():.4f}")
    print(f"     Kurtosis    : {returns.kurtosis():.4f}")
    print(f"     Min         : {returns.min():.6f}")
    print(f"     Max         : {returns.max():.6f}")

    # Test de normalité (Jarque-Bera)
    jb_stat, jb_pval = sp_stats.jarque_bera(returns)
    print(f"     Jarque-Bera : stat={jb_stat:.2f}, p-val={jb_pval:.2e}")
    if jb_pval < 0.05:
        print("     → Rendements NON normaux (rejet H0)")
    else:
        print("     → Rendements normaux (H0 non rejeté)")


# ── 2. Volatilité dans le temps ────────────────────────────────────────────
def plot_volatility(df: pd.DataFrame):
    """Rolling volatility + volatilité mensuelle."""
    fig, axes = plt.subplots(2, 1, figsize=FIGSIZE, height_ratios=[2, 1])

    # Rolling std (20 bougies ≈ 5h de marché)
    rolling_vol = df["return_15m"].rolling(window=20).std()
    axes[0].plot(df.index, rolling_vol, linewidth=0.5, color="steelblue", alpha=0.8)
    axes[0].fill_between(df.index, 0, rolling_vol, alpha=0.2, color="steelblue")
    axes[0].set_title("Volatilité glissante (rolling std 20 périodes)", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Écart-type")

    # Coloriser par année
    for year in YEARS:
        mask = df["year"] == year
        axes[0].axvspan(df[mask].index.min(), df[mask].index.max(),
                        alpha=0.05, color="gray")
        mid_date = df[mask].index[len(df[mask]) // 2]
        axes[0].annotate(f"{year}\n({LABELS[year]})", xy=(mid_date, rolling_vol.max() * 0.9),
                         ha="center", fontsize=10, fontweight="bold")

    # Volatilité mensuelle (bar chart)
    monthly_vol = df["return_15m"].resample("ME").std()
    colors = []
    for date in monthly_vol.index:
        if date.year == 2022:
            colors.append("steelblue")
        elif date.year == 2023:
            colors.append("darkorange")
        else:
            colors.append("green")
    axes[1].bar(monthly_vol.index, monthly_vol.values, width=25, color=colors, alpha=0.7)
    axes[1].set_title("Volatilité mensuelle", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Écart-type")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_volatilite.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("  ✓ 02_volatilite.png")


# ── 3. Analyse horaire ─────────────────────────────────────────────────────
def plot_hourly_analysis(df: pd.DataFrame):
    """Rendement moyen et volatilité par heure."""
    df_with_hour = df.copy()
    df_with_hour["hour"] = df_with_hour.index.hour

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)

    # Rendement moyen par heure
    hourly_return = df_with_hour.groupby("hour")["return_15m"].mean()
    colors_ret = ["green" if r > 0 else "red" for r in hourly_return]
    axes[0].bar(hourly_return.index, hourly_return.values * 10000,
                color=colors_ret, alpha=0.7)
    axes[0].set_title("Rendement moyen par heure (en bps)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Heure (UTC)")
    axes[0].set_ylabel("Rendement (basis points)")
    axes[0].axhline(y=0, color="black", linewidth=0.5)
    axes[0].set_xticks(range(24))

    # Volatilité par heure
    hourly_vol = df_with_hour.groupby("hour")["return_15m"].std()
    axes[1].bar(hourly_vol.index, hourly_vol.values * 10000,
                color="steelblue", alpha=0.7)
    axes[1].set_title("Volatilité par heure (en bps)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Heure (UTC)")
    axes[1].set_ylabel("Écart-type (basis points)")
    axes[1].set_xticks(range(24))

    # Zones de marché
    for ax in axes:
        ax.axvspan(7, 16, alpha=0.05, color="blue", label="Session Londres")
        ax.axvspan(13, 21, alpha=0.05, color="red", label="Session New York")

    axes[0].legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_analyse_horaire.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("  ✓ 03_analyse_horaire.png")


# ── 4. Autocorrélation (ACF/PACF) ──────────────────────────────────────────
def plot_autocorrelation(df: pd.DataFrame):
    """ACF et PACF des rendements."""
    returns = df["return_15m"].dropna()

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # ACF des rendements
    plot_acf(returns, lags=50, ax=axes[0, 0], title="ACF – Rendements M15")
    axes[0, 0].set_xlabel("Lag (bougies M15)")

    # PACF des rendements
    plot_pacf(returns, lags=50, ax=axes[0, 1], title="PACF – Rendements M15",
              method="ywm")
    axes[0, 1].set_xlabel("Lag (bougies M15)")

    # ACF des rendements au carré (effet ARCH)
    returns_sq = returns ** 2
    plot_acf(returns_sq, lags=50, ax=axes[1, 0],
             title="ACF – Rendements² (clustering de volatilité)")
    axes[1, 0].set_xlabel("Lag (bougies M15)")

    # ACF des rendements absolus
    returns_abs = returns.abs()
    plot_acf(returns_abs, lags=50, ax=axes[1, 1],
             title="ACF – |Rendements| (persistance volatilité)")
    axes[1, 1].set_xlabel("Lag (bougies M15)")

    plt.suptitle("Autocorrélation des rendements GBP/USD M15",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_autocorrelation.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("  ✓ 04_autocorrelation.png")


# ── 5. Test ADF (stationnarité) ────────────────────────────────────────────
def test_adf(df: pd.DataFrame):
    """Test Augmented Dickey-Fuller sur les prix et les rendements."""
    print("\n" + "=" * 60)
    print("TEST ADF (Augmented Dickey-Fuller)")
    print("=" * 60)
    print("H0 : La série possède une racine unitaire (non stationnaire)")
    print("H1 : La série est stationnaire")
    print("-" * 60)

    results = {}

    series_to_test = {
        "Prix (close_15m)": df["close_15m"].dropna(),
        "Rendements": df["return_15m"].dropna(),
        "Log-rendements": df["log_return"].dropna(),
    }

    for name, series in series_to_test.items():
        result = adfuller(series, autolag="AIC")
        adf_stat, p_value, used_lag, nobs, crit_values, ic = result

        results[name] = {
            "adf_stat": adf_stat,
            "p_value": p_value,
            "used_lag": used_lag,
            "nobs": nobs,
            "crit_values": crit_values,
        }

        print(f"\n  📈 {name}:")
        print(f"     Statistique ADF : {adf_stat:.6f}")
        print(f"     p-value         : {p_value:.2e}")
        print(f"     Lags utilisés   : {used_lag}")
        print(f"     Observations    : {nobs}")
        print(f"     Valeurs critiques :")
        for level, value in crit_values.items():
            marker = "✅" if adf_stat < value else "❌"
            print(f"       {level}: {value:.6f}  {marker}")

        if p_value < 0.05:
            print(f"     → STATIONNAIRE (p < 0.05)")
        else:
            print(f"     → NON STATIONNAIRE (p >= 0.05)")

    return results


# ── 6. Prix et rendements dans le temps ────────────────────────────────────
def plot_price_overview(df: pd.DataFrame):
    """Vue d'ensemble des prix et rendements cumulés."""
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))

    # Prix
    axes[0].plot(df.index, df["close_15m"], linewidth=0.5, color="steelblue")
    axes[0].set_title("Prix GBP/USD (Close M15)", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Prix")

    # Rendements
    axes[1].plot(df.index, df["return_15m"], linewidth=0.3, color="gray", alpha=0.5)
    axes[1].set_title("Rendements M15", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Rendement")

    # Rendements cumulés
    cumret = (1 + df["return_15m"].fillna(0)).cumprod() - 1
    axes[2].plot(df.index, cumret * 100, linewidth=0.8, color="steelblue")
    axes[2].fill_between(df.index, 0, cumret * 100, alpha=0.15, color="steelblue")
    axes[2].set_title("Rendements cumulés (%)", fontsize=13, fontweight="bold")
    axes[2].set_ylabel("Rendement cumulé (%)")
    axes[2].axhline(y=0, color="black", linewidth=0.5)

    # Annotations année
    for ax in axes:
        for year in YEARS:
            mask = df["year"] == year
            if mask.any():
                ax.axvline(x=df[mask].index.min(), color="red", linestyle="--",
                           linewidth=0.8, alpha=0.5)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "00_prix_overview.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("  ✓ 00_prix_overview.png")


# ── 7. Rapport texte ───────────────────────────────────────────────────────
def generate_report(df: pd.DataFrame, adf_results: dict):
    """Génère un rapport texte de l'analyse exploratoire."""
    returns = df["return_15m"].dropna()

    report = []
    report.append("=" * 70)
    report.append("RAPPORT D'ANALYSE EXPLORATOIRE – GBP/USD M15")
    report.append("=" * 70)
    report.append("")

    report.append("1. DONNÉES")
    report.append(f"   Période        : {df.index.min()} → {df.index.max()}")
    report.append(f"   Observations   : {len(df):,}")
    for year in YEARS:
        n = (df["year"] == year).sum()
        report.append(f"   {year} ({LABELS[year]:>10}) : {n:,} bougies")
    report.append("")

    report.append("2. STATISTIQUES DES RENDEMENTS")
    report.append(f"   Moyenne        : {returns.mean():.8f}")
    report.append(f"   Écart-type     : {returns.std():.6f}")
    report.append(f"   Skewness       : {returns.skew():.4f}")
    report.append(f"   Kurtosis       : {returns.kurtosis():.4f}")
    report.append(f"   Min            : {returns.min():.6f}")
    report.append(f"   Max            : {returns.max():.6f}")
    jb_stat, jb_pval = sp_stats.jarque_bera(returns)
    report.append(f"   Jarque-Bera    : stat={jb_stat:.2f}, p={jb_pval:.2e}")
    normal = "OUI" if jb_pval >= 0.05 else "NON"
    report.append(f"   Normalité      : {normal}")
    report.append("")

    report.append("3. TEST ADF (STATIONNARITÉ)")
    for name, res in adf_results.items():
        stationary = "STATIONNAIRE" if res["p_value"] < 0.05 else "NON STATIONNAIRE"
        report.append(f"   {name:.<30} ADF={res['adf_stat']:.4f}  "
                       f"p={res['p_value']:.2e}  → {stationary}")
    report.append("")

    report.append("4. FIGURES GÉNÉRÉES")
    report.append("   00_prix_overview.png           – Prix et rendements cumulés")
    report.append("   01_distribution_rendements.png – Histogramme + QQ plot")
    report.append("   02_volatilite.png              – Volatilité glissante et mensuelle")
    report.append("   03_analyse_horaire.png         – Rendement/volatilité par heure")
    report.append("   04_autocorrelation.png         – ACF/PACF")
    report.append("")
    report.append("=" * 70)

    report_text = "\n".join(report)

    report_path = OUTPUT_DIR / "rapport_eda.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\n{report_text}")
    print(f"\n  ✓ Rapport sauvegardé : {report_path}")


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("T04 – Analyse exploratoire GBP/USD M15")
    print("=" * 60)

    # Charger les données
    print("\n📂 Chargement des données M15...")
    df = load_all_m15()
    print(f"  Total : {len(df):,} bougies")

    # 0. Vue d'ensemble
    print("\n📊 Vue d'ensemble des prix...")
    plot_price_overview(df)

    # 1. Distribution
    print("\n📊 Distribution des rendements...")
    plot_return_distribution(df)

    # 2. Volatilité
    print("\n📊 Volatilité...")
    plot_volatility(df)

    # 3. Analyse horaire
    print("\n📊 Analyse horaire...")
    plot_hourly_analysis(df)

    # 4. Autocorrélation
    print("\n📊 Autocorrélation...")
    plot_autocorrelation(df)

    # 5. Test ADF
    adf_results = test_adf(df)

    # 6. Rapport
    print("\n📝 Génération du rapport...")
    generate_report(df, adf_results)

    print(f"\n{'=' * 60}")
    print("✅ Analyse exploratoire terminée!")
    print(f"   Figures dans : {OUTPUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
