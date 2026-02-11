"""
T02 – Agrégation M1 → M15
Transforme les données brutes GBP/USD en bougies 15 minutes.

Règles :
  open_15m  = open de la 1ère minute
  high_15m  = max(high) sur 15 minutes
  low_15m   = min(low) sur 15 minutes
  close_15m = close de la dernière minute

Usage :
  python training/aggregate_m15.py
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIRS = {
    2022: PROJECT_ROOT / "HISTDATA_COM_MT_GBPUSD_M12022" / "DAT_MT_GBPUSD_M1_2022.csv",
    2023: PROJECT_ROOT / "HISTDATA_COM_MT_GBPUSD_M12023" / "DAT_MT_GBPUSD_M1_2023.csv",
    2024: PROJECT_ROOT / "HISTDATA_COM_MT_GBPUSD_M12024" / "DAT_MT_GBPUSD_M1_2024.csv",
}
OUTPUT_DIR = PROJECT_ROOT / "data" / "m15"
M1_COLUMNS = ["date", "time", "open", "high", "low", "close", "volume"]


def load_m1(filepath: Path) -> pd.DataFrame:
    """Charge un CSV M1 brut et crée un index DatetimeIndex."""
    df = pd.read_csv(filepath, header=None, names=M1_COLUMNS)

    # Fusion date + time → timestamp
    df["timestamp"] = pd.to_datetime(df["date"] + " " + df["time"], format="%Y.%m.%d %H:%M")
    df = df.set_index("timestamp").sort_index()

    # Suppression colonnes texte
    df = df.drop(columns=["date", "time"])

    print(f"  ✓ Chargé {filepath.name}: {len(df):,} lignes "
          f"({df.index.min()} → {df.index.max()})")

    return df


def check_m1_quality(df: pd.DataFrame, year: int) -> dict:
    """Vérifie la qualité des données M1."""
    stats = {}

    # Vérification prix négatifs
    neg_prices = (df[["open", "high", "low", "close"]] < 0).any().any()
    stats["prix_negatifs"] = neg_prices
    if neg_prices:
        print(f"  ⚠ Prix négatifs détectés en {year}!")

    # Vérification régularité (écarts > 1 minute)
    time_diffs = df.index.to_series().diff().dropna()
    gaps = time_diffs[time_diffs > pd.Timedelta(minutes=2)]
    stats["nb_gaps"] = len(gaps)
    stats["max_gap"] = time_diffs.max()
    print(f"  ℹ {len(gaps)} gaps > 2min (max: {time_diffs.max()})")

    # Vérification high >= low
    invalid_bars = (df["high"] < df["low"]).sum()
    stats["bars_high_lt_low"] = invalid_bars
    if invalid_bars > 0:
        print(f"  ⚠ {invalid_bars} bougies avec high < low!")

    return stats


def aggregate_m1_to_m15(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrège les données M1 en bougies M15.

    Règles du sujet :
    - open_15m  : open de la première minute du bloc
    - high_15m  : max(high) sur le bloc de 15 minutes
    - low_15m   : min(low) sur le bloc de 15 minutes
    - close_15m : close de la dernière minute du bloc
    """
    agg_rules = {
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }

    df_m15 = df.resample("15min").agg(agg_rules)

    # Renommer les colonnes selon le format imposé
    df_m15 = df_m15.rename(columns={
        "open":   "open_15m",
        "high":   "high_15m",
        "low":    "low_15m",
        "close":  "close_15m",
        "volume": "volume_15m",
    })

    # Supprimer les bougies vides (pas de données M1 dans la fenêtre)
    df_m15 = df_m15.dropna(subset=["open_15m", "close_15m"])

    return df_m15


def filter_incomplete_candles(df_m15: pd.DataFrame, df_m1: pd.DataFrame,
                               min_ticks: int = 5) -> pd.DataFrame:
    """
    Supprime les bougies M15 qui contiennent trop peu de ticks M1.
    En heures de marché normales, on attend ~15 ticks par bougie.
    On utilise un seuil minimal (default 5) pour garder les bougies
    en début/fin de session.
    """
    counts = df_m1.resample("15min")["close"].count()
    counts.name = "tick_count"

    df_m15 = df_m15.join(counts)

    # Filtrer les bougies avec trop peu de données
    incomplete = df_m15["tick_count"] < min_ticks
    n_removed = incomplete.sum()
    if n_removed > 0:
        print(f"  ℹ {n_removed} bougies M15 supprimées (< {min_ticks} ticks M1)")

    df_m15 = df_m15[~incomplete].drop(columns=["tick_count"])

    return df_m15


def validate_m15(df_m15: pd.DataFrame, year: int):
    """Validation post-agrégation."""
    print(f"\n  📊 Validation M15 {year}:")
    print(f"     Lignes : {len(df_m15):,}")
    print(f"     Période : {df_m15.index.min()} → {df_m15.index.max()}")
    print(f"     Prix moyen : {df_m15['close_15m'].mean():.5f}")
    print(f"     Min/Max : {df_m15['low_15m'].min():.5f} / {df_m15['high_15m'].max():.5f}")

    # Vérification cohérence prix
    assert (df_m15["high_15m"] >= df_m15["low_15m"]).all(), "Erreur: high < low dans M15!"
    assert (df_m15["high_15m"] >= df_m15["open_15m"]).all(), "Erreur: high < open dans M15!"
    assert (df_m15["high_15m"] >= df_m15["close_15m"]).all(), "Erreur: high < close dans M15!"
    assert (df_m15["low_15m"] <= df_m15["open_15m"]).all(), "Erreur: low > open dans M15!"
    assert (df_m15["low_15m"] <= df_m15["close_15m"]).all(), "Erreur: low > close dans M15!"
    print("     ✅ Toutes les vérifications passées")


def main():
    print("=" * 60)
    print("T02 – Agrégation M1 → M15 (GBP/USD)")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for year, filepath in DATA_RAW_DIRS.items():
        print(f"\n{'─' * 40}")
        print(f"  Année {year}")
        print(f"{'─' * 40}")

        if not filepath.exists():
            print(f"  ❌ Fichier introuvable : {filepath}")
            continue

        # 1. Charger M1
        df_m1 = load_m1(filepath)

        # 2. Vérification qualité M1
        check_m1_quality(df_m1, year)

        # 3. Agrégation M1 → M15
        df_m15 = aggregate_m1_to_m15(df_m1)
        print(f"  ✓ Agrégation : {len(df_m1):,} M1 → {len(df_m15):,} M15")

        # 4. Filtrer bougies incomplètes
        df_m15 = filter_incomplete_candles(df_m15, df_m1)

        # 5. Validation
        validate_m15(df_m15, year)

        # 6. Sauvegarde
        output_path = OUTPUT_DIR / f"GBPUSD_M15_{year}.csv"
        df_m15.to_csv(output_path, index=True, index_label="timestamp")
        print(f"  ✓ Sauvegardé : {output_path}")

    print(f"\n{'=' * 60}")
    print("✅ Agrégation terminée avec succès!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
