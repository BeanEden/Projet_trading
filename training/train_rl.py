"""
T08 – Entraînement agent RL
Entraîne un agent PPO sur l'environnement de trading GBP/USD M15.

Usage :
  python training/train_rl.py
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

# ── Setup ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.trading_env import TradingEnv

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.callbacks import BaseCallback
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    print("⚠ stable-baselines3 non installé. Exécutez : pip install stable-baselines3")

# ── Config ──────────────────────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data" / "m15"
MODEL_DIR = PROJECT_ROOT / "models" / "v1"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparamètres RL
RL_CONFIG = {
    "algorithm": "PPO",
    "total_timesteps": 100_000,
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "batch_size": 64,
    "n_epochs": 10,
    "clip_range": 0.2,
    "ent_coef": 0.01,        # encourager l'exploration
    "seed": 42,
    "window_size": 20,
    "transaction_cost": 0.0002,
    "drawdown_penalty": 0.5,
}


# ── Callback pour suivi ────────────────────────────────────────────────────
class TradingCallback(BaseCallback):
    """Callback pour afficher les performances pendant l'entraînement."""

    def __init__(self, eval_env=None, eval_freq=10_000, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.best_reward = -np.inf

    def _on_step(self):
        if self.eval_env and self.n_calls % self.eval_freq == 0:
            obs, _ = self.eval_env.reset()
            total_reward = 0
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                total_reward += reward
                done = terminated or truncated

            perf = self.eval_env.get_performance_summary()
            print(f"  [Step {self.n_calls:>7}] "
                  f"Profit: {perf['profit_pct']:+.2f}% | "
                  f"Trades: {perf['nb_trades']} | "
                  f"Reward: {total_reward:.4f}")

            if total_reward > self.best_reward:
                self.best_reward = total_reward

        return True


# ── Fonctions ───────────────────────────────────────────────────────────────
def load_data(year: int) -> pd.DataFrame:
    """Charge les données M15 pour une année."""
    path = DATA_DIR / f"GBPUSD_M15_{year}.csv"
    df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
    return df


def train_agent():
    """Entraîne l'agent PPO."""
    print("=" * 60)
    print("T08 – Entraînement RL (PPO)")
    print("=" * 60)

    # Charger les données
    print("\n📂 Chargement des données...")
    df_train = load_data(2022)
    df_val = load_data(2023)
    print(f"  Train (2022) : {len(df_train):,} bougies")
    print(f"  Val   (2023) : {len(df_val):,} bougies")

    # Créer les environnements
    print("\n🏗️ Création des environnements...")
    env_train = TradingEnv(
        df=df_train,
        window_size=RL_CONFIG["window_size"],
        transaction_cost=RL_CONFIG["transaction_cost"],
        drawdown_penalty=RL_CONFIG["drawdown_penalty"],
    )
    env_val = TradingEnv(
        df=df_val,
        window_size=RL_CONFIG["window_size"],
        transaction_cost=RL_CONFIG["transaction_cost"],
        drawdown_penalty=RL_CONFIG["drawdown_penalty"],
    )

    # Vérification de l'environnement
    print("  Vérification gymnasium...")
    check_env(env_train, warn=True)
    print("  ✅ Environnement valide")

    # Créer le modèle PPO
    print(f"\n🤖 Configuration PPO:")
    for k, v in RL_CONFIG.items():
        print(f"     {k}: {v}")

    model = PPO(
        "MlpPolicy",
        env_train,
        learning_rate=RL_CONFIG["learning_rate"],
        gamma=RL_CONFIG["gamma"],
        batch_size=RL_CONFIG["batch_size"],
        n_epochs=RL_CONFIG["n_epochs"],
        clip_range=RL_CONFIG["clip_range"],
        ent_coef=RL_CONFIG["ent_coef"],
        seed=RL_CONFIG["seed"],
        verbose=0,
    )

    # Callback d'évaluation
    callback = TradingCallback(
        eval_env=env_val,
        eval_freq=10_000,
    )

    # Entraînement
    print(f"\n🚀 Entraînement ({RL_CONFIG['total_timesteps']:,} timesteps)...")
    model.learn(
        total_timesteps=RL_CONFIG["total_timesteps"],
        callback=callback,
    )

    # Sauvegarde
    model_path = MODEL_DIR / "ppo_trading"
    model.save(str(model_path))
    print(f"\n  ✓ Modèle sauvegardé : {model_path}")

    # Évaluation finale
    print(f"\n{'═' * 50}")
    print("  ÉVALUATION FINALE")
    print(f"{'═' * 50}")

    for year, label in [(2022, "Train"), (2023, "Validation"), (2024, "Test")]:
        df = load_data(year)
        env = TradingEnv(
            df=df,
            window_size=RL_CONFIG["window_size"],
            transaction_cost=RL_CONFIG["transaction_cost"],
            drawdown_penalty=RL_CONFIG["drawdown_penalty"],
        )

        obs, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        perf = env.get_performance_summary()
        print(f"\n  📊 {year} ({label}):")
        print(f"     Profit      : {perf['profit_pct']:+.2f}%")
        print(f"     Equity      : {perf['final_equity']:,.2f}")
        print(f"     Max DD      : {perf['max_drawdown_pct']:.2f}%")
        print(f"     Nb Trades   : {perf['nb_trades']}")
        print(f"     Total Reward: {total_reward:.4f}")

    # Sauvegarder la config
    import json
    config_path = MODEL_DIR / "rl_config.json"
    with open(config_path, "w") as f:
        json.dump(RL_CONFIG, f, indent=2)
    print(f"\n  ✓ Config sauvegardée : {config_path}")

    print(f"\n{'=' * 60}")
    print("✅ Entraînement RL terminé!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    if not HAS_SB3:
        print("❌ Installez stable-baselines3 : pip install stable-baselines3")
        sys.exit(1)
    train_agent()
