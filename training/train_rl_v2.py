"""
T08 v2 – Entraînement RL amélioré
Plus de timesteps, meilleure architecture, meilleur reward shaping.

Usage :
  python training/train_rl_v2.py
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

# ── Setup ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.trading_env_v2 import TradingEnvV2

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv
except ImportError:
    print("stable-baselines3 non installe. pip install stable-baselines3")
    sys.exit(1)

# ── Config ──────────────────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data" / "m15"
MODEL_DIR = PROJECT_ROOT / "models" / "v2"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RL_CONFIG = {
    "algorithm": "PPO",
    "total_timesteps": 500_000,
    "learning_rate": 1e-4,       # plus bas = plus stable
    "gamma": 0.995,              # horizon plus long
    "batch_size": 256,           # plus gros batch
    "n_steps": 2048,             # plus de steps par update
    "n_epochs": 5,               # moins d'epochs (évite overfitting)
    "clip_range": 0.15,          # clip plus serré
    "ent_coef": 0.005,           # exploration modérée
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "gae_lambda": 0.95,
    "seed": 42,
    "window_size": 30,
    "transaction_cost": 0.0001,
    "policy_kwargs": {
        "net_arch": {
            "pi": [256, 128, 64],  # policy network
            "vf": [256, 128, 64],  # value network
        }
    },
}


# ── Callback ────────────────────────────────────────────────────────────
class EvalCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq=20_000, save_path=None, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.save_path = save_path
        self.best_profit = -np.inf
        self.results = []

    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            obs, _ = self.eval_env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated

            perf = self.eval_env.get_performance_summary()
            self.results.append({"step": self.n_calls, **perf})

            print(f"  [Step {self.n_calls:>7}] "
                  f"Profit: {perf['profit_pct']:+.2f}% | "
                  f"Trades: {perf['nb_trades']} | "
                  f"WR: {perf['win_rate']:.0f}% | "
                  f"MaxDD: {perf['max_drawdown_pct']:.1f}%")

            # Sauvegarder le meilleur modèle
            if perf["profit_pct"] > self.best_profit and self.save_path:
                self.best_profit = perf["profit_pct"]
                self.model.save(str(self.save_path / "ppo_trading_best"))
                print(f"    -> Nouveau meilleur modele sauvegarde! ({perf['profit_pct']:+.2f}%)")

        return True


# ── Fonctions ───────────────────────────────────────────────────────────
def load_data(year: int) -> pd.DataFrame:
    path = DATA_DIR / f"GBPUSD_M15_{year}.csv"
    return pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")


def evaluate_model(model, df, config, label=""):
    """Evalue le modele sur un dataset."""
    env = TradingEnvV2(
        df=df,
        window_size=config["window_size"],
        transaction_cost=config["transaction_cost"],
    )
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    perf = env.get_performance_summary()
    print(f"\n  {label}:")
    print(f"     Profit      : {perf['profit_pct']:+.2f}%")
    print(f"     Equity      : {perf['final_equity']:,.2f}")
    print(f"     Max DD      : {perf['max_drawdown_pct']:.2f}%")
    print(f"     Nb Trades   : {perf['nb_trades']}")
    print(f"     Win Rate    : {perf['win_rate']:.1f}%")
    print(f"     Avg Win     : {perf['avg_win']:+.3f}%")
    print(f"     Avg Loss    : {perf['avg_loss']:+.3f}%")
    return perf


def train():
    print("=" * 60)
    print("T08 v2 - Entrainement RL ameliore (PPO)")
    print("=" * 60)

    # ── Data ────────────────────────────────────────────────
    print("\nChargement des donnees...")
    df_train = load_data(2022)
    df_val = load_data(2023)
    df_test = load_data(2024)
    print(f"  Train (2022) : {len(df_train):,} bougies")
    print(f"  Val   (2023) : {len(df_val):,} bougies")
    print(f"  Test  (2024) : {len(df_test):,} bougies")

    # ── Environnements ──────────────────────────────────────
    print("\nCreation des environnements v2...")
    env_train = TradingEnvV2(
        df=df_train,
        window_size=RL_CONFIG["window_size"],
        transaction_cost=RL_CONFIG["transaction_cost"],
    )
    env_val = TradingEnvV2(
        df=df_val,
        window_size=RL_CONFIG["window_size"],
        transaction_cost=RL_CONFIG["transaction_cost"],
    )

    print("  Verification gymnasium...")
    check_env(env_train, warn=True)
    print("  Environnement v2 valide")

    # ── Config PPO ──────────────────────────────────────────
    print(f"\nConfiguration PPO v2:")
    for k, v in RL_CONFIG.items():
        if k != "policy_kwargs":
            print(f"     {k}: {v}")
    print(f"     net_arch: pi={RL_CONFIG['policy_kwargs']['net_arch']['pi']}, "
          f"vf={RL_CONFIG['policy_kwargs']['net_arch']['vf']}")

    model = PPO(
        "MlpPolicy",
        env_train,
        learning_rate=RL_CONFIG["learning_rate"],
        gamma=RL_CONFIG["gamma"],
        batch_size=RL_CONFIG["batch_size"],
        n_steps=RL_CONFIG["n_steps"],
        n_epochs=RL_CONFIG["n_epochs"],
        clip_range=RL_CONFIG["clip_range"],
        ent_coef=RL_CONFIG["ent_coef"],
        vf_coef=RL_CONFIG["vf_coef"],
        max_grad_norm=RL_CONFIG["max_grad_norm"],
        gae_lambda=RL_CONFIG["gae_lambda"],
        seed=RL_CONFIG["seed"],
        policy_kwargs={"net_arch": RL_CONFIG["policy_kwargs"]["net_arch"]},
        verbose=0,
    )

    # ── Callback ────────────────────────────────────────────
    callback = EvalCallback(
        eval_env=env_val,
        eval_freq=20_000,
        save_path=MODEL_DIR,
    )

    # ── Entrainement ────────────────────────────────────────
    print(f"\nEntrainement ({RL_CONFIG['total_timesteps']:,} timesteps)...")
    model.learn(
        total_timesteps=RL_CONFIG["total_timesteps"],
        callback=callback,
    )

    # Sauvegarder le modèle final
    model.save(str(MODEL_DIR / "ppo_trading"))
    print(f"\n  Modele final sauvegarde : {MODEL_DIR / 'ppo_trading'}")

    # ── Evaluation finale ───────────────────────────────────
    # Charger le meilleur modèle si disponible
    best_path = MODEL_DIR / "ppo_trading_best.zip"
    if best_path.exists():
        print("\n  Chargement du meilleur modele (validation)...")
        model = PPO.load(str(MODEL_DIR / "ppo_trading_best"))

    print(f"\n{'=' * 50}")
    print("  EVALUATION FINALE (meilleur modele)")
    print(f"{'=' * 50}")

    results = {}
    for year, label in [(2022, "2022 (Train)"), (2023, "2023 (Validation)"), (2024, "2024 (Test)")]:
        df = load_data(year)
        perf = evaluate_model(model, df, RL_CONFIG, label)
        results[year] = perf

    # ── Sauvegarder config + résultats ──────────────────────
    config_save = {k: v for k, v in RL_CONFIG.items() if k != "policy_kwargs"}
    config_save["net_arch_pi"] = RL_CONFIG["policy_kwargs"]["net_arch"]["pi"]
    config_save["net_arch_vf"] = RL_CONFIG["policy_kwargs"]["net_arch"]["vf"]

    with open(MODEL_DIR / "rl_config.json", "w") as f:
        json.dump(config_save, f, indent=2)

    # Sauvegarder les résultats d'évaluation
    with open(MODEL_DIR / "eval_results.json", "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)

    print(f"\n{'=' * 60}")
    print("Entrainement v2 termine!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    train()
