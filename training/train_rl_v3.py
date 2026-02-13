"""
T08 v3 – Entraînement RL anti-overtrading
Min hold period + spread cost réaliste + reward simplifiée.

Usage :
  python training/train_rl_v3.py
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.trading_env_v3 import TradingEnvV3

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.callbacks import BaseCallback
except ImportError:
    print("stable-baselines3 non installe.")
    sys.exit(1)

DATA_DIR = PROJECT_ROOT / "data" / "m15"
MODEL_DIR = PROJECT_ROOT / "models" / "v3"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RL_CONFIG = {
    "algorithm": "PPO",
    "total_timesteps": 500_000,
    "learning_rate": 5e-5,       # très bas = très stable
    "gamma": 0.999,              # horizon très long
    "batch_size": 512,
    "n_steps": 4096,             # beaucoup de données par update
    "n_epochs": 3,               # peu d'epochs
    "clip_range": 0.1,           # clip serré
    "ent_coef": 0.001,           # peu d'exploration (forcer exploitation)
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "gae_lambda": 0.98,
    "seed": 42,
    "window_size": 30,
    "spread": 0.0003,
    "min_hold": 4,
    "policy_kwargs": {
        "net_arch": {"pi": [128, 64], "vf": [128, 64]},
    },
}


class EvalCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq=20_000, save_path=None, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.save_path = save_path
        self.best_profit = -np.inf

    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            obs, _ = self.eval_env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated

            perf = self.eval_env.get_performance_summary()
            print(f"  [Step {self.n_calls:>7}] "
                  f"Profit: {perf['profit_pct']:+.2f}% | "
                  f"Trades: {perf['nb_trades']} | "
                  f"WR: {perf['win_rate']:.0f}% | "
                  f"MaxDD: {perf['max_drawdown_pct']:.1f}% | "
                  f"AvgHold: {perf['avg_hold']:.0f}", flush=True)

            if perf["profit_pct"] > self.best_profit and self.save_path:
                self.best_profit = perf["profit_pct"]
                self.model.save(str(self.save_path / "ppo_trading_best"))
                print(f"    -> BEST ({perf['profit_pct']:+.2f}%)", flush=True)

        return True


def load_data(year):
    return pd.read_csv(DATA_DIR / f"GBPUSD_M15_{year}.csv",
                       parse_dates=["timestamp"], index_col="timestamp")


def evaluate(model, df, config, label=""):
    env = TradingEnvV3(
        df=df,
        window_size=config["window_size"],
        spread=config["spread"],
        min_hold=config["min_hold"],
    )
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    perf = env.get_performance_summary()
    print(f"\n  {label}:")
    print(f"     Profit    : {perf['profit_pct']:+.2f}%")
    print(f"     Equity    : {perf['final_equity']:,.2f}")
    print(f"     Max DD    : {perf['max_drawdown_pct']:.2f}%")
    print(f"     Trades    : {perf['nb_trades']}")
    print(f"     Win Rate  : {perf['win_rate']:.1f}%")
    print(f"     Avg Win   : {perf['avg_win']:+.3f}%")
    print(f"     Avg Loss  : {perf['avg_loss']:+.3f}%")
    print(f"     Avg Hold  : {perf['avg_hold']:.0f} bars", flush=True)
    return perf


def train():
    print("=" * 60)
    print("T08 v3 - Entrainement RL (anti-overtrading)")
    print("=" * 60, flush=True)

    print("\nChargement des donnees...", flush=True)
    df_train = load_data(2022)
    df_val = load_data(2023)
    print(f"  Train (2022) : {len(df_train):,} bougies")
    print(f"  Val   (2023) : {len(df_val):,} bougies", flush=True)

    env_train = TradingEnvV3(
        df=df_train,
        window_size=RL_CONFIG["window_size"],
        spread=RL_CONFIG["spread"],
        min_hold=RL_CONFIG["min_hold"],
    )
    env_val = TradingEnvV3(
        df=df_val,
        window_size=RL_CONFIG["window_size"],
        spread=RL_CONFIG["spread"],
        min_hold=RL_CONFIG["min_hold"],
    )

    print("  Verification...", flush=True)
    check_env(env_train, warn=True)
    print("  OK", flush=True)

    print(f"\nConfig PPO v3:", flush=True)
    for k, v in RL_CONFIG.items():
        if k != "policy_kwargs":
            print(f"     {k}: {v}")
    print(f"     net_arch: {RL_CONFIG['policy_kwargs']['net_arch']}", flush=True)

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

    callback = EvalCallback(
        eval_env=env_val,
        eval_freq=20_000,
        save_path=MODEL_DIR,
    )

    print(f"\nEntrainement ({RL_CONFIG['total_timesteps']:,} timesteps)...", flush=True)
    model.learn(total_timesteps=RL_CONFIG["total_timesteps"], callback=callback)

    model.save(str(MODEL_DIR / "ppo_trading"))
    print(f"\n  Modele final : {MODEL_DIR / 'ppo_trading'}", flush=True)

    # Charger le meilleur
    best_path = MODEL_DIR / "ppo_trading_best.zip"
    if best_path.exists():
        print("  Chargement du meilleur modele...", flush=True)
        model = PPO.load(str(MODEL_DIR / "ppo_trading_best"))

    print(f"\n{'=' * 50}")
    print("  EVALUATION FINALE")
    print(f"{'=' * 50}", flush=True)

    results = {}
    for year, label in [(2022, "2022 (Train)"), (2023, "2023 (Val)"), (2024, "2024 (Test)")]:
        df = load_data(year)
        results[year] = evaluate(model, df, RL_CONFIG, label)

    config_save = {k: v for k, v in RL_CONFIG.items() if k != "policy_kwargs"}
    config_save["net_arch"] = str(RL_CONFIG["policy_kwargs"]["net_arch"])
    with open(MODEL_DIR / "rl_config.json", "w") as f:
        json.dump(config_save, f, indent=2)
    with open(MODEL_DIR / "eval_results.json", "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)

    print(f"\n{'=' * 60}")
    print("Entrainement v3 termine!")
    print(f"{'=' * 60}", flush=True)


if __name__ == "__main__":
    train()
