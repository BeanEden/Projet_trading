"""
T08 v4 – Entraînement RL (pas de stop-loss, log-return, données combinées)

Usage :
  python training/train_rl_v4.py
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.trading_env_v4 import TradingEnvV4

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.callbacks import BaseCallback
except ImportError:
    print("stable-baselines3 non installe.")
    sys.exit(1)

DATA_DIR = PROJECT_ROOT / "data" / "m15"
MODEL_DIR = PROJECT_ROOT / "models" / "v4"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RL_CONFIG = {
    "algorithm": "PPO",
    "total_timesteps": 500_000,
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "batch_size": 256,
    "n_steps": 2048,
    "n_epochs": 10,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "gae_lambda": 0.95,
    "seed": 42,
    "window_size": 30,
    "spread": 0.00015,
    "min_hold": 8,
    "net_arch": [128, 64],
}


class EvalCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq=25_000, save_path=None, verbose=1):
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
                obs, _, terminated, truncated, _ = self.eval_env.step(action)
                done = terminated or truncated

            p = self.eval_env.get_performance_summary()
            print(f"  [Step {self.n_calls:>7}] "
                  f"Profit: {p['profit_pct']:+.2f}% | "
                  f"Trades: {p['nb_trades']} | "
                  f"WR: {p['win_rate']:.0f}% | "
                  f"MaxDD: {p['max_drawdown_pct']:.1f}%", flush=True)

            if p["profit_pct"] > self.best_profit and self.save_path:
                self.best_profit = p["profit_pct"]
                self.model.save(str(self.save_path / "ppo_trading_best"))
                print(f"    -> BEST ({p['profit_pct']:+.2f}%)", flush=True)
        return True


def load_data(year):
    return pd.read_csv(DATA_DIR / f"GBPUSD_M15_{year}.csv",
                       parse_dates=["timestamp"], index_col="timestamp")


def evaluate(model, df, config, label=""):
    env = TradingEnvV4(
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

    p = env.get_performance_summary()
    print(f"\n  {label}:")
    print(f"     Profit    : {p['profit_pct']:+.2f}%")
    print(f"     Equity    : {p['final_equity']:,.2f}")
    print(f"     Max DD    : {p['max_drawdown_pct']:.2f}%")
    print(f"     Trades    : {p['nb_trades']}")
    print(f"     Win Rate  : {p['win_rate']:.1f}%")
    print(f"     Avg Win   : {p['avg_win']:+.3f}%")
    print(f"     Avg Loss  : {p['avg_loss']:+.3f}%", flush=True)
    return p


def train():
    print("=" * 60)
    print("T08 v4 - Entrainement RL (log-return, no stop-loss)")
    print("=" * 60, flush=True)

    # Charger données
    print("\nChargement...", flush=True)
    df_2022 = load_data(2022)
    df_2023 = load_data(2023)
    df_test = load_data(2024)

    # Combiner 2022+2023 pour l'entraînement
    df_train = pd.concat([df_2022, df_2023], ignore_index=False)
    print(f"  Train (2022+2023) : {len(df_train):,} bougies")
    print(f"  Test  (2024)      : {len(df_test):,} bougies", flush=True)

    # Environments
    env_train = TradingEnvV4(
        df=df_train,
        window_size=RL_CONFIG["window_size"],
        spread=RL_CONFIG["spread"],
        min_hold=RL_CONFIG["min_hold"],
    )
    env_test = TradingEnvV4(
        df=df_test,
        window_size=RL_CONFIG["window_size"],
        spread=RL_CONFIG["spread"],
        min_hold=RL_CONFIG["min_hold"],
    )

    check_env(env_train, warn=True)
    print("  Env OK", flush=True)

    print(f"\nConfig:", flush=True)
    for k, v in RL_CONFIG.items():
        print(f"     {k}: {v}")

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
        policy_kwargs={"net_arch": RL_CONFIG["net_arch"]},
        verbose=0,
    )

    callback = EvalCallback(
        eval_env=env_test,  # Évaluer directement sur 2024
        eval_freq=25_000,
        save_path=MODEL_DIR,
    )

    print(f"\nEntrainement ({RL_CONFIG['total_timesteps']:,} timesteps)...", flush=True)
    model.learn(total_timesteps=RL_CONFIG["total_timesteps"], callback=callback)

    model.save(str(MODEL_DIR / "ppo_trading"))

    # Charger le meilleur
    best_path = MODEL_DIR / "ppo_trading_best.zip"
    if best_path.exists():
        model = PPO.load(str(MODEL_DIR / "ppo_trading_best"))
        print("\n  Meilleur modele charge.", flush=True)

    print(f"\n{'=' * 50}")
    print("  EVALUATION FINALE")
    print(f"{'=' * 50}", flush=True)

    results = {}
    for year, label in [(2022, "2022"), (2023, "2023"), (2024, "2024 (Test)")]:
        df = load_data(year)
        results[year] = evaluate(model, df, RL_CONFIG, label)

    with open(MODEL_DIR / "rl_config.json", "w") as f:
        json.dump(RL_CONFIG, f, indent=2)
    with open(MODEL_DIR / "eval_results.json", "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)

    print(f"\n{'=' * 60}")
    print("v4 termine!")
    print(f"{'=' * 60}", flush=True)


if __name__ == "__main__":
    train()
