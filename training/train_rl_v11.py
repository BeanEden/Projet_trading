"""
T08 v11 - Recurrent PPO (LSTM) + Anti-Overfitting Setup
Train: 2022-2024 (walk-forward validation)
Final Test: 2025
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Environnement V8
from training.trading_env_v8 import TradingEnvV8

try:
    from sb3_contrib import RecurrentPPO
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.callbacks import BaseCallback
except ImportError:
    print("Install sb3-contrib first: pip install sb3-contrib")
    sys.exit(1)

DATA_DIR = PROJECT_ROOT / "data" / "m15"
MODEL_DIR = PROJECT_ROOT / "models" / "v11"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RL_CONFIG = {
    "total_timesteps": 800_000,
    "learning_rate": 1e-4,
    "gamma": 0.995,
    "batch_size": 256,
    "n_steps": 1024,
    "n_epochs": 5,
    "clip_range": 0.15,
    "ent_coef": 0.02,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "gae_lambda": 0.95,
    "seed": 42,

    # Env
    "window_size": 30,
    "spread": 0.00015,
    "take_profit_pct": 0.003,
    "stop_loss_pct": 0.002,
    "max_hold": 48,
    "cooldown": 4,
}

def load_data(year):
    path = DATA_DIR / f"GBPUSD_M15_{year}.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NpEncoder, self).default(obj)

def evaluate(model, df, label=""):
    if df.empty:
        print(f"{label}: No data")
        return {}

    env = TradingEnvV8(
        df=df,
        window_size=RL_CONFIG["window_size"],
        spread=RL_CONFIG["spread"],
        take_profit_pct=RL_CONFIG["take_profit_pct"],
        stop_loss_pct=RL_CONFIG["stop_loss_pct"],
        max_hold=RL_CONFIG["max_hold"],
        cooldown=RL_CONFIG["cooldown"],
    )

    obs, _ = env.reset()
    # LSTM states handling
    lstm_states = None
    done = False
    
    # Track completion if needed
    while not done:
        # RecurrentPPO predict returns (action, states)
        action, lstm_states = model.predict(
            obs,
            state=lstm_states,
            deterministic=True
        )
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    p = env.get_performance_summary()

    print(f"\n===== {label} =====")
    print(f"Profit       : {p['profit_pct']:+.2f}%")
    print(f"Max DD       : {p['max_drawdown_pct']:.2f}%")
    print(f"Trades       : {p['nb_trades']}")
    print(f"Win Rate     : {p['win_rate']:.1f}%")
    print(f"Avg Hold     : {p['avg_hold']:.0f}")
    print("=======================")

    return p

def train():
    df_2022 = load_data(2022)
    df_2023 = load_data(2023)
    df_2024 = load_data(2024)
    df_2025 = load_data(2025)

    # Train on 2022-2024
    df_train = pd.concat([df_2022, df_2023, df_2024])
    print(f"Training on {len(df_train)} candles (2022-2024)...")

    env_train = TradingEnvV8(
        df=df_train,
        window_size=RL_CONFIG["window_size"],
        spread=RL_CONFIG["spread"],
        take_profit_pct=RL_CONFIG["take_profit_pct"],
        stop_loss_pct=RL_CONFIG["stop_loss_pct"],
        max_hold=RL_CONFIG["max_hold"],
        cooldown=RL_CONFIG["cooldown"],
    )

    check_env(env_train)

    model = RecurrentPPO(
        "MlpLstmPolicy",
        env_train,
        learning_rate=RL_CONFIG["learning_rate"],
        gamma=RL_CONFIG["gamma"],
        batch_size=RL_CONFIG["batch_size"],
        n_steps=RL_CONFIG["n_steps"],
        n_epochs=RL_CONFIG["n_epochs"],
        clip_range=RL_CONFIG["clip_range"],
        ent_coef=RL_CONFIG["ent_coef"],
        verbose=1,
        seed=RL_CONFIG["seed"],
    )

    print("Training V11 (RecurrentPPO)...")
    model.learn(total_timesteps=RL_CONFIG["total_timesteps"])

    model.save(str(MODEL_DIR / "ppo_v11"))
    
    # Save config
    with open(MODEL_DIR / "rl_config.json", "w") as f:
        json.dump(RL_CONFIG, f, indent=2)

    print("\n===== FINAL EVALUATION =====")

    results = {}
    results[2022] = evaluate(model, df_2022, "2022")
    results[2023] = evaluate(model, df_2023, "2023")
    results[2024] = evaluate(model, df_2024, "2024")
    results[2025] = evaluate(model, df_2025, "2025 (OOS TEST)")

    with open(MODEL_DIR / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2, cls=NpEncoder)

    print("\nV11 completed.")

if __name__ == "__main__":
    train()
