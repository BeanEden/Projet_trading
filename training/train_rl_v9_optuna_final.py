"""
T08 v9 Optuna Final - Training with Best Hyperparameters
Env: V8 (Signal + Auto TP/SL + Time + Regime).
Train: 2022+2023. Test: 2024. Validation: 2025.

Usage:
  python training/train_rl_v9_optuna_final.py
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
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.callbacks import BaseCallback
except ImportError:
    print("stable-baselines3 non installe.")
    sys.exit(1)

DATA_DIR = PROJECT_ROOT / "data" / "m15"
MODEL_DIR = PROJECT_ROOT / "models" / "v9_optuna"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Load best params
PARAMS_FILE = MODEL_DIR / "best_params.json"
if not PARAMS_FILE.exists():
    print(f"Error: {PARAMS_FILE} not found. Please run train_rl_v9_optuna.py first.")
    # Fallback default if needed, but better to fail
    sys.exit(1)

with open(PARAMS_FILE, "r") as f:
    BEST_PARAMS = json.load(f)

print("Loaded Best Params:", BEST_PARAMS)

# Parse net_arch
net_arch_type = BEST_PARAMS.pop("net_arch", "medium")
if net_arch_type == "small":
    net_arch = [64, 64]
elif net_arch_type == "medium":
    net_arch = [128, 128]
else: # large
    net_arch = [256, 128]

# Default fixed params (from V8 config mostly)
RL_CONFIG = {
    "algorithm": "PPO",
    "total_timesteps": 500_000, # Standard training length
    "learning_rate": BEST_PARAMS["learning_rate"],
    "gamma": BEST_PARAMS["gamma"],
    "batch_size": BEST_PARAMS["batch_size"],
    "n_steps": BEST_PARAMS["n_steps"],
    "n_epochs": 10, # Fixed
    "clip_range": BEST_PARAMS["clip_range"],
    "ent_coef": BEST_PARAMS["ent_coef"],
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "gae_lambda": 0.95,
    "seed": 42,
    "net_arch": net_arch,
    
    # Env config (Fixed V8)
    "window_size": 30,
    "spread": 0.00015,
    "take_profit_pct": 0.003,
    "stop_loss_pct": 0.002,
    "max_hold": 48,
    "cooldown": 4,
}


class EvalCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq=25_000, save_path=None):
        super().__init__(verbose=1)
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
                obs, _, t, tr, _ = self.eval_env.step(action)
                done = t or tr

            p = self.eval_env.get_performance_summary()
            msg = "  [Step {:>7}] P:{:+.2f}% T:{} WR:{:.0f}% TP:{} SL:{} TO:{} Hold:{:.0f} MaxDD:{:.2f}%".format(
                self.n_calls, p["profit_pct"], p["nb_trades"],
                p["win_rate"], p["tp_count"], p["sl_count"], p["timeout_count"], p["avg_hold"], p["max_drawdown_pct"]
            )
            print(msg, flush=True)

            if p["profit_pct"] > self.best_profit and self.save_path:
                self.best_profit = p["profit_pct"]
                self.model.save(str(self.save_path / "ppo_trading_best"))
                print("    -> NEW BEST ({:+.2f}%)".format(p["profit_pct"]), flush=True)

        return True


def load_data(year):
    path = DATA_DIR / "GBPUSD_M15_{}.csv".format(year)
    if not path.exists():
        print(f"Warning: {path} not found.")
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")


def evaluate(model, df, label=""):
    if df.empty:
        print(f"  {label}: No data")
        return {}
        
    env = TradingEnvV8(
        df=df, window_size=RL_CONFIG["window_size"],
        spread=RL_CONFIG["spread"],
        take_profit_pct=RL_CONFIG["take_profit_pct"],
        stop_loss_pct=RL_CONFIG["stop_loss_pct"],
        max_hold=RL_CONFIG["max_hold"],
        cooldown=RL_CONFIG["cooldown"],
    )
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, t, tr, _ = env.step(action)
        done = t or tr

    p = env.get_performance_summary()
    print("\n  {}:".format(label))
    print("     Profit    : {:+.2f}%".format(p["profit_pct"]))
    print("     Equity    : {:,.2f}".format(p["final_equity"]))
    print("     Max DD    : {:.2f}%".format(p["max_drawdown_pct"]))
    print("     Trades    : {} (TP:{} SL:{} TO:{})".format(
        p["nb_trades"], p["tp_count"], p["sl_count"], p["timeout_count"]))
    print("     Win Rate  : {:.1f}%".format(p["win_rate"]))
    print("     Avg Win   : {:+.3f}%".format(p["avg_win"]))
    print("     Avg Loss  : {:+.3f}%".format(p["avg_loss"]))
    print("     Avg Hold  : {:.0f} bars".format(p["avg_hold"]), flush=True)
    return p


def train():
    print("=" * 60)
    print("T08 v9 Optuna Final - PPO Training (Final Model)")
    print("=" * 60, flush=True)

    df_train = pd.concat([load_data(y) for y in [2022, 2023]], ignore_index=False)
    df_test = load_data(2024)
    
    print("  Train: {:,} candles (2022-2023)".format(len(df_train)))
    print("  Test : {:,} candles (2024)".format(len(df_test)), flush=True)

    env_train = TradingEnvV8(
        df=df_train, window_size=RL_CONFIG["window_size"],
        spread=RL_CONFIG["spread"],
        take_profit_pct=RL_CONFIG["take_profit_pct"],
        stop_loss_pct=RL_CONFIG["stop_loss_pct"],
        max_hold=RL_CONFIG["max_hold"],
        cooldown=RL_CONFIG["cooldown"],
    )
    
    env_test = TradingEnvV8(
        df=df_test, window_size=RL_CONFIG["window_size"],
        spread=RL_CONFIG["spread"],
        take_profit_pct=RL_CONFIG["take_profit_pct"],
        stop_loss_pct=RL_CONFIG["stop_loss_pct"],
        max_hold=RL_CONFIG["max_hold"],
        cooldown=RL_CONFIG["cooldown"],
    )

    check_env(env_train, warn=True)
    print("  Env V9 Optuna OK", flush=True)

    model = PPO(
        "MlpPolicy", env_train,
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
        verbose=1,
    )

    callback = EvalCallback(eval_env=env_test, eval_freq=25_000, save_path=MODEL_DIR)

    print("\nTraining ({:,} steps)...".format(RL_CONFIG["total_timesteps"]), flush=True)
    model.learn(total_timesteps=RL_CONFIG["total_timesteps"], callback=callback)
    model.save(str(MODEL_DIR / "ppo_trading"))

    best_path = MODEL_DIR / "ppo_trading_best.zip"
    if best_path.exists():
        model = PPO.load(str(MODEL_DIR / "ppo_trading_best"))
        print("\n  Best model loaded.", flush=True)

    print("\n" + "=" * 50)
    print("  FINAL EVALUATION")
    print("=" * 50, flush=True)

    results = {}
    for year, label in [(2022, "2022"), (2023, "2023"), (2024, "2024 (Test)"), (2025, "2025 (Validation)")]:
        results[year] = evaluate(model, load_data(year), label)

    # Add net_arch back to config for display (as string)
    RL_CONFIG["net_arch"] = str(RL_CONFIG["net_arch"])
    
    with open(MODEL_DIR / "rl_config.json", "w") as f:
        json.dump(RL_CONFIG, f, indent=2)
    with open(MODEL_DIR / "eval_results.json", "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)

    print("\n" + "=" * 60)
    print("v9_optuna done!")
    print("=" * 60, flush=True)


if __name__ == "__main__":
    train()
