"""
T08 v5c – RL Signal + TP/SL, optimisé pour selectivité
Même TP/SL que V5 (0.3%/0.2%), cooldown plus long, moins d'exploration.

Usage :
  python training/train_rl_v5c.py
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.trading_env_v5 import TradingEnvV5

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.callbacks import BaseCallback
except ImportError:
    print("stable-baselines3 non installe.")
    sys.exit(1)

DATA_DIR = PROJECT_ROOT / "data" / "m15"
MODEL_DIR = PROJECT_ROOT / "models" / "v5c"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RL_CONFIG = {
    "algorithm": "PPO",
    "total_timesteps": 500_000,
    "learning_rate": 1e-4,          # plus bas pour plus de stabilité
    "gamma": 0.995,                 # horizon plus long
    "batch_size": 512,              # plus gros batch = gradients plus stables
    "n_steps": 4096,                # plus de données par update
    "n_epochs": 5,                  # moins d'epochs
    "clip_range": 0.15,             # plus serré
    "ent_coef": 0.003,              # TRES peu d'exploration → force exploitation
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "gae_lambda": 0.95,
    "seed": 42,
    "window_size": 30,
    "spread": 0.00015,
    "take_profit_pct": 0.003,       # 0.3% TP (identique à V5)
    "stop_loss_pct": 0.002,         # 0.2% SL (identique à V5)
    "max_hold": 48,
    "cooldown": 8,                  # 2h cooldown (double de V5)
    "net_arch": [256, 128, 64],     # réseau plus grand pour de meilleures features
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
            msg = "  [Step {:>7}] P:{:+.2f}% T:{} WR:{:.0f}% TP:{} SL:{} TO:{} Hold:{:.0f}".format(
                self.n_calls, p["profit_pct"], p["nb_trades"],
                p["win_rate"], p["tp_count"], p["sl_count"], p["timeout_count"], p["avg_hold"]
            )
            print(msg, flush=True)

            if p["profit_pct"] > self.best_profit and self.save_path:
                self.best_profit = p["profit_pct"]
                self.model.save(str(self.save_path / "ppo_trading_best"))
                print("    -> NEW BEST ({:+.2f}%)".format(p["profit_pct"]), flush=True)

        return True


def load_data(year):
    return pd.read_csv(DATA_DIR / "GBPUSD_M15_{}.csv".format(year),
                       parse_dates=["timestamp"], index_col="timestamp")


def evaluate(model, df, label=""):
    env = TradingEnvV5(
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
    print("T08 v5c - RL Signal + TP/SL (selective, low entropy)")
    print("=" * 60, flush=True)

    df_train = pd.concat([load_data(2022), load_data(2023)], ignore_index=False)
    df_test = load_data(2024)
    print("  Train: {:,}  Test: {:,}".format(len(df_train), len(df_test)), flush=True)

    env_train = TradingEnvV5(
        df=df_train, window_size=RL_CONFIG["window_size"],
        spread=RL_CONFIG["spread"],
        take_profit_pct=RL_CONFIG["take_profit_pct"],
        stop_loss_pct=RL_CONFIG["stop_loss_pct"],
        max_hold=RL_CONFIG["max_hold"],
        cooldown=RL_CONFIG["cooldown"],
    )
    env_test = TradingEnvV5(
        df=df_test, window_size=RL_CONFIG["window_size"],
        spread=RL_CONFIG["spread"],
        take_profit_pct=RL_CONFIG["take_profit_pct"],
        stop_loss_pct=RL_CONFIG["stop_loss_pct"],
        max_hold=RL_CONFIG["max_hold"],
        cooldown=RL_CONFIG["cooldown"],
    )

    check_env(env_train, warn=True)
    print("  Env OK", flush=True)

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
        verbose=0,
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
    for year, label in [(2022, "2022"), (2023, "2023"), (2024, "2024 (Test)")]:
        results[year] = evaluate(model, load_data(year), label)

    with open(MODEL_DIR / "rl_config.json", "w") as f:
        json.dump(RL_CONFIG, f, indent=2)
    with open(MODEL_DIR / "eval_results.json", "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)

    print("\n" + "=" * 60)
    print("v5c done!")
    print("=" * 60, flush=True)


if __name__ == "__main__":
    train()
