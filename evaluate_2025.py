import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.trading_env_v5 import TradingEnvV5
from stable_baselines3 import PPO

DATA_FILE = PROJECT_ROOT / "data/m15/GBPUSD_M15_2025.csv"
MODEL_PATH = PROJECT_ROOT / "models/v5/ppo_trading_best"

# Config from V5 training (hardcoded or loaded from json)
# Let's load from json if possible to be robust
CONFIG_FILE = PROJECT_ROOT / "models/v5/rl_config.json"

def main():
    print(f"Loading 2025 data from {DATA_FILE}...")
    if not DATA_FILE.exists():
        print("Error: 2025 data file not found.")
        return

    df_2025 = pd.read_csv(DATA_FILE, parse_dates=["timestamp"], index_col="timestamp")
    print(f"Loaded {len(df_2025)} candles.")

    print(f"Loading model config from {CONFIG_FILE}...")
    with open(CONFIG_FILE, "r") as f:
        config = json.load(f)
    
    # We need to filter config to match TradingEnvV5 constructor args
    env_kwargs = {
        "window_size": config["window_size"],
        "spread": config["spread"],
        "take_profit_pct": config["take_profit_pct"],
        "stop_loss_pct": config["stop_loss_pct"],
        "max_hold": config["max_hold"],
        "cooldown": config["cooldown"]
    }

    print("Creating environment...")
    try:
        env = TradingEnvV5(df=df_2025, **env_kwargs)
    except Exception as e:
        print(f"Error creating environment: {e}")
        return

    print(f"Loading model from {MODEL_PATH}...")
    try:
        model = PPO.load(str(MODEL_PATH))
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("\nRunning evaluation on 2025 data...")
    obs, _ = env.reset()
    done = False
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    p = env.get_performance_summary()
    
    output_str = f"""
==================================================
  2025 EVALUATION RESULTS (V5 Model)
==================================================
Profit      : {p["profit_pct"]:+.2f}%
Final Equity: {p["final_equity"]:,.2f}
Max DD      : {p["max_drawdown_pct"]:.2f}%
Trades      : {p["nb_trades"]} (TP:{p["tp_count"]} SL:{p["sl_count"]} TO:{p["timeout_count"]})
Win Rate    : {p["win_rate"]:.1f}%
Avg Win     : {p["avg_win"]:+.3f}%
Avg Loss    : {p["avg_loss"]:+.3f}%
Avg Hold    : {p["avg_hold"]:.0f} bars
==================================================
"""
    print(output_str)
    
    # Save to file
    out_path = PROJECT_ROOT / "models/v5/eval_results_2025.txt"
    with open(out_path, "w") as f:
        f.write(output_str)
    print(f"Results saved to {out_path}")

if __name__ == "__main__":
    main()
