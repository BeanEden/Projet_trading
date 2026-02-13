
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from stable_baselines3 import PPO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Environnement V8
from training.trading_env_v8 import TradingEnvV8

DATA_DIR = PROJECT_ROOT / "data" / "m15"
MODEL_DIR = PROJECT_ROOT / "models" / "v8_fixed"
model_path = MODEL_DIR / "ppo_trading.zip" # Or best? Let's use final for now. Or check if best exists.
if (MODEL_DIR / "ppo_trading_best.zip").exists():
    model_path = MODEL_DIR / "ppo_trading_best.zip"
    print("Loading BEST model.")
else:
    print("Loading FINAL model.")

def load_data(year):
    path = DATA_DIR / "GBPUSD_M15_{}.csv".format(year)
    if not path.exists():
        print(f"Warning: {path} not found.")
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")

# Helper to convert numpy types to python types for JSON
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def evaluate(model, df, label=""):
    if df.empty:
        print(f"  {label}: No data")
        return {}
        
    env = TradingEnvV8(
        df=df, window_size=30,
        spread=0.00015,
        take_profit_pct=0.003,
        stop_loss_pct=0.002,
        max_hold=48,
        cooldown=4,
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

if __name__ == "__main__":
    model = PPO.load(str(model_path))
    print(f"Model loaded from {model_path}")
    
    results = {}
    for year, label in [(2022, "2022"), (2023, "2023"), (2024, "2024 (Test)"), (2025, "2025 (Validation)")]:
        results[year] = evaluate(model, load_data(year), label)
        
    with open(MODEL_DIR / "eval_results.json", "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2, cls=NpEncoder)
    print("\nEvaluation complete. Saved to eval_results.json")
