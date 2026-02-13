import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from stable_baselines3 import PPO, DQN

# Setup path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import Envs
try: from training.trading_env_v2 import TradingEnvV2
except ImportError: TradingEnvV2 = None
try: from training.trading_env_v3 import TradingEnvV3
except ImportError: TradingEnvV3 = None
try: from training.trading_env_v4 import TradingEnvV4
except ImportError: TradingEnvV4 = None
try: from training.trading_env_v5 import TradingEnvV5
except ImportError: TradingEnvV5 = None
try: from training.trading_env_v6 import TradingEnvV6
except ImportError: TradingEnvV6 = None
try: from training.trading_env_v8 import TradingEnvV8
except ImportError: TradingEnvV8 = None

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NpEncoder, self).default(obj)

def load_data(year):
    path = PROJECT_ROOT / "data" / "m15" / f"GBPUSD_M15_{year}.csv"
    if not path.exists(): return None
    return pd.read_csv(path, parse_dates=["timestamp"])

def get_env_class(version):
    v = version.lower()
    if "v2" in v: return TradingEnvV2
    if "v3" in v: return TradingEnvV3
    if "v4" in v: return TradingEnvV4
    if "v5" in v or "v6" in v or "v7" in v: return TradingEnvV5 
    if "v8" in v or "v9" in v or "v10" in v: return TradingEnvV8
    return TradingEnvV8

def load_config(model_dir):
    config_path = model_dir / "rl_config.json"
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}

def evaluate_model(version, model, df, year, config):
    env_class = get_env_class(version)
    if env_class is None: return None

    # Prepare env kwargs from config or defaults
    env_kwargs = {"df": df}
    
    # Common params in TradingEnvV5/V8
    target_params = ["window_size", "spread", "take_profit_pct", "stop_loss_pct", "max_hold", "cooldown"]
    
    for param in target_params:
        if param in config:
            env_kwargs[param] = config[param]
            
    # Init Env
    try:
        env = env_class(**env_kwargs)
    except Exception as e:
        print(f"    Error init env for {version} (config={list(env_kwargs.keys())}): {e}")
        # Try without config if failed (fallback to defaults if signature mismatch)
        try:
            env = env_class(df)
        except:
            return None

    obs, _ = env.reset()
    done = False
    try:
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
    except Exception as e:
        print(f"    Runtime error: {e}")
        return None
        
    if hasattr(env, "get_performance_summary"):
        return env.get_performance_summary()
    elif hasattr(env, "get_metrics"):
        return env.get_metrics()
    return {}

def run_evaluation(version):
    model_dir = PROJECT_ROOT / "models" / version
    if not model_dir.exists(): return

    print(f"\n--- Processing {version} ---")
    
    # Load Config
    config = load_config(model_dir)
    if config:
        print(f"  Loaded config: TP={config.get('take_profit_pct')}, SL={config.get('stop_loss_pct')}")
    else:
        print("  No config found, using defaults.")

    # Find model file
    candidates = ["ppo_trading_best.zip", "dqn_trading_best.zip", "ppo_trading.zip", "ppo_trading_final.zip", "dqn_trading.zip"]
    model_path = None
    model_type = PPO
    
    for c in candidates:
        p = model_dir / c
        if p.exists():
            model_path = p
            if "dqn" in c: model_type = DQN
            break
            
    if not model_path:
        print(f"  No model zip found.")
        return

    try:
        model = model_type.load(model_path)
    except Exception as e:
        print(f"  Failed to load model: {e}")
        return

    results = {}
    
    # Evaluate 2022-2025
    for year in [2022, 2023, 2024, 2025]:
        print(f"  Evaluating {year}...", end=" ", flush=True)
        df = load_data(year)
        if df is not None:
            met = evaluate_model(version, model, df, year, config)
            if met:
                results[str(year)] = met
                print(f"Profit: {met.get('profit_pct', 0):.2f}%")
            else:
                print("Failed.")
        else:
            print("No data.")
    
    # Save
    out_file = model_dir / "eval_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"  Saved results.")

if __name__ == "__main__":
    models_root = PROJECT_ROOT / "models"
    all_versions = [d.name for d in models_root.iterdir() if d.is_dir()]
    
    def version_key(v):
        import re
        num = re.sub(r'\D', '', v)
        return int(num) if num else 0
        
    all_versions.sort(key=version_key)
    
    print(f"Found models: {all_versions}")

    for v in all_versions:
        try:
            run_evaluation(v)
        except Exception as e:
            print(f"CRITICAL ERROR {v}: {e}")
