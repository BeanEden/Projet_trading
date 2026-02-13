"""
T08 v9 - Optimisation Hyperparametres (Optuna) sur Env V8
Objectif : Maximiser le Sharpe Ratio sur 2024.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# On utilise l'environnement V8 (le meilleur jusqu'ici)
from training.trading_env_v8 import TradingEnvV8

DATA_DIR = PROJECT_ROOT / "data" / "m15"
MODEL_DIR = PROJECT_ROOT / "models" / "v9_optuna"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Charger les données une fois pour toutes
def load_data(year):
    path = DATA_DIR / "GBPUSD_M15_{}.csv".format(year)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")

DF_TRAIN = pd.concat([load_data(2022), load_data(2023)], ignore_index=False)
DF_VAL = load_data(2024)

print(f"Data Loaded: Train={len(DF_TRAIN)}, Val={len(DF_VAL)}")

def calculate_sharpe(pnl_history):
    if not pnl_history:
        return -10.0
    returns = np.array(pnl_history)
    if np.std(returns) < 1e-6:
        return -10.0
    return np.mean(returns) / np.std(returns) * np.sqrt(len(pnl_history)) # annualized-ish if len is year? No, just ratio.


class TrialEvalCallback(BaseCallback):
    """
    Callback pour élaguer les mauvais trials (Pruning)
    """
    def __init__(self, eval_env, trial, eval_freq=25000):
        super().__init__(verbose=0)
        self.eval_env = eval_env
        self.trial = trial
        self.eval_freq = eval_freq
        self.best_profit = -np.inf

    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            # Evaluate
            obs, _ = self.eval_env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, t, tr, _ = self.eval_env.step(action)
                done = t or tr
            
            p = self.eval_env.get_performance_summary()
            profit = p["profit_pct"]
            
            # Report to Optuna
            self.trial.report(profit, self.n_calls)
            
            # Prune if bad
            if self.trial.should_prune():
                raise optuna.exceptions.TrialPruned()
                
        return True


def objective(trial):
    # Hyperparameters to tune
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    n_steps = trial.suggest_categorical("n_steps", [1024, 2048, 4096])
    ent_coef = trial.suggest_float("ent_coef", 0.0001, 0.1, log=True)
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3])
    gamma = trial.suggest_categorical("gamma", [0.99, 0.995, 0.999])
    
    # Network Arch
    net_arch_type = trial.suggest_categorical("net_arch", ["small", "medium", "large"])
    if net_arch_type == "small":
        net_arch = [64, 64]
    elif net_arch_type == "medium":
        net_arch = [128, 128]
    else:
        net_arch = [256, 128]

    # Env config (Fixed V8 logic, maybe tune spread sensitivity? No, keep std)
    env_kwargs = {
        "df": DF_TRAIN,
        "window_size": 30,
        "spread": 0.00015,
        "take_profit_pct": 0.003,
        "stop_loss_pct": 0.002,
        "max_hold": 48,
        "cooldown": 4
    }
    
    env_train = TradingEnvV8(**env_kwargs)
    env_val = TradingEnvV8(df=DF_VAL, **{k:v for k,v in env_kwargs.items() if k != "df"})
    
    model = PPO(
        "MlpPolicy",
        env_train,
        learning_rate=learning_rate,
        batch_size=batch_size,
        n_steps=n_steps,
        ent_coef=ent_coef,
        clip_range=clip_range,
        gamma=gamma,
        policy_kwargs={"net_arch": net_arch},
        verbose=0,
        seed=42
    )
    
    # 100k steps is enough to see if it learns something
    TOTAL_STEPS = 100_000 
    
    callback = TrialEvalCallback(env_val, trial, eval_freq=25000)
    
    try:
        model.learn(total_timesteps=TOTAL_STEPS, callback=callback)
    except optuna.exceptions.TrialPruned:
        raise
    except Exception as e:
        print(f"Trial failed: {e}")
        return -100.0 # Bad score

    # Final Eval on Validation Set
    obs, _ = env_val.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, t, tr, _ = env_val.step(action)
        done = t or tr
        
    p = env_val.get_performance_summary()
    
    # We want to maximize... Profit (or Sharpe?). Using Profit for simplicity, but punish DD.
    score = p["profit_pct"] - (p["max_drawdown_pct"] * 0.5) 
    
    # Penalty for 0 trades or very low activity
    if p["nb_trades"] < 20:
        score -= 20
        
    print(f"[Trial {trial.number}] Profit: {p['profit_pct']:.2f}% | MaxDD: {p['max_drawdown_pct']:.2f}% | Trades: {p['nb_trades']} | Score: {score:.2f}")
    print(f"  Params: {trial.params}")
        
    return score


if __name__ == "__main__":
    print("Starting Optuna Study...")
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    try:
        study.optimize(objective, n_trials=30, timeout=3600) # 1 hour max or 30 trials
    except KeyboardInterrupt:
        print("Interrupted by user.")

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Save best params
    import json
    with open(MODEL_DIR / "best_params.json", "w") as f:
        json.dump(trial.params, f, indent=4)
        
    # Re-train best model fully? (Optionally)
