"""
T08 – Environnement de trading RL (Gymnasium)
Environnement custom pour l'entraînement d'un agent RL sur GBP/USD M15.

Usage :
  from training.trading_env import TradingEnv
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from pathlib import Path


class TradingEnv(gym.Env):
    """
    Environnement de trading GBP/USD pour Reinforcement Learning.

    Observation : fenêtre glissante de features techniques + position courante
    Actions     : 0=HOLD, 1=BUY, 2=SELL
    Reward      : PnL réalisé - coûts de transaction - pénalité drawdown
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int = 20,
        transaction_cost: float = 0.0002,
        drawdown_penalty: float = 0.5,
        initial_capital: float = 10_000,
    ):
        """
        Args:
            df              : DataFrame M15 avec colonnes close_15m + features
            window_size     : taille de la fenêtre d'observation
            transaction_cost: coût par trade (proportion)
            drawdown_penalty: facteur de pénalité drawdown dans la reward
            initial_capital : capital de départ
        """
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.transaction_cost = transaction_cost
        self.drawdown_penalty = drawdown_penalty
        self.initial_capital = initial_capital

        # Préparer les features
        self.feature_columns = self._compute_features()
        self.n_features = len(self.feature_columns)

        # Espaces
        # observation = fenêtre de features (window_size x n_features) aplatie + position
        obs_size = self.window_size * self.n_features + 1  # +1 pour la position
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)  # HOLD, BUY, SELL

        # Variables d'état
        self.current_step = 0
        self.position = 0  # 0=flat, 1=long, -1=short
        self.equity = initial_capital
        self.peak_equity = initial_capital
        self.entry_price = 0.0
        self.trades = []

    def _compute_features(self) -> list:
        """Calcule les features techniques à partir du DataFrame."""
        df = self.df

        # Return court terme
        df["return_1"] = df["close_15m"].pct_change(1)
        df["return_4"] = df["close_15m"].pct_change(4)

        # EMA
        df["ema_20"] = df["close_15m"].ewm(span=20, adjust=False).mean()
        df["ema_50"] = df["close_15m"].ewm(span=50, adjust=False).mean()
        df["ema_diff"] = (df["ema_20"] - df["ema_50"]) / df["close_15m"]

        # RSI 14
        delta = df["close_15m"].diff()
        gain = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        df["rsi_14"] = 100 - (100 / (1 + rs))
        df["rsi_14"] = df["rsi_14"] / 100  # normaliser [0,1]

        # Volatilité
        df["rolling_std_20"] = df["return_1"].rolling(20).std()
        df["range_15m"] = (df["high_15m"] - df["low_15m"]) / df["close_15m"]

        # Structure bougie
        df["body"] = (df["close_15m"] - df["open_15m"]) / df["close_15m"]
        df["upper_wick"] = (df["high_15m"] - df[["open_15m", "close_15m"]].max(axis=1)) / df["close_15m"]
        df["lower_wick"] = (df[["open_15m", "close_15m"]].min(axis=1) - df["low_15m"]) / df["close_15m"]

        features = [
            "return_1", "return_4", "ema_diff", "rsi_14",
            "rolling_std_20", "range_15m", "body", "upper_wick", "lower_wick"
        ]

        # Remplir les NaN
        for col in features:
            df[col] = df[col].fillna(0)

        self.df = df
        return features

    def _get_observation(self) -> np.ndarray:
        """Retourne l'observation courante."""
        start = self.current_step - self.window_size
        end = self.current_step

        window = self.df[self.feature_columns].iloc[start:end].values

        # Z-score normalisation sur la fenêtre
        mean = window.mean(axis=0)
        std = window.std(axis=0) + 1e-8
        window_norm = (window - mean) / std

        # Aplatir + ajouter la position
        obs = np.concatenate([window_norm.flatten(), [self.position]])
        return obs.astype(np.float32)

    def reset(self, seed=None, options=None):
        """Reset l'environnement."""
        super().reset(seed=seed)

        self.current_step = self.window_size  # skip warm-up
        self.position = 0
        self.equity = self.initial_capital
        self.peak_equity = self.initial_capital
        self.entry_price = 0.0
        self.trades = []

        obs = self._get_observation()
        info = {"equity": self.equity, "position": self.position}

        return obs, info

    def step(self, action: int):
        """
        Exécute une action.

        Args:
            action : 0=HOLD, 1=BUY, 2=SELL

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Map action → signal
        action_map = {0: 0, 1: 1, 2: -1}
        signal = action_map[action]

        price = self.df["close_15m"].iloc[self.current_step]
        prev_price = self.df["close_15m"].iloc[self.current_step - 1]

        # ── PnL sur position existante ──
        pnl = 0.0
        if self.position != 0:
            pnl = self.position * (price - prev_price) / prev_price * self.equity

        self.equity += pnl

        # ── Coût de transaction si changement de position ──
        trade_cost = 0.0
        if signal != 0 and signal != self.position:
            trade_cost = self.transaction_cost * self.equity
            if self.position != 0:
                trade_cost *= 2  # retournement
            self.equity -= trade_cost
            self.position = signal
            self.entry_price = price
            self.trades.append({
                "step": self.current_step,
                "action": signal,
                "price": price,
                "equity": self.equity,
            })

        # ── Drawdown ──
        self.peak_equity = max(self.peak_equity, self.equity)
        current_dd = (self.peak_equity - self.equity) / self.peak_equity

        # ── Reward ──
        reward = pnl / self.initial_capital  # PnL normalisé
        reward -= trade_cost / self.initial_capital  # pénalité coût
        reward -= self.drawdown_penalty * current_dd * abs(pnl) / self.initial_capital  # pénalité DD

        # ── Avancer ──
        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        truncated = False

        # ── Stop loss si equity trop basse ──
        if self.equity < self.initial_capital * 0.5:
            terminated = True
            reward -= 1.0  # grosse pénalité

        obs = self._get_observation() if not terminated else np.zeros(
            self.observation_space.shape, dtype=np.float32
        )

        info = {
            "equity": self.equity,
            "position": self.position,
            "drawdown": current_dd,
            "nb_trades": len(self.trades),
            "pnl": pnl,
        }

        return obs, reward, terminated, truncated, info

    def get_performance_summary(self) -> dict:
        """Retourne un résumé de la performance de l'épisode."""
        profit_pct = (self.equity / self.initial_capital - 1) * 100
        max_dd = (self.peak_equity - self.equity) / self.peak_equity * 100

        return {
            "profit_pct": profit_pct,
            "final_equity": self.equity,
            "max_drawdown_pct": max_dd,
            "nb_trades": len(self.trades),
        }
