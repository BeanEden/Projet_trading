"""
T08 v2 – Environnement de trading RL amélioré
Reward shaping optimisé pour encourager des trades profitables.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from pathlib import Path


class TradingEnvV2(gym.Env):
    """
    Environnement de trading GBP/USD amélioré.

    Améliorations par rapport à v1 :
    - Reward basée sur le PnL réalisé (à la clôture du trade)
    - Pénalité pour inactivité (trop de HOLD)
    - Bonus pour fermer un trade gagnant
    - Features supplémentaires (MACD, Bollinger, momentum)
    - Gestion de position simplifiée (flat/long uniquement pour commencer)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int = 30,
        transaction_cost: float = 0.0001,
        initial_capital: float = 10_000,
    ):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.transaction_cost = transaction_cost
        self.initial_capital = initial_capital

        # Calculer features
        self.feature_columns = self._compute_features()
        self.n_features = len(self.feature_columns)

        # Observation: window aplatie + position + unrealized_pnl + steps_in_position
        obs_size = self.window_size * self.n_features + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)  # HOLD, BUY, SELL

        # État
        self.current_step = 0
        self.position = 0  # 0=flat, 1=long, -1=short
        self.equity = initial_capital
        self.peak_equity = initial_capital
        self.entry_price = 0.0
        self.steps_in_position = 0
        self.idle_steps = 0
        self.trades = []
        self.trade_results = []

    def _compute_features(self) -> list:
        df = self.df
        close = df["close_15m"]

        # Returns
        df["return_1"] = close.pct_change(1)
        df["return_4"] = close.pct_change(4)
        df["return_16"] = close.pct_change(16)  # 4h

        # EMAs normalisées
        df["ema_10"] = close.ewm(span=10, adjust=False).mean()
        df["ema_30"] = close.ewm(span=30, adjust=False).mean()
        df["ema_60"] = close.ewm(span=60, adjust=False).mean()
        df["ema_10_norm"] = (close - df["ema_10"]) / close
        df["ema_30_norm"] = (close - df["ema_30"]) / close
        df["ema_60_norm"] = (close - df["ema_60"]) / close
        df["ema_cross"] = (df["ema_10"] - df["ema_30"]) / close

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        df["rsi_14"] = (100 - (100 / (1 + rs))) / 100  # [0, 1]

        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        df["macd_norm"] = (macd - signal) / close

        # Bollinger Bands
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        df["bb_upper"] = (close - (sma20 + 2 * std20)) / close
        df["bb_lower"] = (close - (sma20 - 2 * std20)) / close
        df["bb_width"] = (4 * std20) / close

        # Volatilité
        df["vol_20"] = df["return_1"].rolling(20).std()
        df["vol_60"] = df["return_1"].rolling(60).std()

        # ATR (Average True Range)
        high_low = df["high_15m"] - df["low_15m"]
        high_close = abs(df["high_15m"] - close.shift())
        low_close = abs(df["low_15m"] - close.shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr_14"] = tr.rolling(14).mean() / close

        # Structure de bougie
        df["body"] = (close - df["open_15m"]) / close
        df["range_norm"] = (df["high_15m"] - df["low_15m"]) / close

        # Momentum
        df["mom_10"] = close / close.shift(10) - 1
        df["mom_40"] = close / close.shift(40) - 1

        features = [
            "return_1", "return_4", "return_16",
            "ema_10_norm", "ema_30_norm", "ema_60_norm", "ema_cross",
            "rsi_14", "macd_norm",
            "bb_upper", "bb_lower", "bb_width",
            "vol_20", "vol_60", "atr_14",
            "body", "range_norm",
            "mom_10", "mom_40",
        ]

        for col in features:
            df[col] = df[col].fillna(0)

        self.df = df
        return features

    def _get_observation(self) -> np.ndarray:
        start = max(0, self.current_step - self.window_size)
        end = self.current_step
        window = self.df[self.feature_columns].iloc[start:end].values

        # Pad si nécessaire
        if len(window) < self.window_size:
            pad = np.zeros((self.window_size - len(window), self.n_features))
            window = np.concatenate([pad, window])

        # Z-score sur la fenêtre
        mean = window.mean(axis=0)
        std = window.std(axis=0) + 1e-8
        window_norm = (window - mean) / std

        # Infos supplémentaires
        unrealized_pnl = 0.0
        if self.position != 0 and self.entry_price > 0:
            price = self.df["close_15m"].iloc[self.current_step]
            unrealized_pnl = self.position * (price - self.entry_price) / self.entry_price

        obs = np.concatenate([
            window_norm.flatten(),
            [self.position, unrealized_pnl, min(self.steps_in_position / 100, 1.0)]
        ])
        return obs.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.position = 0
        self.equity = self.initial_capital
        self.peak_equity = self.initial_capital
        self.entry_price = 0.0
        self.steps_in_position = 0
        self.idle_steps = 0
        self.trades = []
        self.trade_results = []

        obs = self._get_observation()
        return obs, {"equity": self.equity}

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = int(action)

        action_map = {0: 0, 1: 1, 2: -1}
        signal = action_map[action]

        price = self.df["close_15m"].iloc[self.current_step]
        reward = 0.0

        # ── Gestion de position ──────────────────────────────
        if self.position == 0:
            # Flat → ouvrir position
            if signal != 0:
                self.position = signal
                self.entry_price = price
                self.steps_in_position = 0
                cost = self.transaction_cost * self.equity
                self.equity -= cost
                reward -= cost / self.initial_capital * 10  # petite pénalité coût
                self.trades.append({"step": self.current_step, "action": signal, "price": price})
            else:
                self.idle_steps += 1
                # Légère pénalité si trop d'inactivité
                if self.idle_steps > 50:
                    reward -= 0.0001

        elif signal == 0 or signal == self.position:
            # HOLD position existante → PnL non réalisé
            self.steps_in_position += 1
            prev_price = self.df["close_15m"].iloc[self.current_step - 1]
            unrealized = self.position * (price - prev_price) / prev_price
            self.equity *= (1 + unrealized)
            # Petit reward pour maintenir une position gagnante
            reward += unrealized * 10

        else:
            # Fermer position (signal opposé ou fermeture)
            prev_price = self.df["close_15m"].iloc[self.current_step - 1]
            unrealized = self.position * (price - prev_price) / prev_price
            self.equity *= (1 + unrealized)

            # PnL réalisé sur le trade complet
            trade_pnl = self.position * (price - self.entry_price) / self.entry_price
            cost = self.transaction_cost * self.equity
            self.equity -= cost

            # Reward principal : PnL du trade complet
            reward += trade_pnl * 50  # amplifier le signal

            # Bonus trade gagnant
            if trade_pnl > 0:
                reward += 0.1  # bonus fixe pour gagner
            else:
                reward -= 0.05  # pénalité pour perdre

            self.trade_results.append(trade_pnl)
            self.trades.append({"step": self.current_step, "action": -self.position, "price": price, "pnl": trade_pnl})

            # Si signal opposé, ouvrir nouvelle position
            if signal != 0:
                self.position = signal
                self.entry_price = price
                self.steps_in_position = 0
                cost2 = self.transaction_cost * self.equity
                self.equity -= cost2
            else:
                self.position = 0
                self.entry_price = 0.0
                self.steps_in_position = 0

            self.idle_steps = 0

        # ── Drawdown check ───────────────────────────────────
        self.peak_equity = max(self.peak_equity, self.equity)
        current_dd = (self.peak_equity - self.equity) / self.peak_equity

        # Pénalité drawdown seulement si > 10%
        if current_dd > 0.10:
            reward -= (current_dd - 0.10) * 0.5

        # ── Avancer ──────────────────────────────────────────
        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        truncated = False

        # Stop loss à -30%
        if self.equity < self.initial_capital * 0.7:
            terminated = True
            reward -= 2.0

        # Reward de fin d'épisode
        if terminated and self.equity > self.initial_capital:
            final_return = (self.equity / self.initial_capital - 1)
            reward += final_return * 20  # bonus proportionnel au profit final

        obs = self._get_observation() if not terminated else np.zeros(
            self.observation_space.shape, dtype=np.float32
        )

        info = {
            "equity": self.equity,
            "position": self.position,
            "drawdown": current_dd,
            "nb_trades": len(self.trades),
        }

        return obs, reward, terminated, truncated, info

    def get_performance_summary(self) -> dict:
        profit_pct = (self.equity / self.initial_capital - 1) * 100
        max_dd = (self.peak_equity - min(self.equity, self.peak_equity)) / self.peak_equity * 100
        win_trades = [t for t in self.trade_results if t > 0]
        lose_trades = [t for t in self.trade_results if t <= 0]
        win_rate = len(win_trades) / max(1, len(self.trade_results)) * 100

        return {
            "profit_pct": profit_pct,
            "final_equity": self.equity,
            "max_drawdown_pct": max_dd,
            "nb_trades": len(self.trade_results),
            "win_rate": win_rate,
            "avg_win": np.mean(win_trades) * 100 if win_trades else 0,
            "avg_loss": np.mean(lose_trades) * 100 if lose_trades else 0,
        }
