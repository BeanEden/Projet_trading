"""
T08 v4 – Environnement de trading RL
Approche différente: pas de stop-loss, log-return reward, épisode complet.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class TradingEnvV4(gym.Env):
    """
    Env v4: approche conservative.
    
    Principes:
    - Pas de terminaison anticipée (toujours épisode complet)
    - Reward = log-return de l'equity à chaque step
    - Spread réaliste déduit à chaque changement de position
    - Min hold pour éviter overtrading
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int = 30,
        spread: float = 0.00015,    # 1.5 pip (réaliste)
        min_hold: int = 8,          # 2h minimum de hold
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.spread = spread
        self.min_hold = min_hold

        self.feature_columns = self._compute_features()
        self.n_features = len(self.feature_columns)

        # obs: features + position + unrealized_pnl + hold_time
        obs_size = self.window_size * self.n_features + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)  # 0=FLAT, 1=LONG, 2=SHORT

        self._init_state()

    def _init_state(self):
        self.current_step = self.window_size
        self.position = 0
        self.entry_price = 0.0
        self.hold_time = 0
        self.equity = 1.0  # normalisé à 1.0
        self.prev_equity = 1.0
        self.peak_equity = 1.0
        self.trade_results = []
        self.equity_curve = [1.0]

    def _compute_features(self):
        df = self.df
        c = df["close_15m"]

        # Returns
        df["r1"] = c.pct_change(1)
        df["r4"] = c.pct_change(4)
        df["r16"] = c.pct_change(16)
        df["r64"] = c.pct_change(64)

        # EMAs normalisées
        for s in [8, 21, 55]:
            df[f"e{s}"] = (c - c.ewm(span=s, adjust=False).mean()) / c

        # RSI
        delta = c.diff()
        g = delta.where(delta > 0, 0.0).rolling(14).mean()
        l = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        df["rsi"] = (100 - 100 / (1 + g / (l + 1e-10))) / 100 - 0.5  # centré sur 0

        # MACD signal
        e12 = c.ewm(span=12, adjust=False).mean()
        e26 = c.ewm(span=26, adjust=False).mean()
        macd = e12 - e26
        sig = macd.ewm(span=9, adjust=False).mean()
        df["macd"] = (macd - sig) / c * 100

        # Bollinger position
        sma = c.rolling(20).mean()
        std = c.rolling(20).std()
        df["bb"] = (c - sma) / (2 * std + 1e-10)

        # ATR
        hl = df["high_15m"] - df["low_15m"]
        hc = abs(df["high_15m"] - c.shift())
        lc = abs(df["low_15m"] - c.shift())
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        df["atr"] = tr.rolling(14).mean() / c

        # Volatilité (ratio court/long)
        v5 = df["r1"].rolling(5).std()
        v20 = df["r1"].rolling(20).std()
        df["vratio"] = v5 / (v20 + 1e-10) - 1  # >0 = vol croissante

        # Bougie
        df["body"] = (c - df["open_15m"]) / c

        features = [
            "r1", "r4", "r16", "r64",
            "e8", "e21", "e55",
            "rsi", "macd", "bb",
            "atr", "vratio", "body",
        ]

        for col in features:
            df[col] = df[col].fillna(0).replace([np.inf, -np.inf], 0)

        self.df = df
        return features

    def _get_obs(self):
        s = max(0, self.current_step - self.window_size)
        e = self.current_step
        w = self.df[self.feature_columns].iloc[s:e].values

        if len(w) < self.window_size:
            w = np.vstack([np.zeros((self.window_size - len(w), self.n_features)), w])

        m = w.mean(0)
        s = w.std(0) + 1e-8
        w = np.clip((w - m) / s, -3, 3)

        # PnL non réalisé
        upnl = 0.0
        if self.position != 0 and self.entry_price > 0:
            p = self.df["close_15m"].iloc[self.current_step]
            upnl = self.position * (p - self.entry_price) / self.entry_price
            upnl = np.clip(upnl, -0.05, 0.05)

        return np.concatenate([
            w.flatten(),
            [float(self.position), upnl, min(self.hold_time / 100.0, 1.0)]
        ]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._init_state()
        return self._get_obs(), {}

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = int(action)

        desired = {0: 0, 1: 1, 2: -1}[action]
        price = self.df["close_15m"].iloc[self.current_step]
        prev_price = self.df["close_15m"].iloc[self.current_step - 1]

        # Update equity from position
        if self.position != 0:
            ret = self.position * (price - prev_price) / prev_price
            self.equity *= (1 + ret)
            self.hold_time += 1

        # Position change
        can_change = self.position == 0 or self.hold_time >= self.min_hold

        if desired != self.position and can_change:
            # Close existing
            if self.position != 0:
                pnl = self.position * (price - self.entry_price) / self.entry_price - self.spread
                self.equity *= (1 - self.spread)
                self.trade_results.append(pnl)
                self.position = 0
                self.entry_price = 0
                self.hold_time = 0

            # Open new
            if desired != 0:
                self.position = desired
                self.entry_price = price
                self.hold_time = 0
                self.equity *= (1 - self.spread)

        # Reward = log return of equity
        reward = float(np.log(max(self.equity, 1e-10) / max(self.prev_equity, 1e-10)))
        self.prev_equity = self.equity
        self.peak_equity = max(self.peak_equity, self.equity)
        self.equity_curve.append(self.equity)

        # Advance
        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        truncated = False

        obs = self._get_obs() if not terminated else np.zeros(
            self.observation_space.shape, dtype=np.float32
        )

        return obs, reward, terminated, truncated, {
            "equity": self.equity,
            "position": self.position,
            "nb_trades": len(self.trade_results),
        }

    def get_performance_summary(self):
        ret = (self.equity - 1.0) * 100
        peak = max(self.equity_curve)
        idx = self.equity_curve.index(peak)
        trough = min(self.equity_curve[idx:])
        max_dd = (peak - trough) / peak * 100

        wins = [t for t in self.trade_results if t > 0]
        losses = [t for t in self.trade_results if t <= 0]
        n = len(self.trade_results)
        wr = len(wins) / max(1, n) * 100

        return {
            "profit_pct": ret,
            "final_equity": self.equity * 10000,
            "max_drawdown_pct": max_dd,
            "nb_trades": n,
            "win_rate": wr,
            "avg_win": np.mean(wins) * 100 if wins else 0,
            "avg_loss": np.mean(losses) * 100 if losses else 0,
        }
