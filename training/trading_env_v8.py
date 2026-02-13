"""
T08 v8 – Environnement V6 + Features de Régime (ADX, ATR Ratio)
Objectif : Aider le modèle à distinguer les marchés tendance (Trend) vs range (Choppy).
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class TradingEnvV8(gym.Env):
    """
    Env v8: Basé sur V6 (Optimized numpy).
    Ajouts:
    - ADX (Trend Strength)
    - ATR Ratio (Volatility Regime)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int = 30,
        spread: float = 0.00015,
        take_profit_pct: float = 0.003,
        stop_loss_pct: float = 0.002,
        max_hold: int = 48,
        cooldown: int = 4,
    ):
        super().__init__()
        # Reset index but keep 'timestamp' column if it was the index
        self.df = df.reset_index(drop=False)
        self.window_size = window_size
        self.spread = spread
        self.tp_pct = take_profit_pct
        self.sl_pct = stop_loss_pct
        self.max_hold = max_hold
        self.cooldown = cooldown

        self.feature_columns = self._compute_features()
        self.n_features = len(self.feature_columns)
        
        # Convert to numpy for faster access
        self._data_matrix = self.df[self.feature_columns].values.astype(np.float32)
        self._close_prices = self.df["close_15m"].values.astype(np.float32)

        # obs size
        obs_size = self.window_size * self.n_features + 5
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

        self._init()

    def _init(self):
        self.step_idx = self.window_size
        self.position = 0
        self.entry_price = 0.0
        self.hold_time = 0
        self.cooldown_left = 0
        self.equity = 1.0
        self.peak_equity = 1.0
        self.trade_results = []
        self.equity_curve = [1.0]

    def _compute_features(self):
        df = self.df.copy()
        c = df["close_15m"]
        h = df["high_15m"]
        l = df["low_15m"]

        # Returns
        for p in [1, 2, 4, 8, 16, 32, 64]:
            df[f"r{p}"] = c.pct_change(p)

        # Tech indicators (same as V5)
        for s in [8, 21, 55]:
            df[f"e{s}"] = (c - c.ewm(span=s, adjust=False).mean()) / c
        
        # RSI
        d = c.diff()
        g = d.where(d > 0, 0.0).rolling(14).mean()
        l_loss = (-d.where(d < 0, 0.0)).rolling(14).mean() # renamed l to l_loss to avoid conflict
        df["rsi"] = (100 - 100 / (1 + g / (l_loss + 1e-10))) / 100 - 0.5

        # MACD
        e12 = c.ewm(span=12).mean()
        e26 = c.ewm(span=26).mean()
        macd = e12 - e26
        df["macd"] = (macd - macd.ewm(span=9).mean()) / c * 100

        # Bollinger Bands
        sma = c.rolling(20).mean()
        std = c.rolling(20).std()
        df["bb"] = (c - sma) / (2 * std + 1e-10)

        # ATR & ATR Ratio (New in V8)
        hl = h - l
        hc = abs(h - c.shift())
        lc = abs(l - c.shift())
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        df["atr"] = atr / c
        # Volatility Regime: Current ATR vs Long Term ATR
        df["atr_ratio"] = atr / (tr.rolling(100).mean() + 1e-10)

        # ADX (New in V8)
        # +DM, -DM
        up = h - h.shift()
        down = l.shift() - l
        plus_dm = np.where((up > down) & (up > 0), up, 0.0)
        minus_dm = np.where((down > up) & (down > 0), down, 0.0)
        
        # Rolling sums for DM and TR
        # Wilder's smoothing is ideal but EMA is close enough for ML features
        tr_smooth = tr.ewm(alpha=1/14, adjust=False).mean()
        plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean() / (tr_smooth + 1e-10)
        minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean() / (tr_smooth + 1e-10)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        df["adx"] = dx.ewm(alpha=1/14, adjust=False).mean() / 100.0 # Normalize 0-1

        # Time features (From V6)
        if "timestamp" in df.columns:
            dt = df["timestamp"]
        elif isinstance(df.index, pd.DatetimeIndex):
            dt = df.index.to_series()
        else:
            dt = pd.to_datetime(np.zeros(len(df)))

        df["sin_hour"] = np.sin(2 * np.pi * dt.dt.hour / 24)
        df["cos_hour"] = np.cos(2 * np.pi * dt.dt.hour / 24)
        df["sin_day"] = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
        df["cos_day"] = np.cos(2 * np.pi * dt.dt.dayofweek / 7)

        features = [
            "r1", "r2", "r4", "r8", "r16", "r32", "r64",
            "e8", "e21", "e55",
            "rsi", "macd", "bb", "atr", "atr_ratio", "adx",
            "sin_hour", "cos_hour", "sin_day", "cos_day"
        ]

        for col in features:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = df[col].fillna(0).replace([np.inf, -np.inf], 0)

        self.df = df
        return features

    def _get_obs(self):
        s = max(0, self.step_idx - self.window_size)
        w = self._data_matrix[s:self.step_idx]

        if len(w) < self.window_size:
            w = np.vstack([np.zeros((self.window_size - len(w), self.n_features)), w])

        m = w.mean(0)
        sd = w.std(0) + 1e-8
        w = np.clip((w - m) / sd, -3, 3)

        upnl = 0.0
        if self.position != 0:
            price = self._close_prices[self.step_idx]
            upnl = self.position * (price - self.entry_price) / self.entry_price
            upnl = np.clip(upnl, -0.05, 0.05)

        return np.concatenate([
            w.flatten(),
            [float(self.position != 0),
             float(self.position),
             min(self.hold_time / self.max_hold, 1.0),
             upnl,
             min(self.cooldown_left / self.cooldown, 1.0)]
        ]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._init()
        return self._get_obs(), {}

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = int(action)

        desired = {0: 0, 1: 1, 2: -1}[action]
        price = self._close_prices[self.step_idx]
        # We need high/low for TP/SL checking. 
        # Since I optimized close prices to numpy, I should do the same for high/low or just access df (slower)
        # Accessing df.iloc is slow. Optimize high/low access too?
        # For now, let's just use df.iloc or optimize it. Ideally verify high/low is in df.
        # Let's optimize high/low in __init__ too if I want max speed, but let's stick to df for now to minimize code changes unless crucial.
        # Actually V6 used df.iloc for high/low. I'll stick to that for now.
        
        high = self.df["high_15m"].iloc[self.step_idx]
        low = self.df["low_15m"].iloc[self.step_idx]
        
        reward = 0.0

        if self.position != 0:
            self.hold_time += 1
            if self.position == 1:
                max_pnl = (high - self.entry_price) / self.entry_price
                min_pnl = (low - self.entry_price) / self.entry_price
            else:
                max_pnl = (self.entry_price - low) / self.entry_price
                min_pnl = (self.entry_price - high) / self.entry_price

            close_pnl = self.position * (price - self.entry_price) / self.entry_price
            close_reason = None

            if max_pnl >= self.tp_pct:
                net_pnl = self.tp_pct - self.spread
                close_reason = "TP"
            elif min_pnl <= -self.sl_pct:
                net_pnl = -self.sl_pct - self.spread
                close_reason = "SL"
            elif self.hold_time >= self.max_hold:
                net_pnl = close_pnl - self.spread
                close_reason = "TIMEOUT"

            if close_reason:
                self.equity *= (1 + net_pnl)
                self.trade_results.append({"pnl": net_pnl, "hold": self.hold_time, "reason": close_reason})
                reward = net_pnl * 200
                if close_reason == "TP": reward += 0.5
                elif close_reason == "SL": reward -= 0.3
                self.position = 0; self.entry_price = 0; self.hold_time = 0; self.cooldown_left = self.cooldown
            else:
                reward = close_pnl * 5

        elif self.cooldown_left > 0:
            self.cooldown_left -= 1
            reward = 0.001
        elif desired != 0:
            self.position = desired; self.entry_price = price; self.hold_time = 0;
            reward = -0.02
        else:
            reward = 0.0

        self.peak_equity = max(self.peak_equity, self.equity)
        self.equity_curve.append(self.equity) # Track equity curve
        self.step_idx += 1
        terminated = self.step_idx >= len(self.df) - 1
        truncated = False

        if terminated:
            ret = self.equity - 1.0
            if ret > 0: reward += ret * 50
            elif ret < -0.3: reward -= 1.0

        obs = self._get_obs() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, reward, terminated, truncated, {"equity": self.equity, "position": self.position, "nb_trades": len(self.trade_results)}

    def get_performance_summary(self):
        ret = (self.equity - 1.0) * 100
        ec = self.equity_curve
        peak = max(ec)
        idx = ec.index(peak)
        trough = min(ec[idx:]) if idx < len(ec) else peak
        max_dd = (peak - trough) / peak * 100
        
        trades = self.trade_results
        n = len(trades)
        wins = [t["pnl"] for t in trades if t["pnl"] > 0]
        losses = [t["pnl"] for t in trades if t["pnl"] <= 0]
        wr = len(wins) / max(1, n) * 100
        
        return {
            "profit_pct": ret,
            "final_equity": self.equity * 10000,
            "max_drawdown_pct": max_dd,
            "nb_trades": n,
            "win_rate": wr,
            "avg_win": np.mean(wins) * 100 if wins else 0,
            "avg_loss": np.mean(losses) * 100 if losses else 0,
            "tp_count": sum(1 for t in trades if t["reason"] == "TP"),
            "sl_count": sum(1 for t in trades if t["reason"] == "SL"),
            "timeout_count": sum(1 for t in trades if t["reason"] == "TIMEOUT"),
            "avg_hold": np.mean([t["hold"] for t in trades]) if trades else 0,
        }
