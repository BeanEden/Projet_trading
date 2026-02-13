"""
T08 v3 – Environnement de trading RL (anti-overtrading)
Solution: pénaliser fortement chaque ouverture de trade + reward simplifiée.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class TradingEnvV3(gym.Env):
    """
    Environnement optimisé pour réduire l'overtrading.
    
    Changements clés vs v2:
    - Cooldown obligatoire entre les trades (min_hold)
    - Coût de trade beaucoup plus élevé dans la reward
    - Reward centrée sur l'equity relative (log return)
    - Action space: 0=FLAT (fermer), 1=LONG, 2=SHORT
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int = 30,
        spread: float = 0.0003,   # 3 pips spread (~réaliste GBP/USD)
        initial_capital: float = 10_000,
        min_hold: int = 4,         # minimum 4 bougies (1h) avant de pouvoir clôturer
    ):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.spread = spread
        self.initial_capital = initial_capital
        self.min_hold = min_hold

        self.feature_columns = self._compute_features()
        self.n_features = len(self.feature_columns)

        # obs = window features aplaties + [position, unrealized_pnl_norm, hold_time_norm]
        obs_size = self.window_size * self.n_features + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)  # FLAT, LONG, SHORT

        self._reset_state()

    def _reset_state(self):
        self.current_step = self.window_size
        self.position = 0
        self.equity = self.initial_capital
        self.peak_equity = self.initial_capital
        self.entry_price = 0.0
        self.hold_time = 0
        self.trades = []
        self.trade_results = []
        self.equity_curve = [self.initial_capital]

    def _compute_features(self):
        df = self.df
        close = df["close_15m"]

        # Returns multi-horizon
        df["ret_1"] = close.pct_change(1)
        df["ret_4"] = close.pct_change(4)
        df["ret_16"] = close.pct_change(16)

        # EMAs normalisées
        for span in [10, 30, 60]:
            ema = close.ewm(span=span, adjust=False).mean()
            df[f"ema_{span}_d"] = (close - ema) / close

        # EMA cross
        df["ema_cross"] = (close.ewm(span=10, adjust=False).mean() -
                           close.ewm(span=30, adjust=False).mean()) / close

        # RSI 14
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        df["rsi"] = (100 - (100 / (1 + rs))) / 100

        # MACD normalisé
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        df["macd_d"] = ((ema12 - ema26) - (ema12 - ema26).ewm(span=9, adjust=False).mean()) / close

        # Bollinger band position
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        df["bb_pos"] = (close - sma20) / (2 * std20 + 1e-10)  # [-1, 1]

        # ATR normalisé
        hl = df["high_15m"] - df["low_15m"]
        hc = abs(df["high_15m"] - close.shift())
        lc = abs(df["low_15m"] - close.shift())
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        df["atr"] = tr.rolling(14).mean() / close

        # Volatilité
        df["vol_20"] = df["ret_1"].rolling(20).std()

        # Body et range
        df["body"] = (close - df["open_15m"]) / close
        df["range"] = (df["high_15m"] - df["low_15m"]) / close

        features = [
            "ret_1", "ret_4", "ret_16",
            "ema_10_d", "ema_30_d", "ema_60_d", "ema_cross",
            "rsi", "macd_d", "bb_pos",
            "atr", "vol_20",
            "body", "range",
        ]

        for col in features:
            df[col] = df[col].fillna(0).replace([np.inf, -np.inf], 0)

        self.df = df
        return features

    def _get_observation(self):
        start = max(0, self.current_step - self.window_size)
        end = self.current_step
        window = self.df[self.feature_columns].iloc[start:end].values

        if len(window) < self.window_size:
            pad = np.zeros((self.window_size - len(window), self.n_features))
            window = np.concatenate([pad, window])

        # Z-score
        m = window.mean(axis=0)
        s = window.std(axis=0) + 1e-8
        window = (window - m) / s
        window = np.clip(window, -5, 5)  # clip extrêmes

        # Unrealized PnL
        unrealized = 0.0
        if self.position != 0 and self.entry_price > 0:
            price = self.df["close_15m"].iloc[self.current_step]
            unrealized = self.position * (price - self.entry_price) / self.entry_price
            unrealized = np.clip(unrealized, -0.1, 0.1)  # clip à ±10%

        obs = np.concatenate([
            window.flatten(),
            [float(self.position),
             unrealized,
             min(self.hold_time / 50.0, 1.0)]
        ])
        return obs.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._get_observation(), {"equity": self.equity}

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = int(action)

        # Map: 0=FLAT, 1=LONG, 2=SHORT
        desired = {0: 0, 1: 1, 2: -1}[action]

        price = self.df["close_15m"].iloc[self.current_step]
        prev_price = self.df["close_15m"].iloc[self.current_step - 1]
        reward = 0.0

        # ── Mettre à jour l'equity sur la position courante ──
        if self.position != 0:
            step_return = self.position * (price - prev_price) / prev_price
            self.equity *= (1 + step_return)
            self.hold_time += 1

        # ── Gestion des changements de position ──────────────
        # Bloquer le changement si min_hold pas atteint
        can_change = (self.hold_time >= self.min_hold or self.position == 0)

        if desired != self.position and can_change:
            # Fermer position existante
            if self.position != 0:
                trade_pnl = self.position * (price - self.entry_price) / self.entry_price
                spread_cost = self.spread
                net_pnl = trade_pnl - spread_cost

                # Reward basée sur le PnL net du trade
                reward += net_pnl * 100  # amplifier

                self.equity -= spread_cost * self.equity  # coût spread
                self.trade_results.append(net_pnl)
                self.trades.append({
                    "step": self.current_step,
                    "action": "close",
                    "price": price,
                    "pnl": net_pnl,
                    "hold": self.hold_time
                })

                self.position = 0
                self.entry_price = 0.0
                self.hold_time = 0

            # Ouvrir nouvelle position
            if desired != 0:
                self.position = desired
                self.entry_price = price
                self.hold_time = 0
                self.equity -= self.spread * self.equity  # coût spread
                reward -= self.spread * 100  # pénalité d'ouverture

                self.trades.append({
                    "step": self.current_step,
                    "action": "long" if desired == 1 else "short",
                    "price": price,
                })

        # ── Reward de holding ────────────────────────────────
        # Petit reward continu basé sur le PnL de ce step
        if self.position != 0:
            step_pnl = self.position * (price - prev_price) / prev_price
            reward += step_pnl * 10  # signal continu faible

        # ── Drawdown ─────────────────────────────────────────
        self.peak_equity = max(self.peak_equity, self.equity)
        dd = (self.peak_equity - self.equity) / self.peak_equity

        # ── Avancer ──────────────────────────────────────────
        self.current_step += 1
        self.equity_curve.append(self.equity)
        terminated = self.current_step >= len(self.df) - 1
        truncated = False

        # Stop loss
        if self.equity < self.initial_capital * 0.7:
            terminated = True
            reward -= 1.0

        # Bonus de fin profitable
        if terminated:
            total_return = self.equity / self.initial_capital - 1
            if total_return > 0:
                reward += total_return * 30

        obs = self._get_observation() if not terminated else np.zeros(
            self.observation_space.shape, dtype=np.float32
        )

        info = {
            "equity": self.equity,
            "position": self.position,
            "drawdown": dd,
            "nb_trades": len(self.trade_results),
        }

        return obs, reward, terminated, truncated, info

    def get_performance_summary(self):
        profit_pct = (self.equity / self.initial_capital - 1) * 100
        peak = max(self.equity_curve) if self.equity_curve else self.initial_capital
        trough = min(self.equity_curve[self.equity_curve.index(peak):]) if self.equity_curve else self.initial_capital
        max_dd = (peak - trough) / peak * 100

        wins = [t for t in self.trade_results if t > 0]
        losses = [t for t in self.trade_results if t <= 0]
        n_trades = len(self.trade_results)
        win_rate = len(wins) / max(1, n_trades) * 100

        return {
            "profit_pct": profit_pct,
            "final_equity": self.equity,
            "max_drawdown_pct": max_dd,
            "nb_trades": n_trades,
            "win_rate": win_rate,
            "avg_win": np.mean(wins) * 100 if wins else 0,
            "avg_loss": np.mean(losses) * 100 if losses else 0,
            "avg_hold": np.mean([t.get("hold", 0) for t in self.trades if "hold" in t]) if self.trades else 0,
        }
