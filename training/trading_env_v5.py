"""
T08 v5 – Environnement de trading RL à signaux d'entrée
L'agent décide QUAND entrer et dans quelle direction.
Les sorties sont automatiques : Take-Profit ou Stop-Loss.
Cela empêche structurellement l'overtrading.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class TradingEnvV5(gym.Env):
    """
    Env v5: Signal d'entrée uniquement.
    
    Actions:
    - 0 = WAIT (ne rien faire)
    - 1 = ENTER LONG
    - 2 = ENTER SHORT
    
    Quand on est en position:
    - L'action est IGNORÉE
    - La position est fermée automatiquement quand:
      - TP atteint (take_profit_pct)
      - SL atteint (stop_loss_pct)
      - Timeout (max_hold bougies)
    
    Cela force le modèle à ne trader que quand il est sûr
    et empêche l'overtrading car il ne peut pas sortir manuellement.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int = 30,
        spread: float = 0.00015,
        take_profit_pct: float = 0.003,   # 0.3% TP (30 pips sur GBPUSD ~1.26)
        stop_loss_pct: float = 0.002,     # 0.2% SL (20 pips)
        max_hold: int = 48,               # 12h max hold (48 x 15min)
        cooldown: int = 4,                # 1h cooldown entre trades
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.spread = spread
        self.tp_pct = take_profit_pct
        self.sl_pct = stop_loss_pct
        self.max_hold = max_hold
        self.cooldown = cooldown

        self.feature_columns = self._compute_features()
        self.n_features = len(self.feature_columns)

        # obs: features + [in_position, direction, hold_time_norm, unrealized_pnl, cooldown_left]
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
        df = self.df
        c = df["close_15m"]

        # Multi-horizon returns
        for p in [1, 2, 4, 8, 16, 32, 64]:
            df[f"r{p}"] = c.pct_change(p)

        # EMAs
        for s in [8, 21, 55]:
            df[f"e{s}"] = (c - c.ewm(span=s, adjust=False).mean()) / c

        # RSI centré
        d = c.diff()
        g = d.where(d > 0, 0.0).rolling(14).mean()
        l = (-d.where(d < 0, 0.0)).rolling(14).mean()
        df["rsi"] = (100 - 100 / (1 + g / (l + 1e-10))) / 100 - 0.5

        # MACD
        e12 = c.ewm(span=12).mean()
        e26 = c.ewm(span=26).mean()
        macd = e12 - e26
        df["macd"] = (macd - macd.ewm(span=9).mean()) / c * 100

        # Bollinger
        sma = c.rolling(20).mean()
        std = c.rolling(20).std()
        df["bb"] = (c - sma) / (2 * std + 1e-10)

        # ATR
        hl = df["high_15m"] - df["low_15m"]
        hc = abs(df["high_15m"] - c.shift())
        lc = abs(df["low_15m"] - c.shift())
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        df["atr"] = tr.rolling(14).mean() / c

        # Vol ratio
        v5 = df["r1"].rolling(5).std()
        v20 = df["r1"].rolling(20).std()
        df["vratio"] = v5 / (v20 + 1e-10) - 1

        # Candle
        df["body"] = (c - df["open_15m"]) / c
        df["wick_up"] = (df["high_15m"] - df[["open_15m", "close_15m"]].max(axis=1)) / c
        df["wick_dn"] = (df[["open_15m", "close_15m"]].min(axis=1) - df["low_15m"]) / c

        features = [
            "r1", "r2", "r4", "r8", "r16", "r32", "r64",
            "e8", "e21", "e55",
            "rsi", "macd", "bb",
            "atr", "vratio",
            "body", "wick_up", "wick_dn",
        ]

        for col in features:
            df[col] = df[col].fillna(0).replace([np.inf, -np.inf], 0)

        self.df = df
        return features

    def _get_obs(self):
        s = max(0, self.step_idx - self.window_size)
        w = self.df[self.feature_columns].iloc[s:self.step_idx].values

        if len(w) < self.window_size:
            w = np.vstack([np.zeros((self.window_size - len(w), self.n_features)), w])

        m = w.mean(0)
        sd = w.std(0) + 1e-8
        w = np.clip((w - m) / sd, -3, 3)

        # Current unrealized
        upnl = 0.0
        if self.position != 0:
            p = self.df["close_15m"].iloc[self.step_idx]
            upnl = self.position * (p - self.entry_price) / self.entry_price
            upnl = np.clip(upnl, -0.05, 0.05)

        return np.concatenate([
            w.flatten(),
            [float(self.position != 0),  # in position (binary)
             float(self.position),       # direction
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
        price = self.df["close_15m"].iloc[self.step_idx]
        high = self.df["high_15m"].iloc[self.step_idx]
        low = self.df["low_15m"].iloc[self.step_idx]
        reward = 0.0

        # ── Si en position: vérifier TP/SL/timeout ──────────
        if self.position != 0:
            self.hold_time += 1

            # Calculer PnL intra-barre (utiliser high/low pour TP/SL)
            if self.position == 1:  # Long
                max_pnl = (high - self.entry_price) / self.entry_price
                min_pnl = (low - self.entry_price) / self.entry_price
            else:  # Short
                max_pnl = (self.entry_price - low) / self.entry_price
                min_pnl = (self.entry_price - high) / self.entry_price

            close_pnl = self.position * (price - self.entry_price) / self.entry_price
            close_reason = None

            # Check TP (favorable price during candle)
            if max_pnl >= self.tp_pct:
                net_pnl = self.tp_pct - self.spread
                close_reason = "TP"
            # Check SL
            elif min_pnl <= -self.sl_pct:
                net_pnl = -self.sl_pct - self.spread
                close_reason = "SL"
            # Check timeout
            elif self.hold_time >= self.max_hold:
                net_pnl = close_pnl - self.spread
                close_reason = "TIMEOUT"

            if close_reason:
                self.equity *= (1 + net_pnl)
                self.trade_results.append({
                    "pnl": net_pnl,
                    "hold": self.hold_time,
                    "reason": close_reason,
                })
                # Reward = net PnL du trade (amplifié)
                reward = net_pnl * 200
                # Bonus pour TP, pénalité pour SL
                if close_reason == "TP":
                    reward += 0.5
                elif close_reason == "SL":
                    reward -= 0.3

                self.position = 0
                self.entry_price = 0
                self.hold_time = 0
                self.cooldown_left = self.cooldown
            else:
                # En position, pas de close: petit reward continu
                reward = close_pnl * 5  # signal faible

        # ── Si pas en position: traiter l'action ────────────
        elif self.cooldown_left > 0:
            self.cooldown_left -= 1
            # Action ignorée pendant cooldown
            # Légère récompense pour patience
            reward = 0.001

        elif desired != 0:
            # Ouvrir une position
            self.position = desired
            self.entry_price = price
            self.hold_time = 0
            self.equity *= (1 - self.spread)
            # Petite pénalité d'entrée (encourage la sélectivité)
            reward = -0.02

        else:
            # WAIT - ne rien faire
            # Pas de pénalité ni récompense pour attendre
            reward = 0.0

        # ── Avancer ──────────────────────────────────────────
        self.peak_equity = max(self.peak_equity, self.equity)
        self.equity_curve.append(self.equity)
        self.step_idx += 1
        terminated = self.step_idx >= len(self.df) - 1
        truncated = False

        # Bonus/malus de fin
        if terminated:
            total_ret = self.equity - 1.0
            if total_ret > 0:
                reward += total_ret * 50
            elif total_ret < -0.3:
                reward -= 1.0

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

        tp_count = sum(1 for t in trades if t["reason"] == "TP")
        sl_count = sum(1 for t in trades if t["reason"] == "SL")
        to_count = sum(1 for t in trades if t["reason"] == "TIMEOUT")

        return {
            "profit_pct": ret,
            "final_equity": self.equity * 10000,
            "max_drawdown_pct": max_dd,
            "nb_trades": n,
            "win_rate": wr,
            "avg_win": np.mean(wins) * 100 if wins else 0,
            "avg_loss": np.mean(losses) * 100 if losses else 0,
            "tp_count": tp_count,
            "sl_count": sl_count,
            "timeout_count": to_count,
            "avg_hold": np.mean([t["hold"] for t in trades]) if trades else 0,
        }
