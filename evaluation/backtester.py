"""
T06 – Backtester Engine
Simulation de trading avec gestion des positions, coûts de transaction,
et calcul des métriques de performance.

Usage :
  from evaluation.backtester import Backtester
"""

import numpy as np
import pandas as pd
from typing import List


class Backtester:
    """
    Engine de backtest pour stratégies GBP/USD M15.

    Positions : 0 = flat, 1 = long, -1 = short
    Signaux   : 1 = BUY, -1 = SELL, 0 = HOLD

    Paramètres :
        transaction_cost : coût par aller-retour en proportion (default: 0.0002 ≈ 2 pips)
        initial_capital  : capital initial
    """

    def __init__(self, transaction_cost: float = 0.0002, initial_capital: float = 10_000):
        self.transaction_cost = transaction_cost
        self.initial_capital = initial_capital

    def run(self, prices: pd.Series, signals: pd.Series) -> dict:
        """
        Exécute le backtest.

        Args:
            prices  : série des prix close_15m (indexés par timestamp)
            signals : série des signaux (1=BUY, -1=SELL, 0=HOLD)

        Returns:
            dict avec equity_curve, trades, metrics
        """
        assert len(prices) == len(signals), "prices et signals doivent avoir la même longueur"

        prices = prices.values
        signals = signals.values
        n = len(prices)

        equity = np.zeros(n)
        equity[0] = self.initial_capital
        position = 0  # flat
        entry_price = 0.0
        trades = []
        pnl_per_trade = []

        for i in range(1, n):
            signal = signals[i]
            price = prices[i]
            prev_price = prices[i - 1]

            # PnL sur position existante
            if position != 0:
                pnl = position * (price - prev_price) / prev_price * equity[i - 1]
            else:
                pnl = 0.0

            equity[i] = equity[i - 1] + pnl

            # Changement de position
            if signal != 0 and signal != position:
                # Coût de transaction (entrée ou retournement)
                cost = self.transaction_cost * equity[i]
                if position != 0:
                    # Clôture de la position précédente
                    trade_pnl = position * (price - entry_price) / entry_price
                    pnl_per_trade.append(trade_pnl)
                    cost *= 2  # coût de retournement

                equity[i] -= cost
                position = int(signal)
                entry_price = price

                trades.append({
                    "index": i,
                    "price": price,
                    "signal": signal,
                    "equity": equity[i],
                })

            elif signal == 0 and position != 0:
                # HOLD → on ne change rien, la position reste ouverte
                pass

        # Clôture forcée à la fin
        if position != 0:
            trade_pnl = position * (prices[-1] - entry_price) / entry_price
            pnl_per_trade.append(trade_pnl)

        # Calcul des métriques
        metrics = self._compute_metrics(equity, pnl_per_trade)
        metrics["nb_trades"] = len(trades)

        return {
            "equity_curve": equity,
            "trades": trades,
            "metrics": metrics,
            "pnl_per_trade": pnl_per_trade,
        }

    def _compute_metrics(self, equity: np.ndarray, pnl_per_trade: list) -> dict:
        """Calcule les métriques de performance."""
        metrics = {}

        # Profit cumulé (%)
        metrics["profit_cumule_pct"] = (equity[-1] / equity[0] - 1) * 100
        metrics["profit_cumule_abs"] = equity[-1] - equity[0]

        # Maximum drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        metrics["max_drawdown_pct"] = drawdown.min() * 100

        # Sharpe simplifié (annualisé, 252 jours * ~26 bougies M15 par jour)
        returns = np.diff(equity) / equity[:-1]
        if returns.std() > 0:
            periods_per_year = 252 * 26  # bougies M15 par an
            metrics["sharpe"] = returns.mean() / returns.std() * np.sqrt(periods_per_year)
        else:
            metrics["sharpe"] = 0.0

        # Profit factor
        if len(pnl_per_trade) > 0:
            gains = sum(p for p in pnl_per_trade if p > 0)
            losses = abs(sum(p for p in pnl_per_trade if p < 0))
            metrics["profit_factor"] = gains / losses if losses > 0 else float("inf")
            metrics["win_rate"] = sum(1 for p in pnl_per_trade if p > 0) / len(pnl_per_trade) * 100
        else:
            metrics["profit_factor"] = 0.0
            metrics["win_rate"] = 0.0

        return metrics
