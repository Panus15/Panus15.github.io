"""Minimal, honesty-first backtest engine + risk metrics.

Most options backtests lie. They fill at mid, ignore commissions, and assume
infinite liquidity. This engine makes costs *explicit and mandatory* so the
equity curve you get is the pessimistic, tradeable one — the only kind worth
trusting.

The engine is strategy-agnostic: feed it a list of daily P&L observations
(already net of the costs your strategy chose to model) and it returns an
equity curve plus the metrics that actually matter for survival.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

TRADING_DAYS = 252


@dataclass
class Fill:
    """A modelled fill. Never assume you trade at mid."""
    intended_price: float   # e.g. mid
    side: str               # "buy" or "sell"
    spread: float
    commission: float = 0.65   # per contract, typical retail
    slippage_frac: float = 0.0  # extra adverse move as fraction of price

    @property
    def executed_price(self) -> float:
        # Buys lift the ask, sells hit the bid: you pay half the spread each way.
        half = 0.5 * self.spread
        slip = self.slippage_frac * self.intended_price
        if self.side == "buy":
            return self.intended_price + half + slip
        return self.intended_price - half - slip

    def cost_vs_mid(self, contracts: int, multiplier: int = 100) -> float:
        """Total friction (spread + slippage + commission) in account currency."""
        per_contract = abs(self.executed_price - self.intended_price) * multiplier
        return contracts * (per_contract + self.commission)


@dataclass
class BacktestResult:
    equity_curve: list[float]
    daily_returns: list[float]
    total_return: float
    cagr: float
    sharpe: float
    sortino: float
    max_drawdown: float
    hit_rate: float

    def summary(self) -> str:
        return (
            f"Total return : {self.total_return:+.1%}\n"
            f"CAGR         : {self.cagr:+.1%}\n"
            f"Sharpe       : {self.sharpe:.2f}\n"
            f"Sortino      : {self.sortino:.2f}\n"
            f"Max drawdown : {self.max_drawdown:.1%}\n"
            f"Hit rate     : {self.hit_rate:.1%}"
        )


def max_drawdown(equity: Sequence[float]) -> float:
    peak = equity[0]
    mdd = 0.0
    for v in equity:
        peak = max(peak, v)
        mdd = min(mdd, v / peak - 1.0)
    return mdd


def run_backtest(daily_pnl: Sequence[float], starting_equity: float = 100_000.0) -> BacktestResult:
    """Turn a stream of daily P&L (in currency, net of costs) into metrics."""
    if not daily_pnl:
        raise ValueError("no P&L supplied")

    equity = [starting_equity]
    for pnl in daily_pnl:
        equity.append(equity[-1] + pnl)

    rets = [equity[i] / equity[i - 1] - 1.0 for i in range(1, len(equity))]
    mean = sum(rets) / len(rets)

    var = sum((x - mean) ** 2 for x in rets) / max(len(rets) - 1, 1)
    std = math.sqrt(var)
    downside = [min(x, 0.0) for x in rets]
    dvar = sum(x * x for x in downside) / max(len(downside), 1)
    dstd = math.sqrt(dvar)

    sharpe = (mean / std * math.sqrt(TRADING_DAYS)) if std > 0 else 0.0
    sortino = (mean / dstd * math.sqrt(TRADING_DAYS)) if dstd > 0 else 0.0

    total_return = equity[-1] / equity[0] - 1.0
    years = len(rets) / TRADING_DAYS
    cagr = (equity[-1] / equity[0]) ** (1 / years) - 1.0 if years > 0 else 0.0
    hit_rate = sum(1 for x in rets if x > 0) / len(rets)

    return BacktestResult(
        equity_curve=equity,
        daily_returns=rets,
        total_return=total_return,
        cagr=cagr,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_drawdown(equity),
        hit_rate=hit_rate,
    )
