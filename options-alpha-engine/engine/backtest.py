"""Minimal, honesty-first backtest engine + risk metrics.

Most options backtests lie. They fill at mid, ignore commissions, and assume
infinite liquidity. This engine makes costs *explicit and mandatory* so the
equity curve you get is the pessimistic, tradeable one — the only kind worth
trusting.

The engine is strategy-agnostic: feed it a list of daily P&L observations
(already net of the costs your strategy chose to model) and it returns an
equity curve plus the metrics that actually matter for survival.

RUIN IS A RESULT, NOT AN ERROR, and this module had no concept of it. The one
trade this engine exists to measure is SELLING options, whose loss is unbounded,
so an equity curve reaching zero is the case that matters most -- and it was the
case that broke. Equity passing exactly through zero raised ZeroDivisionError
from the returns loop, and an account ending BELOW zero gave a CAGR of
(-1.0089+0.7051j), a complex number, because a negative base raised to a
fractional power is complex in Python; `summary()` then raised ValueError trying
to format it as a percentage. Both came out of `engine/benchmark.py`, which
prints these numbers.

So the curve now STOPS at the first non-positive equity. P&L after that point is
fiction: the position is closed out and there is no capital behind it, and
averaging post-ruin returns into a Sharpe ratio would report a survivor's
statistic for an account that did not survive. `ruined` and `ruin_day` say so,
`cagr` is -100% rather than a complex number, and `total_return` stays
arithmetic -- it can read below -100% on a margin account, which is the honest
number when you end up owing money.

SHARPE AND SORTINO USE DIFFERENT DIVISORS, deliberately and conventionally:
the Sharpe's standard deviation is a SAMPLE one over n-1, while the Sortino's
target downside deviation normalises by the FULL n, counting days with no
downside as zeros rather than dropping them. They are therefore not two views of
one number and should not be read as a ratio of each other.
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
        #
        # The side is VALIDATED rather than defaulted. `if side == "buy"` meant
        # every other string -- "Buy", "BUY", "sel", "" and None included -- was
        # silently filled as a SELL, a full spread away from the intended fill
        # and in the wrong direction. `cost_vs_mid` takes an absolute value and
        # so reported the right friction either way, which is what made it
        # invisible. This matches `pricing.price`'s handling of `kind`.
        side = self.side.lower() if isinstance(self.side, str) else self.side
        if side not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got {self.side!r}")
        half = 0.5 * self.spread
        slip = self.slippage_frac * self.intended_price
        if side == "buy":
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
    #: True when the equity curve reached zero or below. The curve is truncated
    #: there and every metric describes only the period the account existed for,
    #: so a Sharpe ratio reported alongside ``ruined`` is not a reason to trade.
    ruined: bool = False
    #: 1-based index into the supplied P&L of the day the account was wiped out.
    ruin_day: int | None = None

    def summary(self) -> str:
        lines = [
            f"Total return : {self.total_return:+.1%}",
            f"CAGR         : {self.cagr:+.1%}",
            f"Sharpe       : {self.sharpe:.2f}",
            f"Sortino      : {self.sortino:.2f}",
            f"Max drawdown : {self.max_drawdown:.1%}",
            f"Hit rate     : {self.hit_rate:.1%}",
        ]
        if self.ruined:
            lines.insert(0, f"RUINED       : equity hit zero on day {self.ruin_day}")
        return "\n".join(lines)


def max_drawdown(equity: Sequence[float]) -> float:
    """Worst peak-to-trough fraction, as a NEGATIVE number (0.0 for a curve that
    only rises).

    The starting value has to be positive or there is no capital to measure a
    drawdown against, and `equity[0] / peak` divided by zero instead of saying
    so. Note that a curve reaching zero needs no special case here -- the peak
    is already positive, so the ratio is -1.0 on its own -- and a MARGIN account
    going negative gives a reading below -100%, which is the honest number when
    the loss exceeds the capital.
    """
    if not equity:
        raise ValueError("no equity curve supplied")
    if equity[0] <= 0.0:
        raise ValueError(f"equity curve must start above zero, got {equity[0]!r}")
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
    if starting_equity <= 0:
        raise ValueError(f"starting_equity must be positive, got {starting_equity!r}")

    # Stop at ruin. Everything after the account reaches zero is fiction, and
    # letting it through is what produced a complex CAGR and a division by zero.
    equity = [starting_equity]
    ruin_day = None
    for day, pnl in enumerate(daily_pnl, start=1):
        equity.append(equity[-1] + pnl)
        if equity[-1] <= 0.0:
            ruin_day = day
            break

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
    # A non-positive final equity has no real growth rate: the formula returns a
    # complex number for a negative base and a fractional exponent. -100% is the
    # limit as the final value approaches zero from above, and it is what ruin is.
    if equity[-1] <= 0.0:
        cagr = -1.0
    else:
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
        ruined=ruin_day is not None,
        ruin_day=ruin_day,
    )
