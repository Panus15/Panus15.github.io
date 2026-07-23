"""Benchmark harness — does the signal beat the strategies you could buy off
the shelf?

Every claim of edge needs a reference. This runs THREE books over the SAME
price path and chain series, so the comparison is apples-to-apples:

  signal-gated   sell vol only when the day's chain is RICH vs the P-forecast
                 and the regime is calm (the engine's actual strategy).
  always-sell    sell vol every eligible period, no questions asked — this is
                 mechanically what covered-call / option-income ETFs (QQQI,
                 JEPQ, ...) do. Their headline "yield" is harvested premium,
                 NOT evidence of a high win rate; this book is that strategy's
                 honest, cost-inclusive equity curve.
  buy-and-hold   own the underlying, do nothing. The bar every option overlay
                 must clear before it is worth its complexity.

The interesting read-outs: does signal-gating beat always-selling (is the
timing worth anything?), and does either beat just holding (is the whole
overlay worth anything?). On a strong bull path buy-and-hold usually wins
total return — the overlays earn their keep on drawdown/Sharpe, or not at all.
"""

from __future__ import annotations

from dataclasses import dataclass

from engine import backtest
from engine.signal_backtest import run_signal_backtest


@dataclass
class BookResult:
    name: str
    metrics: backtest.BacktestResult
    n_trades: int
    note: str = ""


def _buy_and_hold(prices, warmup: int, starting_equity: float) -> backtest.BacktestResult:
    """Hold starting_equity of the underlying from the warmup date. No costs
    beyond the implicit full-price entry (one-off spread is negligible here)."""
    shares = starting_equity / prices[warmup]
    pnl = [shares * (prices[i] - prices[i - 1]) for i in range(warmup + 1, len(prices))]
    return backtest.run_backtest(pnl, starting_equity=starting_equity)


def compare_books(prices, chain_at, forecaster, *, dte: int = 21, warmup: int = 63,
                  starting_equity: float = 100_000.0, **signal_kw) -> list[BookResult]:
    """Run all three books on one path; returns [signal, always-sell, buy-hold]."""
    sig = run_signal_backtest(prices, chain_at, forecaster, dte=dte, warmup=warmup,
                              starting_equity=starting_equity, **signal_kw)
    alw = run_signal_backtest(prices, chain_at, forecaster, dte=dte, warmup=warmup,
                              starting_equity=starting_equity, always_sell=True,
                              **signal_kw)
    bh = _buy_and_hold(prices, warmup, starting_equity)
    skips = "; ".join(f"{k}:{v}" for k, v in sig.skip_reasons.items()) or "none"
    return [
        BookResult("signal-gated", sig.metrics, sig.n_sold, f"skips {skips}"),
        BookResult("always-sell (income-ETF style)", alw.metrics, alw.n_sold,
                   "sells every usable period — the QQQI/JEPQ mechanic"),
        BookResult("buy-and-hold underlying", bh, 1, "the bar to clear"),
    ]


def summary_table(books: list[BookResult]) -> str:
    head = (f"  {'book':32} {'Sharpe':>7} {'Sortino':>8} {'MaxDD':>7} "
            f"{'total':>8} {'trades':>7}")
    lines = [head]
    for b in books:
        m = b.metrics
        lines.append(f"  {b.name:32} {m.sharpe:>7.2f} {m.sortino:>8.2f} "
                     f"{m.max_drawdown:>7.1%} {m.total_return:>+8.1%} {b.n_trades:>7}")
    for b in books:
        if b.note:
            lines.append(f"    - {b.name}: {b.note}")
    return "\n".join(lines)
