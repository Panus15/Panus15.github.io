"""Overnight-gap / jump stress — the risk a continuously-hedged backtest hides.

A delta-hedged short-vol book looks wonderful when you assume you can rebalance
continuously: the hedge neutralises the delta and you quietly collect theta. Real
markets do not cooperate. They GAP — overnight, over a weekend, on an earnings or
macro print — and you are short gamma across that gap with no chance to rebalance.
The jump goes straight into your P&L, and it is always the wrong way (a short
straddle loses on any large move, up or down).

This module injects those jumps into a price path and re-runs the book, so the
gamma bleed the smooth backtest omitted becomes a number you can look at. Use it
to pressure-test any result from signal_backtest / the paper ledger before you
believe its Sharpe.

  inject_gaps   apply multiplicative jumps at chosen bars (level shifts forward).
  gap_stress    run a signal book on the clean path vs the gapped path -> both
                results, so you can read the drawdown the gaps added.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def inject_gaps(prices, gaps):
    """Return a copy of ``prices`` with overnight jumps applied.

    ``gaps`` is a list of ``(index, log_return)``: at each index the price level
    jumps by ``exp(log_return)`` and the shift is CARRIED FORWARD (the gap moves
    the whole subsequent path, as a real overnight gap does — it is not a
    one-bar spike that mean-reverts). A short-gamma book cannot hedge across it."""
    out = list(prices)
    for idx, lr in gaps:
        if idx < 0 or idx >= len(out):
            continue
        f = math.exp(lr)
        for i in range(idx, len(out)):
            out[i] *= f
    return out


def evenly_spaced_gaps(n, *, every: int, size: float, start: int = 0,
                       offset: int = 0):
    """Alternating ±``size`` (log) jumps every ``every`` bars — a simple, brutal
    weekend-gap regime for stress runs. Deterministic (no RNG).

    ``offset`` shifts the whole schedule. It exists because of a trap: the caller
    also chooses ``dte``, and if ``every`` is a multiple of ``dte`` then EVERY gap
    lands on a trade entry/expiry boundary, where remaining time — and therefore
    gamma — is zero. The stress then measures nothing, because a short-gamma book
    is only hurt by a jump it is holding gamma across. ``gap_stress`` guards
    against this; if you call this function directly, pick ``every`` coprime with
    your holding period or set ``offset``.
    """
    gaps = []
    sign = 1.0
    i = start + every + offset
    while i < n:
        gaps.append((i, sign * size))
        sign = -sign
        i += every
    return gaps


@dataclass
class GapStressResult:
    clean: object          # SignalBacktestResult on the smooth path
    gapped: object         # SignalBacktestResult on the gapped path
    n_gaps: int
    gap_size: float

    def summary(self) -> str:
        c, g = self.clean.metrics, self.gapped.metrics
        return (f"gap stress: {self.n_gaps} jumps of ±{self.gap_size:.0%} (log)\n"
                f"  {'':10}{'Sharpe':>8} {'MaxDD':>8} {'total':>9}\n"
                f"  clean     {c.sharpe:>8.2f} {c.max_drawdown:>8.1%} {c.total_return:>+9.1%}\n"
                f"  gapped    {g.sharpe:>8.2f} {g.max_drawdown:>8.1%} {g.total_return:>+9.1%}\n"
                f"  -> gaps cost {g.total_return - c.total_return:+.1%} return, "
                f"{g.max_drawdown - c.max_drawdown:+.1%} drawdown (short gamma)")


def gap_stress(prices, chain_at, forecaster, *, every: int = 42, size: float = 0.10,
               **bt_kwargs) -> GapStressResult:
    """Run the signal book on the clean path and on the same path with alternating
    ±``size`` gaps every ``every`` bars. Returns both so the gap-induced drawdown
    is explicit. Extra kwargs pass straight through to run_signal_backtest (dte,
    warmup, always_sell, ...).

    A jump only hurts a short-gamma book if the book is HOLDING gamma when it
    lands. The default ``every=42`` against the default ``dte=21`` puts all of
    them on entry/expiry boundaries, where remaining time is zero and the stress
    silently measures nothing — so when the schedule resonates with the holding
    period the gaps are nudged half a cycle into the trade instead.
    """
    from engine.signal_backtest import run_signal_backtest

    warmup = bt_kwargs.get("warmup", 63)
    dte = bt_kwargs.get("dte", 21)
    offset = (dte // 2) if (dte and every % dte == 0) else 0
    gaps = evenly_spaced_gaps(len(prices), every=every, size=size,
                              start=warmup, offset=offset)
    gapped_prices = inject_gaps(prices, gaps)
    clean = run_signal_backtest(prices, chain_at, forecaster, **bt_kwargs)
    gapped = run_signal_backtest(gapped_prices, chain_at, forecaster, **bt_kwargs)
    return GapStressResult(clean, gapped, len(gaps), size)
