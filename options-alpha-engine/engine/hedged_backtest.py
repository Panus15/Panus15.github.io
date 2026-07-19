"""Delta-hedged, walk-forward short-vol backtest — the honest proof engine.

Harvesting the variance risk premium does NOT mean holding a naked option; you
sell the straddle and DELTA-HEDGE it daily, so the P&L is a bet on implied vs
subsequently-realised variance:

    delta-hedged short-straddle P&L  ~  (IV_entry^2 - RV_realised^2) x dollar-gamma
                                        - transaction + hedging costs

This module simulates that day by day, with the disciplines the review demanded:

  * WALK-FORWARD   — every decision (vol forecast, sizing, regime gate) uses only
                     trailing data; no future price ever enters a forecast.
  * NON-OVERLAPPING — trades are back-to-back (open at t, held to expiry, next
                     opens at expiry), so daily P&L is not autocorrelated across
                     overlapping positions and the Sharpe is not inflated.
  * REAL COSTS     — option spread at entry + a daily delta-hedge/rebalance cost.
  * RISK-GOVERNED  — size via portfolio.size_by_cvar (fat left tail, not Kelly);
                     every trade must clear portfolio.RiskGovernor (short-vega cap
                     + drawdown kill-switch); a regime gate skips selling vol when
                     realised vol is accelerating.
  * CRASH-AWARE    — meant to be run over a path that INCLUDES a vol spike, because
                     a crash-free sample makes short-vol look like free money.

Honest limitation: implied vol is held constant within each trade, so the
gamma (realised-variance) loss in a crash IS captured but the vega loss from IV
itself spiking is not — this understates crash losses. A vol-of-vol path is the
next refinement. Everything here is pure stdlib.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from engine import backtest, pricing, portfolio, volforecast
from models.edge import regime_stressed

MULT = 100


def price_path_with_crash(
    n_days: int = 756,
    *,
    seed: int = 7,
    start: float = 100.0,
    calm_vol: float = 0.13,
    crash_at: float = 0.62,
    crash_drop: float = 0.22,
    crash_vol: float = 0.48,
    recovery_days: int = 45,
) -> list[float]:
    """Deterministic 3-year-ish daily path: calm regime, a crash, elevated vol.

    Exists so the backtest spans at least one stress regime. NOT a market model.
    """
    rng = random.Random(seed)
    prices = [start]
    crash_idx = int(n_days * crash_at)
    for i in range(1, n_days):
        if i == crash_idx:
            prices.append(prices[-1] * (1.0 - crash_drop))   # the gap-down
            continue
        if crash_idx < i <= crash_idx + recovery_days:
            vol = crash_vol
            drift = 0.10 / 252            # sharp bounce-and-chop
        else:
            vol = calm_vol
            drift = 0.07 / 252
        dv = vol / math.sqrt(252)
        z = rng.gauss(0.0, 1.0)
        prices.append(prices[-1] * math.exp(drift - 0.5 * dv * dv + dv * z))
    return prices


def _straddle(S: float, K: float, T: float, r: float, q: float, iv: float):
    """(value, delta_sum) of a LONG ATM straddle (call+put) per single contract."""
    if T <= 0:
        val = max(S - K, 0.0) + max(K - S, 0.0)
        return val, 0.0
    c = pricing.greeks(S, K, T, r, q, iv, "call")
    p = pricing.greeks(S, K, T, r, q, iv, "put")
    return c.price + p.price, c.delta + p.delta


@dataclass
class HedgedBacktestResult:
    metrics: backtest.BacktestResult
    n_trades: int
    n_skipped: int
    skip_reasons: dict = field(default_factory=dict)
    trade_pnl: list = field(default_factory=list)

    def summary(self) -> str:
        skips = "; ".join(f"{k}:{v}" for k, v in self.skip_reasons.items()) or "none"
        return (f"trades={self.n_trades}  skipped={self.n_skipped} ({skips})\n"
                + self.metrics.summary())


def run_hedged_backtest(
    prices,
    *,
    dte: int = 21,
    warmup: int = 63,
    premium: float = 0.15,
    starting_equity: float = 100_000.0,
    r: float = 0.03,
    q: float = 0.0,
    hedge_bps: float = 5e-4,
    spread_frac: float = 0.015,
    cvar_limit: float = 0.03,
    limits: portfolio.RiskLimits | None = None,
) -> HedgedBacktestResult:
    """Walk-forward delta-hedged short-straddle backtest over ``prices``.

    ``dte`` is the holding period in TRADING days (non-overlapping). The market
    charges IV = forecast_vol * (1 + ``premium``); we harvest that premium when
    realised vol comes in below it, and pay when it doesn't.
    """
    limits = limits or portfolio.RiskLimits(
        max_net_short_vega=8_000.0, max_drawdown=0.25
    )
    gov = portfolio.RiskGovernor(limits)

    n = len(prices)
    pnl_by_day = [0.0] * n
    equity = starting_equity
    peak = starting_equity
    trade_pnl: list[float] = []
    skip_reasons: dict[str, int] = {}
    n_trades = 0

    i0 = warmup
    while i0 + dte < n:
        trailing = prices[:i0 + 1]
        spot = prices[i0]
        fvol = volforecast.har_rv_forecast(trailing)
        iv_entry = max(fvol * (1.0 + premium), 0.05)
        K = round(spot, 0)
        drawdown = max(0.0, 1.0 - equity / peak)

        # --- gates: regime + governor + CVaR sizing ------------------------ #
        def _skip(reason_key: str):
            skip_reasons[reason_key] = skip_reasons.get(reason_key, 0) + 1

        if regime_stressed(trailing):
            _skip("regime")
            i0 += dte
            continue

        # Size on the crash-exposed short put via CVaR; governor caps the rest.
        template = portfolio.Position(
            underlying="U", strike=K, expiry_days=dte, kind="put",
            quantity=-1, entry_price=pricing.price(spot, K, dte / 252, r, q, iv_entry, "put"),
            spot=spot, r=r, q=q, vol=iv_entry,
        )
        size = portfolio.size_by_cvar(
            equity, template,
            shock_scenarios=portfolio.vol_spike_scenarios(spot_shock=-0.12, vol_bump=0.15),
            cvar_limit=cvar_limit,
        )
        if size <= 0:
            _skip("size_zero")
            i0 += dte
            continue

        short_straddle_leg = portfolio.Position(
            underlying="U", strike=K, expiry_days=dte, kind="put",
            quantity=-size, entry_price=template.entry_price,
            spot=spot, r=r, q=q, vol=iv_entry,
        )
        ok, reason = gov.can_add(short_straddle_leg, portfolio.Portfolio(), equity, drawdown)
        if not ok:
            _skip("kill_switch" if "KILL" in reason else "vega_cap")
            i0 += dte
            continue

        # --- simulate the delta-hedged short straddle day by day ----------- #
        v_prev, d_prev = _straddle(spot, K, dte / 252, r, q, iv_entry)
        hedge_prev = size * MULT * d_prev            # shares to hold (neutralise)
        # entry cost: cross half the spread on both legs.
        entry_spread = spread_frac * v_prev + 0.04
        trade_cost = size * MULT * 0.5 * entry_spread
        pnl_by_day[i0] -= trade_cost
        trade_total = -trade_cost

        for s in range(i0 + 1, i0 + dte + 1):
            rem = (i0 + dte - s) / 252.0
            S, S_prev = prices[s], prices[s - 1]
            v_now, d_now = _straddle(S, K, rem, r, q, iv_entry)
            option_mtm = size * MULT * (v_prev - v_now)      # short: gain if value falls
            shares_pnl = hedge_prev * (S - S_prev)
            hedge_now = size * MULT * d_now
            rebal_cost = abs(hedge_now - hedge_prev) * S * hedge_bps
            day_pnl = option_mtm + shares_pnl - rebal_cost
            pnl_by_day[s] += day_pnl
            trade_total += day_pnl
            v_prev, hedge_prev = v_now, hedge_now

        equity += trade_total
        peak = max(peak, equity)
        trade_pnl.append(trade_total)
        n_trades += 1
        i0 += dte

    active = pnl_by_day[warmup:]
    metrics = backtest.run_backtest(active, starting_equity=starting_equity)
    return HedgedBacktestResult(
        metrics=metrics,
        n_trades=n_trades,
        n_skipped=sum(skip_reasons.values()),
        skip_reasons=skip_reasons,
        trade_pnl=trade_pnl,
    )
