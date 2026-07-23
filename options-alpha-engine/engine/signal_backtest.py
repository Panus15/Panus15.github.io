"""Multi-date signal-driven walk-forward — does the P-vs-Q signal add value?

engine/hedged_backtest.py sells vol every eligible period. This harness instead
drives ENTRY from the actual distributional signal computed on a CHAIN SNAPSHOT
at each date: sell only when the market's risk-neutral vol (read off that day's
chain) is richer than the physical forecast AND the regime gate is clear. It
runs the signal-driven book alongside an always-sell baseline so the signal's
contribution is measurable — the honest question "is the edge real?" answered as
an out-of-sample equity curve once you feed it real chain snapshots.

You supply `chain_at(t, trailing_prices) -> OptionChain | None` — a live/replayed
snapshot for date t (walk-forward: it may only see prices up to t). Everything is
pure stdlib; the realisation reuses the delta-hedged mechanics of hedged_backtest.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from engine import backtest, pricing
from engine.hedged_backtest import MULT, _straddle
from engine.iv import implied_vol


def _atm_iv(chain, dte: int, T: float):
    """Implied vol of the ATM option on this snapshot (the vol we sell at)."""
    best, atm, kind = 1e18, None, None
    for qt in chain.quotes:
        if qt.expiry_days == dte and abs(qt.strike - chain.spot) < best:
            best, atm, kind = abs(qt.strike - chain.spot), qt, qt.kind
    if atm is None:
        return None, None
    iv = implied_vol(atm.mid, chain.spot, atm.strike, T, chain.r, chain.q, kind)
    return iv, atm.strike


def _realize_short_straddle(prices, t, dte, K, iv, r, q, hedge_bps, spread_frac):
    """Daily-hedged short-straddle P&L over [t, t+dte]; returns per-day list."""
    T0 = dte / 252.0
    v_prev, d_prev = _straddle(prices[t], K, T0, r, q, iv)
    hedge_prev = MULT * d_prev
    entry_cost = MULT * 0.5 * (spread_frac * v_prev + 0.04)
    day_pnl = {t: -entry_cost}
    for s in range(t + 1, t + dte + 1):
        rem = (t + dte - s) / 252.0
        S, S_prev = prices[s], prices[s - 1]
        v_now, d_now = _straddle(S, K, rem, r, q, iv)
        option_mtm = MULT * (v_prev - v_now)          # short: gain if value falls
        shares_pnl = hedge_prev * (S - S_prev)
        hedge_now = MULT * d_now
        rebal = abs(hedge_now - hedge_prev) * S * hedge_bps
        day_pnl[s] = option_mtm + shares_pnl - rebal
        v_prev, hedge_prev = v_now, hedge_now
    return day_pnl


@dataclass
class SignalBacktestResult:
    metrics: backtest.BacktestResult
    n_periods: int
    n_sold: int
    skip_reasons: dict = field(default_factory=dict)

    def summary(self) -> str:
        skips = "; ".join(f"{k}:{v}" for k, v in self.skip_reasons.items()) or "none"
        return (f"periods={self.n_periods} sold={self.n_sold} (skips: {skips})\n"
                + self.metrics.summary())


def run_signal_backtest(
    prices, chain_at, forecaster, *,
    dte: int = 21, warmup: int = 63, r: float = 0.03, q: float = 0.0,
    hedge_bps: float = 5e-4, spread_frac: float = 0.015,
    starting_equity: float = 100_000.0, min_vrp: float = 0.0,
    always_sell: bool = False,
) -> SignalBacktestResult:
    """Walk-forward, non-overlapping. Sells vol only when the day's chain signal
    is RICH and the regime is clear (unless ``always_sell`` — the baseline)."""
    from models import edge, rnd

    n = len(prices)
    pnl = [0.0] * n
    n_sold = 0
    n_periods = 0
    skips: dict[str, int] = {}

    def _skip(key):
        skips[key] = skips.get(key, 0) + 1

    t = warmup
    while t + dte < n:
        n_periods += 1
        trailing = prices[:t + 1]
        chain = chain_at(t, trailing)
        if chain is None:
            _skip("no_chain")
            t += dte
            continue
        T = dte / 365.0
        try:
            q_vol = rnd.model_free_implied_vol(chain, T, dte)
        except ValueError:
            _skip("bad_chain")
            t += dte
            continue
        iv, K = _atm_iv(chain, dte, T)
        if iv is None or K is None:
            _skip("no_atm")
            t += dte
            continue

        if not always_sell:
            p = forecaster.forecast(trailing, T, r=r, q=q, spot=prices[t])
            vrp = q_vol ** 2 - p.log_return_vol(prices[t], T) ** 2
            if edge.regime_stressed(trailing):
                _skip("regime")
                t += dte
                continue
            if vrp <= min_vrp:
                _skip("not_rich")
                t += dte
                continue

        for s, pv in _realize_short_straddle(
                prices, t, dte, K, iv, r, q, hedge_bps, spread_frac).items():
            pnl[s] += pv
        n_sold += 1
        t += dte

    metrics = backtest.run_backtest(pnl[warmup:], starting_equity=starting_equity)
    return SignalBacktestResult(metrics, n_periods, n_sold, skips)


# --------------------------------------------------------------------------- #
# Synthetic chain series — a chain_at() for offline demo/tests
# --------------------------------------------------------------------------- #
def synthetic_chain_series(*, dte: int, r: float = 0.03, q: float = 0.0,
                           base_premium: float = 0.12, smile: float = 0.9):
    """Return a chain_at(t, trailing) that builds a market chain each date whose
    ATM IV = trailing realised vol * (1 + time-varying premium) + a smile. The
    premium wobbles (some dates rich, some cheap) so the signal has something to
    discriminate; realised-vol spikes make the chain IV spike in stress too."""
    from engine.data import OptionChain, OptionQuote
    from engine import volforecast

    def chain_at(t, trailing):
        if len(trailing) < 22:
            return None
        spot = trailing[-1]
        rv = volforecast.close_to_close_vol(trailing, 21)
        premium = base_premium + 0.10 * math.sin(t / 9.0)   # rich/cheap cycle
        atm_vol = max(rv * (1.0 + premium), 0.03)
        T = dte / 365.0
        quotes = []
        for mny in (-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15):
            K = round(spot * (1 + mny), 2)
            vol = atm_vol + smile * mny * mny
            for kind in ("call", "put"):
                px = pricing.price(spot, K, T, r, q, vol, kind)
                if px < 0.02:
                    continue
                half = max(0.02, 0.02 * px)
                quotes.append(OptionQuote(dte, K, kind, round(max(px - half, 0.01), 2),
                                          round(px + half, 2)))
        return OptionChain("SYN", spot, r, q, quotes)

    return chain_at
