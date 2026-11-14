"""Many positions open at once — the case the risk governor was written for.

Every other harness in this repo holds ONE position at a time. That is deliberate
(non-overlapping trades keep the Sharpe honest) and it has an unnoticed cost: the
risk governor is handed an empty book on every single entry, so the
correlation-aware aggregation in ``portfolio.Portfolio.net_short_vega`` — the
reason that module exists — has never once run inside a backtest. The cap was
tested in isolation and never in the loop.

This harness runs several underlyings with STAGGERED entries, so at any moment a
real book of overlapping short-vol positions is open and the governor sees it. It
exists to answer one question the single-position harnesses structurally cannot:

    does the correlation-aware cap actually bind, and does it change the outcome
    versus sizing each trade as if it were the only one?

WHY THE ANSWER IS NOT OBVIOUS. Short vol on five names is not five independent
bets — in a crash every short-vol position loses together, so the book's risk
scales closer to N than to sqrt(N). A per-trade view that passes each position
individually will happily assemble a book that no single position would justify.
That is the classic way a short-vol desk discovers its risk model was per-trade.

The comparison run (``govern=False``) sizes every trade exactly as the existing
harnesses do — against an empty book — so the difference between the two runs IS
the value of the aggregation, in dollars, on the same paths.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from engine import backtest, portfolio, pricing, volforecast
from engine.hedged_backtest import MULT, _straddle


@dataclass
class BookBacktestResult:
    metrics: backtest.BacktestResult
    n_trades: int
    n_blocked_by_vega_cap: int = 0
    max_open: int = 0
    peak_net_short_vega: float = 0.0
    peak_naive_vega: float = 0.0       # what a per-trade view would have reported
    skip_reasons: dict = field(default_factory=dict)
    trade_pnl: list = field(default_factory=list)
    entries: list = field(default_factory=list)   # [{'t':bar,'symbol':sym,'size':n}]

    @property
    def understatement(self) -> float:
        """How much a per-trade view understates the book's real short vega."""
        if self.peak_naive_vega <= 0:
            return 0.0
        return 1.0 - self.peak_naive_vega / max(self.peak_net_short_vega, 1e-9)

    def summary(self) -> str:
        skips = "; ".join(f"{k}:{v}" for k, v in self.skip_reasons.items()) or "none"
        return (f"Multi-position book: {self.n_trades} trades, up to "
                f"{self.max_open} open at once (skips: {skips})\n"
                f"  peak correlation-aware net short vega {self.peak_net_short_vega:>12,.0f}\n"
                f"  peak per-trade (naive) view           {self.peak_naive_vega:>12,.0f}"
                f"   -> a per-trade view understates by {self.understatement:.0%}\n"
                f"  vega cap blocked {self.n_blocked_by_vega_cap} entries\n"
                + self.metrics.summary())


def run_book_backtest(
    prices_by_symbol: dict, *, dte: int = 21, warmup: int = 63,
    premium: float = 0.15, r: float = 0.03, q: float = 0.0,
    hedge_bps: float = 5e-4, spread_frac: float = 0.015,
    cvar_limit: float = 0.03, starting_equity: float = 100_000.0,
    limits: portfolio.RiskLimits | None = None, contract_mult: int = MULT,
    rho: float = 0.8, govern: bool = True, stagger: bool = True,
) -> BookBacktestResult:
    """Short a delta-hedged straddle on each symbol, entries staggered.

    ``rho`` is the assumed correlation between the positions' vega. 0.8 is a
    deliberate choice for equity index / sector short vol: these do not diversify
    each other in the regime that matters. ``govern=False`` reproduces the
    single-position behaviour (each trade sized against an empty book) so the two
    can be compared on identical paths.
    """
    from models.edge import regime_stressed

    syms = sorted(prices_by_symbol)
    if not syms:
        raise ValueError("no symbols: run_book_backtest needs at least one "
                         "price series, and its whole point is several at once")
    n = min(len(p) for p in prices_by_symbol.values())
    limits = limits or portfolio.RiskLimits(max_net_short_vega=8_000.0,
                                            max_drawdown=0.25)
    gov = portfolio.RiskGovernor(limits)

    pnl = [0.0] * n
    equity, peak = starting_equity, starting_equity
    skips: dict[str, int] = {}
    trades: list[float] = []
    book = portfolio.Portfolio([], rho=rho)
    open_until: dict = {}                      # symbol -> (expiry_bar, position)
    entries: list = []
    n_trades = n_blocked = max_open = 0
    peak_vega = peak_naive = 0.0

    def _skip(k):
        skips[k] = skips.get(k, 0) + 1

    # Stagger so the book is genuinely overlapping rather than N synchronised
    # trades that all open and close together (which would aggregate to the same
    # thing as one big position and prove nothing).
    offset = {s: (i * max(1, dte // len(syms)) if stagger else 0)
              for i, s in enumerate(syms)}

    for t in range(warmup, n - 1):
        # retire anything that expired at this bar
        for s in [s for s, (exp, _) in open_until.items() if exp <= t]:
            _, pos = open_until.pop(s)
            book.positions.remove(pos)

        drawdown = max(0.0, 1.0 - equity / peak)
        for s in syms:
            if s in open_until or (t - warmup - offset[s]) % dte != 0:
                continue
            if t + dte >= n:
                continue
            px = prices_by_symbol[s]
            trailing = px[:t + 1]
            spot = px[t]
            fvol = volforecast.har_rv_forecast(trailing)
            iv = max(fvol * (1.0 + premium), 0.05)
            K = round(spot, 0)
            if regime_stressed(trailing):
                _skip("regime")
                continue

            template = portfolio.Position(
                underlying=s, strike=K, expiry_days=dte, kind="put", quantity=-1,
                entry_price=pricing.price(spot, K, dte / 252, r, q, iv, "put"),
                spot=spot, r=r, q=q, vol=iv, multiplier=contract_mult,
                days_per_year=252.0)
            size = portfolio.size_by_cvar(
                equity, template,
                shock_scenarios=portfolio.vol_spike_scenarios(spot_shock=-0.12,
                                                              vol_bump=0.15),
                cvar_limit=cvar_limit)
            if size <= 0:
                _skip("size_zero")
                continue

            leg = portfolio.Position(
                underlying=s, strike=K, expiry_days=dte, kind="put", quantity=-size,
                entry_price=template.entry_price, spot=spot, r=r, q=q, vol=iv,
                multiplier=contract_mult, days_per_year=252.0)

            # THE POINT OF THIS MODULE: the governor sees the live book, not an
            # empty one. With govern=False it sees an empty book, exactly as every
            # other harness in the repo does.
            against = book if govern else portfolio.Portfolio([], rho=rho)
            ok, reason = gov.can_add(leg, against, equity, drawdown)
            if not ok:
                _skip("kill_switch" if "KILL" in reason else "vega_cap")
                n_blocked += "KILL" not in reason
                continue

            book.add(leg)
            open_until[s] = (t + dte, leg)
            n_trades += 1
            entries.append({"t": t, "symbol": s, "size": size})

            # realise it day by day
            v_prev, d_prev = _straddle(spot, K, dte / 252, r, q, iv)
            hedge_prev = size * contract_mult * d_prev
            entry_cost = size * contract_mult * 0.5 * (spread_frac * v_prev + 0.04)
            pnl[t] -= entry_cost
            total = -entry_cost
            for u in range(t + 1, min(t + dte + 1, n)):
                rem = (t + dte - u) / 252.0
                S, S_prev = px[u], px[u - 1]
                v_now, d_now = _straddle(S, K, rem, r, q, iv)
                hedge_now = size * contract_mult * d_now
                day = (size * contract_mult * (v_prev - v_now)
                       + hedge_prev * (S - S_prev)
                       - abs(hedge_now - hedge_prev) * S * hedge_bps)
                pnl[u] += day
                total += day
                v_prev, hedge_prev = v_now, hedge_now
            trades.append(total)
            equity += total
            peak = max(peak, equity)

        if open_until:
            max_open = max(max_open, len(open_until))
            peak_vega = max(peak_vega, book.net_short_vega())
            naive = max((abs(portfolio.Portfolio([p], rho=rho).net_short_vega())
                         for _, p in open_until.values()), default=0.0)
            peak_naive = max(peak_naive, naive)

    return BookBacktestResult(
        metrics=backtest.run_backtest(pnl[warmup:], starting_equity=starting_equity),
        n_trades=n_trades, n_blocked_by_vega_cap=n_blocked, max_open=max_open,
        peak_net_short_vega=peak_vega, peak_naive_vega=peak_naive,
        skip_reasons=skips, trade_pnl=trades, entries=entries,
    )
