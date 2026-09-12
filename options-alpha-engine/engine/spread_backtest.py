"""Do defined-risk spreads actually beat selling naked vol? — the missing harness.

models/spreads.py builds iron condors and put credit spreads and scores each one
`EV = credit - E_P[loss] - cost` against the physical density. ARCHITECTURE.md
calls that "the structural answer to the tail." It was, until now, an expected
value and nothing else: no backtest in this repo could reach it, so the structure
sold as the answer to the crash problem had never produced a single simulated
dollar. This harness closes that.

WHAT IS DIFFERENT ABOUT REALISING A SPREAD

  It is HELD TO EXPIRY AND NEVER HEDGED, and that is the point rather than a
  simplification. A naked short straddle needs daily delta-hedging because its
  loss is unbounded; the long wings of a condor cap the loss by construction, so
  the position that has to be nursed through a crash is replaced by one whose
  worst case is known when it is opened. Realisation is therefore the terminal
  payoff — no hedge slippage, no gamma path-dependence, no rebalancing cost.

  The comparison is what makes the number mean anything. On EVERY date this sells
  the spread AND the delta-hedged naked straddle the rest of the repo trades, so
  the two books are exposed to identical paths. Giving up premium for a capped
  tail must pay for itself, and paired against the naked book is the only way to
  see whether it does.

WHAT IT ACTUALLY MEASURED, which is not what this module was written expecting.
Run on the repo's own crash path, the condor loses to the delta-hedged straddle
on BOTH counts — -0.9% vs +4.5% return, and a worse worst trade (-559 vs -87).
Inject +/-18% gaps into the same path, so the hedge can no longer be maintained,
and it inverts: the naked book's worst trade blows out to -2,901 while the
spread's stays capped at -694, and the spread now loses less overall too.

    path              spread tot   naked tot   spread worst   naked worst
    clean                  -0.9%       +4.5%           -559           -87
    with +/-18% gaps       -2.7%       -3.9%           -694        -2,901

That is the honest statement of what these structures are for, and it is sharper
than "spreads cap the tail". The comparison book is DELTA-HEDGED, and a working
hedge already removes most of the directional damage the wings are sold to cap —
so against a book that can hedge continuously, the wings are a pure cost. They
earn their price exactly where the hedge fails: gaps, weekends, illiquidity, an
account too small to rebalance. Buy them for the risk you cannot hedge, not for
the risk you can.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from engine import backtest


def spread_payoff(spread, terminal_spot: float) -> float:
    """What the structure costs to settle at expiry, PER SHARE.

    Positive = the short legs finished in the money and you owe. Signed by side,
    so the long wings subtract — which is exactly the mechanism that caps the loss.
    """
    total = 0.0
    for leg in spread.legs:
        intrinsic = (max(terminal_spot - leg.strike, 0.0) if leg.kind == "call"
                     else max(leg.strike - terminal_spot, 0.0))
        total += intrinsic if leg.side == "short" else -intrinsic
    return total


def realize_spread(spread, terminal_spot: float, *, commission: float = 0.65) -> float:
    """Realised P&L in DOLLARS for one spread held to expiry.

    credit taken in, payoff paid out, commission on every leg both ways. The
    result is bounded below by -(contract_mult * width + commission * n_legs)
    however far the underlying runs — that bound IS the product.
    """
    mult = spread.contract_mult
    return mult * (spread.net_credit - spread_payoff(spread, terminal_spot)) \
        - commission * len(spread.legs)


def mark_spread_daily(spread, prices, t, dte, *, r, q, commission=0.65):
    """Daily mark-to-market P&L for one spread, as a {bar: pnl} dict.

    Why this exists rather than booking the whole result on expiry day: the naked
    straddle this harness compares against is marked every day, and Sharpe on a
    series that is zero for twenty days and lumpy on the twenty-first is
    structurally lower than Sharpe on a smooth one. Comparing the two directly
    would penalise spreads for an accounting choice rather than for their economics.

    Each leg's implied vol is recovered from its OWN executable entry price and
    then frozen, exactly as the naked harness freezes its entry IV. That makes the
    day-one liability equal the entry credit by construction, so the daily series
    telescopes to precisely ``realize_spread`` — the total is unchanged, only its
    distribution over time.
    """
    from engine import pricing
    from engine.iv import implied_vol

    legs = []
    for leg in spread.legs:
        v = implied_vol(leg.price, prices[t], leg.strike, dte / 252.0, r, q, leg.kind)
        legs.append((leg, v if (v and v > 0) else None))
    if any(v is None for _, v in legs):
        return None                       # cannot mark it honestly -> do not pretend

    def liability(S, rem):
        total = 0.0
        for leg, v in legs:
            px = (max(S - leg.strike, 0.0) if leg.kind == "call"
                  else max(leg.strike - S, 0.0)) if rem <= 0 else \
                pricing.price(S, leg.strike, rem, r, q, v, leg.kind)
            total += px if leg.side == "short" else -px
        return total

    mult = spread.contract_mult
    prev = liability(prices[t], dte / 252.0)
    day = {t: -commission * len(spread.legs)}
    for s in range(t + 1, min(t + dte + 1, len(prices))):
        rem = (t + dte - s) / 252.0
        now = liability(prices[s], rem)
        day[s] = mult * (prev - now)
        prev = now
    return day


@dataclass
class SpreadBacktestResult:
    metrics: backtest.BacktestResult
    n_periods: int
    n_sold: int
    worst_trade: float = 0.0
    best_trade: float = 0.0
    max_loss_budgeted: float = 0.0     # the worst loss the structures ALLOWED
    naked: backtest.BacktestResult | None = None
    naked_worst: float = 0.0
    n_capped: int = 0                  # trades where the long wing actually paid
    n_naked: int = 0                   # comparison straddles actually sold
    skip_reasons: dict = field(default_factory=dict)
    trades: list = field(default_factory=list)

    def summary(self) -> str:
        skips = "; ".join(f"{k}:{v}" for k, v in self.skip_reasons.items()) or "none"
        out = [f"Defined-risk spreads, held to expiry (unhedged by design)",
               f"  periods={self.n_periods} sold={self.n_sold} (skips: {skips})",
               f"  worst trade {self.worst_trade:>+12,.2f}   "
               f"best {self.best_trade:>+12,.2f}   "
               f"wing paid on {self.n_capped}/{self.n_sold} trades",
               f"  worst loss the structure ALLOWED: "
               f"{-abs(self.max_loss_budgeted):>+12,.2f} (the cap, known at entry)",
               self.metrics.summary()]
        if self.naked is not None:
            out.append(f"  vs the delta-hedged NAKED straddle on the same dates:")
            out.append(f"    {'':10}{'Sharpe':>8} {'MaxDD':>8} {'total':>9} "
                       f"{'worst trade':>14}")
            out.append(f"    spread   {self.metrics.sharpe:>8.2f} "
                       f"{self.metrics.max_drawdown:>8.1%} "
                       f"{self.metrics.total_return:>+9.1%} {self.worst_trade:>+14,.2f}")
            out.append(f"    naked    {self.naked.sharpe:>8.2f} "
                       f"{self.naked.max_drawdown:>8.1%} "
                       f"{self.naked.total_return:>+9.1%} {self.naked_worst:>+14,.2f}")
            out.append("  -> " + self.verdict())
        return "\n".join(out)

    def verdict(self) -> str:
        if self.n_sold == 0 or self.naked is None:
            return "no comparison available"
        ret_given_up = self.naked.total_return - self.metrics.total_return
        # If the naked book never had a losing trade, there was no tail to cap and
        # the difference of the two worsts is not a saving — saying otherwise would
        # dress up a calm sample as evidence the wings worked.
        if self.naked_worst >= 0:
            return (f"NOTHING STRESSED THE WINGS — the naked book's worst trade was "
                    f"itself a profit ({self.naked_worst:+,.0f}) over {self.n_naked} "
                    f"trades, so the {ret_given_up:.1%} of return the spreads gave up "
                    f"bought insurance this sample never claimed on. Re-run it over a "
                    f"crash before drawing any conclusion.")
        # Both worsts are losses, i.e. negative. The spread did BETTER when its
        # worst is the less-negative one, so the saving is worst - naked_worst.
        tail_saved = self.worst_trade - self.naked_worst
        if tail_saved <= 0:
            return (f"the wings cost {ret_given_up:.1%} of return AND still took the "
                    f"bigger loss ({self.worst_trade:+,.0f} vs {self.naked_worst:+,.0f}) "
                    f"— on this sample the structure is simply worse")
        if ret_given_up <= 0:
            return ("the spread book gave up NOTHING and still capped the tail — "
                    "too good to accept without checking the sample and the costs")
        return (f"bought a {tail_saved:,.0f} smaller worst trade for "
                f"{ret_given_up:.1%} of return — the trade the wings exist to make")


def run_spread_backtest(
    prices, chain_at, forecaster, *,
    dte: int = 21, warmup: int = 63, r: float = 0.03, q: float = 0.0,
    structure: str = "iron_condor", body: float = 0.05, wing: float = 0.05,
    commission: float = 0.65, contract_mult: float = 100.0,
    min_ev: float = 0.0, starting_equity: float = 100_000.0,
    always_sell: bool = False, event_at=None, compare_naked: bool = True,
    hedge_bps: float = 5e-4, spread_frac: float = 0.015,
) -> SpreadBacktestResult:
    """Walk-forward, non-overlapping. Sells one defined-risk structure per period.

    ``structure`` is 'iron_condor' or 'put_credit_spread'. Entry requires the model
    EV to clear ``min_ev`` unless ``always_sell``; the EV is computed from the
    physical density on trailing prices only, so the gate is walk-forward.
    """
    from models import spreads

    build = {"iron_condor": spreads.iron_condor,
             "put_credit_spread": spreads.put_credit_spread}[structure]

    n = len(prices)
    pnl = [0.0] * n
    naked_pnl = [0.0] * n
    skips: dict[str, int] = {}
    trades: list = []
    n_periods = n_sold = n_capped = n_naked = 0
    worst = best = None
    budgeted = 0.0
    naked_worst = None

    def _skip(k):
        skips[k] = skips.get(k, 0) + 1

    t = warmup
    while t + dte < n:
        n_periods += 1
        trailing = prices[:t + 1]
        chain = chain_at(t, trailing)
        if chain is None:
            _skip("no_chain")
            t += dte
            continue
        if event_at is not None and event_at(t, trailing):
            _skip("event")
            t += dte
            continue
        try:
            sp = build(chain, forecaster, trailing, dte=dte, body=body, wing=wing,
                       r=r, q=q, commission=commission, contract_mult=contract_mult)
        except (ValueError, ZeroDivisionError, ArithmeticError):
            sp = None
        if sp is None:
            _skip("no_structure")          # the four strikes are not all quoted
            t += dte
            continue
        if sp.max_loss <= 0:
            _skip("degenerate")            # zero-width wings: not a defined-risk trade
            t += dte
            continue
        if not always_sell and sp.ev <= min_ev:
            _skip("negative_ev")
            t += dte
            continue

        terminal = prices[t + dte]
        realised = realize_spread(sp, terminal, commission=commission)
        marked = mark_spread_daily(sp, prices, t, dte, r=r, q=q,
                                   commission=commission)
        if marked is None:
            _skip("unmarkable")           # a leg's IV could not be recovered
            t += dte
            continue
        for s, pv in marked.items():
            pnl[s] += pv
        n_sold += 1
        worst = realised if worst is None else min(worst, realised)
        best = realised if best is None else max(best, realised)
        budgeted = max(budgeted, contract_mult * sp.max_loss + commission * len(sp.legs))
        # did a long wing actually absorb something? (a short leg finished ITM
        # far enough that the wing beyond it also had value)
        if any(l.side == "long" and (
                (l.kind == "call" and terminal > l.strike) or
                (l.kind == "put" and terminal < l.strike)) for l in sp.legs):
            n_capped += 1
        trades.append({"t": t, "terminal": terminal, "credit": sp.net_credit,
                       "max_loss": sp.max_loss, "ev": sp.ev, "pnl": realised})

        if compare_naked:
            from engine.signal_backtest import _atm_iv, _realize_short_straddle
            iv, K = _atm_iv(chain, dte, dte / 365.0)
            if iv is not None and K is not None:
                day = _realize_short_straddle(prices, t, dte, K, iv, r, q,
                                              hedge_bps, spread_frac)
                for s, pv in day.items():
                    naked_pnl[s] += pv
                n_naked += 1
                tot = sum(day.values())
                naked_worst = tot if naked_worst is None else min(naked_worst, tot)
        t += dte

    metrics = backtest.run_backtest(pnl[warmup:], starting_equity=starting_equity)
    naked = (backtest.run_backtest(naked_pnl[warmup:], starting_equity=starting_equity)
             if compare_naked else None)
    return SpreadBacktestResult(
        metrics=metrics, n_periods=n_periods, n_sold=n_sold,
        worst_trade=worst if worst is not None else 0.0,
        best_trade=best if best is not None else 0.0,
        max_loss_budgeted=budgeted, naked=naked,
        naked_worst=naked_worst if naked_worst is not None else 0.0,
        n_capped=n_capped, n_naked=n_naked,
        skip_reasons=skips, trades=trades,
    )
