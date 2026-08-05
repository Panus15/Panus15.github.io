"""How much of that Sharpe is the strategy, and how much is the sample?

Every backtest in this repo printed a single number off a single path. On the
repo's own fixture, holding the strategy and every parameter fixed and changing
ONLY the random seed, the delta-hedged book's Sharpe runs:

    seed  1     2     3     4     5     6     7     8
    Sharpe -0.02 +0.42 +0.66 +1.27 +1.84 +1.85 +1.87 +3.91

Those are the same strategy. A reader shown "Sharpe 1.85" would conclude
something a reader shown "Sharpe -0.02" would not, and neither number deserves
that. The spread is not noise around a true value to be averaged away either —
it is the honest width of what this evidence can support, and reporting a point
estimate without it is the single easiest way to fool yourself in this whole
codebase.

Two instruments, because they answer different questions:

  SEED ENSEMBLE       re-runs the strategy on independently generated paths.
                      Answers "would this have worked in a different world?" —
                      the right question for a synthetic fixture, and the only
                      one available when you have no real history.

  BLOCK BOOTSTRAP     resamples the REALISED trade P&L in contiguous blocks.
                      Answers "how much of this equity curve is a handful of
                      trades?" — the right question on real data, where you have
                      one path and cannot rerun the world. Blocks (rather than
                      single trades) preserve the short-run autocorrelation that
                      makes a losing streak a losing streak.

Neither turns a bad backtest into a good one. What they do is stop a good-looking
one from being read as more than it is.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


@dataclass
class Interval:
    """A point estimate with the range the evidence actually supports."""
    point: float
    lo: float
    hi: float
    n: int
    label: str = ""

    @property
    def spans_zero(self) -> bool:
        return self.lo <= 0.0 <= self.hi

    def line(self, pct: bool = False) -> str:
        """One row. ``pct=True`` for intervals whose units are returns, not cash.

        A return of 0.0049 printed at two decimals reads as +0.00 — the interval
        would look empty and the reader would conclude the opposite of the truth.
        """
        flag = "   <- spans zero" if self.spans_zero else ""
        if pct:
            return (f"  {self.label:22}{self.point:>+9.2%}   "
                    f"[{self.lo:>+7.2%}, {self.hi:>+7.2%}]  n={self.n}{flag}")
        return (f"  {self.label:22}{self.point:>+9.2f}   "
                f"[{self.lo:>+7.2f}, {self.hi:>+7.2f}]  n={self.n}{flag}")


def _quantile(sorted_vals, p: float) -> float:
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    i = p * (len(sorted_vals) - 1)
    lo = int(math.floor(i))
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = i - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def interval_from(samples, *, point=None, level: float = 0.90,
                  label: str = "") -> Interval:
    """Percentile interval over a list of resampled statistics."""
    vals = sorted(x for x in samples if x is not None and not math.isnan(x))
    if not vals:
        return Interval(float("nan"), float("nan"), float("nan"), 0, label)
    a = (1.0 - level) / 2.0
    mid = point if point is not None else _quantile(vals, 0.5)
    return Interval(mid, _quantile(vals, a), _quantile(vals, 1.0 - a),
                    len(vals), label)


# --------------------------------------------------------------------------- #
# 1. Seed ensemble — rerun the world
# --------------------------------------------------------------------------- #
@dataclass
class EnsembleResult:
    sharpe: Interval
    total_return: Interval
    max_drawdown: Interval
    seeds: list = field(default_factory=list)
    per_seed: list = field(default_factory=list)

    def summary(self) -> str:
        out = [f"Seed ensemble over {len(self.seeds)} independently generated paths",
               f"  {'':22}{'point':>9}   {'90% interval':^17}",
               self.sharpe.line(), self.total_return.line(),
               self.max_drawdown.line()]
        if self.sharpe.spans_zero:
            out.append("  -> the Sharpe interval SPANS ZERO: this evidence cannot "
                       "distinguish the strategy from no edge at all")
        return "\n".join(out)


def seed_ensemble(run, path_for_seed, seeds=range(1, 21), *,
                  level: float = 0.90) -> EnsembleResult:
    """Run ``run(prices) -> result-with-.metrics`` over ``path_for_seed(seed)``.

    Deliberately takes callables rather than a strategy enum: it should work on
    any harness in this repo (hedged, signal, spread) without knowing about them.
    """
    sh, tr, dd, used, rows = [], [], [], [], []
    for s in seeds:
        try:
            res = run(path_for_seed(s))
        except (ValueError, ZeroDivisionError, ArithmeticError):
            continue
        m = getattr(res, "metrics", res)
        sh.append(m.sharpe)
        tr.append(m.total_return)
        dd.append(m.max_drawdown)
        used.append(s)
        rows.append({"seed": s, "sharpe": m.sharpe, "total_return": m.total_return,
                     "max_drawdown": m.max_drawdown,
                     "trades": getattr(res, "n_trades", None)})
    return EnsembleResult(
        sharpe=interval_from(sh, point=(sum(sh) / len(sh)) if sh else None,
                             level=level, label="Sharpe"),
        total_return=interval_from(tr, point=(sum(tr) / len(tr)) if tr else None,
                                   level=level, label="total return"),
        max_drawdown=interval_from(dd, point=(sum(dd) / len(dd)) if dd else None,
                                   level=level, label="max drawdown"),
        seeds=used, per_seed=rows,
    )


# --------------------------------------------------------------------------- #
# 2. Block bootstrap — one path is all you get
# --------------------------------------------------------------------------- #
def block_bootstrap(trade_pnl, *, n_boot: int = 2000, block: int = 3,
                    seed: int = 0, level: float = 0.90,
                    starting_equity: float = 100_000.0) -> dict:
    """Resample realised trade P&L in contiguous blocks -> intervals for the stats.

    ``block`` > 1 keeps neighbouring trades together, so a run of losses can still
    appear as a run. Sampling trades independently would quietly break up exactly
    the clustering that produces a drawdown, and report a tighter interval than the
    data supports.

    Returns intervals for mean trade P&L, total return, and a per-trade Sharpe
    (mean/sd of trade P&L — NOT annualised; it is a shape statistic for comparing
    resamples, not a number to quote).
    """
    pnl = [float(x) for x in trade_pnl]
    n = len(pnl)
    if n < 2:
        raise ValueError("need at least 2 trades to bootstrap")
    blk = max(1, min(int(block), n))
    rng = random.Random(seed)
    means, totals, sharpes = [], [], []
    n_blocks = int(math.ceil(n / blk))
    for _ in range(int(n_boot)):
        sample = []
        for _ in range(n_blocks):
            start = rng.randrange(n)
            for j in range(blk):
                sample.append(pnl[(start + j) % n])
        sample = sample[:n]
        m = sum(sample) / n
        v = sum((x - m) ** 2 for x in sample) / max(n - 1, 1)
        sd = math.sqrt(v)
        means.append(m)
        totals.append(sum(sample) / starting_equity)
        sharpes.append(m / sd if sd > 1e-12 else 0.0)

    obs_m = sum(pnl) / n
    obs_v = sum((x - obs_m) ** 2 for x in pnl) / max(n - 1, 1)
    obs_sd = math.sqrt(obs_v)
    return {
        "n_trades": n,
        "block": blk,
        "mean_trade": interval_from(means, point=obs_m, level=level,
                                    label="mean trade P&L"),
        "total_return": interval_from(totals, point=sum(pnl) / starting_equity,
                                      level=level, label="total return"),
        "trade_sharpe": interval_from(
            sharpes, point=(obs_m / obs_sd if obs_sd > 1e-12 else 0.0),
            level=level, label="per-trade Sharpe"),
    }


def bootstrap_summary(boot: dict) -> str:
    out = [f"Block bootstrap: {boot['n_trades']} trades, "
           f"blocks of {boot['block']}, 90% percentile intervals",
           f"  {'':22}{'point':>9}   {'90% interval':^17}",
           boot["mean_trade"].line(), boot["total_return"].line(),
           boot["trade_sharpe"].line()]
    if boot["mean_trade"].spans_zero:
        out.append("  -> mean trade P&L spans zero: the profit is not distinguishable "
                   "from luck on this many trades")
    return "\n".join(out)
