"""Does the rotation chart predict anything? — the test the pretty picture needs.

A Relative Rotation Graph is a transform of past relative price. That it looks
organised is not evidence: any smoothed two-axis view of eleven correlated series
produces tidy loops. The only question worth asking is whether a sector's quadrant
TODAY says anything about its excess return TOMORROW, and this measures it.

TWO READINGS, because they can disagree and the disagreement is informative:

  QUADRANT PANEL   every sector, every rebalance date, bucketed by the quadrant it
                   was in, scored on its forward excess return over the benchmark.
                   If the chart works, Leading is positive and Lagging negative.
                   This ignores costs and is the kindest possible test — a signal
                   that fails HERE cannot be rescued by better execution.

  LONG/SHORT BOOK  buy the top ``k`` of the leaderboard, sell the bottom ``k``,
                   equal weight, rebalanced every ``horizon`` bars, costs charged
                   both ways. Long and short are held over the SAME dates, so the
                   market return cancels and what is left is the rotation call.
                   This is the tradeable version and it is much less forgiving.

THREE THINGS KEEP IT HONEST. Everything is POINT-IN-TIME: the map at bar t is
built from ``prices[:t+1]`` and scored on bars after t, never the reverse. Trades
are NON-OVERLAPPING, so a good call is not counted once per day for a month.
And a PLACEBO runs alongside — the same book with the sector labels shuffled,
which must produce nothing. If the placebo also makes money, the harness is
measuring sector beta or a rebalancing artifact, not rotation, and the real result
must be thrown away rather than explained.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from engine import backtest
from models.rotation import CYCLE, leaderboard, quadrant, rotation_map


def _excess(px, bench, t, h) -> float | None:
    """Forward return of a symbol minus the benchmark's, over [t, t+h]."""
    if t + h >= len(px) or t + h >= len(bench):
        return None
    if px[t] <= 0 or bench[t] <= 0:
        return None
    return (px[t + h] / px[t]) - (bench[t + h] / bench[t])


@dataclass
class QuadrantPanel:
    stats: dict = field(default_factory=dict)   # quadrant -> {n, mean, sd, t}
    n_dates: int = 0
    min_obs: int = 30
    t_thresh: float = 2.0

    def spread_t(self) -> tuple | None:
        """(Leading − Lagging, its t). None when a quadrant is too thin to read.

        Exposed rather than left inside ``verdict``'s prose so the pre-registered
        criterion can be evaluated by a machine instead of by whoever is reading
        the sentence after seeing the number.
        """
        lead = self.stats.get("Leading", {})
        lag = self.stats.get("Lagging", {})
        if not lead or not lag or min(lead["n"], lag["n"]) < self.min_obs:
            return None
        spread = lead["mean"] - lag["mean"]
        se = math.sqrt(lead["sd"] ** 2 / lead["n"] + lag["sd"] ** 2 / lag["n"])
        return spread, (spread / se if se > 1e-12 else 0.0)

    def verdict(self) -> str:
        lead = self.stats.get("Leading", {})
        lag = self.stats.get("Lagging", {})
        if not lead or not lag:
            return "NOT ENOUGH DATA — a quadrant never filled"
        if min(lead["n"], lag["n"]) < self.min_obs:
            return (f"NOT ENOUGH DATA — Leading n={lead['n']}, Lagging n={lag['n']} "
                    f"(need >= {self.min_obs} each)")
        spread, t = self.spread_t()
        if abs(t) < self.t_thresh:
            return (f"NO SIGNIFICANT EFFECT — Leading beats Lagging by "
                    f"{spread:+.2%} per period, t={t:+.2f}. The quadrant a sector "
                    f"sits in does not tell you its next move on this sample.")
        if spread < 0:
            return (f"THE CHART IS INVERTED HERE — Lagging outperformed Leading by "
                    f"{-spread:.2%} (t={t:+.2f}); this sample rewarded mean "
                    f"reversion, not momentum.")
        return (f"LEADING BEATS LAGGING by {spread:.2%} per period (t={t:+.2f}) "
                f"— before costs, which the long/short book charges.")

    def summary(self) -> str:
        lines = [f"Quadrant panel over {self.n_dates} rebalance dates "
                 f"(forward excess return vs the benchmark, BEFORE costs)",
                 f"  {'quadrant':11}{'n':>6}{'mean':>10}{'sd':>10}{'t':>8}"]
        for q in CYCLE:
            s = self.stats.get(q)
            if not s:
                lines.append(f"  {q:11}{'-':>6}")
                continue
            lines.append(f"  {q:11}{s['n']:>6}{s['mean']:>+10.2%}{s['sd']:>10.2%}"
                         f"{s['t']:>+8.2f}")
        lines.append("  -> " + self.verdict())
        return "\n".join(lines)


def quadrant_panel(prices_by_symbol: dict, benchmark, *, window: int = 63,
                   mom_lag: int = 5, horizon: int = 21, warmup: int | None = None,
                   min_obs: int = 30, t_thresh: float = 2.0) -> QuadrantPanel:
    """Bucket every (sector, date) by quadrant and score its forward excess return."""
    warm = warmup if warmup is not None else window + mom_lag + window
    n = min([len(benchmark)] + [len(p) for p in prices_by_symbol.values()])
    buckets: dict = {q: [] for q in CYCLE}
    dates = 0
    t = warm
    while t + horizon < n:
        pts = rotation_map({s: p[:t + 1] for s, p in prices_by_symbol.items()},
                           benchmark[:t + 1], window=window, mom_lag=mom_lag)
        if pts:
            dates += 1
        for p in pts:
            fwd = _excess(prices_by_symbol[p.symbol], benchmark, t, horizon)
            if fwd is not None:
                buckets[p.quadrant].append(fwd)
        t += horizon
    stats = {}
    for q, vals in buckets.items():
        if not vals:
            continue
        m = sum(vals) / len(vals)
        var = sum((x - m) ** 2 for x in vals) / max(len(vals) - 1, 1)
        sd = math.sqrt(var)
        se = sd / math.sqrt(len(vals)) if len(vals) else 0.0
        stats[q] = {"n": len(vals), "mean": m, "sd": sd,
                    "t": (m / se if se > 1e-12 else 0.0)}
    return QuadrantPanel(stats=stats, n_dates=dates, min_obs=min_obs,
                         t_thresh=t_thresh)


@dataclass
class RotationBacktestResult:
    metrics: backtest.BacktestResult
    n_rebalances: int
    period_returns: list = field(default_factory=list)
    placebo: backtest.BacktestResult | None = None
    placebo_returns: list = field(default_factory=list)
    picks: list = field(default_factory=list)
    cost_per_rebalance: float = 0.0

    def verdict(self) -> str:
        if self.n_rebalances < 12:
            return (f"NOT ENOUGH DATA — {self.n_rebalances} rebalances. Nothing here "
                    f"is a result.")
        real = self.metrics.total_return
        if self.placebo is None:
            return f"total {real:+.1%} over {self.n_rebalances} rebalances (no placebo run)"
        fake = self.placebo.total_return
        if real <= 0:
            return (f"NO EDGE — the rotation book returned {real:+.1%} net of costs. "
                    f"The chart may describe the past well and still not pay.")
        if fake >= real * 0.5:
            return (f"REJECTED BY THE PLACEBO — shuffling the sector labels earned "
                    f"{fake:+.1%} against the real book's {real:+.1%}. Most of this "
                    f"is not the rotation call.")
        return (f"the rotation book returned {real:+.1%} net of costs vs the "
                f"label-shuffled placebo's {fake:+.1%} — worth a longer sample and "
                f"a live forward test, not a position")

    def summary(self) -> str:
        lines = [f"Rotation long/short: top-k vs bottom-k of the leaderboard, "
                 f"{self.n_rebalances} rebalances, costs charged",
                 self.metrics.summary()]
        if self.placebo is not None:
            lines.append(f"  PLACEBO (sector labels shuffled): "
                         f"total {self.placebo.total_return:+.2%}  "
                         f"Sharpe {self.placebo.sharpe:+.2f}")
        lines.append("  -> " + self.verdict())
        return "\n".join(lines)


def run_rotation_backtest(prices_by_symbol: dict, benchmark, *, window: int = 63,
                          mom_lag: int = 5, horizon: int = 21, k: int = 3,
                          warmup: int | None = None, cost_bps: float = 10.0,
                          starting_equity: float = 100_000.0,
                          placebo_seed: int | None = 0,
                          rank_by: str = "both") -> RotationBacktestResult:
    """Long the leaderboard's top ``k``, short its bottom ``k``, equal weight.

    ``cost_bps`` is charged on every leg every rebalance — 2k legs in and 2k out,
    so a 21-bar horizon with k=3 pays it 12 times a period. Sector ETFs are liquid
    and 10bp is generous; the point is that the number is charged at all, because a
    long/short book that rebalances monthly is a costly thing to run and a gross
    return hides that entirely.
    """
    warm = warmup if warmup is not None else window + mom_lag + window
    n = min([len(benchmark)] + [len(p) for p in prices_by_symbol.values()])
    rng = random.Random(placebo_seed if placebo_seed is not None else 0)

    rets: list[float] = []
    fake_rets: list[float] = []
    picks: list = []
    cost = 4.0 * k * cost_bps / 10_000.0 / (2.0 * k)   # per-unit-of-book, round trip

    t = warm
    while t + horizon < n:
        pts = rotation_map({s: p[:t + 1] for s, p in prices_by_symbol.items()},
                           benchmark[:t + 1], window=window, mom_lag=mom_lag)
        if len(pts) < 2 * k:
            t += horizon
            continue
        board = leaderboard(pts, rank_by=rank_by)
        longs = [p.symbol for p in board[:k]]
        shorts = [p.symbol for p in board[-k:]]

        def _book(long_syms, short_syms):
            lo = [_excess(prices_by_symbol[s], benchmark, t, horizon) for s in long_syms]
            sh = [_excess(prices_by_symbol[s], benchmark, t, horizon) for s in short_syms]
            if any(x is None for x in lo + sh):
                return None
            return (sum(lo) / len(lo) - sum(sh) / len(sh)) / 2.0 - cost

        real = _book(longs, shorts)
        if real is None:
            t += horizon
            continue
        rets.append(real)

        if placebo_seed is not None:
            syms = [p.symbol for p in pts]
            rng.shuffle(syms)
            fake = _book(syms[:k], syms[-k:])
            if fake is not None:
                fake_rets.append(fake)

        picks.append({"t": t, "long": longs, "short": shorts, "ret": real})
        t += horizon

    def _curve(seq):
        pnl = [x * starting_equity for x in seq]
        return backtest.run_backtest(pnl, starting_equity=starting_equity)

    return RotationBacktestResult(
        metrics=_curve(rets), n_rebalances=len(rets), period_returns=rets,
        placebo=_curve(fake_rets) if fake_rets else None,
        placebo_returns=fake_rets, picks=picks, cost_per_rebalance=cost,
    )


# --------------------------------------------------------------------------
# The pre-registered decision, evaluated mechanically
# --------------------------------------------------------------------------

# PREREGISTRATION.md §2.1: "does the quadrant panel report LEADING BEATS LAGGING
# with |t| >= 2, AND does the long/short book beat its label-shuffled placebo by
# more than 2x?" Both must hold.
PREREG_T_THRESH = 2.0
PREREG_PLACEBO_RATIO = 2.0


def preregistered_verdict(panel: QuadrantPanel,
                          book: RotationBacktestResult) -> dict:
    """Answer §2.1's primary question with arithmetic, not with prose.

    Written down as code because the failure mode here is not a wrong number, it
    is a reader who has already seen the number deciding what the criterion
    meant. "Beat the placebo by 2x" is exactly the phrase that becomes elastic
    when the book loses money and the placebo loses more.

    So: a book that lost money FAILS, whatever the placebo did. There is no
    reading of §2.1 on which "lost 27% while a coin-flip lost 18%" is a pass, and
    the ratio is only consulted once the book is above zero.
    """
    st = panel.spread_t()
    if st is None:
        panel_pass, why_panel = False, "the panel had too few observations to read"
    else:
        spread, t = st
        panel_pass = spread > 0 and abs(t) >= PREREG_T_THRESH
        why_panel = (f"Leading-Lagging {spread:+.2%}/period, t={t:+.2f} "
                     f"(needs a positive spread at |t| >= {PREREG_T_THRESH:g})")

    real = book.metrics.total_return
    fake = book.placebo.total_return if book.placebo is not None else None
    if book.n_rebalances < 12:
        book_pass, why_book = False, f"only {book.n_rebalances} rebalances"
    elif fake is None:
        book_pass, why_book = False, "no placebo was run, so nothing is comparable"
    elif real <= 0:
        book_pass = False
        why_book = (f"the book returned {real:+.1%} net of costs; a losing book "
                    f"does not pass on the placebo's {fake:+.1%} being worse")
    else:
        book_pass = fake < real / PREREG_PLACEBO_RATIO
        why_book = (f"book {real:+.1%} vs placebo {fake:+.1%} "
                    f"(needs the placebo below {real / PREREG_PLACEBO_RATIO:+.1%})")

    passed = panel_pass and book_pass
    return {
        "passed": passed,
        "panel_passed": panel_pass,
        "book_passed": book_pass,
        "why_panel": why_panel,
        "why_book": why_book,
        "headline": ("PRE-REGISTERED QUESTION: ANSWERED YES" if passed else
                     "PRE-REGISTERED QUESTION: ANSWERED NO"),
    }
