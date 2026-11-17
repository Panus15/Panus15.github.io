"""Sector rotation — which part of the market is being paid, and which is being sold.

A Relative Rotation Graph plots every sector on two axes measured AGAINST a
benchmark (SPY for US sectors): how strong it is relative to the market, and
whether that strength is building or fading. Sectors trace a clockwise loop
through four quadrants:

        RS-Momentum
             ^
   IMPROVING | LEADING          clockwise:
   weak but  | strong and         improving -> leading -> weakening -> lagging
   building  | still building
   ----------+----------> RS-Ratio (100 = the benchmark)
   LAGGING   | WEAKENING
   weak and  | strong but
   fading    | fading

WHAT THIS IS, SAID PLAINLY. It is a transform of RELATIVE PRICE. Nothing here
reads fund flows, net creations, or anyone's order book — a sector "leading" means
its price has outrun SPY and is still outrunning it, and that is all it means. The
distinction matters because rotation charts are routinely described as showing
"where the money is going", which they do not: they show where the price has
already gone. `models/fund_flow.py` is the module in this repo that reads actual
published positions, and it covers option books, not sector flows.

THE FORMULAS. The original JdK RS-Ratio / RS-Momentum are proprietary in their
exact constants; what follows is the standard public reconstruction, stated
explicitly so nobody has to guess what was computed:

    RS_t        = 100 * price_t / benchmark_t          (the relative price line)
    RS-Ratio_t  = 100 + zscore(RS, window)             (is it above its own norm?)
    ROC_t       = RS-Ratio_t / RS-Ratio_{t-mom_lag}    (is that improving?)
    RS-Mom_t    = 100 + zscore(ROC, window)

Both axes are z-scores, so 100 is "in line with its own recent history", NOT "in
line with SPY". A sector can sit at RS-Ratio 101 while still trailing SPY over the
year — it is merely doing better than it has been. Reading the axis as absolute
outperformance is the most common way this chart is misused.

WHETHER IT PREDICTS ANYTHING is a separate question, and this repo answers those
with harnesses rather than assertions: see ``engine/rotation_backtest.py``. What
that harness found is worth knowing before reading the chart — across 15
independent synthetic worlds where sector strength genuinely persists:

    rank on relative strength alone   Sharpe +9.52  [+6.34, +12.79]
    rank on both axes (the diagonal)  Sharpe +7.60  [+3.86, +12.19]
    rank on MOMENTUM alone            Sharpe -1.17  [-3.24,  +1.22]  <- spans zero

The second axis — the thing that makes an RRG an RRG rather than a bar chart —
carried no detectable edge on its own, and adding it to relative strength did not
clearly help. Strength alone won 13 of the 15 worlds, but the interval on that
difference includes zero, so it is suggestive rather than shown. The fixture's
alpha persists but never accelerates, which is precisely the world where a
momentum axis ought to earn its keep, so this is a reason to test it on real
sector history rather than a reason to delete it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

#: The eleven SPDR sector ETFs that partition the S&P 500, plus the benchmark.
SECTOR_ETFS = {
    "XLK": "Technology",
    "XLV": "Health Care",
    "XLF": "Financials",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLE": "Energy",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLU": "Utilities",
    "XLC": "Communication Services",
}
BENCHMARK = "SPY"

LEADING, WEAKENING, LAGGING, IMPROVING = (
    "Leading", "Weakening", "Lagging", "Improving")
#: Clockwise order. A sector moving the other way is unusual and worth noticing.
CYCLE = (IMPROVING, LEADING, WEAKENING, LAGGING)


def quadrant(rs_ratio: float, rs_momentum: float) -> str:
    """Which quadrant a (RS-Ratio, RS-Momentum) point falls in. Axes cross at 100."""
    strong = rs_ratio >= 100.0
    rising = rs_momentum >= 100.0
    if strong:
        return LEADING if rising else WEAKENING
    return IMPROVING if rising else LAGGING


def _zscore_series(vals, window: int):
    """100 + rolling z-score. None until the window fills.

    A flat input has zero dispersion and no meaningful z-score; it maps to exactly
    100 rather than to an infinity, so a sector that tracks the benchmark tick for
    tick lands on the origin instead of blowing up the chart.
    """
    out = [None] * len(vals)
    for i in range(window - 1, len(vals)):
        w = vals[i - window + 1:i + 1]
        m = sum(w) / window
        var = sum((x - m) ** 2 for x in w) / max(window - 1, 1)
        sd = math.sqrt(var)
        out[i] = 100.0 if sd < 1e-12 else 100.0 + (vals[i] - m) / sd
    return out


def relative_strength(prices, benchmark) -> list:
    """RS = 100 * price / benchmark, aligned on the shorter of the two."""
    n = min(len(prices), len(benchmark))
    if n == 0:
        return []
    return [100.0 * prices[len(prices) - n + i] / benchmark[len(benchmark) - n + i]
            if benchmark[len(benchmark) - n + i] > 0 else float("nan")
            for i in range(n)]


def rrg_series(prices, benchmark, *, window: int = 63, mom_lag: int = 5):
    """(rs_ratio, rs_momentum) series for one symbol against a benchmark.

    ``window`` is the z-score lookback — about a quarter of trading days by
    default, which is what makes the axes read as "versus its own recent norm".
    ``mom_lag`` is how far back the momentum ratio looks; small values make the
    chart twitchy, large ones make it late. Both entries are None until enough
    history exists, so a caller can never silently plot a point built on less data
    than it claims.
    """
    rs = relative_strength(prices, benchmark)
    if len(rs) < window:
        return [], []
    ratio = _zscore_series(rs, window)
    roc = [None] * len(ratio)
    for i in range(len(ratio)):
        prev = ratio[i - mom_lag] if i >= mom_lag else None
        if ratio[i] is not None and prev not in (None, 0.0):
            roc[i] = ratio[i] / prev
    valid = [(i, x) for i, x in enumerate(roc) if x is not None]
    mom = [None] * len(ratio)
    if len(valid) >= window:
        z = _zscore_series([x for _, x in valid], window)
        for (idx, _), zz in zip(valid, z):
            mom[idx] = zz
    return ratio, mom


@dataclass
class RotationPoint:
    symbol: str
    name: str
    rs_ratio: float
    rs_momentum: float
    quadrant: str
    tail: list = field(default_factory=list)   # [(rs_ratio, rs_momentum), ...] oldest first
    excess_return: float = 0.0                 # vs benchmark over the tail window

    @property
    def distance(self) -> float:
        """How far from the origin — how pronounced the rotation is, either way."""
        return math.hypot(self.rs_ratio - 100.0, self.rs_momentum - 100.0)

    def line(self) -> str:
        return (f"  {self.symbol:5} {self.name:24} RS={self.rs_ratio:7.2f} "
                f"Mom={self.rs_momentum:7.2f}  {self.quadrant:10} "
                f"excess {self.excess_return:+7.2%}")


def rotation_map(prices_by_symbol: dict, benchmark_prices, *, window: int = 63,
                 mom_lag: int = 5, tail: int = 12, names: dict | None = None) -> list:
    """One RotationPoint per symbol, ranked by RS-Ratio (strongest first).

    Symbols without enough history are DROPPED rather than plotted at a default —
    an absent sector is honest, a sector pinned at the origin because its data was
    short is a lie the chart tells silently.
    """
    names = names or SECTOR_ETFS
    out = []
    # Only the last `tail` points are ever plotted, and every statistic here is
    # ROLLING: a point depends on `window` of RS, which depends on `window` more
    # through the momentum z-score, plus `mom_lag`. So anything older than that
    # cannot change the answer, and trimming to it turns an O(history) recompute
    # per date into a constant one — which is the difference between a backtest
    # that runs and one that does not. `test_trimming_the_history_changes_nothing`
    # pins the equality.
    need = tail + 2 * window + mom_lag + 2
    for sym, px in prices_by_symbol.items():
        n_keep = min(len(px), len(benchmark_prices), need)
        px_w = px[-n_keep:] if n_keep < len(px) else px
        bench_w = (benchmark_prices[-n_keep:] if n_keep < len(benchmark_prices)
                   else benchmark_prices)
        ratio, mom = rrg_series(px_w, bench_w, window=window, mom_lag=mom_lag)
        if not ratio or ratio[-1] is None or not mom or mom[-1] is None:
            continue
        pts = [(ratio[i], mom[i]) for i in range(max(0, len(ratio) - tail), len(ratio))
               if ratio[i] is not None and mom[i] is not None]
        n = min(len(px_w), len(bench_w), tail + 1)
        exc = 0.0
        if n >= 2:
            p0, p1 = px_w[-n], px_w[-1]
            b0, b1 = bench_w[-n], bench_w[-1]
            if p0 > 0 and b0 > 0:
                exc = (p1 / p0) - (b1 / b0)
        out.append(RotationPoint(
            symbol=sym, name=names.get(sym, sym), rs_ratio=ratio[-1],
            rs_momentum=mom[-1], quadrant=quadrant(ratio[-1], mom[-1]),
            tail=pts, excess_return=exc))
    return sorted(out, key=lambda p: p.rs_ratio, reverse=True)


def leaderboard(points, *, top: int | None = None, rank_by: str = "both") -> list:
    """Sectors ranked by how far into Leading (or out of Lagging) they sit.

    ``rank_by='both'`` scores signed distance along the diagonal, so strength AND
    momentum count — the stated reason to look at two axes instead of one. The
    other two settings exist so that claim can be TESTED rather than assumed:
    ``'rs'`` uses relative strength alone and ``'momentum'`` the second axis alone.
    If 'rs' does as well as 'both', the momentum axis is decoration, and
    ``engine/rotation_backtest`` is where that gets measured.
    """
    key = {
        "both": lambda p: (p.rs_ratio - 100.0) + (p.rs_momentum - 100.0),
        "rs": lambda p: p.rs_ratio - 100.0,
        "momentum": lambda p: p.rs_momentum - 100.0,
    }[rank_by]
    scored = sorted(points, key=key, reverse=True)
    return scored[:top] if top else scored


def quadrant_history(prices, benchmark, *, window: int = 63, mom_lag: int = 5,
                     bars: int = 60) -> list:
    """[(index, quadrant)] over the last ``bars`` — how the sector actually moved.

    Read it for the SEQUENCE, not the label: a sector that has sat in Leading for
    forty bars is a different proposition from one that entered yesterday, and the
    single current label cannot tell them apart.
    """
    ratio, mom = rrg_series(prices, benchmark, window=window, mom_lag=mom_lag)
    out = []
    for i in range(max(0, len(ratio) - bars), len(ratio)):
        if ratio[i] is None or mom[i] is None:
            continue
        out.append((i, quadrant(ratio[i], mom[i])))
    return out


def transitions(history) -> list:
    """The quadrant CHANGES in a history, as [(index, from, to, clockwise?)].

    Rotation is supposed to run clockwise. A counter-clockwise hop is not a bug in
    the data — it is a sector reversing before completing the loop, which is
    exactly the case where the tidy narrative of the chart breaks down.
    """
    out = []
    for k in range(1, len(history)):
        prev_q, cur_q = history[k - 1][1], history[k][1]
        if prev_q == cur_q:
            continue
        step = (CYCLE.index(cur_q) - CYCLE.index(prev_q)) % len(CYCLE)
        out.append((history[k][0], prev_q, cur_q, step == 1))
    return out


def summary_table(points) -> str:
    if not points:
        return "no sector has enough history to place on the chart"
    by_q: dict = {}
    for p in points:
        by_q.setdefault(p.quadrant, []).append(p.symbol)
    lines = [f"  {'sym':5} {'sector':24} {'RS':>7} {'Mom':>7}  {'quadrant':10} "
             f"{'excess':>8}"]
    lines += [p.line() for p in points]
    lines.append("  " + "   ".join(
        f"{q}: {'/'.join(by_q.get(q, [])) or '-'}" for q in CYCLE))
    lines.append("  NOTE: axes are z-scores vs each sector's OWN recent history, so")
    lines.append("        100 means 'in line with its own norm', not 'in line with SPY'.")
    return "\n".join(lines)
