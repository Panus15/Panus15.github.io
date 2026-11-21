"""How many trades a claim would need — the question that ends most FX research.

`backtest.py` refuses to print a Sharpe ratio and warns below 30 trades, and
`verdict.py` refuses a result that is positive but inside its own noise. This
module is where those thresholds come from, and it exists because the honest
answer to "is my system good?" is almost always "you cannot tell yet, and here
is how long it would take to find out".

THE NUMBER THAT SURPRISES PEOPLE. A rule that wins 55% at a payoff of 1.0 has a
real edge. Distinguishing that 55% from the 50% it must beat takes 389 trades —
seven and a half years at one trade a week. Shave the edge to 52% and it takes
2,425 trades, or forty-six years. That is the quadratic in action: the edge
fell by half and the data requirement rose six-fold. The first years of trading
are exactly when people conclude their system works, or doesn't, and act on it,
and they are precisely the years in which the sample cannot yet say.

WHY THE NORMAL APPROXIMATION IS NOT USED FOR WIN RATES. p̂ ± 1.96·sqrt(p̂(1-p̂)/n)
is what everyone reaches for, and at small n it fails in the dangerous
direction. At 19 wins in 20 trades it returns (0.854, 1.046) — an upper bound
above 100%, which is not an approximation but an impossibility, and a lower
bound 9 points too high. `wilson_interval` gives (0.764, 0.991) for the same
data. The gap matters most exactly where people have the least data and the
most confidence.

WHAT EVERY FUNCTION HERE ASSUMES, AND WHY THE TRUE ANSWER IS WORSE. All of it
treats trades as INDEPENDENT. Overlapping positions in one pair are not, and
correlated pairs traded together are not — EURUSD and GBPUSD share a dollar leg,
so "200 trades across four majors" may be closer to 60 independent bets. Every
sample size below is therefore a FLOOR. The direction of the error is always the
same: you need more data than this says, never less.
"""

from __future__ import annotations

import math
from statistics import NormalDist

#: Default two-sided confidence for interval estimates.
CONFIDENCE = 0.95

#: Upper bound on the search in `trades_to_separate`. A requirement past this is
#: reported as unreachable rather than as a number nobody will ever trade to.
SEARCH_CAP = 10_000_000


def z_for(confidence: float = CONFIDENCE) -> float:
    """Two-sided normal quantile for a confidence level."""
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence!r}")
    return NormalDist().inv_cdf(1.0 - (1.0 - confidence) / 2.0)


def wilson_interval(wins: int, n: int, confidence: float = CONFIDENCE) -> tuple:
    """Wilson score interval for a win rate — the one that is honest at small n.

    At 15 wins in 20 trades the textbook interval is (0.560, 0.940) and this one
    is (0.531, 0.888) — wider at both ends, because 20 trades really do tell you
    that little. At 19 wins in 20 the textbook version returns an upper bound of
    1.046, which is not a conservative estimate but an impossible one.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= wins <= n:
        raise ValueError(f"wins must be in [0, {n}], got {wins!r}")
    z = z_for(confidence)
    p = wins / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return (max(centre - half, 0.0), min(centre + half, 1.0))


def trades_to_separate(win_rate: float, break_even: float,
                       confidence: float = CONFIDENCE,
                       cap: int = SEARCH_CAP) -> float:
    """Trades until ``win_rate`` is distinguishable from ``break_even``.

    Defined as the smallest n whose Wilson LOWER bound clears the break-even
    rate — i.e. the point at which "I win more often than I need to" stops being
    compatible with "I do not". Returns ``inf`` when the claimed rate does not
    exceed break-even at all, because then no amount of data establishes it.

    This is the function that turns "my system wins 55%" into "come back in 389
    trades", which is the single most useful sentence in retail system
    evaluation — and into "come back in 2,425" when the claim is 52%.
    """
    if not 0.0 <= win_rate <= 1.0:
        raise ValueError(f"win_rate must be in [0, 1], got {win_rate!r}")
    if win_rate <= break_even:
        # Not an optimisation. The search below would also run to the cap and
        # return inf, but then "this claim is false" and "this claim is true and
        # needs more than ten million trades" would be the same answer arrived at
        # the same way. The early return keeps inf meaning the first thing.
        return math.inf
    lo, hi = 1, 1
    while hi <= cap:
        # the Wilson lower bound rises with n, so the first n that clears is
        # found by doubling and then bisecting rather than by a closed form
        if wilson_interval(round(win_rate * hi), hi, confidence)[0] > break_even:
            break
        lo, hi = hi, hi * 2
    if hi > cap:
        return math.inf
    while lo < hi:
        mid = (lo + hi) // 2
        if wilson_interval(round(win_rate * mid), mid, confidence)[0] > break_even:
            hi = mid
        else:
            lo = mid + 1
    return float(lo)


def trades_needed(edge_pips: float, sd_pips: float, t_required: float) -> float:
    """n = (t·sd / edge)² — trades to detect an edge of this size.

    The quadratic is the whole problem. Halving the edge does not double the
    data you need, it QUADRUPLES it, which is why a thin edge is not merely
    harder to find but practically unverifiable: +0.2 pips a trade against a
    50-pip standard deviation needs about 250,000 trades at t = 2.
    """
    if edge_pips <= 0:
        return math.inf
    if sd_pips < 0 or t_required <= 0:
        raise ValueError("sd_pips must be >= 0 and t_required > 0")
    return (t_required * sd_pips / edge_pips) ** 2


def detectable_edge(n: int, sd_pips: float, t_required: float) -> float:
    """The smallest edge ``n`` trades could establish — `trades_needed` inverted.

    Ask this BEFORE running a backtest. If your sample can only establish an
    edge of 3 pips a trade and no FX pattern plausibly has one, the study cannot
    succeed and its result will be noise whichever way it comes out.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    return t_required * sd_pips / math.sqrt(n)


def years_needed(n: float, trades_per_year: float) -> float:
    """How long ``n`` trades takes at your actual frequency.

    The translation that settles most arguments. A requirement of 1,000 trades
    is abstract; "nineteen years at one trade a week" is not.
    """
    if trades_per_year <= 0:
        raise ValueError("trades_per_year must be positive")
    return n / trades_per_year


def sample_size_report(win_rate: float, break_even: float,
                       trades_per_year: float,
                       confidence: float = CONFIDENCE) -> str:
    """The plain-language version, with the years attached to the trades."""
    n = trades_to_separate(win_rate, break_even, confidence)
    if n == math.inf:
        return (f"a {win_rate:.1%} win rate never separates from the "
                f"{break_even:.1%} it must beat — no sample size establishes a "
                f"claim that is not true to begin with")
    yrs = years_needed(n, trades_per_year)
    return (f"{win_rate:.1%} vs a {break_even:.1%} break-even needs about "
            f"{n:,.0f} trades at {confidence:.0%} confidence — {yrs:.1f} years "
            f"at {trades_per_year:,.0f} trades a year, and that assumes the "
            f"trades are independent, which overlapping FX positions are not")
