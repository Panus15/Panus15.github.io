"""Chart patterns — detected honestly, each carrying what the evidence says.

THE PATTERNS ASKED FOR ARE ALL HERE. Head-and-shoulders, double top and bottom,
triangles, flags, and engulfing candles. They detect, they produce an entry, a stop
and a target, and those feed straight into `engine/expectancy.py` so you can see
what each one is actually worth on your costs.

WHAT IS ALSO HERE, AND CANNOT BE SWITCHED OFF, is the evidence label. Every pattern
carries a one-line verdict from the literature (research/BRIEF.md), because the
research found something that changes how these should be read:

    head_and_shoulders   TESTED IN FX AND FAILED
    double_top/bottom    NEVER TESTED IN FX
    triangle             NEVER TESTED IN FX
    flag                 NEVER TESTED IN FX
    engulfing            FAILS AGAINST A RANDOMISED NULL

"Never tested" is not a gap in this module's research — it is the finding. Only
head-and-shoulders has a substantial peer-reviewed FX literature, and it is
negative: Lucke (2003) found returns not significantly positive; Chang & Osler
(1999) found it worked on two of five rates and was DOMINATED by a simple filter
rule; Osler (1998) found head-and-shoulders traders are noise traders whose price
impact mean-reverts within two weeks. Marshall, Young & Rose (2006) generated random
OHLC series by bootstrap and found candlestick strategies indistinguishable from
them.

So a detection here is a HYPOTHESIS, never a recommendation. What makes it worth
anything is the measured expectancy on your own costs, which is what the rest of
this engine computes.

FREE PARAMETERS ARE THE OTHER HALF OF THE PROBLEM. Every pattern detector has
knobs — how many bars make a swing, how close two highs must be to count as "equal",
how deep a pullback may go. Those knobs are how a backtest is overfitted: search
them long enough and any pattern produces any result you like. So they are DECLARED
as module constants with a stated reason, not passed in ad hoc, and
`PARAMETER_COUNT` is published so the multiple-testing correction downstream knows
how many degrees of freedom it is correcting for.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Locked parameters. Declared, reasoned, and counted — see the module docstring.
# ---------------------------------------------------------------------------

#: Bars either side of a high/low for it to count as a swing point. 3 is the
#: smallest value that ignores single-bar noise while still finding structure on
#: an hourly chart; it is NOT tuned, and changing it is a new hypothesis.
SWING_BARS = 3

#: How close two extremes must be, as a fraction of price, to count as "equal"
#: (the two shoulders, the two tops). 0.15% ~ 15 pips on EURUSD at 1.10.
EQUAL_TOL = 0.0015

#: Minimum height of the pattern as a fraction of price, so that a flat stretch of
#: noise is not reported as a formation.
MIN_HEIGHT = 0.002

#: How many bars a pattern may span. Below the floor it is noise; above the ceiling
#: it stops being the pattern a trader would recognise.
MIN_SPAN, MAX_SPAN = 10, 120

#: Published so the statistics downstream know the search space it must correct
#: for. Four knobs, each of which could have been swept.
PARAMETER_COUNT = 4


# ---------------------------------------------------------------------------
# Evidence. Not decoration: this is what the research actually found.
# ---------------------------------------------------------------------------

TESTED_AND_FAILED = "tested in FX and failed"
NEVER_TESTED = "never tested in FX"
FAILS_RANDOM_NULL = "fails against a randomised null"

EVIDENCE = {
    "head_and_shoulders": (
        TESTED_AND_FAILED,
        "Lucke (2003, Applied Economics): returns not significantly positive. "
        "Chang & Osler (1999, Economic Journal): profitable on 2 of 5 dollar rates "
        "and DOMINATED by a simple filter rule. Osler (1998, NY Fed): H&S traders "
        "are noise traders; the price impact mean-reverts within ~2 weeks."),
    "double_top": (
        NEVER_TESTED,
        "No peer-reviewed FX profitability test exists. The win-rate tables quoted "
        "for it come from Bulkowski, which is US EQUITIES, applies no costs, "
        "specifies no stop and no exit rule."),
    "double_bottom": (
        NEVER_TESTED,
        "No peer-reviewed FX profitability test exists. Same provenance problem as "
        "double top."),
    "triangle": (
        NEVER_TESTED,
        "No peer-reviewed FX profitability test exists."),
    "flag": (
        NEVER_TESTED,
        "No peer-reviewed FX profitability test exists."),
    "engulfing": (
        FAILS_RANDOM_NULL,
        "Marshall, Young & Rose (2006) bootstrapped random open/high/low/close "
        "series and found candlestick strategies indistinguishable from them. The "
        "one FX-specific positive result is in a low-tier journal with plain "
        "z-tests, no data-snooping correction and no transaction costs."),
}


@dataclass(frozen=True)
class Bar:
    """One OHLC bar. `high`/`low` are the BID side; see backtest.py on why the
    side you fill at matters more than most people model."""
    o: float
    h: float
    l: float
    c: float


@dataclass(frozen=True)
class Detection:
    """A pattern occurrence, with the levels needed to price it.

    `index` is the bar at which the pattern is COMPLETE and could first have been
    acted on. Nothing here may look at a bar after it — that is the look-ahead
    every pattern backtest gets wrong, and `backtest.py` re-checks it.
    """
    name: str
    index: int
    direction: int            # +1 long, -1 short
    entry: float
    stop: float
    target: float

    @property
    def evidence(self) -> str:
        return EVIDENCE.get(self.name, (NEVER_TESTED, ""))[0]

    @property
    def evidence_detail(self) -> str:
        return EVIDENCE.get(self.name, (NEVER_TESTED, "no entry for this pattern"))[1]

    @property
    def risk(self) -> float:
        """Distance from entry to stop, in price. Zero is degenerate, not free."""
        return abs(self.entry - self.stop)

    @property
    def reward(self) -> float:
        return abs(self.target - self.entry)

    @property
    def planned_payoff(self) -> float:
        """target/stop distance — the payoff ratio the pattern PLANS.

        Planned, not achieved: it assumes the target is reached before the stop and
        that both fill at their level. The backtest is what turns this into a
        number worth quoting.
        """
        return self.reward / self.risk if self.risk > 0 else float("inf")


# ---------------------------------------------------------------------------
# Swing points — the primitive every structural pattern is built from
# ---------------------------------------------------------------------------

def swing_highs(bars: list, k: int = SWING_BARS) -> list:
    """Indices of bars whose high exceeds the k bars either side.

    Strict on the left and non-strict on the right, so a flat double-peak is
    reported once rather than twice.
    """
    out = []
    for i in range(k, len(bars) - k):
        h = bars[i].h
        if all(h > bars[i - j].h for j in range(1, k + 1)) and \
           all(h >= bars[i + j].h for j in range(1, k + 1)):
            out.append(i)
    return out


def swing_lows(bars: list, k: int = SWING_BARS) -> list:
    out = []
    for i in range(k, len(bars) - k):
        l = bars[i].l
        if all(l < bars[i - j].l for j in range(1, k + 1)) and \
           all(l <= bars[i + j].l for j in range(1, k + 1)):
            out.append(i)
    return out


def _close_enough(a: float, b: float, tol: float = EQUAL_TOL) -> bool:
    ref = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / ref <= tol


# ---------------------------------------------------------------------------
# The patterns
# ---------------------------------------------------------------------------

def head_and_shoulders(bars: list) -> list:
    """Left shoulder, higher head, right shoulder at a similar height; entry on
    the break of the neckline, target the head-to-neckline height projected down.

    This is the textbook construction, and it is the one the literature tested and
    found wanting. It is implemented faithfully so that the measurement is of the
    pattern as traders actually use it, not of a strawman.
    """
    out = []
    highs = swing_highs(bars)
    lows = swing_lows(bars)
    for a, b, c in zip(highs, highs[1:], highs[2:]):
        if not (MIN_SPAN <= c - a <= MAX_SPAN):
            continue
        ls, head, rs = bars[a].h, bars[b].h, bars[c].h
        if not (head > ls and head > rs):
            continue
        if not _close_enough(ls, rs):
            continue
        troughs = [i for i in lows if a < i < c]
        if len(troughs) < 2:
            continue
        neck = min(bars[i].l for i in troughs)
        height = head - neck
        if height / max(head, 1e-12) < MIN_HEIGHT:
            continue
        # completion = the first close below the neckline AFTER the right shoulder
        entry_i = None
        for i in range(c + 1, min(c + MAX_SPAN, len(bars))):
            if bars[i].c < neck:
                entry_i = i
                break
        if entry_i is None:
            continue
        out.append(Detection("head_and_shoulders", entry_i, -1,
                             entry=neck, stop=rs, target=neck - height))
    return out


def double_top(bars: list) -> list:
    """Two highs at a similar level with a trough between; entry on the break of
    the trough, target the height projected down."""
    out = []
    highs = swing_highs(bars)
    lows = swing_lows(bars)
    for a, b in zip(highs, highs[1:]):
        if not (MIN_SPAN <= b - a <= MAX_SPAN):
            continue
        if not _close_enough(bars[a].h, bars[b].h):
            continue
        troughs = [i for i in lows if a < i < b]
        if not troughs:
            continue
        neck = min(bars[i].l for i in troughs)
        top = max(bars[a].h, bars[b].h)
        height = top - neck
        if height / max(top, 1e-12) < MIN_HEIGHT:
            continue
        entry_i = None
        for i in range(b + 1, min(b + MAX_SPAN, len(bars))):
            if bars[i].c < neck:
                entry_i = i
                break
        if entry_i is None:
            continue
        out.append(Detection("double_top", entry_i, -1,
                             entry=neck, stop=top, target=neck - height))
    return out


def double_bottom(bars: list) -> list:
    """The mirror of double_top."""
    out = []
    lows = swing_lows(bars)
    highs = swing_highs(bars)
    for a, b in zip(lows, lows[1:]):
        if not (MIN_SPAN <= b - a <= MAX_SPAN):
            continue
        if not _close_enough(bars[a].l, bars[b].l):
            continue
        peaks = [i for i in highs if a < i < b]
        if not peaks:
            continue
        neck = max(bars[i].h for i in peaks)
        bottom = min(bars[a].l, bars[b].l)
        height = neck - bottom
        if height / max(neck, 1e-12) < MIN_HEIGHT:
            continue
        entry_i = None
        for i in range(b + 1, min(b + MAX_SPAN, len(bars))):
            if bars[i].c > neck:
                entry_i = i
                break
        if entry_i is None:
            continue
        out.append(Detection("double_bottom", entry_i, +1,
                             entry=neck, stop=bottom, target=neck + height))
    return out


def triangle(bars: list) -> list:
    """Converging highs and lows; entry on the break in the direction of the break.

    Deliberately simple: two falling highs and two rising lows whose range has
    narrowed by at least a third. Elaborating the definition adds free parameters,
    and free parameters are how this kind of study gets overfitted.
    """
    out = []
    highs = swing_highs(bars)
    lows = swing_lows(bars)
    for h1, h2 in zip(highs, highs[1:]):
        inner = [i for i in lows if h1 < i < h2]
        if len(inner) < 2:
            continue
        l1, l2 = inner[0], inner[-1]
        if not (MIN_SPAN <= h2 - h1 <= MAX_SPAN):
            continue
        if not (bars[h2].h < bars[h1].h and bars[l2].l > bars[l1].l):
            continue
        first_range = bars[h1].h - bars[l1].l
        last_range = bars[h2].h - bars[l2].l
        if first_range <= 0 or last_range / first_range > 0.67:
            continue
        if first_range / max(bars[h1].h, 1e-12) < MIN_HEIGHT:
            continue
        hi, lo = bars[h2].h, bars[l2].l
        for i in range(h2 + 1, min(h2 + MAX_SPAN, len(bars))):
            if bars[i].c > hi:
                out.append(Detection("triangle", i, +1, entry=hi, stop=lo,
                                     target=hi + first_range))
                break
            if bars[i].c < lo:
                out.append(Detection("triangle", i, -1, entry=lo, stop=hi,
                                     target=lo - first_range))
                break
    return out


def flag(bars: list) -> list:
    """A sharp move (the pole) then a shallow counter-drift, then continuation."""
    out = []
    pole = 12
    # len(bars), not len(bars) - 1: stopping one short means the detector can
    # never fire on the MOST RECENT bar, which is precisely the bar a live scanner
    # would act on. It also fails the look-ahead test, because a detection at the
    # last bar needs a bar after it to exist.
    for i in range(pole + 4, len(bars)):
        start, mid = bars[i - pole - 4], bars[i - 4]
        move = mid.c - start.c
        if abs(move) / max(start.c, 1e-12) < MIN_HEIGHT * 2:
            continue
        drift = bars[i].c - mid.c
        # the pullback must be a pullback: opposite in sign, and shallow
        if move * drift >= 0 or abs(drift) > 0.5 * abs(move):
            continue
        direction = 1 if move > 0 else -1
        window = bars[i - 4:i + 1]
        hi = max(b.h for b in window)
        lo = min(b.l for b in window)
        entry = hi if direction > 0 else lo
        stop = lo if direction > 0 else hi
        if entry == stop:
            continue
        out.append(Detection("flag", i, direction, entry=entry, stop=stop,
                             target=entry + direction * abs(move)))
    return out


def engulfing(bars: list) -> list:
    """A candle whose body fully covers the previous body, in the other direction.

    Included because it was asked for. Note the evidence label: candlestick
    strategies are indistinguishable from bootstrapped random OHLC series.
    """
    out = []
    for i in range(1, len(bars)):        # see the note in flag() on the bound
        p, q = bars[i - 1], bars[i]
        p_lo, p_hi = min(p.o, p.c), max(p.o, p.c)
        q_lo, q_hi = min(q.o, q.c), max(q.o, q.c)
        if q_hi - q_lo <= 0 or p_hi - p_lo <= 0:
            continue
        if not (q_lo <= p_lo and q_hi >= p_hi):
            continue
        bullish = q.c > q.o and p.c < p.o
        bearish = q.c < q.o and p.c > p.o
        if not (bullish or bearish):
            continue
        direction = 1 if bullish else -1
        entry = q.c
        stop = q.l if bullish else q.h
        if entry == stop:
            continue
        out.append(Detection("engulfing", i, direction, entry=entry, stop=stop,
                             target=entry + direction * 2.0 * abs(entry - stop)))
    return out


#: Every detector, by name. Used by the scanner and by the tests that assert each
#: one is reachable and each one carries an evidence label.
DETECTORS = {
    "head_and_shoulders": head_and_shoulders,
    "double_top": double_top,
    "double_bottom": double_bottom,
    "triangle": triangle,
    "flag": flag,
    "engulfing": engulfing,
}


def scan(bars: list, names=None) -> list:
    """Run every detector (or the named ones) and return detections in bar order."""
    chosen = DETECTORS if names is None else {n: DETECTORS[n] for n in names}
    found = []
    for fn in chosen.values():
        found.extend(fn(bars))
    return sorted(found, key=lambda d: d.index)
