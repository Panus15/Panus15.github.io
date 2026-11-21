"""How often this engine is wrong, measured, next to what the research expects.

Every other tool here judges a trading rule. This one judges the judge. It runs
the complete pipeline — detect, backtest at realistic costs, correct for the
search, hold out the later part of the series — over data whose truth is known
by construction, and counts how often the answer is right.

    python3 tools/calibrate.py                  # 20 trials each way
    python3 tools/calibrate.py --trials 50

TWO CONDITIONS, BECAUSE ONE NUMBER IS NOT AN ACCURACY.

    NOISE   a seeded random walk. Nothing can be true, so every CONFIRM is a
            false positive and the rate is this engine's risk of telling you to
            trade something that is not there.
    TREND   a series with a genuine, persistent drift. Every CONFIRM is correct
            and every refusal is a miss, so the rate is its power.

A filter that refuses everything scores perfectly on the first and zero on the
second, and the two are indistinguishable until both are measured. That is why
both are here and why neither is reported alone.

THE NUMBER THAT HAS TO BE READ WITH ITS CAVEAT. The TREND condition uses a
drift far stronger than any currency produces — calibrated to roughly double
the price over the sample, about three times faster than the strongest trend
EURUSD has ever had (0.85 in 2000 to 1.60 in 2008). So the power figure below is
an UPPER BOUND. Against a real FX trend it is lower, and this engine cannot
measure how much lower, because FX does not supply a trend that strong to
measure against. Treating the upper bound as the expected rate is the single
easiest way to misread this report.

WHAT IT IS NOT. It is not a backtest of your data and it says nothing about
your broker. It is the engine's own error rate on synthetic series, which is the
only data whose answer is known. On your data nobody knows the answer, which is
the whole reason the refusals exist.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import statistics
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.costs import CostModel
from engine.holdout import SPLIT_DEFAULT, evaluate
from engine.stats import trades_to_separate
from models.patterns import DETECTORS, PARAMETER_COUNT, Bar

#: Costs used throughout. Deliberately at the CHEAP end of what retail pays, so
#: nothing below can be dismissed as a pessimistic cost model.
COSTS = CostModel(pair="EURUSD", round_turn_pips=1.1, swap_markup_annual=0.008)

#: Bars per financing rollover. The synthetic series is hourly-sized.
BARS_PER_NIGHT = 24

#: Drift for the TREND condition, calibrated rather than picked: it roughly
#: doubles the price over 15,000 bars. See the module docstring on why the power
#: figure it produces is an upper bound.
TREND_DRIFT = 0.00004

#: Nominal one-sided significance after the Bonferroni correction across the
#: six detectors' declared parameters. What the arithmetic PROMISES, against
#: which the measured rate below is the reality.
NOMINAL_ALPHA = 0.05 / (len(DETECTORS) * PARAMETER_COUNT)


@dataclass(frozen=True)
class Benchmark:
    """A published figure this engine's own measurement is placed beside."""
    label: str
    value: str
    source: str
    bearing: str


BENCHMARKS = (
    Benchmark(
        "in-sample to out-of-sample decay, currency technical rules",
        "Sharpe 0.66 -> 0.06, and returns do not survive modest costs",
        "Hutchinson et al. (2022), Research in International Business and Finance",
        "the decay this engine measures below should be the same order. If it "
        "measures much less, its hold-out is too easy"),
    Benchmark(
        "post-publication decay, currency strategies",
        "out-of-sample Sharpe +0.39 -> -0.32",
        "Hutchinson & O'Brien",
        "an edge that was real can stop being real once it is published, which "
        "no amount of in-sample testing detects"),
    Benchmark(
        "real retail outcomes",
        "61% win rate and still losing: 48 pip wins against 83 pip losses",
        "FXCM / DailyFX, 43 million real trades",
        "the failure mode is not a bad win rate, so an engine that only "
        "reported win rates would have called these traders successful"),
    Benchmark(
        "share of retail accounts that lose money",
        "74-89%",
        "ESMA / FCA / AMF",
        "the base rate any claim of edge is competing against"),
    Benchmark(
        "chart patterns in FX, peer-reviewed",
        "head-and-shoulders not significantly profitable; candlesticks "
        "indistinguishable from bootstrapped random OHLC",
        "Lucke (2003); Chang & Osler (1999); Marshall, Young & Rose (2006)",
        "the prior against which a CONFIRM on any of the six shipped patterns "
        "should be read"),
)


def series(n: int, seed: int, drift: float = 0.0, sigma: float = 0.0004,
           start: float = 1.1000, substeps: int = 4) -> list:
    """OHLC bars with a known drift. The intrabar path is simulated so highs
    and lows behave like real ones and the ambiguous-bar problem shows up."""
    rng = random.Random(seed)
    out, px, s = [], start, sigma / math.sqrt(substeps)
    for _ in range(n):
        o = px
        path = [o]
        for _ in range(substeps):
            px *= math.exp(rng.gauss(drift / substeps, s))
            path.append(px)
        out.append(Bar(o, max(path), min(path), px))
    return out


@dataclass
class Condition:
    """What the full pipeline did over one condition's trials."""
    name: str
    truth: bool                 # is there anything to find?
    trials: int = 0
    rules: int = 0              # rule-trials attempted
    in_sample_passes: int = 0
    confirmed: int = 0
    not_tested: int = 0
    #: Trials in which AT LEAST ONE rule confirmed. The per-rule rate divides
    #: by six detectors, most of which are reversal patterns and have no
    #: business firing on a trend — so it understates whether the engine would
    #: have FOUND the thing, which is the question power actually asks.
    trials_with_confirm: int = 0
    decays: list = None
    move: float = 1.0

    def __post_init__(self):
        if self.decays is None:
            self.decays = []

    @property
    def in_sample_rate(self) -> float:
        return self.in_sample_passes / self.rules if self.rules else 0.0

    @property
    def confirm_rate(self) -> float:
        return self.confirmed / self.rules if self.rules else 0.0

    @property
    def survival(self) -> float:
        """Share of in-sample passes that survived the hold-out."""
        return (self.confirmed / self.in_sample_passes
                if self.in_sample_passes else 0.0)

    @property
    def found_rate(self) -> float:
        return self.trials_with_confirm / self.trials if self.trials else 0.0

    @property
    def median_decay(self) -> float:
        """Median out-of-sample / in-sample net pips, among in-sample passes.

        Below 1.0 means the second half delivered less than the first promised.
        This is the quantity Hutchinson et al. put at roughly 0.06/0.66.
        """
        return statistics.median(self.decays) if self.decays else float("nan")


def run_condition(name: str, truth: bool, trials: int, bars_n: int,
                  drift: float, seed0: int) -> Condition:
    c = Condition(name=name, truth=truth, trials=trials)
    for t in range(trials):
        bars = series(bars_n, seed0 + t, drift=drift)
        if t == 0:
            c.move = bars[-1].c / bars[0].o
        h = evaluate(bars, COSTS, frac=SPLIT_DEFAULT,
                     bars_per_night=BARS_PER_NIGHT, costs_confirmed=True)
        c.rules += len(h.results)
        if any(r.confirmed for r in h.results):
            c.trials_with_confirm += 1
        for r in h.results:
            if not r.in_sample.tradeable:
                continue
            c.in_sample_passes += 1
            if r.out_of_sample is None:
                c.not_tested += 1
                continue
            if r.confirmed:
                c.confirmed += 1
            a, b = r.in_sample.net.net, r.out_of_sample.net.net
            if a > 0:
                c.decays.append(b / a)
    return c


def report(noise: Condition, trend: Condition) -> str:
    L = []
    L.append("CALIBRATION — this engine's own error rate, measured")
    L.append("")
    L.append(f"  costs {COSTS.round_turn_pips} pip round turn, "
             f"{COSTS.swap_markup_annual:.1%}/yr markup · "
             f"{int(SPLIT_DEFAULT * 100)}/{100 - int(SPLIT_DEFAULT * 100)} "
             f"hold-out split")
    L.append("")
    L.append(f"  {'condition':<10} {'rule-trials':>11} {'in-sample':>10} "
             f"{'CONFIRMED':>10} {'any rule':>9}  reading")
    for c, reading in ((noise, "false positives — nothing was there"),
                       (trend, "power — UPPER BOUND, see below")):
        L.append(f"  {c.name:<10} {c.rules:>11,} {c.in_sample_rate:>9.1%} "
                 f"{c.confirm_rate:>10.1%} {c.found_rate:>9.0%}  {reading}")
    L.append("  'CONFIRMED' is per rule-trial, across all six detectors. "
             "'any rule' is the share of")
    L.append("  SERIES where at least one confirmed — the figure that answers "
             "whether the engine")
    L.append("  would have found the thing at all, since most of the six are "
             "reversal patterns")
    L.append("  with no business firing on a trend.")
    L.append("")

    L.append("WHAT THE HOLD-OUT IS WORTH")
    if noise.in_sample_passes == 0:
        # 0/0 is not 100%. With nothing reaching the hold-out it removed
        # nothing, and crediting it with a perfect score here would be the
        # report flattering the engine it is supposed to be auditing.
        L.append(f"  Not measurable at this size: no rule-trial passed "
                 f"in-sample on noise, so no false")
        L.append(f"  positive ever reached the hold-out and it cannot be "
                 f"credited with removing any.")
        L.append(f"  Costs and the correction stopped all "
                 f"{noise.rules:,} of them first. Raise --trials to see the "
                 f"hold-out work.")
    else:
        L.append(f"  On noise, {noise.in_sample_rate:.1%} of rule-trials "
                 f"passed in-sample and {noise.confirm_rate:.1%} survived the "
                 f"hold-out, so it")
        L.append(f"  removed {1 - noise.survival:.0%} of the false positives "
                 f"an in-sample test alone would have handed you.")
    if noise.confirm_rate > 0:
        L.append(f"  It did NOT remove all of them: {noise.confirmed} of "
                 f"{noise.rules:,} rule-trials confirmed on data with nothing")
        L.append(f"  in it, against a nominal {NOMINAL_ALPHA:.2%} after the "
                 f"Bonferroni correction. The gap is overlapping trades, which")
        L.append(f"  edge_t_stat treats as independent and which no correction "
                 f"here repairs.")
    else:
        L.append(f"  Nothing confirmed on noise in {noise.trials} trials. That "
                 f"bounds the false-positive rate below "
                 f"{1.0 / max(noise.rules, 1):.2%}, it does not make it zero.")
    L.append("")

    L.append("DECAY, IN-SAMPLE TO OUT-OF-SAMPLE")
    for c in (noise, trend):
        d = c.median_decay
        L.append(f"  {c.name:<8} median out-of-sample / in-sample net pips: "
                 + ("no in-sample passes to measure" if d != d else f"{d:+.2f}"))
    L.append("  The NOISE row is the one that belongs beside Hutchinson et al.")
    L.append("  (2022), who measured 0.06/0.66 = 0.09 on currency technical "
             "rules: both are")
    L.append("  spurious edges meeting data they were not fitted to, and a "
             "number far above 0.09")
    L.append("  would mean this hold-out is too easy.")
    L.append("  The TREND row must NOT be compared to it. There the edge is "
             "real, so a figure")
    L.append("  near or above 1.0 is correct — a hold-out that decayed the "
             "trend condition too")
    L.append("  would be destroying edges that are genuinely present, which is "
             "the opposite fault.")
    L.append("")

    L.append(f"THE POWER FIGURE IS AN UPPER BOUND. The trend condition moves "
             f"the price x{trend.move:.2f}")
    L.append("  over the sample — roughly three times faster than the strongest "
             "trend EURUSD has")
    L.append("  ever had. Against a real FX trend the power is lower, and this "
             "engine cannot say")
    L.append("  how much lower, because FX does not supply one that strong to "
             "measure against.")
    L.append("")

    L.append("AGAINST THE RESEARCH")
    for b in BENCHMARKS:
        L.append(f"  · {b.label}")
        L.append(f"      {b.value}")
        L.append(f"      {b.source}")
        L.append(f"      bearing: {b.bearing}")
    L.append("")
    need = trades_to_separate(0.55, 0.50)
    L.append(f"AND THE COST OF FINDING OUT. Separating a 55% win rate from the "
             f"50% it must beat")
    L.append(f"  takes {need:,.0f} trades. At one trade a week that is "
             f"{need / 52:.0f} years. Most of what this")
    L.append("  engine refuses, it refuses because nobody could yet have known "
             "either way.")
    return "\n".join(L)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Measure this engine's own false-positive rate and power.")
    p.add_argument("--trials", type=int, default=20,
                   help="independent series per condition (default 20)")
    p.add_argument("--bars", type=int, default=15_000,
                   help="bars per series (default 15000)")
    a = p.parse_args(argv)
    if a.trials < 1 or a.bars < 1000:
        print("need --trials >= 1 and --bars >= 1000", file=sys.stderr)
        return 2
    noise = run_condition("noise", False, a.trials, a.bars, 0.0, 7000)
    trend = run_condition("trend", True, a.trials, a.bars, TREND_DRIFT, 9000)
    print(report(noise, trend))
    return 0


if __name__ == "__main__":
    sys.exit(main())
