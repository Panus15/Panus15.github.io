"""The out-of-sample test — the only one that settles anything.

`demo.py` measures what an in-sample pass is worth: run this pipeline over
random walks, where nothing can be true, and 3 of 150 pattern-tests still come
back tradeable after costs AND the multiple-testing correction. So a verdict
from `analyse.py` is a hypothesis that survived one test on the data it was
found in. This module is where it meets data it has never seen.

THE SPLIT IS CHRONOLOGICAL AND THE REASON IS NOT STYLE. Shuffling rows before
splitting is standard practice on cross-sectional data and catastrophic here.
It leaks in three ways at once: neighbouring bars overlap in the trades they
generate, the price LEVEL itself carries the answer (a bar at 1.45 tells you
the year), and a pattern's target is resolved by the bars immediately after it,
which shuffling scatters into the training half. A shuffled FX backtest reports
an edge on pure noise. The split here is a single cut in time, and the later
half is never touched until the earlier half is finished with.

EACH HALF IS ANALYSED AS IF IT WERE A SEPARATE FILE. Detections are recomputed
inside each segment rather than carried across the cut, so no trade can enter
before the boundary and exit after it. The cost is that the first hundred-odd
bars of the out-of-sample half cannot form a pattern needing that much history
— which is right, because you would not have had it either.

THE CORRECTION DENOMINATOR CHANGES, AND IT MUST. In-sample you searched every
pattern, so the denominator is the whole search. Out-of-sample you are testing
only the ones that survived, so it is the number of survivors. Reusing the
original width would be too strict and would hide real results; ignoring the
correction entirely would let the survivor with the luckiest half look
confirmed. `n_survivors` is what this uses, and it is stated in the report.

WHAT THE LITERATURE SAYS THIS STEP USUALLY DOES, IN CURRENCIES SPECIFICALLY.
Hutchinson et al. (2022, Research in International Business and Finance) took a
portfolio of currency technical trading rules and split it the way this module
does. The mean Sharpe fell from 0.66 in-sample to 0.06 out-of-sample, and the
returns did not survive modest transaction costs out of sample. They also found
that whatever the rules earned was fully explained by time-series momentum, so
"the pattern failed, I will use a trend filter instead" is not a second idea.

That is the expected shape of the result below: roughly a tenth of what the
in-sample number promised, and then nothing once costs are charged. If your
hold-out CONFIRMS, the base rate says look for the leak before celebrating.

WHAT IT TOOK TO GET A CONFIRMATION AT ALL, MEASURED. This module's own
positive control — a synthetic series with a genuine, persistent trend — only
confirms out-of-sample when the trend multiplies the price SEVENFOLD over the
sample. At a fifth of that length and the same drift, the held-back half
produced 63 trades against the ~305 a +7 pip edge on a 76 pip standard
deviation needs, and refused. Sample length and trend strength trade off
against each other and FX supplies too little of both, so expect this module to
refuse. That is not it failing; it is `stats.py` arriving at the same answer
from the other direction.

YOU GET ONE. The out-of-sample half stops being out-of-sample the moment you
look at it, change something, and look again. There is no counter in this code
that can enforce that and no way to detect a violation from the outside, so the
report says it in words every time. If you find yourself re-running this after
an adjustment, you are no longer testing — you are searching a second data set,
and the honest thing is to say so and find a third.
"""

from __future__ import annotations

from dataclasses import dataclass

from engine.backtest import run
from engine.costs import CostModel
from engine.verdict import judge
from models.patterns import DETECTORS

#: Fraction of the series used to FIND things. The remainder is held back. 0.7
#: is a convention, not a result: too small a training half finds nothing, too
#: small a test half cannot confirm anything, and there is no optimum to tune
#: toward — tuning it against the outcome is exactly the sin this module exists
#: to prevent.
SPLIT_DEFAULT = 0.7

#: Below this many bars, the held-back half cannot produce enough trades for a
#: verdict to mean anything, and the honest report is that the file is too short
#: to validate rather than a confirmation nobody should believe.
MIN_HOLDOUT_BARS = 500


@dataclass
class HoldoutResult:
    """One pattern, tested twice: where it was found, and where it was not."""

    name: str
    in_sample: object                  # Verdict
    out_of_sample: object = None       # Verdict, or None if never tested
    confirmed: bool = False

    @property
    def status(self) -> str:
        if not self.in_sample.tradeable:
            return "refused in-sample"
        if self.out_of_sample is None:
            return "not tested out-of-sample"
        return "CONFIRMED" if self.confirmed else "FAILED out-of-sample"

    def summary(self) -> str:
        i = self.in_sample
        lines = [f"{self.name}: {self.status}",
                 f"  in-sample      {i.n_trades:>6,} trades  "
                 f"{i.net.net:>+8.2f} pips/trade  "
                 f"{'passed' if i.tradeable else i.codes[0] if i.codes else '-'}"]
        o = self.out_of_sample
        if o is not None:
            lines.append(
                f"  out-of-sample  {o.n_trades:>6,} trades  "
                f"{o.net.net:>+8.2f} pips/trade  "
                f"{'passed' if o.tradeable else o.codes[0] if o.codes else '-'}")
            if not self.confirmed and o.refusals:
                # guarded on the list being indexed, not on `codes`, which is
                # derived from it: checking one and indexing the other works
                # only for as long as they cannot drift apart
                lines.append(f"  why: {o.refusals[0]}")
        return "\n".join(lines)


@dataclass
class Holdout:
    """The whole test: where the cut fell, and what happened either side.

    The sizes live here rather than being recomputed by whoever prints them.
    The first version had the caller derive `int(len(bars) * frac)` for the
    report while `evaluate` split the bars itself — two computations of one
    fact, which is the shape of every disagreement bug in this repository. A
    mutation that changed the split silently kept the old numbers in the
    report, and no test could see it.
    """

    frac: float
    n_train: int
    n_test: int
    results: list

    @property
    def survivors(self) -> list:
        return [r for r in self.results if r.in_sample.tradeable]

    @property
    def confirmed(self) -> list:
        return [r for r in self.results if r.confirmed]

    def report(self) -> str:
        """The findings, and the sentence about using the holdout only once."""
        lines = [f"HOLD-OUT: found on {self.n_train:,} bars, tested on "
                 f"{self.n_test:,} never seen ({1 - self.frac:.0%} held back)"]
        for r in self.results:
            lines.append("  " + r.summary().replace("\n", "\n  "))

        if not self.survivors:
            lines.append("")
            lines.append("Nothing passed in-sample, so there was nothing to "
                         "test. The held-back half is\nstill untouched — which "
                         "is worth more than a result would have been.")
            return "\n".join(lines)
        if self.n_test < MIN_HOLDOUT_BARS:
            lines.append("")
            lines.append(f"NOT TESTED: {self.n_test:,} bars held back is too "
                         f"few to confirm anything, so the\nin-sample pass "
                         f"stands unvalidated. A longer file, not a smaller "
                         f"split.")
            return "\n".join(lines)

        lines.append("")
        lines.append(f"{len(self.confirmed)} of {len(self.survivors)} in-sample "
                     f"pass(es) survived data they had never seen,\ncorrected "
                     f"for {len(self.survivors)} hypothes(es) rather than the "
                     f"full search — because only the\nsurvivors were tested.")
        if not self.confirmed:
            lines.append("A pattern that passes in-sample and fails "
                         "out-of-sample is the ORDINARY outcome,\nnot a "
                         "malfunction. It is what the in-sample pass was always "
                         "most likely to be.")
        lines.append("")
        lines.append("YOU GET ONE. The held-back half stopped being "
                     "out-of-sample the moment you read\nthis. Nothing in this "
                     "code can enforce that or detect a violation. If you "
                     "adjust\nanything and run again, you are searching a "
                     "second data set, not testing — and the\nhonest move is "
                     "to say so and go and find a third.")
        return "\n".join(lines)


def split(bars: list, frac: float = SPLIT_DEFAULT) -> tuple:
    """One cut in time. Never shuffled — see the module docstring for why.

    Returns (earlier, later). The later half is what you have not looked at.
    """
    if not 0.0 < frac < 1.0:
        raise ValueError(f"frac must be in (0, 1), got {frac!r}")
    cut = int(len(bars) * frac)
    return bars[:cut], bars[cut:]


def evaluate(bars: list, costs: CostModel, *, frac: float = SPLIT_DEFAULT,
             bars_per_night: float = 1.0, costs_confirmed: bool = False,
             patterns=None) -> Holdout:
    """Find on the earlier half, test the survivors on the later half.

    Only patterns that PASS in-sample are tested out-of-sample. Testing all of
    them and reporting the best would put the multiple-testing problem straight
    back into the step that exists to escape it.
    """
    chosen = DETECTORS if not patterns else {p: DETECTORS[p] for p in patterns}
    train, test = split(bars, frac)

    results = []
    for name, fn in chosen.items():
        res = run(train, fn(train), costs, bars_per_night=bars_per_night)
        v = judge(res, costs, name=name, n_hypotheses=len(chosen) * 4,
                  costs_confirmed=costs_confirmed)
        results.append(HoldoutResult(name=name, in_sample=v))

    out = Holdout(frac=frac, n_train=len(train), n_test=len(test),
                  results=results)
    survivors = out.survivors
    if not survivors or len(test) < MIN_HOLDOUT_BARS:
        return out

    # the denominator is now the number of survivors, not the original search
    for r in survivors:
        fn = chosen[r.name]
        res = run(test, fn(test), costs, bars_per_night=bars_per_night)
        r.out_of_sample = judge(res, costs, name=r.name,
                                n_hypotheses=len(survivors),
                                costs_confirmed=costs_confirmed)
        r.confirmed = r.out_of_sample.tradeable
    return out
