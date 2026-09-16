"""Run the whole engine over your own bars.

    python3 tools/analyse.py data/EURUSD_H1.csv --pair EURUSD --side bid \
        --round-turn 1.3 --swap-markup 0.012 --costs-measured

It loads the file, says what is wrong with it, refuses to go on if the defects
break the engine's assumptions, and then scans, backtests and judges every
pattern against the costs YOU gave it.

THE TWO ARGUMENTS THAT ARE NOT CONVENIENCES.

``--side`` says whether the prices are bid, ask or mid. Nothing in the numbers
can tell you, and a study run on mid quietly collects half the spread twice per
trade. It is required.

``--costs-measured`` is your signature on the cost model. Without it every
verdict carries COST_NOT_MEASURED, because the broker's swap markup is not
published, is charged in BOTH directions, and decides more than any indicator
in this engine. Measuring it is an afternoon: compare the quoted swap against
the published tom-next differential, long and short, for a full week including
the Wednesday triple charge.

WHAT A PASS FROM THIS TOOL IS. A hypothesis that survived one in-sample test.
It is not a prediction, it is not a bound on your loss, and it is not out-of-
sample evidence — `demo.py` measures how often this same pipeline passes data
with nothing in it at all.

``--holdout`` is the answer to that, and the only result here worth much: it
finds patterns on the earlier part of your file and tests the survivors on the
later part, which nothing has looked at. You get ONE, because the held-back
half stops being held back the moment you read the result and adjust something.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.backtest import run
from engine.costs import CostModel
from engine.data import load_csv
from engine.holdout import SPLIT_DEFAULT
from engine.levers import advise
from engine.holdout import evaluate as holdout_evaluate
from engine.verdict import (NEGATIVE_NET_EXPECTANCY, NO_WIN_RATE_SAVES_IT,
                            TOO_FEW_TRADES, judge)
from models.patterns import DETECTORS


def analyse(path: str, pair: str, side: str, costs: CostModel, *,
            costs_measured: bool = False, time_format: str = None,
            patterns=None, holdout: float = None, out=print) -> int:
    """Load, check, scan, backtest, judge. Returns a process exit code.

    With ``holdout`` set, the file is cut in time instead: patterns are found
    on the earlier part and the survivors are tested on the later part, which
    nothing has looked at. That is the only test here whose result means much,
    and it can be used once — see `engine/holdout.py`.
    """
    series = load_csv(path, pair, side=side, time_format=time_format)
    out(series.report())
    out("")

    if not series.usable:
        out("STOPPED. The defects above break assumptions the rest of the engine "
            "rests on, so\nanything it computed would be arithmetic on data that "
            "is not what it claims to be.\nNothing was repaired for you: what to "
            "do about them is a decision with consequences.")
        return 2
    if series.n < 200:
        out(f"NOTE: {series.n} bars is a very short history. Expect "
            f"TOO_FEW_TRADES on most patterns.")

    try:
        bpn = series.bars_per_night()
        out(f"financing: {bpn:g} bars per rollover, from the measured interval")
    except ValueError as e:
        bpn = 1.0
        out(f"financing: assuming {bpn:g} bar per night — {e}")
    out(f"costs:     {costs.round_turn_pips} pip round turn, "
        f"{costs.swap_markup_annual:.2%}/yr markup"
        f"{'' if costs_measured else '  (NOT confirmed — see --costs-measured)'}")
    out("")

    if holdout is not None:
        out(holdout_evaluate(series.bars, costs, frac=holdout,
                             bars_per_night=bpn,
                             costs_confirmed=costs_measured,
                             patterns=patterns).report())
        return 0

    chosen = DETECTORS if not patterns else {p: DETECTORS[p] for p in patterns}
    # every detector run over one file is one search, and the correction has to
    # know how wide it was; the four declared knobs in patterns.py are the rest
    n_hyp = len(chosen) * 4
    out(f"search:    {n_hyp} hypotheses ({len(chosen)} pattern(s) x 4 declared "
        f"parameters) — a FLOOR,\n           since the honest count also "
        f"includes every variant you tried and dropped")
    out("")
    out(f"{'pattern':<20} {'n':>6} {'win%':>6} {'need%':>6} {'pips/trade':>11}"
        f"  verdict")
    passed, judged = [], {}
    for name, fn in chosen.items():
        res = run(series.bars, fn(series.bars), costs, bars_per_night=bpn)
        v = judge(res, costs, name=name, n_hypotheses=n_hyp,
                  costs_confirmed=costs_measured)
        head = "TRADEABLE" if v.tradeable else (v.codes[0] if v.codes else "-")
        out(f"{name:<20} {v.n_trades:>6,} {v.gross.win_rate:>6.1%} "
            f"{v.win_rate_needed:>6.1%} {v.net.net:>11.2f}  {head}")
        judged[name] = v
        if v.tradeable:
            passed.append(v)
    out("")

    for v in passed:
        out(v.summary())
        out("")

    # A refusal without a next step leaves you holding a named no and nothing
    # to do with it. But the levers only answer ONE kind of no -- the one where
    # the arithmetic does not work -- and running them on a rule refused for
    # TOO_FEW_TRADES prints "it already clears its costs" directly under a table
    # saying it was refused. That reads as a contradiction because it is one:
    # a sample-size refusal is answered by more data, which stats.py sizes, not
    # by a cheaper broker. So only arithmetic refusals get the lever treatment.
    refused = [v for v in judged.values() if not v.tradeable and v.n_trades]
    # TOO_FEW_TRADES disqualifies a rule from this section whatever ELSE it was
    # refused for. Refusals accumulate, so a rule with one trade can carry both
    # "too few trades" and "negative expectancy" -- and running the levers on it
    # announces "your rule has no edge before costs, find a different one" on
    # the strength of a single trade. Every number here is derived from the
    # expectancy, and a sample too small to measure an expectancy is too small
    # to advise on it.
    arithmetic = [v for v in refused
                  if TOO_FEW_TRADES not in v.codes
                  and (NEGATIVE_NET_EXPECTANCY in v.codes
                       or NO_WIN_RATE_SAVES_IT in v.codes)]
    if not passed and arithmetic:
        best = max(arithmetic, key=lambda v: v.gross.net)
        # trades per year from the sample's own span, not a round guess
        days = max(len(series.bars) / bpn, 1.0)
        out(f"CLOSEST TO WORKING — {best.name}, and what you could change:")
        out(advise(best.gross, trades_per_year=best.n_trades * 365.0 / days))
        out("")
    elif not passed and refused:
        out("No rule here failed on ARITHMETIC — every refusal above is about "
            "sample size or\nsignificance. The binding constraint is data, "
            "not your broker or your targets, and\nthe refusals say how many "
            "trades each claim would need. A cheaper account changes\nnothing "
            "about a number nobody can yet measure.")
        out("")
    if passed:
        out(f"{len(passed)} pattern(s) survived one IN-SAMPLE test on this file. "
            f"That is a hypothesis,\nnot a finding. Split the file and re-run on "
            f"the half this never saw; demo.py shows\nhow often this same "
            f"pipeline passes data with nothing in it at all.")
    else:
        out("Nothing survived. That is the usual outcome and it is a result: it "
            "cost you an\nafternoon instead of an account.")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Run the FX engine over a CSV of OHLC bars.")
    p.add_argument("path", help="CSV file of OHLC bars")
    p.add_argument("--pair", required=True,
                   help="e.g. EURUSD. Decides pip size; JPY crosses differ 100x")
    p.add_argument("--side", required=True, choices=("bid", "ask", "mid"),
                   help="which side of the book the prices are; not guessable")
    p.add_argument("--round-turn", type=float, default=1.5,
                   help="ALL-IN cost of getting in and out once, in pips: "
                        "spread + commission + expected slippage")
    p.add_argument("--swap-markup", type=float, default=0.008,
                   help="broker markup over tom-next, per year, as a fraction")
    p.add_argument("--carry", type=float, default=0.0,
                   help="net yield differential you expect to earn (+) or pay (-)")
    p.add_argument("--costs-measured", action="store_true",
                   help="confirm you MEASURED the two above rather than "
                        "accepting the defaults")
    p.add_argument("--time-format",
                   help="strptime layout, needed only when the date order is "
                        "genuinely ambiguous")
    p.add_argument("--pattern", action="append", choices=sorted(DETECTORS),
                   help="restrict to one pattern; repeatable")
    p.add_argument("--holdout", nargs="?", type=float, const=SPLIT_DEFAULT,
                   metavar="FRAC",
                   help=f"find patterns on the first FRAC of the file and test "
                        f"the survivors on the rest, which nothing has looked "
                        f"at (default {SPLIT_DEFAULT}). You get ONE of these")
    a = p.parse_args(argv)

    costs = CostModel(pair=a.pair, round_turn_pips=a.round_turn,
                      swap_markup_annual=a.swap_markup, carry_annual=a.carry)
    try:
        return analyse(a.path, a.pair, a.side, costs,
                       costs_measured=a.costs_measured,
                       time_format=a.time_format, patterns=a.pattern,
                       holdout=a.holdout)
    except (OSError, ValueError) as e:
        print(f"could not analyse {a.path}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
