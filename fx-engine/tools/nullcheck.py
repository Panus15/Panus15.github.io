"""The test that asks whether YOUR file's result could be your file's noise.

`calibrate.py` measures this engine against data whose truth is known by
construction. This measures it against one specific file — yours — by building
nulls out of that file's own moves and counting how often a result as good as
the one you got shows up in them.

    python3 -m tools.nullcheck data.csv --pair USDJPY --side bid --pattern double_top

WHY A RANDOM WALK IS NOT A GOOD ENOUGH NULL. A Gaussian walk with constant
volatility is too easy to beat. On the USDJPY M1 file this tool was written for
it produced a median of 13 double_top detections where the real file produced
33 — so the detector appeared to be finding something, when the gap could
equally have been volatility clustering, fat tails or the session pattern, none
of which a constant-sigma walk has. A null that does not look like your data
cannot tell you anything about your data.

So this resamples your file's own bar-to-bar moves in contiguous BLOCKS. Fat
tails, clustering and the shape of each bar survive; the relationship between
one block and the next does not.

BLOCK LENGTH IS THE WHOLE ASSUMPTION, AND IT IS REPORTED, NOT CHOSEN FOR YOU.
A block longer than a pattern plus its outcome preserves the patterns too, so
such a null is partly not a null — it reshuffles your days rather than
destroying their structure. Patterns here span `MIN_SPAN`..`MAX_SPAN` bars and
are held for up to `MAX_HOLD_BARS`, so:

    block << pattern span      the pattern cannot form: a strict null
    block ~ pattern span       the pattern forms, its outcome is cut off
    block >> pattern + hold    pattern AND outcome survive: NOT a null

The sweep runs several lengths precisely so you can see where your result stops
being distinguishable from a reshuffle. If it already is at the shortest block,
the pass rate is your false-positive risk. If it only becomes so at the longest,
read it the other way: your result does not depend on the ORDER of your days,
only on which days were in the file — which is a statement about a handful of
sessions, not about the pair.

WHAT THIS CANNOT DO. It cannot confirm an edge. A low pass rate at a strict
block length means the result is not THIS artifact; there are others, and
`holdout.py` is still the only test here whose result means much. It also
inherits every refusal in `verdict.py`: if your sample cannot clear the trade
floor, neither can the nulls, and a 0% pass rate then says only that the floor
refused all of them — which is reported rather than left to be misread.
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys

sys.path.insert(0, __file__.rsplit("/", 2)[0])

from engine.backtest import MAX_HOLD_BARS, run
from engine.costs import CostModel
from engine.data import load_csv
from engine.verdict import MIN_TRADES, TOO_FEW_TRADES, judge
from models.patterns import DETECTORS, MAX_SPAN, MIN_SPAN, PARAMETER_COUNT, Bar

#: Block lengths swept by default, in bars. Chosen against the pattern scale
#: rather than the clock: well inside one pattern, around one pattern, around a
#: pattern plus its hold, and far beyond both.
DEFAULT_BLOCKS = (30, 240, 1440)

#: Resamples per block length. 30 puts a 27% rate inside roughly +/-16 points
#: at 95%, which is wide — the figure separates "routine" from "rare", and is
#: not a precise rate. `--trials` raises it.
DEFAULT_TRIALS = 30


def moves_of(bars: list) -> list:
    """Each bar as (close-to-close move, upper wick, lower wick).

    The wicks travel with the move so a resampled bar keeps a plausible range.
    Rebuilding bars from the close alone would give every bar a zero high and
    low relative to its body, and the detectors read highs and lows.
    """
    return [(bars[i].c - bars[i - 1].c,
             bars[i].h - max(bars[i].o, bars[i].c),
             min(bars[i].o, bars[i].c) - bars[i].l)
            for i in range(1, len(bars))]


def resample(moves: list, block: int, start_px: float, rnd) -> list:
    """A series of the same length, built from contiguous blocks of `moves`."""
    if block < 1:
        raise ValueError(f"block must be at least 1 bar, got {block}")
    if not moves:
        return []
    # No clamp to len(moves) here, deliberately. It would be dead code: for a
    # block at least as long as the file, `moves[j:j + block]` returns the
    # whole list and the loop below runs exactly once, so clamped and
    # unclamped produce the same exact copy of the file. Its mutant survived
    # every test because nothing it could change is observable.
    # The number of blocks is COMPUTED, not looped until long enough. A
    # `while len(seq) < len(moves)` loop makes no progress when a block comes
    # out empty, so it HANGS rather than failing -- and a hang cannot be caught
    # by a test, because the test hangs with it. That is not hypothetical: it
    # cost two nine-minute mutation sweeps here, each killed by a timeout that
    # then left a live mutant in the tree. Dividing by the block length makes
    # the degenerate case raise on the spot instead.
    seq = []
    for _ in range(-(-len(moves) // block)):
        j = rnd.randrange(0, max(len(moves) - block, 0) + 1)
        seq.extend(moves[j:j + block])
    px, out = start_px, []
    for d, up, dn in seq[:len(moves)]:
        o = px
        c = px + d
        out.append(Bar(o, max(o, c) + up, min(o, c) - dn, c))
        px = c
    return out


def sweep(bars: list, costs: CostModel, *, pattern: str, blocks=DEFAULT_BLOCKS,
          trials: int = DEFAULT_TRIALS, bars_per_night: float = 1.0,
          n_hypotheses: int = None, seed: int = 7000, detector=None) -> list:
    """Run `pattern` over resamples at each block length.

    Returns one row per block: (block, passed, trials, refused_for_floor,
    median_trades, best_net_pips).
    """
    fn = detector or DETECTORS[pattern]
    base = moves_of(bars)
    start = bars[0].c if bars else 0.0
    rows = []
    for block in blocks:
        passed = floored = 0
        ns, bests = [], []
        for s in range(trials):
            rnd = random.Random(seed + s)
            sim = resample(base, block, start, rnd)
            res = run(sim, fn(sim), costs, bars_per_night=bars_per_night)
            v = judge(res, costs, name=pattern, n_hypotheses=n_hypotheses,
                      costs_confirmed=True)
            ns.append(res.n)
            bests.append(v.net.net)
            if v.tradeable:
                passed += 1
            if TOO_FEW_TRADES in v.codes:
                floored += 1
        rows.append((block, passed, trials, floored,
                     statistics.median(ns) if ns else 0.0,
                     max(bests) if bests else 0.0))
    return rows


def report(rows: list, real_n: int, real_net: float, pattern: str) -> str:
    """The sweep, with the reading of it that the numbers support."""
    out = [f"NULL CHECK: {pattern} against resamples of your own file",
           f"  your file: {real_n} trades at {real_net:+.2f} pips/trade",
           "",
           f"  {'block':>6}  {'passed':>12}  {'refused: floor':>14}  "
           f"{'trades (med)':>12}  {'best net':>9}",
           f"  {'-'*6}  {'-'*12}  {'-'*14}  {'-'*12}  {'-'*9}"]
    for block, passed, trials, floored, med, best in rows:
        out.append(f"  {block:>6}  {passed:>4}/{trials:<3} {passed/trials:>4.0%}"
                   f"  {floored:>9}/{trials:<4}  {med:>12.0f}  {best:>+9.2f}")
    out.append("")

    strict = rows[0]
    loose = rows[-1]
    # A null whose detection count is nowhere near the real one is not a
    # comparable null, whatever its pass rate. Said before any rate is read.
    comparable = [r for r in rows if real_n and 0.5 <= r[4] / real_n <= 2.0]
    if not comparable:
        out.append(f"  NO COMPARABLE NULL. None of these block lengths "
                   f"reproduces your {real_n} trades, so none of the rates "
                   f"above is a rate for a sample like yours. Widen --blocks.")
        return "\n".join(out)

    # The WORST comparable block is reported, not the average of them. This is
    # a risk figure, and averaging a strict null with a permissive one would
    # report neither.
    worst = max(comparable, key=lambda r: r[1])
    others = [r for r in comparable if r is not worst]
    out.append(f"  {len(comparable)} of these block lengths reproduce your "
               f"{real_n} trades closely enough to compare "
               f"({', '.join(str(r[0]) for r in comparable)}). The WORST of "
               f"them is reported, because this is a risk figure.")
    out.append(f"  At block={worst[0]} the nulls find {worst[4]:.0f} trades "
               f"against your {real_n}, and {worst[1]}/{worst[2]} of them "
               f"({worst[1]/worst[2]:.0%}) pass as TRADEABLE."
               + (f" The others: "
                  + ", ".join(f"{r[1]/r[2]:.0%} at {r[0]}" for r in others)
                  + "." if others else ""))
    if worst[1] / worst[2] >= 0.1:
        out.append(f"  That is your result's false-positive risk AT THAT BLOCK "
                   f"LENGTH: roughly {worst[1]/worst[2]:.0%} of reshuffles of "
                   f"your own data look this good.")
        if worst[0] > MAX_SPAN + MAX_HOLD_BARS:
            out.append(f"  Read the caveat. {worst[0]} bars exceeds a pattern "
                       f"({MIN_SPAN}-{MAX_SPAN}) plus its hold "
                       f"({MAX_HOLD_BARS}), so these blocks keep whole "
                       f"patterns AND their outcomes. The reshuffle changes "
                       f"the ORDER of your sessions, not their content — so "
                       f"what this says is that your result does not need that "
                       f"order. It rests on which sessions were in the file.")
    else:
        out.append(f"  Below one in ten, so this particular artifact does not "
                   f"account for your result. It is not evidence the edge is "
                   f"real: other artifacts exist, and the hold-out is still "
                   f"the test that matters.")

    if strict[3] == strict[2]:
        out.append(f"  Note: at block={strict[0]} every one of the "
                   f"{strict[2]} nulls was refused for TOO_FEW_TRADES "
                   f"(under {MIN_TRADES}), so that 0% is the trade floor "
                   f"refusing them, not merit being assessed.")
    if loose[1] > strict[1]:
        out.append(f"  The rate RISES with block length "
                   f"({strict[1]/strict[2]:.0%} at {strict[0]} -> "
                   f"{loose[1]/loose[2]:.0%} at {loose[0]}), which is what it "
                   f"should do: longer blocks keep more of your file intact. "
                   f"The shortest block is the strict null; the longest is "
                   f"closer to your file than to noise.")
    return "\n".join(out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="How often a reshuffle of your own file looks as good as "
                    "your result")
    p.add_argument("path")
    p.add_argument("--pair", required=True)
    p.add_argument("--side", required=True, choices=("bid", "ask", "mid"))
    p.add_argument("--pattern", required=True, choices=sorted(DETECTORS))
    p.add_argument("--round-turn", type=float, default=1.5)
    p.add_argument("--swap-markup", type=float, default=0.008)
    p.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    p.add_argument("--blocks", type=int, nargs="+", default=list(DEFAULT_BLOCKS),
                   help="block lengths in bars (default 30 240 1440)")
    p.add_argument("--time-format")
    a = p.parse_args(argv)

    series = load_csv(a.path, a.pair, side=a.side, time_format=a.time_format)
    if not series.usable:
        print(series.report(), file=sys.stderr)
        print("STOPPED: the defects above break the assumptions this rests on.",
              file=sys.stderr)
        return 2
    costs = CostModel(pair=a.pair, round_turn_pips=a.round_turn,
                      swap_markup_annual=a.swap_markup)
    try:
        bpn = series.bars_per_night()
    except ValueError:
        bpn = 1.0
    n_hyp = len(DETECTORS) * PARAMETER_COUNT
    fn = DETECTORS[a.pattern]
    real = run(series.bars, fn(series.bars), costs, bars_per_night=bpn)
    rows = sweep(series.bars, costs, pattern=a.pattern, blocks=sorted(a.blocks),
                 trials=a.trials, bars_per_night=bpn, n_hypotheses=n_hyp)
    print(report(rows, real.n, real.expectancy().net, a.pattern))
    return 0


if __name__ == "__main__":
    sys.exit(main())
