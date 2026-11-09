"""Tests for the null check, written against the ways a null can flatter a file.

A tool that answers "could this be noise?" fails in two directions. It can
build a null too easy to beat, in which case every result looks real and the
tool is worse than nothing — that is the actual history here, a constant-sigma
walk that found 13 detections where the file found 33. Or it can build a null
so faithful it reproduces the result by construction, and call a real edge an
artifact.

The resampler is therefore tested on what it PRESERVES and what it DESTROYS,
and the report is tested on the readings it must refuse to give: a rate from
nulls that are not comparable to the file, and a 0% that is only the trade
floor refusing everything.

Run: python3 tests/test_nullcheck.py
"""

import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.backtest import MAX_HOLD_BARS
from models.patterns import MAX_SPAN, Bar, Detection
from tools.nullcheck import (DEFAULT_BLOCKS, moves_of, report, resample, sweep)
from engine.costs import CostModel


def _walk(n=600, sigma=0.01, seed=1):
    rnd = random.Random(seed)
    px, bars = 150.0, []
    for _ in range(n):
        o = px
        c = px + rnd.gauss(0.0, sigma)
        bars.append(Bar(o, max(o, c) + 0.004, min(o, c) - 0.003, c))
        px = c
    return bars


def test_a_resample_is_the_same_length_as_the_file():
    bars = _walk(600)
    sim = resample(moves_of(bars), 50, bars[0].c, random.Random(0))
    assert len(sim) == len(bars) - 1, (
        f"{len(sim)} bars from {len(bars)}: a null of a different length is "
        f"not comparable to the file")


def test_a_resample_keeps_the_move_distribution_it_was_built_from():
    # The whole point of resampling rather than simulating: fat tails and
    # volatility clustering survive because the moves are the FILE's moves. If
    # the standard deviation drifted, the null would be an easier or harder
    # market than the one being tested.
    bars = _walk(4000, sigma=0.02)
    base = moves_of(bars)
    sim = resample(base, 100, bars[0].c, random.Random(3))
    got = statistics.stdev(m[0] for m in moves_of(sim))
    want = statistics.stdev(m[0] for m in base)
    assert abs(got - want) / want < 0.10, (
        f"per-bar sd {got:.5f} against the file's {want:.5f}: the null is not "
        f"the same market")


def test_every_bar_of_a_resample_is_a_valid_bar():
    sim = resample(moves_of(_walk(800)), 40, 150.0, random.Random(5))
    for i, b in enumerate(sim):
        assert b.h >= max(b.o, b.c) and b.l <= min(b.o, b.c) and b.h >= b.l, (
            f"bar {i} is malformed: {b}")


def test_a_move_is_close_to_close_and_a_wick_is_measured_from_the_body():
    # Literal values, because every way of getting this wrong produces a
    # plausible-looking series. Open-to-open has almost the same standard
    # deviation as close-to-close, and wicks of zero still make valid bars --
    # so neither a distribution check nor a validity check can see them.
    bars = [Bar(100.0, 101.0, 99.0, 100.5),
            Bar(102.0, 104.0, 101.5, 103.0)]
    (d, up, dn), = moves_of(bars)
    assert abs(d - 2.5) < 1e-12, f"move {d}, expected 103.0 - 100.5 = 2.5"
    assert abs(up - 1.0) < 1e-12, f"upper wick {up}, expected 104.0 - 103.0"
    assert abs(dn - 0.5) < 1e-12, f"lower wick {dn}, expected 102.0 - 101.5"


def test_a_resample_keeps_the_bar_RANGES_not_only_the_closes():
    # Bars rebuilt from closes alone have h == max(o, c) and l == min(o, c):
    # bodies with no wicks. They are still VALID bars, which is why the check
    # above cannot catch it -- `h >= max(o, c)` is satisfied by equality, the
    # exact broken version. The swing detectors read highs and lows, so a
    # wickless null is a different market.
    bars = _walk(2000, sigma=0.02)
    sim = resample(moves_of(bars), 100, bars[0].c, random.Random(4))
    want = statistics.fmean(b.h - b.l for b in bars)
    got = statistics.fmean(b.h - b.l for b in sim)
    assert abs(got - want) / want < 0.15, (
        f"mean bar range {got:.5f} against the file's {want:.5f}")
    wicked = sum(1 for b in sim if b.h > max(b.o, b.c) + 1e-12)
    assert wicked > len(sim) * 0.9, (
        f"only {wicked} of {len(sim)} bars have an upper wick at all: the "
        f"bars were rebuilt from closes")


def test_a_short_block_destroys_structure_a_long_block_preserves_it():
    # The property the whole tool rests on. A deterministic saw-tooth has
    # perfect 100-bar structure; resampled in 10-bar blocks the period is
    # broken, in 100-bar blocks it survives. Measured as the autocorrelation
    # of the moves at the saw's own lag.
    bars, px = [], 150.0
    for i in range(3000):
        o = px
        c = px + (0.05 if (i // 50) % 2 == 0 else -0.05)
        bars.append(Bar(o, max(o, c) + 0.001, min(o, c) - 0.001, c))
        px = c
    base = moves_of(bars)

    def autocorr(ms, lag):
        xs = [m[0] for m in ms]
        mu = statistics.fmean(xs)
        num = sum((xs[i] - mu) * (xs[i + lag] - mu)
                  for i in range(len(xs) - lag))
        den = sum((x - mu) ** 2 for x in xs)
        return num / den

    short = autocorr(moves_of(resample(base, 10, 150.0, random.Random(1))), 100)
    long_ = autocorr(moves_of(resample(base, 400, 150.0, random.Random(1))), 100)
    assert long_ > 0.5, f"a 400-bar block destroyed 100-bar structure: {long_:.2f}"
    assert short < long_ - 0.2, (
        f"a 10-bar block preserved as much structure as a 400-bar one "
        f"({short:.2f} vs {long_:.2f}) — the sweep would mean nothing")


def test_the_default_blocks_straddle_the_pattern_scale():
    # The sweep is only informative if it spans both sides of "a block long
    # enough to contain a pattern and its outcome". Asserted against the
    # pattern constants rather than against the literals, so moving MAX_SPAN
    # or MAX_HOLD_BARS without revisiting the blocks fails here.
    assert min(DEFAULT_BLOCKS) < MAX_SPAN, (
        f"the strictest block {min(DEFAULT_BLOCKS)} is not shorter than a "
        f"pattern ({MAX_SPAN}), so no block is a strict null")
    assert max(DEFAULT_BLOCKS) > MAX_SPAN + MAX_HOLD_BARS, (
        f"the loosest block {max(DEFAULT_BLOCKS)} does not exceed a pattern "
        f"plus its hold ({MAX_SPAN + MAX_HOLD_BARS}), so the sweep never "
        f"reaches the case that is not a null")


def test_a_block_shorter_than_one_bar_is_refused():
    try:
        resample(moves_of(_walk(100)), 0, 150.0, random.Random(0))
    except ValueError as e:
        assert "at least 1" in str(e), e
    else:
        assert False, "a zero-length block was accepted"


def _rising(n=1200, step=0.05):
    """A series that only ever goes up, so a long's outcome is not a coin toss.

    Resampling preserves it: every move in the file is +step, so every move in
    any block of them is too. That is what makes a planted long's result
    KNOWN after resampling, which is what the bookkeeping tests need.
    """
    px, bars = 150.0, []
    for _ in range(n):
        o = px
        c = px + step
        bars.append(Bar(o, c + 0.01, o - 0.01, c))
        px = c
    return bars


def _stub(n):
    """A detector that plants exactly `n` winning trades on a rising series.

    The real detectors make the sweep's bookkeeping untestable: how many
    trades a resample yields is whatever the data gives, so every count the
    sweep reports is consistent with every way of mis-wiring it. With the
    count AND the outcome fixed, each field has one right answer.

    The target is 20 pips out against a 1 pip cost. A target inside the cost
    would make every "winner" a net loss, which is how the first draft of
    these tests reported 0 passes out of 3 and looked like a bookkeeping bug.
    """
    def detect(bars):
        out = []
        for k in range(n):
            i = 5 + k * 12
            if i + 2 >= len(bars):
                break
            px = bars[i].c
            out.append(Detection(name="double_top", index=i, direction=1,
                                 entry=px, stop=px - 1.00, target=px + 0.20))
        return out
    return detect


def test_the_sweep_reports_one_row_per_block_in_the_order_given():
    bars = _walk(1200, seed=9)
    costs = CostModel(pair="USDJPY", round_turn_pips=1.0,
                      swap_markup_annual=0.006)
    rows = sweep(bars, costs, pattern="double_top", blocks=(20, 300),
                 trials=2, bars_per_night=1440.0, n_hypotheses=24)
    assert [r[0] for r in rows] == [20, 300], rows
    for block, passed, trials, floored, med, _best in rows:
        assert trials == 2 and 0 <= passed <= trials, (block, passed, trials)
        assert 0 <= floored <= trials, floored
        assert med >= 0, med


def test_a_pass_is_counted_as_a_pass_and_a_floor_refusal_as_a_floor_refusal():
    # 60 planted winners: comfortably over the 30-trade floor and plainly
    # tradeable, so passed must be ALL the trials and floored none of them.
    # Swapping the two, or counting refusals as passes, has a different answer
    # here -- which it does not when the trade count is left to the data.
    bars = _rising(1200)
    costs = CostModel(pair="USDJPY", round_turn_pips=1.0,
                      swap_markup_annual=0.006)
    (block, passed, trials, floored, med, best), = sweep(
        bars, costs, pattern="double_top", blocks=(300,), trials=3,
        bars_per_night=1440.0, n_hypotheses=4, detector=_stub(60))
    assert med == 60, f"the stub plants 60 trades, the sweep reported {med}"
    assert passed == trials == 3, f"{passed}/{trials} passed, expected 3/3"
    assert floored == 0, f"{floored} refused for the trade floor, expected 0"
    assert best > 0, best


def test_a_sample_under_the_floor_is_counted_as_floored_and_not_as_a_pass():
    bars = _rising(1200)
    costs = CostModel(pair="USDJPY", round_turn_pips=1.0,
                      swap_markup_annual=0.006)
    (_b, passed, trials, floored, med, _best), = sweep(
        bars, costs, pattern="double_top", blocks=(300,), trials=3,
        bars_per_night=1440.0, n_hypotheses=4, detector=_stub(5))
    assert med == 5, med
    assert floored == trials == 3, f"{floored}/{trials} floored, expected 3/3"
    assert passed == 0, f"{passed} passed on 5 trades, expected 0"


def test_the_nulls_are_judged_with_costs_CONFIRMED():
    # The nulls are run at the cost model the caller is testing, so refusing
    # them for COST_NOT_MEASURED would make every pass rate zero and the whole
    # sweep read as "no false positives". The cost model here is all defaults,
    # which is the only case where the flag changes anything.
    bars = _rising(1200)
    (_b, passed, trials, _f, _m, _best), = sweep(
        bars, CostModel(pair="USDJPY"), pattern="double_top", blocks=(300,),
        trials=2, bars_per_night=1440.0, n_hypotheses=4, detector=_stub(60))
    assert passed == trials == 2, (
        f"{passed}/{trials}: the nulls were refused for unmeasured costs, so "
        f"every rate this tool reports would be zero")


def test_the_trade_count_reported_is_the_median_not_the_maximum():
    # One trial in three finds far more trades than the others. A tool that
    # reported the maximum would overstate how comparable its nulls are to the
    # file, which is the judgement the whole report hangs on.
    bars = _rising(1200)
    costs = CostModel(pair="USDJPY", round_turn_pips=1.0,
                      swap_markup_annual=0.006)
    counts = iter([40, 40, 95])
    planted = []

    def varying(bs):
        n = next(counts)
        planted.append(n)
        return _stub(n)(bs)

    (_b, _p, _t, _f, med, _best), = sweep(
        bars, costs, pattern="double_top", blocks=(300,), trials=3,
        bars_per_night=1440.0, n_hypotheses=4, detector=varying)
    assert planted == [40, 40, 95], planted
    assert med == 40, f"reported {med}: that is the maximum, not the median"


def test_every_trial_is_a_different_draw():
    # Seeded per trial, not once: `Random(seed)` for all of them would run the
    # same resample `trials` times and report a rate of 0% or 100% with no
    # sampling in between.
    bars = _walk(3000, seed=13, sigma=0.03)
    costs = CostModel(pair="USDJPY", round_turn_pips=1.0,
                      swap_markup_annual=0.006)
    seen = []

    def spy(bs):
        seen.append(tuple(round(b.c, 6) for b in bs[:20]))
        return []

    sweep(bars, costs, pattern="double_top", blocks=(50,), trials=4,
          bars_per_night=1440.0, n_hypotheses=4, detector=spy)
    assert len(set(seen)) == 4, (
        f"{len(set(seen))} distinct resamples out of 4: the trials are not "
        f"independent draws")


def test_a_block_longer_than_the_file_is_clamped_to_it():
    # Unclamped, `moves[j:j + block]` still returns the whole list, so the
    # series comes out the right length and the bug is invisible there. What
    # changes is that a block longer than the file can only ever be the file
    # itself: every "resample" is then an exact copy, and the tool would be
    # comparing the file against itself.
    bars = _walk(400, seed=17)
    base = moves_of(bars)
    a = resample(base, len(base) * 3, bars[0].c, random.Random(1))
    b = resample(base, len(base) * 3, bars[0].c, random.Random(2))
    assert len(a) == len(base) and len(b) == len(base), (len(a), len(b))
    ca = [round(x.c, 9) for x in a]
    cb = [round(x.c, 9) for x in b]
    orig = [round(x.c, 9) for x in bars[1:]]
    assert ca == cb == orig, (
        "a block longer than the file should clamp to the file, making the "
        "resample an exact copy — it is not")


def test_a_rate_from_nulls_that_look_nothing_like_the_file_is_refused():
    # 4 trades in the nulls against 300 in the file is not a 0% false-positive
    # rate, it is a null that never attempted the question. Quoting a rate off
    # it would be the most flattering possible misreading.
    rows = [(30, 0, 30, 30, 4.0, 1.0), (240, 0, 30, 30, 5.0, 1.0)]
    text = report(rows, 300, 12.0, "double_top")
    assert "NO COMPARABLE NULL" in text, text
    assert "0%" not in text.split("NO COMPARABLE NULL")[1], (
        "a rate was quoted anyway after refusing to quote one")


def test_a_floor_only_zero_is_not_reported_as_a_clean_result():
    # Every null refused for TOO_FEW_TRADES: the 0% is the trade floor, not
    # merit. A reader told "0% of nulls passed" and not told why would draw
    # the opposite conclusion from the right one.
    rows = [(30, 0, 30, 30, 31.0, 1.0)]
    text = report(rows, 33, 17.5, "double_top")
    assert "trade floor refusing them" in text, text


def test_the_caveat_appears_when_the_block_is_longer_than_pattern_plus_hold():
    rows = [(MAX_SPAN + MAX_HOLD_BARS + 10, 9, 30, 0, 34.0, 20.0)]
    text = report(rows, 33, 17.5, "double_top")
    assert "keep whole patterns AND their outcomes" in text, text
    assert "does not need that order" in text, text


def test_a_low_rate_is_not_reported_as_evidence_the_edge_is_real():
    rows = [(30, 1, 30, 0, 34.0, 5.0)]
    text = report(rows, 33, 17.5, "double_top")
    assert "not evidence the edge is real" in text, text
    assert "hold-out is still" in text, text


def test_the_worst_comparable_block_is_the_one_quoted():
    rows = [(30, 1, 30, 0, 34.0, 5.0), (1440, 9, 30, 0, 36.0, 21.0)]
    text = report(rows, 33, 17.5, "double_top")
    assert "block=1440" in text, text
    assert "WORST of them is reported" in text, text
    assert "30%" in text, f"the worst rate 9/30 was not quoted: {text}"


def _run_all():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
