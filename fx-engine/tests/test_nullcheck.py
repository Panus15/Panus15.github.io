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
from models.patterns import MAX_SPAN, Bar
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
    # Rebuilding from closes alone would give highs inside the body, which the
    # loader calls fatal and the detectors read as fills that cannot happen.
    sim = resample(moves_of(_walk(800)), 40, 150.0, random.Random(5))
    for i, b in enumerate(sim):
        assert b.h >= max(b.o, b.c) and b.l <= min(b.o, b.c) and b.h >= b.l, (
            f"bar {i} is malformed: {b}")


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


def test_a_block_longer_than_the_file_is_clamped_not_an_error():
    base = moves_of(_walk(200))
    sim = resample(base, 10_000, 150.0, random.Random(0))
    assert len(sim) == len(base), len(sim)


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
