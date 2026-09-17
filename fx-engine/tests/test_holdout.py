"""Tests for the hold-out — the step whose whole value is that it can say no.

Three things are pinned, because each can be weakened without anything failing:

  1. THE CUT IS IN TIME. A shuffled split reports an edge on pure noise, since
     neighbouring bars overlap, the price level leaks the date, and a trade's
     outcome lives in the bars right after it. The test asserts the halves are
     contiguous slices, not just the right sizes.
  2. ONLY SURVIVORS ARE RE-TESTED. Testing everything out-of-sample and
     reporting the best puts the multiple-testing problem back into the step
     that exists to escape it.
  3. THE DENOMINATOR CHANGES. Out-of-sample you are testing the survivors, not
     the whole search, so the correction is over the survivor count.

Both directions are exercised: a trending series that should CONFIRM, and a
random walk that should not. A validator that only ever says no is worth as
little as one that only ever says yes.

Run: python3 tests/test_holdout.py
"""

import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.costs import CostModel
from engine.holdout import (MIN_HOLDOUT_BARS, SPLIT_DEFAULT, Holdout,
                            HoldoutResult, evaluate, split)
from models.patterns import Bar

COSTS = CostModel(pair="EURUSD", round_turn_pips=1.1, swap_markup_annual=0.008)
HOURLY = 24


def _series(n, seed, drift=0.0, sigma=0.0004):
    rng = random.Random(seed)
    out, px = [], 1.1000
    for _ in range(n):
        o, path = px, [px]
        for _ in range(4):
            px *= math.exp(rng.gauss(drift / 4, sigma))
            path.append(px)
        out.append(Bar(o, max(path), min(path), px))
    return out


# ---------------------------------------------------------------------------
# the cut
# ---------------------------------------------------------------------------

def test_the_split_is_a_single_cut_in_time():
    """Not just the right sizes — the right BARS, in order, with nothing from
    the later half appearing in the earlier one."""
    bars = _series(1000, 1)
    a, b = split(bars, 0.7)
    assert len(a) == 700 and len(b) == 300
    assert a == bars[:700] and b == bars[700:], "the split was not contiguous"
    assert a[-1] is bars[699] and b[0] is bars[700]


def test_the_split_fraction_is_respected_and_validated():
    bars = _series(1000, 1)
    assert len(split(bars, 0.5)[0]) == 500
    assert len(split(bars, 0.9)[0]) == 900
    assert len(split(bars)[0]) == int(1000 * SPLIT_DEFAULT)
    for bad in (0.0, 1.0, -0.1, 1.5):
        try:
            split(bars, bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"frac={bad} was allowed")


# ---------------------------------------------------------------------------
# it can say yes
# ---------------------------------------------------------------------------

def test_a_real_trend_is_confirmed_out_of_sample():
    """The positive control. A validator that only ever refuses is worth as
    little as one that only ever confirms, and the two look identical until
    something that should pass is put through.

    The series needed to get here multiplies the price SEVENFOLD, which no
    currency does. At 12,000 bars and the same drift the out-of-sample half
    produced 63 trades against the ~305 that a +7 pip edge on a 76 pip standard
    deviation needs, so it refused — correctly, and for the reason the refusal
    named. That is the module working, and it is also the measured cost of
    confirming anything: sample length and trend strength trade off, and FX
    supplies too little of both.
    """
    bars = _series(30000, 7, drift=0.00006)
    h = evaluate(bars, COSTS, bars_per_night=HOURLY, costs_confirmed=True)
    passed = h.survivors
    assert passed, "nothing passed in-sample; the control cannot run"
    assert any(r.confirmed for r in h.results), [r.summary() for r in passed]
    for r in h.results:
        if r.confirmed:
            assert r.out_of_sample.n_trades > 0
            assert r.status == "CONFIRMED"


# ---------------------------------------------------------------------------
# it can say no
# ---------------------------------------------------------------------------

def test_a_random_walk_is_not_confirmed():
    bars = _series(12000, 3)
    h = evaluate(bars, COSTS, bars_per_night=HOURLY, costs_confirmed=True)
    assert not any(r.confirmed for r in h.results), [r.summary() for r in h.results]


def test_an_in_sample_pass_that_does_not_survive_is_reported_as_failed():
    """The case the random walk cannot produce, because nothing passes there at
    all — so "mark every survivor confirmed" would go unnoticed. A fifth of the
    trending series is enough to pass in-sample and too little to confirm."""
    bars = _series(12000, 7, drift=0.00006)
    h = evaluate(bars, COSTS, bars_per_night=HOURLY, costs_confirmed=True)
    tested = [r for r in h.results if r.out_of_sample is not None]
    assert tested, "nothing was tested out-of-sample; the fixture is wrong"
    assert not any(r.confirmed for r in h.results), [r.summary() for r in tested]
    for r in tested:
        assert r.status == "FAILED out-of-sample", r.status
        assert "why:" in r.summary(), r.summary()


def test_the_out_of_sample_half_is_the_one_held_back():
    """Testing on the training half would confirm almost anything, and on a
    trending series it looks identical from the outside — except that the trade
    counts would match, because it is the same data."""
    bars = _series(30000, 7, drift=0.00006)
    h = evaluate(bars, COSTS, bars_per_night=HOURLY, costs_confirmed=True)
    tested = [r for r in h.results if r.out_of_sample is not None]
    assert tested
    for r in tested:
        assert r.out_of_sample.n_trades < r.in_sample.n_trades, (
            r.name, r.out_of_sample.n_trades, r.in_sample.n_trades)


def test_patterns_refused_in_sample_are_never_tested_out_of_sample():
    """Testing everything out-of-sample and reporting the best would put the
    multiple-testing problem straight back into the step that escapes it."""
    bars = _series(12000, 3)
    h = evaluate(bars, COSTS, bars_per_night=HOURLY, costs_confirmed=True)
    for r in h.results:
        if not r.in_sample.tradeable:
            assert r.out_of_sample is None, f"{r.name} was tested anyway"
            assert r.status == "refused in-sample"


def test_the_correction_is_over_the_survivors_not_the_whole_search():
    bars = _series(30000, 7, drift=0.00006)
    h = evaluate(bars, COSTS, bars_per_night=HOURLY, costs_confirmed=True)
    survivors = h.survivors
    assert survivors
    assert all(r.in_sample.n_hypotheses == 24 for r in h.results), "in-sample width"
    for r in survivors:
        assert r.out_of_sample.n_hypotheses == len(survivors), (
            r.out_of_sample.n_hypotheses, len(survivors))


# ---------------------------------------------------------------------------
# when it cannot answer
# ---------------------------------------------------------------------------

def test_too_small_a_holdout_is_reported_rather_than_confirmed():
    """A confirmation from 200 bars would be worth less than no answer, and it
    would read exactly the same."""
    bars = _series(30000, 7, drift=0.00006)
    frac = 1.0 - (MIN_HOLDOUT_BARS - 100) / len(bars)
    h = evaluate(bars, COSTS, frac=frac, bars_per_night=HOURLY,
                  costs_confirmed=True)
    assert h.survivors, "the control did not pass"
    assert all(r.out_of_sample is None for r in h.results)
    for r in h.results:
        if r.in_sample.tradeable:
            assert r.status == "not tested out-of-sample", r.status
            assert not r.confirmed
    text = h.report()
    assert "NOT TESTED" in text and "too few to confirm" in text, text


def test_nothing_passing_leaves_the_holdout_untouched():
    bars = _series(12000, 3)
    h = evaluate(bars, COSTS, bars_per_night=HOURLY, costs_confirmed=True)
    text = h.report()
    assert "nothing to test" in text, text
    assert "still untouched" in text
    assert "YOU GET ONE" not in text, "it warned about spending an unspent test"


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------

def test_the_report_states_the_split_it_actually_made():
    """The sizes live on the Holdout rather than being recomputed by whoever
    prints them. Two computations of one fact is how a report ends up
    describing a split that did not happen."""
    bars = _series(30000, 7, drift=0.00006)
    h = evaluate(bars, COSTS, frac=0.6, bars_per_night=HOURLY,
                 costs_confirmed=True)
    assert (h.n_train, h.n_test) == (18000, 12000), (h.n_train, h.n_test)
    assert h.n_train + h.n_test == len(bars)
    assert h.frac == 0.6, h.frac
    text = h.report()
    assert "found on 18,000 bars, tested on 12,000" in text, text
    assert "40% held back" in text, text          # and not the default 30%


def test_the_report_says_the_holdout_can_only_be_used_once():
    bars = _series(30000, 7, drift=0.00006)
    h = evaluate(bars, COSTS, bars_per_night=HOURLY, costs_confirmed=True)
    text = h.report()
    assert "YOU GET ONE" in text, text
    assert "searching a second data set" in text
    assert "hypothes" in text, "the correction denominator was not stated"


def test_a_failure_out_of_sample_is_called_ordinary_not_a_malfunction():
    """Constructed rather than hoped for: an in-sample pass whose out-of-sample
    verdict refuses."""
    class _V:
        """A stand-in shaped like Verdict, with `codes` DERIVED from `refusals`
        exactly as the real class derives it — a stub that lets the two drift
        apart tests a object that does not exist."""
        def __init__(self, ok):
            self.tradeable, self.n_trades, self.n_hypotheses = ok, 50, 1
            self.refusals = [] if ok else ["NEGATIVE_NET_EXPECTANCY: it lost"]
            self.net = type("n", (), {"net": 1.0})()

        @property
        def codes(self):
            return [str(r).split(":")[0] for r in self.refusals]
    h = Holdout(frac=0.7, n_train=8400, n_test=3600,
                results=[HoldoutResult("flag", _V(True), _V(False), False)])
    text = h.report()
    assert "FAILED out-of-sample" in text, text
    assert "ORDINARY outcome" in text and "not a malfunction" in text
    assert "0 of 1" in text


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
