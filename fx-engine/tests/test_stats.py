"""Tests for the sample-size arithmetic — the part that says "you cannot tell yet".

Two things are being pinned. First, that the Wilson interval is genuinely
different from the textbook one at the sample sizes people actually have; a
"Wilson" implementation that happens to reproduce p +/- 1.96*se has silently
reverted to the thing it was chosen to replace. Second, that the sample-size
answers are MINIMAL and not merely sufficient — returning a number ten times too
large would pass every "does it separate?" assertion while making the module
useless.

Expected values here are written as literals, computed by hand from the
published formulae, never derived from the module's own constants.

Run: python3 tests/test_stats.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.stats import (detectable_edge, sample_size_report,
                          trades_needed, trades_to_separate, wilson_interval,
                          years_needed, z_for)


def _normal_interval(wins, n):
    """The textbook interval Wilson is here to replace, for comparison only."""
    p = wins / n
    se = math.sqrt(p * (1.0 - p) / n)
    return (p - 1.959963984540054 * se, p + 1.959963984540054 * se)


# ---------------------------------------------------------------------------
# the interval
# ---------------------------------------------------------------------------

def test_z_matches_the_published_quantiles():
    assert abs(z_for(0.95) - 1.959964) < 1e-5, z_for(0.95)
    assert abs(z_for(0.99) - 2.575829) < 1e-5, z_for(0.99)
    assert z_for(0.99) > z_for(0.95), "more confidence must mean a wider z"


def test_the_textbook_interval_leaves_the_unit_range_and_wilson_does_not():
    """19 wins in 20 is where the approximation stops approximating."""
    lo_n, hi_n = _normal_interval(19, 20)
    assert hi_n > 1.0, f"the comparison case is not extreme enough: {hi_n}"
    lo_w, hi_w = wilson_interval(19, 20)
    assert hi_w <= 1.0, f"Wilson left the unit range: {hi_w}"
    assert abs(hi_w - 0.9913) < 1e-3, hi_w
    assert abs(lo_w - 0.7640) < 1e-3, lo_w
    assert lo_w < lo_n, "Wilson should be the more cautious lower bound here"


def test_wilson_is_not_quietly_the_textbook_interval():
    """At 15/20 the two differ by ~3 points at the bottom and ~5 at the top.
    If this test ever passes with near-equality the implementation has reverted."""
    lo_w, hi_w = wilson_interval(15, 20)
    lo_n, hi_n = _normal_interval(15, 20)
    assert abs(lo_w - 0.5310) < 1e-3, lo_w
    assert abs(hi_w - 0.8881) < 1e-3, hi_w
    assert abs(lo_w - lo_n) > 0.02 and abs(hi_w - hi_n) > 0.02


def test_the_interval_stays_inside_the_unit_range_at_the_extremes():
    """Swept rather than sampled. Wilson is provably inside [0,1], so the clamps
    only ever catch float error — and they DO: at 0 wins in 2 trades the
    unclamped lower bound is -5.6e-17. A handful of hand-picked (wins, n) pairs
    misses that, which is how a clamp ends up untested and then deleted."""
    for n in range(1, 200):
        for wins in (0, n):
            lo, hi = wilson_interval(wins, n)
            assert 0.0 <= lo <= hi <= 1.0, (wins, n, lo, hi)


def test_the_interval_narrows_as_the_sample_grows():
    widths = [wilson_interval(round(0.6 * n), n)[1]
              - wilson_interval(round(0.6 * n), n)[0]
              for n in (20, 100, 1000, 10_000)]
    assert widths == sorted(widths, reverse=True), widths
    assert widths[-1] < 0.02, widths[-1]


def test_more_confidence_buys_a_wider_interval():
    lo95, hi95 = wilson_interval(60, 100, 0.95)
    lo99, hi99 = wilson_interval(60, 100, 0.99)
    assert lo99 < lo95 and hi99 > hi95


def test_the_interval_rejects_impossible_inputs():
    for wins, n in ((6, 5), (-1, 5), (0, 0)):
        try:
            wilson_interval(wins, n)
        except ValueError:
            pass
        else:
            raise AssertionError(f"wilson_interval({wins}, {n}) was allowed")


# ---------------------------------------------------------------------------
# how many trades
# ---------------------------------------------------------------------------

def test_a_claim_that_is_not_true_never_separates():
    assert trades_to_separate(0.50, 0.50) == math.inf
    assert trades_to_separate(0.45, 0.50) == math.inf


def test_the_answer_is_minimal_and_not_merely_sufficient():
    """The returned n must separate and be the FIRST that does. A function
    returning ten times the right answer passes every other test here."""
    n = int(trades_to_separate(0.60, 0.50))
    assert wilson_interval(round(0.60 * n), n)[0] > 0.50, n
    assert wilson_interval(round(0.60 * (n - 1)), n - 1)[0] <= 0.50, n - 1


def test_a_thinner_edge_costs_disproportionately_more_data():
    wide = trades_to_separate(0.60, 0.50)
    mid = trades_to_separate(0.55, 0.50)
    thin = trades_to_separate(0.52, 0.50)
    assert wide < mid < thin, (wide, mid, thin)
    # halving the edge from 10 points to 5 more than QUADRUPLES the requirement
    assert thin / mid > 4.0, thin / mid
    assert abs(wide - 96) < 1e-9 and abs(mid - 389) < 1e-9, (wide, mid)


def test_a_higher_break_even_is_harder_to_clear():
    assert (trades_to_separate(0.60, 0.55) > trades_to_separate(0.60, 0.50))


def test_an_unreachable_requirement_is_reported_rather_than_returned():
    assert trades_to_separate(0.5001, 0.50, cap=1000) == math.inf


# ---------------------------------------------------------------------------
# edge, in pips
# ---------------------------------------------------------------------------

def test_the_data_requirement_is_quadratic_in_the_edge():
    big = trades_needed(1.0, 50.0, 2.0)
    half = trades_needed(0.5, 50.0, 2.0)
    assert abs(half / big - 4.0) < 1e-9, half / big
    # +0.2 pips against a 50-pip standard deviation, at t = 2
    assert abs(trades_needed(0.2, 50.0, 2.0) - 250_000) < 1e-6


def test_no_edge_needs_infinite_data():
    assert trades_needed(0.0, 50.0, 2.0) == math.inf
    assert trades_needed(-1.0, 50.0, 2.0) == math.inf


def test_the_detectable_edge_inverts_the_sample_size():
    n, sd, t = 10_000, 40.0, 2.24
    e = detectable_edge(n, sd, t)
    assert abs(trades_needed(e, sd, t) - n) < 1e-6, (e, trades_needed(e, sd, t))
    assert detectable_edge(40_000, sd, t) < e, "more data must detect less"


def test_years_translate_the_trades_into_something_arguable():
    assert abs(years_needed(1040, 520) - 2.0) < 1e-12
    for bad in (0, -5):
        try:
            years_needed(100, bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"years_needed(100, {bad}) was allowed")


def test_the_report_says_the_years_and_names_its_own_assumption():
    text = sample_size_report(0.55, 0.50, 52)
    assert "389" in text, text
    assert "7.5 years" in text, text
    assert "independent" in text, "the report hid the assumption it rests on"
    dead = sample_size_report(0.48, 0.50, 52)
    assert "never separates" in dead, dead


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
