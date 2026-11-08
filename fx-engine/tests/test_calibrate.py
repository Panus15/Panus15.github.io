"""Tests for the self-calibration — mostly that it does not flatter the engine.

A tool that measures its own accuracy has an obvious failure mode: every
rounding of a doubt in its own favour. The things pinned here are the ones where
that already happened once and had to be fixed:

  - with ZERO false positives reaching the hold-out, `1 - 0/0` reads as "removed
    100% of them". It removed none, because none arrived;
  - the TREND decay sitting next to Hutchinson's 0.09 invites the reading that
    no decay is a pass. There the edge is real, so no decay is CORRECT, and only
    the NOISE row is comparable;
  - a per-rule confirm rate divides by six detectors, most of them reversal
    patterns with no business firing on a trend, so it understates whether the
    engine would have found the thing at all.

Run: python3 tests/test_calibrate.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.calibrate import (BENCHMARKS, NOMINAL_ALPHA, TREND_DRIFT, Condition,
                             main, report, run_condition, series)

SMALL = dict(trials=3, bars_n=6000)


# ---------------------------------------------------------------------------
# the two conditions
# ---------------------------------------------------------------------------

def test_noise_and_trend_are_actually_different_series():
    flat = series(4000, 1, drift=0.0)
    up = series(4000, 1, drift=TREND_DRIFT)
    assert flat[-1].c / flat[0].o < up[-1].c / up[0].o, (
        "the trend condition did not trend")
    assert up[-1].c > up[0].o, "the trend condition fell"
    # same seed, so the difference is the drift and nothing else
    assert flat[0].o == up[0].o


def test_nothing_confirms_on_noise_and_something_does_on_a_trend():
    """Both directions. A filter that refuses everything scores perfectly on
    the first and zero on the second, and they look identical until both run."""
    noise = run_condition("noise", False, seed0=7000, drift=0.0, **SMALL)
    trend = run_condition("trend", True, seed0=9000, drift=TREND_DRIFT, **SMALL)
    assert noise.rules > 0 and trend.rules > 0
    assert noise.confirmed == 0, (
        f"{noise.confirmed} false positives confirmed on noise")
    assert trend.confirmed > 0, "nothing confirmed on a real trend: no power"
    assert trend.found_rate > 0.0


def test_the_per_series_rate_is_not_the_per_rule_rate():
    """Six detectors per series, so the two differ whenever fewer than all six
    fire — which is always. Reporting only the per-rule figure understates
    whether the engine would have found the thing."""
    trend = run_condition("trend", True, seed0=9000, drift=TREND_DRIFT, **SMALL)
    assert trend.rules == trend.trials * 6, (trend.rules, trend.trials)
    assert trend.found_rate > trend.confirm_rate, (
        trend.found_rate, trend.confirm_rate)


# ---------------------------------------------------------------------------
# the arithmetic that flattered the engine
# ---------------------------------------------------------------------------

def test_zero_false_positives_reaching_the_holdout_is_not_a_perfect_score():
    """0/0 is not 100%. The first version credited the hold-out with removing
    every false positive in exactly the case where none arrived."""
    empty = Condition(name="noise", truth=False, trials=5, rules=30)
    assert empty.in_sample_passes == 0
    assert empty.survival == 0.0
    text = report(empty, Condition(name="trend", truth=True, trials=5, rules=30))
    assert "Not measurable at this size" in text, text
    assert "cannot be credited with removing any" in text
    assert "removed 100%" not in text, text


def test_the_removal_credit_appears_once_something_does_reach_it():
    """Isolation the other way: the honest branch must still work."""
    some = Condition(name="noise", truth=False, trials=5, rules=30,
                     in_sample_passes=4, confirmed=1)
    text = report(some, Condition(name="trend", truth=True, trials=5, rules=30))
    assert "removed 75%" in text, text
    assert "Not measurable" not in text


def test_the_decay_comparison_points_at_the_noise_row_only():
    noise = Condition(name="noise", truth=False, trials=5, rules=30,
                      in_sample_passes=2, confirmed=0, decays=[-0.08, -0.10])
    trend = Condition(name="trend", truth=True, trials=5, rules=30,
                      in_sample_passes=6, confirmed=5, decays=[1.3, 1.2])
    text = report(noise, trend)
    assert "The NOISE row is the one that belongs beside" in text, text
    assert "must NOT be compared to it" in text
    assert "no decay is" not in text or "is correct" in text
    assert "0.09" in text, "the published figure was dropped"


def test_a_condition_with_no_passes_reports_no_decay_rather_than_zero():
    c = Condition(name="noise", truth=False, trials=2, rules=12)
    assert math.isnan(c.median_decay)
    text = report(c, Condition(name="trend", truth=True, trials=2, rules=12))
    assert "no in-sample passes to measure" in text


# ---------------------------------------------------------------------------
# the caveat that must not be droppable
# ---------------------------------------------------------------------------

def test_the_power_figure_is_labelled_an_upper_bound_with_its_move():
    trend = Condition(name="trend", truth=True, trials=3, rules=18,
                      in_sample_passes=3, confirmed=3, move=2.02)
    text = report(Condition(name="noise", truth=False, trials=3, rules=18), trend)
    assert "UPPER BOUND" in text
    assert "x2.02" in text, "the price move was not stated"
    assert "EURUSD has" in text and "ever had" in text
    assert "cannot say" in text and "how much lower" in text


def test_every_benchmark_carries_a_source_and_a_bearing():
    assert len(BENCHMARKS) >= 4
    for b in BENCHMARKS:
        assert b.source.strip(), b.label
        assert b.bearing.strip(), b.label
        assert b.value.strip(), b.label
    text = report(Condition(name="noise", truth=False, trials=1, rules=6),
                  Condition(name="trend", truth=True, trials=1, rules=6))
    for b in BENCHMARKS:
        assert b.source in text, f"{b.label} lost its source in the report"


def test_the_report_states_what_a_55_percent_claim_costs_to_establish():
    text = report(Condition(name="noise", truth=False, trials=1, rules=6),
                  Condition(name="trend", truth=True, trials=1, rules=6))
    assert "389 trades" in text, text
    assert "years" in text


def test_the_nominal_alpha_is_the_corrected_one():
    # 0.05 over six detectors x four declared parameters
    assert abs(NOMINAL_ALPHA - 0.05 / 24) < 1e-12, NOMINAL_ALPHA


# ---------------------------------------------------------------------------
# the entry point
# ---------------------------------------------------------------------------

def test_the_cli_refuses_a_sample_too_small_to_mean_anything():
    assert main(["--trials", "0", "--bars", "6000"]) == 2
    assert main(["--trials", "3", "--bars", "10"]) == 2


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
