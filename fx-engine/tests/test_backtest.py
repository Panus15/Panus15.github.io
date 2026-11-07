"""Tests for the backtest, written against the five ways it could lie.

Each test builds a series where the honest answer is known by construction, so
the assertion is on the ANSWER rather than on the code path:

  1. an ambiguous bar (stop and target both inside it) must resolve to the STOP
     and be counted, because OHLC cannot order them;
  2. a gap through the stop must fill at the OPEN, not the stop level;
  3. financing must be charged for the nights actually held;
  4. entry is the bar AFTER the pattern completed, never the completing bar;
  5. an unresolved trade must be closed and counted, never dropped.

Run: python3 tests/test_backtest.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.backtest import (AMBIGUITY_WARN_FRAC, MAX_HOLD_BARS, run, summary)
from engine.costs import CostModel
from models.patterns import Bar, Detection

FREE = CostModel(pair="EURUSD", round_turn_pips=0.0, swap_markup_annual=0.0)


def _flat(price, n, half=0.0002):
    return [Bar(price, price + half, price - half, price) for _ in range(n)]


def _long(index=0, entry=1.1000, stop=1.0950, target=1.1100):
    return Detection("test", index, +1, entry=entry, stop=stop, target=target)


def _short(index=0, entry=1.1000, stop=1.1050, target=1.0900):
    return Detection("test", index, -1, entry=entry, stop=stop, target=target)


# --------------------------------------------------------------------------
# 1. The ambiguous bar — the commonest way a backtest is inflated
# --------------------------------------------------------------------------

def test_a_bar_holding_both_levels_resolves_to_the_stop_and_is_counted():
    """OHLC cannot say which came first. Assuming the target is how an equity
    curve bends upward with nothing erroring — so the stop is assumed, and the
    count is reported so a reader can see how much of the result rests on it."""
    bars = _flat(1.1000, 2) + [Bar(1.1000, 1.1150, 1.0900, 1.1000)] + _flat(1.1000, 2)
    res = run(bars, [_long(index=0)], FREE)
    assert res.n == 1
    t = res.trades[0]
    assert t.outcome == "stop", "the ambiguous bar was resolved in our favour"
    assert t.ambiguous is True
    assert res.ambiguous_bars == 1
    assert t.net_pips < 0


def test_a_high_share_of_ambiguous_bars_is_warned_about():
    bars = _flat(1.1000, 2) + [Bar(1.1000, 1.1150, 1.0900, 1.1000)] + _flat(1.1000, 2)
    res = run(bars, [_long(index=0)], FREE)
    assert res.ambiguous_frac > AMBIGUITY_WARN_FRAC
    assert any("BOTH" in w for w in res.warnings), res.warnings


def test_an_unambiguous_target_is_still_allowed_to_win():
    """The guard must not make winning impossible — a bar that touches only the
    target resolves to the target."""
    bars = _flat(1.1000, 2) + [Bar(1.1000, 1.1150, 1.0990, 1.1100)] + _flat(1.11, 2)
    res = run(bars, [_long(index=0)], FREE)
    assert res.trades[0].outcome == "target"
    assert res.trades[0].ambiguous is False
    assert res.trades[0].net_pips > 0


# --------------------------------------------------------------------------
# 2. The gap — a stop is an instruction, not a guarantee
# --------------------------------------------------------------------------

def test_a_gap_through_the_stop_fills_at_the_open_not_at_the_stop():
    """The 2015-01-15 mechanism. Filling at the stop level is how a backtest hides
    the risk that actually ends accounts."""
    bars = _flat(1.1000, 2) + [Bar(1.0800, 1.0810, 1.0790, 1.0800)] + _flat(1.08, 2)
    res = run(bars, [_long(index=0, stop=1.0950)], FREE)
    t = res.trades[0]
    assert t.outcome == "stop"
    assert abs(t.exit - 1.0800) < 1e-12, "the fill was not the open"
    assert res.slipped_stops == 1
    assert abs(res.worst_slippage_pips - 150.0) < 0.5, res.worst_slippage_pips
    # the loss is bigger than the stop distance implies
    assert t.pips < -150.0
    assert any("gapped THROUGH" in w for w in res.warnings)


def test_a_short_gapping_up_through_its_stop_is_handled_too():
    bars = _flat(1.1000, 2) + [Bar(1.1200, 1.1210, 1.1190, 1.1200)] + _flat(1.12, 2)
    res = run(bars, [_short(index=0, stop=1.1050)], FREE)
    t = res.trades[0]
    assert t.outcome == "stop" and res.slipped_stops == 1
    assert abs(t.exit - 1.1200) < 1e-12
    assert t.pips < -150.0


# --------------------------------------------------------------------------
# 3. Financing on the nights actually held
# --------------------------------------------------------------------------

def test_financing_is_charged_for_the_nights_actually_held():
    """The cost low turnover does not reduce. Two identical trades, one held ten
    times longer, must not cost the same."""
    costs = CostModel(pair="EURUSD", round_turn_pips=0.0, swap_markup_annual=0.03)
    quick = _flat(1.1000, 2) + [Bar(1.1000, 1.1150, 1.0990, 1.1100)] + _flat(1.11, 2)
    slow = _flat(1.1000, 2) + _flat(1.1000, 40) + [Bar(1.1000, 1.1150, 1.0990, 1.1100)]
    r_quick = run(quick, [_long(index=0)], costs)
    r_slow = run(slow, [_long(index=0)], costs)
    assert r_quick.n == r_slow.n == 1
    assert r_slow.trades[0].cost_pips > r_quick.trades[0].cost_pips * 5, (
        r_quick.trades[0].cost_pips, r_slow.trades[0].cost_pips)
    assert r_slow.trades[0].nights() > r_quick.trades[0].nights()


def test_a_bar_is_not_a_night():
    """The conversion that is a 24x error on hourly data if it is left out.

    The same 40-bar hold is 40 nights of financing on daily bars and one night
    on hourly, because a bar count means nothing until you say how big a bar is.
    """
    costs = CostModel(pair="EURUSD", round_turn_pips=0.0, swap_markup_annual=0.03)
    bars = _flat(1.1000, 2) + _flat(1.1000, 40) + [Bar(1.1000, 1.1150, 1.0990, 1.1100)]
    daily = run(bars, [_long(index=0)], costs).trades[0]
    hourly = run(bars, [_long(index=0)], costs, bars_per_night=24).trades[0]
    assert daily.nights() == 41 and hourly.nights(24) == 1, (
        daily.nights(), hourly.nights(24))
    assert hourly.cost_pips < daily.cost_pips / 20, (
        hourly.cost_pips, daily.cost_pips)


def test_a_hold_that_never_crosses_a_rollover_pays_no_financing():
    """A day trader flat by 5pm New York pays no swap at all, so counting
    fractions of a night would charge for financing that never happened."""
    costs = CostModel(pair="EURUSD", round_turn_pips=0.9, swap_markup_annual=0.03)
    bars = _flat(1.1000, 2) + _flat(1.1000, 3) + [Bar(1.1000, 1.1150, 1.0990, 1.1100)]
    t = run(bars, [_long(index=0)], costs, bars_per_night=24).trades[0]
    assert t.nights(24) == 0, t.nights(24)
    assert abs(t.cost_pips - 0.9) < 1e-12, "financing was charged on an intraday hold"


def test_a_bar_cannot_be_zero_or_negative_nights():
    for bad in (0, -1, -0.5):
        try:
            run(_flat(1.1, 5), [], CostModel(), bars_per_night=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"bars_per_night={bad} was allowed")


def test_the_same_trade_is_worse_with_costs_than_without():
    """And the gap is the round turn PLUS the financing, separately accounted.

    The first version of this test asserted the gap was exactly the round turn,
    forgetting that CostModel carries a default swap markup — so it was asserting
    that financing is NOT charged, which is the opposite of what this engine is
    for. The two components are checked separately now.
    """
    bars = _flat(1.1000, 2) + [Bar(1.1000, 1.1150, 1.0990, 1.1100)] + _flat(1.11, 2)
    model = CostModel(round_turn_pips=2.0)          # default markup applies too
    free = run(bars, [_long(index=0)], FREE).trades[0]
    paid = run(bars, [_long(index=0)], model).trades[0]
    assert abs(free.pips - paid.pips) < 1e-9, "the gross result must not change"
    assert paid.net_pips < free.net_pips

    gap = free.net_pips - paid.net_pips
    financing = model.financing_pips(paid.nights(), paid.entry)
    assert financing > 0, "the default markup should cost something for a night held"
    assert abs(gap - (2.0 + financing)) < 1e-6, (gap, financing)
    # and with the markup switched off the gap is exactly the round turn
    only_spread = run(bars, [_long(index=0)],
                      CostModel(round_turn_pips=2.0,
                                swap_markup_annual=0.0)).trades[0]
    assert abs((free.net_pips - only_spread.net_pips) - 2.0) < 1e-9


# --------------------------------------------------------------------------
# 4. No look-ahead
# --------------------------------------------------------------------------

def test_entry_is_the_bar_after_the_pattern_completed():
    """Entering at the completing bar's close assumes you acted on a bar while it
    was still forming — which you could not have."""
    # the entry bar's OPEN and CLOSE must differ, or the test cannot tell which
    # one was used — the first version had them equal and a mutant swapping .o
    # for .c passed the whole sweep
    bars = [Bar(1.1000, 1.1005, 1.0995, 1.1000),
            Bar(1.1020, 1.1060, 1.1015, 1.1050)] + _flat(1.1050, 4)
    res = run(bars, [_long(index=0, stop=1.0900, target=1.1500)], FREE)
    assert res.n == 1
    assert res.trades[0].entry_index == 1, "the trade started on the detection bar"
    assert bars[1].o != bars[1].c, "the fixture cannot distinguish open from close"
    assert abs(res.trades[0].entry - 1.1020) < 1e-12, (
        f"entry was {res.trades[0].entry}, not the next bar's OPEN (1.1020)")


def test_a_detection_on_the_last_bar_produces_no_trade():
    """There is no next bar to enter on, and inventing one is look-ahead."""
    bars = _flat(1.1000, 4)
    assert run(bars, [_long(index=len(bars) - 1)], FREE).n == 0


def test_nothing_before_the_detection_can_affect_the_result():
    """Prepending history must not change a trade that starts after it."""
    tail = _flat(1.1000, 2) + [Bar(1.1000, 1.1150, 1.0990, 1.1100)] + _flat(1.11, 2)
    a = run(tail, [_long(index=0)], FREE)
    pre = _flat(1.0500, 20)
    b = run(pre + tail, [_long(index=len(pre))], FREE)
    assert a.n == b.n == 1
    assert abs(a.trades[0].net_pips - b.trades[0].net_pips) < 1e-9


# --------------------------------------------------------------------------
# 5. Unresolved trades are counted, never dropped
# --------------------------------------------------------------------------

def test_a_trade_that_never_resolves_is_closed_and_counted():
    """Dropping unresolved trades silently deletes the worst of them."""
    bars = _flat(1.1000, 2) + _flat(1.0960, 30)      # drifts down, never hits either
    res = run(bars, [_long(index=0, stop=1.0900, target=1.1500)], FREE)
    assert res.n == 1, "the unresolved trade was dropped"
    t = res.trades[0]
    assert t.outcome in ("timeout", "end-of-data")
    assert t.net_pips < 0, "the loss it was sitting on must be booked"


def test_the_hold_cap_is_applied_and_reported():
    bars = _flat(1.1000, 2) + _flat(1.1000, MAX_HOLD_BARS + 10)
    res = run(bars, [_long(index=0, stop=1.0000, target=1.5000)], FREE,
              max_hold=10)
    assert res.n == 1
    assert res.trades[0].outcome == "timeout"
    assert res.timeouts == 1
    assert any("hold cap" in w for w in res.warnings)


# --------------------------------------------------------------------------
# The aggregate, and what it refuses to claim
# --------------------------------------------------------------------------

def test_the_measured_expectancy_does_not_double_count_cost():
    """Costs are already inside each trade's net_pips. Charging them again in the
    Expectancy would report a loss twice."""
    bars = _flat(1.1000, 2) + [Bar(1.1000, 1.1150, 1.0990, 1.1100)] + _flat(1.11, 2)
    res = run(bars, [_long(index=0)], CostModel(round_turn_pips=2.0))
    e = res.expectancy()
    assert e.cost_pips == 0.0
    assert abs(e.net - e.gross) < 1e-12
    assert abs(e.gross - res.trades[0].net_pips) < 1e-9


def test_a_small_sample_is_flagged_rather_than_quietly_reported():
    bars = _flat(1.1000, 2) + [Bar(1.1000, 1.1150, 1.0990, 1.1100)] + _flat(1.11, 2)
    res = run(bars, [_long(index=0)], FREE)
    assert any("too few" in w for w in res.warnings), res.warnings


def test_win_rate_and_averages_are_computed_on_NET_pips():
    """A trade that is gross-positive and net-negative is a LOSS. Counting it as a
    win is how a 61%-win-rate losing system gets reported as a winner."""
    bars = _flat(1.1000, 2) + [Bar(1.1000, 1.1012, 1.0990, 1.1010)] + _flat(1.101, 2)
    res = run(bars, [_long(index=0, stop=1.0900, target=1.1010)],
              CostModel(round_turn_pips=20.0))
    t = res.trades[0]
    assert t.pips > 0, "the fixture must be gross-positive"
    assert t.net_pips < 0, "and net-negative after a 20 pip cost"
    assert res.win_rate == 0.0, "a net-losing trade was counted as a win"


def test_the_summary_carries_the_warnings_with_the_numbers():
    bars = _flat(1.1000, 2) + [Bar(1.0800, 1.0810, 1.0790, 1.0800)] + _flat(1.08, 2)
    text = summary(run(bars, [_long(index=0, stop=1.0950)], FREE), "gap case")
    assert "gap case" in text and "!!" in text
    assert "gapped THROUGH" in text
    assert summary(run(_flat(1.10, 3), [], FREE)) == "backtest: no trades"


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
