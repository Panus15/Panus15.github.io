"""Tests for the expectancy arithmetic and the cost model.

These two modules encode the answer to the question that started this project —
"which pattern has the highest win rate?" — so the tests are written against the
CLAIMS, not against the happy path:

  * a high win rate can be a losing strategy, and the real FXCM numbers are the
    oracle for that;
  * cost raises the required win rate by exactly c/(W+L), which punishes small
    targets hardest;
  * no position size converts negative expectancy into positive;
  * a pip is a different fraction of notional on every pair;
  * the carry markup is charged on GROSS notional, which is what killed the
    retail carry trade.

Run: python3 tests/test_expectancy.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.costs import (FINANCING_DAYS, CostModel, bp_per_pip, pip_size)
from engine.expectancy import (Expectancy, break_even_payoff, kelly_fraction,
                               required_win_rate, sizing_cannot_fix)


# --------------------------------------------------------------------------
# The headline claim: win rate is not edge
# --------------------------------------------------------------------------

def test_the_real_fxcm_numbers_reproduce_a_losing_strategy_at_a_61_percent_win_rate():
    """The oracle for this whole project. FXCM's published figures over 43 million
    real trades: EUR/USD closed at a profit 61% of the time, average winner 48
    pips, average loser 83. If this module says that is profitable, it is wrong."""
    e = Expectancy(win_rate=0.61, avg_win_pips=48.0, avg_loss_pips=83.0)
    assert e.gross < 0, f"gross expectancy came out {e.gross:+.2f}"
    assert not e.profitable
    # and the size of the hole, so a refactor cannot quietly halve it
    assert abs(e.gross - (0.61 * 48.0 - 0.39 * 83.0)) < 1e-9
    assert -3.5 < e.gross < -3.0, e.gross
    # it needed a win rate this high to break even, and did not have it
    assert e.break_even_win_rate > 0.61
    assert abs(e.break_even_win_rate - 83.0 / 131.0) < 1e-9


def test_a_ninety_percent_win_rate_can_still_lose():
    """The direct answer to "find the highest win rate". 90% needs a payoff of only
    0.111 — and anything below it loses, however impressive the win rate looks."""
    assert abs(break_even_payoff(0.90) - (0.10 / 0.90)) < 1e-12
    assert abs(break_even_payoff(0.90) - 0.1111111) < 1e-6
    losing = Expectancy(win_rate=0.90, avg_win_pips=1.0, avg_loss_pips=20.0)
    assert losing.gross < 0
    assert losing.payoff_ratio < break_even_payoff(0.90)
    # while a 35% win rate with a good payoff wins
    winning = Expectancy(win_rate=0.35, avg_win_pips=30.0, avg_loss_pips=10.0)
    assert winning.gross > 0
    assert winning.payoff_ratio > break_even_payoff(0.35)


def test_win_rate_alone_orders_nothing():
    """Two rules, the higher win rate is the worse strategy. If a ranking function
    is ever added to this repo, this is what it must not do."""
    high = Expectancy(win_rate=0.85, avg_win_pips=5.0, avg_loss_pips=40.0)
    low = Expectancy(win_rate=0.40, avg_win_pips=40.0, avg_loss_pips=15.0)
    assert high.win_rate > low.win_rate
    assert high.gross < 0 < low.gross


# --------------------------------------------------------------------------
# Cost, and why it punishes the systems people ask for
# --------------------------------------------------------------------------

def test_cost_raises_the_required_win_rate_by_exactly_c_over_w_plus_l():
    """The identity, checked rather than asserted in prose."""
    for w, l, c in ((5, 5, 1.5), (20, 20, 1.5), (50, 50, 1.5), (30, 10, 0.8)):
        free = required_win_rate(w, l, 0.0)
        paid = required_win_rate(w, l, c)
        assert abs((paid - free) - c / (w + l)) < 1e-12, (w, l, c)


def test_the_penalty_is_worst_for_the_small_target_high_win_rate_systems():
    """The counterintuitive part, pinned with the numbers from the brief: a 5/5
    scalp needs 65% instead of 50%; a 50/50 swing needs 51.5%."""
    scalp = required_win_rate(5, 5, 1.5)
    mid = required_win_rate(20, 20, 1.5)
    swing = required_win_rate(50, 50, 1.5)
    assert abs(scalp - 0.650) < 1e-9, scalp
    assert abs(mid - 0.5375) < 1e-9, mid
    assert abs(swing - 0.515) < 1e-9, swing
    assert scalp > mid > swing, "the penalty must shrink as the target widens"


def test_the_break_even_property_carries_the_cost_too():
    """`required_win_rate` and `Expectancy.break_even_win_rate` are two
    implementations of one identity, and only the free function was pinned. With
    cost_pips=0 they agree trivially, so the cost term has to be tested with a
    cost."""
    e = Expectancy(win_rate=0.5, avg_win_pips=20.0, avg_loss_pips=20.0,
                   cost_pips=1.5)
    assert abs(e.break_even_win_rate - 21.5 / 40.0) < 1e-12, e.break_even_win_rate
    assert abs(e.break_even_win_rate - 0.5375) < 1e-12
    # the two implementations must not drift apart
    for w, l, c in ((5, 5, 1.5), (48, 83, 0.8), (30, 10, 2.0), (20, 20, 0.0)):
        prop = Expectancy(0.5, w, l, cost_pips=c).break_even_win_rate
        assert abs(prop - required_win_rate(w, l, c)) < 1e-12, (w, l, c)
    # and the penalty property agrees with the difference it claims to be
    assert abs(e.cost_penalty_points - 1.5 / 40.0) < 1e-12


def test_a_cost_bigger_than_the_whole_range_cannot_be_won_by_any_win_rate():
    """p* above 1.0 is the honest output: no win rate saves it."""
    p = required_win_rate(avg_win_pips=2.0, avg_loss_pips=2.0, cost_pips=5.0)
    assert p > 1.0, p
    e = Expectancy(win_rate=1.0, avg_win_pips=2.0, avg_loss_pips=2.0, cost_pips=5.0)
    assert not e.profitable, "even a 100% win rate loses when cost exceeds the win"


def test_net_is_always_gross_minus_cost_and_never_flatters():
    for c in (0.0, 0.5, 1.5, 3.0):
        e = Expectancy(win_rate=0.55, avg_win_pips=20.0, avg_loss_pips=18.0,
                       cost_pips=c)
        assert abs(e.net - (e.gross - c)) < 1e-12
        assert e.net <= e.gross


# --------------------------------------------------------------------------
# The theorem people argue with
# --------------------------------------------------------------------------

def test_no_position_size_converts_negative_expectancy_into_positive():
    """Martingale, grid, averaging down, 'recovery'. Expectation is linear in
    size, so scaling a negative by any positive weight stays negative — and a sum
    of negatives is negative whatever the weights."""
    neg = Expectancy(win_rate=0.95, avg_win_pips=2.0, avg_loss_pips=60.0).gross
    assert neg < 0
    for size in (0.1, 1.0, 10.0, 1000.0):
        assert sizing_cannot_fix(neg, size) < 0, size
    # a whole martingale sequence, which is just a weighted sum of the same sign
    sequence = [1, 2, 4, 8, 16, 32, 64]
    assert sum(sizing_cannot_fix(neg, s) for s in sequence) < 0


def test_kelly_returns_zero_rather_than_a_negative_size_when_there_is_no_edge():
    assert kelly_fraction(0.40, 1.0) == 0.0
    assert kelly_fraction(0.61, 48.0 / 83.0) == 0.0, "the FXCM case has no edge to size"
    assert kelly_fraction(0.60, 2.0) > 0.0


# --------------------------------------------------------------------------
# A pip is not a pip
# --------------------------------------------------------------------------

def test_pip_size_follows_the_jpy_convention():
    assert pip_size("USDJPY") == 0.01
    assert pip_size("EURJPY") == 0.01
    assert pip_size("eurjpy") == 0.01
    assert pip_size("EURUSD") == 0.0001
    assert pip_size("GBPUSD") == 0.0001


def test_one_pip_is_a_different_fraction_of_notional_on_every_pair():
    """The conversion nobody does. Comparing "1.2 pips" across pairs is not a
    like-for-like comparison, and a cost tested on one pair is not the other's."""
    assert abs(bp_per_pip("EURUSD", 1.10) - 0.909) < 0.002
    assert abs(bp_per_pip("GBPUSD", 1.27) - 0.787) < 0.002
    assert abs(bp_per_pip("USDJPY", 150.0) - 0.667) < 0.002
    # the same nominal pip cost is a 36% larger drag on EURUSD than on USDJPY
    assert bp_per_pip("EURUSD", 1.10) / bp_per_pip("USDJPY", 150.0) > 1.35


def test_a_nonsense_price_is_refused_rather_than_dividing_by_zero():
    for bad in (0.0, -1.10):
        try:
            bp_per_pip("EURUSD", bad)
        except ValueError:
            continue
        raise AssertionError(f"price {bad} was accepted")


# --------------------------------------------------------------------------
# Financing: the cost low turnover does not reduce
# --------------------------------------------------------------------------

def test_financing_accrues_on_calendar_nights_and_is_a_debit_by_default():
    c = CostModel(pair="EURUSD", swap_markup_annual=0.008, carry_annual=0.0)
    assert c.financing_pips(0, 1.10) == 0.0
    week = c.financing_pips(7, 1.10)
    assert week > 0, "the markup is a debit; a positive return value is a cost"
    # linear in nights, and using 365 not 252 — a 5-day-week assumption would
    # understate a month-long hold by ~40%
    assert abs(c.financing_pips(14, 1.10) - 2 * week) < 1e-9
    # 365 is written out rather than read from FINANCING_DAYS: deriving the
    # expectation from the constant under test makes the test move with the bug,
    # and a 252-day year would understate a month-long hold by ~40%.
    assert FINANCING_DAYS == 365.0, "financing accrues on calendar days"
    expected_bp = 0.008 * (7 / 365.0) * 10_000.0
    assert abs(week - expected_bp / bp_per_pip("EURUSD", 1.10)) < 1e-9
    # a full year of markup costs the whole markup, in bp of notional
    year_bp = c.financing_pips(365, 1.10) * bp_per_pip("EURUSD", 1.10)
    assert abs(year_bp - 0.008 * 10_000.0) < 1e-6, year_bp


def test_a_carry_credit_can_offset_the_markup_but_only_up_to_it():
    """Positive carry reduces the financing cost; it does not make the markup
    disappear. At carry exactly equal to the markup, financing is zero."""
    base = CostModel(swap_markup_annual=0.02, carry_annual=0.0)
    matched = CostModel(swap_markup_annual=0.02, carry_annual=0.02)
    rich = CostModel(swap_markup_annual=0.02, carry_annual=0.05)
    assert base.financing_pips(30, 1.10) > 0
    assert abs(matched.financing_pips(30, 1.10)) < 1e-12
    assert rich.financing_pips(30, 1.10) < 0, "net carry credit should be a credit"


def test_holding_longer_can_turn_a_winning_rule_into_a_losing_one():
    """The cost low turnover does not reduce. A rule with a +2 pip gross edge is
    profitable on a day trade and dead if it is held a month."""
    c = CostModel(pair="EURUSD", round_turn_pips=1.0, swap_markup_annual=0.02)
    e_short = Expectancy(0.55, 20.0, 18.0, cost_pips=c.total_cost_pips(1, 1.10))
    e_long = Expectancy(0.55, 20.0, 18.0, cost_pips=c.total_cost_pips(90, 1.10))
    assert e_short.profitable
    assert not e_long.profitable, e_long.net
    assert e_long.cost_pips > e_short.cost_pips


# --------------------------------------------------------------------------
# The arithmetic that killed retail carry
# --------------------------------------------------------------------------

def test_the_markup_is_charged_on_gross_notional_not_on_equity():
    """A long-3/short-3 G10 book is 200% gross and the markup is a debit on EVERY
    leg, so the drag is markup x 2 — while the carry is earned on the NET spread.
    At the cheapest published markup that is 1.6%/yr against a post-2008 index
    return of about 0.85%/yr: negative before a trade is placed."""
    c = CostModel(swap_markup_annual=0.008)
    assert abs(c.gross_notional_drag(2.0) - 0.016) < 1e-12
    assert abs(c.gross_notional_drag(1.0) - 0.008) < 1e-12
    index_return = 0.0085
    assert c.gross_notional_drag(2.0) > index_return, (
        "the fixture no longer reproduces the finding that killed retail carry")


def test_what_share_of_a_carry_spread_the_markup_takes():
    """The table from the brief: at a 3%/yr gross spread, a 0.8% markup takes 53%
    of it and a 1.5% markup takes all of it."""
    spread = 0.03
    for markup, want in ((0.008, 0.533), (0.015, 1.0), (0.02, 1.333)):
        drag = CostModel(swap_markup_annual=markup).gross_notional_drag(2.0)
        assert abs(drag / spread - want) < 0.005, (markup, drag / spread)


def test_annual_friction_reproduces_the_published_figures():
    """3.4% of equity at 1:1 and 10.2% at 3:1, at 250 trades a year and 1.5 pips."""
    c = CostModel(pair="EURUSD", round_turn_pips=1.5)
    at_1x = c.annual_friction(250, 1.0, 1.10)
    at_3x = c.annual_friction(250, 3.0, 1.10)
    assert abs(at_1x - 0.034) < 0.001, at_1x
    assert abs(at_3x - 0.102) < 0.002, at_3x
    assert abs(at_3x - 3 * at_1x) < 1e-9, "friction is linear in leverage"


def test_negative_inputs_are_refused_rather_than_producing_a_negative_cost():
    try:
        CostModel(round_turn_pips=-1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("a negative round-turn cost was accepted")
    try:
        CostModel().financing_pips(-5, 1.10)
    except ValueError:
        pass
    else:
        raise AssertionError("negative nights were accepted")


# --------------------------------------------------------------------------
# The things that cannot be modelled are still said
# --------------------------------------------------------------------------

def test_the_notes_state_what_the_numbers_cannot_cover():
    c = CostModel()
    assert "2015" in c.gap_risk_note and "does NOT bound" in c.gap_risk_note
    assert "CALENDAR" in c.financing_note and "triple" in c.financing_note
    assert "tom-next" in c.measured_markup_hint and "MEASURE" in c.measured_markup_hint


def test_the_summary_leads_with_the_verdict_not_the_win_rate():
    e = Expectancy(win_rate=0.61, avg_win_pips=48.0, avg_loss_pips=83.0,
                   cost_pips=1.5)
    text = e.summary()
    assert "LOSING" in text
    assert "NET expectancy" in text
    assert "break even" in text
    good = Expectancy(0.45, 40.0, 20.0, cost_pips=1.5).summary()
    assert "PROFITABLE" in good


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
