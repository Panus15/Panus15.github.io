"""Tests for the constructive half — and mostly for when it refuses to be.

The failure mode of an advice module is that it always finds something helpful
to say. The thing being pinned here is the opposite: that when the rule loses
BEFORE costs, every lever reports unachievable and the verdict says so in
plain words, because that is the case where a trader spends years changing
brokers over a problem that was never the broker.

The arithmetic is checked against values worked by hand from the formulae, not
from the module's own constants.

Run: python3 tests/test_levers.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.expectancy import Expectancy
from engine.levers import (advise, cost_lever, frequency_note, geometry_lever,
                           levers, payoff_lever, win_rate_lever)


def _e(p, w=10.0, l=10.0, c=1.5):
    return Expectancy(win_rate=p, avg_win_pips=w, avg_loss_pips=l, cost_pips=c)


# p=0.55, W=L=10 -> gross = 5.5 - 4.5 = +1.0, net = -0.5
THIN = _e(0.55)
# p=0.45 -> gross = 4.5 - 5.5 = -1.0, net = -2.5
DEAD = _e(0.45)
# p=0.70 -> gross = 7 - 3 = +4.0, net = +2.5
GOOD = _e(0.70)


def test_the_fixtures_are_what_the_arithmetic_says():
    """Every assertion below rests on these three signs being right."""
    assert abs(THIN.gross - 1.0) < 1e-9 and THIN.net < 0
    assert abs(DEAD.gross + 1.0) < 1e-9 and DEAD.net < 0
    assert abs(GOOD.gross - 4.0) < 1e-9 and GOOD.net > 0


# ---------------------------------------------------------------------------
# the result worth the module
# ---------------------------------------------------------------------------

def test_a_rule_that_loses_before_costs_cannot_be_fixed_by_anything():
    """The check that decides whether your problem is your broker or your idea."""
    assert all(not l.achievable for l in levers(DEAD)), [str(l) for l in levers(DEAD)]
    text = advise(DEAD)
    assert "NOTHING YOU CONTROL FIXES THIS" in text, text
    assert "different rule, not a different setup" in text


def test_an_edge_of_exactly_zero_is_hopeless_not_merely_expensive():
    """gross == 0 needs a cost BELOW zero to break even, so it belongs with the
    losers rather than with the rules that just need a cheaper broker. One
    character decides which, and the boundary is where that shows."""
    flat = _e(0.5)                                    # gross = 5 - 5 = 0
    assert flat.gross == 0.0, flat.gross
    l = cost_lever(flat)
    assert not l.achievable
    assert "free broker" in l.detail, l.detail
    assert "no broker is cheap enough" not in l.detail, l.detail
    assert "NOTHING YOU CONTROL FIXES THIS" in advise(flat)


def test_the_cost_lever_says_a_free_broker_would_not_help_either():
    l = cost_lever(DEAD)
    assert not l.achievable
    assert "free broker" in l.detail, l.detail
    assert math.isnan(l.needed), l.needed


def test_widening_a_negative_edge_is_refused():
    l = geometry_lever(DEAD)
    assert not l.achievable
    assert "nothing here to scale up" in l.detail, l.detail


# ---------------------------------------------------------------------------
# when there IS something to do
# ---------------------------------------------------------------------------

def test_a_real_but_thin_edge_points_at_the_cost():
    l = cost_lever(THIN)
    assert l.achievable
    assert abs(l.needed - 1.0) < 1e-9, l.needed      # break-even cost IS gross
    assert abs(l.current - 1.5) < 1e-9


def test_the_cost_lever_quotes_what_you_would_EARN_not_what_you_would_SAVE():
    """Saving your way to break-even earns nothing, and reporting the saving as
    a gain is the flattering version of this number."""
    l = cost_lever(THIN)
    # gross 1.0 at the cheapest 0.9 pip account leaves +0.10 a trade, not +0.50
    assert "+0.10" in l.detail, l.detail
    assert "+0.50" not in l.detail, l.detail
    assert "thin" in l.detail


def test_an_edge_smaller_than_the_cheapest_spread_is_unreachable():
    """gross = +0.2 pips: real, and smaller than trading costs anywhere."""
    tiny = _e(0.51)                                   # gross = 5.1 - 4.9 = 0.2
    assert abs(tiny.gross - 0.2) < 1e-9
    l = cost_lever(tiny)
    assert not l.achievable, l.detail
    assert "no broker is cheap enough" in l.detail, l.detail
    text = advise(tiny)
    assert "works on paper only" in text, text


def test_the_widening_factor_is_cost_over_gross():
    l = geometry_lever(THIN)
    assert abs(l.needed - 1.5) < 1e-9, l.needed       # 1.5 / 1.0
    assert l.achievable
    assert "15/15 pips" in l.detail, l.detail         # 10 x 1.5 each way


def test_widening_is_reported_as_a_hypothesis_not_a_projection():
    l = geometry_lever(THIN)
    assert "IF the win rate held" in l.detail and "it will not" in l.detail
    assert "wider stop is hit less often" in l.detail, l.detail
    assert "in opposite directions" in l.detail, l.detail
    assert "Re-run the backtest" in l.detail


def test_widening_past_a_point_is_a_different_strategy():
    far = _e(0.505, c=1.5)                            # gross = 0.1, k = 15
    l = geometry_lever(far)
    assert l.needed > 5.0 and not l.achievable, l.needed
    assert "different strategy, not a wider one" in l.detail


def test_a_rule_already_clearing_its_costs_needs_no_widening():
    l = geometry_lever(GOOD)
    assert l.achievable and l.needed == 1.0
    assert "no widening needed" in l.detail
    # the VERDICT line, not the geometry lever's text, which says the same words
    assert "VERDICT: it already clears its costs" in advise(GOOD), advise(GOOD)


# ---------------------------------------------------------------------------
# the two that are diagnostics, never dials
# ---------------------------------------------------------------------------

def test_the_win_rate_is_reported_and_never_recommended():
    l = win_rate_lever(THIN)
    assert not l.achievable, "a win rate was offered as something to adjust"
    assert abs(l.needed - 0.575) < 1e-9, l.needed     # (10 + 1.5) / 20
    assert "not a dial" in l.detail
    # "7.5%" alone also matches inside "57.5%" earlier in the same sentence,
    # so the assertion has to carry enough context to mean what it says
    assert "7.5% of the requirement is the cost" in l.detail, l.detail
    assert abs(l.gap - 0.025) < 1e-9, l.gap


def test_the_payoff_lever_is_a_hypothesis_too():
    l = payoff_lever(THIN)
    assert not l.achievable
    # W' = ((1-p)L + c)/p = (4.5 + 1.5)/0.55 = 10.909
    assert abs(l.needed - 10.909090909 / 10.0) < 1e-6, l.needed
    assert "lowers the win rate at the same time" in l.detail


def test_a_rule_with_no_winners_is_handled():
    l = payoff_lever(_e(0.0))
    assert not l.achievable and l.needed == math.inf
    assert "no winning trades" in l.detail


# ---------------------------------------------------------------------------
# the advice everyone gives that does not work
# ---------------------------------------------------------------------------

def test_trading_less_often_is_not_offered_as_a_fix():
    note = frequency_note(THIN, 250)
    assert "does not help" in note, note
    assert "frequency cannot change a sign" in note
    assert "-125" in note, note                       # -0.5 x 250


def test_frequency_is_the_lever_that_compounds_when_the_sign_is_right():
    note = frequency_note(GOOD, 250)
    assert "compounds" in note, note
    assert "+625" in note, note                       # +2.5 x 250
    assert "does not help" not in note


def test_the_advice_lists_every_lever_with_its_verdict():
    text = advise(THIN)
    for name in ("cost", "geometry", "win rate", "payoff", "frequency"):
        assert name in text, f"{name} missing"
    ls = levers(THIN)
    assert [l.achievable for l in ls] == [True, True, False, False], ls
    # four levers listed, plus the recommendation echoing the first achievable
    assert text.count("[CAN]") == 3 and text.count("[cannot]") == 2, text
    assert "-> [CAN] cost" in text, "the recommendation did not name one thing"


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
