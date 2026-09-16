"""Tests for the verdict, written against the ways a "no" can be wrong.

A refusal engine fails in two directions and both are expensive. It can refuse
something that was fine, in which case it is ignored and might as well not
exist. Or it can pass something that loses, which is the failure it was built to
prevent. So every refusal here is tested twice: once on a fixture that must trip
it, and once on the clean baseline that must NOT.

ISOLATION IS THE POINT. Each negative fixture is the clean baseline with exactly
ONE thing changed, and the test asserts both that the intended code fired and
that the others did not. Without that, a fixture rejected for the wrong reason
passes a test that never exercised the rule it names — the way a guard can be
dead for months while its test stays green.

THE ONE PAIR THAT CANNOT BE SEPARATED is NO_WIN_RATE_SAVES_IT and
NEGATIVE_NET_EXPECTANCY: if the cost exceeds the whole win+loss range then even
a perfect win rate loses, so the second follows from the first as arithmetic.
That test asserts both and says why.

Run: python3 tests/test_verdict.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.backtest import BacktestResult, Trade
from engine.costs import CostModel
from engine.verdict import (COST_NOT_MEASURED, COSTS_DISAGREE,
                            GOVERNED_BY_ASSUMPTION, NEGATIVE_NET_EXPECTANCY,
                            NOT_SIGNIFICANT, NOTHING_MEASURED,
                            NO_WIN_RATE_SAVES_IT, TOO_FEW_TRADES,
                            edge_t_stat, judge, judge_planned, required_t)
from models.patterns import Detection

#: A cost model the caller plainly typed in: neither field is a default, so
#: COST_NOT_MEASURED cannot fire and every other rule is tested in isolation.
MINE = CostModel(pair="EURUSD", round_turn_pips=1.2, swap_markup_annual=0.012)


def _trade(gross_pips, cost=1.5, name="double_top", i=0):
    """One trade with a KNOWN gross result. `cost` is what the backtest charged;
    net is gross - cost, which is what the verdict judges on."""
    return Trade(name=name, entry_index=i, exit_index=i + 1, direction=1,
                 entry=1.1000, exit=1.1000 + gross_pips * 0.0001,
                 pips=gross_pips, cost_pips=cost,
                 outcome="target" if gross_pips > 0 else "stop")


def _result(wins, win_pips, losses, loss_pips, *, cost=1.5, name="double_top",
            ambiguous=0):
    """`win_pips` and `loss_pips` are GROSS magnitudes, before cost."""
    res = BacktestResult()
    for k in range(wins):
        res.trades.append(_trade(win_pips, cost, name, k))
    for k in range(losses):
        res.trades.append(_trade(-loss_pips, cost, name, wins + k))
    res.ambiguous_bars = ambiguous
    return res


def _baseline():
    """60 trades, 60% gross win rate, +20/-10 gross at 1.5 pips cost.

    Net: +18.5 on a win, -11.5 on a loss, so +6.5 pips per trade. Every gate is
    cleared with room, which is what makes it usable as the control.
    """
    return _result(36, 20.0, 24, 10.0)


# ---------------------------------------------------------------------------
# the control
# ---------------------------------------------------------------------------

def test_a_clean_positive_result_is_tradeable():
    v = judge(_baseline(), MINE, costs_confirmed=True)
    assert v.tradeable, f"clean baseline refused: {v.codes}"
    assert v.codes == [], f"unexpected refusals on the control: {v.codes}"
    assert abs(v.net.net - 6.5) < 1e-9, v.net.net


def test_the_control_is_not_passing_by_accident():
    """The baseline must clear each gate on its own merits, not because the gate
    is unreachable. If any of these drift, every isolation test below is void."""
    v = judge(_baseline(), MINE, costs_confirmed=True)
    assert v.n_trades == 60
    assert v.t_stat > v.t_required, f"{v.t_stat} vs {v.t_required}"
    assert v.net.net > 0
    assert v.win_rate_needed < 1.0
    assert v.gross.win_rate > v.win_rate_needed
    assert v.win_rate_gap < 0, "a passing rule should have room to spare"


# ---------------------------------------------------------------------------
# NEGATIVE_NET_EXPECTANCY — the one the whole module exists for
# ---------------------------------------------------------------------------

def test_a_losing_rule_is_refused_by_name():
    # 60% win rate, and it still loses: +10 gross win against a -20 gross loss.
    # This is the FXCM result — winning more often than you lose is not an edge.
    v = judge(_result(36, 10.0, 24, 20.0), MINE, costs_confirmed=True)
    assert not v.tradeable
    assert NEGATIVE_NET_EXPECTANCY in v.codes, v.codes
    assert TOO_FEW_TRADES not in v.codes, "rejected for the wrong reason"
    assert COSTS_DISAGREE not in v.codes, "rejected for the wrong reason"
    assert v.gross.win_rate == 0.6, "the fixture was meant to WIN 60% of the time"


def test_the_refusal_states_the_win_rate_it_would_have_needed():
    # W=20, L=10, c=1.5 -> p* = (10 + 1.5) / 30 = 38.333%, computed here from
    # the fixture rather than from the module, so a change in the formula shows.
    v = judge(_result(6, 20.0, 54, 10.0), MINE, costs_confirmed=True)
    assert abs(v.win_rate_needed - 11.5 / 30.0) < 1e-9, v.win_rate_needed
    assert abs(v.gross.win_rate - 0.1) < 1e-9
    # short by 28.3 points, and SHORT is the direction: a sign error here reads
    # as "you have 28 points to spare" on a rule that loses money.
    assert abs(v.win_rate_gap - (11.5 / 30.0 - 0.1)) < 1e-9, v.win_rate_gap
    assert v.win_rate_gap > 0, "a losing rule was reported as having room to spare"
    detail = [r.detail for r in v.refusals if r.code == NEGATIVE_NET_EXPECTANCY][0]
    assert "38.3%" in detail, detail
    assert "10.0%" in detail, detail
    assert "+28.3%" in detail, detail


def test_sizing_cannot_rescue_a_negative_expectancy():
    """The claim every martingale seller makes, refuted with the number."""
    v = judge(_result(36, 10.0, 24, 20.0), MINE, costs_confirmed=True)
    detail = [r.detail for r in v.refusals if r.code == NEGATIVE_NET_EXPECTANCY][0]
    assert "100 units" in detail, detail
    # per-trade net is 0.6*8.5 - 0.4*21.5 = -3.5; a hundred of them is -350.
    assert "-350" in detail, detail
    assert v.suggested_risk_fraction == 0.0, "a refused rule was still given a size"


# ---------------------------------------------------------------------------
# COST_NOT_MEASURED / COSTS_DISAGREE
# ---------------------------------------------------------------------------

def test_a_rule_that_exactly_breaks_even_is_refused():
    """Zero is not positive. The boundary matters because a break-even rule
    still pays you nothing for carrying the gap risk."""
    v = judge(_result(30, 11.5, 30, 8.5), MINE, costs_confirmed=True)
    assert abs(v.net.net) < 1e-9, v.net.net
    assert NEGATIVE_NET_EXPECTANCY in v.codes, v.codes
    assert not v.tradeable


def test_default_costs_are_refused_until_they_are_confirmed():
    v = judge(_baseline(), CostModel())
    assert COST_NOT_MEASURED in v.codes, v.codes
    assert not v.tradeable
    ok = judge(_baseline(), CostModel(), costs_confirmed=True)
    assert COST_NOT_MEASURED not in ok.codes, ok.codes
    assert ok.tradeable, ok.codes


def test_changing_either_cost_field_counts_as_entering_your_own():
    """Both fields are checked; matching only one is still the default model."""
    one = CostModel(round_turn_pips=2.2)          # markup still default
    other = CostModel(swap_markup_annual=0.021)   # round turn still default
    assert COST_NOT_MEASURED not in judge(_baseline(), one).codes
    assert COST_NOT_MEASURED not in judge(_baseline(), other).codes
    assert COST_NOT_MEASURED in judge(_baseline(), CostModel()).codes


def test_a_result_measured_on_cheaper_costs_is_refused():
    # the backtest charged 0.4 pips; the model being judged says 1.2 is the
    # round turn ALONE. Nothing else can catch this: costs live in the trades.
    cheap = _result(36, 20.0, 24, 10.0, cost=0.4)
    v = judge(cheap, MINE, costs_confirmed=True)
    assert COSTS_DISAGREE in v.codes, v.codes
    assert NEGATIVE_NET_EXPECTANCY not in v.codes, "rejected for the wrong reason"
    assert "0.40" in str(v.refusals[0]) and "1.20" in str(v.refusals[0])
    # and when only SOME trades were under-costed, which is the case that hides:
    # the cheapest is what matters, not the dearest or the average.
    mostly = _result(36, 20.0, 23, 10.0, cost=1.5)
    mostly.trades.append(_trade(20.0, 0.4, "double_top", 99))
    assert COSTS_DISAGREE in judge(mostly, MINE, costs_confirmed=True).codes


def test_costs_at_or_above_the_model_do_not_trip_the_disagreement():
    assert COSTS_DISAGREE not in judge(
        _result(36, 20.0, 24, 10.0, cost=1.2), MINE, costs_confirmed=True).codes


# ---------------------------------------------------------------------------
# TOO_FEW_TRADES, and the higher bar for patterns already rejected in print
# ---------------------------------------------------------------------------

def test_too_few_trades_is_refused_however_good_it_looks():
    # 9 wins of +20 against 1 loss of -10: hugely profitable, t well clear of
    # the bar, so TOO_FEW_TRADES is the only thing that can be firing.
    v = judge(_result(9, 20.0, 1, 10.0), MINE, costs_confirmed=True)
    assert TOO_FEW_TRADES in v.codes, v.codes
    assert v.codes == [TOO_FEW_TRADES], f"not isolated: {v.codes}"
    assert v.net.net > 0 and v.t_stat > v.t_required
    # quarter-Kelly on a 90% win rate would size this at the cap; a refused rule
    # gets nothing, whatever the arithmetic would otherwise have allowed.
    assert v.suggested_risk_fraction == 0.0, v.suggested_risk_fraction
    # a floor alone is not actionable: the refusal must also say what THIS
    # claim would take, which is a different number from the generic 30.
    detail = v.refusals[0].detail
    assert "90%" in detail and "38%" in detail, detail
    assert "trades," in detail and "distinguishable" in detail, detail


def test_a_pattern_the_literature_rejected_needs_a_bigger_sample():
    """Same trades, same costs, same everything but the name.

    head_and_shoulders has been tested in FX and failed; double_top has never
    been tested. 60 trades is enough to ask the question of the untested one and
    not enough to overturn a published rejection of the other.
    """
    untested = judge(_result(36, 20.0, 24, 10.0, name="double_top"),
                     MINE, costs_confirmed=True)
    contested = judge(_result(36, 20.0, 24, 10.0, name="head_and_shoulders"),
                      MINE, costs_confirmed=True)
    assert untested.tradeable, untested.codes
    assert TOO_FEW_TRADES in contested.codes, contested.codes
    assert "rejected it" in str(contested.refusals[0]), contested.refusals[0]


def test_a_contested_pattern_with_enough_trades_passes_but_is_flagged():
    v = judge(_result(72, 20.0, 48, 10.0, name="head_and_shoulders"),
              MINE, costs_confirmed=True)
    assert v.tradeable, v.codes
    assert any("contradicts it" in n for n in v.notes), v.notes


def test_the_evidence_label_travels_with_the_verdict():
    v = judge(_result(36, 20.0, 24, 10.0, name="engulfing"), MINE,
              costs_confirmed=True)
    assert "randomised null" in v.evidence, v.evidence
    assert "Marshall" in v.evidence_detail, v.evidence_detail


# ---------------------------------------------------------------------------
# NO_WIN_RATE_SAVES_IT
# ---------------------------------------------------------------------------

def test_no_win_rate_saves_a_target_smaller_than_the_cost():
    """A 1-pip target and a 1-pip stop at 3 pips of cost needs p* = 350%.

    NEGATIVE_NET_EXPECTANCY necessarily fires too: if the cost exceeds the whole
    range then even a 100% win rate loses, so the two are the same fact stated
    twice and cannot be separated by any fixture.
    """
    v = judge(_result(30, 1.0, 30, 1.0, cost=3.0), MINE, costs_confirmed=True)
    assert NO_WIN_RATE_SAVES_IT in v.codes, v.codes
    assert NEGATIVE_NET_EXPECTANCY in v.codes, v.codes
    assert TOO_FEW_TRADES not in v.codes and COSTS_DISAGREE not in v.codes
    assert abs(v.win_rate_needed - 2.0) < 1e-9, v.win_rate_needed
    # and the boundary itself: W=2, L=1, c=2 puts p* at exactly 100%, which is
    # unreachable rather than merely hard, so the refusal must include it.
    edge = judge(_result(30, 2.0, 30, 1.0, cost=2.0), MINE, costs_confirmed=True)
    assert edge.win_rate_needed == 1.0, edge.win_rate_needed
    assert NO_WIN_RATE_SAVES_IT in edge.codes, edge.codes


# ---------------------------------------------------------------------------
# NOT_SIGNIFICANT
# ---------------------------------------------------------------------------

def test_a_positive_result_inside_the_noise_is_refused():
    # net +50 on a win and -49.6 on a loss at 50/50: +0.2 pips a trade, drowned
    # by a ~50 pip standard deviation. Profitable, and meaningless.
    v = judge(_result(30, 51.5, 30, 48.1), MINE, costs_confirmed=True)
    assert NOT_SIGNIFICANT in v.codes, v.codes
    assert v.codes == [NOT_SIGNIFICANT], f"not isolated: {v.codes}"
    assert v.net.net > 0, "the fixture was meant to be nominally profitable"
    # +0.2 pips against a 50.2 pip sd at t=2.241 is (2.241*50.2/0.2)^2 trades:
    # 316,765, against the 60 in hand. That number is the answer to "so should I
    # collect more data?", and no is a different answer from not yet.
    detail = v.refusals[0].detail
    assert "316,765" in detail, detail
    assert "and you have 60" in detail, detail


def test_a_losing_rule_is_not_also_called_insignificant():
    """NEGATIVE and NOT_SIGNIFICANT are alternatives, not a pile-on: telling
    someone their losing system is statistically unconvincing is noise."""
    v = judge(_result(36, 10.0, 24, 20.0), MINE, costs_confirmed=True)
    assert NOT_SIGNIFICANT not in v.codes, v.codes


def test_more_hypotheses_make_the_bar_higher():
    """The same result, judged against a wider search, stops being a discovery."""
    res = _result(36, 20.0, 24, 10.0)
    assert judge(res, MINE, costs_confirmed=True, n_hypotheses=1).tradeable
    strict = judge(res, MINE, costs_confirmed=True, n_hypotheses=100_000)
    assert NOT_SIGNIFICANT in strict.codes, strict.codes


def test_the_hypothesis_count_defaults_to_the_declared_parameters():
    v = judge(_baseline(), MINE, costs_confirmed=True)
    # one pattern name, four declared knobs
    assert v.n_hypotheses == 4, v.n_hypotheses
    mixed = _result(18, 20.0, 12, 10.0, name="double_top")
    mixed.trades.extend(_result(18, 20.0, 12, 10.0, name="flag").trades)
    assert judge(mixed, MINE, costs_confirmed=True).n_hypotheses == 8


def test_required_t_rises_with_the_number_of_hypotheses():
    assert required_t(1) < required_t(10) < required_t(1000)
    assert abs(required_t(1) - 1.6449) < 1e-3, required_t(1)   # one-sided 5%
    assert abs(required_t(20) - 2.8070) < 1e-3, required_t(20)  # one-sided 0.25%
    for bad in (0, -1):
        try:
            required_t(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"required_t({bad}) should not be allowed")


def test_the_t_stat_needs_at_least_two_trades_and_never_flatters_a_loss():
    assert edge_t_stat([]) == 0.0
    assert edge_t_stat([50.0]) == 0.0, "one trade is not a standard error"
    assert edge_t_stat([-5.0] * 40) == 0.0, "a zero-variance LOSS scored as an edge"
    assert edge_t_stat([5.0] * 40) == math.inf
    assert abs(edge_t_stat([1.0, -1.0, 1.0, -1.0])) < 1e-12


# ---------------------------------------------------------------------------
# GOVERNED_BY_ASSUMPTION
# ---------------------------------------------------------------------------

def test_a_result_the_ambiguity_rule_decided_is_refused():
    v = judge(_result(36, 20.0, 24, 10.0, ambiguous=30), MINE,
              costs_confirmed=True)
    assert GOVERNED_BY_ASSUMPTION in v.codes, v.codes
    assert v.codes == [GOVERNED_BY_ASSUMPTION], f"not isolated: {v.codes}"


def test_a_few_ambiguous_bars_are_noted_but_do_not_refuse():
    v = judge(_result(36, 20.0, 24, 10.0, ambiguous=3), MINE, costs_confirmed=True)
    assert v.tradeable, v.codes
    assert v.net.net > 0


def test_the_ambiguity_threshold_is_a_ceiling_not_a_target():
    """Exactly a quarter is the documented limit, not past it. Pinned because
    "above this share" and "at this share" are one character apart."""
    at = judge(_result(36, 20.0, 24, 10.0, ambiguous=15), MINE, costs_confirmed=True)
    assert abs(at.net.win_rate - 0.6) < 1e-9          # 15 of 60 = exactly 25%
    assert at.tradeable, at.codes
    over = judge(_result(36, 20.0, 24, 10.0, ambiguous=16), MINE,
                 costs_confirmed=True)
    assert GOVERNED_BY_ASSUMPTION in over.codes, over.codes


# ---------------------------------------------------------------------------
# what cost did to the trades, made visible
# ---------------------------------------------------------------------------

def test_trades_that_won_and_still_lost_money_are_counted_and_reclassified():
    # +1.0 gross against a 1.5 pip cost: a winning trade that lost money. The
    # gross view must still call it a win, and the net view must not.
    # 20 winners big enough to survive the cost, 20 that are not, 20 losers.
    # The mixture matters: a fixture where EVERY winner flips cannot tell
    # "winners eaten by cost" apart from "winners".
    res = _result(20, 30.0, 20, 30.0)
    res.trades.extend(_result(20, 1.0, 0, 0.0).trades)
    v = judge(res, MINE, costs_confirmed=True)
    assert v.flipped_by_cost == 20, v.flipped_by_cost
    assert abs(v.gross.win_rate - 40 / 60) < 1e-9, "gross win rate was net-classified"
    assert abs(v.net.win_rate - 20 / 60) < 1e-9, "a trade that lost money was a win"


def test_the_cost_penalty_is_reported_separately_from_the_rest():
    # W=20, L=10, c=1.5 -> the cost alone adds 1.5/30 = 5.0 points of win rate.
    v = judge(_baseline(), MINE, costs_confirmed=True)
    assert abs(v.gross.cost_penalty_points - 0.05) < 1e-9, v.gross.cost_penalty_points
    assert abs(v.net.cost_pips) < 1e-12, "cost was charged twice"


# ---------------------------------------------------------------------------
# sizing, the summary, and the empty case
# ---------------------------------------------------------------------------

def test_risk_is_capped_however_good_the_edge_looks():
    v = judge(_result(59, 100.0, 1, 1.0), MINE, costs_confirmed=True)
    assert v.tradeable, v.codes
    assert 0 < v.suggested_risk_fraction <= 0.01, v.suggested_risk_fraction
    # quarter-Kelly on a 98% win rate is enormous; the cap is what binds
    assert v.suggested_risk_fraction == 0.01, v.suggested_risk_fraction


def test_a_thin_edge_is_sized_below_the_cap():
    """The cap is a ceiling, not the answer. A 19% win rate at a payoff of 5 is
    worth +1.4 pips a trade and quarter-Kelly puts 0.7% of equity behind it."""
    v = judge(_result(380, 51.5, 1620, 8.5), MINE, costs_confirmed=True)
    assert v.tradeable, v.codes
    assert abs(v.net.net - 1.4) < 1e-9, v.net.net
    # f* = 0.19 - 0.81/5 = 0.028, quartered = 0.007, and the cap does not bind
    assert abs(v.suggested_risk_fraction - 0.007) < 1e-9, v.suggested_risk_fraction


def test_nothing_measured_when_there_are_no_trades():
    v = judge(BacktestResult(), MINE, name="flag", costs_confirmed=True)
    assert v.codes == [NOTHING_MEASURED], v.codes
    assert not v.tradeable
    assert "intention, not an edge" in v.refusals[0].detail
    # and with no name either: an empty result must not divide by an empty set
    assert judge(BacktestResult(), MINE).codes == [NOTHING_MEASURED]


def test_the_summary_names_every_refusal_and_both_win_rates():
    v = judge(_result(36, 10.0, 24, 20.0), CostModel())
    text = v.summary()
    assert "STAND DOWN" in text
    for code in v.codes:
        assert code in text, f"{code} missing from the summary"
    assert "required win rate" in text
    assert "t-stat" in text and "evidence:" in text
    ok = judge(_baseline(), MINE, costs_confirmed=True).summary()
    assert "TRADEABLE" in ok and "not a floor" in ok


# ---------------------------------------------------------------------------
# judge_planned — the question asked before any code is written
# ---------------------------------------------------------------------------

def test_a_planned_scalp_is_priced_before_it_is_ever_backtested():
    """5 pips each way at 1.5 pips of cost needs 65%, not 50%."""
    d = Detection(name="double_top", index=10, direction=1,
                  entry=1.1000, stop=1.0995, target=1.1005)
    v = judge_planned(d, CostModel(), costs_confirmed=True)
    assert v.codes == [NOTHING_MEASURED], v.codes
    detail = v.refusals[0].detail
    assert "65.0%" in detail, detail
    assert "15.0%" in detail, detail        # the cost's share: 1.5 / 100 pips
    assert not v.tradeable


def test_a_wider_target_is_punished_far_less_by_the_same_cost():
    wide = Detection(name="double_top", index=10, direction=1,
                     entry=1.1000, stop=1.1050, target=1.0500)
    v = judge_planned(wide, CostModel(), costs_confirmed=True)
    detail = v.refusals[0].detail
    assert "9.4%" in detail, detail         # (50 + 1.5) / 550
    assert "0.3%" in detail, detail         # the cost's share: 1.5 / 550


def test_planned_verdicts_also_refuse_unconfirmed_default_costs():
    d = Detection(name="flag", index=5, direction=-1,
                  entry=1.1000, stop=1.1050, target=1.0900)
    codes = judge_planned(d, CostModel()).codes
    assert codes == [NOTHING_MEASURED, COST_NOT_MEASURED], codes
    assert COST_NOT_MEASURED not in judge_planned(d, MINE).codes


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
