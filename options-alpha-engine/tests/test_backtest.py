"""Tests for the backtest metrics engine (engine/backtest.py).

This module had NO test file at all and 678 tests passed around it, which is the
same hole `engine/sizing.py` was in. It is not a quiet corner: it computes the
Sharpe, Sortino, drawdown and CAGR a human reads to decide whether a strategy
works, and `engine/benchmark.py` prints them.

Every metric here is pinned against arithmetic computed independently with exact
rationals, not against the implementation. The oracles:

  * start 100, P&L [10, 10]      -> rets [0.1, 1/11], Sharpe 235.72441537
  * start 100, P&L [10, -5.5]    -> rets [0.1, -0.05], Sharpe 3.74165739,
                                    Sortino 11.22497216
  * 252 days that exactly double  -> CAGR exactly 1.0
  * equity [100,120,90,150,75]   -> max drawdown exactly -0.5

Run: python3 tests/test_backtest.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.backtest import TRADING_DAYS, BacktestResult, Fill, max_drawdown, run_backtest


def test_equity_curve_is_the_running_sum_including_the_start():
    res = run_backtest([100.0, -50.0], starting_equity=1000.0)
    assert res.equity_curve == [1000.0, 1100.0, 1050.0]
    assert len(res.equity_curve) == len(res.daily_returns) + 1


def test_daily_returns_are_ratios_of_consecutive_equity():
    res = run_backtest([10.0, 10.0], starting_equity=100.0)
    assert abs(res.daily_returns[0] - 0.1) < 1e-15
    assert abs(res.daily_returns[1] - (120.0 / 110.0 - 1.0)) < 1e-15
    assert abs(res.daily_returns[1] - 0.09090909090909091) < 1e-15


def test_total_return_and_cagr_against_an_exact_doubling():
    """252 trading days is exactly one year, so a doubling is a CAGR of exactly
    1.0 -- no tolerance needed on the exponent."""
    res = run_backtest([100000.0 / 252] * 252, starting_equity=100000.0)
    assert abs(res.equity_curve[-1] - 200000.0) < 1e-6
    assert abs(res.total_return - 1.0) < 1e-12
    assert abs(res.cagr - 1.0) < 1e-9
    assert len(res.daily_returns) == TRADING_DAYS


def test_cagr_is_annualised_not_the_period_return():
    """Half a year of doubling annualises to +300%, not +100%: (2)^2 - 1."""
    res = run_backtest([100000.0 / 126] * 126, starting_equity=100000.0)
    assert abs(res.total_return - 1.0) < 1e-12
    assert abs(res.cagr - 3.0) < 1e-9


def test_sharpe_uses_a_sample_standard_deviation():
    """rets [0.1, -0.05]: mean 0.025, SAMPLE sd (n-1) 0.10606602, so the Sharpe is
    3.74165739. A population sd (n) would be 0.075 and give 5.29150262."""
    res = run_backtest([10.0, -5.5], starting_equity=100.0)
    assert abs(res.daily_returns[0] - 0.1) < 1e-15
    assert abs(res.daily_returns[1] - (-0.05)) < 1e-15
    assert abs(res.sharpe - 3.74165739) < 1e-7, res.sharpe


def test_sharpe_is_annualised_by_the_root_of_the_trading_year():
    res = run_backtest([10.0, -5.5], starting_equity=100.0)
    daily = res.sharpe / (TRADING_DAYS ** 0.5)
    assert abs(daily - 0.23570226) < 1e-7, daily       # the un-annualised ratio


def test_sortino_normalises_by_every_day_not_only_the_down_days():
    """Target downside deviation counts a day with no downside as a ZERO, it does
    not drop it. For rets [0.1, -0.05] that is sqrt(0.0025/2) = 0.03535534 and a
    Sortino of 11.22497216; dividing by the one down day gives 0.05 and 7.93725393.
    """
    res = run_backtest([10.0, -5.5], starting_equity=100.0)
    assert abs(res.sortino - 11.22497216) < 1e-7, res.sortino
    assert res.sortino > res.sharpe                   # by construction here


def test_sortino_is_zero_when_nothing_ever_went_down():
    """No downside deviation at all -> the ratio is undefined and reported as 0.0
    rather than as an infinity or a division error."""
    res = run_backtest([10.0, 10.0], starting_equity=100.0)
    assert res.sortino == 0.0
    assert abs(res.sharpe - 235.72441537) < 1e-6, res.sharpe


def test_sharpe_is_zero_when_returns_never_vary():
    """A single observation has no sample dispersion, so the divisor is zero."""
    res = run_backtest([10.0], starting_equity=100.0)
    assert len(res.daily_returns) == 1
    assert res.sharpe == 0.0


def test_max_drawdown_against_hand_arithmetic():
    """[100,120,90,150,75]: peaks 100,120,120,150,150; the worst ratio is
    75/150 - 1 = -0.5, not 90/120 - 1 = -0.25 and not the last-to-peak move."""
    assert abs(max_drawdown([100.0, 120.0, 90.0, 150.0, 75.0]) + 0.5) < 1e-15
    assert abs(max_drawdown([100.0, 120.0, 90.0]) + 0.25) < 1e-15


def test_max_drawdown_is_zero_for_a_curve_that_only_rises():
    assert max_drawdown([100.0, 110.0, 120.0]) == 0.0
    assert max_drawdown([100.0]) == 0.0


def test_max_drawdown_runs_from_the_peak_not_from_the_start():
    """A curve that rises then falls back to its starting value has drawn down
    from the PEAK. Measuring from the start would report 0.0."""
    assert abs(max_drawdown([100.0, 200.0, 100.0]) + 0.5) < 1e-15


def test_max_drawdown_refuses_a_curve_with_no_capital_behind_it():
    """`equity[0] / peak` divided by zero instead of saying so."""
    for bad in ([0.0, 100.0, 50.0], [-10.0, 5.0]):
        try:
            max_drawdown(bad)
            assert False, f"should refuse {bad}"
        except ValueError:
            pass
    try:
        max_drawdown([])
        assert False, "should refuse an empty curve"
    except ValueError:
        pass


def test_hit_rate_does_not_count_a_flat_day_as_a_win():
    """Strictly positive. A flat day is not a winning day, and counting it as one
    flatters every strategy that spends most days doing nothing -- which is what
    a short-premium book does."""
    res = run_backtest([10.0, 0.0, -5.0], starting_equity=100.0)
    assert len(res.daily_returns) == 3
    assert res.daily_returns[1] == 0.0
    assert abs(res.hit_rate - 1.0 / 3.0) < 1e-15
    assert run_backtest([10.0, 10.0], starting_equity=100.0).hit_rate == 1.0


def test_ruin_stops_the_curve_and_says_so():
    """Equity passing exactly through zero used to raise ZeroDivisionError from
    the returns loop. -400 a day from 100000 hits exactly zero on day 250, and
    the 128 days of P&L after that describe an account that no longer exists."""
    res = run_backtest([-400.0] * 378, starting_equity=100000.0)
    assert res.ruined and res.ruin_day == 250, (res.ruined, res.ruin_day)
    assert res.equity_curve[-1] == 0.0
    assert len(res.equity_curve) == 251                  # truncated at ruin
    assert len(res.daily_returns) == 250
    assert res.cagr == -1.0
    assert abs(res.total_return + 1.0) < 1e-12
    assert abs(res.max_drawdown + 1.0) < 1e-12


def test_an_account_ending_below_zero_still_has_a_real_cagr():
    """A negative base raised to a fractional power is COMPLEX in Python, so this
    returned (-1.0089+0.7051j) and `summary()` raised ValueError formatting it as
    a percentage. The engine's only trade is selling options, whose loss is
    unbounded, so this is the case that matters most."""
    res = run_backtest([-300.0] * 500, starting_equity=100000.0)
    assert res.ruined and res.ruin_day == 334
    assert res.equity_curve[-1] == -200.0
    assert isinstance(res.cagr, float) and res.cagr == -1.0
    assert res.total_return < -1.0                       # -100.2%: honest arithmetic
    assert "RUINED" in res.summary()
    assert "day 334" in res.summary()


def test_a_healthy_backtest_is_not_flagged_ruined():
    """The ruin path must not fire on a drawdown, however deep.

    A CAGR OF -100% IS NOT A RUIN FLAG, which this test originally assumed. An
    80% loss over three days annualises to 0.201^84 - 1, about 2.5e-59 - 1, which
    IS -1.0 in floating point -- on an account that survived with 20% of its
    capital intact. `ruined` has to be its own field because no metric here can
    stand in for it.
    """
    res = run_backtest([-40000.0, -40000.0, 100.0], starting_equity=100000.0)
    assert res.equity_curve[-1] == 20100.0
    assert not res.ruined and res.ruin_day is None
    assert "RUINED" not in res.summary()
    assert abs(res.max_drawdown + 0.8) < 1e-9        # -80%, and it survived
    assert res.cagr == -1.0                          # the point above, pinned

    # Over a year the same survival reads as an ordinary negative CAGR.
    slow = run_backtest([-100.0] * 252, starting_equity=100000.0)
    assert not slow.ruined
    assert abs(slow.equity_curve[-1] - 74800.0) < 1e-9
    assert abs(slow.cagr + 0.252) < 1e-9             # -25.2% for exactly one year


def test_empty_pnl_and_capital_that_is_not_capital_are_refused():
    for bad_pnl in ([], ()):
        try:
            run_backtest(bad_pnl)
            assert False, "should refuse empty P&L"
        except ValueError:
            pass
    for bad_eq in (0.0, -1000.0):
        try:
            run_backtest([10.0], starting_equity=bad_eq)
            assert False, f"should refuse starting_equity={bad_eq}"
        except ValueError:
            pass


def test_summary_prints_every_metric_it_holds():
    res = run_backtest([10.0, -5.5], starting_equity=100.0)
    text = res.summary()
    for label in ("Total return", "CAGR", "Sharpe", "Sortino", "Max drawdown", "Hit rate"):
        assert label in text, label
    assert "+4.5%" in text                               # total return, as a percentage
    assert "3.74" in text                                # Sharpe, two decimals


def test_fill_lifts_the_ask_to_buy_and_hits_the_bid_to_sell():
    """Half the spread each way, in the direction that costs you."""
    buy = Fill(intended_price=2.00, side="buy", spread=0.10)
    sell = Fill(intended_price=2.00, side="sell", spread=0.10)
    assert abs(buy.executed_price - 2.05) < 1e-15
    assert abs(sell.executed_price - 1.95) < 1e-15
    assert buy.executed_price > 2.00 > sell.executed_price


def test_fill_side_is_validated_not_defaulted_to_a_sell():
    """`if side == "buy"` filled every other value as a SELL -- a full spread away
    and in the wrong direction. `cost_vs_mid` takes an absolute value and so
    reported the right friction either way, which is what hid it."""
    for ok in ("buy", "Buy", "BUY"):
        assert abs(Fill(2.00, ok, 0.10).executed_price - 2.05) < 1e-15
    for ok in ("sell", "Sell", "SELL"):
        assert abs(Fill(2.00, ok, 0.10).executed_price - 1.95) < 1e-15
    for bad in ("sel", "", "b", "short", None, 1):
        try:
            Fill(2.00, bad, 0.10).executed_price
            assert False, f"should refuse side={bad!r}"
        except ValueError:
            pass


def test_fill_cost_counts_the_spread_the_slippage_and_the_commission():
    """Per contract: |executed - intended| x 100, plus the commission."""
    f = Fill(intended_price=2.00, side="buy", spread=0.10, commission=0.65)
    assert abs(f.cost_vs_mid(1) - 5.65) < 1e-12          # 0.05 x 100 + 0.65
    assert abs(f.cost_vs_mid(3) - 16.95) < 1e-12         # and it scales per contract
    assert abs(f.cost_vs_mid(1, multiplier=1) - 0.70) < 1e-12
    slipped = Fill(2.00, "buy", 0.10, commission=0.65, slippage_frac=0.01)
    assert abs(slipped.executed_price - 2.07) < 1e-15    # 0.05 half-spread + 0.02 slip
    assert abs(slipped.cost_vs_mid(1) - 7.65) < 1e-12
    free = Fill(2.00, "buy", 0.10, commission=0.0)
    assert abs(free.cost_vs_mid(1) - 5.00) < 1e-12


def test_the_shipped_commission_is_the_one_in_force():
    """Every cost test above passes `commission` explicitly, so the default could
    have been zero and nothing would have failed.

    0.65 per contract is not a rounding detail: against a 0.10 spread on a single
    contract it is 0.65 of 5.65 of total friction, 11.5%, and it is charged per
    contract so it does not shrink with size. A backtest that drops it is the
    "ignores commissions" lie this module's own docstring opens with.
    """
    from dataclasses import fields
    default = next(f.default for f in fields(Fill) if f.name == "commission")
    assert default == 0.65, default
    f = Fill(intended_price=2.00, side="buy", spread=0.10)      # no commission given
    assert abs(f.cost_vs_mid(1) - 5.65) < 1e-12
    assert abs(f.cost_vs_mid(10) - 56.5) < 1e-12
    spread_only = Fill(2.00, "buy", 0.10, commission=0.0).cost_vs_mid(1)
    assert abs(f.cost_vs_mid(1) - spread_only - 0.65) < 1e-12


def test_fill_cost_is_the_same_friction_whichever_way_you_trade():
    """The magnitude is symmetric even though the signed price is not -- which is
    why a wrong `side` cannot be caught through the cost."""
    buy = Fill(2.00, "buy", 0.10, commission=0.65)
    sell = Fill(2.00, "sell", 0.10, commission=0.65)
    assert abs(buy.cost_vs_mid(2) - sell.cost_vs_mid(2)) < 1e-12


def test_the_result_carries_its_own_curve_and_returns():
    res = run_backtest([10.0, -5.5], starting_equity=100.0)
    assert isinstance(res, BacktestResult)
    assert res.equity_curve[0] == 100.0
    assert abs(res.equity_curve[-1] - 104.5) < 1e-12
    assert len(res.daily_returns) == 2


def _run_all():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
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
