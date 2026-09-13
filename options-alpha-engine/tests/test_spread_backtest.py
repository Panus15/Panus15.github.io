"""Tests for the defined-risk spread harness (engine/spread_backtest.py).

The load-bearing claims are: the daily marks telescope to the held-to-expiry
result exactly (so the fairer accounting did not change the economics), the loss
is genuinely BOUNDED whatever the underlying does, and the comparison against the
delta-hedged naked book flips in the direction the theory says it should when the
hedge stops working.

The exit-rule tests exist because the engine was RECOMMENDING a management rule
("close at 50% of max profit, or at 7 DTE") that nothing here implemented, so every
reported number described a different strategy from the one on the screen. The
property that matters is not that the rule helps — measured, its P&L effect is
noise — it is that the rule cannot be measured into looking free. Closing early
costs a round trip that holding to expiry never pays, and a harness that forgets
that charge will report a management rule as a money machine.

Run: python3 tests/test_spread_backtest.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.hedged_backtest import price_path_with_crash
from engine.signal_backtest import synthetic_chain_series
from engine.spread_backtest import (EXIT_RULE_TEXT, MIN_DTE_REMAINING,
                                    TAKE_PROFIT_FRAC, apply_exit_rule,
                                    mark_spread_daily, realize_spread,
                                    run_spread_backtest, spread_payoff)
from engine.stress import evenly_spaced_gaps, inject_gaps
from models import spreads
from models.baseline import BaselineDensityForecaster

# fine ladder + low quote floor: an iron condor needs FOUR strikes to exist at
# once, and the coarse default drops the wings in calm regimes
LADDER = tuple(round(-0.20 + 0.025 * i, 3) for i in range(17))
PRICES = price_path_with_crash(900)
FC = BaselineDensityForecaster()


def _chains():
    return synthetic_chain_series(dte=21, ladder=LADDER, min_px=0.005)


def _one_spread(t=200, structure="iron_condor"):
    chain = _chains()(t, PRICES[:t + 1])
    build = {"iron_condor": spreads.iron_condor,
             "put_credit_spread": spreads.put_credit_spread}[structure]
    return chain, build(chain, FC, PRICES[:t + 1], dte=21)


def test_daily_marks_telescope_to_the_held_to_expiry_result():
    """The fairer accounting must move P&L in TIME, never change its total."""
    for structure in ("iron_condor", "put_credit_spread"):
        _, sp = _one_spread(structure=structure)
        assert sp is not None, structure
        held = realize_spread(sp, PRICES[221])
        marks = mark_spread_daily(sp, PRICES, 200, 21, r=0.03, q=0.0)
        assert marks is not None
        assert abs(held - sum(marks.values())) < 1e-3, (
            f"{structure}: held-to-expiry {held:.4f} != marked {sum(marks.values()):.4f}")
        assert len(marks) == 22                      # entry bar + 21 holding bars


def test_the_loss_is_actually_bounded():
    """The whole product is the cap. Push the underlying anywhere and check it."""
    _, sp = _one_spread()
    assert sp is not None
    bound = -(sp.contract_mult * sp.max_loss + 0.65 * len(sp.legs))
    for terminal in (1.0, 20.0, 50.0, 90.0, 100.0, 110.0, 150.0, 400.0, 5_000.0):
        pnl = realize_spread(sp, terminal)
        assert pnl >= bound - 1e-6, f"S_T={terminal}: {pnl:.2f} breached the cap {bound:.2f}"
    # and a naked short call at the same strike is NOT bounded — the contrast
    short_call = max(5_000.0 - sp.legs[0].strike, 0.0)
    assert short_call > abs(bound) * 5


def test_the_long_wings_are_what_caps_it():
    """Delete the long legs and the same structure becomes unbounded."""
    _, sp = _one_spread()
    naked_legs = [l for l in sp.legs if l.side == "short"]
    assert len(naked_legs) < len(sp.legs), "fixture must contain long wings"
    capped = spread_payoff(sp, 5_000.0)
    uncapped = sum(max(5_000.0 - l.strike, 0.0) if l.kind == "call"
                   else max(l.strike - 5_000.0, 0.0) for l in naked_legs)
    assert uncapped > capped * 5, (capped, uncapped)


def test_a_clean_path_favours_the_hedged_naked_book():
    """Against a book that CAN hedge continuously, the wings are a pure cost.

    This is the finding the module was written expecting to be the other way
    round, so it is pinned deliberately: nothing here should quietly drift back
    to flattering the structure.
    """
    res = run_spread_backtest(PRICES, _chains(), FC, dte=21, warmup=63,
                              structure="iron_condor", always_sell=True)
    assert res.n_sold > 20, res.skip_reasons
    assert res.naked is not None and res.n_naked > 20
    assert res.naked.total_return > res.metrics.total_return
    assert res.worst_trade < res.naked_worst          # the spread took the bigger hit
    assert "simply worse" in res.verdict(), res.verdict()
    # the cap still held, even while losing
    assert res.worst_trade >= -res.max_loss_budgeted - 1e-6


def test_gaps_invert_it_and_the_cap_is_what_saves_the_book():
    """Where the hedge fails, the wings earn their price — the whole thesis."""
    gapped = inject_gaps(PRICES, evenly_spaced_gaps(900, every=40, size=0.18,
                                                    start=63))
    res = run_spread_backtest(gapped, _chains(), FC, dte=21, warmup=63,
                              structure="iron_condor", always_sell=True)
    assert res.naked is not None
    assert res.naked_worst < res.worst_trade, (
        f"under gaps the naked book must take the worse trade: "
        f"naked {res.naked_worst:,.0f} vs spread {res.worst_trade:,.0f}")
    assert res.naked_worst < -1_000, "the fixture must actually hurt the naked book"
    assert res.worst_trade >= -res.max_loss_budgeted - 1e-6, "the cap must hold"
    assert res.metrics.total_return > res.naked.total_return


def test_a_calm_sample_is_not_reported_as_a_win_for_the_wings():
    """If the naked book never lost, there was no tail to cap — say so."""
    class _Fake:
        total_return = 0.05
        sharpe = 1.0
        max_drawdown = -0.01
    from engine.spread_backtest import SpreadBacktestResult
    r = SpreadBacktestResult(metrics=_Fake(), n_periods=10, n_sold=8,
                             worst_trade=-50.0, best_trade=10.0,
                             max_loss_budgeted=500.0, naked=_Fake(),
                             naked_worst=+145.0, n_naked=8)
    v = r.verdict()
    assert "NOTHING STRESSED THE WINGS" in v, v
    assert "smaller worst trade" not in v, "a calm sample must not read as a saving"


def test_skips_are_counted_and_the_event_gate_reaches_this_harness_too():
    res = run_spread_backtest(PRICES, lambda t, tr: None, FC, dte=21, warmup=63,
                              always_sell=True)
    assert res.n_sold == 0 and res.skip_reasons.get("no_chain", 0) > 30
    assert res.verdict() == "no comparison available"

    gated = run_spread_backtest(PRICES, _chains(), FC, dte=21, warmup=63,
                                always_sell=True, event_at=lambda t, tr: True)
    assert gated.n_sold == 0 and gated.skip_reasons.get("event", 0) > 30


def test_a_spread_it_cannot_mark_is_dropped_rather_than_guessed():
    """An un-invertible quote must skip the date, not silently book a wrong P&L."""
    from engine.data import OptionChain, OptionQuote

    base = _chains()

    def broken(t, trailing):
        ch = base(t, trailing)
        if ch is None:
            return None
        # price the far wings absurdly high: no vol in the solver's range
        # reproduces them, so implied_vol correctly refuses
        qs = []
        for q in ch.quotes:
            if abs(q.strike / ch.spot - 1.0) > 0.09:
                qs.append(OptionQuote(q.expiry_days, q.strike, q.kind,
                                      ch.spot * 0.95, ch.spot * 0.96))
            else:
                qs.append(q)
        return OptionChain(ch.symbol, ch.spot, ch.r, ch.q, qs, asof=ch.asof)

    res = run_spread_backtest(PRICES, broken, FC, dte=21, warmup=63,
                              always_sell=True)
    assert res.skip_reasons.get("unmarkable", 0) > 0, res.skip_reasons
    assert res.n_sold < 29, "unmarkable dates must not be traded"


def test_the_ev_gate_is_walk_forward_and_binds():
    """Without always_sell, entry needs positive model EV — and that must matter."""
    free = run_spread_backtest(PRICES, _chains(), FC, dte=21, warmup=63,
                               always_sell=True)
    gated = run_spread_backtest(PRICES, _chains(), FC, dte=21, warmup=63,
                                min_ev=1e9)
    assert gated.n_sold == 0 and gated.skip_reasons.get("negative_ev", 0) > 20
    assert free.n_sold > gated.n_sold


# --------------------------------------------------------------------------
# The management rule the ticket recommends
# --------------------------------------------------------------------------

def _marked_trade(t=200, dte=21):
    chain, sp = _one_spread(t=t)
    marked = mark_spread_daily(sp, PRICES, t, dte, r=0.03, q=0.0)
    assert marked is not None
    return sp, marked, t, dte


class _FakeSpread:
    """A spread with numbers chosen so every branch of the rule is REACHABLE.

    A real trade exercises whichever branch its path happens to hit — the one at
    t=200 exits on time and never touches the profit target — so testing the rule
    through it leaves most of the rule unmeasured. 7 of 13 mutants survived a sweep
    that relied on it.
    """
    contract_mult = 100.0
    max_gain = 2.00               # = $200 per contract, so 50% is exactly $100
    max_loss = 8.00

    def __init__(self, n_legs=4):
        self.legs = [object()] * n_legs


def _series(t, pnls):
    """{bar: pnl} starting at t, so cum and remaining are both exactly known."""
    return {t + i: v for i, v in enumerate(pnls)}


def test_holding_to_expiry_is_still_the_default():
    """Every number published before the rule existed must be reproducible. A new
    parameter that silently changes past results makes the history unreadable."""
    r1 = run_spread_backtest(PRICES, _chains(), FC, dte=21, warmup=120,
                             compare_naked=False, always_sell=True)
    r2 = run_spread_backtest(PRICES, _chains(), FC, dte=21, warmup=120,
                             compare_naked=False, always_sell=True,
                             take_profit_frac=None, min_dte_remaining=None)
    assert r1.metrics.total_return == r2.metrics.total_return
    assert r1.exit_reasons == {} and r1.exit_rule == "held to expiry"


def test_the_rule_actually_fires_and_says_why():
    r = run_spread_backtest(PRICES, _chains(), FC, dte=21, warmup=120,
                            compare_naked=False, always_sell=True,
                            take_profit_frac=TAKE_PROFIT_FRAC,
                            min_dte_remaining=MIN_DTE_REMAINING)
    assert r.n_sold > 0
    assert sum(r.exit_reasons.values()) == r.n_sold, r.exit_reasons
    assert set(r.exit_reasons) <= {"profit", "time", "expiry"}
    assert "50%" in r.exit_rule and "7 DTE" in r.exit_rule
    assert "held to expiry" not in r.summary(), "the summary must not misdescribe it"


def test_an_early_exit_truncates_the_series_rather_than_rescaling_it():
    sp, marked, t, dte = _marked_trade()
    out, bar, why = apply_exit_rule(marked, sp, t, dte)
    assert bar is not None and why in ("profit", "time", "expiry")
    if why != "expiry":
        assert max(out) == bar < max(marked), (bar, max(marked))
        # every bar before the exit is untouched; only the exit bar carries the cost
        for b in sorted(out):
            if b != bar:
                assert out[b] == marked[b], b


def test_the_profit_target_fires_on_the_exact_bar_it_is_crossed():
    """$200 max gain, 50% target = $100. Cum is 60 at t+2 and 110 at t+3, so the
    exit is t+3 and nothing earlier."""
    sp, t, dte = _FakeSpread(), 100, 10
    marked = _series(t, [-2.6, 30.0, 30.0, 50.0, 20.0, 20.0, 20.0, 5.0, 5.0, 5.0, 5.0])
    out, bar, why = apply_exit_rule(marked, sp, t, dte, min_dte_remaining=None)
    assert (bar, why) == (t + 3, "profit"), (bar, why)
    assert max(out) == t + 3


def test_one_cent_below_the_target_does_not_fire():
    """The comparison is >=, so the boundary is a real decision and is pinned."""
    sp, t, dte = _FakeSpread(), 100, 10
    just_under = _series(t, [0.0, 50.0, 49.99] + [0.0] * 8)
    _, _, why = apply_exit_rule(just_under, sp, t, dte, min_dte_remaining=None)
    assert why == "expiry", why
    just_over = _series(t, [0.0, 50.0, 50.0] + [0.0] * 8)
    _, bar, why = apply_exit_rule(just_over, sp, t, dte, min_dte_remaining=None)
    assert (bar, why) == (t + 2, "profit")


def test_the_time_stop_fires_at_exactly_its_threshold_not_a_day_late():
    """`remaining <= 7` must fire ON the 7-DTE bar. `<` would hold one more day,
    through the part of the tenor the stop exists to avoid."""
    sp, t, dte = _FakeSpread(), 100, 21
    marked = _series(t, [0.0] * 22)
    _, bar, why = apply_exit_rule(marked, sp, t, dte, take_profit_frac=None,
                                  min_dte_remaining=7)
    assert (bar, why) == (t + 14, "time"), (bar, why, t + dte - bar)
    assert t + dte - bar == 7


def test_closing_early_is_never_free_and_the_charge_scales():
    """The whole trade-off. A rule measured without its round trip reads as free
    money — the easiest way to publish an improvement that does not exist."""
    sp, t, dte = _FakeSpread(), 100, 10
    marked = _series(t, [0.0, 60.0, 60.0] + [0.0] * 8)   # exits t+2 on profit
    free, bar, why = apply_exit_rule(marked, sp, t, dte, min_dte_remaining=None,
                                     close_spread_frac=0.0, commission=0.0)
    assert why == "profit"
    assert sum(free.values()) == 120.0, sum(free.values())
    # residual = 200 - 120 = 80; at 5% that is 4.00, plus 4 legs x 0.65 = 2.60
    paid, _, _ = apply_exit_rule(marked, sp, t, dte, min_dte_remaining=None,
                                 close_spread_frac=0.05, commission=0.65)
    assert abs(sum(paid.values()) - (120.0 - 4.00 - 2.60)) < 1e-9, sum(paid.values())
    dear, _, _ = apply_exit_rule(marked, sp, t, dte, min_dte_remaining=None,
                                 close_spread_frac=0.20, commission=0.65)
    assert abs(sum(dear.values()) - (120.0 - 16.00 - 2.60)) < 1e-9, sum(dear.values())
    assert sum(dear.values()) < sum(paid.values()) < sum(free.values())


def test_commission_is_charged_per_leg_not_per_trade():
    sp2, sp4, t, dte = _FakeSpread(2), _FakeSpread(4), 100, 10
    marked = _series(t, [0.0, 60.0, 60.0] + [0.0] * 8)
    two, _, _ = apply_exit_rule(marked, sp2, t, dte, min_dte_remaining=None,
                                close_spread_frac=0.0, commission=1.0)
    four, _, _ = apply_exit_rule(marked, sp4, t, dte, min_dte_remaining=None,
                                 close_spread_frac=0.0, commission=1.0)
    assert abs((sum(two.values()) - sum(four.values())) - 2.0) < 1e-9


def test_a_profit_beyond_the_maximum_cannot_produce_a_rebate():
    """Without the floor on the residual, a cum above max_gain makes the closing
    cost NEGATIVE — the harness would pay you to close, which is not a trade."""
    sp, t, dte = _FakeSpread(), 100, 10
    marked = _series(t, [0.0, 500.0] + [0.0] * 9)        # cum 500 vs max gain 200
    out, bar, why = apply_exit_rule(marked, sp, t, dte, min_dte_remaining=None,
                                    close_spread_frac=0.50, commission=0.0)
    assert (bar, why) == (t + 1, "profit")
    assert sum(out.values()) <= 500.0, "closing paid a rebate"
    assert sum(out.values()) == 500.0   # residual floored at 0, so cost is 0


def test_a_disabled_threshold_disables_only_its_own_half():
    sp, t, dte = _FakeSpread(), 100, 21
    # crosses the profit target at t+2 AND would hit the time stop at t+14
    marked = _series(t, [0.0, 60.0, 60.0] + [0.0] * 19)
    _, bar, why = apply_exit_rule(marked, sp, t, dte, take_profit_frac=None)
    assert (bar, why) == (t + 14, "time"), (bar, why)
    _, bar, why = apply_exit_rule(marked, sp, t, dte, min_dte_remaining=None)
    assert (bar, why) == (t + 2, "profit"), (bar, why)
    out, bar, why = apply_exit_rule(marked, sp, t, dte, take_profit_frac=None,
                                    min_dte_remaining=None)
    assert why == "expiry" and out == marked, "a disabled rule must change nothing"


def test_the_rule_cannot_close_on_the_entry_bar_or_after_settlement():
    """Closing on the open would book a round trip the position never had, and
    'closing' after expiry is not a trade at all."""
    sp, t, dte = _FakeSpread(), 100, 10
    # already past the target on the entry bar, and a second chance at t+1
    marked = _series(t, [150.0, 10.0] + [0.0] * 9)
    out, bar, why = apply_exit_rule(marked, sp, t, dte, min_dte_remaining=None)
    assert bar == t + 1, f"closed on the entry bar: {bar} == {t}"
    # the bar at expiry is not a close either: a zero-remaining bar must not be
    # charged a round trip the position never paid
    flat = _series(t, [0.0] * 11)
    out, bar, why = apply_exit_rule(flat, sp, t, dte, take_profit_frac=None,
                                    min_dte_remaining=0)
    assert why == "expiry", (bar, why)
    assert out == flat, "a settlement bar was charged a closing cost"


def test_the_managed_arm_reports_the_managed_pnl_not_the_terminal_payoff():
    """The defect this guards: `realised` was computed from the terminal spot
    before the rule ran, so worst_trade and best_trade described a trade the
    harness did not take."""
    held = run_spread_backtest(PRICES, _chains(), FC, dte=21, warmup=120,
                               compare_naked=False, always_sell=True)
    mgd = run_spread_backtest(PRICES, _chains(), FC, dte=21, warmup=120,
                              compare_naked=False, always_sell=True,
                              take_profit_frac=TAKE_PROFIT_FRAC,
                              min_dte_remaining=MIN_DTE_REMAINING)
    assert mgd.worst_trade != held.worst_trade, (
        "the managed arm is reporting the held arm's worst trade")
    assert mgd.worst_trade >= held.worst_trade, (
        f"cutting the trade short made the worst trade worse: "
        f"{mgd.worst_trade:,.0f} vs {held.worst_trade:,.0f}")


def test_the_recommended_rule_and_the_measured_rule_are_one_string():
    """They drifted once: the ticket printed a rule the harness did not run. One
    constant, so the screen cannot advise something unmeasured again."""
    assert f"{TAKE_PROFIT_FRAC:.0%}" in EXIT_RULE_TEXT
    assert str(MIN_DTE_REMAINING) in EXIT_RULE_TEXT
    assert "max profit" in EXIT_RULE_TEXT and "DTE" in EXIT_RULE_TEXT


def test_the_loss_stays_bounded_under_the_exit_rule_too():
    """The wings are the product. A management rule must not be able to book a
    loss larger than the structure allowed."""
    r = run_spread_backtest(PRICES, _chains(), FC, dte=21, warmup=120,
                            compare_naked=False, always_sell=True,
                            take_profit_frac=TAKE_PROFIT_FRAC,
                            min_dte_remaining=MIN_DTE_REMAINING)
    assert r.worst_trade >= -r.max_loss_budgeted - 1e-6, (
        f"worst {r.worst_trade:,.2f} breaches the budgeted {r.max_loss_budgeted:,.2f}")


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
