"""Tests for the defined-risk spread harness (engine/spread_backtest.py).

The load-bearing claims are: the daily marks telescope to the held-to-expiry
result exactly (so the fairer accounting did not change the economics), the loss
is genuinely BOUNDED whatever the underlying does, and the comparison against the
delta-hedged naked book flips in the direction the theory says it should when the
hedge stops working.

Run: python3 tests/test_spread_backtest.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.hedged_backtest import price_path_with_crash
from engine.signal_backtest import synthetic_chain_series
from engine.spread_backtest import (mark_spread_daily, realize_spread,
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
