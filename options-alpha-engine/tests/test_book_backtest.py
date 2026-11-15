"""Tests for the multi-position book (engine/book_backtest.py).

This harness exists because of a gap the other tests could not see: every
single-position backtest hands the risk governor an EMPTY book, so the
correlation-aware aggregation that justifies portfolio.py had never run in a
loop. These tests pin what happens when it finally does.

The headline the fixture produces, and the reason the module was worth building:
sized against an empty book the run peaks at 17,562 of net short vega against a
stated cap of 8,000 — it breaches its own risk limit by 2.2x and nothing notices,
because no single position is anywhere near the cap on its own.

Run: python3 tests/test_book_backtest.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import portfolio
from engine.book_backtest import run_book_backtest
from engine.hedged_backtest import price_path_with_crash

PX = {f"S{i}": price_path_with_crash(900, seed=i) for i in range(1, 6)}


def test_the_book_really_holds_several_positions_at_once():
    """Without overlap this harness would be five copies of the old one."""
    r = run_book_backtest(PX)
    assert r.n_trades > 40, r.n_trades
    assert r.max_open >= 2, (
        f"only {r.max_open} position open at a time — the stagger is not working, "
        f"and the governor is seeing an effectively empty book again")
    # The stagger must put symbols on DIFFERENT points of the cycle, not merely
    # produce several open positions — five synchronised trades aggregate to one
    # big position and would test nothing about a real book. Checked on the entry
    # PHASE rather than the first entry bar, because the vega cap delays some
    # symbols to a later cycle and that would blur a first-bar comparison.
    phase = {(e["t"] - 63) % 21 for e in r.entries}
    assert len(phase) > 1, f"all entries share one phase {phase} — not staggered"

    flat = run_book_backtest(PX, stagger=False)
    flat_phase = {(e["t"] - 63) % 21 for e in flat.entries}
    assert flat_phase == {0}, (
        f"unstaggered entries must all land on the same point of the cycle, "
        f"got {sorted(flat_phase)}")


def test_a_per_trade_view_materially_understates_the_book():
    """N short-vol positions are close to N times the risk, not sqrt(N)."""
    r = run_book_backtest(PX)
    assert r.peak_net_short_vega > r.peak_naive_vega * 1.5, (
        f"correlated aggregation {r.peak_net_short_vega:,.0f} vs per-trade "
        f"{r.peak_naive_vega:,.0f} — the correlation model is not doing anything")
    assert r.understatement > 0.3
    assert "understates by" in r.summary()


def test_the_cap_binds_and_ungoverned_sizing_breaches_it():
    """The finding: an empty-book view assembles a book no limit would allow."""
    lim = portfolio.RiskLimits(max_net_short_vega=8_000.0, max_drawdown=0.25)
    on = run_book_backtest(PX, limits=lim, govern=True)
    off = run_book_backtest(PX, limits=lim, govern=False)

    assert on.n_blocked_by_vega_cap > 10, on.skip_reasons
    assert on.peak_net_short_vega <= lim.max_net_short_vega * 1.02, (
        f"the governed book peaked at {on.peak_net_short_vega:,.0f} against a cap "
        f"of {lim.max_net_short_vega:,.0f}")
    assert off.peak_net_short_vega > lim.max_net_short_vega * 1.5, (
        f"the ungoverned book should blow through the cap; it peaked at "
        f"{off.peak_net_short_vega:,.0f}")
    assert off.n_blocked_by_vega_cap == 0
    # and the cap is not free: it costs return and buys tail
    assert off.metrics.total_return > on.metrics.total_return
    assert on.metrics.max_drawdown > off.metrics.max_drawdown


def test_a_tighter_cap_binds_harder():
    tight = run_book_backtest(
        PX, limits=portfolio.RiskLimits(max_net_short_vega=3_000.0, max_drawdown=0.25))
    loose = run_book_backtest(
        PX, limits=portfolio.RiskLimits(max_net_short_vega=40_000.0, max_drawdown=0.25))
    assert tight.n_trades < loose.n_trades
    assert tight.peak_net_short_vega < loose.peak_net_short_vega
    assert loose.skip_reasons.get("vega_cap", 0) < tight.skip_reasons.get("vega_cap", 0)


def test_correlation_assumption_changes_the_aggregate():
    """rho is a judgement call, so it must visibly matter rather than sit unused."""
    hi = run_book_backtest(PX, rho=0.95)
    lo = run_book_backtest(PX, rho=0.05)
    assert hi.peak_net_short_vega != lo.peak_net_short_vega
    # near-independence aggregates to less risk, so more trades clear the cap
    assert lo.n_trades >= hi.n_trades


def test_positions_are_retired_at_expiry():
    """A book that never releases capacity would strangle itself after one cycle."""
    r = run_book_backtest(PX, dte=21)
    assert r.n_trades > 5 * 2, (
        "with 5 symbols on a 21-bar cycle over ~840 usable bars there should be "
        f"many cycles; only {r.n_trades} trades means positions are never retired")
    assert r.max_open <= 5


def test_it_refuses_an_empty_universe_with_a_useful_message():
    try:
        run_book_backtest({})
    except ValueError as e:
        assert "symbol" in str(e).lower(), (
            f"the error should say what is missing, got {e!r}")
        return
    raise AssertionError("an empty symbol set should raise, not return a fake result")



def _gauss_world(n=700, seed=1, vol=0.012, crash_at=None):
    """Three symbols on seeded random walks, with an optional 12-bar crash."""
    import math
    import random
    rng = random.Random(seed)
    out = {}
    for sym in ("AAA", "BBB", "CCC"):
        pr = [100.0]
        for i in range(n - 1):
            step = rng.gauss(0.0002, vol)
            if crash_at is not None and crash_at <= i < crash_at + 12:
                step = rng.gauss(-0.035, vol * 4)
            pr.append(pr[-1] * math.exp(step))
        out[sym] = pr
    return out


def test_equity_accrues_as_pnl_lands_not_when_the_trade_is_entered():
    """The kill-switch and the CVaR sizer both read `equity`, and positions here
    overlap by design, so the next entry -- one bar later -- used to be sized and
    gated against equity that already contained the WHOLE outcome of trades that
    had barely started. The day-by-day P&L array was right; the scalar feeding
    the decisions ran ahead of it.

    Measured on the pre-fix code, 3 symbols, 700 bars, max_drawdown 0.2%:

        seed 1, crash at 300:  1 trade made (kill-switch latched for 79 entries)
                               against 23 once equity accrues as it lands.
        seed 2, no crash, 0.5%: kill-switch never fired, against once.

    The first is the dramatic one -- a single early losing trade, debited in full
    at entry, held the book out of the market for the rest of the run -- and the
    second is the quiet one: a real drawdown the old scalar could not see.
    """
    lim = portfolio.RiskLimits(max_net_short_vega=8_000.0, max_drawdown=0.002)
    r = run_book_backtest(_gauss_world(seed=1, crash_at=300), dte=21, warmup=63,
                          limits=lim)
    assert r.n_trades == 23, r.n_trades
    assert r.skip_reasons.get("kill_switch", 0) == 51, r.skip_reasons

    lim2 = portfolio.RiskLimits(max_net_short_vega=8_000.0, max_drawdown=0.005)
    q = run_book_backtest(_gauss_world(seed=2), dte=21, warmup=63, limits=lim2)
    assert q.skip_reasons.get("kill_switch", 0) == 1, q.skip_reasons
    assert q.n_trades == 57, q.n_trades


def test_a_book_that_never_draws_down_is_unchanged_by_accrual():
    """Accrual must not move a run whose equity only ever rises between entries:
    with a drawdown limit it never approaches, the result is the same trades."""
    wide = portfolio.RiskLimits(max_net_short_vega=8_000.0, max_drawdown=0.9)
    r = run_book_backtest(_gauss_world(n=900, seed=1), dte=21, warmup=63, limits=wide)
    assert r.skip_reasons.get("kill_switch", 0) == 0
    assert (r.n_trades, r.n_blocked_by_vega_cap) == (82, 19), (
        r.n_trades, r.n_blocked_by_vega_cap)
    assert abs(r.metrics.total_return - 0.0903) < 5e-4, r.metrics.total_return

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
