"""Tests for the delta-hedged walk-forward backtest.

The point is not a specific Sharpe — it's that the engine behaves HONESTLY:
a crash-free sample looks like free money, a crash reveals the tail, and the
risk governor (regime gate, vega cap, drawdown kill-switch) actually fires.
Run: python3 tests/test_hedged_backtest.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import portfolio
from engine.data import SyntheticAdapter
from engine.hedged_backtest import price_path_with_crash, run_hedged_backtest


def test_calm_path_harvests_premium():
    calm = SyntheticAdapter(seed=5).price_history("X", 620)
    r = run_hedged_backtest(calm)
    assert r.n_trades > 5
    assert r.metrics.total_return > 0          # premium harvested when no crash
    assert r.metrics.sharpe > 0
    assert r.metrics.max_drawdown > -0.05       # shallow drawdown in a calm sample


def test_crash_reveals_the_tail():
    calm = SyntheticAdapter(seed=5).price_history("X", 620)
    crash = price_path_with_crash(756)
    rc = run_hedged_backtest(calm)
    rx = run_hedged_backtest(crash)
    # A crash makes the drawdown deeper and the Sharpe worse than the calm sample.
    assert rx.metrics.max_drawdown < rc.metrics.max_drawdown
    assert rx.metrics.sharpe < rc.metrics.sharpe


def test_regime_gate_skips_during_spike():
    crash = price_path_with_crash(756)
    r = run_hedged_backtest(crash)
    assert r.skip_reasons.get("regime", 0) > 0   # short-vol suppressed as vol accelerates


def test_drawdown_kill_switch_fires():
    crash = price_path_with_crash(756)
    r = run_hedged_backtest(
        crash, limits=portfolio.RiskLimits(max_net_short_vega=8000, max_drawdown=0.03)
    )
    assert r.skip_reasons.get("kill_switch", 0) > 0   # blocks new risk after breach


def test_vega_cap_blocks_oversized_book():
    crash = price_path_with_crash(756)
    r = run_hedged_backtest(
        crash, limits=portfolio.RiskLimits(max_net_short_vega=50, max_drawdown=0.9)
    )
    assert r.n_trades == 0
    assert r.skip_reasons.get("vega_cap", 0) > 0


def test_walk_forward_non_overlapping_coverage():
    # trades + skips must tile the timeline in non-overlapping dte blocks.
    calm = SyntheticAdapter(seed=5).price_history("X", 620)
    dte, warmup = 21, 63
    r = run_hedged_backtest(calm, dte=dte, warmup=warmup)
    blocks = (len(calm) - warmup) // dte
    assert r.n_trades + r.n_skipped == blocks


def test_vega_loss_deepens_crash_drawdown():
    # Same crash path, same params — only the opt-in vega-loss model is toggled.
    # Modelling the IV spike must make the SHORT vega book's crash mark-to-market
    # DEEPER (more negative max_drawdown) and the Sharpe WORSE than the constant-IV
    # (gamma-only) run. Entry sizing/gates are IV-entry based, so the SAME trades
    # are selected either way and the effect is purely the vega mark-to-market.
    crash = price_path_with_crash(756)
    off = run_hedged_backtest(crash, model_vega_loss=False)
    on = run_hedged_backtest(crash, model_vega_loss=True)
    assert on.n_trades == off.n_trades                          # identical trade set
    assert on.metrics.max_drawdown < off.metrics.max_drawdown   # strictly deeper crash DD
    assert on.metrics.sharpe < off.metrics.sharpe               # strictly worse Sharpe


def test_vega_loss_negligible_on_calm_path():
    # With no vol spike, trailing realised vol never runs hot enough to clear the
    # deadband, so the vega-loss model leaves a calm run essentially unchanged —
    # the vega loss is negligible when vol does not spike.
    calm = SyntheticAdapter(seed=5).price_history("X", 620)
    off = run_hedged_backtest(calm, model_vega_loss=False)
    on = run_hedged_backtest(calm, model_vega_loss=True)
    assert on.n_trades == off.n_trades
    assert abs(on.metrics.max_drawdown - off.metrics.max_drawdown) < 1e-4
    assert abs(on.metrics.sharpe - off.metrics.sharpe) < 0.05


def test_contract_mult_lets_a_crypto_level_path_trade():
    # Found on the first real Deribit run: BTC spot ~64,000 with the hardcoded
    # 100-shares multiplier made every position 100x too large, so CVaR sizing
    # returned 0 and the backtest reported trades=0 / size_zero. One BTC option is
    # ONE coin — with contract_mult=1 the same path trades normally.
    import math
    import random
    rng = random.Random(2)
    p = [64482.0]
    for _ in range(400):
        p.append(p[-1] * math.exp(rng.gauss(0, 0.025)))
    wrong = run_hedged_backtest(p, dte=21)                      # default 100
    right = run_hedged_backtest(p, dte=21, contract_mult=1)
    assert wrong.n_trades == 0 and wrong.skip_reasons.get("size_zero", 0) > 0
    assert right.n_trades > 0
    assert right.metrics.sharpe == right.metrics.sharpe         # finite

    # The multiplier must reach the P&L ARITHMETIC, not only the sizing. The right
    # invariant is INVARIANCE: CVaR sizing targets a dollar risk budget, so
    # doubling the contract size halves the contract count and total exposure
    # (size x mult) is unchanged -- P&L must stay the same. A HALF-FIX (sizing
    # threaded through but the P&L legs still hardcoded at 100) breaks exactly
    # this: the count halves while the P&L multiplier does not, so P&L collapses
    # to ~half. Asserting invariance therefore catches the half-fix that an
    # only-trades>0 test cannot.
    e = [x / 1000 for x in p]                                   # ~$64 underlying
    a = run_hedged_backtest(e, dte=21, contract_mult=100)
    b = run_hedged_backtest(e, dte=21, contract_mult=200)
    assert a.n_trades == b.n_trades > 0
    assert a.trade_pnl and any(abs(x) > 1e-9 for x in a.trade_pnl)
    tot_a, tot_b = sum(a.trade_pnl), sum(b.trade_pnl)
    assert abs(tot_a) > 1.0                                     # non-trivial
    assert abs(tot_b - tot_a) < 0.05 * abs(tot_a)               # invariant (~half if broken)


# --------------------------------------------------------------------------
# the drawdown kill-switch must work WHILE a position is open
# --------------------------------------------------------------------------

def _bleeding_path(n=400, start=100.0, vol=0.85, warm=63, seed=3):
    """Calm, then a long grinding high-vol regime — loss ACCUMULATES day by day.

    Deliberately NOT a gap: this is the regime a kill-switch can actually act in,
    and price_path_with_crash cannot test it because its damage lands in one bar.
    """
    import math
    import random
    rng = random.Random(seed)
    p = [start]
    for i in range(1, n):
        v = 0.10 if i < warm + 21 else vol
        dv = v / math.sqrt(252)
        p.append(p[-1] * math.exp(-0.5 * dv * dv + dv * rng.gauss(0, 1)))
    return p


def test_kill_switch_liquidates_a_position_that_is_bleeding():
    """The fix: equity is marked DAILY, not only when the position closes.

    Before this, drawdown was read once before entry and equity updated only after
    the whole holding period, so a limit of 25% could be blown through by 60% and
    the switch would not learn about it until the trade expired. A limit that is
    only checked when you are flat is not a limit.
    """
    P = _bleeding_path()
    lim = portfolio.RiskLimits(max_net_short_vega=1e9, max_drawdown=0.02)
    kw = dict(prices=P, dte=42, warmup=63, limits=lim, cvar_limit=0.10,
              model_vega_loss=True)
    on = run_hedged_backtest(**kw)
    off = run_hedged_backtest(**kw, kill_midtrade=False)

    assert on.n_killed_midtrade >= 1, "the switch never fired on a bleeding path"
    assert on.worst_intratrade_drawdown > lim.max_drawdown
    assert min(on.trade_pnl) > min(off.trade_pnl) * 0.5, (
        f"liquidating must materially cut the worst trade: "
        f"{min(on.trade_pnl):,.0f} vs {min(off.trade_pnl):,.0f}")
    assert on.metrics.total_return > off.metrics.total_return
    assert "LIQUIDATED" in on.summary()


def test_the_kill_switch_charges_the_cost_of_getting_out():
    """Liquidating is not free — the exit crosses the spread and unwinds the hedge.

    Checked on a path where the switch fires but the damage has ALREADY landed in a
    single gap bar: there is nothing left to save, so the only difference between
    the two runs is the exit cost, and it must make the killed run slightly worse.
    A switch that looked free here would be one that forgot to charge for exiting.
    """
    lim = portfolio.RiskLimits(max_net_short_vega=8_000.0, max_drawdown=0.03)
    kw = dict(prices=price_path_with_crash(900), limits=lim, model_vega_loss=True)
    on = run_hedged_backtest(**kw)
    off = run_hedged_backtest(**kw, kill_midtrade=False)
    assert on.n_killed_midtrade == 1, on.n_killed_midtrade
    assert min(on.trade_pnl) < min(off.trade_pnl), (
        "a gap that already happened cannot be un-lost; exiting should COST a little")
    assert abs(min(on.trade_pnl) - min(off.trade_pnl)) < 0.05 * abs(min(off.trade_pnl))

    # The two ledgers must not drift apart. trade_pnl is the per-trade log and the
    # metrics are built from the per-DAY array; if a cost is charged to one and not
    # the other, the equity curve and the trade list quietly tell different stories.
    for res in (on, off):
        by_day = res.metrics.total_return * 100_000.0
        assert abs(sum(res.trade_pnl) - by_day) < 1.0, (
            f"per-trade total {sum(res.trade_pnl):,.2f} != per-day total {by_day:,.2f}")


def test_disabling_the_switch_reproduces_the_old_behaviour_exactly():
    """On a path where the limit never binds, the two runs must be identical."""
    calm = SyntheticAdapter(seed=5).price_history("X", 620)
    on = run_hedged_backtest(calm)
    off = run_hedged_backtest(calm, kill_midtrade=False)
    assert on.n_killed_midtrade == 0
    assert on.trade_pnl == off.trade_pnl
    assert on.metrics.sharpe == off.metrics.sharpe



def test_the_kill_switch_has_never_actually_protected_anything():
    """A safety feature the repo advertises in three places and which has never
    fired. Recorded as a test so the claim cannot quietly become true-sounding
    again.

    At the shipped default (max_drawdown=0.25) the CVaR-sized book never gets
    near a 25% drawdown - it tops out around 8% - so the switch is INERT:
    kill_midtrade True and False produce byte-identical results on every path
    tried, calm and crash alike.

    That is not a bug. It is a risk control that has never been exercised, and
    describing it as protection implies evidence that does not exist.
    """
    paths = [("calm", SyntheticAdapter(seed=s).price_history("X", 900))
             for s in (1, 4, 7)]
    paths += [("crash", price_path_with_crash(900))]
    for label, px in paths:
        on = run_hedged_backtest(px, kill_midtrade=True)
        off = run_hedged_backtest(px, kill_midtrade=False)
        assert on.n_killed_midtrade == 0, (
            f"{label}: the kill-switch fired at defaults. That is NEWS - update "
            f"the docs that say it never has, rather than deleting this test")
        assert on.metrics.total_return == off.metrics.total_return, label


def test_the_kill_switch_works_when_it_is_reachable_and_costs_more_than_it_saves():
    """Separates "inert" from "broken", and measures the trade it makes.

    Lowering the threshold until it is reachable shows the mechanism is fine: it
    fires at max_drawdown <= 0.06 on the crash path. What it buys is the finding:

        max_drawdown 0.25 (default)   0 kills   total -0.56%   maxDD -8.4%
        max_drawdown 0.06             1 kill    total -6.90%   maxDD -7.6%

    6.3 points of return for 0.8 points of drawdown. On this fixture the switch
    is a bad trade when it acts, which is why the default is NOT being lowered to
    make it fire - that would be tuning a locked parameter (PREREGISTRATION.md
    §2.3) toward a worse outcome in order to justify a feature.
    """
    px = price_path_with_crash(900)
    loose = run_hedged_backtest(
        px, limits=portfolio.RiskLimits(max_net_short_vega=8_000.0,
                                        max_drawdown=0.25), kill_midtrade=True)
    tight = run_hedged_backtest(
        px, limits=portfolio.RiskLimits(max_net_short_vega=8_000.0,
                                        max_drawdown=0.06), kill_midtrade=True)
    assert loose.n_killed_midtrade == 0
    assert tight.n_killed_midtrade >= 1, "the mechanism is broken, not merely inert"
    assert tight.metrics.total_return < loose.metrics.total_return, (
        "the kill-switch stopped costing return when it fires; re-measure the "
        "trade before any doc claims it is protection")
    assert tight.metrics.max_drawdown > loose.metrics.max_drawdown, (
        "it fired and did not even reduce the drawdown")


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
