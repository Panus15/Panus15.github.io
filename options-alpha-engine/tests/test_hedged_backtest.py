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



# --------------------------------------------------------------------------- #
# Numeric oracle. The tests above are all QUALITATIVE -- the kill-switch fires,
# a crash hurts -- so every cost term in the daily P&L could be wrong or absent
# and none of them would notice. 29 of 78 first-sweep defects survived for that
# reason, including the rebalancing cost, the entry spread, the opening hedge
# and the unwind on liquidation.
#
# The oracle below re-derives ONE hedged trade from the module's own docstring:
# short a straddle, hold yesterday's delta in shares, and pay the cost of
# rebalancing and of crossing the spread. It is written differently from the
# engine (a single function, |S-K| for the expired straddle) and agreed with it
# bit-for-bit on four paths before it was trusted.
# --------------------------------------------------------------------------- #
import math
import random

from engine import pricing, volforecast


def _straddle(S, K, T, iv, r=0.03, q=0.0):
    if T <= 0:
        return abs(S - K), 0.0
    c = pricing.greeks(S, K, T, r, q, iv, "call")
    p = pricing.greeks(S, K, T, r, q, iv, "put")
    return c.price + p.price, c.delta + p.delta


def _oracle_trade(path, i0, dte, size, iv=0.05, bps=5e-4, sf=0.015, mult=100,
                  kill=None, equity=100_000.0, peak=100_000.0):
    """P&L of one delta-hedged short straddle; (total, killed, worst_dd)."""
    K = round(path[i0])
    v, d = _straddle(path[i0], K, dte / 252, iv)
    hedge = size * mult * d
    total = -(size * mult * 0.5 * (sf * v + 0.04))
    for s in range(i0 + 1, i0 + dte + 1):
        rem = (i0 + dte - s) / 252
        S, Sp = path[s], path[s - 1]
        vn, dn = _straddle(S, K, rem, iv)
        hn = size * mult * dn
        total += size * mult * (v - vn) + hedge * (S - Sp) - abs(hn - hedge) * S * bps
        v, hedge = vn, hn
        if kill is not None and rem > 0:
            dd = max(0.0, 1.0 - (equity + total) / max(peak, equity + total))
            if dd > kill:
                total -= size * mult * 0.5 * (sf * vn + 0.04) + abs(hn) * S * bps
                return total, True, dd
    return total, False, 0.0


def _gw(n, seed, vol, drift=0.0003):
    rng = random.Random(seed)
    pr = [100.0]
    for _ in range(n - 1):
        pr.append(pr[-1] * math.exp(rng.gauss(drift, vol)))
    return pr


def _lim(dd, vega=8_000.0):
    return portfolio.RiskLimits(max_net_short_vega=vega, max_drawdown=dd)


# A flat 63-bar warmup gives HAR a zero forecast, so IV is the 5% floor, and one
# 21-bar trade then fits exactly (63 + 21 < 85). CVaR sizing gives 2 contracts.
_FLAT = [100.0] * 85
_WIGGLE = [100.0] * 64 + [100.0 * (1 + 0.01 * math.sin(k)) for k in range(1, 22)]
_SIZE = 2


def test_a_flat_path_trade_matches_the_oracle_to_the_cent():
    """No price move, so the shares P&L is zero and the whole trade is theta, the
    entry spread and the rebalancing drag -- every friction in isolation.

    It also pins the IV FLOOR: HAR on a flat path forecasts exactly 0.0, so the
    straddle is sold at the 5% floor. Without the floor it would be priced at a
    zero vol and the pricer refuses.
    """
    assert volforecast.har_rv_forecast(_FLAT[:64]) == 0.0
    r = run_hedged_backtest(_FLAT)
    assert r.n_trades == 1 and r.skip_reasons == {}
    want, killed, _ = _oracle_trade(_FLAT, 63, 21, _SIZE)
    assert not killed
    assert abs(r.trade_pnl[0] - want) < 1e-9, (r.trade_pnl[0], want)
    assert abs(r.trade_pnl[0] - 226.2989189975) < 1e-6, r.trade_pnl[0]


def test_a_moving_path_trade_matches_the_oracle_to_the_cent():
    """A 1% sine wiggle makes the opening hedge, yesterday's-delta shares P&L and
    the daily re-hedge all matter. Using today's delta would be a look-ahead; a
    zero or wrong-sized opening hedge changes day one; expiry days measured in
    calendar rather than trading days change every mark."""
    r = run_hedged_backtest(_WIGGLE)
    want, killed, _ = _oracle_trade(_WIGGLE, 63, 21, _SIZE)
    assert not killed and r.n_trades == 1
    assert abs(r.trade_pnl[0] - want) < 1e-9, (r.trade_pnl[0], want)
    assert abs(r.trade_pnl[0] - (-656.9115437216)) < 1e-6, r.trade_pnl[0]


def test_each_friction_moves_the_trade_by_the_amount_it_should():
    """Isolating each cost term, in the direction it should move."""
    base = run_hedged_backtest(_WIGGLE).trade_pnl[0]
    hi_bps = run_hedged_backtest(_WIGGLE, hedge_bps=5e-3).trade_pnl[0]
    hi_spread = run_hedged_backtest(_WIGGLE, spread_frac=0.05).trade_pnl[0]
    assert hi_bps < base and hi_spread < base
    want_bps, _, _ = _oracle_trade(_WIGGLE, 63, 21, _SIZE, bps=5e-3)
    want_sf, _, _ = _oracle_trade(_WIGGLE, 63, 21, _SIZE, sf=0.05)
    assert abs(hi_bps - want_bps) < 1e-9, (hi_bps, want_bps)
    assert abs(hi_spread - want_sf) < 1e-9, (hi_spread, want_sf)


def test_the_kill_switch_liquidates_mid_trade_and_pays_to_leave():
    """A 5% gap on trade day 4 breaches a 0.3% limit. The trade is closed that day
    and pays the exit spread AND the cost of unwinding the hedge: -896.0623
    against -925.2948 for riding it to expiry. The oracle reproduces the
    liquidation, so the exit cost is checked term by term, not just its sign."""
    path = [100.0] * 67 + [95.0] * 18
    r = run_hedged_backtest(path, limits=_lim(0.003))
    want, killed, dd = _oracle_trade(path, 63, 21, _SIZE, kill=0.003)
    assert killed and r.n_killed_midtrade == 1
    assert abs(r.trade_pnl[0] - want) < 1e-9, (r.trade_pnl[0], want)
    assert abs(r.trade_pnl[0] - (-896.0623056417)) < 1e-6
    assert abs(r.worst_intratrade_drawdown - dd) < 1e-12
    assert abs(r.worst_intratrade_drawdown - 0.0087536695) < 1e-9

    ride = run_hedged_backtest(path, limits=_lim(0.01))
    assert ride.n_killed_midtrade == 0
    assert abs(ride.trade_pnl[0] - (-925.2948190234)) < 1e-6
    # the exit unwind is real money: dropping it would make leaving cheaper
    hedge_unwind = abs(_oracle_trade(path, 63, 21, _SIZE)[0] - want)
    assert hedge_unwind > 1.0, hedge_unwind


def test_the_kill_switch_does_not_liquidate_on_the_expiry_bar():
    """A loss that first breaches the limit on the LAST bar has nothing left to
    liquidate: the position is expiring, so there is no exit to pay for. Firing
    there would charge an exit spread and unwind a hedge on a trade that is
    already over."""
    path = [100.0] * 84 + [94.0]
    r = run_hedged_backtest(path, limits=_lim(0.003))
    want, killed, _ = _oracle_trade(path, 63, 21, _SIZE, kill=0.003)
    assert not killed and r.n_killed_midtrade == 0
    assert abs(r.trade_pnl[0] - want) < 1e-9
    assert abs(r.trade_pnl[0] - (-1011.3690031906)) < 1e-6


def test_drawdown_is_measured_from_the_peak_across_trades():
    """The entry governor reads drawdown from the PEAK equity, so an early winner
    raises the bar the later trades are judged against. Pinned on scenarios where
    measuring from the starting equity instead, or failing to update the peak
    after a trade, changes which entries are refused.

      g12 seed 2: 9 trades, 6 refused by the kill-switch, one liquidated
      crash seed 2: 9 trades, 4 refused, one liquidated
      crash seed 4: 7 trades, 6 refused, none liquidated -- and the trade whose
      only breach is on the expiry bar keeps its full holding period
    """
    a = run_hedged_backtest(_gw(400, 2, 0.012), limits=_lim(0.02), hedge_bps=5e-3)
    assert (a.n_trades, a.skip_reasons, a.n_killed_midtrade) == (
        9, {"kill_switch": 6, "regime": 1}, 1), (a.n_trades, a.skip_reasons)
    assert abs(a.metrics.total_return - (-0.01750482)) < 1e-7
    assert abs(a.worst_intratrade_drawdown - 0.020103) < 1e-6

    b = run_hedged_backtest(price_path_with_crash(400, seed=2), limits=_lim(0.02),
                            hedge_bps=5e-3)
    assert (b.n_trades, b.skip_reasons, b.n_killed_midtrade) == (
        9, {"kill_switch": 4, "regime": 3}, 1), (b.n_trades, b.skip_reasons)
    assert abs(b.metrics.total_return - (-0.02078833)) < 1e-7
    assert abs(b.worst_intratrade_drawdown - 0.020320) < 1e-6

    c = run_hedged_backtest(price_path_with_crash(400, seed=4), limits=_lim(0.02),
                            hedge_bps=5e-3)
    assert (c.n_trades, c.skip_reasons, c.n_killed_midtrade) == (
        7, {"kill_switch": 6, "regime": 3}, 0), (c.n_trades, c.skip_reasons)
    assert abs(c.metrics.total_return - (-0.02099018)) < 1e-7


def test_a_scheduled_event_skips_the_trade_and_sees_only_the_past():
    """`event_at(i, trailing)` is the gate for a jump whose DATE is public. It had
    no test at all: dropping the gate, handing it the whole price list, and
    shifting its bar by one all passed.

    It must be called on each entry bar with exactly the history up to and
    including that bar -- len(trailing) == i + 1 -- or it can see the event it is
    meant to anticipate.
    """
    path = price_path_with_crash(400, seed=1)
    seen = []

    def spy(i, trailing):
        seen.append((i, len(trailing)))
        return False

    base = run_hedged_backtest(path, event_at=spy)
    assert seen and all(n == i + 1 for i, n in seen), seen[:3]
    # Every call is on a 21-bar cycle, in order. The regime gate runs FIRST, so
    # bars it skips (147, 252, ...) never reach the event gate -- 11 calls here,
    # not one per cycle.
    bars = [i for i, _ in seen]
    assert all((i - 63) % 21 == 0 for i in bars) and bars == sorted(bars)
    assert bars[:5] == [63, 84, 105, 126, 168], bars[:5]
    assert base.skip_reasons.get("event", 0) == 0

    allev = run_hedged_backtest(path, event_at=lambda i, tr: True)
    assert allev.n_trades == 0 and allev.skip_reasons["event"] == 11
    assert allev.metrics.total_return == 0.0

    one = run_hedged_backtest(path, event_at=lambda i, tr: i == 63 + 21)
    assert one.skip_reasons["event"] == 1
    assert one.n_trades == base.n_trades - 1


def test_the_run_handles_a_path_that_ends_exactly_on_an_expiry():
    """When (n - warmup) is a multiple of dte the last trade ends on the final bar.
    `i0 + dte < n` is the guard that keeps the loop off the end; relaxing it asks
    for a bar that does not exist."""
    path = _gw(63 + 21 * 10, 5, 0.012)
    r = run_hedged_backtest(path, limits=_lim(0.02))
    assert r.n_trades == 8 and r.skip_reasons == {"regime": 1}
    assert abs(r.metrics.total_return - 0.01322060) < 1e-7


def test_the_equity_curve_starts_after_the_warmup():
    """Warmup bars carry no position, so the curve is n - warmup bars of P&L plus
    the starting point. Counting them would dilute every Sharpe and drawdown with
    63 days of nothing."""
    r = run_hedged_backtest(_gw(400, 1, 0.012))
    assert len(r.metrics.equity_curve) == 400 - 63 + 1
    assert len(r.metrics.daily_returns) == 400 - 63


def test_a_quiet_market_is_priced_at_the_vol_floor_not_the_forecast():
    """In a very quiet market the HAR forecast sits well under 5%, and the floor is
    what keeps the straddle priced at a vol anyone would quote."""
    r = run_hedged_backtest(_gw(400, 1, 0.002), limits=_lim(0.25))
    assert r.n_trades == 14 and r.skip_reasons == {"regime": 2}
    assert abs(r.metrics.total_return - 0.01043994) < 1e-7


def test_the_vega_model_reduces_to_the_constant_iv_run_when_switched_off_by_its_knobs():
    """Two exact identities that pin the model's parameters: with the shock cap at
    zero the IV never moves, and with a zero beta it does not respond to the shock.
    Both must reproduce the constant-IV run to the last digit, and the model with
    its defaults must NOT, or it is not a model.
    """
    path = price_path_with_crash(400, seed=1)
    off = run_hedged_backtest(path, limits=_lim(0.02))
    on = run_hedged_backtest(path, limits=_lim(0.02), model_vega_loss=True)
    cap0 = run_hedged_backtest(path, limits=_lim(0.02), model_vega_loss=True,
                               iv_shock_cap=0.0)
    beta0 = run_hedged_backtest(path, limits=_lim(0.02), model_vega_loss=True,
                                iv_beta=0.0)
    assert cap0.trade_pnl == off.trade_pnl
    assert beta0.trade_pnl == off.trade_pnl
    assert max(abs(a - b) for a, b in zip(on.trade_pnl, off.trade_pnl)) > 1e-6


def test_realised_vol_uses_only_the_window_ending_today():
    """The vega model reprices off trailing realised vol, and that estimate has to
    be walk-forward: nothing after day `s` may enter it. Altering every later
    price must leave it unchanged, and it must equal the textbook figure."""
    from engine.hedged_backtest import _recent_realized_vol
    path = _gw(120, 3, 0.02)
    s, w = 80, 21
    got = _recent_realized_vol(path, s, w)
    rets = [math.log(path[k] / path[k - 1]) for k in range(s - w + 1, s + 1)]
    m = sum(rets) / w
    want = math.sqrt(sum((x - m) ** 2 for x in rets) / (w - 1) * 252)
    assert abs(got - want) < 1e-12, (got, want)
    tampered = path[:s + 1] + [p * 3.0 for p in path[s + 1:]]
    assert _recent_realized_vol(tampered, s, w) == got

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
