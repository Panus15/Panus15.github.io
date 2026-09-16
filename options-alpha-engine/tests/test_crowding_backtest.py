"""Tests for the crowding experiment (engine/crowding_backtest.py).

The harness is only worth running if it can do BOTH things: report an effect that
was deliberately injected, with the right sign, AND report nothing when nothing is
there. A study that only ever finds effects is a bug detector for its own author.

The fixture ALTERNATES which strike is the crowded one from block to block, so the
crowded leg is not always the same distance from spot. Without that, "crowded is
worse" and "the nearer strike has more gamma" would be the same measurement, and
the test would pass for the wrong reason.

Run: python3 tests/test_crowding_backtest.py
"""

import datetime
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import pricing
from engine.crowding_backtest import _realize_short_call, run_crowding_backtest
from engine.data import OptionChain, OptionQuote
from models.fund_flow import SupplyBucket

R, Q = 0.03, 0.0
DTE = 21
WARMUP = 63
BASE_DAY = datetime.date(2024, 1, 2)
MULTS = (1.04, 1.07)          # the two candidate strikes, in % of spot


def _walk(n, vol=0.20, s0=100.0, seed=7):
    """Deterministic lognormal path with a known annualised vol."""
    x, out = seed, [s0]
    dv = vol / math.sqrt(252)
    for _ in range(n):
        x = (1103515245 * x + 12345) & 0x7FFFFFFF
        u1 = x / 0x7FFFFFFF
        x = (1103515245 * x + 12345) & 0x7FFFFFFF
        u2 = x / 0x7FFFFFFF
        z = math.sqrt(-2.0 * math.log(max(u1, 1e-12))) * math.cos(2 * math.pi * u2)
        out.append(out[-1] * math.exp(-0.5 * dv * dv + dv * z))
    return out


PRICES = _walk(800)           # ~20% realised vol, enough bars for >30 pairs


def _fixture(dent, *, base_iv=0.24, supply_mult=None, alternate=True):
    """chain_at / supply_at where the CROWDED strike is mispriced by ``dent``.

    dent < 0  -> crowded vol is crushed (selling it should lose)
    dent > 0  -> crowded vol is inflated (selling it should pay)
    dent == 0 -> nothing to find
    """
    def crowded_i(t):
        return (t // DTE) % 2 if alternate else 0

    def chain_at(t, trailing):
        spot = trailing[-1]
        ci = crowded_i(t)
        quotes = []
        for i, m in enumerate(MULTS):
            K = round(spot * m, 2)
            iv = base_iv + (dent if i == ci else 0.0)
            px = pricing.price(spot, K, DTE / 365.0, R, Q, iv, "call")
            quotes.append(OptionQuote(DTE, K, "call", px, px))
        asof = (BASE_DAY + datetime.timedelta(days=t)).isoformat()
        return OptionChain("TEST", spot, R, Q, quotes, asof=asof)

    def supply_at(t, trailing):
        spot = trailing[-1]
        K = round(spot * MULTS[crowded_i(t)], 2)
        exp = (BASE_DAY + datetime.timedelta(days=t + DTE)).isoformat()
        heavy = supply_mult if supply_mult is not None else K
        return [SupplyBucket(exp, heavy, "call", 9000, 900e6, ["QQQI"]),
                SupplyBucket(exp, round(spot * 1.20, 2), "call", 500, 50e6, ["JEPQ"])]

    return chain_at, supply_at


# --------------------------------------------------------------------------
# 1. the P&L engine itself, before any crowding question
# --------------------------------------------------------------------------

def test_short_call_pnl_has_the_right_sign_against_realised_vol():
    """Selling vol above what realises pays; selling below it loses. No exceptions."""
    rich = _realize_short_call(PRICES, 100, DTE, PRICES[100] * 1.04, 0.45, R, Q,
                               5e-4, 0.015)
    cheap = _realize_short_call(PRICES, 100, DTE, PRICES[100] * 1.04, 0.06, R, Q,
                                5e-4, 0.015)
    assert rich > 0, f"selling 45% vol into ~20% realised should pay, got {rich}"
    assert cheap < 0, f"selling 6% vol into ~20% realised should lose, got {cheap}"
    assert rich > cheap


def test_costs_are_actually_charged():
    """At iv == realised the trade is not free — and EACH cost bites on its own."""
    K = PRICES[100] * 1.04
    free = _realize_short_call(PRICES, 100, DTE, K, 0.20, R, Q, 0.0, 0.0)
    spread_only = _realize_short_call(PRICES, 100, DTE, K, 0.20, R, Q, 0.0, 0.02)
    hedge_only = _realize_short_call(PRICES, 100, DTE, K, 0.20, R, Q, 5e-3, 0.0)
    both = _realize_short_call(PRICES, 100, DTE, K, 0.20, R, Q, 5e-4, 0.015)
    assert spread_only < free, "the entry spread must be charged"
    assert hedge_only < free, "hedge slippage must be charged"
    assert both < free


def test_the_position_is_really_delta_hedged():
    """The hedge exists to strip the DIRECTIONAL term, so prove it on a trend.

    On a path that runs +9% into a short call, a naked seller is destroyed by the
    payoff. If the hedge is wired correctly the same position is roughly flat,
    because what is left is only variance: realised vs implied.
    """
    n, s0, dv, mu = 30, 100.0, 0.12 / math.sqrt(252), 0.45 / 252
    path, x = [s0], 11
    for _ in range(n):
        x = (1103515245 * x + 12345) & 0x7FFFFFFF
        u1 = x / 0x7FFFFFFF
        x = (1103515245 * x + 12345) & 0x7FFFFFFF
        u2 = x / 0x7FFFFFFF
        z = math.sqrt(-2.0 * math.log(max(u1, 1e-12))) * math.cos(2 * math.pi * u2)
        path.append(path[-1] * math.exp(mu - 0.5 * dv * dv + dv * z))
    assert path[DTE] / path[0] - 1.0 > 0.05, "fixture must actually trend"

    K, iv = round(path[0] * 1.02, 2), 0.12
    hedged = _realize_short_call(path, 0, DTE, K, iv, R, Q, 0.0, 0.0)
    v0 = pricing.price(path[0], K, DTE / 252.0, R, Q, iv, "call")
    naked = 100.0 * (v0 - max(path[DTE] - K, 0.0))
    assert naked < -300.0, f"fixture should ruin a naked seller, got {naked:.2f}"
    assert abs(hedged) < 0.10 * abs(naked), (
        f"hedged P&L {hedged:+.2f} is not decoupled from the {naked:+.2f} "
        f"directional loss — the delta hedge is not doing its job")


# --------------------------------------------------------------------------
# 2. it finds an injected effect — in BOTH directions
# --------------------------------------------------------------------------

def test_finds_an_injected_penalty_on_crowded_strikes():
    chain_at, supply_at = _fixture(dent=-0.08)
    res = run_crowding_backtest(PRICES, chain_at, supply_at, dte=DTE, warmup=WARMUP)
    assert res.n_pairs >= 30, f"fixture must clear min_pairs, got {res.n_pairs}"
    assert res.mean_diff < 0, res.summary()
    assert res.t_stat < -2.0, res.summary()
    assert "CROWDED IS WORSE" in res.verdict, res.verdict
    assert res.win_rate_diff < 0.5
    assert res.mean_crowded < res.mean_uncrowded


def test_finds_an_injected_advantage_too():
    """Same harness, opposite injection — it is not hardwired to one answer."""
    chain_at, supply_at = _fixture(dent=+0.08)
    res = run_crowding_backtest(PRICES, chain_at, supply_at, dte=DTE, warmup=WARMUP)
    assert res.mean_diff > 0, res.summary()
    assert res.t_stat > 2.0, res.summary()
    assert "CROWDED IS BETTER" in res.verdict, res.verdict
    assert res.win_rate_diff > 0.5


# --------------------------------------------------------------------------
# 3. it stays quiet when there is nothing to find
# --------------------------------------------------------------------------

def test_reports_no_effect_when_none_was_injected():
    chain_at, supply_at = _fixture(dent=0.0)
    res = run_crowding_backtest(PRICES, chain_at, supply_at, dte=DTE, warmup=WARMUP)
    assert res.n_pairs >= 30
    assert abs(res.t_stat) < 2.0, f"false positive: {res.summary()}"
    assert "NO SIGNIFICANT EFFECT" in res.verdict, res.verdict


def test_a_thin_sample_refuses_to_conclude():
    """Six pairs of a HUGE injected effect still must not be called a result."""
    short_prices = PRICES[:WARMUP + DTE * 7]
    chain_at, supply_at = _fixture(dent=-0.15)
    res = run_crowding_backtest(short_prices, chain_at, supply_at,
                                dte=DTE, warmup=WARMUP)
    assert 0 < res.n_pairs < 30, res.n_pairs
    assert "NOT ENOUGH DATA" in res.verdict, res.verdict
    assert str(res.n_pairs) in res.verdict


# --------------------------------------------------------------------------
# 4. point-in-time: the study may never see its own future
# --------------------------------------------------------------------------

def test_chain_and_supply_are_strictly_point_in_time():
    chain_at, supply_at = _fixture(dent=-0.08)
    seen = {"max_len": 0, "calls": 0}

    def check(t, trailing):
        assert len(trailing) == t + 1, f"trailing len {len(trailing)} != t+1 {t + 1}"
        assert trailing[-1] == PRICES[t], "trailing must end exactly at bar t"
        assert list(trailing) == PRICES[:t + 1], "trailing leaked a future bar"
        seen["max_len"] = max(seen["max_len"], len(trailing))
        seen["calls"] += 1

    def spy_chain(t, trailing):
        check(t, trailing)
        return chain_at(t, trailing)

    def spy_supply(t, trailing):
        check(t, trailing)
        return supply_at(t, trailing)

    res = run_crowding_backtest(PRICES, spy_chain, spy_supply, dte=DTE, warmup=WARMUP)
    assert seen["calls"] == 2 * res.n_pairs, "every date must query both sources"
    assert seen["max_len"] <= len(PRICES) - DTE, "the last entry needs a full holding window"


# --------------------------------------------------------------------------
# 5. a thin or unusable sample is COUNTED, never silently dropped
# --------------------------------------------------------------------------

def test_every_skip_reason_is_counted():
    chain_at, supply_at = _fixture(dent=-0.08)

    def skips(res, key):
        return res.skipped.get(key, 0)

    no_chain = run_crowding_backtest(PRICES, lambda t, tr: None, supply_at,
                                     dte=DTE, warmup=WARMUP)
    assert no_chain.n_pairs == 0 and skips(no_chain, "no_chain") > 30
    assert "NOT ENOUGH DATA" in no_chain.verdict
    assert "no_chain:" in no_chain.summary(), "skips must be visible in the report"

    no_supply = run_crowding_backtest(PRICES, chain_at, lambda t, tr: [],
                                      dte=DTE, warmup=WARMUP)
    assert no_supply.n_pairs == 0 and skips(no_supply, "no_supply_data") > 30

    # supply sits at a strike our chain does not quote -> nothing is crowded
    elsewhere_chain, elsewhere_supply = _fixture(dent=-0.08, supply_mult=9999.0)
    no_pair = run_crowding_backtest(PRICES, elsewhere_chain, elsewhere_supply,
                                    dte=DTE, warmup=WARMUP)
    assert no_pair.n_pairs == 0 and skips(no_pair, "no_crowded_strike") > 30

    # and the crowded threshold is really wired to the data
    strict = run_crowding_backtest(PRICES, chain_at, supply_at, dte=DTE,
                                   warmup=WARMUP, crowded_min=0.999)
    assert strict.n_pairs == 0 and skips(strict, "no_crowded_strike") > 30

    # every strike is crowded -> there is no control leg left to compare against
    nofree = run_crowding_backtest(PRICES, chain_at, supply_at, dte=DTE,
                                   warmup=WARMUP, uncrowded_max=-1.0)
    assert nofree.n_pairs == 0 and skips(nofree, "no_uncrowded_strike") > 30

    # a clean run leaves NO skips behind -- the counter is not decorative
    clean = run_crowding_backtest(PRICES, chain_at, supply_at, dte=DTE, warmup=WARMUP)
    assert clean.skipped == {} and "skips: none" in clean.summary()


# --------------------------------------------------------------------------
# 6. the control leg must be COMPARABLE, or the study measures gamma
# --------------------------------------------------------------------------

def _ladder(mults_shares, *, iv=0.22):
    """chain/supply from [(spot multiplier, crowding share)], IN CHAIN ORDER.

    Strikes are kept >1% apart so ``crowding_score``'s strike tolerance treats them
    as genuinely distinct — otherwise a 'control' strike would inherit the crowded
    one's supply and the test would be measuring nothing.
    """
    total = sum(s for _, s in mults_shares) or 1.0

    def _strikes(spot):
        return [(round(spot * m, 2), sh) for m, sh in mults_shares]

    def chain_at(t, trailing):
        spot = trailing[-1]
        quotes = []
        for K, _ in _strikes(spot):
            px = pricing.price(spot, K, DTE / 365.0, R, Q, iv, "call")
            quotes.append(OptionQuote(DTE, K, "call", px, px))
        asof = (BASE_DAY + datetime.timedelta(days=t)).isoformat()
        return OptionChain("T", spot, R, Q, quotes, asof=asof)

    def supply_at(t, trailing):
        exp = (BASE_DAY + datetime.timedelta(days=t + DTE)).isoformat()
        return [SupplyBucket(exp, K, "call", 1.0, sh / total * 1e9, ["QQQI"])
                for K, sh in _strikes(trailing[-1]) if sh > 0]

    return chain_at, supply_at


def test_control_leg_is_matched_on_moneyness_not_chain_order():
    """The regression guard for a bias that made the null look real.

    Chain order puts the near-the-money strike first. If the harness took the first
    uncrowded strike it found, the control leg would be the highest-gamma contract
    on the board almost every time, and the crowded leg would lose for a reason that
    has nothing to do with crowding. The control must be the NEAREST strike.
    """
    # +1% is uncrowded and FIRST in the chain; +8% is crowded; +6% is the fair match
    chain_at, supply_at = _ladder([(1.01, 0.0), (1.06, 0.0), (1.08, 1.0)])
    res = run_crowding_backtest(PRICES, chain_at, supply_at, dte=DTE, warmup=WARMUP,
                                min_pairs=1)
    assert res.n_pairs > 30, res.summary()
    for p in res.pairs:
        ratio = p["uncrowded_strike"] / p["crowded_strike"]
        assert abs(ratio - 1.06 / 1.08) < 1e-3, (
            f"control leg drifted to the money: {p['uncrowded_strike']} paired with "
            f"{p['crowded_strike']} (ratio {ratio:.4f})")


def test_the_most_crowded_strike_is_the_one_tested():
    """Two strikes clear the threshold — the study must take the heavier one."""
    chain_at, supply_at = _ladder([(1.04, 0.35), (1.06, 0.0), (1.08, 0.65)])
    res = run_crowding_backtest(PRICES, chain_at, supply_at, dte=DTE, warmup=WARMUP,
                                min_pairs=1)
    assert res.n_pairs > 30
    for p in res.pairs:
        assert abs(p["crowded_share"] - 0.65) < 1e-6, p
        assert abs(p["crowded_strike"] / p["uncrowded_strike"] - 1.08 / 1.06) < 1e-3


def test_incomparable_strikes_are_dropped_not_compared():
    """No nearby control -> throw the date away rather than compare two trades."""
    chain_at, supply_at = _ladder([(1.03, 1.0), (1.25, 0.0)])
    res = run_crowding_backtest(PRICES, chain_at, supply_at, dte=DTE, warmup=WARMUP,
                                min_pairs=1)
    assert res.n_pairs == 0, "a 21% strike gap is not a comparable control"
    assert res.skipped.get("no_comparable_strike", 0) > 30

    # widen the tolerance and the same date becomes usable -> the cap is the reason
    loose = run_crowding_backtest(PRICES, chain_at, supply_at, dte=DTE, warmup=WARMUP,
                                  min_pairs=1, max_moneyness_gap=0.30)
    assert loose.n_pairs > 30


def test_the_null_fixture_does_not_manufacture_an_effect():
    """The shipped synthetic fixture with dent=0 must report nothing, repeatedly."""
    from engine.crowding_backtest import synthetic_crowding_series
    from engine.hedged_backtest import price_path_with_crash
    for n in (1600, 2400, 3200):
        ch, su = synthetic_crowding_series(dte=DTE, dent=0.0)
        res = run_crowding_backtest(price_path_with_crash(n), ch, su,
                                    dte=DTE, warmup=WARMUP)
        assert res.n_pairs >= 30, res.summary()
        assert abs(res.t_stat) < 2.0, f"false positive on {n} bars: {res.summary()}"
        # and the control leg is never the near-the-money strike
        ratios = {round(p["crowded_strike"] / p["uncrowded_strike"], 3)
                  for p in res.pairs}
        assert max(ratios) < 1.05, f"control drifted to the money: {sorted(ratios)}"


def test_the_shipped_fixture_detects_a_real_dent():
    """...and the same fixture with a dent injected must find it, both signs."""
    from engine.crowding_backtest import synthetic_crowding_series
    from engine.hedged_backtest import price_path_with_crash
    prices = price_path_with_crash(3000)
    worse = run_crowding_backtest(prices, *synthetic_crowding_series(dte=DTE, dent=0.05),
                                  dte=DTE, warmup=WARMUP)
    assert "CROWDED IS WORSE" in worse.verdict, worse.summary()
    better = run_crowding_backtest(prices, *synthetic_crowding_series(dte=DTE, dent=-0.03),
                                   dte=DTE, warmup=WARMUP)
    assert "CROWDED IS BETTER" in better.verdict, better.summary()


# --------------------------------------------------------------------------
# 7. the record it leaves behind
# --------------------------------------------------------------------------

def test_pairs_are_auditable_and_the_summary_renders():
    chain_at, supply_at = _fixture(dent=-0.08)
    res = run_crowding_backtest(PRICES, chain_at, supply_at, dte=DTE, warmup=WARMUP)
    assert len(res.pairs) == res.n_pairs
    p = res.pairs[0]
    assert p["crowded_share"] >= 0.30 and p["crowded_strike"] != p["uncrowded_strike"]
    assert abs(p["diff"] - (p["crowded_pnl"] - p["uncrowded_pnl"])) < 1e-9
    # both strikes get used across the sample -> the strike effect is not the finding
    assert len({round(x["crowded_strike"] / PRICES[x["t"]], 2) for x in res.pairs}) == 2
    # the paired mean is exactly the mean of the pairs, not a re-derived number
    assert abs(res.mean_diff - sum(x["diff"] for x in res.pairs) / res.n_pairs) < 1e-9
    text = res.summary()
    for want in ("pairs=", "mean paired difference", "t=", "CROWDED IS WORSE"):
        assert want in text, text


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
