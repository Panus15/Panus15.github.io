"""Tests for the per-strike scanner (models/strike_scan.py).

The oracle: price a chain FROM the forecaster's own P-density -> no contract can
be +EV after costs (no free lunch against yourself), so every verdict is FAIR.
Then inject one mispriced contract each way and require the scanner to find
exactly it. Run: python3 tests/test_strike_scan.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.data import OptionChain, OptionQuote
from models.baseline import BaselineDensityForecaster
from models.strike_scan import scan_strikes

R, Q, DTE = 0.03, 0.0, 30
T = DTE / 365.0
F = BaselineDensityForecaster()

# ~20%-vol wiggle path, calm (regime gate must NOT fire)
PRICES = [100.0 * math.exp(0.0126 * math.sin(i / 3.0)) for i in range(80)]
SPOT = PRICES[-1]


def _fair_chain():
    """Chain priced exactly from the forecaster's own P-density (zero spread)."""
    p = F.forecast(PRICES, T, r=R, q=Q, spot=SPOT)
    quotes = []
    for k in range(80, 125, 4):
        K = float(k)
        for kind in ("call", "put"):
            px = p.price(K, R, T, kind)
            if px < 0.02:
                continue
            quotes.append(OptionQuote(DTE, K, kind, round(px, 4), round(px, 4)))
    return OptionChain("FAIR", SPOT, R, Q, quotes)


def test_no_free_lunch_on_self_priced_chain():
    signals = scan_strikes(_fair_chain(), F, PRICES)
    assert signals
    for s in signals:
        assert s.verdict == "FAIR", (s.strike, s.kind, s.verdict, s.edge_buy)
        assert s.edge_buy <= 0 and s.edge_write <= 0


def test_finds_the_one_cheap_call():
    ch = _fair_chain()
    victim = next(q for q in ch.quotes if q.kind == "call" and abs(q.strike - SPOT) < 3)
    object.__setattr__(victim, "bid", round(victim.bid - 0.60, 4))
    object.__setattr__(victim, "ask", round(victim.ask - 0.60, 4))
    signals = scan_strikes(ch, F, PRICES)
    buys = [s for s in signals if s.verdict == "BUY"]
    assert len(buys) == 1
    assert buys[0].strike == victim.strike and buys[0].kind == "call"
    assert signals[0] is buys[0]                       # ranked first (vol-points)
    assert buys[0].edge_buy * 100 > 40                 # ~$60 gap minus costs
    # ...and first under every ranking, since it is the only dislocation
    for rank in ("dollar", "premium", "vol_points"):
        assert scan_strikes(ch, F, PRICES, rank_by=rank)[0].strike == victim.strike


def test_finds_the_one_rich_put_and_regime_suppresses_it():
    ch = _fair_chain()
    victim = next(q for q in ch.quotes if q.kind == "put" and abs(q.strike - SPOT) < 3)
    object.__setattr__(victim, "bid", round(victim.bid + 0.60, 4))
    object.__setattr__(victim, "ask", round(victim.ask + 0.60, 4))
    signals = scan_strikes(ch, F, PRICES)
    writes = [s for s in signals if s.verdict == "WRITE"]
    assert len(writes) == 1
    assert writes[0].strike == victim.strike and writes[0].kind == "put"

    # Same rich contract under a stressed regime (forced, so the forecast itself
    # is unchanged) -> the write must be vetoed: verdict FAIR, note says why.
    stressed = scan_strikes(ch, F, PRICES, stressed=True)
    assert not [s for s in stressed if s.verdict == "WRITE"]
    flagged = [s for s in stressed if "regime" in s.note]
    assert flagged and flagged[0].strike == victim.strike


def test_itm_probabilities_are_consistent():
    signals = scan_strikes(_fair_chain(), F, PRICES)
    by_strike = {}
    for s in signals:
        by_strike.setdefault(s.strike, {})[s.kind] = s
    checked = 0
    for K, pair in by_strike.items():
        if "call" not in pair or "put" not in pair:
            continue
        # model call-ITM + put-ITM = 1 exactly (same density, complementary events)
        assert abs(pair["call"].p_itm_model + pair["put"].p_itm_model - 1.0) < 1e-12
        # market-implied N(d2) + N(-d2) = 1 when both sides have an IV
        if pair["call"].p_itm_market is not None and pair["put"].p_itm_market is not None:
            checked += 1
        # probabilities are probabilities
        assert 0.0 <= pair["call"].p_itm_model <= 1.0
    assert checked >= 0


def test_slippage_penalises_wide_spread_edges():
    # Give the cheap call a WIDE spread; slippage must shrink its edge and can
    # push a marginal signal back to FAIR. Deep/illiquid strikes pay the most.
    ch = _fair_chain()
    victim = next(q for q in ch.quotes if q.kind == "call" and abs(q.strike - SPOT) < 3)
    object.__setattr__(victim, "bid", round(victim.bid - 0.80, 4))   # 0.60 under fair...
    object.__setattr__(victim, "ask", round(victim.ask - 0.40, 4))   # ...with a 0.40 spread
    no_slip = scan_strikes(ch, F, PRICES, slippage_frac=0.0)
    with_slip = scan_strikes(ch, F, PRICES, slippage_frac=0.5)
    b0 = next(s for s in no_slip if s.strike == victim.strike and s.kind == "call")
    b1 = next(s for s in with_slip if s.strike == victim.strike and s.kind == "call")
    spread = victim.ask - victim.bid
    assert abs((b0.edge_buy - b1.edge_buy) - 0.5 * spread) < 1e-9   # exactly half the spread
    assert b1.edge_buy < b0.edge_buy


def test_break_even_slippage_margin():
    # A wide-spread cheap call: break_even_slip_frac = gross edge / spread — how
    # many spreads of slippage the edge survives. The decision metric for thin VRP.
    ch = _fair_chain()
    v = next(q for q in ch.quotes if q.kind == "call" and abs(q.strike - SPOT) < 3)
    object.__setattr__(v, "bid", round(v.bid - 0.80, 4))
    object.__setattr__(v, "ask", round(v.ask - 0.40, 4))
    s = next(x for x in scan_strikes(ch, F, PRICES, slippage_frac=0.0)
             if x.strike == v.strike and x.kind == "call")
    assert s.verdict == "BUY"
    gross = s.fair_value - v.ask - 0.65 / 100.0
    assert abs(s.break_even_slip_frac - gross / (v.ask - v.bid)) < 1e-6
    assert 0.5 < s.break_even_slip_frac < 1.5

    # A zero-spread edge: slippage penalty (∝ spread) is always 0 -> infinite margin.
    ch2 = _fair_chain()
    v2 = next(q for q in ch2.quotes if q.kind == "call" and abs(q.strike - SPOT) < 3)
    object.__setattr__(v2, "bid", round(v2.bid - 0.60, 4))
    object.__setattr__(v2, "ask", round(v2.ask - 0.60, 4))
    s2 = next(x for x in scan_strikes(ch2, F, PRICES)
              if x.strike == v2.strike and x.kind == "call")
    assert s2.break_even_slip_frac == float("inf")


def test_top_truncates_ranked_list():
    signals = scan_strikes(_fair_chain(), F, PRICES, top=3)
    assert len(signals) == 3
    vp = [s.edge_vol_points for s in signals]
    assert vp == sorted(vp, reverse=True)              # default rank: vol points
    dollars = [max(s.edge_buy, s.edge_write) * s.contract_mult
               for s in scan_strikes(_fair_chain(), F, PRICES, rank_by="dollar", top=3)]
    assert dollars == sorted(dollars, reverse=True)
    try:
        scan_strikes(_fair_chain(), F, PRICES, rank_by="nonsense")
        assert False, "unknown rank_by should raise"
    except ValueError:
        pass


def test_dollar_ranking_favours_long_dated_but_vol_points_does_not():
    # The real-data defect: on a chain with several maturities, a DOLLAR-ranked
    # board is topped by the longest expiry purely because its premium is biggest,
    # even though it is the worst edge per unit of premium. Ranking in vol points
    # (edge / vega) is maturity-neutral, which is what makes a board comparable.
    import random
    from engine import pricing as _pr
    rng = random.Random(11)
    p = [100.0]
    for _ in range(280):
        p.append(p[-1] * math.exp(rng.gauss(0.0006, 0.030)))
    for _ in range(120):
        p.append(p[-1] * math.exp(rng.gauss(0.0002, 0.017)))
    spot = p[-1]
    quotes = []
    for dte, iv in ((19, 0.378), (61, 0.428), (334, 0.472)):     # upward-sloping IV
        t = dte / 365.0
        for mny in (-0.05, 0.0, 0.05):
            K = round(spot * (1 + mny), 2)
            for kind in ("call", "put"):
                px = _pr.price(spot, K, t, 0.03, 0.0, iv, kind)
                if px < 0.02:
                    continue
                quotes.append(OptionQuote(dte, K, kind, round(px * 0.998, 4),
                                          round(px * 1.002, 4)))
    ch = OptionChain("MULTI", spot, 0.03, 0.0, quotes)
    by_dollar = scan_strikes(ch, F, p, rank_by="dollar")
    by_vp = scan_strikes(ch, F, p, rank_by="premium")
    assert by_dollar[0].expiry_days == 334          # dollars -> the longest, always
    assert by_vp[0].expiry_days < 334               # per-premium -> a shorter one
    # and the longest expiry is genuinely the WORST per unit of premium
    longest = max(s.edge_frac for s in by_dollar if s.expiry_days == 334)
    shortest = max(s.edge_frac for s in by_dollar if s.expiry_days == 19)
    assert shortest > longest


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
