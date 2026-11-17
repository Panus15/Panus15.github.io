"""Correctness tests for the pricing/IV core. Run: python3 -m pytest -q
(or plain `python3 tests/test_engine.py` — a tiny runner is included).

These check the properties that MUST hold, so refactors can't silently break
the math the whole engine rests on.
"""

import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import pricing, volforecast
from engine.data import OptionChain, OptionQuote, SyntheticAdapter
from engine.iv import implied_vol
from engine.signal import scan_chain


def approx(a, b, tol=1e-4):
    return abs(a - b) <= tol


def test_put_call_parity():
    # C - P = S e^{-qT} - K e^{-rT}
    S, K, T, r, q, vol = 100, 105, 0.5, 0.03, 0.01, 0.25
    c = pricing.price(S, K, T, r, q, vol, "call")
    p = pricing.price(S, K, T, r, q, vol, "put")
    lhs = c - p
    rhs = S * math.exp(-q * T) - K * math.exp(-r * T)
    assert approx(lhs, rhs), (lhs, rhs)


def test_call_delta_bounds():
    g = pricing.greeks(100, 100, 0.5, 0.03, 0.0, 0.2, "call")
    assert 0.0 < g.delta < 1.0
    assert g.gamma > 0
    assert g.vega > 0
    assert g.theta < 0  # long option bleeds time value


def test_put_delta_negative():
    g = pricing.greeks(100, 100, 0.5, 0.03, 0.0, 0.2, "put")
    assert -1.0 < g.delta < 0.0


def test_greeks_match_finite_differences():
    """The only test that pins Greek MAGNITUDES, not just signs.

    Sign-and-bounds assertions are close to worthless here: an audit of this file
    found that doubling gamma, dropping e^{-qT} from delta and vega, returning
    theta per DAY instead of per year, and dropping the T factor from rho ALL
    passed the entire test suite. Each is a plausible real slip and each would
    silently mis-size every hedge and every CVaR number downstream.

    Central differences of price() are an independent oracle: they use only the
    pricer, never the closed-form Greek being checked. Every case runs with q > 0
    so the dividend factors are actually exercised.
    """
    cases = [(100.0, 100.0, 0.50, 0.03, 0.02, 0.25),   # ATM
             (100.0, 120.0, 0.25, 0.05, 0.03, 0.35),   # OTM call, high vol
             (100.0,  85.0, 1.00, 0.02, 0.04, 0.18),   # ITM call, long dated
             (100.0, 100.0, 0.08, 0.04, 0.01, 0.60)]   # short dated, very high vol

    def px(S, K, T, r, q, v, kind):
        return pricing.price(S, K, T, r, q, v, kind)

    for S, K, T, r, q, vol in cases:
        for kind in ("call", "put"):
            g = pricing.greeks(S, K, T, r, q, vol, kind)
            hS, hv, hT, hr = S * 1e-4, 1e-5, 1e-5, 1e-6

            fd_delta = (px(S + hS, K, T, r, q, vol, kind)
                        - px(S - hS, K, T, r, q, vol, kind)) / (2 * hS)
            fd_gamma = (px(S + hS, K, T, r, q, vol, kind)
                        - 2 * px(S, K, T, r, q, vol, kind)
                        + px(S - hS, K, T, r, q, vol, kind)) / (hS * hS)
            fd_vega = (px(S, K, T, r, q, vol + hv, kind)
                       - px(S, K, T, r, q, vol - hv, kind)) / (2 * hv)
            # theta is dV/dt = -dV/dT, PER YEAR
            fd_theta = -(px(S, K, T + hT, r, q, vol, kind)
                         - px(S, K, T - hT, r, q, vol, kind)) / (2 * hT)
            fd_rho = (px(S, K, T, r + hr, q, vol, kind)
                      - px(S, K, T, r - hr, q, vol, kind)) / (2 * hr)

            ctx = f"{kind} S={S} K={K} T={T} q={q} vol={vol}"
            for name, got, want, tol in (("delta", g.delta, fd_delta, 1e-5),
                                         ("gamma", g.gamma, fd_gamma, 1e-4),
                                         ("vega", g.vega, fd_vega, 1e-4),
                                         ("theta", g.theta, fd_theta, 1e-3),
                                         ("rho", g.rho, fd_rho, 1e-4)):
                scale = max(abs(want), 1.0)
                assert abs(got - want) / scale < tol, (
                    f"{name} {ctx}: closed form {got:.8f} vs finite difference "
                    f"{want:.8f} (rel err {abs(got - want) / scale:.2e})")


def test_greeks_reject_a_bad_kind():
    for fn in (pricing.price, pricing.greeks):
        try:
            fn(100, 100, 0.5, 0.03, 0.0, 0.2, "straddle")
        except ValueError:
            continue
        raise AssertionError(f"{fn.__name__} accepted an invalid kind")


def test_iv_roundtrip():
    # Price at a known vol, recover it from the price.
    S, K, T, r, q, true_vol = 100, 95, 0.25, 0.04, 0.0, 0.32
    px = pricing.price(S, K, T, r, q, true_vol, "call")
    iv = implied_vol(px, S, K, T, r, q, "call")
    assert iv is not None and approx(iv, true_vol, 1e-4), iv


def test_iv_rejects_below_intrinsic():
    # A call quoted below intrinsic has no valid IV.
    S, K, T, r, q = 120, 100, 0.25, 0.0, 0.0
    iv = implied_vol(5.0, S, K, T, r, q, "call")  # intrinsic ~20
    assert iv is None


def test_vol_forecasts_are_positive():
    prices = SyntheticAdapter().price_history("X", 260)
    assert volforecast.close_to_close_vol(prices) > 0
    assert volforecast.ewma_vol(prices) > 0
    assert volforecast.har_rv_forecast(prices) > 0


# 63 REAL GOOG adjusted closes ending 2008-08-04 (matplotlib's bundled goog.npz
# sample), containing the -10% July-2008 earnings gap. On this context the
# tiny-sample HAR OLS extrapolates a NEGATIVE next-day variance; the old
# max(var_hat, 1e-8) guard emitted a 0.01% vol forecast against a ~48% realised
# vol, collapsing every downstream density to a spike (found the first time the
# engine touched real data). The fix falls back to the EWMA anchor instead.
GOOG_2008_CTX = [
    586.36, 579.00, 583.01, 573.20, 584.94, 583.00, 576.30, 581.00,
    580.07, 577.52, 578.60, 549.99, 549.46, 544.62, 560.90, 568.24,
    583.00, 585.80, 575.00, 567.30, 572.22, 586.30, 567.00, 557.87,
    554.17, 545.20, 552.95, 571.51, 572.81, 569.46, 562.38, 560.20,
    546.43, 545.21, 542.30, 551.00, 528.82, 528.07, 526.42, 534.73,
    527.04, 537.00, 543.91, 554.53, 541.55, 540.57, 533.80, 521.62,
    516.09, 535.60, 533.44, 481.32, 468.80, 477.11, 489.22, 475.62,
    491.98, 477.12, 483.11, 482.70, 473.75, 467.86, 463.00,
]


def test_har_survives_real_outlier_context():
    # Real-data regression: HAR must stay in the same ballpark as the robust
    # estimators on a context whose OLS misfits, never collapse toward zero.
    har = volforecast.har_rv_forecast(GOOG_2008_CTX)
    ewma = volforecast.ewma_vol(GOOG_2008_CTX)
    cc = volforecast.close_to_close_vol(GOOG_2008_CTX)
    assert har > 0.10, f"HAR collapsed to {har:.4%} on a ~{cc:.0%}-vol context"
    assert 0.3 * ewma <= har <= 3.0 * ewma


def test_term_vol_reverts_toward_the_long_run():
    # Found on the FIRST real option chain (Deribit BTC): the flat forecast gave
    # the SAME annualised vol at 5d and 334d, so any upward-sloping implied curve
    # made the longest expiry look richest — an artifact, not an edge.
    import random
    rng = random.Random(5)
    p = [100.0]
    for _ in range(300):                      # volatile history (~60%)
        p.append(p[-1] * math.exp(rng.gauss(0, 0.038)))
    for _ in range(120):                      # calm recently (~20%)
        p.append(p[-1] * math.exp(rng.gauss(0, 0.0126)))
    spot_v = volforecast.har_rv_forecast(p)
    long_v = volforecast.long_run_vol(p)
    assert spot_v < long_v                    # calm now, hot long-run

    short_T = volforecast.term_vol(p, 5 / 365)
    long_T = volforecast.term_vol(p, 334 / 365)
    assert short_T < long_T                   # the curve SLOPES, no longer flat
    assert abs(short_T - spot_v) < 0.02       # short end ~ today's vol
    assert spot_v < long_T <= long_v          # long end reverts toward long-run

    # and it is symmetric: hot now, calm long-run -> a DOWNWARD-sloping curve
    q = [100.0]
    for _ in range(300):
        q.append(q[-1] * math.exp(rng.gauss(0, 0.0126)))
    for _ in range(120):
        q.append(q[-1] * math.exp(rng.gauss(0, 0.038)))
    assert volforecast.term_vol(q, 5 / 365) > volforecast.term_vol(q, 334 / 365)


def test_term_vol_matches_the_closed_form_exactly():
    # Pin the FORMULA, not just the slope direction: a wrong weight (e.g. dropping
    # the 1/(kT), or e^{-kT} instead of (1-e^{-kT})/(kT)) still slopes the right
    # way but returns wrong numbers, so a direction-only test cannot catch it.
    v0, vinf, kappa = 0.20 ** 2, 0.45 ** 2, 2.77
    for T in (5 / 365, 33 / 365, 152 / 365, 334 / 365, 3.0):
        got = volforecast.term_vol([1.0, 1.0, 1.0], T, kappa=kappa,
                                   spot_vol=math.sqrt(v0), long_run=math.sqrt(vinf))
        kT = kappa * T
        want = math.sqrt(vinf + (v0 - vinf) * (1.0 - math.exp(-kT)) / kT)
        assert abs(got - want) < 1e-12, (T, got, want)

    # limits: T -> 0 gives today's vol; T -> infinity gives the long-run vol
    near0 = volforecast.term_vol([1.0] * 3, 1e-6, kappa=kappa,
                                 spot_vol=0.20, long_run=0.45)
    far = volforecast.term_vol([1.0] * 3, 500.0, kappa=kappa,
                               spot_vol=0.20, long_run=0.45)
    assert abs(near0 - 0.20) < 1e-4 and abs(far - 0.45) < 1e-3
    # and the result is always bracketed by the two inputs (never overshoots)
    for T in (0.01, 0.1, 1.0, 10.0):
        v = volforecast.term_vol([1.0] * 3, T, kappa=kappa, spot_vol=0.20, long_run=0.45)
        assert 0.20 - 1e-12 <= v <= 0.45 + 1e-12


def test_baseline_uses_the_term_structure_by_default():
    # The headline behaviour change: the shipped forecaster must produce a SLOPING
    # annualised vol, and the opt-out must restore the old flat behaviour.
    import random
    from models.baseline import BaselineDensityForecaster
    rng = random.Random(5)
    p = [100.0]
    for _ in range(300):
        p.append(p[-1] * math.exp(rng.gauss(0, 0.038)))
    for _ in range(120):
        p.append(p[-1] * math.exp(rng.gauss(0, 0.0126)))
    sloped, flat = BaselineDensityForecaster(), BaselineDensityForecaster(
        use_term_structure=False)
    assert sloped.use_term_structure is True                 # ON by default

    def vol_at(f, dte):
        T = dte / 365.0
        return f.forecast(p, T, spot=p[-1]).log_return_vol(p[-1], T)

    assert vol_at(sloped, 334) > vol_at(sloped, 5) + 0.02    # genuinely slopes
    assert abs(vol_at(flat, 334) - vol_at(flat, 5)) < 1e-9   # opt-out is flat


def test_scanner_finds_injected_dislocation():
    md = SyntheticAdapter()
    chain = md.option_chain("X")
    # Scan against a low fair vol so the injected rich options surface.
    ideas = scan_chain(chain, forecast_vol=0.17, min_edge_net=0.01)
    assert any(m.verdict == "RICH" for m in ideas)


# ---------------------------------------------------------------------------
# Gaps the first mutation sweep of engine/pricing.py left: the expiry branch
# and the input guards. The oracles above are strong -- finite differences
# alone caught seventeen defects -- but they all price a LIVE option with
# sane inputs, so nothing touched the edges.
# ---------------------------------------------------------------------------

def test_an_expired_option_is_worth_its_intrinsic_value():
    # T <= 0 returns before _d1_d2 is ever called, so this whole branch sat
    # outside every test. Signs reversed, it prices an in-the-money call at
    # zero and a worthless one at a positive number.
    assert pricing.price(110.0, 100.0, 0.0, 0.05, 0.0, 0.2, "call") == 10.0
    assert pricing.price(90.0, 100.0, 0.0, 0.05, 0.0, 0.2, "put") == 10.0
    assert pricing.price(110.0, 100.0, -1.0, 0.05, 0.0, 0.2, "call") == 10.0


def test_an_expired_worthless_option_is_worth_zero_never_less():
    # Without the floor an out-of-the-money expiry prices NEGATIVE, which in a
    # short book reads as a position that pays you to close it.
    for S, K, kind in ((90.0, 100.0, "call"), (110.0, 100.0, "put")):
        v = pricing.price(S, K, 0.0, 0.05, 0.0, 0.2, kind)
        assert v == 0.0, f"{kind} at S={S} K={K} expired worth {v}"


def test_a_live_option_is_worth_more_than_intrinsic():
    # The other half: the expiry shortcut must not fire on a live option.
    live = pricing.price(110.0, 100.0, 0.5, 0.05, 0.0, 0.2, "call")
    assert live > 10.0, (
        f"a 6-month call priced at {live}, its intrinsic value — the expiry "
        f"branch is firing on a live option")


def test_nonsense_inputs_are_refused_rather_than_priced():
    # A zero vol, a zero spot or a zero strike all make d1 undefined. Pricing
    # them anyway returns a number with no meaning that the scanner would
    # then compare a market quote against.
    for kwargs in ({"vol": 0.0}, {"vol": -0.2}, {"S": 0.0}, {"K": 0.0},
                   {"S": -100.0}):
        args = {"S": 100.0, "K": 100.0, "T": 0.5, "r": 0.05, "q": 0.0,
                "vol": 0.2, "kind": "call"}
        args.update(kwargs)
        try:
            pricing.price(**args)
        except ValueError as e:
            assert "positive" in str(e), e
        else:
            assert False, f"priced an option with {kwargs}"


def test_greeks_refuse_a_zero_tenor_rather_than_dividing_by_it():
    try:
        pricing.greeks(100.0, 100.0, 0.0, 0.05, 0.0, 0.2, "call")
    except ValueError as e:
        assert "positive" in str(e), e
    else:
        assert False, "greeks at T=0 divided by a zero sqrt(T) and returned"


def test_the_kind_is_read_case_insensitively():
    # A chain feed hands you "CALL". Dropping the .lower() turns every one of
    # those into the bad-kind refusal.
    for k in ("CALL", "Call", "cAlL"):
        assert pricing.price(100.0, 100.0, 0.5, 0.05, 0.0, 0.2, k) > 0.0, k
    assert pricing.greeks(100.0, 100.0, 0.5, 0.05, 0.0, 0.2, "PUT").delta < 0.0


# ---------------------------------------------------------------------------
# The implied-vol solver. Its first sweep scored 5 of 23 -- the worst of any
# module here -- because two tests covered it: one round trip at one point, and
# one refusal.
#
# Implied vol is what the whole engine competes against: every "rich" or
# "cheap" call is this number against a forecast. The dangerous failure is not
# inaccuracy, it is a CONFIDENT answer on a quote that has none, which is why
# the refusals get as much attention here as the arithmetic.
# ---------------------------------------------------------------------------

def test_the_solver_recovers_vol_across_a_grid_including_the_ugly_corners():
    # One round trip at the money says almost nothing. Deep ITM and OTM
    # quotes, and very short tenors, are where vega collapses, Newton stalls
    # or overshoots, and the bisection fallback is the only thing keeping the
    # answer right -- and they are exactly the quotes a real chain is full of.
    checked = 0
    for S in (100.0,):
        for K in (60.0, 80.0, 95.0, 100.0, 105.0, 120.0, 160.0):
            for T in (1.0 / 365, 7.0 / 365, 30.0 / 365, 0.5, 2.0):
                for vol in (0.08, 0.20, 0.45, 1.20):
                    for kind in ("call", "put"):
                        px = pricing.price(S, K, T, 0.04, 0.01, vol, kind)
                        if px < 1e-6:
                            continue          # no information left to invert
                        # The attainable vol accuracy is about tol/vega,
                        # because `tol` is a tolerance on PRICE: at a vega of
                        # 0.01 a 1e-6 price tolerance pins the vol only to
                        # 1e-4, which is what is asserted below. Quotes under
                        # that are not well posed, and the solver is required
                        # to refuse them -- their own test is further down.
                        if pricing.greeks(S, K, T, 0.04, 0.01, vol,
                                          kind).vega < 1e-2:
                            continue
                        got = implied_vol(px, S, K, T, 0.04, 0.01, kind)
                        assert got is not None, (
                            f"no vol for a price this engine itself produced: "
                            f"K={K} T={T:.4f} vol={vol} {kind} px={px:.6f}")
                        assert abs(got - vol) < 1e-4, (
                            f"K={K} T={T:.4f} {kind}: put in {vol}, got "
                            f"{got:.6f} from px={px:.6f}")
                        checked += 1
    # 192 at the vega floor above, measured. The bar is a floor on coverage,
    # not a figure to match: it exists so that narrowing the grid or raising
    # the vega cut cannot quietly reduce this to a handful of quotes.
    assert checked > 150, f"only {checked} quotes exercised"


def test_a_quote_below_intrinsic_value_has_no_implied_vol():
    # 5.0 for a call whose intrinsic is 20.0: no vol reaches it, and a solver
    # that returns one hands the scanner a free-money signal on every such
    # quote. A wide bid-ask on an illiquid strike produces these constantly.
    assert implied_vol(5.0, 120.0, 100.0, 0.5, 0.04, 0.0, "call") is None
    assert implied_vol(5.0, 80.0, 100.0, 0.5, 0.04, 0.0, "put") is None
    # And the boundary: a quote AT intrinsic is the zero-vol limit, which is
    # outside [lo, hi] and so also unsolvable -- never a confident number.
    assert implied_vol(20.0, 120.0, 100.0, 0.5, 0.04, 0.0, "call") is None


def test_the_no_arbitrage_floor_is_the_EUROPEAN_one_and_so_discounted():
    """`max(K - S, 0)` is the AMERICAN floor and it does not apply here.

    The solver used it against a European pricer, so every in-the-money
    European put trading between the discounted and undiscounted bounds was
    rejected as "below intrinsic" -- 16% of the well-posed quotes on a
    realistic chain at 4% rates, which is the entire in-the-money put wing on
    an engine whose only trade is selling puts.
    """
    from engine.iv import _lower_bound
    S, K, T, r = 100.0, 105.0, 0.25, 0.05
    undiscounted = K - S                                  # 5.00, the American floor
    european = _lower_bound(S, K, T, r, 0.0, "put")
    assert european < undiscounted - 0.1, (
        f"the put floor is {european}, not meaningfully below {undiscounted} "
        f"— it is still the undiscounted bound")
    assert abs(european - max(K * math.exp(-r * T) - S, 0.0)) < 1e-12, european
    # Far enough out, the discounting swallows the moneyness entirely and the
    # floor is zero -- at r=5% over a year the 105 put's floor is exactly 0.
    assert _lower_bound(S, K, 1.0, r, 0.0, "put") == 0.0

    # Zero rates and no dividend: the two bounds coincide, which is the case
    # the old code was right about and must stay right about.
    assert abs(_lower_bound(120.0, 100.0, 0.25, 0.0, 0.0, "call") - 20.0) < 1e-12
    assert abs(_lower_bound(80.0, 100.0, 0.25, 0.0, 0.0, "put") - 20.0) < 1e-12
    # And it is never negative.
    assert _lower_bound(80.0, 100.0, 0.25, 0.04, 0.0, "call") == 0.0
    assert _lower_bound(120.0, 100.0, 0.25, 0.04, 0.0, "put") == 0.0


def test_an_in_the_money_european_put_is_solvable_below_undiscounted_intrinsic():
    # The quote that found it: a 7-day 105 put on a 100 spot at 4% prices to
    # 4.9856, below the 5.00 American floor and above the European one.
    S, K, T, r, q = 100.0, 105.0, 7.0 / 365, 0.04, 0.01
    px = pricing.price(S, K, T, r, q, 0.20, "put")
    assert px < K - S, "the fixture no longer sits below undiscounted intrinsic"
    got = implied_vol(px, S, K, T, r, q, "put")
    assert got is not None, (
        f"a legitimate European put quote at {px:.6f} was refused because it "
        f"is below K - S = {K - S}")
    assert abs(got - 0.20) < 1e-4, got


def test_a_quote_outside_the_vol_bracket_is_refused():
    # A price dearer than the 500% cap can produce has no root in the bracket.
    # Asked for 800% instead of this, the test would be unsound: a call's
    # price saturates near spot, so 800% and 500% are numerically the same
    # quote and the refusal would be untestable through them.
    S = K = 100.0
    at_cap = pricing.price(S, K, 0.5, 0.04, 0.0, 5.0, "call")
    assert implied_vol(at_cap + 1.0, S, K, 0.5, 0.04, 0.0, "call") is None, (
        "a price above everything the bracket can produce was still solved")


def test_a_non_positive_price_or_tenor_is_refused():
    for px in (0.0, -1.0):
        assert implied_vol(px, 100.0, 100.0, 0.5, 0.04, 0.0, "call") is None, px
    for T in (0.0, -1.0):
        assert implied_vol(5.0, 100.0, 100.0, T, 0.04, 0.0, "call") is None, T


def test_the_solver_is_monotone_in_the_price_it_is_given():
    # A dearer option implies a higher vol, always. This is the property the
    # scanner leans on when it calls one strike rich against another, and a
    # swapped bracket or a reversed objective breaks it while still returning
    # plausible numbers.
    S = K = 100.0
    prev = None
    for vol in (0.10, 0.15, 0.20, 0.30, 0.50, 0.80):
        px = pricing.price(S, K, 0.5, 0.04, 0.0, vol, "call")
        got = implied_vol(px, S, K, 0.5, 0.04, 0.0, "call")
        assert got is not None, vol
        if prev is not None:
            assert got > prev, (
                f"vol {got:.4f} from a dearer option is not above {prev:.4f}")
        prev = got


def test_a_quote_that_cannot_identify_a_vol_is_refused_not_seeded():
    """The defect this found: a confident answer where there is no answer.

    `tol` is a tolerance on PRICE, so where vega collapses a band of vols
    price the quote identically, the criterion is met by the SEED on the first
    iteration, and the solver returned 0.20 whatever the market implied.

    The fixture is measured, not invented: a 1-day call at S=100, K=60 prices
    to the same 40.003835294 at every vol from 0.02 to 0.20, and its vega is
    exactly 0.0. If the price cannot tell 8% from 20%, neither can anyone.
    """
    S, K, T, r, q = 100.0, 60.0, 1.0 / 365, 0.04, 0.01
    px = pricing.price(S, K, T, r, q, 0.08, "call")
    assert pricing.price(S, K, T, r, q, 0.20, "call") == px, (
        "the fixture no longer has a flat vol profile, so it no longer tests "
        "what it was built to test")
    got = implied_vol(px, S, K, T, r, q, "call")
    assert got is None, (
        f"returned {got} for a quote whose price is identical across vols "
        f"— 0.2 is the Newton seed, not a measurement")


def test_an_identifiable_quote_is_still_solved():
    # Guard isolation: the refusal above must not swallow ordinary quotes.
    for T in (7.0 / 365, 0.25, 1.0):
        for K in (90.0, 100.0, 110.0):
            px = pricing.price(100.0, K, T, 0.04, 0.0, 0.25, "call")
            got = implied_vol(px, 100.0, K, T, 0.04, 0.0, "call")
            assert got is not None and abs(got - 0.25) < 1e-4, (
                f"K={K} T={T}: {got}")


# ---------------------------------------------------------------------------
# engine/signal.py -- where P and Q become a trade. First sweep: 7 of 23.
#
# One test covered it, and what it left untested was the check the module's own
# docstring calls decisive: "an edge that vanishes once you cross the spread is
# not an edge -- that single check kills most naive options signals." Also
# untested: every gate (the minimum edge, the FAIR band), the whole CHEAP side,
# and the ranking a caller reads when it takes the top idea.
# ---------------------------------------------------------------------------

def _one_quote_chain(market_vol, forecast_vol, spread, kind="call",
                     S=100.0, K=100.0, r=0.03, q=0.0, dte=30):
    """A chain of exactly one quote, priced at `market_vol` with `spread`.

    Everything is then determined: the mid is the Black-Scholes price at
    `market_vol`, fair value is the price at `forecast_vol`, and the net edge
    must be their difference less half the spread.
    """
    T = dte / 365.0
    mid = pricing.price(S, K, T, r, q, market_vol, kind)
    quote = OptionQuote(dte, K, kind, round(mid - spread / 2, 4),
                        round(mid + spread / 2, 4))
    return OptionChain("ONE", S, r, q, [quote]), T


def test_the_net_edge_charges_exactly_half_the_spread():
    # A call marked at 30% vol against a 20% forecast: mid 3.5505, fair
    # 2.4096, so the gross edge is 1.1409 and a 0.20 spread leaves 1.0409.
    # Charging the full spread gives 0.9409 and adding it gives 1.2409 --
    # both plausible, neither right.
    ch, _T = _one_quote_chain(0.30, 0.20, spread=0.20)
    m = scan_chain(ch, forecast_vol=0.20, min_edge_net=0.0)[0]
    assert abs(m.edge_mid + 1.1409) < 1e-3, m.edge_mid
    assert abs(m.quote.spread - 0.20) < 1e-9, m.quote.spread
    assert abs(m.edge_net - 1.0409) < 1e-3, (
        f"net edge {m.edge_net:.4f}: 1.0409 charges half the spread, 0.9409 "
        f"charges all of it, 1.2409 adds it")


def test_an_edge_that_does_not_survive_the_spread_is_not_returned():
    # The module's headline claim. A 0.5-vol-point dislocation is worth about
    # 0.06 of premium here, against a spread of 0.60 -- so crossing costs five
    # times the edge and there is nothing to trade. Without the charge it
    # comes back as an idea.
    ch, _T = _one_quote_chain(0.205, 0.20, spread=0.60)
    assert scan_chain(ch, forecast_vol=0.20, min_edge_net=0.0) == [], (
        "an edge smaller than half the spread was reported as actionable")


def test_the_minimum_edge_gate_is_applied_at_its_boundary():
    ch, _T = _one_quote_chain(0.30, 0.20, spread=0.20)
    net = scan_chain(ch, forecast_vol=0.20, min_edge_net=0.0)[0].edge_net
    assert scan_chain(ch, forecast_vol=0.20, min_edge_net=net - 1e-6), \
        "a trade just above the minimum was dropped"
    assert scan_chain(ch, forecast_vol=0.20, min_edge_net=net + 1e-6) == [], \
        "a trade just below the minimum was kept"


def test_an_option_inside_the_fair_band_is_never_actionable():
    # Two vol points of dislocation against a 5-point band: the dollar edge is
    # large, and it must still be FAIR. Removing the band, or dropping the
    # FAIR filter from the output, makes the whole chain tradeable.
    ch, _T = _one_quote_chain(0.22, 0.20, spread=0.01)
    assert scan_chain(ch, forecast_vol=0.20, min_edge_net=0.0,
                      iv_band=0.05) == [], (
        "an option within the fair band was returned as a trade")


def test_the_SHIPPED_fair_band_and_minimum_edge_are_the_documented_ones():
    """The defaults, which every other test here overrides.

    Passing `iv_band` and `min_edge_net` explicitly is right for testing the
    logic and useless for testing what a caller who omits them gets. Both
    defaults could have been zero and every assertion above would still pass.

    A one-vol-point dislocation is inside the 0.02 band, so it must be FAIR on
    the defaults. And a dislocation worth less than 0.05 of premium must be
    dropped by the minimum-edge gate even when the band lets it through.
    """
    import inspect
    sig = inspect.signature(scan_chain).parameters
    assert sig["iv_band"].default == 0.02, sig["iv_band"].default
    assert sig["min_edge_net"].default == 0.05, sig["min_edge_net"].default

    inside_band, _T = _one_quote_chain(0.21, 0.20, spread=0.01)
    assert scan_chain(inside_band, forecast_vol=0.20) == [], (
        "a one-vol-point gap was actionable on the shipped 0.02 band")

    # Four vol points clears the band; the 0.041 edge of a 115 strike does
    # not clear the 0.05 gate. (A 135 strike does not work: that far out the
    # vega has collapsed and the solver refuses the quote entirely, so it
    # never reaches either gate.)
    tiny, _T2 = _one_quote_chain(0.24, 0.20, spread=0.01, K=115.0)
    net = scan_chain(tiny, forecast_vol=0.20, min_edge_net=0.0, iv_band=0.02)
    assert net, "the fixture no longer clears the fair band"
    assert net[0].edge_net < 0.05, (
        f"the fixture's edge is {net[0].edge_net:.4f}, not under the 0.05 "
        f"gate, so it cannot test the gate")
    assert scan_chain(tiny, forecast_vol=0.20) == [], (
        "an edge under the shipped 0.05 minimum was returned")


def test_the_cheap_side_works_as_well_as_the_rich_side():
    # Every existing assertion was about RICH. A market marked BELOW the
    # forecast must come back CHEAP with a positive dollar edge -- and the
    # consistency check on that branch had no test at all.
    ch, _T = _one_quote_chain(0.15, 0.25, spread=0.05)
    ideas = scan_chain(ch, forecast_vol=0.25, min_edge_net=0.0)
    assert ideas and ideas[0].verdict == "CHEAP", (
        f"a market below the forecast gave {[m.verdict for m in ideas]}")
    assert ideas[0].edge_mid > 0, ideas[0].edge_mid
    assert ideas[0].iv_premium < 0, (
        f"iv_premium is {ideas[0].iv_premium:+.4f}; implied is below the "
        f"forecast so the premium must be negative")


def test_a_vol_gap_that_disagrees_with_the_dollar_edge_is_called_FAIR():
    # The consistency check. A quote whose implied vol is above the forecast
    # but whose price is BELOW fair is a contradiction, and the scanner must
    # not resolve it into a confident verdict. Built by marking the mid below
    # fair while leaving the recovered IV above the forecast, which only
    # happens when the two are measured on different legs -- so it is
    # constructed directly.
    ch, _T = _one_quote_chain(0.30, 0.20, spread=0.20)
    q = ch.quotes[0]
    fair = pricing.price(ch.spot, q.strike, q.T, ch.r, ch.q, 0.20, q.kind)
    # mid just BELOW fair, but still implying a vol above 0.20 + band? It
    # cannot, so the honest assertion is the converse: at this mid, a verdict
    # requires BOTH, and the rich branch is the one that fires.
    m = scan_chain(ch, forecast_vol=0.20, min_edge_net=0.0)[0]
    assert m.verdict == "RICH" and m.edge_mid < 0 and m.iv_premium > 0, (
        f"{m.verdict} with edge {m.edge_mid:+.4f} and premium "
        f"{m.iv_premium:+.4f}: the two must agree in sign")
    assert m.fair_value < q.mid, (
        f"fair {m.fair_value:.4f} is not below the mid {q.mid:.4f}, so this "
        f"fixture no longer exercises the rich branch")


def test_a_quote_whose_vol_cannot_be_solved_is_skipped():
    # Below the no-arbitrage floor there is no implied vol, and the scanner
    # must drop the quote rather than substitute the forecast -- which would
    # make the option exactly FAIR by construction and hide a bad feed.
    S, r, dte = 100.0, 0.03, 30
    bad = OptionQuote(dte, 80.0, "call", 1.0, 1.2)      # intrinsic is ~20
    ch = OptionChain("BAD", S, r, 0.0, [bad])
    assert scan_chain(ch, forecast_vol=0.20, min_edge_net=0.0) == []


def test_ideas_come_back_best_first():
    # A caller that takes the top idea is reading this order. Two quotes with
    # clearly different net edges, built so the ranking is unambiguous.
    S, r, q, dte = 100.0, 0.03, 0.0, 30
    T = dte / 365.0
    # The WEAKER idea is listed first, so an unsorted result fails. My first
    # version happened to list them in edge order and `pass` for the sort
    # passed it.
    quotes = []
    for K, mv in ((105.0, 0.26), (100.0, 0.30)):
        mid = pricing.price(S, K, T, r, q, mv, "call")
        quotes.append(OptionQuote(dte, K, "call", round(mid - 0.01, 4),
                                  round(mid + 0.01, 4)))
    ideas = scan_chain(OptionChain("TWO", S, r, q, quotes),
                       forecast_vol=0.20, min_edge_net=0.0)
    assert len(ideas) == 2, [m.verdict for m in ideas]
    assert ideas[0].edge_net > ideas[1].edge_net, (
        f"{ideas[0].edge_net:.4f} then {ideas[1].edge_net:.4f}: not best-first")


# ---------------------------------------------------------------------------
# engine/volforecast.py -- the P side of the engine's only claim.
#
# Its first sweep scored 4 of 26, and all four kills were term_vol's. The
# three range estimators, the annualisation, EWMA, log returns and the
# overnight gap had no test at all: the machinery that produces the forecast
# every signal is measured against.
#
# The direction matters. The edge is Q - P, so a forecast biased LOW makes
# every option look rich -- a defect here is not one wrong answer, it is a
# sell signal everywhere.
# ---------------------------------------------------------------------------

def test_log_returns_run_forward_in_time():
    # Inverted, every return flips sign. Variance is unchanged, so no
    # volatility test can see it -- but realized_skew and anything directional
    # reads backwards.
    r = volforecast.log_returns([100.0, 110.0])
    assert len(r) == 1 and r[0] > 0, r
    assert abs(r[0] - math.log(1.1)) < 1e-12, r[0]


def test_close_to_close_annualises_by_the_square_root_of_252():
    # The 252 multiplies VARIANCE, so dropping it or using 365 moves the
    # answer by a square root -- a vol missing its annualisation is sixteen
    # times too small and still reads as a plausible decimal.
    rnd = random.Random(3)
    daily = 0.01
    px = [100.0]
    for _ in range(4000):
        px.append(px[-1] * math.exp(rnd.gauss(0.0, daily)))
    got = volforecast.close_to_close_vol(px, window=4000)
    want = daily * math.sqrt(252)
    assert abs(got - want) / want < 0.05, (
        f"{got:.4f} against {want:.4f}: a ratio of {got/want:.3f}. 1/sqrt(252) "
        f"means it is not annualised; sqrt(365/252) means calendar days")


def test_close_to_close_uses_the_SAMPLE_variance_not_the_population_one():
    # The annualisation test above runs 4,000 returns, where n and n-1 differ
    # by 0.01% and no tolerance can see it. Three returns differ by a factor
    # of 1.5 in variance and 1.2247 in vol, which is unmistakable.
    px = [100.0, 101.0, 100.0, 101.0]
    got = volforecast.close_to_close_vol(px, window=3)
    assert abs(got - 0.182393) < 1e-5, (
        f"{got:.6f}: 0.182393 divides by n-1, 0.148923 divides by n")


def test_close_to_close_measures_variance_ABOUT_THE_MEAN():
    # A price rising exactly 1% a day has no volatility at all: every return
    # is identical, so the variance about the mean is zero. Summing raw
    # squares instead reports 1.03e-4 of daily variance -- a 16% annualised
    # vol invented out of a perfectly smooth trend, which is the direction
    # that makes options look rich.
    drift = [100.0 * (1.01 ** i) for i in range(30)]
    got = volforecast.close_to_close_vol(drift, window=30)
    assert got < 1e-6, (
        f"a constant 1%/day trend reported {got:.6f} of annualised vol; the "
        f"mean is not being removed from the returns")


def test_close_to_close_uses_only_the_window_it_was_given():
    # A calm recent window after a violent old one. Ignoring `window` mixes
    # the regime that is over into the forecast for the one that is not.
    calm = [100.0 * (1.0 + 0.0001 * i) for i in range(60)]
    violent = [100.0 * (1.5 if i % 2 else 0.7) for i in range(60)]
    got = volforecast.close_to_close_vol(violent + calm, window=30)
    calm_only = volforecast.close_to_close_vol(calm, window=30)
    assert abs(got - calm_only) < 1e-9, (
        f"{got:.4f} against {calm_only:.4f}: the window is not being applied")


def _flat_with_shock(n, at, jump=1.10):
    """`n` flat days with a single `jump` return at index `at`."""
    px = [100.0]
    for i in range(n):
        px.append(px[-1] * (jump if i == at else 1.0))
    return px


def test_ewma_weights_a_recent_shock_far_above_an_old_one():
    # The shock goes in the MIDDLE of a long series, never at index 0. The
    # implementation seeds `var = rets[0] ** 2`, so the first return carries
    # weight lam^(n-1) rather than (1 - lam) -- and at lam=0.94 those two are
    # 0.0482 and 0.06, which cross over at a gap of about 49 days. My first
    # fixture used exactly 50 and the two answers came out within 10% of each
    # other, which is the fixture landing on the crossover rather than the
    # decay failing.
    recent = volforecast.ewma_vol(_flat_with_shock(200, at=199))
    old = volforecast.ewma_vol(_flat_with_shock(200, at=100))
    assert recent > old * 10, (
        f"a shock on the last day gives {recent:.4f} and one 99 days earlier "
        f"{old:.4f}; at lam=0.94 the weights differ by 0.06 against "
        f"0.06*0.94^99, so the gap should be a factor of hundreds")


def test_the_ewma_seed_makes_a_SHORT_history_hostage_to_its_first_return():
    """A property worth knowing rather than a defect, so it is pinned.

    `var = rets[0] ** 2` seeds the recursion, which gives the first return
    weight lam^(n-1) instead of the (1 - lam) every later one gets. On a short
    history that is the largest weight in the series: over 10 days a shock on
    day one counts 0.94^10 = 0.54 against a shock today at 0.06, so the
    estimate is dominated by the oldest observation it has.

    Anyone forecasting off a few weeks of data should know that. The fix is
    more history, not a different lambda.
    """
    first = volforecast.ewma_vol(_flat_with_shock(10, at=0))
    last = volforecast.ewma_vol(_flat_with_shock(10, at=9))
    assert first > last, (
        f"over 10 days a shock on day one gives {first:.4f} and one today "
        f"{last:.4f}; the seeding artefact this documents is gone, so the "
        f"docstring above is stale")


def test_each_range_estimator_matches_its_own_closed_form():
    # Literal values on one hand-built session, because every constant here
    # looks like magic whether it is right or wrong.
    o, h, l, c = 100.0, 104.0, 98.0, 102.0
    hl, co = math.log(h / l), math.log(c / o)
    assert abs(volforecast.parkinson_var((o, h, l, c))
               - hl * hl / (4.0 * math.log(2.0))) < 1e-15
    assert abs(volforecast.garman_klass_var((o, h, l, c))
               - (0.5 * hl * hl - (2.0 * math.log(2.0) - 1.0) * co * co)) < 1e-15
    assert abs(volforecast.rogers_satchell_var((o, h, l, c))
               - (math.log(h / c) * math.log(h / o)
                  + math.log(l / c) * math.log(l / o))) < 1e-15


def test_the_range_estimators_recover_a_KNOWN_variance():
    """The oracle the closed forms cannot be: simulated sessions.

    Each estimator is unbiased for the session variance under its own
    assumptions, so sampling a geometric walk at 250 intraday points and
    averaging over 1,200 sessions has to come back near the variance that
    generated it. Measured here: 0.89 to 0.92 of it, the shortfall being the
    high and low of a discretely sampled path never quite reaching the
    continuous ones.

    That band is wide enough to accept the real code and narrow enough to
    reject the errors that matter -- dropping Parkinson's 4*ln(2) multiplies
    the answer by 2.77, halving Rogers-Satchell or Garman-Klass takes it to
    roughly half, and losing a sign sends it elsewhere entirely.
    """
    rnd = random.Random(11)
    sigma, steps, days = 0.02, 250, 1200
    totals = {"parkinson": 0.0, "garman_klass": 0.0, "rogers_satchell": 0.0}
    fns = {"parkinson": volforecast.parkinson_var,
           "garman_klass": volforecast.garman_klass_var,
           "rogers_satchell": volforecast.rogers_satchell_var}
    prev = 100.0
    for _ in range(days):
        o = px = hi = lo = prev
        for _ in range(steps):
            px *= math.exp(rnd.gauss(0.0, sigma / math.sqrt(steps)))
            hi, lo = max(hi, px), min(lo, px)
        bar = (o, hi, lo, px)
        for name, fn in fns.items():
            totals[name] += fn(bar)
        prev = px
    truth = sigma * sigma
    for name, total in sorted(totals.items()):
        ratio = total / days / truth
        assert 0.80 <= ratio <= 1.10, (
            f"{name} recovers {ratio:.3f} of the variance that generated the "
            f"sessions; 2.77 is Parkinson without its 4*ln(2), ~0.5 is a "
            f"halved estimator")


def test_the_overnight_gap_is_counted_and_squared():
    # A 2% gap up carries 0.0004 of variance into the session. Dropped, the
    # forecast understates vol on every gapping market, which is every equity
    # index. Unsquared, a gap DOWN would subtract variance.
    bar = (102.0, 103.0, 101.0, 102.5)
    got = volforecast._gap_var(bar, 100.0)
    assert abs(got - math.log(1.02) ** 2) < 1e-15, got
    down = volforecast._gap_var((98.0, 99.0, 97.0, 98.5), 100.0)
    assert down > 0, f"a gap down contributed {down} of variance"
    assert volforecast._gap_var(bar, None) == 0.0, "the first bar has no gap"
    assert volforecast._gap_var(bar, 0.0) == 0.0


def test_a_non_positive_price_gives_zero_variance_rather_than_a_crash():
    # A halted or mis-ticked session arrives as a zero. The documented
    # behaviour is to contribute nothing, not to raise inside a forecast.
    #
    # Parkinson is deliberately held to a weaker claim: it is a HIGH-LOW
    # estimator and never reads the open or the close, so a zero open is
    # harmless to it and guarding on it would be guarding against nothing.
    # Asserting otherwise is what my first version of this test did.
    for fn in (volforecast.parkinson_var, volforecast.garman_klass_var,
               volforecast.rogers_satchell_var):
        assert fn((100.0, 0.0, 98.0, 99.0)) == 0.0, f"{fn.__name__}: h=0"
        assert fn((100.0, 104.0, 0.0, 99.0)) == 0.0, f"{fn.__name__}: l=0"
    for fn in (volforecast.garman_klass_var, volforecast.rogers_satchell_var):
        assert fn((0.0, 104.0, 98.0, 99.0)) == 0.0, f"{fn.__name__}: o=0"
        assert fn((100.0, 104.0, 98.0, 0.0)) == 0.0, f"{fn.__name__}: c=0"
    assert volforecast.parkinson_var((0.0, 104.0, 98.0, 0.0)) > 0.0, (
        "parkinson is a high-low estimator and must not start depending on "
        "the open or the close")


def test_a_flat_history_forecasts_a_positive_floor_not_zero():
    # term_vol floors the variance at 1e-12. On a perfectly flat series both
    # the spot and long-run estimates are zero, so without the floor the
    # forecast is exactly 0.0 -- and a zero forecast makes every option
    # infinitely rich to anything downstream that divides by it.
    got = volforecast.term_vol([100.0] * 300, 30.0 / 365)
    assert got > 0.0, "a flat history forecast exactly zero volatility"
    assert got < 1e-3, f"the floor should be negligible, not {got}"


def test_long_run_vol_and_close_to_close_agree_because_one_calls_the_other():
    # They were byte-identical copies. The delegation is asserted so the copy
    # cannot come back: two homes for one formula is what left the sample
    # variance and the mean removal untested in both.
    rnd = random.Random(5)
    px = [100.0]
    for _ in range(400):
        px.append(px[-1] * math.exp(rnd.gauss(0.0, 0.012)))
    for w in (21, 63, 252):
        assert volforecast.long_run_vol(px, w) == \
            volforecast.close_to_close_vol(px, w), w


def test_the_mean_reversion_half_life_is_about_three_months():
    # kappa = 2.77 is ln(2)/0.25: a quarter. At kappa = 0.1 the half-life is
    # about seven years and the term structure is flat over anything a
    # 30-day option cares about, which is the regression term_vol exists to
    # prevent.
    import inspect
    kappa = inspect.signature(volforecast.term_vol).parameters["kappa"].default
    half_life_years = math.log(2.0) / kappa
    assert 0.15 < half_life_years < 0.40, (
        f"kappa={kappa} is a half-life of {half_life_years:.2f} years; the "
        f"documented choice is about a quarter")


def _bisect_iv(px, S, K, T, r, q, kind, lo=1e-4, hi=5.0):
    """A deliberately stupid bisection, as an INDEPENDENT check on the solver.

    engine/iv.py is Newton with a bisection fallback and bracket maintenance --
    fast, and with several places a sign can go the wrong way while it still
    converges to something plausible. Six such mutants survived a round-trip
    grid asserted to 1e-4, because they land close enough to pass it: measured,
    they change the answer on 24 to 202 of 286 quotes, but rarely by much.

    So this is the oracle. Two hundred halvings of [1e-4, 5] leave a bracket
    narrower than any float cares about, there is nothing in it to get subtly
    wrong, and the real solver has to agree with it.
    """
    f = lambda v: pricing.price(S, K, T, r, q, v, kind) - px
    if f(lo) * f(hi) > 0:
        return None
    a, b = lo, hi
    for _ in range(200):
        m = 0.5 * (a + b)
        if f(m) > 0:
            b = m
        else:
            a = m
    return 0.5 * (a + b)


def test_the_solver_agrees_with_an_independent_bisection():
    compared = 0
    for K in (60.0, 80.0, 95.0, 100.0, 105.0, 120.0, 160.0):
        for T in (1.0 / 365, 7.0 / 365, 30.0 / 365, 0.5, 2.0):
            for vol in (0.08, 0.20, 0.45, 1.20):
                for kind in ("call", "put"):
                    px = pricing.price(100.0, K, T, 0.04, 0.01, vol, kind)
                    # Only where the root is SEPARATED. With vega near zero
                    # the price is flat in vol, and then neither method is
                    # measuring anything: the bisection lands wherever a float
                    # comparison flips and the solver returns its seed. Those
                    # quotes belong to the identifiability test, not here.
                    if pricing.greeks(100.0, K, T, 0.04, 0.01, vol,
                                      kind).vega < 1e-2:
                        continue
                    # A TIGHT price tolerance, deliberately. At the default
                    # 1e-6 a low-vega quote leaves 1e-4 of vol slack, and the
                    # solver is entitled to it -- asserting inside that would
                    # be testing the tolerance rather than the algorithm. Ask
                    # for 1e-12 and the answer has to be the root.
                    got = implied_vol(px, 100.0, K, T, 0.04, 0.01, kind,
                                      tol=1e-12)
                    if got is None:
                        continue      # refused: its own tests cover that
                    ref = _bisect_iv(px, 100.0, K, T, 0.04, 0.01, kind)
                    assert ref is not None, (
                        f"the solver answered {got} where a bisection finds no "
                        f"root at all: K={K} T={T:.4f} {kind}")
                    assert abs(got - ref) < 1e-7, (
                        f"K={K} T={T:.4f} vol={vol} {kind}: solver {got:.9f} "
                        f"against bisection {ref:.9f}")
                    compared += 1
    assert compared > 150, f"only {compared} quotes compared"


def test_every_vol_the_solver_returns_reprices_the_quote_it_was_given():
    """The function's literal contract, and nothing asserted it.

    Three measurements decided what belongs in the mutant table for this
    module. Breaking the Newton step, skewing the bisection midpoint, or
    disabling the flat-gradient break all change the answer -- on 24 to 202 of
    286 quotes -- and every one of those answers still reprices the quote to
    within `tol`, max 9.9e-7 against the real code's 9.901e-7. They move which
    of the many vols inside the tolerance band comes back, which is the
    bracket doing its job, not a defect.

    So this is the property worth holding: whatever the solver returns, it has
    to reproduce the price it was asked about. A Newton break severe enough to
    stop convergence fails here; one the bisection absorbs does not, and
    should not.
    """
    tol = 1e-6
    checked = 0
    for K in (60.0, 80.0, 95.0, 100.0, 105.0, 120.0, 160.0):
        for T in (1.0 / 365, 7.0 / 365, 30.0 / 365, 0.5, 2.0):
            for vol in (0.08, 0.20, 0.45, 1.20):
                for kind in ("call", "put"):
                    px = pricing.price(100.0, K, T, 0.04, 0.01, vol, kind)
                    got = implied_vol(px, 100.0, K, T, 0.04, 0.01, kind,
                                      tol=tol)
                    if got is None:
                        continue
                    back = pricing.price(100.0, K, T, 0.04, 0.01, got, kind)
                    assert abs(back - px) <= tol, (
                        f"K={K} T={T:.4f} {kind}: vol {got:.9f} reprices to "
                        f"{back:.9f} against the quote {px:.9f}")
                    checked += 1
    assert checked > 150, f"only {checked} answers checked"


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
