"""Correctness tests for the pricing/IV core. Run: python3 -m pytest -q
(or plain `python3 tests/test_engine.py` — a tiny runner is included).

These check the properties that MUST hold, so refactors can't silently break
the math the whole engine rests on.
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import pricing, volforecast
from engine.data import SyntheticAdapter
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
