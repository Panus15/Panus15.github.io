"""Tests for the portfolio-level risk layer (engine/portfolio.py).

The properties under test are the ones the adversarial review said the per-trade
Kelly + 2% cap could NOT guarantee for a short-vol book:

  1. Correlation-aware aggregation does not falsely diversify a single-factor
     book (N identical shorts => ~N x vega, not sqrt(N)).
  2. The vega cap bites on the *correlated* exposure.
  3. The drawdown kill-switch blocks ALL new risk, vega headroom or not.
  4. CVaR sizing shrinks the trade as the modeled shock grows.
  5. Defined-risk conversion bounds the naked-short tail.

Run: python3 tests/test_portfolio.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import portfolio

from engine.portfolio import (
    DEFAULT_VEGA_RHO,
    cvar_per_contract,
    Portfolio,
    Position,
    RiskGovernor,
    RiskLimits,
    Scenario,
    size_by_cvar,
    to_defined_risk,
    vol_spike_scenarios,
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _short_put(underlying="SPX", strike=95.0, qty=-1, spot=100.0, vol=0.20):
    """A representative short-vol leg (short OTM put)."""
    return Position(
        underlying=underlying, strike=strike, expiry_days=30, kind="put",
        quantity=qty, entry_price=1.50, spot=spot, r=0.03, q=0.0, vol=vol,
    )


def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol


# --------------------------------------------------------------------------- #
# 1. aggregation: signs + no false diversification
# --------------------------------------------------------------------------- #
def test_net_greeks_have_correct_sign():
    long_call = Position("SPX", 100, 30, "call", quantity=2, entry_price=2.0,
                         spot=100, r=0.03, q=0.0, vol=0.20)
    short_put = _short_put(qty=-3)

    book = Portfolio([long_call, short_put])

    # Expected = linear sum of per-leg dollar Greeks.
    exp_vega = long_call.net_vega() + short_put.net_vega()
    exp_delta = long_call.net_delta() + short_put.net_delta()
    assert approx(book.net_vega(), exp_vega), (book.net_vega(), exp_vega)
    assert approx(book.net_delta(), exp_delta), (book.net_delta(), exp_delta)

    # Short leg carries NEGATIVE vega/gamma (short vol / short gamma) ...
    assert short_put.net_vega() < 0
    assert short_put.net_gamma() < 0
    # ... but POSITIVE theta (a short option collects time decay).
    assert short_put.net_theta() > 0
    # Long call is the opposite: positive vega, negative theta.
    assert long_call.net_vega() > 0
    assert long_call.net_theta() < 0


def test_correlated_short_vega_does_not_diversify():
    # N identical short-vol legs on the SAME underlying = one factor.
    N = 5
    single = _short_put()
    v1 = abs(single.net_vega())          # single-leg vega magnitude
    book = Portfolio([_short_put() for _ in range(N)])

    eff = book.effective_short_vega()    # correlation-aware factor magnitude
    indep = book.independent_vega()      # correlation-blind "diversified" number

    # Correlation-aware exposure keeps the FULL ~N x single (no diversification).
    assert approx(eff, N * v1, tol=1e-6 * N * v1 + 1e-6), (eff, N * v1)
    # The naive per-trade view would report only sqrt(N) x single -> false comfort.
    assert approx(indep, math.sqrt(N) * v1, tol=1e-6 * v1 + 1e-6), (indep, math.sqrt(N) * v1)
    # And the correlated number is materially larger than the diversified one.
    assert eff > indep * 1.5
    # net_short_vega (positive, book is net short) equals the effective magnitude.
    assert approx(book.net_short_vega(), eff)


def test_cross_underlying_correlation_still_concentrates():
    # Different equity indices are NOT independent: rho keeps most of the risk.
    legs = [_short_put(underlying=u) for u in ("SPX", "NDX", "RUT")]
    v1 = abs(legs[0].net_vega())
    book = Portfolio(legs, rho=0.8)
    eff = book.effective_short_vega()
    indep = book.independent_vega()      # = sqrt(3) x v1
    # Effective sits well above the "diversified" sqrt(N) number and near ~N.
    assert eff > indep
    assert eff > 2.5 * v1                 # 3-name book at rho=0.8 stays concentrated


# --------------------------------------------------------------------------- #
# 2. vega cap
# --------------------------------------------------------------------------- #
def test_vega_cap_rejects_over_and_allows_under():
    single = _short_put()
    v1 = abs(single.net_vega())
    # Cap allows ~2 legs but not a 3rd (correlated exposure is linear in N).
    limits = RiskLimits(max_net_short_vega=2.5 * v1, max_drawdown=0.20)
    gov = RiskGovernor(limits)

    # Under the cap: adding the 1st leg to an empty book is allowed.
    empty = Portfolio([])
    ok, reason = gov.can_add(single, empty, equity=100_000, current_drawdown=0.0)
    assert ok, reason
    assert "OK" in reason

    # Over the cap: a book already holding 2 legs rejects a 3rd (3 v1 > 2.5 v1).
    two = Portfolio([_short_put(), _short_put()])
    ok, reason = gov.can_add(single, two, equity=100_000, current_drawdown=0.0)
    assert not ok
    assert "VEGA CAP" in reason, reason


# --------------------------------------------------------------------------- #
# 3. drawdown kill-switch
# --------------------------------------------------------------------------- #
def test_kill_switch_blocks_all_new_risk():
    # Wide-open vega cap so ONLY the drawdown gate can reject.
    limits = RiskLimits(max_net_short_vega=1e12, max_drawdown=0.20)
    gov = RiskGovernor(limits)
    empty = Portfolio([])

    # Below the drawdown threshold -> allowed despite being short vol.
    ok, _ = gov.can_add(_short_put(), empty, equity=100_000, current_drawdown=0.10)
    assert ok

    # Past the drawdown threshold -> blocked regardless of vega headroom.
    ok, reason = gov.can_add(_short_put(), empty, equity=100_000, current_drawdown=0.25)
    assert not ok
    assert "KILL-SWITCH" in reason, reason


# --------------------------------------------------------------------------- #
# 4. CVaR sizing monotonicity
# --------------------------------------------------------------------------- #
def test_cvar_sizing_shrinks_with_bigger_shock():
    equity = 1_000_000.0
    template = _short_put()               # selling a put: fat left tail on a spike

    small = vol_spike_scenarios(spot_shock=-0.05, vol_bump=0.05)
    big = vol_spike_scenarios(spot_shock=-0.15, vol_bump=0.20)

    n_small = size_by_cvar(equity, template, shock_scenarios=small, cvar_limit=0.05)
    n_big = size_by_cvar(equity, template, shock_scenarios=big, cvar_limit=0.05)

    assert isinstance(n_small, int) and isinstance(n_big, int)
    assert n_small >= 0 and n_big >= 0
    assert n_small > 0                    # a modest shock still permits a position
    assert n_big < n_small, (n_small, n_big)   # bigger tail -> fewer contracts


def test_cvar_sizing_scales_with_limit():
    equity = 1_000_000.0
    template = _short_put()
    sc = vol_spike_scenarios(spot_shock=-0.10, vol_bump=0.10)
    tight = size_by_cvar(equity, template, shock_scenarios=sc, cvar_limit=0.01)
    loose = size_by_cvar(equity, template, shock_scenarios=sc, cvar_limit=0.05)
    assert loose > tight >= 0


# --------------------------------------------------------------------------- #
# 5. defined-risk conversion bounds the tail
# --------------------------------------------------------------------------- #
def test_to_defined_risk_bounds_worst_case_loss():
    offset = 5.0
    naked = _short_put(strike=95.0, qty=-1)     # short 95 put, unbounded-ish tail
    legs = to_defined_risk(naked, wing_offset=offset)

    # Two legs: the original short + a long protective wing.
    assert len(legs) == 2
    short_leg, long_wing = legs
    assert short_leg.quantity < 0 and long_wing.quantity > 0
    assert long_wing.strike == 95.0 - offset    # long put below the short strike
    assert long_wing.kind == "put"

    # Deep adverse terminal price (crash to ~0): the naked short loses hugely,
    # the vertical's loss is capped at the strike width (x multiplier).
    crash_spot = 1.0
    naked_loss = -naked.pnl_at_expiry(crash_spot)               # positive = loss
    vertical_loss = -Portfolio(legs).pnl_at_expiry(crash_spot)  # positive = loss

    assert naked_loss > vertical_loss                # the wing genuinely helps
    # Bounded by strike width x multiplier (credit only makes it smaller).
    max_defined_loss = offset * naked.multiplier
    assert vertical_loss <= max_defined_loss + 1e-6, (vertical_loss, max_defined_loss)
    # And the naked tail is far larger than the defined-risk cap.
    assert naked_loss > 5 * max_defined_loss


# --------------------------------------------------------------------------- #
# self-runner (same pattern as the other tests)
# --------------------------------------------------------------------------- #
def test_position_tenor_honours_the_callers_day_count():
    """A backtest that counts TRADING bars must not be sized on a calendar tenor.

    Position.T was hardcoded to expiry_days/365 while every simulation loop in
    this repo prices the same position at dte/252, because their `dte` counts
    trading bars. At 21 bars that is a 1.45x difference in T, and the governor was
    therefore sizing on an option with ~17% less vega than the one being held —
    it under-measured the risk of every position it approved.
    """
    kw = dict(underlying="U", strike=100.0, expiry_days=21, kind="put", quantity=-1,
              entry_price=1.0, spot=100.0, r=0.03, q=0.0, vol=0.25, multiplier=100)
    cal = Position(**kw)                      # default: calendar days
    trd = Position(**kw, days_per_year=252.0)  # caller counts trading bars
    assert abs(cal.T - 21 / 365.0) < 1e-12
    assert abs(trd.T - 21 / 252.0) < 1e-12
    v_cal = cal.contract_greeks().vega
    v_trd = trd.contract_greeks().vega
    assert v_trd > v_cal, (v_cal, v_trd)
    assert 0.14 < 1 - v_cal / v_trd < 0.20, (
        f"a 21-bar tenor on the wrong clock understates vega by "
        f"{1 - v_cal / v_trd:.1%}; the measured figure was 16.8%")



def test_cvar_alpha_must_actually_move_the_number():
    """The knob named `alpha` did nothing at shipped defaults.

    With a two-point scenario set there is ONE loss state carrying tail_prob of
    the mass, so every alpha whose tail falls at or below it averages the same
    single state: 0.90, 0.95, 0.99 and 0.999 all returned 2,443.31. The engine
    advertised "CVaR at 95% confidence" while reporting the loss in one
    hand-written scenario. On a richer set the level must resolve and RISE.
    """
    t = portfolio.Position(underlying="SPY", strike=700.0, expiry_days=30,
                           kind="put", quantity=-1, entry_price=1.10, spot=765.0,
                           r=0.04, q=0.0, vol=0.16, multiplier=100,
                           days_per_year=252.0)
    rich = [portfolio.Scenario(spot_mult=1.00, vol_shift=0.00, prob=0.90),
            portfolio.Scenario(spot_mult=0.95, vol_shift=0.05, prob=0.07),
            portfolio.Scenario(spot_mult=0.88, vol_shift=0.15, prob=0.025),
            portfolio.Scenario(spot_mult=0.75, vol_shift=0.40, prob=0.005)]
    lo = portfolio.cvar_per_contract(t, rich, 0.90)
    mid = portfolio.cvar_per_contract(t, rich, 0.95)
    hi = portfolio.cvar_per_contract(t, rich, 0.99)
    assert lo < mid < hi, (lo, mid, hi)


def test_a_confidence_level_the_scenarios_cannot_resolve_is_refused():
    """Returning a number that looks like a 99% figure and is not is worse than
    refusing, because the number gets quoted."""
    t = portfolio.Position(underlying="SPY", strike=700.0, expiry_days=30,
                           kind="put", quantity=-1, entry_price=1.10, spot=765.0,
                           r=0.04, q=0.0, vol=0.16, multiplier=100,
                           days_per_year=252.0)
    two = portfolio.vol_spike_scenarios()          # tail_prob = 0.10
    portfolio.cvar_per_contract(t, two, 0.90)      # exactly resolvable: fine
    for bad in (0.95, 0.99, 0.999):
        try:
            portfolio.cvar_per_contract(t, two, bad)
        except ValueError as e:
            assert "cannot resolve" in str(e) or "same" in str(e), str(e)
        else:
            raise AssertionError(f"alpha={bad} returned a number it cannot support")


def test_the_shipped_default_alpha_is_one_the_default_scenarios_can_answer():
    """A default that raises on its own default scenario set is a broken default."""
    import inspect
    sig = inspect.signature(portfolio.size_by_cvar)
    alpha = sig.parameters["alpha"].default
    t = portfolio.Position(underlying="SPY", strike=700.0, expiry_days=30,
                           kind="put", quantity=-1, entry_price=1.10, spot=765.0,
                           r=0.04, q=0.0, vol=0.16, multiplier=100,
                           days_per_year=252.0)
    portfolio.cvar_per_contract(t, portfolio.vol_spike_scenarios(), alpha)


# ---------------------------------------------------------------------------
# The thirteen survivors of this module's first sweep. The caps that had no
# test at all were the gross-vega one, the delta one, and the kill-switch
# boundary -- which is to say, two of the three gates and the edge of the
# third. A cap that fails open is worse than no cap, because it is believed.
# ---------------------------------------------------------------------------

def test_the_correlation_is_strictly_between_independent_and_identical():
    # Nothing distinguished rho=0.8 from rho=1.0, so "treat every pair as
    # perfectly correlated" survived. Two different underlyings at rho must
    # sit between the independent and the identical answers, and the DEFAULT
    # has to be the one that does that -- a default of 0.0 is the false
    # diversification this module exists to remove.
    book = Portfolio([_short_put(underlying="SPX"), _short_put(underlying="NDX")])
    one = abs(_short_put().net_vega())
    indep = book.independent_vega()
    identical = book.effective_short_vega(rho=1.0)
    at_rho = book.effective_short_vega(rho=0.8)
    assert approx(identical, 2.0 * one, tol=1e-6), (identical, one)
    assert indep < at_rho < identical, (
        f"rho=0.8 gives {at_rho}, not between independent {indep} and "
        f"identical {identical}")
    assert approx(book.effective_short_vega(), at_rho), (
        f"the default correlation is not {DEFAULT_VEGA_RHO}: "
        f"{book.effective_short_vega()} against {at_rho}")
    assert DEFAULT_VEGA_RHO > 0.5, (
        f"a default of {DEFAULT_VEGA_RHO} lets a short-vol book diversify "
        f"away, which is the false comfort this module was written against")


def test_the_gross_vega_cap_rejects_and_admits_on_the_right_side():
    # No test touched this gate: disabling it entirely, and inverting its
    # comparison, both passed the suite.
    book = Portfolio([_short_put(qty=-10)])
    gross = book.gross_vega()
    gov_tight = RiskGovernor(RiskLimits(max_gross_vega=gross * 0.5))
    ok, why = gov_tight.can_add(_short_put(), book, 100_000.0, 0.0)
    assert not ok, f"a book already over the gross cap was allowed: {why}"
    assert "gross" in why.lower(), why
    gov_loose = RiskGovernor(RiskLimits(max_gross_vega=gross * 100.0))
    ok2, why2 = gov_loose.can_add(_short_put(), book, 100_000.0, 0.0)
    assert ok2, f"a trade far inside the gross cap was vetoed: {why2}"


def test_the_net_delta_cap_rejects_and_admits_on_the_right_side():
    book = Portfolio([_short_put(qty=-10)])
    nd = abs(book.net_delta())
    assert nd > 0, "the fixture carries no delta, so this proves nothing"
    gov_tight = RiskGovernor(RiskLimits(max_net_delta=nd * 0.5))
    ok, why = gov_tight.can_add(_short_put(), book, 100_000.0, 0.0)
    assert not ok, f"a book already over the delta cap was allowed: {why}"
    assert "delta" in why.lower(), why
    gov_loose = RiskGovernor(RiskLimits(max_net_delta=nd * 100.0))
    ok2, why2 = gov_loose.can_add(_short_put(), book, 100_000.0, 0.0)
    assert ok2, f"a trade far inside the delta cap was vetoed: {why2}"


def test_the_kill_switch_trips_AT_its_threshold_not_past_it():
    # `>=` loosened to `>` lets one more trade through at exactly the limit,
    # which is the trade taken at the worst possible moment. Nothing tested
    # the boundary.
    gov = RiskGovernor(RiskLimits(max_drawdown=0.20))
    book = Portfolio([])
    ok_at, why_at = gov.can_add(_short_put(), book, 100_000.0, 0.20)
    assert not ok_at, (
        f"at exactly the 20% drawdown limit a new position was allowed: "
        f"{why_at}")
    ok_under, _ = gov.can_add(_short_put(), book, 100_000.0, 0.1999)
    assert ok_under, "just under the limit must still trade"


def test_cvar_is_a_positive_loss_and_not_the_best_case():
    # Two scenarios, one a loss and one a gain, with all the probability on
    # the loss: the expected shortfall is that loss, as a POSITIVE number.
    # Returning 0.0, or letting the sign through, sizes a position against a
    # loss that cannot happen.
    loss = cvar_per_contract(
        _short_put(),
        [Scenario(spot_mult=0.80, vol_shift=0.30, prob=1.0)],
        alpha=1.0)
    assert loss > 0.0, f"a crash scenario gave a CVaR of {loss}"


def test_cvar_of_a_tail_that_is_PROFITABLE_is_zero_rather_than_negative():
    """A CVaR is a loss. When the modelled tail makes money it is zero.

    Reaching the clamp that enforces this takes some care, and a single
    profitable scenario does not do it -- that exits through an earlier
    branch. The tail slice itself has to be profitable while the set is still
    resolvable at the requested alpha, so the big-mass scenario has to be the
    MOST profitable one and the thin tail the least:

        5%  spot flat, vol flat    +97.71 per contract
        5%  spot +5%, vol -2 pts  +145.48
        90% spot +20%, vol -10    +150.00

    Without the clamp this returns about -98: a negative "loss", which
    size_by_cvar would divide a positive budget by and size the book UP on
    the strength of the tail being good.
    """
    got = cvar_per_contract(
        _short_put(),
        [Scenario(spot_mult=1.00, vol_shift=0.00, prob=0.05),
         Scenario(spot_mult=1.05, vol_shift=-0.02, prob=0.05),
         Scenario(spot_mult=1.20, vol_shift=-0.10, prob=0.90)],
        alpha=0.90)
    assert got == 0.0, f"a profitable tail produced a CVaR of {got}"


def test_a_correlation_outside_zero_to_one_is_refused():
    # Outside [0, 1] the correlation matrix stops being positive
    # semi-definite, so the quadratic form can come out negative and its
    # square root is not an exposure. It used to be clamped silently, which
    # turned a nonsensical input into a number that gets compared to a cap.
    book = Portfolio([_short_put(underlying="SPX"),
                      _short_put(underlying="NDX")])
    for bad in (-0.5, 1.5, 2.0):
        try:
            book.effective_short_vega(rho=bad)
        except ValueError as e:
            assert "rho" in str(e), e
        else:
            assert False, f"rho={bad} was accepted"
    book.effective_short_vega(rho=0.0)      # the boundaries must still work
    book.effective_short_vega(rho=1.0)


def test_no_scenarios_means_no_size_rather_than_unlimited_size():
    assert cvar_per_contract(_short_put(), [], alpha=0.95) == 0.0
    n = size_by_cvar(100_000.0, _short_put(), shock_scenarios=[],
                     cvar_limit=0.02)
    assert n == 0, f"sized {n} contracts against an empty scenario set"


def test_a_non_positive_equity_or_limit_sizes_nothing():
    scen = [Scenario(spot_mult=0.80, vol_shift=0.30, prob=1.0)]
    for eq, lim in ((0.0, 0.02), (-100.0, 0.02), (100_000.0, 0.0),
                    (100_000.0, -0.01)):
        n = size_by_cvar(eq, _short_put(), shock_scenarios=scen,
                         cvar_limit=lim)
        assert n == 0, f"equity {eq} and limit {lim} sized {n} contracts"


def test_a_short_position_reports_itself_short():
    assert _short_put(qty=-1).is_short is True
    long_put = _short_put(qty=1)
    assert long_put.is_short is False
    assert _short_put(qty=-1).net_vega() < 0.0, (
        "a short option's vega must be negative, or the book's short exposure "
        "reads as long and every cap is measured on the wrong side")


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
