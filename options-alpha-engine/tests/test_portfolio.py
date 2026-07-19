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

from engine.portfolio import (
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
