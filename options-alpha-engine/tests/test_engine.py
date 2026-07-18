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


def test_scanner_finds_injected_dislocation():
    md = SyntheticAdapter()
    chain = md.option_chain("X")
    # Scan against a low fair vol so the injected rich options surface.
    ideas = scan_chain(chain, forecast_vol=0.17, min_edge_net=0.01)
    assert any(m.verdict == "RICH" for m in ideas)


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
