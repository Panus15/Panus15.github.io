"""Oracle tests for the distributional core.

The key identity: pricing a European option against the Black-Scholes
risk-neutral law (as a one-component MixtureLogNormal) MUST reproduce the
closed-form Black-Scholes price. If that holds, the whole expected-payoff /
moment machinery is trustworthy. Run: python3 tests/test_density.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import pricing
from models.density import (
    LogNormalComponent,
    MixtureLogNormal,
    single_lognormal_riskneutral,
)


def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol


def test_lognormal_matches_black_scholes():
    S, T, r, q, vol = 100.0, 0.5, 0.04, 0.01, 0.28
    dist = single_lognormal_riskneutral(S, T, r, q, vol)
    for K in (80, 95, 100, 110, 130):
        for kind in ("call", "put"):
            model = dist.price(K, r, T, kind)
            bsm = pricing.price(S, K, T, r, q, vol, kind)
            assert approx(model, bsm, 1e-6), (K, kind, model, bsm)


def test_cdf_monotone_and_bounded():
    dist = single_lognormal_riskneutral(100, 0.5, 0.03, 0.0, 0.2)
    prev = 0.0
    for x in range(1, 400, 5):
        c = dist.cdf(x)
        assert 0.0 <= c <= 1.0
        assert c >= prev - 1e-12
        prev = c


def test_quantile_inverts_cdf():
    dist = single_lognormal_riskneutral(100, 1.0, 0.03, 0.0, 0.25)
    for p in (0.1, 0.25, 0.5, 0.75, 0.9):
        x = dist.quantile(p)
        assert approx(dist.cdf(x), p, 1e-4), (p, dist.cdf(x))


def test_mean_of_riskneutral_is_forward():
    # Under Q, E[S_T] = S * exp((r-q)T)  (the forward price).
    S, T, r, q, vol = 100.0, 0.75, 0.05, 0.02, 0.3
    dist = single_lognormal_riskneutral(S, T, r, q, vol)
    fwd = S * math.exp((r - q) * T)
    assert approx(dist.mean(), fwd, 1e-6), (dist.mean(), fwd)


def test_mixture_has_negative_skew_when_stress_left():
    # A calm component + a lower-mean stress component -> left (crash) skew.
    calm = LogNormalComponent(0.85, math.log(100) + 0.0, 0.15)
    stress = LogNormalComponent(0.15, math.log(100) - 0.20, 0.35)
    dist = MixtureLogNormal([calm, stress])
    assert dist.log_return_skew() < 0.0


def test_distribution_reduces_to_scalar_vol():
    # A single log-normal's log-return vol must equal its input sigma/sqrt(T)... i.e. the vol.
    S, T, vol = 100.0, 0.5, 0.22
    dist = single_lognormal_riskneutral(S, T, 0.0, 0.0, vol)
    assert approx(dist.log_return_vol(S, T), vol, 1e-9)


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
