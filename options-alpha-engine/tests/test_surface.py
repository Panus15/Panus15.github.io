"""Tests for the SVI IV-surface fit (models/surface.py).

Two things matter: (1) the fit reproduces a known smile, and (2) densifying a
SPARSE / truncated chain through SVI recovers the true risk-neutral vol far
better than reading raw moments off the sparse chain — the whole point.
Run: python3 tests/test_surface.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import pricing
from engine.data import OptionChain, OptionQuote
from models import rnd, surface

S, R, Q, DTE = 100.0, 0.03, 0.0, 30
T = DTE / 365.0


def _true_iv(K):
    m = K / S - 1.0
    return 0.22 - 0.5 * m + 0.6 * m * m          # crash-skew smile


def _chain(strikes):
    quotes = []
    for K in strikes:
        v = _true_iv(K)
        for kind in ("call", "put"):
            px = pricing.price(S, K, T, R, Q, v, kind)
            if px < 0.02:
                continue
            quotes.append(OptionQuote(DTE, float(K), kind, round(px, 4), round(px, 4)))
    return OptionChain("X", S, R, Q, quotes)


def test_svi_fit_reproduces_smile():
    surf = surface.fit_chain(_chain(range(80, 121, 2)), T, DTE)
    for K in (85, 90, 95, 100, 105, 110, 115):
        assert abs(surf.iv(K) - _true_iv(K)) < 0.01, (K, surf.iv(K), _true_iv(K))


def test_svi_no_arbitrage():
    surf = surface.fit_chain(_chain(range(80, 121, 2)), T, DTE)
    assert surf.no_arb_ok()
    assert surf.b >= 0 and abs(surf.rho) < 1 and surf.sigma > 0


def test_densified_chain_recovers_atm_vol():
    surf = surface.fit_chain(_chain(range(80, 121, 2)), T, DTE)
    dense = surf.dense_chain(dte=DTE)
    q = rnd.model_free_implied_vol(dense, T, DTE)
    assert abs(q - _true_iv(100)) < 0.02, q     # ~ATM vol back out of the smooth chain


def test_svi_beats_raw_on_sparse_truncated_chain():
    true_q = rnd.model_free_implied_vol(_chain(range(80, 121, 2)), T, DTE)  # dense truth
    sparse = _chain(range(94, 107, 3))          # few strikes, wings truncated
    raw = rnd.model_free_implied_vol(sparse, T, DTE)
    moments, _ = surface.robust_q_moments(sparse, T, DTE)
    # SVI densification recovers the true vol at least as well as raw extraction,
    # and here clearly better (raw is biased low by the missing wings).
    assert abs(moments.vol - true_q) <= abs(raw - true_q)


def test_fit_requires_enough_strikes():
    try:
        surface.fit_chain(_chain([98, 100, 102]), T, DTE)   # too few
        assert False, "should require >= 5 strikes"
    except ValueError:
        pass


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
