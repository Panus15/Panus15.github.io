"""Tests for the SVI IV-surface fit (models/surface.py).

Three things matter: (1) the fit reproduces a known smile, (2) densifying a
SPARSE / truncated chain through SVI recovers the true risk-neutral vol far
better than reading raw moments off the sparse chain — the whole point — and
(3) a smile SVI CANNOT represent is refused rather than densified, because
densification turns a bad fit into 81 confident quotes that `rnd` reads as
market prices.
Run: python3 tests/test_surface.py
"""

import math
import os
import random
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


def _noisy_chain(strikes, jitter, seed):
    """The same smile with uniform vol noise, to drive the fit error up."""
    rng = random.Random(seed)
    quotes = []
    for K in strikes:
        v = max(0.01, _true_iv(K) + rng.uniform(-jitter, jitter))
        for kind in ("call", "put"):
            px = pricing.price(S, float(K), T, R, Q, v, kind)
            if px < 0.02:
                continue
            quotes.append(OptionQuote(DTE, float(K), kind, round(px, 4), round(px, 4)))
    return OptionChain("X", S, R, Q, quotes)


def _sawtooth_chain(strikes):
    """A smile with no SVI representation at all: +/-6 vol points, strike to
    strike. SVI is a smooth hyperbola, so it can only split the difference."""
    quotes = []
    for K in strikes:
        v = 0.22 + (0.06 if int(K) % 4 == 0 else -0.06)
        for kind in ("call", "put"):
            px = pricing.price(S, float(K), T, R, Q, v, kind)
            if px < 0.02:
                continue
            quotes.append(OptionQuote(DTE, float(K), kind, round(px, 4), round(px, 4)))
    return OptionChain("X", S, R, Q, quotes)


def test_vol_rmse_is_in_vol_points():
    """A flat slice gives the conversion an exact oracle.

    With b = 0 the total variance is `a` at every strike, so the ATM vol is
    sqrt(a / T) exactly and d_sigma = dw / (2 * sigma * T) can be checked
    against arithmetic rather than against the fitter.
    """
    vol = 0.20
    w = vol * vol * T
    surf = surface.SVISurface(a=w, b=0.0, rho=0.0, m=0.0, sigma=0.1,
                              F=100.0, T=T, rmse=1e-4)
    assert abs(surf.iv(100.0) - vol) < 1e-12
    assert abs(surf.iv(130.0) - vol) < 1e-12          # flat: no strike dependence
    expect = 1e-4 / (2.0 * vol * T)                   # 0.00304...
    assert abs(surf.vol_rmse() - expect) < 1e-12, (surf.vol_rmse(), expect)
    assert abs(surf.vol_rmse() - 0.00304167) < 1e-7   # the number, not just the formula


def test_representable_smile_passes_the_fit_gate():
    """The gate must not refuse ordinary chains. 2 vol points of quote noise is
    an ordinary chain: measured fit error 0.0118, recovered-vol error 0.0031."""
    surf = surface.fit_chain(_noisy_chain(range(80, 121, 2), 0.02, 3), T, DTE)
    assert surf.vol_rmse() < surface.MAX_VOL_RMSE, surf.vol_rmse()
    moments, _ = surface.robust_q_moments(_noisy_chain(range(80, 121, 2), 0.02, 3), T, DTE)
    assert 0.15 < moments.vol < 0.35, moments.vol


def test_unrepresentable_smile_is_refused_not_densified():
    """The defect this gate was added for: `rmse` was computed and discarded,
    and the docstring promised a fallback that did not exist in any form."""
    saw = _sawtooth_chain(range(80, 121, 2))
    surf = surface.fit_chain(saw, T, DTE)
    assert surf.vol_rmse() > 0.04, surf.vol_rmse()     # measured 0.0499
    try:
        surface.robust_q_moments(saw, T, DTE)          # NOTE: default tolerance
        assert False, "a smile SVI cannot represent must not become 81 quotes"
    except ValueError as e:
        assert "vol points" in str(e), e


def test_shipped_fit_tolerance_is_the_one_in_force():
    """The sawtooth test above would pass with any tolerance below 0.0499, so
    the shipped value gets pinned on its own -- and it has to sit above the
    noisy-chain measurement and below the un-representable one."""
    assert surface.MAX_VOL_RMSE == 0.02
    noisy = surface.fit_chain(_noisy_chain(range(80, 121, 2), 0.02, 3), T, DTE).vol_rmse()
    saw = surface.fit_chain(_sawtooth_chain(range(80, 121, 2)), T, DTE).vol_rmse()
    assert noisy < surface.MAX_VOL_RMSE < saw, (noisy, saw)


def test_a_poor_fit_is_refused_even_with_a_generous_caller():
    """An explicit tolerance still gates; it does not switch the gate off."""
    saw = _sawtooth_chain(range(80, 121, 2))
    moments, surf = surface.robust_q_moments(saw, T, DTE, max_vol_rmse=0.10)
    assert surf.vol_rmse() > 0.04
    try:
        surface.robust_q_moments(saw, T, DTE, max_vol_rmse=0.001)
        assert False, "a tighter tolerance must still refuse"
    except ValueError:
        pass


def test_dense_chain_writes_both_kinds_at_each_strike():
    """The docstring used to claim calls above the forward and puts below it.
    It writes both, and the consumers depend on that: `rnd` picks its OTM leg by
    K0 and `bkm_moments` by spot, so a one-sided chain halves one wing."""
    surf = surface.fit_chain(_chain(range(80, 121, 2)), T, DTE)
    dense = surf.dense_chain(dte=DTE)
    calls = {q.strike for q in dense.quotes if q.kind == "call"}
    puts = {q.strike for q in dense.quotes if q.kind == "put"}
    both = calls & puts
    assert len(both) > 40, len(both)                   # measured 60 of 81
    assert min(both) < surf.F < max(both)              # straddling, not one-sided
    # and the grid is WIDE: that width is the truncation repair.
    assert min(calls | puts) < 0.7 * S < 1.5 * S < max(calls | puts)


def test_narrow_densification_reinstates_the_truncation_bias():
    """`lo`/`hi` are the repair, not cosmetics. Reading Q off a grid clipped
    back towards spot is biased LOW again -- the bias the module exists to
    remove, and the direction that makes the market look cheap."""
    sparse = _chain(range(94, 107, 3))
    surf = surface.fit_chain(sparse, T, DTE)
    wide = rnd.bkm_moments(surf.dense_chain(dte=DTE), T, DTE).vol
    narrow = rnd.bkm_moments(surf.dense_chain(dte=DTE, lo=0.9, hi=1.1), T, DTE).vol
    assert narrow < wide - 0.01, (narrow, wide)


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
