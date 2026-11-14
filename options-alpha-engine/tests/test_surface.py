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


def _shaped_chain(strikes, ivfn, rounding=4):
    """A chain priced at an arbitrary smile, with a settable tick size."""
    quotes = []
    for K in strikes:
        v = max(0.02, ivfn(K))
        for kind in ("call", "put"):
            px = pricing.price(S, float(K), T, R, Q, v, kind)
            if px < 0.02:
                continue
            quotes.append(OptionQuote(DTE, float(K), kind,
                                      round(px, rounding), round(px, rounding)))
    return OptionChain("X", S, R, Q, quotes)


#: Peak ATM, falling in both wings. An unconstrained SVI wants b < 0 for this
#: (total variance DECREASING in |k|), which is outside the admissible region,
#: so it is the shape that makes the slope penalty bind. Measured: the real fit
#: pins b at exactly 0; without the penalty it reaches -0.028 and the slice
#: leaves the region. Its fit error is 0.0164, inside MAX_VOL_RMSE, so this is
#: an ACCEPTED fit rather than one the gate would have refused anyway.
_FROWN = lambda K: 0.30 - 1.8 * (K / S - 1) ** 2

#: Monotone decreasing: no turning point in the quoted range at all. This drives
#: the fit to the min-variance boundary -- measured min total variance is
#: +1.4e-04 over 80-120 strikes and +1.0e-09 over 60-140, i.e. zero to nine
#: decimals, held there by the penalty.
_PURE_SKEW = lambda K: max(0.05, 0.22 - 1.2 * (K / S - 1))


def test_min_total_variance_matches_a_brute_force_scan():
    """The closed form is the no-arbitrage condition, so it gets an exact oracle.

    w(k) is minimised at k - m = -rho*sigma/sqrt(1-rho^2); brute-forcing 80001
    points of k over [-8, 8] has to agree. This lived inline in the objective's
    closure where nothing could call it, and both mutants of it -- 1+rho^2 for
    1-rho^2, and dropping the sigma factor -- survived a sweep for that reason.
    """
    for a, b, rho, sg in ((0.004, 0.05, -0.47, 0.75), (-0.1, 0.53, 0.79, 0.32),
                          (0.01, 0.0, 0.0, 0.1), (0.002, 0.2, -0.9, 0.4),
                          (0.02, 0.3, 0.6, 1.0)):
        brute = min(surface.svi_total_variance(-8.0 + 16.0 * i / 80000, a, b, rho, 0.3, sg)
                    for i in range(80001))
        exact = surface.min_total_variance(a, b, rho, sg)
        assert abs(brute - exact) < 1e-7, (a, b, rho, sg, brute, exact)
    # and the sigma factor is load-bearing: at sigma != 1 the two differ.
    assert abs(surface.min_total_variance(0.0, 1.0, 0.0, 0.25) - 0.25) < 1e-12
    # independent of m, which only translates the slice
    assert surface.min_total_variance(0.004, 0.05, -0.47, 0.75) > 0


def test_no_arb_rejects_each_way_out_of_the_region():
    """Guard isolation: the old test only ever called this on a surface that
    passes, so `return True` and dropping any single clause all survived.

    Each fixture below is rejectable by EXACTLY ONE clause -- verified by the
    comments, which are measured values, not intentions.
    """
    good = surface.SVISurface(a=0.01, b=0.05, rho=-0.5, m=0.0, sigma=0.1, F=100.0, T=T)
    assert good.no_arb_ok()                      # min_tv +0.014330, all clauses pass

    bad = [
        ("negative slope",   dict(a=0.01, b=-0.05, rho=0.0, sigma=0.1)),   # min_tv +0.005
        ("unit correlation", dict(a=0.01, b=0.05, rho=1.0, sigma=0.1)),    # min_tv +0.010
        ("over correlation", dict(a=0.01, b=0.05, rho=1.5, sigma=0.1)),    # min_tv +0.010
        ("zero width",       dict(a=0.01, b=0.05, rho=0.0, sigma=0.0)),    # min_tv +0.010
        ("negative minimum", dict(a=-0.01, b=0.05, rho=0.0, sigma=0.1)),   # min_tv -0.005
    ]
    for label, kw in bad:
        surf = surface.SVISurface(m=0.0, F=100.0, T=T, **kw)
        assert not surf.no_arb_ok(), label
    # the five fail for five different reasons, so no one clause covers another
    assert surface.min_total_variance(0.01, -0.05, 0.0, 0.1) >= 0      # slope only
    assert surface.min_total_variance(0.01, 0.05, 1.0, 0.1) >= 0       # rho only
    assert surface.min_total_variance(0.01, 0.05, 0.0, 0.0) >= 0       # width only
    assert surface.min_total_variance(-0.01, 0.05, 0.0, 0.1) < 0       # minimum only


def test_the_variance_floor_keeps_the_slice_real():
    """A slice can be fitted whose raw total variance is negative somewhere, and
    `iv` takes a square root of it. Without the floor that is a ValueError from
    math.sqrt, not a vol -- so the floor is what makes `iv` total."""
    surf = surface.SVISurface(a=-0.01, b=0.05, rho=0.0, m=0.0, sigma=0.1, F=100.0, T=T)
    assert surface.svi_total_variance(0.0, -0.01, 0.05, 0.0, 0.0, 0.1) < 0   # -0.005
    assert surf.total_variance(100.0) == 1e-8
    assert 0.0 < surf.iv(100.0) < 0.001                                      # 0.000349
    assert not surf.no_arb_ok()                                              # and it is flagged


def test_forward_is_recovered_from_parity_not_assumed():
    """Put-call parity on a Black-Scholes chain gives F = S*exp((r-q)T) exactly,
    which is an arithmetic oracle rather than a comparison against spot.

    Measured: recovered 100.24690806 against an oracle of 100.24687959, so the
    error is 2.8e-05. Dropping the discount factor gives 5.8e-04 and flipping
    the parity sign gives 4.9e-01, so a 1e-4 tolerance separates all three.
    """
    surf = surface.fit_chain(_chain(range(80, 121, 2)), T, DTE)
    oracle = S * math.exp((R - Q) * T)
    assert abs(surf.F - oracle) < 1e-4, (surf.F, oracle)
    assert surf.F > S                       # r > q, so the forward sits above spot


def test_forward_falls_back_to_cost_of_carry_with_no_two_sided_strike():
    """Parity needs a strike quoting BOTH legs. With none, the fallback must
    still carry spot forward -- returning spot itself understates it by the
    whole cost of carry, and `dense_chain` is anchored on that number."""
    quotes = [OptionQuote(DTE, 95.0, "call", 6.0, 6.0),
              OptionQuote(DTE, 105.0, "put", 6.0, 6.0)]
    F, strikes, calls, puts = surface._forward(OptionChain("X", S, R, Q, quotes), DTE, T)
    assert strikes == []                    # the parity branch is genuinely unreachable here
    assert abs(F - S * math.exp((R - Q) * T)) < 1e-9
    assert F > S, F


def test_a_crash_skew_fits_with_negative_rho():
    """rho IS the skew direction in this parameterisation, so its sign is a
    claim about the market and not an internal detail. A left-skewed smile must
    fit with rho < 0; with the skew term's sign flipped the fitter recovers
    +0.564 for the same chain and the fit error stays tiny either way."""
    surf = surface.fit_chain(_chain(range(80, 121, 2)), T, DTE)
    assert surf.rho < -0.1, surf.rho                   # measured -0.468
    assert _true_iv(90) > _true_iv(110)                # the fixture really is left-skewed


def test_a_representable_smile_fits_far_tighter_than_a_vol_point():
    """The 0.01 tolerance in test_svi_fit_reproduces_smile is a sanity bound, not
    a measurement of the fit. On a clean 21-strike smile the fit is essentially
    exact, and that is what pins the parameters that only show up as precision:
    dropping the `m` shift leaves the slice centred on the forward and costs a
    factor of 8.5 here (6.0e-05 -> 5.1e-04) while staying inside the gate.
    """
    assert surface.fit_chain(_chain(range(80, 121, 2)), T, DTE).vol_rmse() < 2e-4
    shifted = _shaped_chain(range(80, 121, 2), lambda K: 0.22 + 1.2 * (K / S - 1.20) ** 2)
    assert surface.fit_chain(shifted, T, DTE).vol_rmse() < 2e-4   # minimum away from ATM


def test_dense_chain_is_anchored_on_the_true_spot():
    """`dense_chain` prices off a spot discounted back from the fitted forward,
    and `rnd.bkm_moments` selects its OTM leg by that spot. Anchoring on the
    forward instead moves the whole grid and shifts the recovered vol."""
    surf = surface.fit_chain(_chain(range(80, 121, 2)), T, DTE)
    dense = surf.dense_chain(dte=DTE)
    assert abs(dense.spot - S) < 1e-3, dense.spot      # measured 100.000028
    assert dense.spot < surf.F                         # discounted back, not carried forward


def test_dense_grid_reaches_both_ends_of_its_range():
    """n points spanning [lo, hi] need (n-1) gaps. Off by one and the grid stops
    1.2 strikes short of its top, which is the wing the fit exists to supply."""
    surf = surface.fit_chain(_chain(range(80, 121, 2)), T, DTE)
    dense = surf.dense_chain(dte=DTE)
    ks = sorted({q.strike for q in dense.quotes})
    assert abs(min(ks) - dense.spot * 0.6) < 1e-3, min(ks)      # 60.0
    assert abs(max(ks) - dense.spot * 1.6) < 1e-3, max(ks)      # 160.0, not 158.77
    assert len(ks) == 81


def test_dense_chain_drops_the_worthless_strikes():
    """The far wings price to nothing and are filtered, so the chain is shorter
    than the grid: 141 quotes from 81 strikes x 2 kinds.

    The filter tests the UNROUNDED price while the quote stores the rounded one,
    so 10 of the 141 it keeps still record 0.000000 at six decimals. Harmless
    for `rnd` -- a zero-priced option contributes zero to the integral exactly
    as an absent one does -- but it is why this asserts on the count rather than
    on every mid being positive.
    """
    surf = surface.fit_chain(_chain(range(80, 121, 2)), T, DTE)
    dense = surf.dense_chain(dte=DTE)
    assert len(dense.quotes) < 2 * 81, len(dense.quotes)        # 141 of 162
    assert all(q.mid >= 0.0 for q in dense.quotes)
    assert len(dense.quotes) > 100                              # and most survive


def test_the_slope_penalty_keeps_an_inverted_smile_in_the_region():
    """An inverted smile wants total variance to FALL in the wings, i.e. b < 0,
    which is outside the admissible region. The penalty pins b at zero; without
    it the fit reaches -0.028 and reports itself as arbitrage-free anyway."""
    surf = surface.fit_chain(_shaped_chain(range(80, 121, 2), _FROWN), T, DTE)
    assert surf.b >= 0.0, surf.b                       # pinned at +0.000000
    assert surf.no_arb_ok(), (surf.a, surf.b, surf.rho, surf.sigma)
    assert surf.vol_rmse() < surface.MAX_VOL_RMSE      # 0.0164: an ACCEPTED fit
    assert _FROWN(100) > _FROWN(80) and _FROWN(100) > _FROWN(120)


def test_the_min_variance_penalty_keeps_a_pure_skew_in_the_region():
    """A monotone smile pushes the fit onto the min-variance boundary, where the
    measured minimum is +1.4e-04 over 80-120 and +1.0e-09 over 60-140. Both
    ranges are needed: dropping the penalty leaves the region on the first, and
    weakening its formula to 1+rho^2 leaves it on the second."""
    for ks in (range(80, 121, 2), range(60, 141, 5)):
        surf = surface.fit_chain(_shaped_chain(ks, _PURE_SKEW), T, DTE)
        mtv = surface.min_total_variance(surf.a, surf.b, surf.rho, surf.sigma)
        assert mtv >= 0.0, (list(ks)[:3], mtv)
        assert surf.no_arb_ok(), (surf.a, surf.b, surf.rho, surf.sigma)


def test_the_otm_leg_is_what_survives_a_realistic_tick():
    """Fitting the ITM quote instead of the OTM one is not a cosmetic swap. ITM
    value is mostly intrinsic, so the vega is small and the recovered IV is
    dominated by the tick rather than by the market.

    On exact synthetic prices the two are indistinguishable -- which is why this
    survived a sweep. At a DIME tick, the real SPX increment above $3, over
    60-140 strikes, the OTM fit errs by 0.00062 vol points and the ITM fit by
    0.02052, so the ITM fit is REFUSED by MAX_VOL_RMSE while the OTM fit stands.
    """
    dimes = _shaped_chain(range(60, 141, 5), _true_iv, rounding=1)
    surf = surface.fit_chain(dimes, T, DTE)
    assert surf.vol_rmse() < 0.005, surf.vol_rmse()            # measured 0.00062
    moments, _ = surface.robust_q_moments(dimes, T, DTE)
    assert 0.15 < moments.vol < 0.35, moments.vol


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
