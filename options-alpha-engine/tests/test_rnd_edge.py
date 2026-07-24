"""Oracle tests for the Q-extractor (rnd.py) and the P-vs-Q comparator (edge.py).

Everything ties back to the Black-Scholes ground truth already used in
test_density.py. The oracle fixture is a DENSE, ZERO-SPREAD, flat-vol BS chain
(NOT SyntheticAdapter, which injects a smile) — on it Q-skew must be ~0.
Run: python3 tests/test_rnd_edge.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import pricing
from engine.data import OptionChain, OptionQuote
from models import rnd
from models.baseline import BaselineDensityForecaster
from models.edge import compare


def _chain(S=100.0, r=0.03, q=0.0, dte=30, n=61, lo=0.55, hi=1.55, iv_fn=None):
    """Build a dense zero-spread chain. iv_fn(moneyness)->vol; None = flat 0.25."""
    T = dte / 365.0
    iv_fn = iv_fn or (lambda m: 0.25)
    quotes = []
    for i in range(n):
        K = round(S * (lo + (hi - lo) * i / (n - 1)), 2)
        vol = iv_fn(K / S - 1.0)
        for kind in ("call", "put"):
            px = pricing.price(S, K, T, r, q, vol, kind)
            quotes.append(OptionQuote(dte, K, kind, px, px))
    return OptionChain("ORACLE", S, r, q, quotes), T


def approx(a, b, tol):
    return abs(a - b) <= tol


# --- rnd.py oracle checks -------------------------------------------------
def test_model_free_iv_recovers_flat_vol():
    ch, T = _chain()
    assert approx(rnd.model_free_implied_vol(ch, T), 0.25, 0.01)


def test_bkm_flat_vol_zero_skew_zero_exkurt():
    ch, T = _chain()
    m = rnd.bkm_moments(ch, T)
    assert approx(m.vol, 0.25, 0.01), m.vol
    assert approx(m.skew, 0.0, 0.05), m.skew          # symmetric -> no skew
    assert approx(m.kurtosis, 0.0, 0.15), m.kurtosis  # ~mesokurtic
    assert m.coverage_ok


def test_bl_density_integrates_to_one():
    ch, T = _chain()
    f = rnd.risk_neutral_pdf(ch, T)
    xs = [40 + 0.5 * i for i in range(280)]
    mass = sum(f(x) * 0.5 for x in xs)
    assert approx(mass, 1.0, 0.05), mass


def test_bkm_negative_skew_on_crash_smile():
    # IV higher for low strikes (OTM puts bid up) -> negative risk-neutral skew.
    ch, T = _chain(iv_fn=lambda m: 0.25 - 0.6 * m + 0.5 * m * m)
    assert rnd.bkm_moments(ch, T).skew < -0.05


def test_bkm_positive_skew_sign():
    # IV higher for high strikes -> positive risk-neutral skew (sign is real, not assumed).
    ch, T = _chain(iv_fn=lambda m: 0.25 + 0.6 * m + 0.5 * m * m)
    assert rnd.bkm_moments(ch, T).skew > 0.05


# --- edge.py: honest verdict logic ---------------------------------------
def _p_at(vol, S=100.0, T=30 / 365):
    # A P-forecast pinned to a chosen vol (skew from data disabled for control).
    f = BaselineDensityForecaster(use_realized_skew=False)
    return f.forecast([S] * 80, T, spot=S, vol=vol)


def _q(vol, skew=0.0, cov=True):
    return rnd.RiskNeutralMoments(vol, skew, 0.0, 30 / 365, 61, cov)


def test_matched_vol_is_fair():
    s = compare(_p_at(0.20), _q(0.20), spot=100, T=30 / 365, expiry_days=30, edge_net=50)
    assert abs(s.variance_risk_premium) < 1e-3
    assert s.verdict == "FAIR", s.verdict


def test_rich_when_q_vol_exceeds_p():
    s = compare(_p_at(0.18), _q(0.26), spot=100, T=30 / 365, expiry_days=30, edge_net=120)
    assert s.variance_risk_premium > 0
    assert s.verdict == "RICH", (s.verdict, s.basis)


def test_regime_gate_suppresses_short_vol():
    s = compare(_p_at(0.18), _q(0.26), spot=100, T=30 / 365, expiry_days=30,
                edge_net=120, stressed=True)
    assert s.verdict == "NO-TRADE" and s.basis == "regime"


def test_cost_downgrades_to_fair():
    # Same rich VRP but the edge does not clear round-trip cost.
    s = compare(_p_at(0.18), _q(0.26), spot=100, T=30 / 365, expiry_days=30, edge_net=-5)
    assert s.verdict == "FAIR" and "cost" in s.note


def test_zscore_trades_deviation_not_level():
    # A positive VRP that is NORMAL vs history must NOT flag RICH (the whole
    # point: harvest deviation, not the ever-present premium level).
    hist = [0.004 + 0.0005 * i for i in range(25)]  # VRP history ~ 0.010 mean, real dispersion
    p, q = _p_at(0.19), _q(0.2166)          # VRP ~ 0.2166^2 - 0.19^2 ~ 0.0108 (normal vs hist)
    s = compare(p, q, spot=100, T=30 / 365, expiry_days=30, edge_net=120, vrp_history=hist)
    assert s.basis == "vrp_zscore" and s.verdict == "FAIR", (s.vrp_z, s.verdict)
    # Now an UNUSUALLY rich VRP relative to the same history -> RICH.
    s2 = compare(_p_at(0.15), _q(0.30), spot=100, T=30 / 365, expiry_days=30,
                 edge_net=120, vrp_history=hist)
    assert s2.verdict == "RICH", (s2.vrp_z, s2.verdict)


def test_srp_sign_is_measured_when_p_and_q_disagree():
    # Baseline P has negative physical skew; a positive-Q-skew market disagrees.
    f = BaselineDensityForecaster()
    p = f.forecast([100 + 0.1 * i for i in range(120)], 30 / 365, spot=100, vol=0.2)
    s = compare(p, _q(0.22, skew=+0.4), spot=100, T=30 / 365, expiry_days=30, edge_net=50)
    assert s.p_skew < 0 < s.q_skew                      # genuinely opposite signs
    assert approx(s.skew_risk_premium, s.q_skew - s.p_skew, 1e-9)


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
