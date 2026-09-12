"""Tests for the range-based variance estimators (engine/volforecast.py).

Why these are ORACLE tests and not backtests: the true volatility of a simulated
session is KNOWN before the code runs, so both the bias and the efficiency of an
estimator can be checked against theory rather than against a price series that
happens to be lying around. The published efficiency constants — Parkinson ~5.2x
and Garman-Klass ~7.4x the close-to-close estimator — are the reference, and an
implementation that does not reproduce them is wrong no matter how good its
forecasts look.

The second half of this file is about a specific way to lose money. The intraday
estimators measure the TRADING SESSION only and are blind to the overnight gap.
Used raw on an index they understate volatility by roughly 15%, and in an engine
whose signal is VRP = q_vol^2 - p_vol^2, understating p makes EVERY expiry look
rich. That is not a small error in a number; it is a silent conversion of the
engine into a machine that is permanently short volatility.

Run: python3 tests/test_range_vol.py
"""

import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.volforecast import (calibration_scale, daily_variance_series,
                                garman_klass_var, parkinson_var,
                                rogers_satchell_var)


def _session(sigma, rng, *, steps=200, drift=0.0, start=1.0):
    """One session of discretised GBM with a KNOWN sigma -> (o, h, l, c)."""
    s = start
    hi = lo = start
    dt = 1.0 / steps
    for _ in range(steps):
        s *= math.exp((drift - 0.5 * sigma * sigma) * dt
                      + sigma * math.sqrt(dt) * rng.gauss(0.0, 1.0))
        hi = max(hi, s)
        lo = min(lo, s)
    return (start, hi, lo, s)


def _stats(vals):
    m = sum(vals) / len(vals)
    v = sum((x - m) ** 2 for x in vals) / max(len(vals) - 1, 1)
    return m, v


def test_the_published_efficiency_constants_are_reproduced():
    """The load-bearing oracle. True sigma is fixed and known, so E[v] and Var(v)
    are measurable and comparable to theory. If this drifts, the arithmetic is
    wrong — no forecast evaluation can tell you that."""
    sigma, n = 0.20, 12_000
    rng = random.Random(7)
    bars = [_session(sigma, rng) for _ in range(n)]

    cc = [math.log(b[3] / b[0]) ** 2 for b in bars]
    _, v_cc = _stats(cc)
    eff = {}
    for name, fn in (("parkinson", parkinson_var),
                     ("garman_klass", garman_klass_var),
                     ("rogers_satchell", rogers_satchell_var)):
        m, v = _stats([fn(b) for b in bars])
        eff[name] = v_cc / v
        # discretely-sampled highs and lows understate the continuous extremes,
        # so a small downward bias is expected and REAL — markets are sampled
        # discretely too. What is not allowed is a large one.
        assert 0.85 < m / (sigma ** 2) < 1.05, f"{name} level: {m / sigma**2:.3f}"

    assert 4.0 < eff["parkinson"] < 6.5, eff
    assert 6.0 < eff["garman_klass"] < 10.0, eff
    assert 5.0 < eff["rogers_satchell"] < 9.0, eff
    assert eff["garman_klass"] > eff["parkinson"], eff


def test_only_rogers_satchell_refuses_to_read_a_trend_as_volatility():
    """Measured, not assumed. Bias vs the true sigma^2 as drift/sigma rises:

        drift/sigma   parkinson   garman-klass   rogers-satchell
              0.0       0.913        0.881            0.883
              1.0       1.210        0.971            0.858
              2.0       2.226        1.286            0.810
              4.0       6.375        2.531            0.691

    Parkinson uses only the high-low range, and a trending session makes a wide
    range without making a volatile one, so it reads the trend as risk. Garman-
    Klass inflates too, more slowly. Rogers-Satchell is constructed to cancel the
    drift term and never inflates - it only drifts DOWN, which is the known
    discretisation bias shared by all three.

    The direction matters for this engine. An inflated p_vol makes options look
    CHEAP and suppresses selling, which is the safe way to be wrong; but it is
    still wrong, and on a strongly trending underlying it is wrong by multiples.
    """
    sigma, n = 0.20, 6_000
    def bias(fn, k):
        rng = random.Random(11)
        bars = [_session(sigma, rng, drift=k * sigma) for _ in range(n)]
        m, _ = _stats([fn(b) for b in bars])
        return m / sigma ** 2

    p0, p2 = bias(parkinson_var, 0.0), bias(parkinson_var, 2.0)
    r0, r2 = bias(rogers_satchell_var, 0.0), bias(rogers_satchell_var, 2.0)

    assert p2 > 1.8 * p0, f"Parkinson stopped inflating with drift ({p0:.3f}->{p2:.3f})"
    assert r2 <= r0 + 0.02, f"Rogers-Satchell inflated with drift ({r0:.3f}->{r2:.3f})"
    assert p2 > 2.0 * r2, "the two estimators stopped being distinguishable"



def _gapped_world(n, sigma_session, gap_share, seed):
    """Sessions plus overnight gaps carrying `gap_share` of the TOTAL variance."""
    rng = random.Random(seed)
    total = sigma_session ** 2 / (1.0 - gap_share)
    gap_sd = math.sqrt(total * gap_share)
    bars, prev_close = [], 1.0
    for _ in range(n):
        o = prev_close * math.exp(rng.gauss(0.0, gap_sd))
        bar = _session(sigma_session, rng, start=o)
        bars.append(bar)
        prev_close = bar[3]
    return bars, total


def test_a_gap_blind_estimator_understates_vol_and_that_is_the_dangerous_direction():
    """The safety property, stated as the failure it prevents.

    With a quarter of the variance arriving overnight, a session-only estimator
    must read about sqrt(0.75) = 0.87 of the truth. In this engine that makes
    every expiry look rich.
    """
    bars, total = _gapped_world(4_000, 0.18, 0.25, seed=3)
    blind = daily_variance_series(bars, estimator="parkinson")
    m_blind = sum(blind) / len(blind)
    assert m_blind / total < 0.90, (
        f"gap-blind estimator read {m_blind / total:.3f} of true variance; if this "
        f"is ever near 1.0 the fixture stopped testing anything")

    gapped = daily_variance_series(bars, estimator="gkyz")
    m_gap = sum(gapped[1:]) / len(gapped[1:])
    assert 0.88 < m_gap / total < 1.12, (
        f"the gap-aware estimator is off by {m_gap / total:.3f}x — it is supposed "
        f"to be the one that can be trusted on level")


def test_calibration_puts_a_gap_blind_series_back_on_level():
    bars, total = _gapped_world(3_000, 0.18, 0.25, seed=5)
    raw = daily_variance_series(bars, estimator="parkinson")
    scale = calibration_scale(bars, estimator="parkinson")
    fixed = sum(raw) / len(raw) * scale
    assert scale > 1.0, "a session-only series must be scaled UP, not down"
    assert 0.85 < fixed / total < 1.15, fixed / total


def test_calibration_refuses_to_invent_a_scale_from_nothing():
    """A short or flat window must return 1.0 rather than a wild multiplier — a
    rescale computed from two bars would move every forecast that followed."""
    assert calibration_scale([], estimator="gkyz") == 1.0
    assert calibration_scale([(1.0, 1.0, 1.0, 1.0)], estimator="gkyz") == 1.0
    flat = [(1.0, 1.0, 1.0, 1.0)] * 50
    assert calibration_scale(flat, estimator="gkyz") == 1.0


def test_the_close_to_close_arm_still_reproduces_the_old_behaviour():
    """The A/B control. If 'cc' ever stops matching squared log close-to-close
    returns, every comparison in this file is measuring two new things at once."""
    rng = random.Random(2)
    bars = [_session(0.2, rng, start=1.0) for _ in range(50)]
    cc = daily_variance_series(bars, estimator="cc")
    assert cc[0] == 0.0, "the first bar has no yesterday"
    for i in range(1, len(bars)):
        expect = math.log(bars[i][3] / bars[i - 1][3]) ** 2
        assert abs(cc[i] - expect) < 1e-15


def test_an_unknown_estimator_is_refused_rather_than_defaulted():
    try:
        daily_variance_series([(1.0, 1.0, 1.0, 1.0)], estimator="magic")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown estimator silently picked one")


def test_a_degenerate_bar_contributes_zero_not_a_nan():
    for bad in ((0.0, 1.0, 1.0, 1.0), (1.0, -1.0, 1.0, 1.0), (1.0, 1.0, 0.0, 1.0)):
        for fn in (parkinson_var, garman_klass_var, rogers_satchell_var):
            v = fn(bad)
            assert v == 0.0 or (v == v and abs(v) < float("inf")), (fn.__name__, bad, v)


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
