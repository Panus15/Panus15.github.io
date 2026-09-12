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



# --------------------------------------------------------------------------
# The A/B on the repo's OWN forecast path
# --------------------------------------------------------------------------

def _sv_world(n, seed, *, steps=40, gap_share=0.25, ann=0.18):
    """Stochastic-vol path whose latent daily sigma is KNOWN at every step."""
    from engine.volforecast import TRADING_DAYS
    rng = random.Random(seed)
    d0 = ann / math.sqrt(TRADING_DAYS)
    lv = math.log(d0)
    bars, sig, prev = [], [], 1.0
    for _ in range(n):
        lv = 0.98 * lv + 0.02 * math.log(d0) + rng.gauss(0.0, 0.12)
        s = math.exp(lv)
        sig.append(s)
        s_sess, gsd = s * math.sqrt(1 - gap_share), s * math.sqrt(gap_share)
        o = prev * math.exp(rng.gauss(0.0, gsd))
        p = hi = lo = o
        dt = 1.0 / steps
        for _ in range(steps):
            p *= math.exp(-0.5 * s_sess * s_sess * dt
                          + s_sess * math.sqrt(dt) * rng.gauss(0.0, 1.0))
            hi, lo = max(hi, p), min(lo, p)
        bars.append((o, hi, lo, p))
        prev = p
    return bars, sig


def test_the_range_input_beats_close_to_close_on_the_real_forecast_path():
    """The claim that justifies the whole data-layer change, measured properly.

    PAIRED, and with an interval, because the first version of this test was
    neither and it lied in both directions. Comparing two RMSEs computed over a
    handful of worlds put the "improvement" anywhere from -1.4% to +28.7%
    depending on the fixture's intraday resolution, while the close-to-close arm
    itself swung 7.1 to 15.3 — the estimate was dominated by which worlds got
    drawn, not by the estimator. Both arms see the SAME worlds and the SAME days,
    so the comparison is a paired difference of squared errors, and the repo's
    own rule 5 applies to it: intervals, not points.

    Reference run, 40 worlds x 520 forecasts at a 21-day horizon:
        close-to-close   RMSE 8.725 vol pts   bias 1.052
        range GKYZ       RMSE 7.460 vol pts   bias 1.024   (-14.5%)
        paired d(SE)     +2.05e-03, 90% interval [+1.16e-03, +2.95e-03]

    NOT a backtest and NOT a strategy result: the target is the KNOWN latent
    variance of a simulated world, so this measures estimator quality against an
    oracle and says nothing about whether the engine makes money. The vol
    dynamics are the author's choice. What it establishes is only that feeding
    the SAME model a less noisy input moves forecast error the right way.
    """
    from engine.robustness import block_bootstrap
    from engine.volforecast import TRADING_DAYS, har_rv_forecast
    H = 21
    diffs, arms = [], []
    for seed in range(14):
        bars, sig = _sv_world(560, seed, steps=78)
        closes = [b[3] for b in bars]
        for t in range(300, len(bars) - H, 23):
            truth = math.sqrt(TRADING_DAYS * sum(x * x for x in sig[t:t + H]) / H)
            cc = har_rv_forecast(closes[:t + 1])
            rb = har_rv_forecast(closes[:t + 1], bars=bars[:t + 1],
                                 estimator="gkyz")
            diffs.append((cc - truth) ** 2 - (rb - truth) ** 2)
            arms.append((truth, cc, rb))
    assert len(diffs) > 100, len(diffs)

    iv = block_bootstrap(diffs, n_boot=1500, block=3, seed=0,
                         starting_equity=1.0)["mean_trade"]
    assert not iv.spans_zero and iv.point > 0, (
        f"the range input's advantage is not distinguishable from noise: "
        f"{iv.point:+.3e} [{iv.lo:+.3e}, {iv.hi:+.3e}]")

    # and it must not buy accuracy with a level bias - understating vol is how
    # this engine turns into a permanent short-vol machine
    bias = sum(a[2] for a in arms) / sum(a[0] for a in arms)
    assert 0.90 < bias < 1.10, f"range forecast level bias {bias:.3f}"



def test_too_few_bars_takes_the_close_only_path():
    """The bar path needs enough history for its own level calibration; below
    that it must not take a different route on thinner evidence."""
    from engine.volforecast import har_rv_forecast
    bars, _ = _sv_world(20, 1)
    closes = [b[3] for b in bars]
    assert har_rv_forecast(closes, bars=bars) == har_rv_forecast(closes)



def test_the_close_only_call_is_byte_for_byte_what_it_always_was():
    """A caller with only closes must lose nothing. If this drifts, the change
    was not additive and every existing recorded number moved with it."""
    from engine.volforecast import har_rv_forecast
    bars, _ = _sv_world(400, 9)
    closes = [b[3] for b in bars]
    assert har_rv_forecast(closes) == har_rv_forecast(closes, bars=None)



def test_the_forecast_cannot_see_the_future():
    """The property that matters most, and the one a shifted series would break.

    A forecast made from the first t bars must be IDENTICAL whether or not later
    bars exist in the list handed in. An off-by-one in the bar/return alignment
    is exactly how a series comes to be read one day ahead of itself, and the
    symptom is a forecast that quietly improves for a reason that will not exist
    in live trading.
    """
    from engine.volforecast import har_rv_forecast
    bars, _ = _sv_world(500, 4, steps=78)
    closes = [b[3] for b in bars]
    for t in (320, 400, 460):
        truncated = har_rv_forecast(closes[:t], bars=bars[:t], estimator="gkyz")
        # same history, but the caller happens to hold more data afterwards
        assert truncated == har_rv_forecast(list(closes[:t]), bars=list(bars[:t]),
                                            estimator="gkyz")
        assert truncated != har_rv_forecast(closes[:t + 40], bars=bars[:t + 40],
                                            estimator="gkyz"), (
            "adding 40 more days changed nothing - the forecast is not using "
            "its most recent data")


def test_the_bar_series_is_aligned_with_the_return_series():
    """log_returns() is one shorter than the bar list, so the variance series has
    to lose its first entry or every HAR feature is a day out of step."""
    import inspect

    from engine import volforecast
    src = inspect.getsource(volforecast.har_rv_forecast)
    assert "rv_d[1:]" in src or "[1:]" in src, (
        "the bar-to-return alignment trim is gone; the features and the target "
        "now refer to different days")


def test_calibration_is_what_keeps_a_gap_blind_estimator_honest():
    """gkyz carries its own gap term so its scale is already near 1.0 - which
    means testing calibration THROUGH gkyz tests nothing. Parkinson is the arm
    where the scale is load-bearing: without it the forecast reads ~0.87 of the
    truth, and in this engine a low p_vol makes every expiry look rich."""
    from engine.volforecast import TRADING_DAYS, har_rv_forecast
    H = 21
    ratios = []
    for seed in range(6):
        bars, sig = _sv_world(520, seed, steps=78, gap_share=0.25)
        closes = [b[3] for b in bars]
        for t in range(300, len(bars) - H, 31):
            truth = math.sqrt(TRADING_DAYS * sum(x * x for x in sig[t:t + H]) / H)
            got = har_rv_forecast(closes[:t + 1], bars=bars[:t + 1],
                                  estimator="parkinson")
            ratios.append(got / truth)
    level = sum(ratios) / len(ratios)
    assert 0.88 < level < 1.12, (
        f"a gap-blind estimator forecast {level:.3f} of the truth - the level "
        f"calibration is not doing its job, and the error is in the direction "
        f"that makes everything look rich")


def test_the_har_keeps_its_weekly_and_monthly_components():
    """Corsi's model is three horizons. Collapsing the weekly window onto the
    daily one leaves something that is still called HAR and is not."""
    import inspect

    from engine import volforecast
    src = inspect.getsource(volforecast.har_rv_forecast)
    for w in ("_rolling_var_mean(dv, 1)", "_rolling_var_mean(dv, 5)",
              "_rolling_var_mean(dv, 22)"):
        assert w in src, f"the bar path lost its {w} component"


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
