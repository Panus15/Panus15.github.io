"""Tests for PIT calibration (models/calibration.py).

The oracle: generate outcomes from a KNOWN log-normal, then forecast with a
density whose vol is deliberately right / too low / too high. PIT must be uniform
in the first case and diagnose the direction of the error in the other two, and
the suggested vol scale must point back at the truth.
Run: python3 tests/test_calibration.py
"""

import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.calibration import (UNIFORM_SD, calibration_report, coverage,
                                ks_uniform, pit_histogram, pit_values,
                                suggest_vol_scale)
from models.density import single_lognormal_riskneutral

TRUE_VOL, H, SPOT = 0.30, 30, 100.0
T = H / 365.0


class FixedVolForecaster:
    """Emits a log-normal at a FIXED annualised vol — so the calibration error is
    exactly known. Honours the `vol=` override, like every forecaster here."""

    def __init__(self, vol):
        self.vol = vol

    def forecast(self, prices, T, *, r=0.0, q=0.0, spot=None, vol=None):
        return single_lognormal_riskneutral(spot if spot is not None else prices[-1],
                                            T, r, q, vol if vol is not None else self.vol)


def _windows(n=600, seed=0, vol=TRUE_VOL):
    """Outcomes drawn from the TRUE log-normal: ln(S_T/S) ~ N(-v²T/2, v²T)."""
    rng = random.Random(seed)
    s = vol * math.sqrt(T)
    out = []
    for _ in range(n):
        s_realized = SPOT * math.exp(rng.gauss(-0.5 * s * s, s))
        out.append(([SPOT] * 5, H, s_realized))
    return out


def test_correct_density_is_uniform_and_calibrated():
    rep = calibration_report(FixedVolForecaster(TRUE_VOL), _windows())
    assert abs(rep.mean - 0.5) < 0.05
    assert abs(rep.sd - UNIFORM_SD) < 0.02
    assert rep.ks_p > 0.05                              # cannot reject uniformity
    assert rep.verdict.startswith("CALIBRATED")
    assert abs(rep.vol_scale - 1.0) < 0.12              # scale points back at truth
    assert abs(rep.coverage_90 - 0.90) < 0.05


def test_too_narrow_forecast_is_diagnosed_and_scaled_up():
    # Forecast vol HALF the truth -> outcomes hit the tails -> U-shaped PIT.
    rep = calibration_report(FixedVolForecaster(TRUE_VOL * 0.5), _windows())
    assert rep.verdict.startswith("TOO NARROW")
    assert rep.sd > UNIFORM_SD                          # mass pushed to the edges
    assert rep.vol_scale > 1.4                          # tells you to roughly double
    assert rep.coverage_90 < 0.90                       # 90% interval under-covers
    h = rep.histogram
    assert h[0] + h[-1] > h[4] + h[5]                   # U-shape: edges beat middle


def test_too_wide_forecast_is_diagnosed_and_scaled_down():
    rep = calibration_report(FixedVolForecaster(TRUE_VOL * 2.0), _windows())
    assert rep.verdict.startswith("TOO WIDE")
    assert rep.sd < UNIFORM_SD                          # mass pulled to the middle
    assert rep.vol_scale < 0.75
    h = rep.histogram
    assert h[4] + h[5] > h[0] + h[-1]                   # hump: middle beats edges


def test_suggested_scale_recovers_a_known_bias():
    # A 1.6x-too-narrow forecast should be corrected by a ~1.6x scale.
    w = _windows(n=800, seed=3)
    scale = suggest_vol_scale(FixedVolForecaster(TRUE_VOL / 1.6), w)
    assert 1.35 < scale < 1.90, scale
    # applying it flattens the PIT: KS distance must drop materially
    before = ks_uniform(pit_values(FixedVolForecaster(TRUE_VOL / 1.6), w))[0]
    after = ks_uniform(pit_values(FixedVolForecaster(TRUE_VOL / 1.6), w,
                                  vol_scale=scale))[0]
    assert after < before / 2.0


def test_ks_and_histogram_primitives():
    u = [(i + 0.5) / 100 for i in range(100)]           # perfectly uniform grid
    d, p = ks_uniform(u)
    assert d < 0.02 and p > 0.9
    assert pit_histogram(u, bins=10) == [10] * 10
    assert abs(coverage(u, 0.5) - 0.5) < 0.02
    clustered = [0.5] * 50
    assert ks_uniform(clustered)[0] > 0.4               # far from uniform
    try:
        ks_uniform([])
        assert False, "empty PIT should raise"
    except ValueError:
        pass


def test_calibrated_forecaster_applies_the_scale_and_fixes_the_pit():
    from models.calibration import CalibratedForecaster
    w = _windows(n=600, seed=7)
    biased = FixedVolForecaster(TRUE_VOL / 1.6)               # 1.6x too narrow
    fixed = CalibratedForecaster(biased, vol_scale=1.6)

    # the wrapper genuinely widens the emitted density
    d0 = biased.forecast([SPOT] * 5, T, spot=SPOT)
    d1 = fixed.forecast([SPOT] * 5, T, spot=SPOT)
    assert abs(d1.log_return_vol(SPOT, T) / d0.log_return_vol(SPOT, T) - 1.6) < 1e-6

    # and the correction actually calibrates it
    before = calibration_report(biased, w, with_scale=False)
    after = calibration_report(fixed, w, with_scale=False)
    assert before.verdict.startswith("TOO NARROW")
    assert after.verdict.startswith("CALIBRATED")
    assert abs(after.centered_sd - UNIFORM_SD) < abs(before.centered_sd - UNIFORM_SD)

    # scale 1.0 is a strict no-op
    passthrough = CalibratedForecaster(biased, vol_scale=1.0)
    assert (passthrough.forecast([SPOT] * 5, T, spot=SPOT).log_return_vol(SPOT, T)
            == d0.log_return_vol(SPOT, T))


def test_walk_forward_scale_never_peeks_and_degrades_safely():
    from models.calibration import walk_forward_scale
    rng = random.Random(11)
    prices = [100.0]
    for _ in range(600):
        prices.append(prices[-1] * math.exp(rng.gauss(0.0, TRUE_VOL / math.sqrt(252))))

    seen = []

    class Spy(FixedVolForecaster):
        def forecast(self, p, T, *, r=0.0, q=0.0, spot=None, vol=None):
            seen.append(len(p))
            return super().forecast(p, T, r=r, q=q, spot=spot, vol=vol)

    cut = int(len(prices) * 0.7)
    walk_forward_scale(Spy(TRUE_VOL / 1.5), prices, dte=21, train_frac=0.7)
    assert seen, "the scale search must actually run"
    # every context came from the TRAINING split; nothing after the cut was touched
    assert max(seen) <= cut

    # too little history -> no correction rather than a fitted-on-itself one
    assert walk_forward_scale(FixedVolForecaster(TRUE_VOL), [100.0] * 50, dte=21) == 1.0
    # and the estimate is clamped to a sane band
    s = walk_forward_scale(FixedVolForecaster(TRUE_VOL / 1.5), prices, dte=21)
    assert 0.5 <= s <= 2.0


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
