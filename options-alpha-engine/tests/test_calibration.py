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
    """Outcomes drawn from the TRUE log-normal: ln(S_T/S) ~ N(-v²T/2, v²T).

    These are INDEPENDENT draws, not a rolling window over one price path, so they
    carry n genuinely independent observations. Callers must say so with
    ``stride=H`` — ``calibration_report`` conservatively assumes every-bar overlap
    otherwise, and would count 600 of these as 20.
    """
    rng = random.Random(seed)
    s = vol * math.sqrt(T)
    out = []
    for _ in range(n):
        s_realized = SPOT * math.exp(rng.gauss(-0.5 * s * s, s))
        out.append(([SPOT] * 5, H, s_realized))
    return out


def test_correct_density_is_uniform_and_calibrated():
    rep = calibration_report(FixedVolForecaster(TRUE_VOL), _windows(), stride=H)
    assert abs(rep.mean - 0.5) < 0.05
    assert abs(rep.sd - UNIFORM_SD) < 0.02
    assert rep.ks_p > 0.05                              # cannot reject uniformity
    assert rep.verdict.startswith("CALIBRATED")
    assert abs(rep.vol_scale - 1.0) < 0.12              # scale points back at truth
    assert abs(rep.coverage_90 - 0.90) < 0.05


def test_too_narrow_forecast_is_diagnosed_and_scaled_up():
    # Forecast vol HALF the truth -> outcomes hit the tails -> U-shaped PIT.
    rep = calibration_report(FixedVolForecaster(TRUE_VOL * 0.5), _windows(), stride=H)
    assert rep.verdict.startswith("TOO NARROW")
    assert rep.sd > UNIFORM_SD                          # mass pushed to the edges
    assert rep.vol_scale > 1.4                          # tells you to roughly double
    assert rep.coverage_90 < 0.90                       # 90% interval under-covers
    h = rep.histogram
    assert h[0] + h[-1] > h[4] + h[5]                   # U-shape: edges beat middle


def test_too_wide_forecast_is_diagnosed_and_scaled_down():
    rep = calibration_report(FixedVolForecaster(TRUE_VOL * 2.0), _windows(), stride=H)
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
    before = calibration_report(biased, w, with_scale=False, stride=H)
    after = calibration_report(fixed, w, with_scale=False, stride=H)
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


# --------------------------------------------------------------------------
# overlapping windows are not independent draws, and the p-value must know it
# --------------------------------------------------------------------------

def test_overlapping_windows_do_not_get_independent_credit():
    """objective.build_windows emits a window per bar; they share almost everything.

    Consecutive windows share context-1 of their context and h-1 of their horizon,
    so ~300 of them carry roughly 10 independent observations. The Kolmogorov tail
    scales by sqrt(n), so handing it the raw count overstates significance by about
    sqrt(h) — measured here at 5.5x, turning p=1.00 into p=0.005.
    """
    from engine.data import SyntheticAdapter
    from models import objective
    from models.baseline import BaselineDensityForecaster

    prices = SyntheticAdapter(seed=4).price_history("X", 400)
    w = objective.build_windows(prices, context=63, horizons=(30,))
    assert len(w) > 250, len(w)

    n_eff = objective.effective_sample_size(w, stride=1)
    assert 5 <= n_eff <= 15, f"~n/h expected, got {n_eff} from {len(w)}"

    fc = BaselineDensityForecaster()
    u = pit_values(fc, w, centered=True)
    d_naive, p_naive = ks_uniform(u)
    d_eff, p_eff = ks_uniform(u, n_effective=n_eff)
    assert d_naive == d_eff, "the DISTANCE uses all the data; only p changes"
    assert p_eff > p_naive * 10, (p_naive, p_eff)
    assert p_naive < 0.05 < p_eff, (
        f"the naive p claims significance ({p_naive:.4f}) that the effective "
        f"sample does not support ({p_eff:.4f})")


def test_non_overlapping_windows_get_full_credit():
    from engine.data import SyntheticAdapter
    from models import objective
    prices = SyntheticAdapter(seed=4).price_history("X", 1200)
    every = objective.build_windows(prices, context=63, horizons=(30,))
    w = objective.build_windows(prices, context=63, horizons=(30,), stride=30)
    # the stride must actually SPACE the windows, not just be recorded
    expected = len(range(63, len(prices) - 30, 30))
    assert len(w) == expected, f"stride=30 should give {expected} windows, got {len(w)}"
    assert len(w) < len(every) / 20
    assert objective.effective_sample_size(w, stride=30) == len(w)
    # and the offsets carve out genuinely different sub-samples
    a = objective.build_windows(prices, context=63, horizons=(30,), stride=30, offset=0)
    b = objective.build_windows(prices, context=63, horizons=(30,), stride=30, offset=7)
    assert [x[2] for x in a] != [x[2] for x in b]


def test_a_verdict_is_withheld_when_too_few_independent_windows_back_it():
    """The verdict tells the operator the VRP is overstated — it must be earned."""
    from models.calibration import MIN_EFFECTIVE_WINDOWS

    thin = calibration_report(FixedVolForecaster(TRUE_VOL * 0.5), _windows(n=600),
                              with_scale=False, stride=1)          # -> n_eff = 20
    assert thin.n_effective < MIN_EFFECTIVE_WINDOWS
    assert "NOT ENOUGH INDEPENDENT DATA" in thin.verdict, thin.verdict
    assert "leans narrow" in thin.verdict, "it should still say which way it leans"

    fat = calibration_report(FixedVolForecaster(TRUE_VOL * 0.5), _windows(n=600),
                             with_scale=False, stride=H)           # -> n_eff = 600
    assert fat.n_effective >= MIN_EFFECTIVE_WINDOWS
    assert fat.verdict.startswith("TOO NARROW"), fat.verdict
    # the point estimates are identical either way — only the CLAIM changed
    assert abs(thin.sd - fat.sd) < 1e-12 and abs(thin.ks_d - fat.ks_d) < 1e-12


def test_the_run_live_window_helper_reports_its_own_stride():
    """The old thinning computed to a step of 1 and thinned nothing."""
    from engine.data import SyntheticAdapter
    from tools.run_live import objective_windows
    from models import objective

    short = SyntheticAdapter(seed=4).price_history("X", 400)
    w, stride = objective_windows(short, 30)
    assert stride == 1 and len(w) > 250
    assert objective.effective_sample_size(w, stride=stride) < 20

    # a long history is thinned, and the stride it reports reflects the thinning —
    # which is the whole point: thinning without saying so was the original bug
    long_p = SyntheticAdapter(seed=4).price_history("X", 3000)
    raw = objective.build_windows(long_p, context=63, horizons=(30,))
    w2, stride2 = objective_windows(long_p, 30)
    assert stride2 > 1 and len(w2) < len(raw) / 5
    assert len(w2) * stride2 >= len(raw) - stride2, "thinning must not drop coverage"
    assert objective.effective_sample_size(w2, stride=stride2) > \
        objective.effective_sample_size(w, stride=stride)


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
