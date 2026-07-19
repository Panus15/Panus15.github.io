"""Tests for the trainable MDN density forecaster (models/mdn.py).

The MDN is a strict DROP-IN for the HAR baseline: same `forecast(prices, T, *,
r, q, spot)` signature, same emitted `MixtureLogNormal`. These tests check it is
(1) well-posed before and after training, (2) call-compatible with the baseline,
(3) actually trainable (NLL drops), (4) free of the sigma-collapse pathology the
review flagged, and (5) usable by the objective promotion gate.

Fast by design (small context, few epochs, subsampled windows). Run:
    python3 tests/test_mdn.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.data import SyntheticAdapter
from models import objective
from models.baseline import BaselineDensityForecaster
from models.density import MixtureLogNormal
from models.mdn import SequenceMDNForecaster

# Shared, deterministic fixtures (small -> fast).
PRICES = SyntheticAdapter().price_history("X", 180)
CTX = 40
HORIZONS = (10,)


def _integrates_to_one(dist: MixtureLogNormal, grid: int = 2000) -> float:
    """Trapezoidal integral of the pdf over the mass-bearing region."""
    lo = dist.quantile(1e-4)
    hi = dist.quantile(1 - 1e-4)
    dx = (hi - lo) / grid
    total = 0.0
    prev = dist.pdf(lo)
    for i in range(1, grid + 1):
        cur = dist.pdf(lo + i * dx)
        total += 0.5 * (prev + cur) * dx
        prev = cur
    return total


def _assert_valid(dist: MixtureLogNormal, sigma_floor: float, tag: str) -> None:
    wsum = sum(c.weight for c in dist.components)
    assert abs(wsum - 1.0) < 1e-9, f"{tag}: weights sum to {wsum}, not 1"
    for c in dist.components:
        assert c.sigma >= sigma_floor - 1e-12, f"{tag}: sigma {c.sigma} < floor {sigma_floor}"
        assert c.weight >= 0.0, f"{tag}: negative weight"
    assert dist.mean() > 0.0, f"{tag}: non-positive mean"
    integ = _integrates_to_one(dist)
    assert abs(integ - 1.0) < 0.03, f"{tag}: pdf integrates to {integ}, not ~1"


def test_forecast_valid_before_fit():
    m = SequenceMDNForecaster(context=CTX, seed=0)
    dist = m.forecast(PRICES, 30 / 365, spot=PRICES[-1])
    assert isinstance(dist, MixtureLogNormal)
    _assert_valid(dist, m.sigma_floor, "before-fit")


def test_forecast_valid_after_fit():
    m = SequenceMDNForecaster(context=CTX, seed=0)
    m.fit(PRICES, horizons=HORIZONS, epochs=40, lr=0.04, max_windows=100)
    for T in (7 / 365, 30 / 365, 60 / 365):
        dist = m.forecast(PRICES, T, spot=PRICES[-1])
        _assert_valid(dist, m.sigma_floor, f"after-fit T={T:.3f}")


def test_dropin_interface_matches_baseline():
    """The exact call that works on the baseline works here, same return type."""
    prices, T, spot = PRICES, 30 / 365, PRICES[-1]
    base = BaselineDensityForecaster().forecast(prices, T, spot=spot)
    mdn = SequenceMDNForecaster(context=CTX).forecast(prices, T, spot=spot)
    assert isinstance(base, MixtureLogNormal) and isinstance(mdn, MixtureLogNormal)
    # keyword args r/q accepted identically
    mdn2 = SequenceMDNForecaster(context=CTX).forecast(prices, T, r=0.03, q=0.01, spot=spot)
    assert isinstance(mdn2, MixtureLogNormal)


def test_training_reduces_nll():
    """In-sample: mean NLL after fit < mean NLL of the untrained model."""
    windows = objective.build_windows(PRICES, context=CTX, horizons=HORIZONS)
    untrained = SequenceMDNForecaster(context=CTX, seed=0)
    nll_before = objective.dataset_nll(untrained, windows, r=0.0, q=0.0)

    trained = SequenceMDNForecaster(context=CTX, seed=0)
    trained.fit(PRICES, horizons=HORIZONS, epochs=80, lr=0.04)
    nll_after = objective.dataset_nll(trained, windows, r=0.0, q=0.0)

    assert nll_after < nll_before, f"no learning: before={nll_before:.4f} after={nll_after:.4f}"
    # training history should be monotone-ish downward overall
    assert trained.history[-1] < trained.history[0], "train NLL did not fall"


def test_no_sigma_collapse_after_training():
    """The hard floor holds across many forecasts after training."""
    m = SequenceMDNForecaster(context=CTX, seed=0)
    m.fit(PRICES, horizons=HORIZONS, epochs=40, lr=0.04, max_windows=100)
    min_sigma = float("inf")
    for t in range(CTX, len(PRICES) - 12, 5):
        ctx = PRICES[t - CTX:t]
        for T in (7 / 365, 30 / 365, 60 / 365):
            dist = m.forecast(ctx, T, spot=ctx[-1])
            for c in dist.components:
                min_sigma = min(min_sigma, c.sigma)
    assert min_sigma >= m.sigma_floor - 1e-12, f"sigma collapsed to {min_sigma}"


def test_compatible_with_dataset_nll():
    """objective.dataset_nll on a fitted model returns a finite number."""
    windows = objective.build_windows(PRICES, context=CTX, horizons=HORIZONS)[:60]
    m = SequenceMDNForecaster(context=CTX, seed=0)
    m.fit(PRICES, horizons=HORIZONS, epochs=20, lr=0.04, max_windows=100)
    val = objective.dataset_nll(m, windows, r=0.0, q=0.0)
    assert val == val and abs(val) < 1e9, f"dataset_nll not finite: {val}"


def _run_all():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    tests.sort(key=lambda f: f.__code__.co_firstlineno)
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
