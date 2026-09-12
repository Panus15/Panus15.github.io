"""Tests for the trained neural encoder + MDN head (models/neural.py).

Requires numpy; SKIPS cleanly if numpy is absent, so the stdlib suite is never
broken by it. The key check is that end-to-end backprop actually LEARNS —
training must drive the NLL down monotonically-ish, not up (a sign error in the
MDN gradient would make it rise). Run: python3 tests/test_neural.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import numpy  # noqa
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False

if HAVE_NUMPY:
    from engine.data import SyntheticAdapter
    from models import objective
    from models.density import MixtureLogNormal
    from models.neural import NeuralMDNForecaster

PRICES = SyntheticAdapter(seed=3).price_history("X", 320) if HAVE_NUMPY else []


def _valid(dist, spot, T):     # unannotated on purpose: the name
    # only exists when numpy does, and an annotation is evaluated at
    # def time, so annotating it made the 'clean skip' a NameError
    w = sum(c.weight for c in dist.components)
    assert abs(w - 1.0) < 1e-6
    assert all(c.sigma > 0 for c in dist.components)
    assert dist.mean() > 0
    assert dist.log_return_vol(spot, T) > 0


def test_forecast_valid_before_and_after_fit():
    m = NeuralMDNForecaster(context=63, seed=0)
    _valid(m.forecast(PRICES, 30 / 365, spot=PRICES[-1]), PRICES[-1], 30 / 365)
    m.fit(PRICES, horizons=(30,), epochs=60, lr=0.05, max_windows=150)
    _valid(m.forecast(PRICES, 30 / 365, spot=PRICES[-1]), PRICES[-1], 30 / 365)


def test_dropin_interface():
    # Same call shape as the baseline / stdlib MDN.
    m = NeuralMDNForecaster(context=63)
    d = m.forecast(PRICES, 30 / 365, r=0.03, q=0.01, spot=PRICES[-1])
    assert isinstance(d, MixtureLogNormal)


def test_training_reduces_nll_end_to_end():
    m = NeuralMDNForecaster(context=63, seed=0)
    m.fit(PRICES, horizons=(30,), epochs=200, lr=0.05, max_windows=200)
    assert m.history[-1] < m.history[0], (m.history[0], m.history[-1])
    # a meaningful drop, not a rounding wiggle
    assert m.history[-1] < m.history[0] - 0.05


def test_no_sigma_collapse():
    m = NeuralMDNForecaster(context=63, seed=0, sigma_floor=0.03)
    m.fit(PRICES, horizons=(30,), epochs=150, lr=0.05, max_windows=150)
    for T in (7 / 365, 30 / 365, 60 / 365):
        d = m.forecast(PRICES, T, spot=PRICES[-1])
        assert min(c.sigma for c in d.components) >= 0.03 * (T ** 0.5) - 1e-9


def test_compatible_with_dataset_nll():
    m = NeuralMDNForecaster(context=63, seed=0).fit(
        PRICES, horizons=(30,), epochs=80, lr=0.05, max_windows=120)
    windows = objective.build_windows(PRICES, context=63, horizons=(30,))[:40]
    val = objective.dataset_nll(m, windows, r=0.0, q=0.0)
    assert val == val and val < 1e9


def _run_all():
    if not HAVE_NUMPY:
        print("SKIP (numpy not installed) — neural encoder is an optional module")
        print("\n0/0 passed")
        return 0
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
