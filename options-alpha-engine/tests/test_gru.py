"""Tests for the GRU-MDN sequence forecaster (models/gru.py).

The load-bearing test is the finite-difference gradient check: a hand-derived
backpropagation-through-time is only trustworthy if its analytic gradients match
numerical ones on every parameter block. The rest confirm the drop-in contract
(valid MixtureLogNormal), that training reduces NLL, and that it plugs into the
same promotion gate as every other forecaster. Needs numpy.
Run: python3 tests/test_gru.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The repo's contract is that the core runs on the standard library ALONE, so an
# optional dependency must produce a SKIP, not a red suite. This file needs numpy
# and said so in its docstring while importing it unguarded, which meant every
# stdlib-only machine — including the operator's — saw a hard failure.
try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False

if HAVE_NUMPY:
    from models import objective
    from models.baseline import BaselineDensityForecaster
    from models.gru import GRUMDNForecaster, _seq_features

    _rng = np.random.default_rng(0)
    PRICES = [100.0]
    for _ in range(400):
        PRICES.append(PRICES[-1] * math.exp(_rng.normal(0.0002, 0.012)))
else:
    PRICES = []


def test_bptt_gradients_match_finite_difference():
    # THE test: analytic BPTT grads must equal central finite differences.
    m = GRUMDNForecaster(n_components=2, hidden=4, seq_len=6, context=20, seed=1)
    rng = np.random.default_rng(2)
    X = rng.standard_normal((5, m.L, m.D))
    z = rng.standard_normal((5, 1)) * 0.1
    _, grads = m._loss_and_grads(X, z)
    eps = 1e-5
    for p, g in zip(m._param_list(), grads):
        flat, gflat = p.reshape(-1), g.reshape(-1)
        for i in rng.choice(len(flat), size=min(4, len(flat)), replace=False):
            orig = flat[i]
            flat[i] = orig + eps
            plus, _ = m._loss_and_grads(X, z)
            flat[i] = orig - eps
            minus, _ = m._loss_and_grads(X, z)
            flat[i] = orig
            fd = (plus - minus) / (2 * eps)
            assert abs(fd - gflat[i]) < 1e-4, (p.shape, i, fd, gflat[i])


def test_seq_features_shape_and_padding():
    f = _seq_features([100, 101, 102], seq_len=5)
    assert f.shape == (5, 2)
    assert np.allclose(f[:3], 0.0)                    # left-padded
    assert f[-1, 1] == abs(f[-1, 0])                  # |r| column


def test_training_reduces_nll():
    m = GRUMDNForecaster(hidden=6, seq_len=20, context=63, seed=0)
    m.fit(PRICES, horizons=(21,), epochs=60, lr=0.05, max_windows=200)
    assert m.history[-1] < m.history[0]
    assert all(math.isfinite(v) for v in m.history)


def test_forecast_is_a_valid_density():
    m = GRUMDNForecaster(hidden=6, seq_len=20, context=63, seed=0)
    m.fit(PRICES, horizons=(21,), epochs=40, lr=0.05, max_windows=150)
    d = m.forecast(PRICES, 21 / 365.0, spot=PRICES[-1])
    assert abs(sum(c.weight for c in d.components) - 1.0) < 1e-9
    assert all(c.sigma > 0 for c in d.components)
    assert d.pdf(PRICES[-1]) > 0 and math.isfinite(d.pdf(PRICES[-1]))
    assert d.log_return_vol(PRICES[-1], 21 / 365.0) > 0
    # vol override keeps a valid density
    d2 = m.forecast(PRICES, 21 / 365.0, spot=PRICES[-1], vol=0.25)
    assert abs(d2.log_return_vol(PRICES[-1], 21 / 365.0) - 0.25) < 1e-6


def test_plugs_into_the_promotion_gate():
    # It must be scorable by the same OOS gate as every forecaster — the gate,
    # not the author, decides if it ships. (No claim it beats the baseline here.)
    cut = int(len(PRICES) * 0.75)
    m = GRUMDNForecaster(hidden=6, seq_len=20, context=63, seed=0)
    m.fit(PRICES[:cut], horizons=(21,), epochs=40, lr=0.05, max_windows=150)
    oos = objective.build_windows(PRICES[cut - 63:], context=63, horizons=(21,))
    gru_nll = objective.dataset_nll(m, oos)
    base_nll = objective.dataset_nll(BaselineDensityForecaster(), oos)
    assert math.isfinite(gru_nll) and math.isfinite(base_nll)


def _run_all():
    if not HAVE_NUMPY:
        # printing PASS for a test that returned without asserting anything is
        # the same lie the rest of this repo has been hunting: work that did not
        # happen must not look like work that succeeded
        print("SKIP (numpy not installed) — the GRU-MDN forecaster is optional")
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
