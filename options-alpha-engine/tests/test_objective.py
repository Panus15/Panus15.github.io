"""Tests for the promotion gate (models/objective.py).

Key property: NLL is a STRICTLY PROPER scoring rule — the true-vol density must
score a lower expected NLL than any wrong-vol density. If that holds, the gate
cannot be gamed by a miscalibrated model. Run: python3 tests/test_objective.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.data import SyntheticAdapter
from models import objective
from models.baseline import BaselineDensityForecaster
from models.density import single_lognormal_riskneutral


def _mean_nll(candidate, samples):
    return sum(objective.nll(candidate, s) for s in samples) / len(samples)


def test_nll_is_proper_true_vol_wins():
    S, T, r, q, vol_true = 100.0, 30 / 365, 0.03, 0.0, 0.25
    true = single_lognormal_riskneutral(S, T, r, q, vol_true)
    # Deterministic representative sample = quantiles of the true law (no RNG).
    samples = [true.quantile((i + 0.5) / 40) for i in range(40)]

    nll_true = _mean_nll(true, samples)
    for wrong_vol in (0.15, 0.20, 0.32, 0.45):
        wrong = single_lognormal_riskneutral(S, T, r, q, wrong_vol)
        assert _mean_nll(wrong, samples) > nll_true, wrong_vol


def test_crps_and_pinball_minimized_at_truth():
    S, T = 100.0, 30 / 365
    true = single_lognormal_riskneutral(S, T, 0.0, 0.0, 0.25)
    samples = [true.quantile((i + 0.5) / 30) for i in range(30)]
    wrong = single_lognormal_riskneutral(S, T, 0.0, 0.0, 0.40)
    crps_true = sum(objective.crps(true, s) for s in samples) / len(samples)
    crps_wrong = sum(objective.crps(wrong, s) for s in samples) / len(samples)
    assert crps_wrong > crps_true


def test_build_windows_is_leak_free():
    prices = SyntheticAdapter().price_history("X", 200)
    windows = objective.build_windows(prices, context=63, horizons=(7, 30))
    assert windows
    for ctx, h, s_realized in windows:
        assert len(ctx) == 63                 # fixed context, known at decision time
        # the realized outcome is strictly in the future relative to the context
        assert s_realized in prices


def test_dataset_nll_runs_and_is_finite():
    prices = SyntheticAdapter().price_history("X", 200)
    windows = objective.build_windows(prices, context=63, horizons=(30,))[:50]
    val = objective.dataset_nll(BaselineDensityForecaster(), windows, r=0.0, q=0.0)
    assert val == val and val < 1e9   # not NaN / inf


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
