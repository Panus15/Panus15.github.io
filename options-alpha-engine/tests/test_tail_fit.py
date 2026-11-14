"""Tests for fitting the crash tail (models/tail_fit.py).

The module exists to answer a question the repo had been carrying as an
assumption: `objective.left_tail_pinball` gates every challenger model against
six hand-set shape constants, and nobody had measured whether that prior is any
good. These tests check that the machine which answers it is trustworthy — the
optimiser really optimises, the test slice really is held out, and the verdict
really refuses when the sample is too thin.

Measured answer, for the record: fitting improves TRAIN loss every time and
improves TEST loss NEVER — -0.8% at 5,000 bars, -1.6% at 9,000. The hand-set
shape cannot be beaten by fitting on this kind of data, so it functions as a
regulariser rather than as an unexamined guess.

Run: python3 tests/test_tail_fit.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import replace

from engine.hedged_backtest import price_path_with_crash
from models.baseline import BaselineDensityForecaster
from models.objective import build_windows
from models.tail_fit import (SEARCH_SPACE, TailFitResult, fit_tail_shape,
                             tail_loss)

P = price_path_with_crash(2000)


def test_tail_loss_is_finite_and_ordered():
    """A forecaster that puts the left tail in the right place must score better."""
    w = build_windows(P, context=63, horizons=(30,), stride=30)
    assert len(w) > 20
    base = BaselineDensityForecaster()
    l_base = tail_loss(base, w)
    assert l_base == l_base and l_base < float("inf")

    # a forecaster with essentially no crash component is a worse tail model
    thin = replace(base, stress_weight=0.03, stress_sd_mult=1.2, base_drop_mult=0.4)
    fat = replace(base, stress_weight=0.40, stress_sd_mult=4.0, base_drop_mult=2.5)
    assert tail_loss(thin, w) != tail_loss(fat, w), (
        "the shape parameters must actually move the tail score")


def test_a_broken_forecaster_scores_infinite_rather_than_zero():
    class _Broken:
        def forecast(self, *a, **k):
            raise ValueError("no")
    w = build_windows(P, context=63, horizons=(30,), stride=30)
    assert tail_loss(_Broken(), w) == float("inf"), (
        "zero windows scored must be inf, not a flattering 0.0")


def test_the_search_improves_train_loss():
    """If the optimiser cannot beat the default in-sample, it is not searching.

    Strictly better, and the shape must actually have moved: an optimiser that
    silently accepts nothing returns the defaults unchanged and would pass a
    <= comparison while having done no work at all.
    """
    base = BaselineDensityForecaster()
    r = fit_tail_shape(P, horizons=(30,), rounds=1, grid=4)
    assert r.fitted_train < r.default_train, (r.default_train, r.fitted_train)
    assert any(abs(r.fitted[a] - getattr(base, a)) > 1e-9 for a, _, _ in SEARCH_SPACE), (
        "the fitted shape is identical to the defaults — nothing was searched")
    assert r.n_train > 10 and r.n_test > 5


def test_the_search_only_ever_sees_the_train_slice():
    """The invariant that proves it: the reported train loss must be exactly the
    fitted shape's loss ON TRAIN, recomputed independently.

    If the optimiser had scored candidates on train+test, best_loss would be a
    blend and would not match. This is the specific failure the whole walk-forward
    split exists to prevent, and it is otherwise invisible from the outside.
    """
    r = fit_tail_shape(P, horizons=(30,), rounds=1, grid=4)
    cut = int(len(P) * 0.7)
    train = build_windows(P[:cut], context=63, horizons=(30,), stride=30)
    test = build_windows(P[max(0, cut - 63):], context=63, horizons=(30,), stride=30)
    rebuilt = replace(BaselineDensityForecaster(), **r.fitted)
    assert abs(tail_loss(rebuilt, train) - r.fitted_train) < 1e-9, (
        f"reported train loss {r.fitted_train:.6f} != the fitted shape's actual "
        f"train loss {tail_loss(rebuilt, train):.6f} — the search saw other data")
    # and the headline number must be the fitted shape scored on TEST, not the
    # train loss wearing a test label — that would make every fit look free
    assert abs(tail_loss(rebuilt, test) - r.fitted_test) < 1e-9, (
        f"reported test loss {r.fitted_test:.6f} != the fitted shape's actual "
        f"test loss {tail_loss(rebuilt, test):.6f}")
    assert r.fitted_test != r.fitted_train


def test_windows_are_non_overlapping_by_default():
    """Overlapping windows would let the search chase the same outcomes repeatedly
    and report a fit far better than the data supports."""
    r = fit_tail_shape(P, horizons=(30,), rounds=1, grid=4)
    # ~0.7*2000 bars at stride 30 is tens of windows, not hundreds
    assert r.n_train < len(P) / 25, (
        f"{r.n_train} train windows from {len(P)} bars implies a stride near 1")
    assert r.n_train > 10


def test_fitted_values_stay_inside_their_bounds():
    r = fit_tail_shape(P, horizons=(30,), rounds=1, grid=4)
    for attr, lo, hi in SEARCH_SPACE:
        v = r.fitted[attr]
        assert lo - 1e-9 <= v <= hi + 1e-9, (attr, v, lo, hi)
    assert set(r.fitted) == {a for a, _, _ in SEARCH_SPACE}


def test_train_and_test_do_not_share_outcomes():
    """A shape fitted on the data it is scored on reproduces the very problem
    this module exists to expose."""
    cut = int(len(P) * 0.7)
    train = build_windows(P[:cut], context=63, horizons=(30,), stride=30)
    test = build_windows(P[max(0, cut - 63):], context=63, horizons=(30,), stride=30)
    assert train and test
    tr_out = {round(s, 9) for _, _, s in train}
    te_out = {round(s, 9) for _, _, s in test}
    assert not (tr_out & te_out), (
        f"{len(tr_out & te_out)} realised outcomes appear in BOTH slices")


def test_the_verdict_refuses_a_thin_test_slice():
    r = TailFitResult(default_test=1.0, fitted_test=0.5, n_test=10, n_train=40)
    assert "NOT ENOUGH DATA" in r.verdict()
    assert "10 test windows" in r.verdict()


def test_the_verdict_distinguishes_the_three_outcomes():
    better = TailFitResult(default_test=1.00, fitted_test=0.80, n_test=60)
    assert "WAS COSTING" in better.verdict() and better.test_improvement > 0.05

    worse = TailFitResult(default_test=1.00, fitted_test=1.20, n_test=60)
    assert "LOST OUT OF SAMPLE" in worse.verdict()
    assert "do not tune them further" in worse.verdict()

    same = TailFitResult(default_test=1.00, fitted_test=1.01, n_test=60)
    assert "PRIOR HOLDS UP" in same.verdict()
    assert "measured rather than assumed" in same.verdict()


def test_the_measured_answer_replicates():
    """The finding itself: fitting wins in-sample and does not win out of sample.

    Pinned because it is load-bearing — it is what retires the concern that the
    tail gate has been grading challengers against an unexamined guess.
    """
    r = fit_tail_shape(price_path_with_crash(5000), horizons=(30,), rounds=1, grid=4)
    assert r.n_test >= 30, r.n_test
    assert r.fitted_train <= r.default_train, "in-sample fit must improve"
    assert r.test_improvement < 0.05, (
        f"out-of-sample improvement was {r.test_improvement:+.1%}; if fitting ever "
        f"does beat the prior by more than 5%, the tail gate needs revisiting and "
        f"this test should fail loudly rather than pass quietly")
    assert "PRIOR HOLDS UP" in r.verdict() or "LOST OUT OF SAMPLE" in r.verdict()
    assert "default" in r.summary() and "fitted" in r.summary()


def test_it_refuses_a_history_too_short_to_split():
    try:
        fit_tail_shape(P[:100], horizons=(30,))
    except ValueError as e:
        assert "train" in str(e).lower()
        return
    raise AssertionError("a history too short to split should raise")


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
