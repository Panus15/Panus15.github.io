"""Fit the crash-tail shape instead of guessing it — and report what the guess cost.

`baseline.BaselineDensityForecaster` carries six numbers that decide the shape of
its left tail:

    stress_weight  stress_sd_mult  base_drop_mult
    leverage_exp   horizon_exp     realized_gain

All six were set by hand. That would be unremarkable except for what depends on
them: `objective.left_tail_pinball` is the gate a challenger model must pass before
it can ship, and it scores that challenger AGAINST THIS SHAPE. So "passed the tail
gate" has never meant "models crashes well" — it has meant "beat one person's
prior about crashes", and nobody has measured how good that prior is.

This module measures it. It fits the six numbers by coordinate descent on the
left-tail pinball loss over a TRAIN slice, scores the result on a held-out TEST
slice, and reports both against the shipped defaults. Three outcomes, all useful:

  the fit clearly beats the defaults   the hand-set tail was costing accuracy, and
                                       the gate has been grading on a soft curve
  the fit matches the defaults         the prior was good; now that is a finding
                                       rather than an assumption
  the fit LOSES out of sample          the shape cannot be identified from this
                                       much data, which is itself worth knowing
                                       before anyone tunes it further

The split is walk-forward and the search never sees the test slice. A shape fitted
on the data it is then scored on would reproduce exactly the problem this module
exists to expose.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

from models.baseline import BaselineDensityForecaster
from models.objective import build_windows, left_tail_pinball

#: (attribute, lower, upper) for every shape parameter that gets fitted.
#: Bounds are wide enough to contain shapes the author would not have chosen —
#: a search confined to a neighbourhood of the guess could only ever confirm it.
SEARCH_SPACE = (
    ("stress_weight", 0.03, 0.45),
    ("stress_sd_mult", 1.20, 4.00),
    ("base_drop_mult", 0.40, 2.50),
    ("leverage_exp", 0.00, 1.20),
    ("horizon_exp", 0.00, 0.80),
    ("realized_gain", 0.00, 4.00),
)


def tail_loss(forecaster, windows, *, r: float = 0.0, q: float = 0.0,
              taus=(0.05, 0.10)) -> float:
    """Mean left-tail pinball loss — lower is better. NaN-safe."""
    total, n = 0.0, 0
    for ctx, h, s_real in windows:
        try:
            d = forecaster.forecast(ctx, h / 365.0, r=r, q=q, spot=ctx[-1])
            v = left_tail_pinball(d, s_real, taus=taus)
        except (ValueError, ZeroDivisionError, ArithmeticError, OverflowError):
            continue
        if v == v and abs(v) != float("inf"):
            total += v
            n += 1
    return total / n if n else float("inf")


@dataclass
class TailFitResult:
    fitted: dict = field(default_factory=dict)
    default_train: float = 0.0
    fitted_train: float = 0.0
    default_test: float = 0.0
    fitted_test: float = 0.0
    n_train: int = 0
    n_test: int = 0

    @property
    def test_improvement(self) -> float:
        """Fraction by which the fit beat the defaults out of sample (>0 = better)."""
        if self.default_test <= 0:
            return 0.0
        return 1.0 - self.fitted_test / self.default_test

    def verdict(self) -> str:
        imp = self.test_improvement
        if self.n_test < 30:
            return (f"NOT ENOUGH DATA — {self.n_test} test windows. The tail is the "
                    f"rarest thing in the sample and needs the most of it.")
        if imp > 0.05:
            return (f"THE HAND-SET TAIL WAS COSTING {imp:.1%} of left-tail accuracy "
                    f"out of sample. objective.left_tail_pinball has been grading "
                    f"challengers against a prior that is beatable, so 'passed the "
                    f"tail gate' meant less than it sounded.")
        if imp < -0.05:
            return (f"THE FIT LOST OUT OF SAMPLE by {-imp:.1%} — the shape cannot be "
                    f"identified from this much data. Treat the shipped defaults as "
                    f"a regulariser, and do not tune them further on this sample.")
        return (f"THE PRIOR HOLDS UP — fitting moved out-of-sample tail loss by only "
                f"{imp:+.1%}. The hand-set shape was a good guess, and that is now "
                f"measured rather than assumed.")

    def summary(self) -> str:
        rows = "\n".join(
            f"    {k:16}{getattr(BaselineDensityForecaster(), k):>8.3f}  ->{v:>8.3f}"
            for k, v in self.fitted.items())
        return (f"Crash-tail shape: fitted vs the shipped hand-set defaults\n"
                f"  train {self.n_train} windows / test {self.n_test} windows "
                f"(walk-forward, the search never sees test)\n"
                f"    {'':16}{'default':>8}   {'fitted':>8}\n{rows}\n"
                f"  left-tail pinball   train {self.default_train:.4f} -> "
                f"{self.fitted_train:.4f}   test {self.default_test:.4f} -> "
                f"{self.fitted_test:.4f}\n"
                f"  -> {self.verdict()}")


def fit_tail_shape(prices, *, horizons=(30,), context: int = 63,
                   train_frac: float = 0.7, stride: int | None = None,
                   rounds: int = 2, grid: int = 5, r: float = 0.0,
                   q: float = 0.0, base=None) -> TailFitResult:
    """Coordinate-descent the six shape parameters on a TRAIN slice only.

    ``stride`` defaults to the longest horizon, i.e. non-overlapping windows —
    overlapping ones would let the search chase the same few realised outcomes
    many times and report a fit far better than the data supports.
    """
    base = base or BaselineDensityForecaster()
    step = stride if stride else max(horizons)
    cut = int(len(prices) * train_frac)
    train = build_windows(prices[:cut], context=context, horizons=horizons,
                          stride=step)
    # the test slice starts far enough in that no training outcome is reused
    test = build_windows(prices[max(0, cut - context):], context=context,
                         horizons=horizons, stride=step)
    if not train or not test:
        raise ValueError("not enough history to split into train and test")

    best = replace(base)
    best_loss = tail_loss(best, train, r=r, q=q)
    for _ in range(max(1, rounds)):
        for attr, lo, hi in SEARCH_SPACE:
            cur = getattr(best, attr)
            span = (hi - lo) / 2.0
            for _ in range(2):                       # coarse then fine
                cands = [lo + (hi - lo) * i / (grid - 1) for i in range(grid)] \
                    if span > (hi - lo) / 2.5 else \
                    [max(lo, min(hi, cur + span * (2 * i / (grid - 1) - 1)))
                     for i in range(grid)]
                for v in cands:
                    trial = replace(best, **{attr: v})
                    lv = tail_loss(trial, train, r=r, q=q)
                    if lv < best_loss:
                        best_loss, best, cur = lv, trial, v
                span /= 3.0

    return TailFitResult(
        fitted={a: getattr(best, a) for a, _, _ in SEARCH_SPACE},
        default_train=tail_loss(base, train, r=r, q=q), fitted_train=best_loss,
        default_test=tail_loss(base, test, r=r, q=q),
        fitted_test=tail_loss(best, test, r=r, q=q),
        n_train=len(train), n_test=len(test),
    )
