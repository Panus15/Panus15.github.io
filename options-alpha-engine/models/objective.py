"""Promotion gate — how you PROVE the future MDN beats the HAR baseline.

Phase-1b (ships after rnd.py/edge.py). The core ML technique is only worth
upgrading to a TFT+MDN if the neural density genuinely scores better OUT OF
SAMPLE on the physical S_T distribution. This module is the honest scoreboard:
proper scoring rules over a leak-free supervised dataset.

  nll        negative log-likelihood — the MDN's training loss AND the headline
             model-selection scalar (a strictly proper scoring rule).
  crps       continuous ranked probability score — robust, validation-only.
  pinball    quantile loss across the distribution.
  left_tail  pinball at tau in {0.05, 0.10} ONLY — a DECOMPOSED gate for the
             crash-tail/skew dimension, which the aggregate NLL (dominated by
             the vol-level term) otherwise masks. An MDN must beat the baseline
             HERE to claim a real skew edge, not just better vol.

Why a decomposed gate: getting the vol LEVEL right dominates NLL, and HAR-RV is a
genuine wall there — so an MDN can pass aggregate NLL while emitting garbage
tails, or improve real tails yet fail the aggregate. Score the tail separately.
"""

from __future__ import annotations

import math

from .density import MixtureLogNormal


def nll(dist: MixtureLogNormal, s_realized: float) -> float:
    """-log f(s). Lower is better. The MDN training loss / selection scalar."""
    p = dist.pdf(s_realized)
    return -math.log(max(p, 1e-300))


def crps(dist: MixtureLogNormal, s_realized: float, *, grid: int = 512) -> float:
    """CRPS = ∫ (F(x) - 1{x >= s})^2 dx, integrated over the bulk of the density."""
    lo = dist.quantile(0.001)
    hi = dist.quantile(0.999)
    hi = max(hi, s_realized * 1.05)
    lo = min(lo, s_realized * 0.95)
    dx = (hi - lo) / grid
    total = 0.0
    for i in range(grid + 1):
        x = lo + i * dx
        indicator = 1.0 if x >= s_realized else 0.0
        total += (dist.cdf(x) - indicator) ** 2 * dx
    return total


def pinball(dist: MixtureLogNormal, s_realized: float, taus=(0.1, 0.25, 0.5, 0.75, 0.9)) -> float:
    """Mean quantile (pinball) loss across the distribution."""
    total = 0.0
    for tau in taus:
        qh = dist.quantile(tau)
        total += (s_realized - qh) * tau if s_realized >= qh else (qh - s_realized) * (1 - tau)
    return total / len(taus)


def left_tail_pinball(dist: MixtureLogNormal, s_realized: float, taus=(0.05, 0.10)) -> float:
    """Decomposed crash-tail score — the dimension the aggregate NLL hides."""
    return pinball(dist, s_realized, taus=taus)


def build_windows(prices, *, context: int = 63, horizons=(7, 30, 60),
                  stride: int = 1, offset: int = 0):
    """Leak-free supervised dataset: (context_prices, horizon_days, s_realized).

    The context window ENDS strictly before the horizon it predicts, so no future
    price ever enters a forecast. This is the single most important guard against
    the look-ahead that silently inflates OOS scores.

    OVERLAP, and why ``stride`` exists. With the default ``stride=1`` a window is
    emitted for every t, so consecutive windows share ``context - 1`` of their
    context and ``h - 1`` of their horizon — from 400 bars at h=30 this yields 307
    windows of which only ~11 are independent. That is fine for TRAINING (more,
    correlated examples still carry signal) and wrong for any TEST that assumes
    iid draws: a Kolmogorov-Smirnov p-value computed on 307 overlapping windows
    reports significance it has not earned. Pass ``stride=h`` for a genuinely
    non-overlapping sample, and ``offset`` to pick which of the h disjoint
    sub-samples you get, so the choice of start bar is not silently load-bearing.
    """
    out = []
    n = len(prices)
    step = max(1, int(stride))
    for h in horizons:
        start = context + (int(offset) % step)
        for t in range(start, n - h, step):
            ctx = prices[t - context:t]          # known at decision time t-1
            s_realized = prices[t + h - 1]       # the outcome h days later
            out.append((ctx, h, s_realized))
    return out


def effective_sample_size(windows, *, stride: int = 1) -> int:
    """How many of these windows are approximately INDEPENDENT.

    Two windows whose decision dates are fewer than ``h`` bars apart share part of
    the same realised path, so they are not two observations of the model being
    right or wrong — they are close to one. Windows ``stride`` apart carry roughly
    ``min(stride, h) / h`` of an independent observation each; once stride reaches
    the horizon they are independent and the count is the raw one.

    This is the honest denominator for any test that assumes iid draws. Returns at
    least 1 so callers can divide by it.
    """
    if not windows:
        return 0
    counts: dict = {}
    for _, h, _ in windows:
        counts[h] = counts.get(h, 0) + 1
    step = max(1, int(stride))
    total = 0.0
    for h, n in counts.items():
        total += n * min(step, max(1, int(h))) / max(1, int(h))
    return max(1, int(total))


def dataset_nll(forecaster, windows, *, r: float = 0.0, q: float = 0.0) -> float:
    """Mean NLL of a forecaster over a window set. The OOS promotion number."""
    if not windows:
        raise ValueError("no windows")
    total = 0.0
    for ctx, h, s_realized in windows:
        dist = forecaster.forecast(ctx, h / 365.0, r=r, q=q, spot=ctx[-1])
        total += nll(dist, s_realized)
    return total / len(windows)
