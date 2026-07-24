"""Realised-volatility estimators and forecasters.

The single most durable edge in options is the *variance risk premium*: implied
vol tends to trade above the volatility that is actually realised. To harvest it
you need an honest forecast of future realised vol to compare against the market.

Start SIMPLE. A HAR-RV or EWMA model beats most fancy deep nets on vol, and it is
interpretable. Graduate to a Temporal Fusion Transformer only after you have
beaten these baselines out-of-sample.
"""

from __future__ import annotations

import math
from typing import Sequence

TRADING_DAYS = 252


def log_returns(prices: Sequence[float]) -> list[float]:
    return [math.log(prices[i] / prices[i - 1]) for i in range(1, len(prices))]


def close_to_close_vol(prices: Sequence[float], window: int = 21) -> float:
    """Annualised close-to-close realised volatility over the last ``window`` days."""
    rets = log_returns(prices)[-window:]
    if len(rets) < 2:
        raise ValueError("need at least 3 prices")
    mean = sum(rets) / len(rets)
    var = sum((x - mean) ** 2 for x in rets) / (len(rets) - 1)
    return math.sqrt(var * TRADING_DAYS)


def ewma_vol(prices: Sequence[float], lam: float = 0.94) -> float:
    """RiskMetrics-style EWMA volatility forecast (annualised).

    ``lam`` = 0.94 is the classic RiskMetrics daily decay. Higher = smoother.
    """
    rets = log_returns(prices)
    if not rets:
        raise ValueError("need at least 2 prices")
    var = rets[0] ** 2
    for x in rets[1:]:
        var = lam * var + (1 - lam) * x * x
    return math.sqrt(var * TRADING_DAYS)


def _rolling_rv(rets: Sequence[float], window: int) -> list[float]:
    """Rolling annualised realised variance (not vol) ending at each day."""
    out = []
    for i in range(len(rets)):
        lo = max(0, i - window + 1)
        chunk = rets[lo:i + 1]
        rv = sum(x * x for x in chunk) / len(chunk) * TRADING_DAYS
        out.append(rv)
    return out


def _ols(X: list[list[float]], y: list[float]) -> list[float]:
    """Tiny ordinary-least-squares via normal equations + Gaussian elimination.

    Pure stdlib so the baseline has no numpy dependency. For production swap in
    numpy.linalg.lstsq — this exists to keep the MVP runnable anywhere.
    """
    n_feat = len(X[0])
    # Build X'X (n_feat x n_feat) and X'y (n_feat).
    xtx = [[0.0] * n_feat for _ in range(n_feat)]
    xty = [0.0] * n_feat
    for row, target in zip(X, y):
        for a in range(n_feat):
            xty[a] += row[a] * target
            for b in range(n_feat):
                xtx[a][b] += row[a] * row[b]
    # Solve (X'X) beta = X'y with partial pivoting.
    for a in range(n_feat):
        xtx[a].append(xty[a])
    for col in range(n_feat):
        piv = max(range(col, n_feat), key=lambda r: abs(xtx[r][col]))
        xtx[col], xtx[piv] = xtx[piv], xtx[col]
        if abs(xtx[col][col]) < 1e-12:
            continue
        for r in range(n_feat):
            if r == col:
                continue
            factor = xtx[r][col] / xtx[col][col]
            for c in range(col, n_feat + 1):
                xtx[r][c] -= factor * xtx[col][c]
    return [xtx[a][n_feat] / xtx[a][a] if abs(xtx[a][a]) > 1e-12 else 0.0 for a in range(n_feat)]


def realized_skew(prices: Sequence[float], window: int = 63) -> float:
    """Skewness of recent daily log-returns (physical, backward-looking).

    Equities print persistent NEGATIVE return skew (crashes are sharper than
    rallies). Used to make the baseline density's shape data-driven instead of a
    hardcoded constant — see models/baseline.py. Returns 0.0 if under-sampled.
    """
    rets = log_returns(prices)[-window:]
    n = len(rets)
    if n < 10:
        return 0.0
    mean = sum(rets) / n
    var = sum((x - mean) ** 2 for x in rets) / n
    if var <= 0:
        return 0.0
    third = sum((x - mean) ** 3 for x in rets) / n
    return third / var ** 1.5


def har_rv_forecast(prices: Sequence[float]) -> float:
    """Heterogeneous Auto-Regressive Realised Volatility (Corsi, 2009) forecast.

    Regresses next-day realised variance on daily / weekly (5d) / monthly (22d)
    realised-variance averages. Returns an annualised vol forecast. This is the
    workhorse baseline every serious vol desk starts from.
    """
    rets = log_returns(prices)
    if len(rets) < 30:
        # Not enough history for HAR -> fall back to EWMA.
        return ewma_vol(prices)

    rv_d = _rolling_rv(rets, 1)
    rv_w = _rolling_rv(rets, 5)
    rv_m = _rolling_rv(rets, 22)

    X, y = [], []
    # Predict day t's RV from features known at t-1.
    for t in range(22, len(rets)):
        X.append([1.0, rv_d[t - 1], rv_w[t - 1], rv_m[t - 1]])
        y.append(rv_d[t])
    beta = _ols(X, y)

    last = [1.0, rv_d[-1], rv_w[-1], rv_m[-1]]
    var_hat = sum(b * f for b, f in zip(beta, last))

    # Sanity bracket — learned from REAL data (GOOG July-2008 earnings gap).
    # On short samples containing one huge outlier day, the tiny-sample OLS can
    # extrapolate a NEGATIVE (or absurd) next-day variance; the old guard
    # (max(var_hat, 1e-8)) then emitted a ~0.01% vol forecast on a 48%-vol
    # context, collapsing every downstream density to a spike. A negative or
    # out-of-bracket prediction means the regression misfit -> fall back to the
    # robust EWMA anchor instead of trusting the extrapolation.
    anchor = ewma_vol(prices) ** 2
    if not math.isfinite(var_hat) or var_hat < 0.1 * anchor or var_hat > 10.0 * anchor:
        return math.sqrt(anchor)
    return math.sqrt(var_hat)
