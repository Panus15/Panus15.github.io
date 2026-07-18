"""Implied volatility solver.

Given a *market* option price, recover the volatility the market is pricing in.
This is the number your models compete against: your job is to decide whether
that implied vol is too high (option rich) or too low (option cheap).

Strategy: Newton-Raphson using vega, with a robust bisection fallback so the
solver never diverges on ugly quotes (deep ITM/OTM, wide spreads, etc.).
"""

from __future__ import annotations

import math

from .pricing import price


def _intrinsic(S: float, K: float, kind: str) -> float:
    return max((S - K) if kind.lower() == "call" else (K - S), 0.0)


def implied_vol(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    kind: str,
    *,
    lo: float = 1e-4,
    hi: float = 5.0,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float | None:
    """Return the implied volatility, or ``None`` if the price is not arbitrageable.

    ``None`` means the quote sits below intrinsic value or outside [lo, hi] vol,
    i.e. no Black-Scholes vol reproduces it. Callers should skip such quotes
    rather than trust a garbage number.
    """
    if T <= 0 or market_price <= 0:
        return None
    # A price below intrinsic value has no valid implied vol.
    if market_price < _intrinsic(S, K, kind) - 1e-8:
        return None

    def f(vol: float) -> float:
        return price(S, K, T, r, q, vol, kind) - market_price

    f_lo, f_hi = f(lo), f(hi)
    # Price outside the [lo, hi] vol range -> unsolvable within bounds.
    if f_lo * f_hi > 0:
        return None

    # Newton-Raphson from a sensible seed, guarded by the bracket.
    vol = 0.20
    a, b = lo, hi
    for _ in range(max_iter):
        diff = f(vol)
        if abs(diff) < tol:
            return vol
        # Keep the bracket tight for the fallback.
        if diff > 0:
            b = vol
        else:
            a = vol
        # Vega (unnormalised) for the Newton step.
        d1 = (math.log(S / K) + (r - q + 0.5 * vol * vol) * T) / (vol * math.sqrt(T))
        vega = S * math.exp(-q * T) * (math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi)) * math.sqrt(T)
        if vega < 1e-8:
            break  # flat gradient -> hand over to bisection
        step = diff / vega
        vol -= step
        if not (a < vol < b):  # Newton left the bracket -> bisect
            vol = 0.5 * (a + b)

    # Bisection fallback: guaranteed to converge inside the bracket.
    for _ in range(max_iter):
        vol = 0.5 * (a + b)
        diff = f(vol)
        if abs(diff) < tol:
            return vol
        if diff > 0:
            b = vol
        else:
            a = vol
    return vol
