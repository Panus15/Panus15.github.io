"""Implied volatility solver.

Given a *market* option price, recover the volatility the market is pricing in.
This is the number your models compete against: your job is to decide whether
that implied vol is too high (option rich) or too low (option cheap).

Strategy: Newton-Raphson using vega, with a robust bisection fallback so the
solver never diverges on ugly quotes (deep ITM/OTM, wide spreads, etc.).
"""

from __future__ import annotations

import math

from .pricing import greeks, price


def _lower_bound(S: float, K: float, T: float, r: float, q: float,
                 kind: str) -> float:
    """The no-arbitrage floor for a EUROPEAN option, which is DISCOUNTED.

    `max(K - S, 0)` is the AMERICAN floor: it is what exercising today
    guarantees you. A European option cannot be exercised today, so its floor
    is the discounted one -- K*e^{-rT} - S*e^{-qT} for a put -- which at any
    positive rate sits BELOW K - S for every in-the-money strike.

    This mattered. The solver used the American floor against a European
    pricer, so it rejected every in-the-money European put trading between the
    two bounds as "below intrinsic" and returned None. On a realistic chain at
    4% rates that was 16% of the well-posed quotes -- the whole in-the-money
    put wing, on an engine whose only trade is selling puts. American options
    are handled by `engine/american.py`, not here.
    """
    fwd_s = S * math.exp(-q * T)
    fwd_k = K * math.exp(-r * T)
    return max((fwd_s - fwd_k) if kind.lower() == "call" else (fwd_k - fwd_s),
               0.0)


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
    vol_tol: float = 1e-3,
    max_iter: int = 100,
) -> float | None:
    """Return the implied volatility, or ``None`` if the price is not arbitrageable.

    ``None`` means the quote sits below intrinsic value, outside [lo, hi] vol,
    or does not IDENTIFY a vol at all. Callers should skip such quotes rather
    than trust a garbage number.

    That third case was a live defect and it is worth stating plainly, because
    it returned a confident number rather than failing. ``tol`` is a tolerance
    on PRICE. Where vega collapses -- a deep in-the-money option days from
    expiry -- a whole band of vols price the quote identically, so the price
    criterion is met by the SEED on the first iteration and the function
    returns 0.20 whatever the market is implying. Measured on a 1-day call at
    S=100, K=60: the price is bit-identical (40.003835294) from vol 0.02 to
    0.20 and vega is exactly 0.0, and the solver answered 0.20.

    So ``vol_tol`` is a tolerance on VOL: a quote is only solved when moving
    the answer by that much moves the price by more than ``tol``. Refusing
    here is not conservatism. If the market price cannot tell 8% vol from 20%,
    then neither can you, and no view about that option being rich or cheap
    exists to be had.
    """
    if T <= 0 or market_price <= 0:
        return None
    # A price below the European no-arbitrage floor has no implied vol.
    if market_price < _lower_bound(S, K, T, r, q, kind) - 1e-8:
        return None

    def f(vol: float) -> float:
        return price(S, K, T, r, q, vol, kind) - market_price

    def identified(v: float) -> bool:
        """Does this quote pin the vol down, or merely fail to contradict it?"""
        step = max(vol_tol, 1e-12)
        return abs(price(S, K, T, r, q, v + step, kind)
                   - price(S, K, T, r, q, max(v - step, 1e-12), kind)) > tol

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
            return vol if identified(vol) else None
        # Keep the bracket tight for the fallback.
        if diff > 0:
            b = vol
        else:
            a = vol
        # Vega for the Newton step, from the pricer rather than restated
        # here. It used to be a second copy of the formula, and a copy is a
        # place for the two to disagree: dropping the sqrt(T) from this one
        # left the solver still converging, via bisection, with nothing to
        # show for it. `pricing.greeks` is covered by a finite-difference
        # oracle, so there is exactly one vega in this engine and it is tested.
        vega = greeks(S, K, T, r, q, vol, kind).vega
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
            return vol if identified(vol) else None
        if diff > 0:
            b = vol
        else:
            a = vol
    return vol if identified(vol) else None
