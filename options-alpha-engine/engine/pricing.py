"""Black-Scholes-Merton pricing and Greeks (pure standard library).

This is the deterministic, verifiable core of the engine. Everything else
(vol forecasting, mispricing signals, backtests) plugs into these functions.

Conventions
-----------
- ``S``   spot price of the underlying
- ``K``   strike
- ``T``   time to expiry in YEARS (e.g. 30 calendar days -> 30/365)
- ``r``   continuously-compounded risk-free rate (annualised, e.g. 0.05)
- ``q``   continuous dividend yield (annualised, 0.0 if none)
- ``vol`` annualised volatility (e.g. 0.20 for 20%)
- ``kind`` "call" or "put"

Greeks are returned in "raw" units:
- vega  is per 1.00 change in vol (divide by 100 for per-1-vol-point)
- theta is per YEAR (divide by 365 for per-calendar-day decay)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

SQRT_2PI = math.sqrt(2.0 * math.pi)


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQRT_2PI


def _norm_cdf(x: float) -> float:
    # Standard normal CDF via the error function (stdlib, high precision).
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _d1_d2(S: float, K: float, T: float, r: float, q: float, vol: float):
    if T <= 0 or vol <= 0 or S <= 0 or K <= 0:
        raise ValueError("S, K, T, vol must be positive")
    v = vol * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * vol * vol) * T) / v
    d2 = d1 - v
    return d1, d2


def price(S: float, K: float, T: float, r: float, q: float, vol: float, kind: str) -> float:
    """Black-Scholes-Merton fair value of a European option."""
    kind = kind.lower()
    if T <= 0:  # expired -> intrinsic value
        intrinsic = (S - K) if kind == "call" else (K - S)
        return max(intrinsic, 0.0)
    d1, d2 = _d1_d2(S, K, T, r, q, vol)
    df_r = math.exp(-r * T)
    df_q = math.exp(-q * T)
    if kind == "call":
        return S * df_q * _norm_cdf(d1) - K * df_r * _norm_cdf(d2)
    if kind == "put":
        return K * df_r * _norm_cdf(-d2) - S * df_q * _norm_cdf(-d1)
    raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")


@dataclass
class Greeks:
    price: float
    delta: float
    gamma: float
    vega: float   # per 1.00 vol
    theta: float  # per year
    rho: float    # per 1.00 rate


def greeks(S: float, K: float, T: float, r: float, q: float, vol: float, kind: str) -> Greeks:
    """Full first/second-order Greeks for a European option."""
    kind = kind.lower()
    d1, d2 = _d1_d2(S, K, T, r, q, vol)
    df_r = math.exp(-r * T)
    df_q = math.exp(-q * T)
    nd1 = _norm_pdf(d1)
    sqrtT = math.sqrt(T)

    gamma = df_q * nd1 / (S * vol * sqrtT)
    vega = S * df_q * nd1 * sqrtT

    if kind == "call":
        delta = df_q * _norm_cdf(d1)
        theta = (-(S * df_q * nd1 * vol) / (2 * sqrtT)
                 - r * K * df_r * _norm_cdf(d2)
                 + q * S * df_q * _norm_cdf(d1))
        rho = K * T * df_r * _norm_cdf(d2)
    elif kind == "put":
        delta = -df_q * _norm_cdf(-d1)
        theta = (-(S * df_q * nd1 * vol) / (2 * sqrtT)
                 + r * K * df_r * _norm_cdf(-d2)
                 - q * S * df_q * _norm_cdf(-d1))
        rho = -K * T * df_r * _norm_cdf(-d2)
    else:
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")

    return Greeks(
        price=price(S, K, T, r, q, vol, kind),
        delta=delta,
        gamma=gamma,
        vega=vega,
        theta=theta,
        rho=rho,
    )
