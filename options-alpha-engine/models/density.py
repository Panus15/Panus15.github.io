"""Distributional forecast — the OUTPUT TYPE of the core ML model.

The engine's core ML technique is *probabilistic (distributional) forecasting*:
predict the full probability distribution of the underlying's terminal price
S_T, not a point estimate and not a direction. This module defines that output
as a **mixture of log-normals**, which:

  - keeps S_T strictly positive (a price cannot go negative),
  - has closed-form mean / variance / CDF / European-payoff expectations,
  - represents skew and fat tails via multiple components (e.g. calm + stress),
  - is exactly the object a Mixture Density Network (MDN) head emits.

The whole point of pinning the interface here: the Phase-1 baseline
(models/baseline.py) fills these parameters from a HAR-RV vol forecast, while
the production model swaps that for a TFT-encoder + MDN head that emits the
SAME `MixtureLogNormal`. Nothing downstream (pricing, edge, backtest) changes.

Measure note: a `MixtureLogNormal` is just a distribution. Whether it is a
*physical* (P) forecast or a *risk-neutral* (Q) object depends on who built it.
`models/baseline.py` builds a P-forecast; `models/rnd.py` recovers Q-moments
from the market. The edge lives in the gap between them — see models/edge.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

SQRT2 = math.sqrt(2.0)
SQRT2PI = math.sqrt(2.0 * math.pi)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / SQRT2))


@dataclass
class LogNormalComponent:
    """One log-normal piece: ln(S_T) ~ Normal(mu, sigma)."""
    weight: float
    mu: float      # mean of ln(S_T)
    sigma: float   # std  of ln(S_T), > 0

    def __post_init__(self) -> None:
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")
        if self.weight < 0:
            raise ValueError("weight must be non-negative")


@dataclass
class MixtureLogNormal:
    """A mixture of log-normals over the terminal price S_T.

    This is the canonical forecast object of the engine. Build it from a model
    (baseline or MDN), then price any European option against it or compare its
    moments to the market's risk-neutral moments.
    """
    components: list[LogNormalComponent]

    def __post_init__(self) -> None:
        total = sum(c.weight for c in self.components)
        if total <= 0:
            raise ValueError("weights must sum to a positive number")
        # Normalise so weights form a proper probability mixture.
        self.components = [
            LogNormalComponent(c.weight / total, c.mu, c.sigma) for c in self.components
        ]

    # --- density / distribution -------------------------------------------
    def pdf(self, x: float) -> float:
        if x <= 0:
            return 0.0
        out = 0.0
        for c in self.components:
            z = (math.log(x) - c.mu) / c.sigma
            out += c.weight * math.exp(-0.5 * z * z) / (x * c.sigma * SQRT2PI)
        return out

    def cdf(self, x: float) -> float:
        if x <= 0:
            return 0.0
        return sum(c.weight * _norm_cdf((math.log(x) - c.mu) / c.sigma) for c in self.components)

    def prob_above(self, level: float) -> float:
        """P(S_T > level) — probability the option ends in/out of the money."""
        return 1.0 - self.cdf(level)

    def quantile(self, p: float, *, lo: float = 1e-6, hi: float = 1e9) -> float:
        """Inverse CDF via bisection (robust for any mixture)."""
        if not 0.0 < p < 1.0:
            raise ValueError("p must be in (0, 1)")
        a, b = lo, hi
        for _ in range(200):
            m = 0.5 * (a + b)
            if self.cdf(m) < p:
                a = m
            else:
                b = m
        return 0.5 * (a + b)

    # --- moments of S_T ----------------------------------------------------
    def mean(self) -> float:
        return sum(c.weight * math.exp(c.mu + 0.5 * c.sigma ** 2) for c in self.components)

    def var(self) -> float:
        e_x2 = sum(c.weight * math.exp(2 * c.mu + 2 * c.sigma ** 2) for c in self.components)
        return e_x2 - self.mean() ** 2

    def std(self) -> float:
        return math.sqrt(max(self.var(), 0.0))

    # --- moments of the LOG price (returns) --------------------------------
    # ln(S_T) is a mixture of NORMALS -> closed-form central moments. These are
    # what we compare against risk-neutral (BKM) return moments in edge.py.
    def _log_moments(self):
        m = sum(c.weight * c.mu for c in self.components)
        var = sum(c.weight * (c.sigma ** 2 + (c.mu - m) ** 2) for c in self.components)
        third = sum(
            c.weight * ((c.mu - m) ** 3 + 3 * (c.mu - m) * c.sigma ** 2)
            for c in self.components
        )
        return m, var, third

    def log_return_vol(self, spot: float, T: float) -> float:
        """Annualised volatility of the log-return implied by this forecast.

        This is the single number the scalar scanner (signal.py) consumes, so a
        distributional forecast is a strict superset of a point vol forecast.
        (spot is accepted for interface symmetry; vol depends only on the spread
        of ln S_T, not on its level.)
        """
        _, var, _ = self._log_moments()
        if T <= 0:
            raise ValueError("T must be positive")
        return math.sqrt(var / T)

    def log_return_skew(self) -> float:
        """Skewness of the log-return (0 for a single log-normal; <0 = crash-skew)."""
        _, var, third = self._log_moments()
        if var <= 0:
            return 0.0
        return third / var ** 1.5

    # --- European option pricing against this distribution -----------------
    # E[(S_T - K)^+] for a log-normal component has the Black-Scholes closed
    # form; the mixture is the weighted sum. Discounting by exp(-r T) turns an
    # expectation under a RISK-NEUTRAL mixture into a price. Feed a physical
    # forecast instead and you get the model's "fair value" under P — useful for
    # ranking, but remember the P/Q gap is the risk premium, not pure alpha.
    def expected_call_payoff(self, K: float) -> float:
        out = 0.0
        for c in self.components:
            d1 = (c.mu + c.sigma ** 2 - math.log(K)) / c.sigma
            d2 = d1 - c.sigma
            out += c.weight * (math.exp(c.mu + 0.5 * c.sigma ** 2) * _norm_cdf(d1) - K * _norm_cdf(d2))
        return out

    def expected_put_payoff(self, K: float) -> float:
        out = 0.0
        for c in self.components:
            d1 = (c.mu + c.sigma ** 2 - math.log(K)) / c.sigma
            d2 = d1 - c.sigma
            out += c.weight * (K * _norm_cdf(-d2) - math.exp(c.mu + 0.5 * c.sigma ** 2) * _norm_cdf(-d1))
        return out

    def price(self, K: float, r: float, T: float, kind: str) -> float:
        """Discounted expected payoff. If this mixture is the risk-neutral law,
        this equals the arbitrage-free option price."""
        disc = math.exp(-r * T)
        if kind.lower() == "call":
            return disc * self.expected_call_payoff(K)
        if kind.lower() == "put":
            return disc * self.expected_put_payoff(K)
        raise ValueError("kind must be 'call' or 'put'")


def single_lognormal_riskneutral(S: float, T: float, r: float, q: float, vol: float) -> MixtureLogNormal:
    """The Black-Scholes risk-neutral law of S_T as a one-component mixture.

    Pricing against this MUST reproduce Black-Scholes-Merton exactly — that
    identity is the oracle test that validates the whole distribution machinery.
    """
    mu = math.log(S) + (r - q - 0.5 * vol * vol) * T
    sigma = vol * math.sqrt(T)
    return MixtureLogNormal([LogNormalComponent(1.0, mu, sigma)])
