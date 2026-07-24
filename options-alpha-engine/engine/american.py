"""American options — binomial pricing + de-Americanization to a European chain.

Every Q-extractor in this engine (models.rnd: BKM / VIX-style model-free /
Breeden-Litzenberger) integrates EUROPEAN option prices. US single-name and ETF
options — QQQ, SPY, and the holdings inside covered-call income funds like QQQI —
are AMERICAN: their market price carries an early-exercise premium that European
replication has no term for. Feeding raw American prices into BKM biases the
recovered variance HIGH (the premium looks like extra option value = extra
implied variance), exactly the opposite of what you want.

This module removes that bias the standard (ORATS / CBOE) way:

  1. price American options by a Cox-Ross-Rubinstein binomial tree with an
     early-exercise test at every node (``american_price``),
  2. back out the American implied vol the market is pricing (``american_iv``),
  3. RE-PRICE a European option at that same vol (``de_americanize_price``): the
     European value strips the early-exercise premium while preserving the vol
     the market quoted. That European-equivalent price is what BKM needs.

``de_americanize_chain`` maps a whole American ``OptionChain`` to its
European-equivalent, a drop-in for models.rnd. Pure stdlib; the tree is the only
new numerical primitive and it is validated against Black-Scholes in the no-
early-exercise limit (American call, no dividend == European call).
"""

from __future__ import annotations

import math

from engine import pricing
from engine.data import OptionChain, OptionQuote


def american_price(S: float, K: float, T: float, r: float, q: float, vol: float,
                   kind: str, *, steps: int = 128) -> float:
    """Cox-Ross-Rubinstein binomial value of an AMERICAN option (early exercise
    tested at every node). Converges to Black-Scholes as ``steps`` grows for
    cases where early exercise is never optimal (e.g. a call with no dividend)."""
    kind = kind.lower()
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
    if T <= 0:
        return max((S - K) if kind == "call" else (K - S), 0.0)
    if S <= 0 or K <= 0 or vol <= 0 or steps < 1:
        raise ValueError("S, K, vol must be positive and steps >= 1")

    dt = T / steps
    u = math.exp(vol * math.sqrt(dt))
    disc = math.exp(-r * dt)
    p = (math.exp((r - q) * dt) - 1.0 / u) / (u - 1.0 / u)
    p = min(1.0, max(0.0, p))                    # clamp degenerate low-vol nodes
    pu, pd = disc * p, disc * (1.0 - p)

    # Asset price at layer i, node j (j up-moves) = S * u^(2j - i). Precompute u^k.
    pw = [u ** k for k in range(-steps, steps + 1)]

    def spot(i, j):
        return S * pw[(2 * j - i) + steps]

    # Terminal payoffs.
    if kind == "call":
        val = [max(spot(steps, j) - K, 0.0) for j in range(steps + 1)]
    else:
        val = [max(K - spot(steps, j), 0.0) for j in range(steps + 1)]

    # Backward induction with the early-exercise test.
    for i in range(steps - 1, -1, -1):
        for j in range(i + 1):
            cont = pu * val[j + 1] + pd * val[j]
            s = spot(i, j)
            intrinsic = (s - K) if kind == "call" else (K - s)
            val[j] = cont if cont >= intrinsic else intrinsic
    return val[0]


def american_iv(market_price: float, S: float, K: float, T: float, r: float,
                q: float, kind: str, *, steps: int = 128, lo: float = 0.01,
                hi: float = 5.0, tol: float = 1e-5, max_iter: int = 60) -> float | None:
    """Implied vol under the AMERICAN (binomial) model. Bisection — the American
    price is monotonic in vol, so this always converges inside a valid bracket.
    Returns ``None`` for a quote below intrinsic or outside [lo, hi] vol."""
    kind = kind.lower()
    if T <= 0 or market_price <= 0:
        return None
    intrinsic = max((S - K) if kind == "call" else (K - S), 0.0)
    if market_price < intrinsic - 1e-8:
        return None

    def f(vol):
        return american_price(S, K, T, r, q, vol, kind, steps=steps) - market_price

    a, b = lo, hi
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        return None
    for _ in range(max_iter):
        m = 0.5 * (a + b)
        fm = f(m)
        if abs(fm) < tol:
            return m
        if (fm > 0) == (fa > 0):
            a, fa = m, fm
        else:
            b, fb = m, fm
    return 0.5 * (a + b)


def de_americanize_price(american_market_price: float, S: float, K: float, T: float,
                         r: float, q: float, kind: str, *, steps: int = 128):
    """American market price -> (European-equivalent price, implied vol) or None.

    Solves the American implied vol, then prices a EUROPEAN option at that vol.
    The European value strips the early-exercise premium while keeping the vol the
    market quoted — the price BKM / model-free replication requires."""
    iv = american_iv(american_market_price, S, K, T, r, q, kind, steps=steps)
    if iv is None:
        return None
    return pricing.price(S, K, T, r, q, iv, kind), iv


def de_americanize_chain(chain: OptionChain, *, steps: int = 128) -> OptionChain:
    """Map an AMERICAN OptionChain to its European-equivalent (drop-in for rnd).

    Each quote's bid and ask are independently de-Americanized (repriced European
    at that side's American implied vol). Quotes whose bid or ask can't be solved
    (below intrinsic, unquoted) are dropped. ``r``/``q``/``asof``/``spot`` pass
    through unchanged. NOTE: European call, no dividend -> a no-op (there is no
    early-exercise premium to strip), which the tests assert."""
    out = []
    for qt in chain.quotes:
        T = qt.expiry_days / 365.0
        eb = de_americanize_price(qt.bid, chain.spot, qt.strike, T, chain.r,
                                  chain.q, qt.kind, steps=steps)
        ea = de_americanize_price(qt.ask, chain.spot, qt.strike, T, chain.r,
                                  chain.q, qt.kind, steps=steps)
        if eb is None or ea is None:
            continue
        out.append(OptionQuote(qt.expiry_days, qt.strike, qt.kind,
                               round(eb[0], 4), round(ea[0], 4)))
    return OptionChain(chain.symbol, chain.spot, chain.r, chain.q, out,
                       asof=chain.asof)


def early_exercise_premium(S: float, K: float, T: float, r: float, q: float,
                           vol: float, kind: str, *, steps: int = 128) -> float:
    """American value minus the European value at the same vol (>= 0). A
    diagnostic: how much of a quoted price is early-exercise optionality."""
    return american_price(S, K, T, r, q, vol, kind, steps=steps) - \
        pricing.price(S, K, T, r, q, vol, kind)
