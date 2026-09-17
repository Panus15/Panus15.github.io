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


def _binomial(S: float, K: float, T: float, r: float, q: float, vol: float,
              kind: str, steps: int, american: bool) -> float:
    """Cox-Ross-Rubinstein binomial value. ``american`` toggles the early-exercise
    test at every node; with it off this is a discrete European (used so the
    early-exercise premium isolates cleanly — the discretization error cancels)."""
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

    # Backward induction, with the early-exercise test when american.
    for i in range(steps - 1, -1, -1):
        for j in range(i + 1):
            cont = pu * val[j + 1] + pd * val[j]
            if american:
                s = spot(i, j)
                intrinsic = (s - K) if kind == "call" else (K - s)
                val[j] = cont if cont >= intrinsic else intrinsic
            else:
                val[j] = cont
    return val[0]


def american_price(S: float, K: float, T: float, r: float, q: float, vol: float,
                   kind: str, *, steps: int = 128) -> float:
    """Cox-Ross-Rubinstein binomial value of an AMERICAN option (early exercise
    tested at every node). Converges to Black-Scholes as ``steps`` grows for
    cases where early exercise is never optimal (e.g. a call with no dividend)."""
    return _binomial(S, K, T, r, q, vol, kind, steps, american=True)


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


def implied_carry(chain: OptionChain, dte: int, T: float, *, r: float | None = None):
    """Recover the market's implied dividend/borrow so the binomial reproduces the
    chain's own implied forward, instead of trusting a static ``q``.

    From put-call parity at the strike nearest spot, F = K + e^{rT}(C - P); then
    q_eff = r - ln(F/S)/T so the tree's forward S·e^{(r-q_eff)T} equals F. For an
    ETF like QQQ/SPY this folds the real dividend yield AND the hard-to-borrow rate
    into one implied number — both of which move the American price the raw ``q``
    would miss. (Parity is exact for European options and slightly biased for
    American, since the put's early-exercise premium ≠ the call's; near ATM that
    bias is second-order and far smaller than using a wrong static q.) Returns
    ``(r, q_eff)`` or ``None`` when there is no ATM call/put pair to anchor it."""
    r = chain.r if r is None else r
    calls, puts = {}, {}
    for qt in chain.quotes:
        if qt.expiry_days != dte:
            continue
        (calls if qt.kind == "call" else puts)[qt.strike] = qt.mid
    common = [k for k in calls if k in puts]
    if not common or T <= 0 or chain.spot <= 0:
        return None
    kstar = min(common, key=lambda k: abs(k - chain.spot))
    fwd = kstar + math.exp(r * T) * (calls[kstar] - puts[kstar])
    if fwd <= 0:
        return None
    return r, r - math.log(fwd / chain.spot) / T


def de_americanize_chain(chain: OptionChain, *, steps: int = 128,
                         use_implied_forward: bool = True) -> OptionChain:
    """Map an AMERICAN OptionChain to its European-equivalent (drop-in for rnd).

    The MID of each quote is de-Americanized (models.rnd integrates mids), and the
    original half-spread is reflowed around the European mid. De-Americanizing the
    mid — not bid and ask independently — is deliberate: a routine zero-bid OTM
    wing has an unsolvable bid but a perfectly solvable mid, so per-side handling
    would drop the whole strike (and rnd's call/put strike intersection would then
    drop the paired side too), silently truncating the wings and biasing the
    recovered variance LOW. Only a quote whose MID is below intrinsic (a genuinely
    broken quote) is dropped.

    ``use_implied_forward`` (default) recovers the dividend/borrow per expiry from
    put-call parity (see ``implied_carry``) so QQQ/SPY-style chains are priced
    against the market's own forward; set False to trust the chain's static ``q``.
    ``r``/``q``/``asof``/``spot`` metadata pass through unchanged — the European
    prices carry the implied forward, which rnd recovers from parity anyway."""
    carry = {}
    for dte in {qt.expiry_days for qt in chain.quotes}:
        T = dte / 365.0
        c = implied_carry(chain, dte, T) if use_implied_forward else None
        carry[dte] = c if c is not None else (chain.r, chain.q)

    out = []
    for qt in chain.quotes:
        T = qt.expiry_days / 365.0
        r_used, q_used = carry[qt.expiry_days]
        de = de_americanize_price(qt.mid, chain.spot, qt.strike, T, r_used,
                                  q_used, qt.kind, steps=steps)
        if de is None:
            continue
        eur_mid = de[0]
        half = max(0.0, 0.5 * (qt.ask - qt.bid))
        out.append(OptionQuote(qt.expiry_days, qt.strike, qt.kind,
                               round(max(eur_mid - half, 0.0), 4),
                               round(eur_mid + half, 4)))
    return OptionChain(chain.symbol, chain.spot, chain.r, chain.q, out,
                       asof=chain.asof)


def early_exercise_premium(S: float, K: float, T: float, r: float, q: float,
                           vol: float, kind: str, *, steps: int = 128) -> float:
    """American minus European value at the same vol — the early-exercise premium,
    guaranteed >= 0. Both legs use the SAME binomial tree so the discretization
    error cancels exactly (subtracting an exact BSM price from a coarse tree price
    would instead leak discretization noise, which can go negative)."""
    am = _binomial(S, K, T, r, q, vol, kind, steps, american=True)
    eu = _binomial(S, K, T, r, q, vol, kind, steps, american=False)
    return am - eu
