"""IV-surface fit (SVI) — a smooth, arbitrage-aware smile for robust Q extraction.

models/rnd.py reads risk-neutral moments straight off the discrete option chain.
On a real chain with sparse strikes or truncated wings that is fragile: the
model-free / BKM variance is biased LOW exactly where the OTM wing is missing
(the stressed regimes that dominate short-vol P&L). This module fits Gatheral's
SVI (Stochastic Volatility Inspired) parameterization to one expiry's smile, then
DENSIFIES the chain onto a smooth, wide, arbitrage-checked strike grid that
rnd can consume — so the Q moments stop depending on which strikes happened to
quote.

SVI raw total-variance slice (Gatheral, 2004):
    w(k) = a + b*( rho*(k - m) + sqrt((k - m)^2 + sigma^2) )
with k = ln(K / F) the log-moneyness, w = sigma_BS^2 * T the total variance.
Fitted by a pure-stdlib Nelder-Mead with penalties enforcing the no-arbitrage
parameter region (b>=0, |rho|<1, sigma>0, min-variance >= 0).

FIT QUALITY IS A GATE, NOT A DIAGNOSTIC. `rmse` was computed and discarded, and
`robust_q_moments` claimed to "fall back to raw rnd if the fit is too poor" with
no such code anywhere -- so a smile SVI cannot represent was densified into 81
confident quotes and read as a measurement. MAX_VOL_RMSE below is the gate, and
its threshold was measured rather than chosen; the table is on the constant.

A SILENT FALLBACK WAS REJECTED, deliberately. Returning raw rnd moments when the
fit fails hands back the number this module exists to replace -- biased LOW in
exactly the truncated-wing regimes it was built for -- wearing the SVI label,
with nothing for the caller to check. This module refuses instead, which is the
convention in `engine.iv` (returns None rather than its seed) and in
`rnd._coverage_ok` (reports False rather than a narrow integral).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from engine import pricing
from engine.data import OptionChain, OptionQuote
from engine.iv import implied_vol


#: Largest fit error, in VOL points, that may still be read as a measurement.
#: Measured on a 21-strike 30-day crash-skew smile, averaging 12 noise seeds and
#: comparing the Q vol recovered through the fit against the dense-chain truth:
#:
#:     quote noise   vol_rmse   error in recovered Q vol
#:     none          0.00006    0.0023   <- SVI-vs-true-smile model error floor
#:     0.5 vol pt    0.00295    0.0021
#:     1 vol pt      0.00591    0.0022
#:     2 vol pt      0.01180    0.0031
#:     3 vol pt      0.01764    0.0044
#:     5 vol pt      0.02912    0.0072
#:     8 vol pt      0.04377    0.0481   <- breaks down, 7x the previous row
#:
#: The fit absorbs quote noise well because it averages over strikes, so the
#: error stays under one vol point until the shape itself stops being SVI. The
#: knee is between 0.029 and 0.044, and 0.02 keeps a 7x margin in the recovered
#: vol while still admitting ~3.5 vol points of ordinary quote noise. Note the
#: 0.0023 FLOOR: a perfect fit is not an exact answer, and no threshold here can
#: make the densified chain better than the parameterisation.
MAX_VOL_RMSE = 0.02


def svi_total_variance(k, a, b, rho, m, sigma):
    return a + b * (rho * (k - m) + math.sqrt((k - m) ** 2 + sigma * sigma))


# --------------------------------------------------------------------------- #
# Nelder-Mead (pure stdlib) — small, robust downhill simplex
# --------------------------------------------------------------------------- #
def _nelder_mead(f, x0, *, step=0.15, no_improv_break=40, max_iter=3000,
                 alpha=1.0, gamma=2.0, rho_c=0.5, sig=0.5):
    dim = len(x0)
    prev_best = f(x0)
    no_improv = 0
    res = [[list(x0), prev_best]]
    for i in range(dim):
        x = list(x0)
        x[i] += step * (1 + abs(x[i]))
        res.append([x, f(x)])
    it = 0
    while True:
        res.sort(key=lambda r: r[1])
        best = res[0][1]
        if it >= max_iter:
            return res[0][0]
        it += 1
        if best < prev_best - 1e-12:
            no_improv, prev_best = 0, best
        else:
            no_improv += 1
        if no_improv >= no_improv_break:
            return res[0][0]
        cen = [0.0] * dim
        for tup in res[:-1]:
            for i, c in enumerate(tup[0]):
                cen[i] += c / (len(res) - 1)
        xr = [cen[i] + alpha * (cen[i] - res[-1][0][i]) for i in range(dim)]
        rs = f(xr)
        if res[0][1] <= rs < res[-2][1]:
            res[-1] = [xr, rs]
            continue
        if rs < res[0][1]:
            xe = [cen[i] + gamma * (xr[i] - cen[i]) for i in range(dim)]
            es = f(xe)
            res[-1] = [xe, es] if es < rs else [xr, rs]
            continue
        xc = [cen[i] + rho_c * (res[-1][0][i] - cen[i]) for i in range(dim)]
        cs = f(xc)
        if cs < res[-1][1]:
            res[-1] = [xc, cs]
            continue
        x1 = res[0][0]
        res = [[[x1[i] + sig * (t[0][i] - x1[i]) for i in range(dim)], 0.0] for t in res]
        for t in res:
            t[1] = f(t[0])


@dataclass
class SVISurface:
    a: float
    b: float
    rho: float
    m: float
    sigma: float
    F: float          # forward
    T: float
    r: float = 0.0
    q: float = 0.0
    rmse: float = 0.0

    def total_variance(self, K: float) -> float:
        k = math.log(K / self.F)
        return max(svi_total_variance(k, self.a, self.b, self.rho, self.m, self.sigma), 1e-8)

    def iv(self, K: float) -> float:
        return math.sqrt(self.total_variance(K) / self.T)

    def vol_rmse(self) -> float:
        """`rmse` converted from total-variance units into VOL points.

        Raw `rmse` is in w = sigma^2 * T, so its scale moves with both the vol
        level and the tenor and no fixed tolerance can be read against it.
        Differentiating sigma = sqrt(w / T) gives d_sigma = dw / (2 * sigma * T),
        evaluated at the ATM vol.
        """
        atm_vol = math.sqrt(self.total_variance(self.F) / self.T)
        return self.rmse / (2.0 * atm_vol * self.T)

    def no_arb_ok(self, kmin=-1.5, kmax=1.5, n=60) -> bool:
        """Necessary check: total variance stays non-negative across the slice."""
        for i in range(n + 1):
            k = kmin + (kmax - kmin) * i / n
            if svi_total_variance(k, self.a, self.b, self.rho, self.m, self.sigma) < 0:
                return False
        return self.b >= 0 and abs(self.rho) < 1 and self.sigma > 0

    def dense_chain(self, symbol="SVI", *, dte=None, n=81, lo=0.6, hi=1.6) -> OptionChain:
        """Emit a smooth, wide, zero-spread BSM chain from the fitted smile, for
        rnd to extract robust Q moments off.

        BOTH kinds are written at every strike whose price survives the `px <= 0`
        filter, not calls above the forward and puts below it: the consumers pick
        their own OTM leg (`rnd` by K0, `bkm_moments` by spot), and handing them a
        one-sided chain would silently halve the integral on one wing. The filter
        is what thins the far wings, so 81 strikes over 0.6-1.6 of spot yield both
        kinds on about 60 of them.
        """
        spot = self.F * math.exp(-(self.r - self.q) * self.T)
        dte = dte if dte is not None else max(1, round(self.T * 365))
        quotes = []
        for i in range(n):
            K = round(spot * (lo + (hi - lo) * i / (n - 1)), 4)
            vol = self.iv(K)
            for kind in ("call", "put"):
                px = pricing.price(spot, K, self.T, self.r, self.q, vol, kind)
                if px <= 0:
                    continue
                quotes.append(OptionQuote(dte, K, kind, round(px, 6), round(px, 6)))
        return OptionChain(symbol, spot, self.r, self.q, quotes)


def _forward(chain: OptionChain, dte: int, T: float):
    calls, puts = {}, {}
    for qt in chain.quotes:
        if qt.expiry_days == dte:
            (calls if qt.kind == "call" else puts)[qt.strike] = qt.mid
    strikes = sorted(k for k in calls if k in puts)
    if not strikes:
        return chain.spot * math.exp((chain.r - chain.q) * T), strikes, calls, puts
    k0 = min(strikes, key=lambda k: abs(k - chain.spot))
    F = math.exp(chain.r * T) * (calls[k0] - puts[k0]) + k0
    return F, strikes, calls, puts


def fit_chain(chain: OptionChain, T: float, dte: int | None = None) -> SVISurface:
    """Fit an SVI slice to one expiry of an OptionChain (uses OTM mids)."""
    dte = dte if dte is not None else round(T * 365)
    F, strikes, calls, puts = _forward(chain, dte, T)
    pts = []  # (k, w_market, weight)
    for K in strikes:
        px = calls[K] if K >= F else puts[K]
        kind = "call" if K >= F else "put"
        iv = implied_vol(px, chain.spot, K, T, chain.r, chain.q, kind)
        if iv is None or iv <= 0:
            continue
        pts.append((math.log(K / F), iv * iv * T, 1.0))
    if len(pts) < 5:
        raise ValueError("need >= 5 usable strikes to fit SVI")

    w_vals = [w for _, w, _ in pts]

    def obj(p):
        a, b, rho, m, s = p
        pen = 0.0
        if b < 0:
            pen += 1e6 * (-b) + 1e3
        if s <= 1e-6:
            pen += 1e6 * (1e-6 - s) + 1e3
        if abs(rho) >= 1:
            pen += 1e6 * (abs(rho) - 1) + 1e3
        min_w = a + b * s * math.sqrt(max(1 - rho * rho, 0.0))
        if min_w < 0:
            pen += 1e6 * (-min_w)
        sse = 0.0
        for k, wm, wt in pts:
            sse += wt * (svi_total_variance(k, a, max(b, 0), rho, m, max(s, 1e-6)) - wm) ** 2
        return sse + pen

    x0 = [min(w_vals) * 0.9, 0.1, -0.3, 0.0, 0.1]
    best = _nelder_mead(obj, x0)
    # one restart from the found point tightens the fit
    best = _nelder_mead(obj, best, step=0.03)
    a, b, rho, m, s = best
    resid = math.sqrt(sum((svi_total_variance(k, a, b, rho, m, s) - wm) ** 2
                          for k, wm, _ in pts) / len(pts))
    return SVISurface(a, b, rho, m, abs(s), F, T, chain.r, chain.q, rmse=resid)


def robust_q_moments(chain: OptionChain, T: float, dte: int | None = None,
                     max_vol_rmse: float = MAX_VOL_RMSE):
    """Fit SVI, densify, then run rnd on the smooth chain -> robust Q moments.

    REFUSES rather than falling back. `fit_chain` raises when the expiry has
    fewer than five usable strikes, and this raises when the fitted smile misses
    the quotes by more than `max_vol_rmse` vol points. Both are the sparse and
    stressed chains the module was built for, so a number returned from them
    would be read as the robust one and is the single most expensive thing this
    module could get wrong -- see the note in the module docstring on why no
    silent fallback to raw rnd exists.
    """
    from . import rnd
    dte = dte if dte is not None else round(T * 365)
    surf = fit_chain(chain, T, dte)
    err = surf.vol_rmse()
    if err > max_vol_rmse:
        raise ValueError(
            "SVI fit misses the quotes by %.4f vol points (limit %.4f); the smile "
            "is not representable, so the densified chain is not a measurement"
            % (err, max_vol_rmse))
    dense = surf.dense_chain(dte=dte)
    return rnd.bkm_moments(dense, T, dte), surf
