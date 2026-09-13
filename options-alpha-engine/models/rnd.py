"""Risk-neutral (Q) density and moments recovered from the live option chain.

This is the Q HALF of the engine. models/baseline.py builds the PHYSICAL (P)
forecast; this module reads what the MARKET implies — the risk-neutral law under
which every discounted option price is an expectation. The tradeable premium is
the gap between them (see models/edge.py).

Three model-independent extractors, all pure stdlib:
  - model_free_implied_vol : CBOE VIX / Britten-Jones-Neuberger (2000) replication
  - bkm_moments            : Bakshi-Kapadia-Madan (2003) variance / skew / kurtosis
  - risk_neutral_pdf       : Breeden-Litzenberger f_Q(K) = e^{rT} d2C/dK2 (diagnostic)

Correctness commitments baked in after adversarial review:
  * The forward F is recovered from PUT-CALL PARITY, never assumed to be spot.
  * BKM is SPOT-anchored (R = ln(S_T/S)); Var_Q = e^{rT}V - mu^2 (mean-corrected);
    annualised vol is sqrt(Var_Q / T) so it is a like-for-like match to the
    physical log_return_vol = sqrt(Var(ln S_T)/T). (Reporting sqrt(V/T) drops the
    e^{rT} growth and the -mu^2 term and biases variance LOW.)
  * A strike-coverage guard flags sparse / truncated-wing chains, where model-free
    and BKM variance are biased low exactly in stressed regimes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from engine.data import OptionChain


@dataclass
class RiskNeutralMoments:
    vol: float          # annualised risk-neutral vol, sqrt(Var_Q / T)
    skew: float         # risk-neutral skewness of the log-return
    kurtosis: float     # EXCESS kurtosis (normal = 0)
    horizon_T: float
    n_strikes: int
    coverage_ok: bool   # False if too few strikes or wings too narrow to trust


def _slice(chain: OptionChain, dte: int):
    """strike -> (call_mid, put_mid) for one expiry; strikes sorted ascending."""
    calls, puts = {}, {}
    for qt in chain.quotes:
        if qt.expiry_days != dte:
            continue
        (calls if qt.kind == "call" else puts)[qt.strike] = qt.mid
    strikes = sorted(k for k in calls if k in puts)
    return strikes, calls, puts


def _dks(strikes):
    """Central-difference strike spacing dK_i for a discretised integral."""
    n = len(strikes)
    out = []
    for i in range(n):
        if i == 0:
            out.append(strikes[1] - strikes[0])
        elif i == n - 1:
            out.append(strikes[-1] - strikes[-2])
        else:
            out.append(0.5 * (strikes[i + 1] - strikes[i - 1]))
    return out


def _forward(chain: OptionChain, strikes, calls, puts, T: float) -> float:
    """Forward price via put-call parity, using the strike nearest the spot.

    C - P = S e^{-qT} - K e^{-rT}  =>  F = S e^{(r-q)T} = e^{rT}(C-P) + K.
    """
    k0 = min(strikes, key=lambda k: abs(k - chain.spot))
    return math.exp(chain.r * T) * (calls[k0] - puts[k0]) + k0


#: How far the listed strikes must reach, in standard deviations of the terminal
#: price (sigma * sqrt(T)). MEASURED, not chosen: on European chains — where early
#: exercise plays no part at all — the model-free integral's own truncation error
#: depends almost entirely on coverage in THESE units and barely on tenor or vol:
#:
#:     coverage    bias in the recovered vol
#:       1.0 sd    -1.16% to -2.68%
#:       1.5 sd    -0.33% to -0.80%
#:       2.0 sd    -0.07% to -0.27%
#:       2.5 sd    -0.001% to -0.10%
#:       3.0 sd    ~0
#:
#: 2.5 is the first level whose worst case (0.10 vol points) is small against the
#: 1-4 vol points of variance premium this engine is built to harvest. 2.0 would
#: admit an error up to a quarter of a one-point edge.
COVERAGE_SIGMAS = 2.5


def _atm_iv(chain: OptionChain, strikes, calls, puts, T: float) -> float | None:
    """Implied vol at the strike nearest spot. Used to SIZE the coverage test.

    Deliberately not the model-free estimate: that is the quantity whose bias is
    being tested, and it is biased LOW exactly when coverage is poor, so using it
    would shrink the required width precisely on the chains that need it widened.
    An at-the-money vol is unaffected by missing wings.
    """
    from engine.iv import implied_vol
    k0 = min(strikes, key=lambda k: abs(k - chain.spot))
    for px, kind in ((calls.get(k0), "call"), (puts.get(k0), "put")):
        if not px or px <= 0:
            continue
        try:
            v = implied_vol(px, chain.spot, k0, T, chain.r, chain.q, kind)
        except (ValueError, ZeroDivisionError, ArithmeticError):
            continue
        if v and v > 0:
            return v
    return None


def _coverage_ok(chain: OptionChain, strikes, calls=None, puts=None,
                 T: float | None = None) -> bool:
    """Do the listed strikes reach far enough for the integral to converge?

    The old rule was a FIXED +/-10% of spot, with no tenor in it at all. The width
    an integral over strikes actually needs scales with sigma*sqrt(T), so that one
    number meant 2.33 standard deviations on a 30-day 15%-vol chain and **0.41** on
    a 180-day 35%-vol one — where the recovered vol is about 2.7 points too LOW
    while this function reports True. Too low is the dangerous direction: it makes
    the market look cheap, which suppresses selling and can invite buying.

    Falls back to the old fixed rule only when an ATM vol cannot be recovered, and
    that case is strictly narrower than before rather than wider.
    """
    if len(strikes) < 5:
        return False
    fixed = strikes[0] <= 0.90 * chain.spot and strikes[-1] >= 1.10 * chain.spot
    if calls is None or puts is None or not T or T <= 0:
        return fixed
    iv = _atm_iv(chain, strikes, calls, puts, T)
    if iv is None:
        return fixed
    half = COVERAGE_SIGMAS * iv * math.sqrt(T)
    return (strikes[0] <= chain.spot * (1.0 - half)
            and strikes[-1] >= chain.spot * (1.0 + half))


def coverage_report(chain: OptionChain, T: float, dte: int | None = None) -> dict:
    """How far the strikes reach against how far they need to, in one dict.

    A bare False tells an operator nothing they can act on. This says the chain
    reached -15.4% when it needed +/-18.3%, which names the fix: a wider ladder, or
    a tenor whose wings still carry a quotable price.
    """
    dte = dte if dte is not None else round(T * 365)
    try:
        strikes, calls, puts = _slice(chain, dte)
    except (ValueError, KeyError):
        strikes, calls, puts = [], {}, {}
    out = {"dte": dte, "n_strikes": len(strikes), "ok": False, "atm_iv": None,
           "required_frac": None, "low_frac": None, "high_frac": None,
           "reason": "no strike had both a call and a put"}
    if not strikes or not chain.spot:
        return out
    out["low_frac"] = strikes[0] / chain.spot - 1.0
    out["high_frac"] = strikes[-1] / chain.spot - 1.0
    out["ok"] = _coverage_ok(chain, strikes, calls, puts, T)
    iv = _atm_iv(chain, strikes, calls, puts, T) if T and T > 0 else None
    out["atm_iv"] = iv
    if iv is None:
        out["required_frac"] = 0.10
        out["reason"] = ("no ATM implied vol could be recovered, so the fixed "
                         "+/-10% fallback was used")
        return out
    req = COVERAGE_SIGMAS * iv * math.sqrt(T)
    out["required_frac"] = req
    if out["ok"]:
        out["reason"] = "ok"
    else:
        short = []
        if strikes[0] > chain.spot * (1.0 - req):
            short.append(f"puts stop at {out['low_frac']:+.1%}")
        if strikes[-1] < chain.spot * (1.0 + req):
            short.append(f"calls stop at {out['high_frac']:+.1%}")
        if len(strikes) < 5:
            short.append(f"only {len(strikes)} paired strikes")
        out["reason"] = (", ".join(short) + f"; {COVERAGE_SIGMAS} sd at an ATM vol "
                         f"of {iv:.1%} over {dte}d needs +/-{req:.1%}")
    return out


def snap_to_listed_expiry(chain: OptionChain, dte: int):
    """Move a requested tenor onto an expiry the vendor actually lists.

    ``_slice`` matches ``expiry_days`` EXACTLY, so a requested 30d against a
    chain listing 7/14/35 yields an empty slice and every Q extractor raises. On
    a vendor that returns only the nearest few expiries — Tradier fetches three,
    and SPY/QQQ expire near-daily — the requested tenor is essentially never
    listed, so "no usable Q" was the permanent state rather than the exception.

    Returns ``(snapped_dte, note)``. ``note`` is None when the requested tenor
    was listed and a sentence describing the move when it was not. The note is
    NOT optional decoration: a silent snap from 30d to 3d would fill a ledger
    with entries at a tenor nobody chose, and every VRP in it would be measured
    against a forecast horizon it does not match. Callers must surface it and
    record the realised tenor alongside the requested one.
    """
    available = sorted({q.expiry_days for q in chain.quotes})
    if not available or dte in available:
        return dte, None
    snapped = min(available, key=lambda d: abs(d - dte))
    shown = ", ".join(str(d) for d in available[:10])
    return snapped, (f"no {dte}d expiry listed; snapped to the nearest: "
                     f"{snapped}d (available: {shown}"
                     f"{'...' if len(available) > 10 else ''})")


def model_free_implied_vol(chain: OptionChain, T: float, dte: int | None = None) -> float:
    """Model-free (VIX-style) risk-neutral volatility for one expiry.

    sigma^2 = (2 e^{rT}/T) Σ (dK_i/K_i^2) Q(K_i) - (1/T)(F/K0 - 1)^2,
    Q(K) = OTM put below K0, OTM call above K0, average at K0.
    """
    dte = dte if dte is not None else round(T * 365)
    strikes, calls, puts = _slice(chain, dte)
    if len(strikes) < 3:
        raise ValueError("need >= 3 strikes with both call and put")
    F = _forward(chain, strikes, calls, puts, T)
    below = [k for k in strikes if k <= F]
    K0 = max(below) if below else strikes[0]
    dks = _dks(strikes)
    disc = math.exp(chain.r * T)
    total = 0.0
    for k, dk in zip(strikes, dks):
        if k < K0:
            o = puts[k]
        elif k > K0:
            o = calls[k]
        else:
            o = 0.5 * (calls[k] + puts[k])
        total += (dk / (k * k)) * disc * o
    var = (2.0 / T) * total - (1.0 / T) * (F / K0 - 1.0) ** 2
    return math.sqrt(max(var, 1e-10))


def bkm_moments(chain: OptionChain, T: float, dte: int | None = None) -> RiskNeutralMoments:
    """Bakshi-Kapadia-Madan (2003) risk-neutral moments of ln(S_T / S).

    Spot-anchored: with x = ln(K/S) the quad/cubic/quartic contract weights unify
    to  2(1-x)/K^2,  (6x-3x^2)/K^2,  (12x^2-4x^3)/K^2,  and the OTM option is a
    call for K>=S, a put for K<S.
    """
    dte = dte if dte is not None else round(T * 365)
    strikes, calls, puts = _slice(chain, dte)
    if len(strikes) < 3:
        raise ValueError("need >= 3 strikes with both call and put")
    S = chain.spot
    er = math.exp(chain.r * T)
    dks = _dks(strikes)
    V = W = X = 0.0
    for k, dk in zip(strikes, dks):
        x = math.log(k / S)
        o = calls[k] if k >= S else puts[k]
        base = o * dk / (k * k)
        V += 2.0 * (1.0 - x) * base
        W += (6.0 * x - 3.0 * x * x) * base
        X += (12.0 * x * x - 4.0 * x ** 3) * base
    mu = er - 1.0 - er / 2.0 * V - er / 6.0 * W - er / 24.0 * X
    var_q = er * V - mu * mu
    if var_q <= 0:
        return RiskNeutralMoments(0.0, 0.0, 0.0, T, len(strikes),
                                  _coverage_ok(chain, strikes, calls, puts, T))
    skew = (er * W - 3.0 * mu * er * V + 2.0 * mu ** 3) / var_q ** 1.5
    kurt = (er * X - 4.0 * mu * er * W + 6.0 * mu * mu * er * V - 3.0 * mu ** 4) / var_q ** 2
    return RiskNeutralMoments(
        vol=math.sqrt(var_q / T),
        skew=skew,
        kurtosis=kurt - 3.0,
        horizon_T=T,
        n_strikes=len(strikes),
        coverage_ok=_coverage_ok(chain, strikes, calls, puts, T),
    )


def risk_neutral_pdf(chain: OptionChain, T: float, dte: int | None = None) -> Callable[[float], float]:
    """Breeden-Litzenberger diagnostic density f_Q(K) = e^{rT} d2C/dK2.

    DIAGNOSTIC ONLY: the second derivative amplifies quote noise and can go
    negative on real chains. Use bkm_moments for any signal; use this to eyeball
    the shape or check it integrates to ~1 on a clean chain.
    """
    dte = dte if dte is not None else round(T * 365)
    strikes, calls, _ = _slice(chain, dte)
    er = math.exp(chain.r * T)
    kmin, kmax = strikes[0], strikes[-1]

    def _call(k: float) -> float:
        # piecewise-linear interpolation of the call curve
        if k <= kmin:
            return calls[kmin]
        if k >= kmax:
            return calls[kmax]
        for i in range(len(strikes) - 1):
            a, b = strikes[i], strikes[i + 1]
            if a <= k <= b:
                w = (k - a) / (b - a)
                return calls[a] * (1 - w) + calls[b] * w
        return 0.0

    def f(k: float, h: float = 0.5 * (kmax - kmin) / max(len(strikes) - 1, 1)) -> float:
        d2 = (_call(k + h) - 2 * _call(k) + _call(k - h)) / (h * h)
        return max(er * d2, 0.0)

    return f


def risk_neutral_density(chain: OptionChain, T: float, dte: int | None = None):
    """Q-density in the shared interface: a lognormal matched to the model-free
    Q variance and the parity forward. Lets P and Q be priced/compared directly.
    (Skew is carried separately in RiskNeutralMoments; a single lognormal is
    symmetric in log-space by construction.)"""
    from .density import single_lognormal_riskneutral
    vol_q = model_free_implied_vol(chain, T, dte)
    return single_lognormal_riskneutral(chain.spot, T, chain.r, chain.q, vol_q)
