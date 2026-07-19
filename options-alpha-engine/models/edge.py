"""Distributional edge — compare the physical (P) forecast to the market (Q).

This generalises the scalar scanner (engine/signal.py: one forecast vol vs one
market IV) into the full-distribution comparison the product is built around:
per expiry, forecast the P-density, read the Q-moments off the chain, and measure
the two premia that pay:

  variance risk premium  VRP = q_vol^2 - p_vol^2   (>0: market implies more
                                                    variance than we forecast)
  skew risk premium      SRP = q_skew - p_skew      (DIAGNOSTIC in Phase-1)

Honesty commitments baked in after adversarial review — the difference between a
signal and a trap:

  1. VRP is POSITIVE in equilibrium (implied > realised most days). Selling
     whenever VRP>0 just harvests the non-arbitrageable premium as carry, and
     mislabels it "mispricing". So when a VRP history is supplied we trade its
     Z-SCORE (unusually rich vs its own past), not its level.
  2. HAR-RV lags vol spikes, so a naive short-vol signal is loudest right into a
     developing crash. A REGIME GATE suppresses SELL when realised vol is
     accelerating.
  3. SRP is diagnostic-only: the baseline's physical skew, though now data-driven,
     is still a coarse prior — do not allocate capital on skew alone yet.
  4. Edge is charged the ROUND-TRIP option spread + commission + an estimated
     delta-hedge cost, converted to vol points via vega. An edge that dies
     crossing costs is dropped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from engine import pricing, volforecast
from engine.data import OptionChain

from . import rnd
from .baseline import BaselineDensityForecaster
from .density import MixtureLogNormal


@dataclass
class DistributionalSignal:
    expiry_days: int
    p_vol: float
    q_vol: float
    p_skew: float
    q_skew: float
    variance_risk_premium: float   # q_vol^2 - p_vol^2
    skew_risk_premium: float       # q_skew - p_skew (diagnostic)
    vol_premium: float             # q_vol - p_vol, in vol points
    vrp_z: float | None            # z-score vs supplied history, or None
    verdict: str                   # RICH / CHEAP / FAIR / NO-TRADE
    basis: str                     # 'vrp_zscore' or 'vrp_level(carry)'
    regime_stressed: bool
    coverage_ok: bool
    edge_net: float                # dollars/contract, net of round-trip cost
    note: str = ""


def regime_stressed(prices, short: int = 5, long: int = 21, thresh: float = 1.3) -> bool:
    """True when short-window realised vol is running hot vs the longer window.

    HAR-RV is backward-looking, so this guard stops the engine from selling
    'cheap-looking' vol into an accelerating move (the short-vol blow-up trap).
    """
    try:
        s = volforecast.close_to_close_vol(prices, short)
        l = volforecast.close_to_close_vol(prices, long)
    except ValueError:
        return False
    return l > 0 and (s / l) > thresh


def compare(
    p_forecast: MixtureLogNormal,
    q: rnd.RiskNeutralMoments,
    *,
    spot: float,
    T: float,
    expiry_days: int,
    vrp_history: list[float] | None = None,
    stressed: bool = False,
    edge_net: float = 0.0,
    z_thresh: float = 1.0,
    min_vrp: float = 1e-4,
) -> DistributionalSignal:
    """Compare one P-forecast against one expiry's Q-moments -> a signal."""
    p_vol = p_forecast.log_return_vol(spot, T)
    p_skew = p_forecast.log_return_skew()
    vrp = q.vol ** 2 - p_vol ** 2
    srp = q.skew - p_skew

    z = None
    if vrp_history and len(vrp_history) >= 20:
        m = sum(vrp_history) / len(vrp_history)
        var = sum((x - m) ** 2 for x in vrp_history) / (len(vrp_history) - 1)
        sd = math.sqrt(var)
        # A near-constant history has no usable dispersion — z-scoring it yields
        # floating-point-dust denominators and absurd scores; fall back to level.
        z = (vrp - m) / sd if sd > 1e-6 else None

    note = ""
    if not q.coverage_ok:
        verdict, basis = "NO-TRADE", "coverage"
        note = "sparse/truncated chain — Q moments unreliable"
    elif stressed and vrp > 0:
        verdict, basis = "NO-TRADE", "regime"
        note = "realised vol accelerating; short-vol suppressed"
    elif z is not None:
        basis = "vrp_zscore"
        verdict = "RICH" if z >= z_thresh else "CHEAP" if z <= -z_thresh else "FAIR"
    else:
        basis = "vrp_level(carry)"
        verdict = "RICH" if vrp > min_vrp else "CHEAP" if vrp < -min_vrp else "FAIR"

    if verdict in ("RICH", "CHEAP") and edge_net <= 0:
        verdict, note = "FAIR", "edge does not survive round-trip cost"

    return DistributionalSignal(
        expiry_days=expiry_days,
        p_vol=p_vol, q_vol=q.vol, p_skew=p_skew, q_skew=q.skew,
        variance_risk_premium=vrp, skew_risk_premium=srp,
        vol_premium=q.vol - p_vol, vrp_z=z,
        verdict=verdict, basis=basis, regime_stressed=stressed,
        coverage_ok=q.coverage_ok, edge_net=edge_net, note=note,
    )


def _atm_cost_and_vega(chain: OptionChain, dte: int, T: float, p_vol: float):
    """Round-trip cost (dollars/contract) and vega (per 1.00 vol) at the ATM strike.

    Converts a vol-point edge into dollars honestly: you pay the FULL option
    spread over a round trip, two commissions, and a delta-hedge/roll estimate.
    """
    atm = None
    best = 1e18
    for qt in chain.quotes:
        if qt.expiry_days == dte and abs(qt.strike - chain.spot) < best:
            best, atm = abs(qt.strike - chain.spot), qt
    if atm is None:
        return None, None
    g = pricing.greeks(chain.spot, atm.strike, T, chain.r, chain.q, p_vol, atm.kind)
    mult = 100
    round_trip_spread = atm.spread * mult            # full spread, entry+exit
    commissions = 2 * 0.65
    hedge_est = 0.20 * atm.spread * mult             # daily delta-hedge/roll proxy
    cost = round_trip_spread + commissions + hedge_est
    vega_contract = g.vega * mult
    return cost, vega_contract


def scan_distribution(
    chain: OptionChain,
    forecaster: BaselineDensityForecaster,
    prices,
    *,
    vrp_history_by_dte: dict | None = None,
    min_edge: float = 0.0,
    actionable_only: bool = True,
) -> list[DistributionalSignal]:
    """One distributional signal per expiry, best (largest |VRP|) first.

    actionable_only=True keeps only cost-surviving RICH/CHEAP signals (falling
    back to the full board if none qualify); False returns every expiry so you
    can see the whole P-vs-Q comparison, FAIR/NO-TRADE included.
    """
    stressed = regime_stressed(prices)
    dtes = sorted({qt.expiry_days for qt in chain.quotes})
    out: list[DistributionalSignal] = []
    for dte in dtes:
        T = dte / 365.0
        try:
            q_vol = rnd.model_free_implied_vol(chain, T, dte)      # robust Q vol
            q_full = rnd.bkm_moments(chain, T, dte)                # skew / kurt / coverage
        except ValueError:
            continue
        q = rnd.RiskNeutralMoments(
            vol=q_vol, skew=q_full.skew, kurtosis=q_full.kurtosis,
            horizon_T=T, n_strikes=q_full.n_strikes, coverage_ok=q_full.coverage_ok,
        )
        p = forecaster.forecast(prices, T, r=chain.r, q=chain.q, spot=chain.spot)

        cost, vega = _atm_cost_and_vega(chain, dte, T, p.log_return_vol(chain.spot, T))
        edge_net = 0.0
        if cost is not None and vega is not None:
            gross = abs(q.vol - p.log_return_vol(chain.spot, T)) * vega  # $ from vol edge
            edge_net = gross - cost

        hist = (vrp_history_by_dte or {}).get(dte)
        out.append(compare(
            p, q, spot=chain.spot, T=T, expiry_days=dte,
            vrp_history=hist, stressed=stressed, edge_net=edge_net,
        ))

    out.sort(key=lambda s: abs(s.variance_risk_premium), reverse=True)
    if not actionable_only:
        return out
    actionable = [s for s in out if s.verdict in ("RICH", "CHEAP") and s.edge_net >= min_edge]
    return actionable if actionable else out
