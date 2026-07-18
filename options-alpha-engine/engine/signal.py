"""Mispricing scanner — the heart of the MVP.

For every quote in the chain we:
  1. Recover the market implied vol from the *mid* price.
  2. Compare it to our forecast of realised vol (the "fair" vol).
  3. Flag the option RICH (IV >> RV -> sell vol) or CHEAP (IV << RV -> buy vol),
     and express the edge in dollars by re-pricing at the forecast vol.

Crucially we also report edge *net of half the bid/ask spread*, because you
never trade at mid. An edge that vanishes once you cross the spread is not an
edge — that single check kills most naive options "signals".
"""

from __future__ import annotations

from dataclasses import dataclass

from .data import OptionChain, OptionQuote
from .iv import implied_vol
from .pricing import price


@dataclass
class Mispricing:
    quote: OptionQuote
    market_iv: float
    forecast_vol: float
    fair_value: float       # price at forecast vol
    edge_mid: float         # fair_value - mid  (signed; +ve = cheap vs fair)
    edge_net: float         # edge after paying half the spread to enter
    verdict: str            # "CHEAP", "RICH", or "FAIR"

    @property
    def iv_premium(self) -> float:
        """How much implied exceeds forecast, in vol points (+ = rich)."""
        return self.market_iv - self.forecast_vol


def scan_chain(
    chain: OptionChain,
    forecast_vol: float,
    *,
    min_edge_net: float = 0.05,
    iv_band: float = 0.02,
) -> list[Mispricing]:
    """Return mispricings sorted by net edge (best opportunities first).

    Parameters
    ----------
    forecast_vol : your realised-vol forecast for the underlying (annualised).
    min_edge_net : minimum dollar edge, after crossing half the spread, to act.
    iv_band      : vol-point tolerance; within this, the option is "FAIR".
    """
    results: list[Mispricing] = []
    for q in chain.quotes:
        iv = implied_vol(q.mid, chain.spot, q.strike, q.T, chain.r, chain.q, q.kind)
        if iv is None:
            continue
        fair = price(chain.spot, q.strike, q.T, chain.r, chain.q, forecast_vol, q.kind)
        edge_mid = fair - q.mid                      # +ve: market too cheap
        # To BUY you pay the ask; to SELL you receive the bid. Either way you
        # give up ~half the spread on entry — charge it against the edge.
        edge_net = abs(edge_mid) - 0.5 * q.spread

        if iv - forecast_vol > iv_band and edge_mid < 0:
            verdict = "RICH"     # implied too high -> option overpriced -> sell
        elif forecast_vol - iv > iv_band and edge_mid > 0:
            verdict = "CHEAP"    # implied too low -> option underpriced -> buy
        else:
            verdict = "FAIR"

        results.append(Mispricing(
            quote=q,
            market_iv=iv,
            forecast_vol=forecast_vol,
            fair_value=fair,
            edge_mid=edge_mid,
            edge_net=edge_net,
            verdict=verdict,
        ))

    actionable = [m for m in results if m.verdict != "FAIR" and m.edge_net >= min_edge_net]
    actionable.sort(key=lambda m: m.edge_net, reverse=True)
    return actionable
