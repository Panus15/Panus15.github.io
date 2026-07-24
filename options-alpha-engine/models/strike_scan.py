"""Per-strike scanner — model vs market, contract by contract.

Answers the question "WHICH call/put, at which strike and expiry, is worth
touching?" by comparing, for every quoted contract:

  P(ITM)  the model's physical probability the contract finishes in the money
          (from the P-density: prob_above(K)) vs the market-implied probability
          (N(d2) at that strike's implied vol — SVI-smoothed when the chain
          supports a fit, raw otherwise), and
  EV      the model's fair value  e^{-rT} E_P[payoff]  against the actual
          bid/ask, charged the full cost of crossing (half-spread is already in
          the ask/bid you trade at, plus commission).

Verdicts are deliberately conservative:

  BUY    fair value exceeds the ASK plus costs — the market sells it cheaper
         than the model thinks it is worth.
  WRITE  the BID exceeds fair value plus costs — you are paid more than model
         fair to take the short side. Suppressed while the regime gate is
         stressed (never sell vol into an accelerating move).
  FAIR   everything else — which, on an efficient chain, is almost everything.

Honesty notes, same as the rest of the engine:
  * The baseline P-density is DIRECTION-NEUTRAL (mean pinned to the forward), so
    BUY/WRITE edges come from the vol level and distribution SHAPE (skew/tails),
    not from a view that the underlying goes up or down.
  * WRITE edges are largely the variance risk premium — compensation for tail
    risk, not free money. Rank with this scanner, but let the distributional
    signal (edge.py: z-score, regime, coverage) decide whether to deploy.
  * A P-vs-Q gap is only as good as the P forecast; the promotion gate + paper
    ledger are the scoreboards that tell you whether to believe it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from engine.data import OptionChain
from engine.iv import implied_vol

from . import surface
from .edge import regime_stressed


def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@dataclass
class StrikeSignal:
    expiry_days: int
    strike: float
    kind: str                     # 'call' | 'put'
    bid: float
    ask: float
    mid: float
    market_iv: float | None       # smoothed (SVI) when available, else raw
    p_itm_model: float            # physical P(finish ITM)
    p_itm_market: float | None    # risk-neutral P(finish ITM) = N(+-d2)
    prob_gap: float | None        # model - market (per-contract diagnostic)
    fair_value: float             # e^{-rT} E_P[payoff]
    edge_buy: float               # fair - ask - commission   (>0: +EV to buy)
    edge_write: float             # bid - fair - commission   (>0: paid over fair)
    verdict: str                  # BUY / WRITE / FAIR
    note: str = ""
    contract_mult: float = 100.0  # shares/contract: 100 US equity/ETF, 1 crypto coin

    def line(self) -> str:
        pm = f"{self.p_itm_market:6.1%}" if self.p_itm_market is not None else "   n/a"
        iv = f"{self.market_iv:6.1%}" if self.market_iv is not None else "   n/a"
        edge = self.edge_buy if self.verdict == "BUY" else self.edge_write
        return (f"{self.expiry_days:>4}d {self.kind:>4} {self.strike:>9.2f} "
                f"iv={iv} P_itm model={self.p_itm_model:6.1%} mkt={pm} "
                f"fair={self.fair_value:>9.2f} mid={self.mid:>9.2f} "
                f"edge${edge * self.contract_mult:>+10.2f} {self.verdict:>6}")


def scan_strikes(
    chain: OptionChain,
    forecaster,
    prices,
    *,
    dtes=None,
    commission: float = 0.65,
    min_edge: float = 0.01,
    use_surface: bool = True,
    top: int | None = None,
    stressed: bool | None = None,
    contract_mult: float = 100.0,
    slippage_frac: float = 0.25,
) -> list[StrikeSignal]:
    """Scan every quoted contract; return signals sorted best-edge-first.

    ``min_edge`` is per SHARE (0.01 = $1/contract) so bid-ask dust never rates a
    verdict. ``top`` truncates the ranked list. Writes are suppressed (verdict
    forced FAIR, note set) while the regime is stressed — by default the price
    gate (edge.regime_stressed); pass ``stressed=`` explicitly to compose in the
    forward-looking news gate (models.news_signal.event_risk).
    """
    if stressed is None:
        stressed = regime_stressed(prices)
    comm_ps = commission / 100.0                    # per-share (100x multiplier)
    dtes = sorted(dtes) if dtes else sorted({q.expiry_days for q in chain.quotes})
    S, r, q_div = chain.spot, chain.r, chain.q
    out: list[StrikeSignal] = []

    for dte in dtes:
        T = dte / 365.0
        if T <= 0:
            continue
        p = forecaster.forecast(prices, T, r=r, q=q_div, spot=S)
        surf = None
        if use_surface:
            try:
                surf = surface.fit_chain(chain, T, dte)
            except (ValueError, ArithmeticError):
                surf = None

        for qt in chain.quotes:
            if qt.expiry_days != dte:
                continue
            mid = qt.mid
            iv_raw = implied_vol(mid, S, qt.strike, T, r, q_div, qt.kind)
            iv_used = None
            if surf is not None:
                try:
                    iv_used = surf.iv(qt.strike)
                except (ValueError, ArithmeticError):
                    iv_used = None
            if iv_used is None or iv_used <= 0:
                iv_used = iv_raw

            p_up = p.prob_above(qt.strike)
            p_itm_model = p_up if qt.kind == "call" else 1.0 - p_up

            p_itm_market = None
            if iv_used is not None and iv_used > 0:
                d2 = ((math.log(S / qt.strike) + (r - q_div - 0.5 * iv_used ** 2) * T)
                      / (iv_used * math.sqrt(T)))
                p_itm_market = _ncdf(d2) if qt.kind == "call" else _ncdf(-d2)

            # You do not get the touch on a wide/illiquid strike — you sweep into
            # the book. Charge the FULL spread (buy@ask, write@bid) PLUS a slippage
            # penalty proportional to the spread itself, so deep ITM/OTM strikes
            # with gaping quotes are penalised most, exactly where they should be.
            fair = p.price(qt.strike, r, T, qt.kind)
            slip = slippage_frac * max(qt.ask - qt.bid, 0.0)
            edge_buy = fair - (qt.ask + slip) - comm_ps
            edge_write = (qt.bid - slip) - fair - comm_ps

            note = ""
            if edge_buy > min_edge:
                verdict = "BUY"
            elif edge_write > min_edge:
                if stressed:
                    verdict, note = "FAIR", "write suppressed: regime stressed"
                else:
                    verdict = "WRITE"
                    note = "short side = risk-premium harvest; gate via edge.py"
            else:
                verdict = "FAIR"

            out.append(StrikeSignal(
                expiry_days=dte, strike=qt.strike, kind=qt.kind,
                bid=qt.bid, ask=qt.ask, mid=mid,
                market_iv=iv_used,
                p_itm_model=p_itm_model, p_itm_market=p_itm_market,
                prob_gap=(p_itm_model - p_itm_market) if p_itm_market is not None else None,
                fair_value=fair, edge_buy=edge_buy, edge_write=edge_write,
                verdict=verdict, note=note, contract_mult=contract_mult,
            ))

    out.sort(key=lambda s: max(s.edge_buy, s.edge_write), reverse=True)
    return out[:top] if top else out
