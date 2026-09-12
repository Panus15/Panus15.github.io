"""Defined-risk spread construction — harvest the premium with a capped tail.

The engine's whole discipline is that the variance risk premium is *compensation
for a crash*, not free money. Every model we tried (MDN, GRU) confirmed it by
losing the crash left-tail. The structural answer is not a better tail forecast —
it is to stop selling NAKED vol and sell DEFINED-RISK structures instead, whose
maximum loss is capped by long wings. You give up some premium; you buy back the
un-hedgeable tail that blows short-vol books up.

This module turns the P-vs-Q signal into concrete, tradeable structures:

  iron_condor        short strangle + long wings — sell vol, capped both sides.
  put_credit_spread  short put + long put below — sell downside vol, capped.

Each is scored against the PHYSICAL (P) forecast, cost-aware:
  net_credit   received, selling shorts at the BID, buying longs at the ASK.
  max_loss     wing width − credit  (the defined, worst-case loss).
  prob_profit  P(terminal price finishes inside the break-evens), from the P CDF.
  ev           net_credit − E_P[structure's terminal loss] − commissions. Positive
               means the market pays more credit than the loss the P-density
               expects — the variance premium, net of the tail you capped.

Direction-neutral by construction: an iron condor profits on the underlying
staying in a range, not on it going up or down.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from engine.data import OptionChain


@dataclass
class SpreadLeg:
    kind: str      # 'call' | 'put'
    strike: float
    side: str      # 'short' | 'long'
    price: float   # executable price (short -> bid, long -> ask)


@dataclass
class Spread:
    name: str
    expiry_days: int
    legs: list = field(default_factory=list)
    net_credit: float = 0.0        # per share (>0 = credit received)
    max_loss: float = 0.0          # per share, defined
    max_gain: float = 0.0          # per share
    break_evens: tuple = ()
    prob_profit: float = 0.0       # P(finish in the profit zone), from the P density
    ev: float = 0.0                # model expected value per share, net of cost
    contract_mult: float = 100.0

    def line(self) -> str:
        legs = " ".join(f"{l.side[0].upper()}{l.kind[0].upper()}{l.strike:g}" for l in self.legs)
        be = "/".join(f"{b:.2f}" for b in self.break_evens)
        return (f"{self.name:16} {self.expiry_days:>3}d  {legs:28} "
                f"cr={self.net_credit * self.contract_mult:>+8.2f} "
                f"maxL={self.max_loss * self.contract_mult:>8.2f} "
                f"pP={self.prob_profit:5.1%} "
                f"EV={self.ev * self.contract_mult:>+8.2f}  be={be}")


def _nearest(chain: OptionChain, dte: int, kind: str, target: float):
    cands = [q for q in chain.quotes if q.expiry_days == dte and q.kind == kind]
    return min(cands, key=lambda q: abs(q.strike - target)) if cands else None


def _finalize(name, dte, legs, forecaster, prices, chain, r, q, commission, mult):
    """Shared scoring: credit, defined max loss/gain, break-evens, prob-profit, EV."""
    T = dte / 365.0
    p = forecaster.forecast(prices, T, r=r, q=q, spot=chain.spot)
    credit = sum((l.price if l.side == "short" else -l.price) for l in legs)
    comm = commission / 100.0 * len(legs)

    puts = sorted((l for l in legs if l.kind == "put"), key=lambda l: l.strike)
    calls = sorted((l for l in legs if l.kind == "call"), key=lambda l: l.strike)
    put_w = (puts[-1].strike - puts[0].strike) if len(puts) == 2 else 0.0
    call_w = (calls[-1].strike - calls[0].strike) if len(calls) == 2 else 0.0
    max_loss = max(put_w, call_w) - credit + comm
    max_gain = credit - comm

    # Expected terminal loss of the short spreads under P (long wing caps it):
    exp_loss = 0.0
    if len(puts) == 2:                                   # short higher put, long lower put
        exp_loss += p.expected_put_payoff(puts[1].strike) - p.expected_put_payoff(puts[0].strike)
    if len(calls) == 2:                                  # short lower call, long higher call
        exp_loss += p.expected_call_payoff(calls[0].strike) - p.expected_call_payoff(calls[1].strike)
    ev = credit - exp_loss - comm

    # Break-evens & profit zone (credit structures: profit between the break-evens).
    lo_be = puts[1].strike - credit if len(puts) == 2 else 0.0
    hi_be = calls[0].strike + credit if len(calls) == 2 else float("inf")
    if len(puts) == 2 and len(calls) == 2:
        be = (lo_be, hi_be)
        prob_profit = max(0.0, p.cdf(hi_be) - p.cdf(lo_be))
    elif len(puts) == 2:                                 # put credit spread: profit above lo_be
        be = (lo_be,)
        prob_profit = p.prob_above(lo_be)
    else:
        be = (hi_be,)
        prob_profit = p.cdf(hi_be)

    return Spread(name, dte, legs, round(credit, 4), round(max_loss, 4),
                  round(max_gain, 4), tuple(round(b, 2) for b in be),
                  prob_profit, round(ev, 4), mult)


def iron_condor(chain, forecaster, prices, *, dte, body=0.05, wing=0.05,
                r=0.03, q=0.0, commission=0.65, contract_mult=100.0):
    """Short strangle (~body OTM each side) + long wings (~wing further out).
    Returns a Spread, or None if the four strikes aren't all quoted."""
    S = chain.spot
    sp = _nearest(chain, dte, "put", S * (1 - body))
    lp = _nearest(chain, dte, "put", S * (1 - body - wing))
    sc = _nearest(chain, dte, "call", S * (1 + body))
    lc = _nearest(chain, dte, "call", S * (1 + body + wing))
    if None in (sp, lp, sc, lc) or lp.strike >= sp.strike or lc.strike <= sc.strike:
        return None
    legs = [SpreadLeg("put", sp.strike, "short", sp.bid),
            SpreadLeg("put", lp.strike, "long", lp.ask),
            SpreadLeg("call", sc.strike, "short", sc.bid),
            SpreadLeg("call", lc.strike, "long", lc.ask)]
    return _finalize("iron condor", dte, legs, forecaster, prices, chain,
                     r, q, commission, contract_mult)


def put_credit_spread(chain, forecaster, prices, *, dte, body=0.05, wing=0.05,
                      r=0.03, q=0.0, commission=0.65, contract_mult=100.0):
    """Short put (~body OTM) + long put (~wing further down). Capped-risk downside
    vol sale. Returns a Spread, or None if the two strikes aren't quoted."""
    S = chain.spot
    sp = _nearest(chain, dte, "put", S * (1 - body))
    lp = _nearest(chain, dte, "put", S * (1 - body - wing))
    if None in (sp, lp) or lp.strike >= sp.strike:
        return None
    legs = [SpreadLeg("put", sp.strike, "short", sp.bid),
            SpreadLeg("put", lp.strike, "long", lp.ask)]
    return _finalize("put credit spread", dte, legs, forecaster, prices, chain,
                     r, q, commission, contract_mult)


def scan_spreads(chain, forecaster, prices, *, dte, r=0.03, q=0.0,
                 commission=0.65, contract_mult=100.0, **geom):
    """Build the candidate defined-risk structures for one expiry, best-EV first.
    Only positive-EV structures are actionable — but all built are returned so you
    can see the whole board. Empty if the chain lacks the strikes."""
    out = []
    for ctor in (iron_condor, put_credit_spread):
        s = ctor(chain, forecaster, prices, dte=dte, r=r, q=q,
                 commission=commission, contract_mult=contract_mult, **geom)
        if s is not None:
            out.append(s)
    out.sort(key=lambda s: s.ev, reverse=True)
    return out
