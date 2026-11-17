"""Does fund crowding actually matter? — the paired experiment that answers it.

models/fund_flow.py can see WHERE the big income ETFs have concentrated their
short calls. What it cannot say is whether that knowledge is worth anything:
selling at a strike a $5B fund already dominates might be worse (you are late,
the vol there is already crushed) or might be fine (the flow is mechanical and
persistent). This module measures it instead of asserting it.

THE DESIGN, and why it is built this way:

  PAIRED, SAME DATE. On every decision date we sell BOTH a crowded strike and an
  uncrowded one, on the same underlying, same expiry, same size. Market-wide moves
  hit both legs identically, so the DIFFERENCE isolates the crowding effect. An
  unpaired comparison would mostly measure which dates each bucket happened to
  trade on — the classic way this study goes wrong.

  A MATCHED CONTROL LEG. The uncrowded strike is the one NEAREST the crowded strike
  in moneyness, and if none is close enough the date is dropped. This is not a
  detail: pairing a crowded 6%-OTM call against whatever uncrowded strike came
  first puts the near-the-money contract — the one with the most gamma and the
  most premium — on the control side almost every time, and the crowded leg then
  loses for a reason that has nothing to do with crowding. Before this rule the
  null fixture below produced a spurious t = -2.27.

  POINT-IN-TIME SUPPLY. ``supply_at(t, trailing)`` may only use holdings published
  on or before date t. Fund books lag by a day or more; using tomorrow's book to
  pick today's strike would manufacture an effect out of nothing.

  PAIRED t-STATISTIC, AND A REFUSAL TO OVERCLAIM. The result carries n, the mean
  paired difference, its t-stat, and a verdict that says NOT SIGNIFICANT unless
  |t| clears the threshold on enough pairs. A crowding "effect" measured on 12
  trades is noise, and the report says so rather than printing a Sharpe.

The harness is the deliverable; the ANSWER depends on real fund holdings, which
have to be supplied. Its tests prove both directions: it detects an effect that
was deliberately injected, and it reports nothing when nothing is there.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from engine import pricing
from engine.iv import implied_vol

MULT = 100


def _realize_short_call(prices, t, dte, K, iv, r, q, hedge_bps, spread_frac,
                        contract_mult=MULT):
    """Delta-hedged short-CALL P&L over [t, t+dte]. One contract, entry cost paid.

    Same convention as engine/signal_backtest: a single 252-day clock, implied
    variance credited at iv^2*dte/252 against the realised variance of the same
    bars, so the position breaks even when realised annual vol == iv.
    """
    T0 = dte / 252.0
    g = pricing.greeks(prices[t], K, T0, r, q, iv, "call")
    v_prev, d_prev = g.price, g.delta
    hedge_prev = contract_mult * d_prev
    entry_cost = contract_mult * 0.5 * (spread_frac * v_prev + 0.04)
    total = -entry_cost
    for s in range(t + 1, min(t + dte + 1, len(prices))):
        rem = (t + dte - s) / 252.0
        S, S_prev = prices[s], prices[s - 1]
        if rem <= 0:
            v_now, d_now = max(S - K, 0.0), (1.0 if S > K else 0.0)
        else:
            gg = pricing.greeks(S, K, rem, r, q, iv, "call")
            v_now, d_now = gg.price, gg.delta
        total += contract_mult * (v_prev - v_now)          # short: gain if value falls
        total += hedge_prev * (S - S_prev)                 # the hedge
        hedge_now = contract_mult * d_now
        total -= abs(hedge_now - hedge_prev) * S * hedge_bps
        v_prev, hedge_prev = v_now, hedge_now
    return total


# --------------------------------------------------------------------------- #
# Synthetic fixture — a (chain_at, supply_at) pair for offline demo/tests
# --------------------------------------------------------------------------- #
def synthetic_crowding_series(*, dte: int, dent: float = 0.0, r: float = 0.03,
                              q: float = 0.0, base_premium: float = 0.12,
                              smile: float = 0.9, rotate: bool = True):
    """Return ``(chain_at, supply_at)`` where a fund crowds one OTM call strike.

    ``dent`` is how many VOL POINTS the crowded strike's IV sits BELOW the smile
    its neighbours describe — the same sign convention as ``fund_flow.iv_dent``.
    dent=0 means the fund is present but its supply moves nothing, which is the
    honest null: run the study on it and it should find nothing.

    ``rotate`` alternates WHICH moneyness is the crowded one from block to block.
    Without that, "the crowded strike" and "the nearer strike" would be the same
    variable and any result would just be measuring gamma.
    """
    import datetime

    from engine import volforecast
    from engine.data import OptionChain, OptionQuote
    from models.fund_flow import SupplyBucket

    day0 = datetime.date(2024, 1, 2)
    # Two CLOSE OTM candidates: far wings price under the quote floor in calm
    # regimes and drop out of the chain, which would silently thin the sample.
    otm = (0.03, 0.06)
    ladder = (-0.15, -0.10, -0.05, 0.0, 0.03, 0.06, 0.10, 0.15)

    def _crowded_strike(t, spot):
        i = (t // max(dte, 1)) % len(otm) if rotate else 0
        return round(spot * (1 + otm[i]), 2)

    def chain_at(t, trailing):
        if len(trailing) < 22:
            return None
        spot = trailing[-1]
        rv = volforecast.close_to_close_vol(trailing, 21)
        premium = base_premium + 0.10 * math.sin(t / 9.0)
        atm_vol = max(rv * (1.0 + premium), 0.03)
        crowded_K = _crowded_strike(t, spot)
        T = dte / 365.0
        quotes = []
        for mny in ladder:
            K = round(spot * (1 + mny), 2)
            vol = atm_vol + smile * mny * mny
            if K == crowded_K:
                vol = max(vol - dent, 0.01)          # the supply dent
            for kind in ("call", "put"):
                px = pricing.price(spot, K, T, r, q, vol, kind)
                if px < 0.01:
                    continue
                half = max(0.02, 0.02 * px)
                quotes.append(OptionQuote(dte, K, kind, round(max(px - half, 0.01), 2),
                                          round(px + half, 2)))
        return OptionChain("SYN", spot, r, q, quotes,
                           asof=(day0 + datetime.timedelta(days=t)).isoformat())

    def supply_at(t, trailing):
        if len(trailing) < 22:
            return []
        spot = trailing[-1]
        exp = (day0 + datetime.timedelta(days=t + dte)).isoformat()
        return [SupplyBucket(exp, _crowded_strike(t, spot), "call",
                             9000.0, 9000.0 * 100.0 * spot, ["QQQI", "JEPQ"]),
                SupplyBucket(exp, round(spot * 1.25, 2), "call",
                             600.0, 600.0 * 100.0 * spot, ["XYLD"])]

    return chain_at, supply_at


@dataclass
class CrowdingResult:
    n_pairs: int
    mean_crowded: float
    mean_uncrowded: float
    mean_diff: float               # crowded - uncrowded, per pair
    t_stat: float
    win_rate_diff: float           # fraction of pairs where crowded beat uncrowded
    verdict: str
    pairs: list = field(default_factory=list)
    skipped: dict = field(default_factory=dict)

    def summary(self) -> str:
        skips = "; ".join(f"{k}:{v}" for k, v in self.skipped.items()) or "none"
        return (f"Crowded vs uncrowded, PAIRED on the same dates\n"
                f"  pairs={self.n_pairs}   (skips: {skips})\n"
                f"  mean P&L  crowded {self.mean_crowded:>+12,.2f}   "
                f"uncrowded {self.mean_uncrowded:>+12,.2f}\n"
                f"  mean paired difference {self.mean_diff:>+12,.2f}   "
                f"t={self.t_stat:+.2f}   crowded wins {self.win_rate_diff:.0%} of pairs\n"
                f"  -> {self.verdict}")


def _verdict(n: int, t: float, diff: float, min_pairs: int, t_thresh: float) -> str:
    if n < min_pairs:
        return (f"NOT ENOUGH DATA — {n} pairs (need >= {min_pairs}). Any apparent "
                f"crowding effect here is noise.")
    if abs(t) < t_thresh:
        return (f"NO SIGNIFICANT EFFECT — |t|={abs(t):.2f} < {t_thresh}. On this "
                f"sample, selling a crowded strike is neither better nor worse.")
    if diff < 0:
        return (f"CROWDED IS WORSE by {abs(diff):,.2f}/trade (t={t:+.2f}) — the vol "
                f"there is already crushed; being the marginal seller costs you.")
    return (f"CROWDED IS BETTER by {diff:,.2f}/trade (t={t:+.2f}) — the mechanical "
            f"flow persists; check this survives costs and a crash before believing it.")


def run_crowding_backtest(
    prices, chain_at, supply_at, *,
    dte: int = 21, warmup: int = 63, r: float = 0.03, q: float = 0.0,
    hedge_bps: float = 5e-4, spread_frac: float = 0.015,
    crowded_min: float = 0.30, uncrowded_max: float = 0.05,
    max_moneyness_gap: float = 0.05,
    contract_mult: int = MULT, min_pairs: int = 30, t_thresh: float = 2.0,
) -> CrowdingResult:
    """Sell a CROWDED and an UNCROWDED strike side by side on each date.

    ``chain_at(t, trailing) -> OptionChain | None`` and
    ``supply_at(t, trailing) -> list[SupplyBucket]`` (point-in-time: neither may
    look past index t). A date contributes a PAIR only when a crowded strike AND an
    uncrowded one within ``max_moneyness_gap`` of it both exist at the same expiry —
    otherwise it is skipped and counted, so a thin sample cannot masquerade as a
    result. The moneyness cap is what keeps the two legs comparable; without it the
    control leg drifts to the money and the study measures gamma instead.
    """
    from models.fund_flow import crowding_score

    diffs: list[float] = []
    c_pnls: list[float] = []
    u_pnls: list[float] = []
    pairs: list = []
    skipped: dict = {}

    def _skip(k):
        skipped[k] = skipped.get(k, 0) + 1

    t = warmup
    while t + dte < len(prices):
        trailing = prices[:t + 1]
        chain = chain_at(t, trailing)
        if chain is None:
            _skip("no_chain")
            t += dte
            continue
        buckets = supply_at(t, trailing) or []
        if not buckets:
            _skip("no_supply_data")
            t += dte
            continue

        T = dte / 365.0
        cands = []
        for qt in chain.quotes:
            if qt.expiry_days != dte or qt.kind != "call" or qt.strike <= chain.spot:
                continue                                   # OTM calls only: what funds sell
            share, _ = crowding_score(qt.strike, "call", dte, chain.asof, buckets)
            iv = implied_vol(qt.mid, chain.spot, qt.strike, T, chain.r, chain.q, "call")
            if iv is None or iv <= 0:
                continue
            cands.append((qt.strike, iv, share))

        crowded = max((c for c in cands if c[2] >= crowded_min),
                      key=lambda c: c[2], default=None)
        if crowded is None:
            _skip("no_crowded_strike")
            t += dte
            continue

        # MATCH THE CONTROL LEG ON MONEYNESS. This is the part that decides whether
        # the study measures anything real. Taking whichever uncrowded strike came
        # first would keep picking the near-the-money one — the highest-gamma, most
        # profitable contract on the board — and the crowded leg would look worse
        # every time for a reason that has nothing to do with crowding. So pair with
        # the NEAREST comparable strike, and if nothing is close enough, throw the
        # date away rather than compare two different trades.
        free = [c for c in cands if c[2] <= uncrowded_max and c[0] != crowded[0]]
        uncrowded = min(free, key=lambda c: abs(math.log(c[0] / crowded[0])),
                        default=None)
        if uncrowded is None:
            _skip("no_uncrowded_strike")
            t += dte
            continue
        if abs(math.log(uncrowded[0] / crowded[0])) > max_moneyness_gap:
            _skip("no_comparable_strike")
            t += dte
            continue

        cp = _realize_short_call(prices, t, dte, crowded[0], crowded[1], r, q,
                                 hedge_bps, spread_frac, contract_mult)
        up = _realize_short_call(prices, t, dte, uncrowded[0], uncrowded[1], r, q,
                                 hedge_bps, spread_frac, contract_mult)
        c_pnls.append(cp)
        u_pnls.append(up)
        diffs.append(cp - up)
        pairs.append({"t": t, "crowded_strike": crowded[0], "crowded_share": crowded[2],
                      "uncrowded_strike": uncrowded[0], "crowded_pnl": cp,
                      "uncrowded_pnl": up, "diff": cp - up})
        t += dte

    n = len(diffs)
    if n == 0:
        return CrowdingResult(0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              _verdict(0, 0.0, 0.0, min_pairs, t_thresh),
                              pairs, skipped)
    mean_d = sum(diffs) / n
    if n > 1:
        var = sum((x - mean_d) ** 2 for x in diffs) / (n - 1)
        se = math.sqrt(var / n)
        tstat = mean_d / se if se > 1e-12 else 0.0
    else:
        tstat = 0.0
    return CrowdingResult(
        n_pairs=n,
        mean_crowded=sum(c_pnls) / n,
        mean_uncrowded=sum(u_pnls) / n,
        mean_diff=mean_d,
        t_stat=tstat,
        win_rate_diff=sum(1 for d in diffs if d > 0) / n,
        verdict=_verdict(n, tstat, mean_d, min_pairs, t_thresh),
        pairs=pairs, skipped=skipped,
    )
