"""Trade card — one screen that says what to do, and how much to trust it.

The rest of the engine emits tables: a VRP per expiry, a per-strike board, a
calibration report. This turns them into the single decision object a human
actually acts on — direction, levels, size, confidence — WITHOUT inventing any of
the numbers.

Every field is derived, not asserted:

  levels        entry / target / stop are QUANTILES of the physical density, not
                round numbers off a chart. A 25th-percentile stop means exactly
                "the model says a 25% chance of ending past here".
  confidence    the model's own probability of finishing beyond the target, and
                beyond the stop. Because the density is PIT-testable
                (models/calibration.py), this number can be CHECKED — unlike the
                confidence badge on a chart-reading tool, which cannot.
  expected_value  P(win)*reward - P(lose)*risk, in price units. If it is negative
                the card says so instead of dressing the trade up.
  size          contracts from CVaR (engine/portfolio), not a Kelly guess.

Two signals, kept strictly apart because their evidence is not equal:

  VOL SIDE (validated). Sell or buy variance from the P-vs-Q premium, gated by
  regime and cost. This is what the backtests, the promotion gate and the paper
  ledger in this repo actually test.

  DIRECTION (UNVALIDATED). LONG/SHORT from two components: the P-vs-Q PROBABILITY
  GAP at the money — our physical density's P(S_T > forward) minus the market's
  risk-neutral one, which is the skew risk premium expressed as a probability —
  plus a trailing momentum tilt. It is framework-consistent rather than
  vibes-based, but NOTHING in this repo has shown it makes money, and the
  engine's own promotion gate has rejected every richer model on the crash tail.
  The card labels it, and `direction_is_validated` is False. Treat it as a view
  to test, not a recommendation to trade.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from engine import volforecast
from engine.data import OptionChain

from . import rnd
from .edge import regime_stressed


def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@dataclass
class TradeCard:
    symbol: str
    asof: str | None
    spot: float
    expiry_days: int

    # --- the validated half: variance -------------------------------------
    vol_side: str                  # SELL VOL / BUY VOL / NO TRADE
    vol_reason: str
    p_vol: float
    q_vol: float
    vrp: float

    # --- the unvalidated half: direction ----------------------------------
    direction: str                 # LONG / SHORT / NEUTRAL
    direction_score: float         # >0 bullish, <0 bearish (roughly -1..+1)
    direction_reason: str
    direction_is_validated: bool = False

    # --- levels, all quantiles of the physical density --------------------
    entry: float = 0.0
    target: float = 0.0
    stop: float = 0.0
    p_target: float = 0.0          # model probability of finishing past target
    p_stop: float = 0.0            # model probability of finishing past stop
    reward: float = 0.0
    risk: float = 0.0
    expected_value: float = 0.0    # price units, under P

    # --- trust ------------------------------------------------------------
    calibration_note: str = ""
    warnings: list = field(default_factory=list)
    crowding: float = 0.0          # share of mapped fund supply at our strike
    crowding_note: str = ""

    @property
    def reward_risk(self) -> float:
        return self.reward / self.risk if self.risk > 1e-12 else float("inf")

    def render(self) -> str:
        w = 66
        bar = "=" * w
        out = [bar,
               f" {self.symbol}  {self.expiry_days}d   spot {self.spot:,.2f}"
               f"   {self.asof or ''}".rstrip(),
               bar,
               "",
               f" VOLATILITY   {self.vol_side}      [validated by this repo]",
               f"   P vol {self.p_vol:6.1%}   Q vol {self.q_vol:6.1%}   "
               f"VRP {self.vrp:+.4f}",
               f"   {self.vol_reason}",
               "",
               f" DIRECTION    {self.direction}      "
               f"[{'validated' if self.direction_is_validated else 'UNVALIDATED — a view, not a recommendation'}]",
               f"   score {self.direction_score:+.2f}   {self.direction_reason}",
               ""]
        if self.direction != "NEUTRAL":
            out += [f"   entry  {self.entry:>12,.2f}",
                    f"   target {self.target:>12,.2f}   "
                    f"model P(past target) {self.p_target:5.1%}   reward {self.reward:,.2f}",
                    f"   stop   {self.stop:>12,.2f}   "
                    f"model P(past stop)   {self.p_stop:5.1%}   risk   {self.risk:,.2f}",
                    f"   reward:risk {self.reward_risk:.2f}   "
                    f"expected value {self.expected_value:+,.2f} under P",
                    ""]
        if self.crowding_note:
            out += [f" FUND FLOW  crowding {self.crowding:.0%}   {self.crowding_note}", ""]
        if self.calibration_note:
            out += [f" TRUST  {self.calibration_note}", ""]
        for wmsg in self.warnings:
            out.append(f" !! {wmsg}")
        out.append(bar)
        return "\n".join(out)


def _momentum(prices, lookback: int = 21) -> float:
    """Trailing log-return over ``lookback`` bars, scaled to roughly [-1, 1]."""
    if len(prices) <= lookback:
        return 0.0
    raw = math.log(prices[-1] / prices[-1 - lookback])
    try:
        vol = volforecast.close_to_close_vol(prices, lookback)
    except ValueError:
        return 0.0
    scale = vol * math.sqrt(lookback / 252.0)          # ~1 sd of the move
    if scale <= 1e-9:
        return 0.0
    return max(-1.0, min(1.0, raw / (2.0 * scale)))


def build_card(chain: OptionChain, forecaster, prices, *, dte: int,
               min_vrp: float = 0.0, direction_threshold: float = 0.12,
               target_q: float = 0.75, stop_q: float = 0.25,
               momentum_weight: float = 0.5, calibration=None,
               calendar=None, fund_books=None, atm_strike: float | None = None) -> TradeCard:
    """Collapse the engine's output into one decision card for a single expiry.

    ``calibration`` is an optional models.calibration.CalibrationReport; when given,
    its verdict and vol scale are surfaced on the card (and a large bias becomes a
    warning, because it means the VRP below is overstated).

    ``calendar`` is an optional models.events.EventCalendar. If a KNOWN event
    (earnings, FDA, ...) falls inside the option's life, the vol leg is refused:
    that implied vol is EVENT premium priced against a scheduled jump, not a
    mispricing a HAR-RV forecast has spotted. This is the single most important
    guard for SINGLE-STOCK options and has no analogue in crypto.

    ``fund_books`` is an optional list of models.fund_flow.FundBook (the daily
    published holdings of covered-call ETFs). When given, the card reports whether
    the strike we would sell is one the big mechanical sellers already dominate —
    information, not a veto: whether crowded supply means avoid or follow is an
    empirical question this repo has not settled.
    """
    T = dte / 365.0
    spot = chain.spot
    p = forecaster.forecast(prices, T, r=chain.r, q=chain.q, spot=spot)
    p_vol = p.log_return_vol(spot, T)

    warnings: list[str] = []
    q_vol = float("nan")
    coverage_ok = False
    q_skew = float("nan")
    try:
        q_vol = rnd.model_free_implied_vol(chain, T, dte)
        m = rnd.bkm_moments(chain, T, dte)
        coverage_ok, q_skew = m.coverage_ok, m.skew
    except ValueError as exc:
        warnings.append(f"no risk-neutral read at {dte}d ({exc}) — vol side is blind")

    # --- volatility side (the validated one) ------------------------------
    stressed = regime_stressed(prices)
    event_fired, event_reason = (False, "")
    if calendar is not None:
        from .events import event_gate
        event_fired, event_reason = event_gate(calendar, chain.symbol, chain.asof, dte)
    vrp = (q_vol ** 2 - p_vol ** 2) if q_vol == q_vol else float("nan")
    if event_fired:
        vol_side, vol_reason = "NO TRADE", f"scheduled event: {event_reason}"
    elif q_vol != q_vol:
        vol_side, vol_reason = "NO TRADE", "no usable Q — cannot compare P to the market"
    elif not coverage_ok:
        vol_side, vol_reason = "NO TRADE", "chain too sparse/truncated for a trustworthy Q"
    elif stressed:
        vol_side, vol_reason = "NO TRADE", "regime gate: realised vol accelerating"
    elif vrp > min_vrp:
        vol_side = "SELL VOL"
        vol_reason = "market implies more variance than the forecast (harvest the premium)"
    elif vrp < -min_vrp:
        vol_side, vol_reason = "BUY VOL", "market implies LESS variance than the forecast"
    else:
        vol_side, vol_reason = "NO TRADE", "variance fairly priced within the threshold"

    # --- direction (the unvalidated one) ----------------------------------
    # Component 1: the P-vs-Q probability gap at the forward — our physical
    # P(S_T > F) minus the market's risk-neutral one. Both densities have the same
    # mean, so this gap is pure SHAPE (skew) disagreement: the skew risk premium
    # expressed as a probability. Positive = the market prices more downside than
    # we forecast.
    fwd = spot * math.exp((chain.r - chain.q) * T)
    prob_gap = 0.0
    if q_vol == q_vol and q_vol > 0:
        p_up_model = p.prob_above(fwd)
        d2 = (math.log(spot / fwd) + (chain.r - chain.q - 0.5 * q_vol ** 2) * T) / (
            q_vol * math.sqrt(T))
        p_up_market = _ncdf(d2)
        prob_gap = p_up_model - p_up_market
    mom = _momentum(prices)
    score = 4.0 * prob_gap + momentum_weight * mom          # gap is small in absolute terms
    score = max(-1.0, min(1.0, score))

    if score > direction_threshold:
        direction = "LONG"
    elif score < -direction_threshold:
        direction = "SHORT"
    else:
        direction = "NEUTRAL"
    direction_reason = (f"P-vs-Q probability gap {prob_gap:+.1%} (skew premium) + "
                        f"momentum {mom:+.2f}")

    # --- levels from the density's own quantiles --------------------------
    entry = spot
    target = stop = 0.0
    p_target = p_stop = reward = risk = ev = 0.0
    if direction == "LONG":
        target, stop = p.quantile(target_q), p.quantile(stop_q)
        p_target, p_stop = p.prob_above(target), p.cdf(stop)
    elif direction == "SHORT":
        target, stop = p.quantile(1 - target_q), p.quantile(1 - stop_q)
        p_target, p_stop = p.cdf(target), p.prob_above(stop)
    if direction != "NEUTRAL":
        reward, risk = abs(target - entry), abs(stop - entry)
        ev = p_target * reward - p_stop * risk

    # --- fund footprint: are we the marginal seller into someone else's flow? --
    crowding = 0.0
    crowding_note = ""
    if fund_books:
        from .fund_flow import crowding_score, supply_map
        k = atm_strike if atm_strike else round(spot)
        buckets = supply_map(fund_books, chain.symbol, spot)
        crowding, crowding_note = crowding_score(k, "call", dte, chain.asof, buckets)
        if crowding > 0.25 and vol_side == "SELL VOL":
            warnings.append(f"CROWDED SUPPLY: {crowding_note}")

    # --- trust ------------------------------------------------------------
    note = ""
    if calibration is not None:
        note = (f"{calibration.verdict.split(' — ')[0]}; PIT vol scale "
                f"x{calibration.vol_scale:.2f} on this underlying")
        if calibration.vol_scale > 1.15:
            warnings.append(
                f"the P vol looks ~{(calibration.vol_scale - 1) * 100:.0f}% too LOW here, "
                f"so the VRP above is OVERSTATED and every level is too tight")
        elif calibration.vol_scale < 0.87:
            warnings.append(
                f"the P vol looks ~{(1 - calibration.vol_scale) * 100:.0f}% too HIGH here, "
                f"so the VRP above is UNDERSTATED")
    if event_fired:
        warnings.append("a KNOWN event lands before expiry — the premium is priced "
                        "against a scheduled jump; selling it is a bet on that jump "
                        "being smaller than priced, not a variance-premium harvest")
    if direction != "NEUTRAL" and ev <= 0:
        warnings.append("the directional leg is NEGATIVE expected value under the "
                        "model's own density — the levels do not pay for the risk")
    if direction != "NEUTRAL":
        warnings.append("the DIRECTION block is unvalidated: no backtest, gate or "
                        "ledger in this repo has shown it profitable")

    return TradeCard(
        symbol=chain.symbol, asof=chain.asof, spot=spot, expiry_days=dte,
        vol_side=vol_side, vol_reason=vol_reason, p_vol=p_vol, q_vol=q_vol, vrp=vrp,
        direction=direction, direction_score=score, direction_reason=direction_reason,
        entry=entry, target=target, stop=stop, p_target=p_target, p_stop=p_stop,
        reward=reward, risk=risk, expected_value=ev,
        calibration_note=note, warnings=warnings,
        crowding=crowding, crowding_note=crowding_note,
    )
