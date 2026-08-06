"""One decision, with every input labelled by how much evidence stands behind it.

`trade_card.py` collapses the OPTIONS side into one screen. This goes one level
up and fuses the three things the repo can actually see about a US equity:

    the variance edge   is this expiry's implied vol rich against our forecast?
    sector rotation     is the market currently paying this sector or selling it?
    fund crowding       are the mechanical option sellers already parked here?

TWO RULES, and together they are the whole design:

    1. an input with no evidence behind it may only REDUCE size. It can never
       flip a direction, create a trade, or raise conviction.
    2. an input that has been TESTED AND FAILED does not move size at all.

Rule 1 is the familiar asymmetry: of the three inputs exactly one — the variance
edge — rests on machinery this repo has verified against oracles, and letting an
unmeasured thing manufacture position size is precisely how a research codebase
turns into a loss. Letting it subtract costs, at worst, some trades that would
have been fine.

Rule 2 is the one that is easy to skip, and it now applies to sector rotation.
The RRG quadrant used to cut size in Lagging and Weakening. Then it was measured
on real sector history and the quadrant predicted nothing (t=+0.71 on a test that
charges no costs at all — PREREGISTRATION.md §7.1). Keeping the cut would not be
prudence. Under rule 1 an untested input is treated as a risk we cannot see; a
DISPROVEN one is different in kind — it spends real position size on noise, and
it leaves the printed decision looking like it reasons about the sector when the
number underneath is empty. The quadrant is still shown, as context for a human.
It is no longer arithmetic.

The crowding signal has never been tested at all, because the archive that would
feed it is days old, so it stays under rule 1.

So `Decision.size_multiplier` starts at 1.0 and only ever goes down, and every
reduction names the input that caused it. If all three agree, you get the same
size the variance edge alone would have justified — agreement buys confidence to
*hold*, not licence to press.

WHAT THIS IS NOT. It is not a signal that says "buy XLK". The direction half of
the trade card remains unvalidated and is carried through as such. What this
produces is a sized, gated, provenance-tagged VOLATILITY decision with the
context a human needs to overrule it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: How much evidence stands behind an input. Printed next to every one of them,
#: because a reader cannot weigh a signal without knowing what backs it.
VALIDATED = "validated"        # verified machinery; still unproven on live markets
FIXTURE_ONLY = "fixture-only"  # tested, but only against data we generated
NO_DATA = "no data"            # wired up and empty — absence of evidence, not of risk
TESTED_NO_EDGE = "tested: no edge"   # run on real data and it did not predict

_RANK = {VALIDATED: 0, FIXTURE_ONLY: 1, NO_DATA: 2, TESTED_NO_EDGE: 3}


@dataclass
class Input:
    """One contributing signal and its provenance."""
    name: str
    reading: str
    status: str
    effect: float = 1.0        # multiplier applied to size; <= 1.0 always
    note: str = ""

    def line(self) -> str:
        eff = "" if self.effect >= 1.0 else f"  size x{self.effect:.2f}"
        return (f"  {self.name:16} {self.reading:34} [{self.status}]{eff}"
                + (f"\n      {self.note}" if self.note else ""))


@dataclass
class Decision:
    symbol: str
    asof: str | None
    expiry_days: int
    action: str                          # SELL VOL / BUY VOL / NO TRADE
    reason: str
    size_multiplier: float = 1.0
    inputs: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    @property
    def weakest_evidence(self) -> str:
        """The least-supported input that is actually INFLUENCING the decision.

        Only inputs that changed the size count. An input sitting at 1.0 has not
        moved anything, so letting it set this field would report a decision as
        weakly-evidenced when it in fact rests entirely on the variance edge —
        the reverse of the mistake this module exists to prevent, and just as
        misleading. With nothing acting, the answer is whatever backs the edge.
        """
        acting = [i for i in self.inputs if i.effect < 1.0]
        if not acting:
            edge = [i for i in self.inputs if i.name == "variance edge"]
            return edge[0].status if edge else NO_DATA
        return max((i.status for i in acting), key=lambda s: _RANK.get(s, 9))

    def render(self) -> str:
        bar = "=" * 66
        out = [bar,
               f" {self.symbol}  {self.expiry_days}d"
               + (f"   {self.asof}" if self.asof else ""),
               bar,
               f" {self.action}   size x{self.size_multiplier:.2f}",
               f"   {self.reason}",
               "",
               " INPUTS"]
        out += [i.line() for i in self.inputs]
        # keyed on what actually cut, not on the multiplier: a NO TRADE zeroes the
        # size through a different path, and the line then named nobody at all
        cut = ", ".join(i.name for i in self.inputs if i.effect < 1.0)
        if cut:
            out += ["", f" size reduced by: {cut}"]
        out += ["", f" weakest evidence acting on this decision: "
                    f"{self.weakest_evidence}"]
        for w in self.warnings:
            out.append(f" !! {w}")
        out.append(bar)
        return "\n".join(out)


def _rotation_input(rotation_point, *, lagging_cut: float, weakening_cut: float):
    """Sector rotation: REPORTED, never acted on. It was tested and it failed.

    This used to cut size in Lagging and Weakening on the reasoning that realised
    vol runs hot where the market is dumping. That reasoning was never measured;
    it was a story about a chart. It has now been measured on real SPDR sector
    history (PREREGISTRATION.md §7.1) and the quadrant carries no forward
    information: Leading beat Lagging by +0.25% per period at t=+0.71, on the
    kindest test available — the panel charges no costs at all.

    So the quadrant no longer moves size. Cutting on a disproven signal is not
    harmless caution: it spends real position size on noise, and worse, it makes
    the printed decision look like it is reasoning about the sector when the
    number it is reasoning from has been shown to be empty. A repo that runs a
    test, gets NO, and leaves the signal wired into sizing has not run the test.

    The reading is still printed, because knowing which quadrant a name sits in
    is worth seeing even when it predicts nothing — it is context for a human,
    not an input to the arithmetic. ``lagging_cut`` and ``weakening_cut`` are
    accepted and deliberately ignored; they stay in the signature so a caller
    that still passes them is not silently broken, and they become live again
    only if some future sample produces a positive registered result.
    """
    if rotation_point is None:
        return Input("sector rotation", "no sector map supplied", NO_DATA,
                     note="absence of a rotation map is absence of DATA, not "
                          "evidence the sector is fine")
    q = rotation_point.quadrant
    reading = (f"{rotation_point.symbol} {q} "
               f"(RS {rotation_point.rs_ratio:.1f} / Mom {rotation_point.rs_momentum:.1f})")
    return Input("sector rotation", reading, TESTED_NO_EDGE, 1.0,
                 note="context only — on real sector history the quadrant did "
                      "not predict forward returns (t=+0.71, §7.1), so it moves "
                      "size in neither direction")


def _crowding_input(share, note, *, crowded_cut: float, crowded_at: float):
    if share is None:
        return Input("fund crowding", "no holdings archive", NO_DATA,
                     note="tools/archive_holdings.py has not been run long enough; "
                          "an empty archive is not an empty market")
    reading = f"{share:.0%} of mapped fund supply at this strike"
    if share >= crowded_at:
        return Input("fund crowding", reading, NO_DATA, crowded_cut,
                     note="we would be the marginal seller into a multi-billion "
                          "mechanical flow. Whether that costs anything is UNTESTED "
                          "(engine/crowding_backtest.py is waiting on data), so it "
                          "trims size rather than blocking the trade")
    return Input("fund crowding", reading, NO_DATA, 1.0)


def build_decision(card, *, rotation_point=None, crowding=None,
                   lagging_cut: float = 0.5, weakening_cut: float = 0.75,
                   crowded_cut: float = 0.6, crowded_at: float = 0.30) -> Decision:
    """Fuse a TradeCard with sector rotation and fund crowding into one decision.

    ``card`` is a models.trade_card.TradeCard — the variance edge and its gates.
    ``rotation_point`` is a models.rotation.RotationPoint for the underlying's
    sector (None if no map was supplied). ``crowding`` is the
    ``(share, note)`` tuple from models.fund_flow.crowding_score, or None.

    The multipliers are deliberately blunt (0.5 / 0.75 / 0.6). They are judgement
    calls, they are locked in PREREGISTRATION.md, and nothing in this repo has
    earned the right to a finer number.
    """
    inputs: list = []
    warnings = list(getattr(card, "warnings", []) or [])

    vol_side = getattr(card, "vol_side", "NO TRADE")
    inputs.append(Input(
        "variance edge",
        f"{vol_side}  VRP {getattr(card, 'vrp', 0.0):+.4f} "
        f"(P {getattr(card, 'p_vol', 0.0):.1%} vs Q {getattr(card, 'q_vol', 0.0):.1%})",
        VALIDATED,
        note="the only input here backed by oracle-tested machinery — and still "
             "unproven on a live option market"))

    rot = _rotation_input(rotation_point, lagging_cut=lagging_cut,
                          weakening_cut=weakening_cut)
    share, cnote = (crowding if crowding else (None, ""))
    crd = _crowding_input(share, cnote, crowded_cut=crowded_cut,
                          crowded_at=crowded_at)
    inputs += [rot, crd]

    # direction is carried through, flagged, and given no influence at all
    if getattr(card, "direction", None):
        inputs.append(Input(
            "direction", f"{card.direction} (score {card.direction_score:+.2f})",
            FIXTURE_ONLY, 1.0,
            note="carried for context only: no backtest, gate or ledger in this "
                 "repo has shown the direction call profitable, so it does not "
                 "size anything"))

    if vol_side in ("NO TRADE", "", None):
        return Decision(
            symbol=getattr(card, "symbol", ""), asof=getattr(card, "asof", None),
            expiry_days=getattr(card, "expiry_days", 0), action="NO TRADE",
            reason=getattr(card, "vol_reason", "the variance edge did not fire"),
            size_multiplier=0.0, inputs=inputs, warnings=warnings)

    mult = 1.0
    for i in inputs:
        mult *= i.effect
    mult = round(mult, 4)

    reason = getattr(card, "vol_reason", "")
    if mult < 1.0:
        reason += (f" — sized down to {mult:.0%} by "
                   + ", ".join(i.name for i in inputs if i.effect < 1.0))
    d = Decision(
        symbol=getattr(card, "symbol", ""), asof=getattr(card, "asof", None),
        expiry_days=getattr(card, "expiry_days", 0), action=vol_side,
        reason=reason, size_multiplier=mult, inputs=inputs, warnings=warnings)

    if d.weakest_evidence == NO_DATA:
        d.warnings.append("a NO-DATA input is moving this decision — it is a "
                          "placeholder for a measurement that has not been made")
    return d
