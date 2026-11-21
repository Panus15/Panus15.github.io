"""What you can actually change — the answer to "so what do I do?"

Every other module here says no. That is correct and it is not enough: a person
holding eight named refusals still has to decide something. This module is the
constructive half, and it stays honest by only ever doing arithmetic on
quantities you control.

YOU CONTROL FOUR THINGS AND NONE OF THEM IS THE MARKET.

    cost        switch broker or account type — subtracts directly
    geometry    how wide the target and the stop are
    hold time   financing is per night, so this is a cost you choose
    frequency   how many times a year you pay all of the above

YOU DO NOT CONTROL THE WIN RATE OR THE PAYOFF. They are properties of the rule
meeting the market, and every retail system that fails does so while trying to
adjust them. They are reported here as DIAGNOSTICS — the gap between what you
have and what you would need — never as dials.

THE ONE RESULT WORTH THE MODULE. If gross expectancy is zero or negative, NO
change to cost, geometry, hold time or frequency helps. Not a cheaper broker,
not a free one. net = gross - cost, so a rule that loses before costs still
loses at zero cost, and a rule whose gross edge is smaller than the cheapest
round turn available anywhere cannot be rescued by shopping. That single check
decides whether your problem is your broker or your idea, and people spend
years on the first when it was always the second.

WIDENING IS A HYPOTHESIS, NOT A CALCULATION. Scaling the target and the stop by
k multiplies gross expectancy by k while cost stays fixed, so the arithmetic
says a wide-enough version of any gross-positive rule clears its costs. The
arithmetic is also wrong, because it holds the win rate constant and a wider
stop is hit LESS often while a wider target is reached less often too. The net
effect is genuinely ambiguous. So `geometry_lever` reports the k that WOULD be
needed if nothing else moved, and says in the same breath that you have to
re-run the backtest at that geometry rather than believe the projection.

TRADING LESS OFTEN DOES NOT FIX NEGATIVE EXPECTANCY. It is the most common
advice given to a losing trader and it is arithmetically empty: per-trade
expectancy is unchanged by frequency, so fewer trades lose less money more
slowly and never make any. What frequency does change is annual FRICTION, which
matters enormously when expectancy is positive and not at all when it is not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from engine.costs import TYPICAL_ROUND_TURN_PIPS
from engine.expectancy import Expectancy

#: Cheapest all-in round turn published anywhere on a major pair (raw/ECN
#: account: near-zero spread plus commission). A cost lever asking for less than
#: this is asking for a broker that does not exist.
CHEAPEST_ROUND_TURN = TYPICAL_ROUND_TURN_PIPS[0]

#: Beyond this multiple, a "wider version of the same rule" is a different rule
#: with different behaviour, and projecting the old win rate onto it is fiction.
MAX_PLAUSIBLE_WIDENING = 5.0


@dataclass(frozen=True)
class Lever:
    """One thing you could change, and whether changing it would be enough."""

    name: str
    current: float
    needed: float
    achievable: bool
    detail: str

    @property
    def gap(self) -> float:
        return self.needed - self.current

    def __str__(self) -> str:
        mark = "CAN" if self.achievable else "cannot"
        return f"[{mark}] {self.name}: {self.detail}"


def _gross_of(e: Expectancy) -> float:
    """Expectancy before costs. Everything below turns on its sign."""
    return e.gross


def cost_lever(e: Expectancy) -> Lever:
    """The round turn that would put this rule at break-even.

    Break-even needs cost < gross, so the required cost IS the gross edge. When
    that is at or below zero the lever does not exist at any price — which is
    the most useful thing this module can tell you.
    """
    gross = _gross_of(e)
    if gross <= 0:
        return Lever(
            "cost", e.cost_pips, math.nan, False,
            f"no cost gets you there. The rule loses {gross:+.2f} pips a trade "
            f"BEFORE any cost, so it loses at a free broker too. Your problem "
            f"is the rule, not the spread")
    ok = gross > CHEAPEST_ROUND_TURN
    # what the rule would earn at the cheapest account that exists -- NOT the
    # saving needed to reach break-even, which is a different and much more
    # flattering number, since reaching break-even earns you nothing
    at_best = gross - CHEAPEST_ROUND_TURN
    return Lever(
        "cost", e.cost_pips, gross, ok,
        f"break-even needs a round turn under {gross:.2f} pips; you pay "
        f"{e.cost_pips:.2f}. " + (
            f"At the cheapest all-in account published (about "
            f"{CHEAPEST_ROUND_TURN} pips) this rule makes {at_best:+.2f} pips a "
            f"trade — reachable, and thin" if ok else
            f"The cheapest all-in round turn published anywhere is about "
            f"{CHEAPEST_ROUND_TURN} pips, so no broker is cheap enough. The "
            f"gross edge is smaller than the cost of trading at all"))


def geometry_lever(e: Expectancy) -> Lever:
    """How much wider the target and stop would have to be.

    Scaling both by k gives net(k) = k*gross - cost, so k = cost/gross clears
    it. That holds the win rate constant, which is FALSE — see the module
    docstring. The number is where to re-run the backtest, not a projection to
    act on.
    """
    gross = _gross_of(e)
    if gross <= 0:
        return Lever(
            "geometry", 1.0, math.nan, False,
            "widening multiplies a negative gross edge into a bigger negative "
            "one. There is nothing here to scale up")
    k = e.cost_pips / gross if gross > 0 else math.inf
    if k <= 1.0:
        return Lever("geometry", 1.0, 1.0, True,
                     "already clears its costs at this geometry; no widening "
                     "needed")
    ok = k <= MAX_PLAUSIBLE_WIDENING
    return Lever(
        "geometry", 1.0, k, ok,
        f"targets and stops {k:.2f}x wider would clear the cost IF the win rate "
        f"held — and it will not, because a wider stop is hit less often and a "
        f"wider target reached less often, in opposite directions. " + (
            f"Re-run the backtest at {e.avg_win_pips * k:.0f}/"
            f"{e.avg_loss_pips * k:.0f} pips and let it answer" if ok else
            f"At {k:.1f}x this is a different strategy, not a wider one, and "
            f"the old win rate tells you nothing about it"))


def win_rate_lever(e: Expectancy) -> Lever:
    """Reported, never recommended. You cannot dial a win rate."""
    return Lever(
        "win rate", e.win_rate, e.break_even_win_rate, False,
        f"would need {e.break_even_win_rate:.1%} against the {e.win_rate:.1%} "
        f"measured ({e.break_even_win_rate - e.win_rate:+.1%}). This is not a "
        f"dial: it is what the rule does when it meets the market, and "
        f"{e.cost_pips / (e.avg_win_pips + e.avg_loss_pips):.1%} of the "
        f"requirement is the cost rather than the market")


def payoff_lever(e: Expectancy) -> Lever:
    """What the average win would have to be, at this win rate and stop.

    Partly controllable — letting winners run is a real choice — but it moves
    the win rate too, and in the wrong direction, so it is a hypothesis to test
    rather than a parameter to set.
    """
    if e.win_rate <= 0:
        return Lever("payoff", e.payoff_ratio, math.inf, False,
                     "no winning trades to run further")
    need_w = ((1.0 - e.win_rate) * e.avg_loss_pips + e.cost_pips) / e.win_rate
    need_b = need_w / e.avg_loss_pips if e.avg_loss_pips > 0 else math.inf
    return Lever(
        "payoff", e.payoff_ratio, need_b, False,
        f"the average win would have to reach {need_w:.1f} pips "
        f"(payoff {need_b:.2f}) against {e.avg_win_pips:.1f} measured. Letting "
        f"winners run is a real choice, but it lowers the win rate at the same "
        f"time, so this is a hypothesis to backtest, not a setting to change")


def frequency_note(e: Expectancy, trades_per_year: float) -> str:
    """Why 'trade less often' is not the answer it is always given as."""
    annual = e.net * trades_per_year
    if e.net > 0:
        return (f"at {trades_per_year:,.0f} trades a year this is "
                f"{annual:+,.0f} pips. Frequency scales a POSITIVE expectancy, "
                f"so here it is the one lever that compounds")
    return (f"at {trades_per_year:,.0f} trades a year this is {annual:+,.0f} "
            f"pips. Trading less often does not help: per-trade expectancy is "
            f"unchanged by frequency, so fewer trades lose less money more "
            f"slowly and never make any. Only the SIGN matters, and frequency "
            f"cannot change a sign")


def levers(e: Expectancy) -> list:
    """Every lever, with the two you control first."""
    return [cost_lever(e), geometry_lever(e), win_rate_lever(e),
            payoff_lever(e)]


def advise(e: Expectancy, trades_per_year: float = 250.0) -> str:
    """The one thing to do next, or the plain statement that there isn't one."""
    gross = _gross_of(e)
    ls = levers(e)
    lines = [f"net {e.net:+.2f} pips/trade "
             f"(gross {gross:+.2f}, cost {e.cost_pips:.2f})"]
    for l in ls:
        lines.append(f"  {l}")
    lines.append("")
    lines.append(f"  frequency: {frequency_note(e, trades_per_year)}")
    lines.append("")

    if e.net > 0:
        lines.append("VERDICT: it already clears its costs. The levers above "
                     "say how much room there is,\nnot what to fix.")
        return "\n".join(lines)
    if gross <= 0:
        lines.append(
            "VERDICT: NOTHING YOU CONTROL FIXES THIS. The rule loses before "
            "costs are charged\nat all, so a cheaper broker, wider targets, "
            "shorter holds and fewer trades all leave\nit losing. The honest "
            "next step is a different rule, not a different setup.")
        return "\n".join(lines)
    doable = [l for l in ls if l.achievable]
    if doable:
        lines.append(f"VERDICT: the gross edge is real but smaller than what "
                     f"you pay for it. In order of\nhow much they are in your "
                     f"hands: " + ", ".join(l.name for l in doable) + ".")
        lines.append(f"  -> {doable[0]}")
    else:
        lines.append(
            "VERDICT: there is a gross edge and nothing in your control is "
            "enough to reach it.\nThe cheapest broker available is still dearer "
            "than the edge is worth, and widening\nfar enough stops being the "
            "same rule. This is a rule that works on paper only.")
    return "\n".join(lines)
