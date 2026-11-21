"""Expectancy — the number that decides whether a rule makes money.

THE QUESTION THAT WAS ASKED was for the chart pattern with the highest WIN RATE.
This module exists to answer it honestly, because win rate alone cannot:

    E = p*W - (1-p)*L          expectancy per trade
    break-even payoff  W/L = (1-p)/p

A 90% win rate breaks even at a payoff of 0.111 — win 1, lose 9, and it is a
losing strategy. Nothing about 90% is good on its own.

THIS IS NOT A THEORETICAL POINT IN FX. It is the measured state of real retail
accounts (research/BRIEF.md):

    FXCM, 43 million real trades:  61% win rate on EUR/USD, STILL LOSING,
                                   because the average winner was 48 pips and
                                   the average loser 83.
    Ben-David/Birru/Prokopenya:    62.8% of trades won, traders still lost.
    ESMA / FCA / AMF:              74-89% of accounts lose money.

COST IS WHY, AND IT BITES BACKWARDS FROM WHAT PEOPLE EXPECT. Adding a cost c
raises the required win rate by exactly c/(W+L) — so the penalty is LARGEST for
the small-target, high-win-rate systems that "highest win rate" leads you to:

    5 / 5 pips  at 1.5 pips cost   50.0% -> 65.0%   (+15.0 points)
    20 / 20     at 1.5 pips cost   50.0% -> 53.8%   (+3.8)
    50 / 50     at 1.5 pips cost   50.0% -> 51.5%   (+1.5)

AND A THEOREM, because it comes up every time: no position-sizing scheme —
martingale, grid, averaging down, "recovery" — converts negative expectancy into
positive. Expectation is linear; scaling a negative number by any positive size
leaves it negative. That is why 95%-win-rate robots exist and then stop existing.
`sizing_cannot_fix` states it so the claim is in the code rather than in a README.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Expectancy:
    """The full arithmetic of one rule, in pips per trade."""

    win_rate: float
    avg_win_pips: float
    avg_loss_pips: float          # positive magnitude
    cost_pips: float = 0.0

    def __post_init__(self):
        if not 0.0 <= self.win_rate <= 1.0:
            raise ValueError(f"win_rate must be in [0, 1], got {self.win_rate!r}")
        if self.avg_win_pips < 0 or self.avg_loss_pips < 0:
            raise ValueError("avg_win_pips and avg_loss_pips are magnitudes; "
                             "pass the loss as a positive number")

    @property
    def gross(self) -> float:
        """Expectancy per trade BEFORE costs. Never the number to act on."""
        p = self.win_rate
        return p * self.avg_win_pips - (1.0 - p) * self.avg_loss_pips

    @property
    def net(self) -> float:
        """Expectancy per trade after the round-turn cost. The only one that counts."""
        return self.gross - self.cost_pips

    @property
    def payoff_ratio(self) -> float:
        """avg win / avg loss. Infinite when there is no loss to divide by."""
        if self.avg_loss_pips == 0:
            return math.inf
        return self.avg_win_pips / self.avg_loss_pips

    @property
    def break_even_win_rate(self) -> float:
        """The win rate this payoff needs to break even, INCLUDING cost.

            p* = (L + c) / (W + L)

        The cost term is what separates this from the textbook L/(W+L), and the
        difference c/(W+L) is the whole reason small targets are punished.
        """
        denom = self.avg_win_pips + self.avg_loss_pips
        if denom <= 0:
            return math.inf
        return (self.avg_loss_pips + self.cost_pips) / denom

    @property
    def cost_penalty_points(self) -> float:
        """How many percentage points of win rate the cost alone costs you."""
        denom = self.avg_win_pips + self.avg_loss_pips
        if denom <= 0:
            return math.inf
        return self.cost_pips / denom

    @property
    def edge_ratio(self) -> float:
        """Net expectancy as a fraction of the amount risked per trade.

        Reported because "+0.3 pips per trade" means nothing without knowing that
        it was earned by risking 80 pips. This is the number that survives being
        compared across pairs and timeframes.
        """
        if self.avg_loss_pips <= 0:
            return math.inf if self.net > 0 else 0.0
        return self.net / self.avg_loss_pips

    @property
    def profitable(self) -> bool:
        return self.net > 0

    def summary(self) -> str:
        arrow = "PROFITABLE" if self.profitable else "LOSING"
        lines = [
            f"  win rate        {self.win_rate:>8.1%}",
            f"  avg win/loss    {self.avg_win_pips:>8.1f} / {self.avg_loss_pips:.1f} pips"
            f"   (payoff {self.payoff_ratio:.2f})",
            f"  cost            {self.cost_pips:>8.2f} pips round turn",
            f"  gross expectancy{self.gross:>8.2f} pips/trade",
            f"  NET expectancy  {self.net:>8.2f} pips/trade   -> {arrow}",
            f"  needs a win rate of {self.break_even_win_rate:.1%} to break even "
            f"({self.cost_penalty_points:+.1%} of that is the cost alone)",
        ]
        return "\n".join(lines)


def required_win_rate(avg_win_pips: float, avg_loss_pips: float,
                      cost_pips: float = 0.0) -> float:
    """p* = (L + c) / (W + L). Above 1.0 means no win rate can save it."""
    denom = avg_win_pips + avg_loss_pips
    if denom <= 0:
        return math.inf
    return (avg_loss_pips + cost_pips) / denom


def break_even_payoff(win_rate: float) -> float:
    """W/L = (1-p)/p — the payoff a given win rate needs, before costs.

    The answer to "is a 90% win rate good?": it needs only 0.111, which sounds
    easy and is exactly why martingale and grid systems report 90% win rates right
    up until the trade that ends them.
    """
    if not 0.0 < win_rate <= 1.0:
        raise ValueError(f"win_rate must be in (0, 1], got {win_rate!r}")
    return (1.0 - win_rate) / win_rate


def sizing_cannot_fix(expectancy_per_trade: float, size: float) -> float:
    """Expected P&L of ``size`` units of a trade with this expectancy.

    A deliberately trivial function, present because the claim it encodes is the
    one most often argued with: expectation is LINEAR in size, so scaling a
    negative expectancy by any positive size leaves it negative. Martingale, grid
    and "recovery" schemes vary `size` over time; none of them can change the sign
    of the sum, because the sum of negatives is negative whatever the weights.

    What they DO change is the shape: they raise the win rate and move the loss
    into a rarer, larger event. That is why a 95%-win-rate robot is not evidence
    of an edge — it is evidence of a payoff ratio nobody quoted.
    """
    if size < 0:
        raise ValueError("size cannot be negative")
    return expectancy_per_trade * size


def kelly_fraction(win_rate: float, payoff_ratio: float) -> float:
    """f* = p - (1-p)/b. Returned for reference and clamped at 0.

    NOT a sizing recommendation. Kelly assumes the edge estimate is correct, and
    an FX edge estimated on a few hundred trades has a standard error wide enough
    that full Kelly is routinely several times over-levered. It is here so that
    `f* <= 0` can be used as one more way of saying "there is no edge to size".
    """
    if payoff_ratio <= 0:
        return 0.0
    f = win_rate - (1.0 - win_rate) / payoff_ratio
    return max(f, 0.0)
