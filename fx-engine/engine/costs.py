"""What an FX trade actually costs — the layer that decides everything else.

WHY THIS IS MODULE ONE. Every failure mode in research/BRIEF.md has the same
shape: an edge that exists GROSS and dies NET.

    intraday predictability   real gross, dead after 1.01bp
    carry                     real gross, dead after a 0.8%/yr markup on 200% notional
    dollar factor             59% of the gain is the carry leg retail does not get
    real retail accounts      61% win rate, still losing, because 48 < 83

So the cost model is not a correction applied at the end of a backtest. It is the
thing the system is built around, and it comes first.

THREE FACTS THAT BACKTESTS USUALLY GET WRONG.

1. A PIP IS NOT A PIP. One pip is a different fraction of notional on every pair:
   0.909bp on EURUSD at 1.10, 0.787bp on GBPUSD at 1.27, 0.667bp on USDJPY at 150.
   Comparing "1.2 pip spread" across pairs is meaningless; `bp_per_pip` converts.

2. SWAP IS CHARGED ON NOTIONAL, IN BOTH DIRECTIONS, AND LOW TURNOVER DOES NOT
   REDUCE IT. The broker funds at tom-next and adds a markup — published figures
   run 0.5%/yr (cheapest) to 3%/yr. It is a DEBIT on every leg: it never pays you,
   on either side. A long-3/short-3 book is 200% gross notional, so the drag is
   markup x 2, while the carry income is earned only on the NET yield spread. That
   arithmetic is what killed the retail carry trade (VERIFICATION.md).

3. COST RAISES THE REQUIRED WIN RATE BY EXACTLY c/(W+L). Which means it bites
   HARDEST on precisely the small-target, high-win-rate systems people ask for. A
   5-pip-target/5-pip-stop scalp needs 65% instead of 50% at a 1.5 pip cost. See
   `expectancy.py`.

WHAT THIS MODULE DELIBERATELY DOES NOT DO. It does not pretend a stop bounds your
loss. On 15 January 2015 EUR/CHF moved more than 30% with no quotes in between and
clients ended up owing money beyond their account equity; on 3 January 2019 the yen
moved 3% in about thirty seconds with no material news. `gap_risk_note` exists so
that a number which cannot be modelled is at least never silently omitted.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Published all-in round-turn cost on EURUSD at a retail broker, in pips. The low
#: end is a raw/ECN account (0.0-0.1 pip spread + ~0.70 pip commission); the high
#: end is a standard account once slippage is counted. Raw and standard converge —
#: the commission is roughly the spread you saved.
TYPICAL_ROUND_TURN_PIPS = (0.9, 2.5)

#: Broker financing markup over tom-next, per year, on NOTIONAL. 0.8%/yr is the
#: cheapest published figure found (tastyfx/IG "admin fee"); market-maker shops
#: commonly run 2-3%. It is charged in BOTH directions.
TYPICAL_SWAP_MARKUP = (0.005, 0.03)

#: Days per year used to accrue financing. Calendar, not trading: a position held
#: over a weekend is financed for those days too, and Wednesday is charged triple
#: to cover the weekend value date.
FINANCING_DAYS = 365.0


def pip_size(pair: str) -> float:
    """0.01 for JPY crosses, 0.0001 for everything else.

    This is the convention, not a modelling choice — quoting a JPY pair to five
    decimal places would make every pip figure in this module wrong by 100x.
    """
    return 0.01 if pair.upper().endswith("JPY") else 0.0001


def bp_per_pip(pair: str, price: float) -> float:
    """How many basis points of NOTIONAL one pip is worth, at this price.

    The conversion nobody does, and the reason "EURUSD costs 1.2 pips and USDJPY
    costs 1.2 pips" is not a like-for-like comparison:

        EURUSD @ 1.10   0.0001 / 1.10  = 0.909 bp
        GBPUSD @ 1.27   0.0001 / 1.27  = 0.787 bp
        USDJPY @ 150    0.01   / 150   = 0.667 bp

    A cost quoted in pips is not comparable across pairs, and a strategy tested on
    one pair does not inherit the other's cost.
    """
    if price <= 0:
        raise ValueError(f"price must be positive, got {price!r}")
    return pip_size(pair) / price * 10_000.0


@dataclass(frozen=True)
class CostModel:
    """Everything a position pays, measured rather than assumed.

    ``round_turn_pips`` is the ALL-IN cost of getting in and out once: spread plus
    commission plus expected slippage. Quoting spread alone understates it, and on
    a raw account the commission is most of it.

    ``swap_markup_annual`` is the broker's markup over tom-next, as a fraction of
    notional per year, charged in both directions. MEASURE IT rather than accept
    this default — see `measured_markup_hint`. It is the single number that decides
    whether any carry-like strategy is alive, and brokers do not advertise it.
    """

    pair: str = "EURUSD"
    round_turn_pips: float = 1.5
    swap_markup_annual: float = 0.008
    #: Net yield differential you expect to EARN (long high-yielder) or PAY
    #: (short it), as a fraction per year. Signed from the position's point of
    #: view, before the markup is applied.
    carry_annual: float = 0.0

    def __post_init__(self):
        if self.round_turn_pips < 0:
            raise ValueError("round_turn_pips cannot be negative")
        if self.swap_markup_annual < 0:
            raise ValueError("swap_markup_annual cannot be negative")

    # ---- per-trade -------------------------------------------------------

    def round_turn_bp(self, price: float) -> float:
        """The round-turn cost in basis points of notional, at this price."""
        return self.round_turn_pips * bp_per_pip(self.pair, price)

    def financing_pips(self, nights: int, price: float) -> float:
        """Financing paid over ``nights``, expressed in PIPS so it can be added to
        a pip-denominated trade result.

        Signed so that a POSITIVE return value is a cost. The markup is always a
        debit; the carry may be a credit or a debit depending on the position.

        Wednesday's triple charge is not modelled per-weekday here — that requires
        a calendar this function does not have — but `nights` must be CALENDAR
        nights, which captures the weekend on average. `financing_note` says so,
        because a financing figure that quietly assumes 5-day weeks understates a
        month-long hold by roughly 40%.
        """
        if nights < 0:
            raise ValueError("nights cannot be negative")
        frac = nights / FINANCING_DAYS
        # markup is always paid; carry is signed (earned if positive)
        net_annual = self.swap_markup_annual - self.carry_annual
        bp = net_annual * frac * 10_000.0
        return bp / bp_per_pip(self.pair, price)

    def total_cost_pips(self, nights: int, price: float) -> float:
        """All-in cost of one round trip held for ``nights``, in pips.

        This is the number that goes into the expectancy calculation. Anything
        smaller is a backtest lying to itself.
        """
        return self.round_turn_pips + self.financing_pips(nights, price)

    # ---- per-year --------------------------------------------------------

    def annual_friction(self, trades_per_year: float, leverage: float,
                        price: float, *, nights_per_trade: int = 0) -> float:
        """Fraction of ACCOUNT EQUITY consumed by costs in a year.

        Annual friction, not any single trade's spread, is what kills retail
        systems. At 1.5 pips and 250 trades a year it is 3.4% of equity at 1:1
        leverage and 10.2% at 3:1 — before the strategy has to beat anything.
        """
        if trades_per_year < 0 or leverage < 0:
            raise ValueError("trades_per_year and leverage cannot be negative")
        per_trade_bp = (self.total_cost_pips(nights_per_trade, price)
                        * bp_per_pip(self.pair, price))
        return per_trade_bp / 10_000.0 * trades_per_year * leverage

    def gross_notional_drag(self, gross_notional: float) -> float:
        """Annual markup drag on a book carrying ``gross_notional`` x equity.

        THE CARRY ARITHMETIC. A long-3/short-3 G10 book is 200% gross notional, and
        the markup is a debit on EVERY leg — it never pays you on either side — so
        the drag is markup x 2.0, not markup x 1.0. Meanwhile the carry income is
        earned only on the NET yield spread. At the cheapest published markup
        (0.8%/yr) that is 1.6%/yr of drag against a post-2008 G10 carry index
        return of about 0.85%/yr, i.e. negative before a single trade is placed.
        """
        if gross_notional < 0:
            raise ValueError("gross_notional cannot be negative")
        return self.swap_markup_annual * gross_notional

    # ---- honesty ---------------------------------------------------------

    @property
    def financing_note(self) -> str:
        return ("financing accrues on CALENDAR nights, so weekends are charged; "
                "Wednesday is charged triple for the weekend value date and is not "
                "modelled per-weekday here")

    @property
    def gap_risk_note(self) -> str:
        """A stop does not bound your loss, and no cost model can price that."""
        return ("a stop does NOT bound the loss: on 2015-01-15 EUR/CHF moved >30% "
                "with no quotes in between and clients owed money beyond their "
                "equity; on 2019-01-03 the yen moved 3% in ~30 seconds with no "
                "material news. EU/UK negative-balance protection caps the loss at "
                "the account balance, not at the stop")

    @property
    def measured_markup_hint(self) -> str:
        """The five minutes of measurement that decides more than any indicator."""
        return ("MEASURE this instead of assuming it: compare your broker's quoted "
                "swap against the published tom-next / rate differential, on BOTH "
                "the long and the short side, for a full week including the "
                "Wednesday triple charge. If the markup exceeds the yield spread "
                "you intend to harvest, the strategy is arithmetically dead and no "
                "signal work changes that")
