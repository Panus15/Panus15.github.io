"""One order ticket: everything needed to place a trade, or a named refusal.

THE GAP THIS CLOSES. `models/decision.py` ends at "SELL VOL, size x0.85". That is
a view, not an order. Six things are missing before a human can type it into a
broker: which structure, which strikes, which expiry DATE, how many contracts, at
what limit, and what the worst case costs in dollars. Until those exist the most
actionable-looking thing on the screen is the direction block, which this repo has
never validated, while the half it HAS verified prints no levels at all.

WHY EVERY TICKET IS DEFINED-RISK, and why that is arithmetic rather than taste.
`engine/sizing.position_size` caps on the LOSS. A naked short call has no finite
loss, so it sizes to zero and is refused. A naked short index put has a loss of
roughly strike x 100, which on a $100,000 account at a 2% cap means zero contracts
too — one SPY 688 put risks $68,800 against a $2,000 budget. The only structure a
retail account can actually size here is one with a wing on it, so that is the
only structure this builds.

REFUSAL IS AN OUTPUT, NOT AN ABSENCE. Every reason this can decline is named,
enumerable and rendered. A surface that shows numbers when it is confident and
falls silent when it is not teaches the reader that silence means "still thinking"
rather than "no". `Ticket.refusals` is a list of `(code, detail)` and an empty list
is the only thing that makes a ticket placeable.

WHY THE FUSED DECISION IS AN INPUT AND NOT A SUGGESTION. `decision.py` exists to
make unproven inputs cut size and never raise it. That design was bypassed here:
this function read only the card, so a decision of "SELL VOL, size x0.60" built
the SAME ticket as "size x1.00". Measured on one put credit spread, the shipped
order carried 67-100% more risk than the decision authorised ($19,550 against an
intended $11,050 on a $1M account). A cut that the last step discards is not a
risk control, it is a comment. So `decision` is now applied to the contract
count, and when it is ABSENT the ticket says so on its face rather than quietly
printing full size — an optional safety check that can be skipped silently is the
same defect wearing a keyword argument.

WHAT IT STILL DOES NOT KNOW. The forward ledger has no settled trades, so no
ticket from this module carries live evidence that its edge exists. The evidence
label says so, and it is derived from a counter rather than from a judgement.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field

from engine import sizing

#: Enumerated refusal codes. Each one must be independently reachable and is
#: tested that way — a code that cannot fire is a claim of safety nobody checked.
NO_VOL_EDGE = "NO_VOL_EDGE"
NO_STRUCTURE = "NO_STRUCTURE"
NON_POSITIVE_EV = "NON_POSITIVE_EV"
UNBOUNDED_RISK = "UNBOUNDED_RISK"
SIZE_ZERO = "SIZE_ZERO"
STALE_QUOTES = "STALE_QUOTES"
#: The fused decision stood the trade down, or cut it below one contract. Kept
#: separate from SIZE_ZERO because the remedy differs: SIZE_ZERO is "the account
#: is too small for this structure", these two are "the context said no".
DECISION_STANDS_DOWN = "DECISION_STANDS_DOWN"
DECISION_SIZED_TO_ZERO = "DECISION_SIZED_TO_ZERO"

#: Settled forward trades below which the engine has no live evidence at all.
#: Matches the >= 30 that PREREGISTRATION.md §2.3 requires before its primary
#: question can even be asked.
MIN_SETTLED_FOR_EVIDENCE = 30


@dataclass
class TicketLeg:
    kind: str          # 'call' | 'put'
    strike: float
    side: str          # 'short' | 'long'
    price: float       # per share, executable side of the quote

    def line(self) -> str:
        return (f"{self.side.upper():5} {self.kind.upper():4} "
                f"{self.strike:>9.2f} @ {self.price:>7.2f}")


@dataclass
class Ticket:
    symbol: str
    asof: str
    structure: str
    expiry_days: int
    expiry_date: str
    legs: list = field(default_factory=list)
    contracts: int = 0
    limit_credit: float = 0.0        # per contract, dollars, >0 = we receive
    max_loss_per_contract: float = 0.0
    equity: float = 0.0
    max_risk_frac: float = 0.02
    prob_profit: float = 0.0
    ev_per_contract: float = 0.0
    exit_rule: str = ""
    settled_trades: int = 0
    refusals: list = field(default_factory=list)   # [(code, detail)]
    size_multiplier: float = 1.0     # from the fused decision; only ever cuts
    uncapped_contracts: int = 0      # what the risk cap alone allowed
    decision_applied: bool = False   # False = built from the card alone
    cut_by: str = ""                 # which inputs caused the reduction

    @property
    def placeable(self) -> bool:
        return not self.refusals and self.contracts > 0

    @property
    def max_loss_total(self) -> float:
        return self.max_loss_per_contract * self.contracts

    @property
    def credit_total(self) -> float:
        return self.limit_credit * self.contracts

    @property
    def evidence(self) -> str:
        """Derived from a counter, never from a judgement about the setup."""
        if self.settled_trades <= 0:
            return ("no live evidence — the forward ledger has settled 0 trades, "
                    "so nothing here has been graded against a real outcome")
        if self.settled_trades < MIN_SETTLED_FOR_EVIDENCE:
            return (f"{self.settled_trades} settled trade(s) — below the "
                    f"{MIN_SETTLED_FOR_EVIDENCE} that §2.3 needs before its "
                    f"question can be asked at all")
        return (f"{self.settled_trades} settled trades — enough to ask the "
                f"question; read the P&L interval, not the point")

    def _context_lines(self) -> list:
        """How the fused decision moved this ticket — including when it did not
        reach it at all. Silence here would be indistinguishable from consent."""
        if not self.decision_applied:
            return [" fused context  NOT APPLIED — this ticket was built from the "
                    "variance card alone,",
                    "                so sector rotation and fund crowding have not "
                    "cut it. Pass",
                    "                decision= to apply them."]
        if self.size_multiplier >= 1.0:
            return [" fused context  applied, size x1.00 — nothing cut it"]
        return [f" fused context  size x{self.size_multiplier:.2f} applied"
                f"  ({self.uncapped_contracts} -> {self.contracts} contracts)"
                + (f", cut by {self.cut_by}" if self.cut_by else "")]

    def render(self) -> str:
        bar = "=" * 66
        out = [bar, f" {self.symbol}  {self.structure}  exp {self.expiry_date} "
                    f"({self.expiry_days}d)", bar]
        if not self.placeable:
            out.append(" REFUSED")
            for code, detail in self.refusals:
                out.append(f"   {code:22} {detail}")
            out += [""] + self._context_lines()
            out += ["", f" evidence: {self.evidence}", bar]
            return "\n".join(out)

        out.append(f" SELL {self.contracts} x {self.structure}")
        out.append("")
        for leg in self.legs:
            out.append("   " + leg.line())
        out += ["",
                f" limit          {self.limit_credit:>10,.2f} credit / contract"
                f"   ({self.credit_total:>+,.2f} total)",
                f" max loss       {self.max_loss_per_contract:>10,.2f} / contract"
                f"   ({self.max_loss_total:>,.2f} total)",
                f" risk budget    {self.max_risk_frac:>10.1%} of "
                f"{self.equity:,.0f} = {self.max_risk_frac * self.equity:,.0f}",
                f" P(profit)      {self.prob_profit:>10.1%}   model EV "
                f"{self.ev_per_contract:>+,.2f}/contract",
                ""]
        out += self._context_lines() + [""]
        if self.exit_rule:
            out += [f" exit: {self.exit_rule}", ""]
        out += [f" evidence: {self.evidence}", bar]
        return "\n".join(out)


def _expiry_date(asof: str, days: int) -> str:
    try:
        d = datetime.date.fromisoformat(str(asof)[:10])
    except (TypeError, ValueError):
        return f"+{days}d"
    return (d + datetime.timedelta(days=int(days))).isoformat()


def build_ticket(card, spread, *, equity: float, max_risk_frac: float = 0.02,
                 settled_trades: int = 0, asof: str | None = None,
                 exit_rule: str = "", decision=None) -> Ticket:
    """Join a vol verdict and a defined-risk structure into one placeable order.

    Refuses rather than guesses. ``card`` is a models.trade_card.TradeCard and
    ``spread`` a models.spreads.Spread; either may be None, which is itself a
    named refusal rather than a crash.

    ``decision`` is a models.decision.Decision. When supplied its
    ``size_multiplier`` is applied AFTER the risk cap — the cap is a ceiling and
    the multiplier only ever cuts further, so the budget property is preserved by
    construction — and its ``action`` must agree with the card or the ticket
    stands down. When it is omitted the ticket is still built, but it says on its
    face that the fused context was not applied.
    """
    mult = getattr(spread, "contract_mult", 100.0) or 100.0
    sym = getattr(card, "symbol", "") or ""
    when = asof or getattr(card, "asof", None) or ""
    dte = int(getattr(spread, "expiry_days", 0) or getattr(card, "expiry_days", 0) or 0)

    refusals: list = []
    d_action = getattr(decision, "action", None) if decision is not None else None
    if d_action is not None and d_action != getattr(card, "vol_side", ""):
        # The two disagree. Whatever the cause, the conservative reading is the
        # one that does not place an order the fused view did not authorise.
        refusals.append((DECISION_STANDS_DOWN,
                         f"the fused decision says {d_action!r} while the card says "
                         f"{getattr(card, 'vol_side', '') !r}; no order is placed on "
                         f"a disagreement"))
    if card is None or getattr(card, "vol_side", "") != "SELL VOL":
        refusals.append((NO_VOL_EDGE,
                         getattr(card, "vol_reason", "no variance edge") if card
                         else "no trade card was produced"))
    if spread is None or not getattr(spread, "legs", None):
        refusals.append((NO_STRUCTURE,
                         "no defined-risk structure could be built from this chain"))
        return Ticket(symbol=sym, asof=str(when), structure="-", expiry_days=dte,
                      expiry_date=_expiry_date(when, dte), equity=equity,
                      max_risk_frac=max_risk_frac, settled_trades=settled_trades,
                      refusals=refusals)
    if not when:
        refusals.append((STALE_QUOTES,
                         "the chain carries no asof date, so the quotes cannot be "
                         "shown to be current"))

    per_contract_loss = float(getattr(spread, "max_loss", 0.0) or 0.0) * mult
    if per_contract_loss <= 0 or per_contract_loss == sizing.UNBOUNDED:
        refusals.append((UNBOUNDED_RISK,
                         "the structure has no finite maximum loss, so there is "
                         "nothing for a risk cap to bind on"))
        per_contract_loss = sizing.UNBOUNDED

    ev = float(getattr(spread, "ev", 0.0) or 0.0) * mult
    if ev <= 0:
        refusals.append((NON_POSITIVE_EV,
                         f"model EV is {ev:+,.2f} per contract before it is sized"))

    n_cap = sizing.position_size(equity, max_loss_per_contract=per_contract_loss,
                                 max_risk_frac=max_risk_frac)
    # The cap is a ceiling; the multiplier is a further cut applied on top of it.
    # int() floors, which is the only rounding that cannot ship more than the
    # decision authorised — 0.6 x 2 must be 1, never 2.
    size_raw = getattr(decision, "size_multiplier", 1.0) if decision is not None else 1.0
    try:
        size_mult = float(size_raw)
    except (TypeError, ValueError):
        size_mult = float("nan")
    if size_mult != size_mult or size_mult in (float("inf"), float("-inf")):
        # NaN survives min/max clamping untouched (every comparison against it is
        # False), and inf clamps UP to 1.0. Either way a corrupt risk input would
        # have become a full-size order, so unusable means zero, not maximum.
        size_mult = 0.0
    size_mult = min(max(size_mult, 0.0), 1.0)
    n = int(n_cap * size_mult)
    # Only when the multiplier reflects an actual CUT. A decision whose action is
    # NO TRADE also reports x0.00, and NO_VOL_EDGE above already says exactly that
    # — raising a second code for it lists one cause twice and names no input,
    # which reads to a human as an extra independent problem to go and fix.
    if n_cap > 0 and n <= 0 and NO_VOL_EDGE not in {c for c, _ in refusals}:
        cut = ", ".join(i.name for i in getattr(decision, "inputs", [])
                        if getattr(i, "effect", 1.0) < 1.0)
        refusals.append((DECISION_SIZED_TO_ZERO,
                         f"the risk cap allowed {n_cap} contract(s) and the fused "
                         f"decision's x{size_mult:.2f} cut that below one"
                         + (f" (cut by {cut})" if cut else "")))
    if n_cap <= 0:
        budget = max_risk_frac * equity
        refusals.append((SIZE_ZERO,
                         f"one contract risks "
                         f"{'unbounded' if per_contract_loss == sizing.UNBOUNDED else f'{per_contract_loss:,.2f}'}"
                         f" against a budget of {budget:,.2f}"))

    legs = [TicketLeg(kind=l.kind, strike=l.strike, side=l.side, price=l.price)
            for l in spread.legs]
    return Ticket(
        symbol=sym, asof=str(when),
        structure=getattr(spread, "name", "spread"),
        expiry_days=dte, expiry_date=_expiry_date(when, dte),
        legs=legs, contracts=(n if not refusals else 0),
        limit_credit=float(getattr(spread, "net_credit", 0.0) or 0.0) * mult,
        max_loss_per_contract=(0.0 if per_contract_loss == sizing.UNBOUNDED
                               else per_contract_loss),
        equity=equity, max_risk_frac=max_risk_frac,
        prob_profit=float(getattr(spread, "prob_profit", 0.0) or 0.0),
        ev_per_contract=ev, exit_rule=exit_rule,
        settled_trades=settled_trades, refusals=refusals,
        size_multiplier=size_mult, uncapped_contracts=n_cap,
        decision_applied=decision is not None,
        cut_by=", ".join(i.name for i in getattr(decision, "inputs", [])
                         if getattr(i, "effect", 1.0) < 1.0))
