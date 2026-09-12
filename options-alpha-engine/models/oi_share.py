"""Does a fund actually own enough of a strike to move it?

THE QUESTION THIS SETTLES, AND WHY IT IS WORTH A DAY.

The premise behind `engine/crowding_backtest.py` and PREREGISTRATION.md §2.2 is
that mechanical option sellers — covered-call and option-income ETFs — park enough
size on particular strikes to leave a mark on the price there. The study designed
to test that needs ~18 months of daily, point-in-time fund books, which cannot be
bought retroactively at any price. That is a long time to wait to discover the
premise was never plausible.

But the premise has a NECESSARY condition that can be checked in an afternoon: the
fund has to hold a meaningful share of the open interest at its own strike. If it
holds 1% of the contracts outstanding there, it is one participant among a hundred
and no amount of archiving will make it the marginal one. This module measures that
share. It is a screen, not the study — passing it does not show an effect exists,
it only shows the mechanism is not arithmetically ruled out.

THE DECISION RULE, WRITTEN DOWN BEFORE ANY DATA WAS SEEN, which is the only way a
threshold means anything:

    max share < 5%      REFUTED. The fund is a marginal participant at its own
                        strike. §2.2 should be CLOSED rather than waited on.
    5% .. 20%           INCONCLUSIVE. Possible but not obviously so; the archive
                        is worth starting, with this recorded as the reason.
    >= 20%              PLAUSIBLE. Large enough that price impact is arguable, and
                        the 18-month study is justified.

The bands come from the demand-pressure literature rather than from taste:
Gârleanu, Pedersen & Poteshman's demand-based option pricing has end-user demand
moving prices when dealers cannot hedge costlessly, and the effect scales with the
size of the imbalance relative to the market. There is no clean published cutoff,
so these are stated as a judgement made in advance and labelled as such — the point
is that the number cannot be chosen after the answer is known.

WHAT IT WILL NOT DO. Open interest of 0 means NOT KNOWN in this codebase, since
most feeds here do not carry it. A strike with unknown OI is reported as unknown
and excluded from the verdict rather than counted as a 100% share, which is the
error that would make every fund look dominant.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Pre-registered bands. Changing these after seeing a result requires the same
#: amendment discipline as PREREGISTRATION.md §2 — a dated note saying what the
#: answer was BEFORE the change.
REFUTED_BELOW = 0.05
PLAUSIBLE_AT = 0.20


@dataclass
class StrikeShare:
    """One (strike, expiry, kind) the fund holds, against the market's size."""
    strike: float
    expiry_days: int
    kind: str
    fund_contracts: float          # absolute, contracts
    open_interest: int             # 0 == unknown
    moneyness: float               # strike / spot - 1

    @property
    def known(self) -> bool:
        return self.open_interest > 0

    @property
    def share(self) -> float | None:
        """Fraction of contracts outstanding that this fund holds, or None."""
        if not self.known:
            return None
        return self.fund_contracts / float(self.open_interest)

    def line(self) -> str:
        sh = "   unknown" if self.share is None else f"{self.share:>9.1%}"
        return (f"  {self.kind:5}{self.strike:>10.2f}{self.expiry_days:>6}d"
                f"{self.moneyness:>+9.1%}{self.fund_contracts:>12,.0f}"
                f"{(self.open_interest or 0):>12,}{sh}")


@dataclass
class ShareReport:
    rows: list = field(default_factory=list)
    underlying: str = ""
    asof: str = ""
    unmatched: int = 0             # fund lines with no quote in the chain

    @property
    def known_rows(self) -> list:
        return [r for r in self.rows if r.known]

    @property
    def max_share(self) -> float | None:
        known = [r.share for r in self.known_rows]
        return max(known) if known else None

    @property
    def weighted_share(self) -> float | None:
        """Fund contracts over market contracts, pooled across known strikes.

        Reported next to the max because they answer different questions: the max
        asks whether the fund dominates ANY strike, the pooled asks whether it is
        large in aggregate. A fund can top out at 40% on one illiquid strike and
        still be 2% of its own book.
        """
        known = self.known_rows
        if not known:
            return None
        oi = sum(r.open_interest for r in known)
        if oi <= 0:
            return None
        return sum(r.fund_contracts for r in known) / float(oi)

    def verdict(self) -> str:
        if not self.known_rows:
            return ("NO OPEN INTEREST DATA — every matched strike had OI 0, which "
                    "means unknown here, not zero. Nothing can be concluded; "
                    "supply a chain that carries open interest.")
        m = self.max_share
        w = self.weighted_share
        head = (f"max share {m:.1%} at one strike, {w:.1%} pooled across "
                f"{len(self.known_rows)} strikes")
        if m < REFUTED_BELOW:
            return (f"PREMISE REFUTED — {head}. The fund is a marginal participant "
                    f"at its own strikes, so it cannot be the marginal seller "
                    f"setting the price there. PREREGISTRATION.md §2.2 should be "
                    f"closed rather than waited on for 18 months.")
        if m < PLAUSIBLE_AT:
            return (f"INCONCLUSIVE — {head}. Between {REFUTED_BELOW:.0%} and "
                    f"{PLAUSIBLE_AT:.0%}: not ruled out, not demonstrated. Starting "
                    f"the archive is defensible; this number is the reason.")
        return (f"MECHANISM PLAUSIBLE — {head}. Large enough that price impact is "
                f"arguable. This does NOT show an effect exists — it shows the "
                f"18-month study is worth running.")

    def summary(self) -> str:
        out = [f"Share of open interest — {self.underlying} as of {self.asof}",
               f"  {'kind':5}{'strike':>10}{'dte':>7}{'moneyness':>9}"
               f"{'fund ct':>12}{'open int':>12}{'share':>10}"]
        for r in sorted(self.rows, key=lambda x: (x.expiry_days, x.strike)):
            out.append(r.line())
        if self.unmatched:
            out.append(f"  ({self.unmatched} fund line(s) had no matching quote in "
                       f"the chain and are excluded)")
        out.append("  -> " + self.verdict())
        return "\n".join(out)


def share_of_open_interest(book, chain, underlying: str,
                           *, tolerance: float = 0.01) -> ShareReport:
    """Match a fund's option lines to a chain and report its share of each strike.

    ``book`` is a models.fund_flow.FundBook, ``chain`` an engine.data.OptionChain.
    Matching is on kind, strike (within ``tolerance``, since published holdings
    round) and expiry in days. A fund line with no matching quote is COUNTED and
    reported rather than dropped silently — a systematic mismatch, say because the
    fund holds FLEX or OTC options that are not listed, is itself the answer to the
    question, and the most likely way to get a falsely small share is to quietly
    discard the lines that did not match.
    """
    spot = getattr(chain, "spot", 0.0) or 0.0
    by_key: dict = {}
    for q in getattr(chain, "quotes", []):
        by_key.setdefault((q.kind, q.expiry_days), []).append(q)

    rows, unmatched = [], 0
    for p in getattr(book, "positions", []):
        if p.underlying.upper() != underlying.upper():
            continue
        if p.strike <= 0:
            continue
        dte = p.days_to_expiry(getattr(book, "asof", None))
        if dte is None:
            unmatched += 1
            continue
        near = by_key.get((p.kind, dte)) or []
        hit = None
        for q in near:
            if abs(q.strike - p.strike) <= tolerance * max(p.strike, 1.0):
                hit = q
                break
        if hit is None:
            unmatched += 1
            continue
        rows.append(StrikeShare(
            strike=p.strike, expiry_days=dte, kind=p.kind,
            fund_contracts=abs(p.contracts),
            open_interest=int(getattr(hit, "open_interest", 0) or 0),
            moneyness=(p.strike / spot - 1.0) if spot > 0 else 0.0))

    return ShareReport(rows=rows, underlying=underlying.upper(),
                       asof=str(getattr(book, "asof", "") or ""),
                       unmatched=unmatched)
