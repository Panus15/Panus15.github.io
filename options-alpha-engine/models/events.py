"""Scheduled-event gate — the trap that eats single-stock vol sellers.

The other three gates are about UNKNOWN risk: the price gate reacts to realised
vol accelerating, the news gate to a burst of headlines, the macro gate to the
economic backdrop. This one is about KNOWN risk — an event whose DATE is already
public: earnings, an FDA decision, an FOMC meeting, a CPI print.

Why it matters more for equities than anything else in this repo. Before an
earnings date a single stock's implied vol is high **for a reason**: the market
knows a jump is coming and is pricing it. The engine, comparing that implied vol
to a HAR-RV forecast built from ordinary trailing days, sees a huge variance risk
premium and shouts RICH. Selling it is not harvesting a premium — it is selling
insurance against a scheduled jump at a price that is usually about right, and
occasionally catastrophically wrong. It is the classic way a short-vol book on
single names blows up.

So: if a known event falls inside the option's life, the premium is EVENT
premium, not mispricing, and this gate says so.

  event_gate      does a dated event fall inside [asof, asof + dte]?
  EventCalendar   dated events per symbol, from a JSON/CSV file (point-in-time)
  earnings_iv_note  the other half of the story — after the print, IV collapses
                  ("IV crush"), which is why buying vol before an event and
                  selling after is a different trade with a different sign.

It composes exactly like the others: OR it into the `stressed` flag that
edge.compare / the backtests / the paper ledger already take.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field

#: Event kinds that reliably move a single name enough to matter for a vol seller.
HIGH_IMPACT = ("earnings", "fda", "trial", "guidance", "split", "merger")


@dataclass
class MarketEvent:
    date: str            # ISO date the event is KNOWN to occur
    symbol: str          # "" for market-wide events (FOMC, CPI)
    kind: str            # 'earnings' | 'fomc' | 'cpi' | 'fda' | ...
    note: str = ""

    @property
    def is_high_impact(self) -> bool:
        return self.kind.lower() in HIGH_IMPACT


def _as_date(x) -> datetime.date | None:
    if isinstance(x, datetime.date):
        return x
    try:
        return datetime.date.fromisoformat(str(x)[:10])
    except (TypeError, ValueError):
        return None


@dataclass
class EventCalendar:
    """Dated events, filterable by symbol. Load from a file or build in code."""
    events: list = field(default_factory=list)

    @classmethod
    def from_file(cls, path: str) -> "EventCalendar":
        """JSON: list of {date, symbol, kind, note?}. CSV: same columns."""
        import csv
        import json
        rows = []
        if path.endswith(".json"):
            with open(path) as fh:
                rows = json.load(fh)
        else:
            with open(path, newline="") as fh:
                rows = list(csv.DictReader(fh))
        return cls([MarketEvent(r["date"], r.get("symbol", ""), r.get("kind", "earnings"),
                                r.get("note", "")) for r in rows])

    def between(self, start, end, *, symbol: str | None = None,
                high_impact_only: bool = False) -> list:
        """Events in [start, end] for ``symbol`` (market-wide events always count)."""
        s, e = _as_date(start), _as_date(end)
        if s is None or e is None:
            return []
        out = []
        for ev in self.events:
            d = _as_date(ev.date)
            if d is None or not (s <= d <= e):
                continue
            if symbol and ev.symbol and ev.symbol.upper() != symbol.upper():
                continue
            if high_impact_only and not ev.is_high_impact:
                continue
            out.append(ev)
        return sorted(out, key=lambda x: x.date)


def event_gate(calendar: EventCalendar | None, symbol: str, asof, dte: int,
               *, high_impact_only: bool = True) -> tuple[bool, str]:
    """(elevated_event_risk, reason) — True when a KNOWN event lands inside the
    option's life, so its implied vol is event premium rather than mispricing.

    ``asof`` is the decision date; the window is [asof, asof + dte calendar days].
    Returns (False, ...) with no calendar, which is the honest default: absence of
    a calendar is not evidence of absence of an event.
    """
    if calendar is None:
        return False, "no event calendar supplied (absence of data, not of events)"
    start = _as_date(asof)
    if start is None:
        return False, "undated snapshot — cannot check the event calendar"
    end = start + datetime.timedelta(days=int(dte))
    hits = calendar.between(start, end, symbol=symbol, high_impact_only=high_impact_only)
    if not hits:
        return False, f"no scheduled events in the next {dte}d"
    first = hits[0]
    days = (_as_date(first.date) - start).days
    return True, (f"{first.kind} for {first.symbol or 'the market'} in {days}d "
                  f"({first.date}) — the implied vol is EVENT premium, not mispricing")


def event_stress_at(calendar: EventCalendar | None, symbol: str, dates, dte: int,
                    *, high_impact_only: bool = True):
    """Build the ``event_at(t, trailing) -> bool`` veto the backtests accept.

    The harnesses walk an integer bar index; a calendar walks dates. ``dates`` is
    the bridge — either a list of ISO dates aligned 1:1 with the price series, or
    a callable ``t -> date``. Anything the mapping cannot date returns False, on
    the same principle as ``event_gate``: missing data is not evidence of a quiet
    calendar, and the honest failure is to leave the veto off rather than to
    invent a quiet day.

    Returning None here (no calendar) is deliberate. It lets a caller pass the
    result straight through to ``run_signal_backtest(event_at=...)`` and get
    exactly the old behaviour when there is nothing to gate on.
    """
    if calendar is None:
        return None

    lookup = dates if callable(dates) else (
        lambda t: dates[t] if 0 <= t < len(dates) else None)

    def event_at(t, trailing=None) -> bool:
        asof = lookup(t)
        if asof is None:
            return False
        hit, _ = event_gate(calendar, symbol, asof, dte,
                            high_impact_only=high_impact_only)
        return hit

    return event_at


def earnings_iv_note(days_to_event: int | None) -> str:
    """The other half of the story, for a human reading the card."""
    if days_to_event is None:
        return ""
    if days_to_event <= 0:
        return ("the event has passed — implied vol typically COLLAPSES right after "
                "(IV crush), so a short-vol position opened before it is now marked "
                "against a much lower vol")
    if days_to_event <= 5:
        return (f"{days_to_event}d to the event: implied vol is usually at its peak "
                f"here and crushes immediately after. Selling now is a bet on the "
                f"jump being smaller than priced, NOT a variance-premium harvest")
    return (f"{days_to_event}d to the event: implied vol will keep building into it; "
            f"any short-vol position must survive that build, not just the jump")
