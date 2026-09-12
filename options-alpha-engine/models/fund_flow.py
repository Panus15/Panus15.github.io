"""Fund option footprint — read what the big option-selling funds are actually doing.

Covered-call / option-income ETFs (QQQI, JEPQ, JEPI, XYLD, ...) run a mechanical
options book worth billions, and by law they publish their FULL HOLDINGS DAILY —
including every option line: underlying, strike, expiry, and contract count. That
makes them the rare case of a large, systematic option seller whose positions are
public. This module turns that file into structure the engine can reason about.

What it is honestly good for:

  SUPPLY MAP    where the mechanical sellers have concentrated their short calls,
                by strike and expiry. Persistent supply at a strike is a real
                reason its implied vol sits lower than its neighbours — the
                cheapness is a supply effect, not a mispricing waiting to be
                harvested.
  CROWDING      when our own scanner says WRITE at a strike where a multi-billion
                fund is already the dominant seller, we are the marginal seller
                into someone else's flow. That is worth KNOWING before sizing.
  ROLL CALENDAR these books roll on a schedule. Position changes between two
                published files show when supply is renewed and where it moves to.

What it is NOT:

  * Not a directional signal. A fund selling calls at 5% OTM is executing a
    mandate, not expressing a view — reading intent into it is a mistake.
  * Not fast. Holdings publish with a lag (typically T+1), so this is a STRUCTURAL
    map, not an order feed. Every function here takes the file's own as-of date
    and refuses to pretend it knew earlier (see `FundBook.asof`).
  * Not proven to pay. Whether crowded supply means AVOID (you are late to a
    crushed vol) or FOLLOW (the flow persists) is an EMPIRICAL question, and
    `engine/crowding_backtest.py` is the paired experiment that answers it — but
    only once real published holdings are fed to it. Until that has been run on
    your data, the default here is to WARN and annotate, never to auto-flip a
    verdict — `crowding_note` is information, `crowding_blocks_trade` is opt-in.
"""

from __future__ import annotations

import datetime
import math
from dataclasses import dataclass, field

#: Funds whose published books are dominated by systematic option selling.
KNOWN_INCOME_FUNDS = ("QQQI", "JEPQ", "JEPI", "XYLD", "RYLD", "QYLD", "SPYI", "DIVO")


def _as_date(x):
    if isinstance(x, datetime.date):
        return x
    try:
        return datetime.date.fromisoformat(str(x)[:10])
    except (TypeError, ValueError):
        return None


@dataclass
class FundOptionPosition:
    """One option line from a fund's published holdings."""
    underlying: str
    expiry: str              # ISO date
    strike: float
    kind: str                # 'call' | 'put'
    contracts: float         # NEGATIVE = short (what a covered-call fund holds)
    multiplier: float = 100.0

    @property
    def is_short(self) -> bool:
        return self.contracts < 0

    def notional(self, spot: float | None = None) -> float:
        """Absolute notional controlled: |contracts| x multiplier x (spot or strike)."""
        ref = spot if spot else self.strike
        return abs(self.contracts) * self.multiplier * ref

    def days_to_expiry(self, asof) -> int | None:
        a, e = _as_date(asof), _as_date(self.expiry)
        return (e - a).days if a and e else None

    def moneyness(self, spot: float) -> float:
        """Strike relative to spot: +0.05 = the strike is 5% above spot."""
        return self.strike / spot - 1.0 if spot > 0 else 0.0


#: OSI (Options Symbology Initiative) is how a listed option is named on every
#: US holdings file that does not break it into columns: a 6-character root,
#: YYMMDD, C or P, then the strike as an 8-digit integer of price x 1000.
#:
#:     "QQQ   261016C00620000"  ->  QQQ, 2026-10-16, call, 620.00
#:
#: This matters more than it looks. Without it a real issuer file parses to zero
#: option lines, tools/archive_holdings rejects the day as "no option lines
#: found (the URL probably returned a web page)", and the operator banks nothing
#: while being told the wrong reason. The data was fine; the reader was not.
OSI_LEN = 21


def parse_osi(symbol: str):
    """(root, 'YYYY-MM-DD', 'call'|'put', strike) from an OSI symbol, or None.

    Returns None rather than raising on anything that is not OSI-shaped, because
    a holdings file is mostly equities and cash and only some rows are options.
    """
    if not symbol:
        return None
    s = str(symbol).strip()
    # tolerate the root being space-padded to 6 or not padded at all
    if len(s) < 15:
        return None
    tail = s[-15:]                     # YYMMDD + C/P + 8 digits
    root = s[:-15].strip().upper()
    if not root or not root.isalnum():
        return None
    ymd, cp, strike = tail[:6], tail[6:7].upper(), tail[7:]
    if not ymd.isdigit() or cp not in ("C", "P") or not strike.isdigit():
        return None
    yy, mm, dd = int(ymd[:2]), int(ymd[2:4]), int(ymd[4:6])
    if not (1 <= mm <= 12 and 1 <= dd <= 31):
        return None
    # OSI carries a two-digit year. These files are current holdings, so 20xx is
    # the only reading that is ever right; a 19xx option is not in a live book.
    return (root, f"20{yy:02d}-{mm:02d}-{dd:02d}",
            "call" if cp == "C" else "put", int(strike) / 1000.0)


@dataclass
class FundBook:
    """A fund's option positions as published on ``asof``."""
    fund: str
    asof: str
    positions: list = field(default_factory=list)

    @classmethod
    def from_file(cls, path: str, *, fund: str = "", asof: str = "") -> "FundBook":
        """CSV/JSON of holdings. Recognised columns (case-insensitive):
        underlying/ticker, expiry/expiration, strike, kind/type/call_put,
        contracts/quantity/shares, multiplier?. A positive quantity is long; these
        funds publish their short calls as NEGATIVE quantities."""
        import csv
        import json
        rows = []
        if path.endswith(".json"):
            with open(path) as fh:
                payload = json.load(fh)
            if isinstance(payload, dict):
                fund = fund or payload.get("fund", "")
                asof = asof or payload.get("asof", "")
                payload = payload.get("positions", payload.get("holdings", []))
            rows = payload
        else:
            with open(path, newline="") as fh:
                rows = list(csv.DictReader(fh))

        def pick(r, *names, default=None):
            low = {str(k).strip().lower(): v for k, v in r.items()}
            for n in names:
                if low.get(n) not in (None, ""):
                    return low[n]
            return default

        pos = []
        for r in rows:
            kind = str(pick(r, "kind", "type", "call_put", "putcall", default="call")).lower()
            kind = "put" if kind.startswith("p") else "call"
            underlying = str(pick(r, "underlying", "ticker", "symbol",
                                  default="")).upper()
            expiry = str(pick(r, "expiry", "expiration", "maturity", default=""))
            try:
                strike = float(pick(r, "strike", "strike_price", default=0) or 0)
            except (TypeError, ValueError):
                strike = 0.0

            # Fall back to OSI when the file names the contract instead of
            # tabulating it, which is what real issuer files do. Explicit columns
            # WIN when present - a file that supplies both and disagrees is more
            # likely to have a stale symbol than a stale column.
            if strike <= 0 or not expiry:
                # Try EVERY field rather than a list of column names. Issuers
                # disagree about what the column is called - StockTicker,
                # SecurityID, Identifier - and guessing the name is how a real
                # file parses to zero option lines. OSI is distinctive enough
                # (root + YYMMDD + C/P + 8 digits) that a false positive on an
                # equity ticker or a cash row is not a realistic risk.
                osi = None
                for v in r.values():
                    osi = parse_osi(v)
                    if osi:
                        break
                if osi:
                    root, exp, k, strk = osi
                    # Precedence, applied to EVERY field and not just to strike:
                    # an explicit column always wins. A file carrying both is
                    # likelier to have a stale symbol than a stale column, and
                    # letting the symbol override even one field silently moves
                    # a contract. `had_cols` records whether this row tabulated
                    # its option at all - if it did not, the symbol IS the row.
                    had_cols = strike > 0
                    if not had_cols:
                        underlying = underlying or root
                        kind = k
                        strike = strk
                    expiry = expiry or exp

            try:
                pos.append(FundOptionPosition(
                    underlying=underlying, expiry=expiry, strike=strike, kind=kind,
                    contracts=float(pick(r, "contracts", "quantity", "qty",
                                         "shares", default=0) or 0),
                    multiplier=float(pick(r, "multiplier", default=100) or 100),
                ))
            except (TypeError, ValueError):
                continue
        return cls(fund=fund or "FUND", asof=asof, positions=[p for p in pos if p.strike > 0])

    def shorts(self, underlying: str | None = None) -> list:
        out = [p for p in self.positions if p.is_short]
        if underlying:
            out = [p for p in out if p.underlying.upper() == underlying.upper()]
        return out

    def total_short_notional(self, spot: float, underlying: str | None = None) -> float:
        return sum(p.notional(spot) for p in self.shorts(underlying))


@dataclass
class SupplyBucket:
    expiry: str
    strike: float
    kind: str
    contracts: float          # total SHORT contracts here (positive magnitude)
    notional: float
    funds: list = field(default_factory=list)

    def line(self, spot: float) -> str:
        mny = self.strike / spot - 1.0 if spot > 0 else 0.0
        return (f"  {self.expiry}  {self.kind:>4} {self.strike:>10,.2f} "
                f"({mny:+.1%} OTM)  {self.contracts:>10,.0f} contracts  "
                f"${self.notional/1e6:>8,.1f}M   {'+'.join(self.funds)}")


def supply_map(books, underlying: str, spot: float, *, top: int = 10) -> list:
    """Aggregate SHORT option supply across funds, bucketed by (expiry, strike, kind).

    Returns the heaviest buckets first — this is the map of where the mechanical
    sellers actually are.
    """
    agg: dict = {}
    for b in books:
        for p in b.shorts(underlying):
            key = (p.expiry, p.strike, p.kind)
            cur = agg.get(key)
            if cur is None:
                cur = SupplyBucket(p.expiry, p.strike, p.kind, 0.0, 0.0, [])
                agg[key] = cur
            cur.contracts += abs(p.contracts)
            cur.notional += p.notional(spot)
            if b.fund not in cur.funds:
                cur.funds.append(b.fund)
    out = sorted(agg.values(), key=lambda s: s.notional, reverse=True)
    return out[:top] if top else out


def crowding_score(strike: float, kind: str, expiry_days: int, asof,
                   buckets, *, strike_tol: float = 0.01,
                   expiry_tol_days: int = 7) -> tuple[float, str]:
    """How crowded is OUR strike by fund supply? -> (0..1 share of mapped notional, note).

    Matches a bucket when the strike is within ``strike_tol`` (relative) and the
    expiry within ``expiry_tol_days``, so a 452.5 vs 452 strike or a neighbouring
    weekly still counts as the same supply.
    """
    if not buckets:
        return 0.0, "no fund supply data for this underlying"
    total = sum(b.notional for b in buckets) or 1.0
    a = _as_date(asof)
    hit = 0.0
    names: list[str] = []
    for b in buckets:
        if b.kind != kind:
            continue
        if strike <= 0 or abs(b.strike / strike - 1.0) > strike_tol:
            continue
        bd = _as_date(b.expiry)
        if a and bd is not None and abs((bd - a).days - expiry_days) > expiry_tol_days:
            continue
        hit += b.notional
        names += [f for f in b.funds if f not in names]
    share = hit / total
    if share <= 0:
        return 0.0, "no mapped fund supply at this strike/expiry"
    return share, (f"{'+'.join(names)} hold ${hit/1e6:,.0f}M of short {kind}s here "
                   f"({share:.0%} of the mapped fund supply) — selling more is joining "
                   f"their flow, not finding an untouched premium")


def roll_activity(prev: FundBook, curr: FundBook, underlying: str,
                  spot: float) -> dict:
    """What changed between two published books — the fund's ROLL, made visible.

    Returns closed / opened / net notional. These books roll on a schedule, so a
    large 'opened' bucket marks where fresh supply just landed.
    """
    def key(p):
        return (p.expiry, p.strike, p.kind)

    a = {key(p): abs(p.contracts) for p in prev.shorts(underlying)}
    b = {key(p): abs(p.contracts) for p in curr.shorts(underlying)}
    closed = {k: v for k, v in a.items() if b.get(k, 0.0) < v}
    opened = {k: v for k, v in b.items() if a.get(k, 0.0) < v}
    mult = 100.0
    return {
        "from": prev.asof, "to": curr.asof,
        "closed_contracts": sum(v - b.get(k, 0.0) for k, v in closed.items()),
        "opened_contracts": sum(v - a.get(k, 0.0) for k, v in opened.items()),
        "opened_strikes": sorted({k[1] for k in opened}),
        "closed_strikes": sorted({k[1] for k in closed}),
        "net_notional": (sum(v - a.get(k, 0.0) for k, v in opened.items())
                         - sum(v - b.get(k, 0.0) for k, v in closed.items())) * mult * spot,
    }


def iv_dent(chain, surface_fit, dte: int, *, min_dent: float = 0.005) -> list:
    """Strikes whose QUOTED implied vol sits materially BELOW the smooth SVI fit.

    A persistent dent is the fingerprint of concentrated supply: someone is selling
    enough of that strike to push its vol under the smile its neighbours describe.
    Returns [(strike, quoted_iv, fitted_iv, dent)] sorted by the deepest dent.

    Read it with care — a dent is equally consistent with a stale or wide quote, so
    it is a HINT to cross-check against the holdings map, never a signal on its own.
    """
    from engine.iv import implied_vol
    out = []
    T = dte / 365.0
    for q in chain.quotes:
        if q.expiry_days != dte:
            continue
        quoted = implied_vol(q.mid, chain.spot, q.strike, T, chain.r, chain.q, q.kind)
        if quoted is None or quoted <= 0:
            continue
        try:
            fitted = surface_fit.iv(q.strike)
        except (ValueError, ArithmeticError):
            continue
        dent = fitted - quoted
        if dent >= min_dent:
            out.append((q.strike, quoted, fitted, dent))
    return sorted(out, key=lambda r: r[3], reverse=True)
