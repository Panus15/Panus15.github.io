"""Loading real bars — the step where a study is usually lost before it starts.

Everything else in this engine assumes the bars are what they claim to be. This
module is where that assumption is checked, because FX vendor data breaks in
ways that do not raise an exception and do not look wrong on a chart. It loads a
CSV, and then it tries to find reasons not to trust it.

THE SEVEN THINGS IT LOOKS FOR, AND WHY EACH ONE MATTERS MORE THAN IT SOUNDS.

1. FROZEN QUOTES. The worst one, and the least known. When a feed goes quiet a
   vendor commonly fills the gap by repeating the last price, producing bars with
   o == h == l == c. They are not ticks; they are the absence of ticks, written
   as if it were data. Pattern detectors read those flat runs as consolidation,
   backtests fill inside them at no cost, and the equity curve improves. Nothing
   errors. `FROZEN_QUOTES` counts them.

2. OHLC VIOLATIONS. A bar whose high is below its open, or whose low is above
   its close, is impossible — and it appears in real vendor files, usually from a
   bad merge of bid and ask. `backtest.py` decides fills by comparing levels to
   `h` and `l`, so an impossible bar silently produces an impossible fill.

3. MISSING BARS, AND THE WEEKEND THAT IS NOT MISSING. FX runs Sunday evening to
   Friday evening, so the weekend hole is correct and an intraweek hole is not.
   They are counted separately, because a loader that flags 52 weekend gaps a
   year teaches you to ignore its warnings.

4. DUPLICATE TIMESTAMPS. Two bars at one instant means a merge went wrong. Left
   in, they double-count trades at that moment.

5. NON-MONOTONIC TIMESTAMPS. Out-of-order rows make "the next bar" meaningless,
   which is the single assumption every look-ahead guard in this engine rests on.

6. NON-POSITIVE PRICES. Zeros from a failed parse. They make `bp_per_pip` raise,
   but only after they have already been scanned for patterns.

7. WHICH SIDE THE PRICES ARE. Bid, ask, or mid — undetectable from the numbers
   and decisive for cost. `side` is a required argument with no default for that
   reason. A backtest on mid quietly hands you half the spread twice per trade.

WHAT IT WILL NOT DO. It will not silently repair anything. A loader that drops
bad rows and carries on produces a clean series with an unknown relationship to
the market. Problems are counted, located and reported, and what to do about
them is a decision with consequences, so it stays with you.
"""

from __future__ import annotations

import csv
import io
import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from models.patterns import Bar

# --- problem codes, named so they can be looked up and argued with ----------

FROZEN_QUOTES = "FROZEN_QUOTES"
OHLC_VIOLATION = "OHLC_VIOLATION"
MISSING_BARS = "MISSING_BARS"
DUPLICATE_TIMESTAMP = "DUPLICATE_TIMESTAMP"
OUT_OF_ORDER = "OUT_OF_ORDER"
NON_POSITIVE_PRICE = "NON_POSITIVE_PRICE"
IRREGULAR_INTERVAL = "IRREGULAR_INTERVAL"

#: Which side of the book the prices are. There is no default: the numbers
#: cannot tell you, and guessing mid is how a backtest gets the spread for free.
SIDES = ("bid", "ask", "mid")

#: A run of identical o/h/l/c at least this long is reported. One flat bar on an
#: illiquid hour is ordinary; three in a row on a major pair is a filled gap.
FROZEN_RUN = 3

#: Share of frozen bars above which the series is called unusable rather than
#: merely flagged. At one bar in twenty, a "consolidation" pattern is as likely
#: to be a dead feed as a market.
FROZEN_UNUSABLE_FRAC = 0.05

#: Column names understood without a mapping, lowercased.
_ALIASES = {
    "o": "o", "open": "o", "bidopen": "o", "askopen": "o",
    "h": "h", "high": "h", "bidhigh": "h", "askhigh": "h", "max": "h",
    "l": "l", "low": "l", "bidlow": "l", "asklow": "l", "min": "l",
    "c": "c", "close": "c", "bidclose": "c", "askclose": "c", "last": "c",
    "t": "t", "time": "t", "date": "t", "datetime": "t", "timestamp": "t",
    "date_time": "t", "gmt time": "t", "local time": "t",
}

#: Timestamp layouts tried in order. Vendors disagree about almost all of this.
_TIME_FORMATS = (
    "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d",
    "%Y/%m/%d %H:%M:%S", "%Y/%m/%d %H:%M", "%Y/%m/%d",
    "%d.%m.%Y %H:%M:%S", "%d.%m.%Y %H:%M",
    "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M",
    "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M",
    "%Y%m%d %H%M%S", "%Y%m%d %H%M", "%Y%m%d",
)


@dataclass(frozen=True)
class Problem:
    """One defect, with enough detail to go and look at it."""
    code: str
    count: int
    detail: str
    first_index: int = -1

    def __str__(self) -> str:
        where = f" (first at bar {self.first_index})" if self.first_index >= 0 else ""
        return f"{self.code} x{self.count}: {self.detail}{where}"


@dataclass
class Series:
    """Bars, when they happened, and everything wrong with them."""

    pair: str
    side: str
    bars: list = field(default_factory=list)
    times: list = field(default_factory=list)
    problems: list = field(default_factory=list)
    interval: object = None          # the modal spacing, or None if irregular
    source: str = ""

    @property
    def n(self) -> int:
        return len(self.bars)

    @property
    def codes(self) -> list:
        return [p.code for p in self.problems]

    def count(self, code: str) -> int:
        return sum(p.count for p in self.problems if p.code == code)

    @property
    def usable(self) -> bool:
        """False when a defect makes the ENGINE's assumptions wrong, rather than
        merely making the data untidy.

        Missing bars and irregular intervals do not appear here: they are facts
        about the market's calendar as often as they are faults, and the caller
        is better placed to judge. The ones listed break look-ahead ordering,
        fill logic, or the arithmetic itself.
        """
        if not self.bars:
            return False
        fatal = (OHLC_VIOLATION, DUPLICATE_TIMESTAMP, OUT_OF_ORDER,
                 NON_POSITIVE_PRICE)
        if any(c in self.codes for c in fatal):
            return False
        return self.count(FROZEN_QUOTES) / self.n <= FROZEN_UNUSABLE_FRAC

    def report(self) -> str:
        head = (f"{self.source or 'series'}: {self.n:,} bars of {self.pair} "
                f"at the {self.side.upper()}")
        if self.times:
            head += f", {self.times[0]:%Y-%m-%d} to {self.times[-1]:%Y-%m-%d}"
        lines = [head]
        if self.interval is not None:
            lines.append(f"  interval       {self.interval}")
        if not self.problems:
            lines.append("  no defects found — which is not the same as none")
            return "\n".join(lines)
        for p in self.problems:
            lines.append(f"  {'FATAL' if not self.usable else 'note '} {p}")
        if not self.usable:
            lines.append("  NOT USABLE. Nothing here was repaired: a loader that "
                         "drops bad rows hands you a clean series with an unknown "
                         "relationship to the market")
        return "\n".join(lines)


def _clean_time(raw: str) -> str:
    """Normalise the spellings vendors differ on, before any format is tried."""
    s = raw.strip().replace("T", " ")
    if s.endswith("Z"):
        s = s[:-1]
    head, _, tail = s.rpartition(".")
    if head and tail.isdigit() and len(tail) in (3, 6):
        s = head                          # drop sub-second precision
    return s


def _resolve_format(samples: list, explicit: str = None) -> str:
    """Decide ONE layout for the whole column, or refuse to decide.

    03/04/2024 is the third of April in most of the world and the fourth of
    March in the United States, and no row containing it can say which. Trying
    formats in a fixed order would silently pick one, shifting every bar in the
    file by up to eleven months along with every pattern found in it — with no
    error and a chart that still looks like EURUSD.

    So the format is resolved against the WHOLE column. A single row with a day
    past the twelfth settles it. If nothing in the file settles it and two
    layouts disagree, this raises and asks, because the alternative is a guess
    the caller never learns was made.
    """
    if explicit:
        return explicit
    if not samples:
        raise ValueError("no timestamps to resolve a format from")
    live = []
    for fmt in _TIME_FORMATS:
        try:
            datetime.strptime(samples[0], fmt)
        except ValueError:
            continue
        live.append(fmt)
    if not live:
        raise ValueError(
            f"unrecognised timestamp {samples[0]!r}; pass time_format=")
    if len(live) == 1:
        return live[0]
    # several layouts fit the first row: let the rest of the file choose
    disagreement = None
    for s in samples:
        parsed, survivors = {}, []
        for fmt in live:
            try:
                parsed[fmt] = datetime.strptime(s, fmt)
            except ValueError:
                continue
            survivors.append(fmt)
        if not survivors:
            raise ValueError(f"timestamp {s!r} does not match {live[0]!r}")
        if len(survivors) < len(live):
            live = survivors              # this row ruled some out: decided
            if len(live) == 1:
                return live[0]
        elif disagreement is None and len(set(parsed.values())) > 1:
            disagreement = (s, dict(parsed))
    if disagreement:
        s, parsed = disagreement
        shown = ", ".join(f"{f} -> {d:%Y-%m-%d}" for f, d in list(parsed.items())[:2])
        raise ValueError(
            f"timestamp layout is ambiguous and nothing in the file settles it: "
            f"{s!r} reads as {shown}. Every bar would shift if this were guessed "
            f"wrong, so pass time_format= rather than let it be chosen for you")
    return live[0]


def _rows(text: str, columns=None, time_format=None):
    """Yield (datetime, o, h, l, c) from CSV text, header or headerless."""
    sample = text[:4096]
    try:
        delim = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        delim = ","
    reader = csv.reader(io.StringIO(text), delimiter=delim)
    rows = [r for r in reader if r and any(f.strip() for f in r)]
    if not rows:
        return

    if columns is None:
        head = [f.strip().lower().lstrip("﻿") for f in rows[0]]
        mapped = {_ALIASES[h]: i for i, h in enumerate(head) if h in _ALIASES}
        if {"o", "h", "l", "c"} <= set(mapped):
            columns, rows = mapped, rows[1:]
            if "t" not in columns:      # date and time in two separate columns
                pass
        else:
            # headerless: the near-universal vendor layout is time,O,H,L,C[,vol]
            columns = {"t": 0, "o": 1, "h": 2, "l": 3, "c": 4}
    def stamp(r):
        ts = r[columns["t"]]
        # HistData and friends split the stamp across two columns
        if columns["t"] + 1 < len(r) and ":" in r[columns["t"] + 1] \
                and columns.get("o") != columns["t"] + 1:
            ts = f"{ts} {r[columns['t'] + 1]}"
        return _clean_time(ts)

    fmt = None
    if "t" in columns:
        try:
            fmt = _resolve_format([stamp(r) for r in rows], time_format)
        except IndexError as e:
            raise ValueError(f"no timestamp column at index "
                             f"{columns['t']}") from e
    for i, r in enumerate(rows):
        try:
            t = (datetime.strptime(stamp(r), fmt) if fmt
                 else datetime.min + timedelta(minutes=i))
            vals = [float(r[columns[k]]) for k in ("o", "h", "l", "c")]
        except (IndexError, ValueError) as e:
            raise ValueError(f"row {i + 1} could not be read: {r} ({e})") from e
        yield (t, *vals)


def inspect(bars: list, times: list) -> tuple:
    """Find every defect. Returns (problems, modal_interval).

    Separated from loading so it can be run over bars from any source — a
    vendor API, a database, a broker export — not only a CSV.
    """
    problems = []
    n = len(bars)
    if n == 0:
        return problems, None

    # --- impossible bars ---------------------------------------------------
    bad = [i for i, b in enumerate(bars)
           if not (b.h >= max(b.o, b.c) and b.l <= min(b.o, b.c) and b.h >= b.l)]
    if bad:
        problems.append(Problem(
            OHLC_VIOLATION, len(bad),
            "high below the body or low above it — impossible bars, usually a "
            "bad bid/ask merge. backtest.py decides fills against h and l, so "
            "these produce fills that could not have happened", bad[0]))

    nonpos = [i for i, b in enumerate(bars)
              if min(b.o, b.h, b.l, b.c) <= 0 or any(
                  math.isnan(x) for x in (b.o, b.h, b.l, b.c))]
    if nonpos:
        problems.append(Problem(
            NON_POSITIVE_PRICE, len(nonpos),
            "zero, negative or NaN prices, normally a failed parse rather than "
            "a market event", nonpos[0]))

    # --- frozen quotes: the absence of ticks, written as data --------------
    frozen, run, start = [], 0, 0
    for i, b in enumerate(bars):
        if b.o == b.h == b.l == b.c:
            if run == 0:
                start = i
            run += 1
        else:
            if run >= FROZEN_RUN:
                frozen.extend(range(start, i))
            run = 0
    if run >= FROZEN_RUN:
        frozen.extend(range(start, n))
    if frozen:
        problems.append(Problem(
            FROZEN_QUOTES, len(frozen),
            f"bars with o==h==l==c in runs of {FROZEN_RUN}+ "
            f"({len(frozen) / n:.1%} of the series). These are a dead feed "
            f"written as a flat market: detectors read them as consolidation "
            f"and backtests fill inside them for nothing", frozen[0]))

    # --- the clock ---------------------------------------------------------
    interval = None
    if len(times) == n and n > 1:
        dupes = [i for i in range(1, n) if times[i] == times[i - 1]]
        if dupes:
            problems.append(Problem(
                DUPLICATE_TIMESTAMP, len(dupes),
                "two bars at one instant — a merge went wrong, and left in they "
                "double-count every trade at that moment", dupes[0]))
        back = [i for i in range(1, n) if times[i] < times[i - 1]]
        if back:
            problems.append(Problem(
                OUT_OF_ORDER, len(back),
                "timestamps run backwards, which makes \"the next bar\" "
                "meaningless — the assumption every look-ahead guard rests on",
                back[0]))

        deltas = [times[i] - times[i - 1] for i in range(1, n)]
        interval = Counter(deltas).most_common(1)[0][0]
        if interval.total_seconds() > 0:
            weekend, intraweek = 0, []
            for i, d in enumerate(deltas, start=1):
                if d <= interval:
                    continue
                # a hole that covers a Saturday is the market being shut
                spans_saturday = any(
                    (times[i - 1] + timedelta(days=k)).weekday() == 5
                    for k in range(d.days + 1))
                if spans_saturday:
                    weekend += 1
                else:
                    intraweek.append(i)
            if intraweek:
                problems.append(Problem(
                    MISSING_BARS, len(intraweek),
                    f"holes in the series that do NOT span a weekend, against "
                    f"{weekend} that do. The weekend ones are the market being "
                    f"shut; these are data you do not have, and a backtest will "
                    f"treat the bars either side as adjacent", intraweek[0]))
            odd = sum(1 for d in deltas if d != interval)
            if odd > len(deltas) * 0.5:
                problems.append(Problem(
                    IRREGULAR_INTERVAL, odd,
                    f"more than half the gaps differ from the modal {interval}, "
                    f"so this is not a fixed-interval series. Financing is "
                    f"charged per calendar night, and backtest.py counts nights "
                    f"in BARS — on an irregular series that count is wrong"))
    return problems, interval


def load_csv(path: str, pair: str, *, side: str, columns=None,
             time_format: str = None, source: str = "") -> Series:
    """Read a CSV of OHLC bars and report everything wrong with it.

    ``side`` is required and must be one of `SIDES`. The numbers cannot tell you
    which side of the book they came from, and assuming mid gives a backtest
    half the spread twice per trade — so it is asked for rather than defaulted.

    ``columns`` overrides detection, e.g. ``{"t": 0, "o": 1, "h": 2, "l": 3,
    "c": 4}``. Without it, a header row is matched against common vendor names
    and a headerless file is read as time,O,H,L,C.

    ``time_format`` is a strptime layout. It is only needed when the file is
    genuinely ambiguous — a slash-dated file where no row has a day past the
    twelfth — and in that case it is REQUIRED rather than guessed.
    """
    if side not in SIDES:
        raise ValueError(
            f"side must be one of {SIDES}, got {side!r}. It cannot be inferred "
            f"from the prices, and it decides what every cost figure means")
    with io.open(path, encoding="utf-8-sig", errors="replace") as fh:
        text = fh.read()
    bars, times = [], []
    for t, o, h, l, c in _rows(text, columns, time_format):
        times.append(t)
        bars.append(Bar(o, h, l, c))
    problems, interval = inspect(bars, times)
    return Series(pair=pair, side=side, bars=bars, times=times,
                  problems=problems, interval=interval,
                  source=source or path.rsplit("/", 1)[-1])
