"""Merge per-symbol price exports into the one wide CSV the tools read.

    python3 -m tools.merge_csv --dir tradingview_exports --out sectors.csv
    python3 -m tools.merge_csv --out sectors.csv "AMEX_XLK, 1D.csv" "BATS_SPY, 1D.csv"

Chart tools export ONE FILE PER SYMBOL, long format, with the ticker buried in the
filename:

    AMEX_XLK, 1D.csv        time,open,high,low,close,Volume
                            2024-01-02T00:00:00-05:00,...,187.44,...

The rotation dashboard and backtests want ONE FILE, wide format, one column per
symbol:

    date,SPY,XLK,XLV,...
    2024-01-02,472.65,187.44,...

This does that conversion, and it is fussy about two things on purpose.

DATES ARE JOINED, NOT ZIPPED. Different symbols have different holidays, listing
dates and occasional missing bars — XLRE only starts in 2015, XLC in 2018. Lining
files up by ROW NUMBER instead of by date would silently pair one sector's Tuesday
with another's Wednesday, and every relative-strength number computed afterwards
would be quietly wrong in a way no test downstream could detect. So rows are
matched on the date, and only dates present in EVERY symbol survive.

WHAT WAS DROPPED IS REPORTED. The intersection can be much smaller than any input
file, and a merge that silently returns 900 rows from twelve 3,000-row exports
looks like success. The summary prints each symbol's own span and the joined span,
so a short one is visible before it becomes a mysteriously thin backtest.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: Column names various exporters use for the date and the close.
DATE_KEYS = ("time", "date", "datetime", "timestamp")
CLOSE_KEYS = ("close", "close/last", "adj close", "adjusted close", "last", "price")


def symbol_from_filename(path: str) -> str:
    """'AMEX_XLK, 1D.csv' -> 'XLK'.  'spy_daily.csv' -> 'SPY'.

    Exchange prefixes (AMEX_, BATS_, NASDAQ_, NYSE_, ...) and the timeframe suffix
    are stripped, because the caller should not have to rename twelve files by hand
    before anything works.
    """
    base = os.path.basename(path)
    base = re.sub(r"\.(csv|txt)$", "", base, flags=re.I)
    base = re.split(r"[,_\-\s]+", base)[0] if "," not in base else base.split(",")[0]
    base = re.sub(r"^(AMEX|BATS|NASDAQ|NYSE|ARCA|CBOE|OTC)[_:\-]", "", base,
                  flags=re.I)
    base = re.sub(r"[_\-]?(daily|1d|1day|eod)$", "", base, flags=re.I)
    return base.strip().upper() or "UNKNOWN"


def _norm_date(raw: str) -> str | None:
    """Anything an exporter emits -> 'YYYY-MM-DD', or None if it is not a date."""
    s = str(raw).strip().strip('"')
    if not s:
        return None
    # unix seconds / milliseconds
    if re.fullmatch(r"\d{9,13}", s):
        v = int(s)
        if v > 1e11:
            v //= 1000
        try:
            return datetime.datetime.utcfromtimestamp(v).date().isoformat()
        except (OverflowError, OSError, ValueError):
            return None
    # ISO date or datetime, with or without an offset
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    for fmt in ("%m/%d/%Y", "%d/%m/%Y", "%m/%d/%y", "%d-%b-%Y", "%b %d, %Y"):
        try:
            return datetime.datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def _pick(fieldnames, wanted) -> str | None:
    low = {(f or "").strip().lower(): f for f in fieldnames}
    for w in wanted:
        if w in low:
            return low[w]
    return None


def read_series(path: str) -> dict:
    """{'YYYY-MM-DD': close} from one export. Raises on a file with no close."""
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{path}: empty")
    fields = list(rows[0])
    dcol = _pick(fields, DATE_KEYS)
    ccol = _pick(fields, CLOSE_KEYS)
    if dcol is None or ccol is None:
        raise ValueError(f"{path}: need a date and a close column; found "
                         f"{', '.join(str(f) for f in fields)}")
    out: dict = {}
    for r in rows:
        d = _norm_date(r.get(dcol, ""))
        if d is None:
            continue
        raw = str(r.get(ccol, "")).replace(",", "").replace("$", "").strip()
        try:
            v = float(raw)
        except (TypeError, ValueError):
            continue
        if v > 0:
            out[d] = v            # a later duplicate date wins (adjusted reruns)
    if not out:
        raise ValueError(f"{path}: no usable rows (checked column {ccol!r})")
    return out


def merge(paths, *, require=None) -> tuple:
    """(dates, {symbol: [close, ...]}, report). Inner join on the date."""
    series: dict = {}
    spans: dict = {}
    for p in paths:
        sym = symbol_from_filename(p)
        s = read_series(p)
        if sym in series:
            raise ValueError(f"two files resolve to the symbol {sym!r}; rename one")
        series[sym] = s
        spans[sym] = (min(s), max(s), len(s))
    if not series:
        raise ValueError("no input files")
    missing = [s for s in (require or ()) if s not in series]
    if missing:
        raise ValueError(f"missing required symbol(s): {', '.join(missing)}")

    common = set.intersection(*(set(s) for s in series.values()))
    dates = sorted(common)
    if not dates:
        raise ValueError("no date is present in every file — check the timeframes "
                         "(all exports must be the same bar size, e.g. 1D)")
    cols = {sym: [series[sym][d] for d in dates] for sym in sorted(series)}

    lines = [f"  {'symbol':8}{'own rows':>10}{'own first':>13}{'own last':>13}"]
    for sym in sorted(spans):
        lo, hi, n = spans[sym]
        lines.append(f"  {sym:8}{n:>10}{lo:>13}{hi:>13}")
    shortest = min(spans, key=lambda s: spans[s][2])
    lines.append(f"\n  joined: {len(dates)} rows  {dates[0]} .. {dates[-1]}")
    dropped = max(n for _, _, n in spans.values()) - len(dates)
    if dropped:
        lines.append(f"  {dropped} row(s) dropped by the join — the binding symbol "
                     f"is {shortest} ({spans[shortest][2]} rows from "
                     f"{spans[shortest][0]})")
    return dates, cols, "\n".join(lines)


def write_wide(path: str, dates, cols) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        syms = sorted(cols)
        w.writerow(["date"] + syms)
        for i, d in enumerate(dates):
            w.writerow([d] + [f"{cols[s][i]:.6g}" for s in syms])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs="*", help="one export per symbol")
    ap.add_argument("--dir", help="take every .csv in this directory instead")
    ap.add_argument("--out", default="merged.csv")
    ap.add_argument("--require", default="",
                    help="comma-separated symbols that MUST be present (e.g. SPY)")
    a = ap.parse_args(argv)

    paths = list(a.files)
    if a.dir:
        paths += [os.path.join(a.dir, f) for f in sorted(os.listdir(a.dir))
                  if f.lower().endswith((".csv", ".txt"))]
    if not paths:
        ap.error("give some files, or --dir")

    req = [s.strip().upper() for s in a.require.split(",") if s.strip()]
    dates, cols, report = merge(paths, require=req)
    write_wide(a.out, dates, cols)
    print(report)
    print(f"\nwrote {a.out}: {len(dates)} rows x {len(cols)} symbols "
          f"({', '.join(sorted(cols))})")
    need = 2 * 63 + 5 + 12
    if len(dates) < need:
        print(f"  NOTE: the rotation chart needs >= {need} rows and the backtest "
              f"wants far more; {len(dates)} will not produce a usable result.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
