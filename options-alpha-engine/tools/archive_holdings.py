"""Archive the income ETFs' daily option books — the one dataset that expires.

    python3 -m tools.archive_holdings fetch --dir holdings          # run daily
    python3 -m tools.archive_holdings status --dir holdings
    python3 -m tools.archive_holdings supply --dir holdings --underlying QQQ

WHY THIS RUNS EVERY DAY AND WHY THAT MATTERS MORE THAN IT SOUNDS.

Covered-call ETFs publish their FULL holdings daily because the law requires it,
and then they overwrite the file. There is no archive, no history endpoint, no
paid tier that sells you last March. A point-in-time option book is available on
exactly one day, and after that it is gone.

That makes this the only piece of work in the repo where waiting costs something
permanent. ``engine/crowding_backtest.py`` is built, tested, and mutation-verified,
and it cannot answer its question — does selling where a fund already dominates
help or hurt? — without months of dated books. Every day this does not run is a
day that can never be recovered, no matter how much is spent later.

WHAT IT REFUSES TO DO. It will not save a file it cannot parse into a FundBook
with at least one option line, because a directory full of HTML error pages that
look like data is worse than an empty one — the gap is invisible until the study
runs and quietly drops those dates. It will not overwrite an existing dated file.
And ``supply_at_from_archive`` will not hand a backtest a book published after the
decision date, which is the specific mistake that would manufacture the effect the
study is trying to measure.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.fund_flow import FundBook, supply_map

#: Daily-holdings files for the funds whose books are dominated by option selling.
#: These URLs move; ``--sources my.json`` overrides the lot without editing code.
#: Funds we want but have no working direct-download URL for. Listed rather than
#: dropped so the gap is visible: JEPQ in particular is a headline target for the
#: crowding study, and silently omitting it would read as "we chose not to".
#: Both previously pointed at am.jpmorgan.com product PAGES, which return HTML;
#: that HTML was saved as .csv, parsed to zero option lines, and rejected every
#: day while the run still reported success. A named gap beats a fake source.
NEEDS_URL = "<no direct-download URL known — supply one with --sources>"

KNOWN_SOURCES = {
    "QQQI": "https://www.neosfunds.com/wp-content/fund-holdings/QQQI_holdings.csv",
    "SPYI": "https://www.neosfunds.com/wp-content/fund-holdings/SPYI_holdings.csv",
    "JEPI": NEEDS_URL,
    "JEPQ": NEEDS_URL,
}


def _today() -> str:
    return datetime.date.today().isoformat()


def fetch_one(fund: str, url: str, outdir: str, *, asof: str | None = None,
              timeout: int = 30, force: bool = False) -> dict:
    """Download one fund's holdings and archive it under ``outdir/FUND/DATE.csv``.

    Returns a status dict rather than raising: one fund's site being down must not
    stop the others from being archived that day.
    """
    import urllib.request

    if url == NEEDS_URL or not url:
        return {"fund": fund, "status": "NO URL CONFIGURED — supply one with "
                                        "--sources; this fund is not being archived"}

    asof = asof or _today()
    dest_dir = os.path.join(outdir, fund)
    os.makedirs(dest_dir, exist_ok=True)
    ext = ".json" if url.lower().endswith(".json") else ".csv"
    dest = os.path.join(dest_dir, asof + ext)
    if os.path.exists(dest) and not force:
        return {"fund": fund, "status": "already have it", "path": dest}

    # The temp name must KEEP the real extension: FundBook.from_file dispatches on
    # it, so a ".part" suffix would send every JSON holdings file down the CSV
    # parser and get it rejected as unparseable.
    tmp = os.path.join(dest_dir, f".partial-{asof}{ext}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "options-alpha-engine/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read()
        with open(tmp, "wb") as fh:
            fh.write(body)
    except Exception as e:                      # noqa: BLE001 - report, never abort the run
        if os.path.exists(tmp):
            os.remove(tmp)
        return {"fund": fund, "status": f"FETCH FAILED: {type(e).__name__}: {e}"}

    # Validate BEFORE committing the name. A saved file that does not parse is a
    # silent hole in the history — the study just drops that date and says nothing.
    try:
        book = FundBook.from_file(tmp, fund=fund, asof=asof)
        n_opts = len([p for p in book.positions if p.strike > 0])
        shorts = len(book.shorts())
    except Exception as e:                      # noqa: BLE001
        os.remove(tmp)
        return {"fund": fund, "status": f"UNPARSEABLE, not archived: "
                                        f"{type(e).__name__}: {e}"}
    if n_opts == 0:
        os.remove(tmp)
        return {"fund": fund, "status": "no option lines found, not archived "
                                        "(the URL probably returned a web page)"}

    os.replace(tmp, dest)
    return {"fund": fund, "status": "archived", "path": dest,
            "positions": n_opts, "shorts": shorts}


def fetch_all(outdir: str, sources: dict | None = None, **kw) -> list:
    sources = sources or KNOWN_SOURCES
    return [fetch_one(f, u, outdir, **kw) for f, u in sources.items()]


# --------------------------------------------------------------------------- #
# Reading the archive back — the part the backtest consumes
# --------------------------------------------------------------------------- #
def load_archive(outdir: str, *, funds=None) -> list:
    """Every archived book, as (date, FundBook), oldest first."""
    out = []
    if not os.path.isdir(outdir):
        return out
    for fund in sorted(os.listdir(outdir)):
        if funds and fund not in funds:
            continue
        fdir = os.path.join(outdir, fund)
        if not os.path.isdir(fdir):
            continue
        for name in sorted(os.listdir(fdir)):
            if not name.endswith((".csv", ".json")):
                continue
            asof = name.rsplit(".", 1)[0]
            try:
                out.append((asof, FundBook.from_file(os.path.join(fdir, name),
                                                     fund=fund, asof=asof)))
            except Exception:                   # noqa: BLE001
                continue
    return sorted(out, key=lambda x: x[0])


def coverage(outdir: str) -> dict:
    """What the archive actually contains — gaps included, because gaps matter."""
    books = load_archive(outdir)
    by_fund: dict = {}
    for asof, b in books:
        by_fund.setdefault(b.fund, []).append(asof)
    out = {}
    for fund, dates in by_fund.items():
        ds = sorted(set(dates))
        first, last = ds[0], ds[-1]
        d0 = datetime.date.fromisoformat(first)
        d1 = datetime.date.fromisoformat(last)
        span = (d1 - d0).days + 1
        weekdays = sum(1 for i in range(span)
                       if (d0 + datetime.timedelta(days=i)).weekday() < 5)
        out[fund] = {"days": len(ds), "first": first, "last": last,
                     "weekdays_in_span": weekdays,
                     "missing": max(0, weekdays - len(ds))}
    return out


def supply_at_from_archive(outdir: str, underlying: str, dates, prices, *,
                           funds=None, top: int = 12, lag_days: int = 1):
    """Build the point-in-time ``supply_at(t, trailing)`` crowding_backtest wants.

    ``dates`` are ISO dates aligned 1:1 with ``prices``. At bar t only books whose
    as-of date is at least ``lag_days`` BEFORE dates[t] are used — the published
    file carries a settlement lag, so treating today's book as knowable today would
    hand the study a small look-ahead exactly where the effect it hunts for lives.
    """
    archive = load_archive(outdir, funds=funds)
    if not archive:
        return None
    by_date: dict = {}
    for asof, b in archive:
        by_date.setdefault(asof, []).append(b)
    known = sorted(by_date)

    def supply_at(t, trailing=None):
        if t >= len(dates):
            return []
        try:
            cutoff = (datetime.date.fromisoformat(dates[t])
                      - datetime.timedelta(days=int(lag_days))).isoformat()
        except (TypeError, ValueError):
            return []
        usable = [d for d in known if d <= cutoff]
        if not usable:
            return []
        latest = usable[-1]
        spot = trailing[-1] if trailing else (prices[t] if t < len(prices) else 0.0)
        return supply_map(by_date[latest], underlying, spot, top=top)

    return supply_at


def _cmd_fetch(a):
    src = json.load(open(a.sources)) if a.sources else None
    rows = fetch_all(a.dir, sources=src, asof=a.asof, force=a.force)
    ok = 0
    for r in rows:
        mark = "ok " if r["status"] in ("archived", "already have it") else "!! "
        extra = (f"  {r.get('positions', 0)} option lines, {r.get('shorts', 0)} short"
                 if r["status"] == "archived" else "")
        print(f"  {mark}{r['fund']:6} {r['status']}{extra}")
        ok += r["status"] == "archived"
    print(f"\n{ok}/{len(rows)} newly archived into {a.dir}/")

    # Exit code policy. Holdings are published daily and overwritten, so a day
    # not captured is gone at any price — which makes a green exit on an empty
    # run the most expensive lie this tool can tell. But "already have it" is a
    # SUCCESS (a re-run on the same day, or a weekend with no new file), so the
    # line to draw is not "did anything change" but "does today's data exist".
    #
    # Partial success exits 0 deliberately. Two of the four funds have no URL
    # and have not for weeks; failing every day on a known gap trains the
    # operator to ignore the alarm, and then the day QQQI breaks goes unnoticed.
    held = sum(r["status"] in ("archived", "already have it") for r in rows)
    if held == 0:
        print("  NOTHING CAPTURED. Holdings are overwritten daily, so today is\n"
              "  gone unless this succeeds. URLs move: point --sources at a JSON\n"
              '  of {"FUND": "url"} rather than editing the module.')
        return 1
    if held < len(rows):
        print(f"  {len(rows) - held} fund(s) did not archive — see the !! lines "
              f"above. Exiting 0 because {held} did.")
    return 0


def _cmd_status(a):
    cov = coverage(a.dir)
    if not cov:
        print(f"{a.dir}/ is empty — nothing has been archived yet.\n"
              f"Until it is, engine/crowding_backtest.py has no data to run on, and\n"
              f"the days that pass in the meantime cannot be recovered later.")
        return 1
    print(f"  {'fund':8}{'days':>6}{'first':>13}{'last':>13}{'missing':>9}")
    for fund, c in sorted(cov.items()):
        print(f"  {fund:8}{c['days']:>6}{c['first']:>13}{c['last']:>13}"
              f"{c['missing']:>9}")
    best = max(c["days"] for c in cov.values())
    print(f"\n  crowding_backtest needs >= 30 paired dates to report anything; the\n"
          f"  deepest archive here has {best} day(s) of books.")
    return 0


def _cmd_supply(a):
    books = load_archive(a.dir)
    if not books:
        print(f"nothing archived in {a.dir}/")
        return 1
    asof, _ = books[-1]
    latest = [b for d, b in books if d == asof]
    buckets = supply_map(latest, a.underlying, a.spot, top=a.top)
    if not buckets:
        print(f"no short {a.underlying} option supply in the {asof} books")
        return 1
    total = sum(b.notional for b in buckets)
    print(f"Short-option supply in {a.underlying}, from the {asof} books "
          f"({'+'.join(sorted({b.fund for b in latest} if hasattr(latest[0],'fund') else []))})")
    for b in buckets:
        print(b.line(a.spot))
    print(f"  total mapped ${total/1e6:,.0f}M across {len(buckets)} buckets")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="download today's books (run once per trading day)")
    f.add_argument("--dir", default="holdings")
    f.add_argument("--sources", help='JSON of {"FUND": "url"} overriding the built-ins')
    f.add_argument("--asof", help="date to file it under (default: today)")
    f.add_argument("--force", action="store_true", help="re-download an existing date")
    f.set_defaults(func=_cmd_fetch)

    s = sub.add_parser("status", help="what the archive holds, and what is missing")
    s.add_argument("--dir", default="holdings")
    s.set_defaults(func=_cmd_status)

    p = sub.add_parser("supply", help="print the supply map from the newest books")
    p.add_argument("--dir", default="holdings")
    p.add_argument("--underlying", default="QQQ")
    p.add_argument("--spot", type=float, default=500.0)
    p.add_argument("--top", type=int, default=12)
    p.set_defaults(func=_cmd_supply)

    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
