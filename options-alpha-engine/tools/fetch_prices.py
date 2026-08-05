"""Fetch daily closes for a basket of symbols — no key, no manual exports.

    python3 -m tools.fetch_prices --out sectors.csv          # the 11 sectors + SPY
    python3 -m tools.fetch_prices --out tech.csv --symbols AAPL,MSFT,SPY
    python3 -m tools.fetch_prices --out sectors.csv --source stooq   # force one

This exists to delete a step. The sector-rotation study is the cheapest real
answer this repo can produce and the only thing standing between it and an answer
was twelve manual chart exports — a chore nobody does twice, which in practice
means the study never gets re-run on fresh data.

WHY THERE IS MORE THAN ONE SOURCE. A keyless feed is a courtesy, and courtesies
get withdrawn: stooq put a JavaScript bot-check in front of its CSV endpoint and
every request started returning an HTML page that says "This site requires
JavaScript" with a 200 status. Nothing was broken, nothing errored, and a fetcher
that trusted the status code would have written twelve HTML files into the cache
and reported success. So sources are a LIST, each one is validated by parsing what
came back rather than by its status code, and ``--source auto`` walks them until
one actually yields prices.

These are convenience feeds, not survivorship-safe research databases. They will
not tell you about delistings and their adjustments are their own. Fine for
deciding whether a rotation chart predicts anything on eleven large liquid ETFs;
not the basis for a claim that money was made.

NETWORK. This will not run inside a sandbox that blocks outbound market-data
hosts; run it on a machine with open egress. ``--base-url`` points a source at a
local directory or mirror, which is also how it is tested offline.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.rotation import BENCHMARK, SECTOR_ETFS
from tools.merge_csv import merge, read_series, write_wide

STOOQ = "https://stooq.com/q/d/l/"
YAHOO = "https://query1.finance.yahoo.com/v8/finance/chart/"
DEFAULT_BASKET = list(SECTOR_ETFS) + [BENCHMARK]

#: Tried in this order by --source auto. Yahoo first because as of writing stooq
#: is behind a JS challenge; if that reverses, reorder here and nothing else.
SOURCE_ORDER = ("yahoo", "stooq")

#: A plain urllib UA gets refused or challenged by most vendors.
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/122.0 Safari/537.36")


def stooq_url(symbol: str, *, base: str = STOOQ, interval: str = "d") -> str:
    """US tickers carry a '.us' suffix on stooq; a dotted symbol is left alone."""
    s = symbol.strip().lower()
    if "." not in s:
        s += ".us"
    sep = "" if base.endswith(("/", "=", "?", "&")) else "/"
    if base.startswith("file://") or not base.startswith("http"):
        return f"{base}{sep}{s}.csv"
    return f"{base}?s={s}&i={interval}"


def yahoo_url(symbol: str, *, base: str = YAHOO, range_: str = "10y") -> str:
    s = symbol.strip().upper()
    if base.startswith("file://") or not base.startswith("http"):
        sep = "" if base.endswith("/") else "/"
        return f"{base}{sep}{s}.json"
    sep = "" if base.endswith("/") else "/"
    return f"{base}{sep}{s}?interval=1d&range={range_}"


def parse_yahoo(raw: bytes) -> dict:
    """{'YYYY-MM-DD': close} from a Yahoo chart response. Raises on anything else."""
    import datetime

    payload = json.loads(raw.decode("utf-8", "replace"))
    chart = payload.get("chart") or {}
    if chart.get("error"):
        raise ValueError(f"vendor error: {chart['error']}")
    results = chart.get("result") or []
    if not results:
        raise ValueError("no result block (unknown or delisted symbol)")
    r = results[0]
    stamps = r.get("timestamp") or []
    quotes = (r.get("indicators") or {}).get("quote") or [{}]
    closes = quotes[0].get("close") or []
    # prefer the split/dividend-adjusted series when the vendor supplies it
    adj = (r.get("indicators") or {}).get("adjclose") or []
    if adj and adj[0].get("adjclose"):
        closes = adj[0]["adjclose"]
    if not stamps or not closes:
        raise ValueError("response carried no timestamps or closes")
    out: dict = {}
    for ts, c in zip(stamps, closes):
        if c is None:
            continue                          # vendor gaps are holes, not zeros
        try:
            d = datetime.datetime.utcfromtimestamp(int(ts)).date().isoformat()
        except (OverflowError, OSError, ValueError):
            continue
        if c > 0:
            out[d] = float(c)
    if not out:
        raise ValueError("every row was null")
    return out


def _short(msg: str, n: int = 160) -> str:
    """One line, truncated. A bot-check page is 40kB and must not fill a terminal."""
    s = " ".join(str(msg).split())
    return s if len(s) <= n else s[:n] + " ..."


SOURCES = {
    "stooq": {"url": stooq_url, "ext": ".csv", "parse": None, "base": STOOQ},
    "yahoo": {"url": yahoo_url, "ext": ".json", "parse": parse_yahoo, "base": YAHOO},
}


def _download(url: str, timeout: int) -> bytes:
    import urllib.request
    req = urllib.request.Request(url, headers={
        "User-Agent": UA, "Accept": "*/*", "Accept-Language": "en-US,en;q=0.9"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def fetch_one(symbol: str, path: str, *, source: str = "yahoo",
              base: str | None = None, timeout: int = 30,
              force: bool = False) -> dict:
    """Download one symbol from ONE source. Returns a status dict; never raises."""
    spec = SOURCES[source]
    if os.path.exists(path) and not force:
        try:
            return {"symbol": symbol, "status": "cached", "source": source,
                    "rows": len(read_series(path)), "path": path}
        except (ValueError, OSError):
            os.remove(path)                   # a bad cache file is worse than none

    url = spec["url"](symbol, base=base or spec["base"])
    try:
        body = _download(url, timeout)
    except Exception as e:                    # noqa: BLE001 — one bad ticker
        return {"symbol": symbol, "source": source,
                "status": f"FETCH FAILED: {type(e).__name__}: {_short(e, 80)}"}

    # Validate by PARSING, never by status code. A vendor that has put a bot-check
    # in front of the data answers 200 with an HTML page, and a fetcher that trusts
    # the status writes that page into the cache and calls it success.
    try:
        if spec["parse"]:
            rows = spec["parse"](body)
        else:
            tmp = path + ".probe"
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(tmp, "wb") as fh:
                fh.write(body)
            try:
                rows = read_series(tmp)
            finally:
                os.remove(tmp)
    except Exception as e:                    # noqa: BLE001
        head = body[:200].decode("utf-8", "replace").lower()
        if "<html" in head or "<!doctype" in head or "javascript" in head:
            return {"symbol": symbol, "source": source,
                    "status": "BLOCKED: the vendor served a web page, not data "
                              "(bot-check or rate limit)"}
        return {"symbol": symbol, "source": source,
                "status": f"UNUSABLE RESPONSE: {_short(e)}"}

    if len(rows) < 30:
        return {"symbol": symbol, "source": source,
                "status": f"only {len(rows)} row(s) returned — treated as a bad "
                          f"ticker rather than cached"}

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".part"
    with open(tmp, "w", newline="") as fh:
        fh.write("date,close\n")
        for d in sorted(rows):
            fh.write(f"{d},{rows[d]:.6f}\n")
    os.replace(tmp, path)
    return {"symbol": symbol, "status": "fetched", "source": source,
            "rows": len(rows), "first": min(rows), "last": max(rows), "path": path}


def fetch_one_auto(symbol: str, path: str, *, sources=SOURCE_ORDER,
                   base: str | None = None, **kw) -> dict:
    """Try each source until one yields prices. Reports every source that failed.

    A single free feed can be withdrawn or challenged at any time, so a fetcher
    with one hardcoded vendor is a fetcher that stops working without warning.
    """
    tried = []
    for src in sources:
        r = fetch_one(symbol, path, source=src, base=base, **kw)
        if r["status"] in ("fetched", "cached"):
            if tried:
                r["status"] += f"  (after {', '.join(tried)} failed)"
            return r
        tried.append(f"{src}: {r['status'].split(':')[0]}")
    return {"symbol": symbol, "status": "ALL SOURCES FAILED — " + "; ".join(tried)}


def fetch_basket(symbols, cache_dir: str, *, sources=SOURCE_ORDER,
                 base: str | None = None, pause: float = 0.3,
                 force: bool = False, timeout: int = 30) -> list:
    os.makedirs(cache_dir, exist_ok=True)
    out = []
    for i, sym in enumerate(symbols):
        if i and pause:
            time.sleep(pause)                 # be a polite client
        out.append(fetch_one_auto(sym, os.path.join(cache_dir, f"{sym.upper()}.csv"),
                                  sources=sources, base=base, timeout=timeout,
                                  force=force))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--symbols", default=",".join(DEFAULT_BASKET),
                    help="comma-separated; defaults to the 11 SPDR sectors + SPY")
    ap.add_argument("--out", default="sectors.csv")
    ap.add_argument("--cache", default="", help="keep the per-symbol files here")
    ap.add_argument("--source", default="auto",
                    choices=["auto"] + list(SOURCES),
                    help="auto tries each in turn: " + ", ".join(SOURCE_ORDER))
    ap.add_argument("--base-url", default="",
                    help="override the source's base (a local dir works)")
    ap.add_argument("--require", default=BENCHMARK,
                    help="symbols that must be present or the run fails")
    ap.add_argument("--pause", type=float, default=0.3)
    ap.add_argument("--force", action="store_true", help="ignore cached files")
    a = ap.parse_args(argv)

    syms = [s.strip().upper() for s in a.symbols.split(",") if s.strip()]
    srcs = SOURCE_ORDER if a.source == "auto" else (a.source,)
    cache = a.cache or os.path.join(os.path.dirname(a.out) or ".", "_prices")
    rows = fetch_basket(syms, cache, sources=srcs, base=a.base_url or None,
                        pause=a.pause, force=a.force)

    ok = []
    for r in rows:
        if r["status"].startswith(("fetched", "cached")):
            ok.append(r["path"])
            span = f"  {r['first']} .. {r['last']}" if "first" in r else ""
            print(f"  ok  {r['symbol']:6}{r['rows']:>7} rows  "
                  f"[{r.get('source', '?')}]{span}")
        else:
            print(f"  !!  {r['symbol']:6}{_short(r['status'], 110)}")

    if not ok:
        blocked = sum("BLOCKED" in r["status"] or "web page" in r["status"]
                      for r in rows)
        net = sum("FETCH FAILED" in r["status"] for r in rows)
        print("\nnothing fetched.")
        if blocked:
            print("  Every source answered with a web page instead of data — that is\n"
                  "  a bot-check or rate limit, not a bug here. Wait a few minutes,\n"
                  "  try --source yahoo or --source stooq explicitly, or fall back to\n"
                  "  chart exports:  python3 -m tools.merge_csv --dir exports "
                  "--out sectors.csv")
        elif net:
            print("  Every symbol failed with a network error, so this machine cannot\n"
                  "  reach the vendor — run it somewhere with open outbound access,\n"
                  "  or use --base-url against a local mirror.")
        return 1

    req = [s.strip().upper() for s in a.require.split(",") if s.strip()]
    try:
        dates, cols, report = merge(ok, require=req)
    except ValueError as e:
        print(f"\nmerge failed: {e}")
        return 1
    write_wide(a.out, dates, cols)
    print("\n" + report)
    print(f"\nwrote {a.out}: {len(dates)} rows x {len(cols)} symbols")
    print(f"  next:  python3 -m tools.rotation_dashboard --csv {a.out} "
          f"--out rotation.html")
    missing = len(syms) - len(ok)
    if missing:
        print(f"  NOTE: {missing} symbol(s) missing from the basket; the chart will "
              f"simply not show them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
