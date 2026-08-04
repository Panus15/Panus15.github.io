"""Fetch daily closes for a basket of symbols — no key, no manual exports.

    python3 -m tools.fetch_prices --out sectors.csv          # the 11 sectors + SPY
    python3 -m tools.fetch_prices --symbols AAPL,MSFT,SPY --out tech.csv
    python3 -m tools.fetch_prices --out sectors.csv --cache raw/   # keep the raw files

This exists to delete a step. The sector-rotation study is the cheapest real
answer this repo can produce and the only thing standing between it and an answer
was twelve manual chart exports — a chore nobody does twice, which in practice
means the study never gets re-run on fresh data.

WHY STOOQ. It serves plain daily CSV over a stable URL with no key, no signup and
no per-minute quota, which makes it the one source that can sit in a script
somebody actually runs. It is a convenience feed, not a survivorship-safe research
database: it will not tell you about delistings, its adjustments are its own, and
it should never be the basis of a claim that money was made. For deciding whether
a rotation chart predicts anything on eleven large liquid ETFs, it is fine.

The fetch is deliberately unclever. One request per symbol, a small pause between
them, each response validated before it counts, and any symbol that fails is
reported and skipped rather than aborting the run — a basket where one ticker
404s should still produce eleven usable series and say which one is missing.

NETWORK. This will not run inside a sandbox that blocks outbound market-data
hosts; run it on a machine with open egress. `--base-url` points the whole thing
at a local directory or a mirror, which is also how it is tested offline.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.rotation import BENCHMARK, SECTOR_ETFS
from tools.merge_csv import merge, read_series, write_wide

STOOQ = "https://stooq.com/q/d/l/"
DEFAULT_BASKET = list(SECTOR_ETFS) + [BENCHMARK]


def stooq_url(symbol: str, *, base: str = STOOQ, interval: str = "d") -> str:
    """US tickers carry a '.us' suffix on stooq; a dotted symbol is left alone."""
    s = symbol.strip().lower()
    if "." not in s:
        s += ".us"
    sep = "" if base.endswith(("/", "=", "?", "&")) else "/"
    if base.startswith("file://") or not base.startswith("http"):
        return f"{base}{sep}{s}.csv"
    return f"{base}?s={s}&i={interval}"


def fetch_one(symbol: str, path: str, *, base: str = STOOQ,
              timeout: int = 30, force: bool = False) -> dict:
    """Download one symbol's daily CSV. Returns a status dict; never raises."""
    import urllib.request

    if os.path.exists(path) and not force:
        try:
            n = len(read_series(path))
            return {"symbol": symbol, "status": "cached", "rows": n, "path": path}
        except (ValueError, OSError):
            os.remove(path)                       # a bad cache file is worse than none

    url = stooq_url(symbol, base=base)
    tmp = path + ".part"
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "options-alpha-engine/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(tmp, "wb") as fh:
            fh.write(body)
    except Exception as e:                        # noqa: BLE001 — one bad ticker
        if os.path.exists(tmp):
            os.remove(tmp)
        return {"symbol": symbol, "status": f"FETCH FAILED: {type(e).__name__}: {e}"}

    # Validate before committing the name. Vendors answer a bad ticker with a 200
    # and the words "No data", which parses as an empty CSV and would otherwise
    # sit in the cache looking like a real file forever.
    try:
        rows = read_series(tmp)
    except (ValueError, OSError) as e:
        os.remove(tmp)
        return {"symbol": symbol, "status": f"UNUSABLE RESPONSE: {e}"}
    if len(rows) < 30:
        os.remove(tmp)
        return {"symbol": symbol,
                "status": f"only {len(rows)} row(s) returned — treated as a bad "
                          f"ticker rather than cached"}
    os.replace(tmp, path)
    return {"symbol": symbol, "status": "fetched", "rows": len(rows),
            "first": min(rows), "last": max(rows), "path": path}


def fetch_basket(symbols, cache_dir: str, *, base: str = STOOQ, pause: float = 0.3,
                 force: bool = False, timeout: int = 30) -> list:
    os.makedirs(cache_dir, exist_ok=True)
    out = []
    for i, sym in enumerate(symbols):
        if i and pause:
            time.sleep(pause)                     # be a polite client
        out.append(fetch_one(sym, os.path.join(cache_dir, f"{sym.upper()}.csv"),
                             base=base, timeout=timeout, force=force))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--symbols", default=",".join(DEFAULT_BASKET),
                    help="comma-separated; defaults to the 11 SPDR sectors + SPY")
    ap.add_argument("--out", default="sectors.csv")
    ap.add_argument("--cache", default="", help="keep the per-symbol files here")
    ap.add_argument("--base-url", default=STOOQ,
                    help="override the source (a local dir works, for testing)")
    ap.add_argument("--require", default=BENCHMARK,
                    help="symbols that must be present or the run fails")
    ap.add_argument("--pause", type=float, default=0.3)
    ap.add_argument("--force", action="store_true", help="ignore cached files")
    a = ap.parse_args(argv)

    syms = [s.strip().upper() for s in a.symbols.split(",") if s.strip()]
    cache = a.cache or os.path.join(os.path.dirname(a.out) or ".", "_prices")
    rows = fetch_basket(syms, cache, base=a.base_url, pause=a.pause, force=a.force)

    ok = []
    for r in rows:
        if r["status"] in ("fetched", "cached"):
            ok.append(r["path"])
            span = (f"  {r['first']} .. {r['last']}"
                    if "first" in r else "  (from cache)")
            print(f"  ok  {r['symbol']:6}{r['rows']:>7} rows{span}")
        else:
            print(f"  !!  {r['symbol']:6}{r['status']}")

    if not ok:
        print("\nnothing fetched. If every symbol failed with a network error, this "
              "machine cannot reach the vendor —\nrun it somewhere with open "
              "outbound access, or use --base-url against a local mirror.")
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
    if len(syms) - len(ok):
        print(f"  NOTE: {len(syms) - len(ok)} symbol(s) missing from the basket; the "
              f"chart will simply not show them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
