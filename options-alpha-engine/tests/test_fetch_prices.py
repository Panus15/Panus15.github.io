"""Tests for the price fetcher (tools/fetch_prices.py).

No network: `--base-url` points a source at a local directory, which is both how
the tool is tested and how it can be run against a mirror.

The behaviours worth defending all come from one real incident. Stooq put a
JavaScript bot-check in front of its CSV endpoint, and every request began
returning an HTML page saying "This site requires JavaScript" — with HTTP 200.
Nothing errored. A fetcher that trusted the status code would have written twelve
HTML files into the cache and reported success, and the cache would have looked
real forever. So: responses are validated by PARSING, a page-instead-of-data is
named as a bot-check rather than as a mystery, one dead source falls through to
the next, and one dead ticker never aborts the basket.

Run: python3 tests/test_fetch_prices.py
"""

import datetime
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.fetch_prices import (DEFAULT_BASKET, SOURCE_ORDER, fetch_basket,
                                fetch_one, fetch_one_auto, main, parse_yahoo,
                                stooq_url, yahoo_url)

BOT_CHECK = (b'<!DOCTYPE html><html><head><meta charset="utf-8"><meta name="robots" '
             b'content="noindex, nofollow"></head><body><noscript>This site '
             b'requires JavaScript. Please enable JavaScript and reload.</noscript>'
             b'<script nonce="fiM_Sya2njLgxg4PR3Tr2Q"></script></body></html>')


def _csv(n=200, start=100.0, first="2024-01-01"):
    d0 = datetime.date.fromisoformat(first)
    out = ["Date,Open,High,Low,Close,Volume"]
    px = start
    for i in range(n):
        px *= 1.001
        out.append(f"{(d0 + datetime.timedelta(days=i)).isoformat()},"
                   f"{px:.4f},{px:.4f},{px:.4f},{px:.4f},1000")
    return "\n".join(out) + "\n"


def _yahoo(n=200, start=100.0, first="2024-01-01", adjusted=True):
    d0 = datetime.datetime.fromisoformat(first + "T00:00:00")
    stamps, closes, px = [], [], start
    for i in range(n):
        px *= 1.001
        stamps.append(int((d0 + datetime.timedelta(days=i)).timestamp()))
        closes.append(round(px, 4))
    ind = {"quote": [{"close": closes}]}
    if adjusted:
        ind["adjclose"] = [{"adjclose": closes}]
    return json.dumps({"chart": {"result": [
        {"meta": {}, "timestamp": stamps, "indicators": ind}], "error": None}}).encode()


def _mirror(d, symbols, kind="yahoo", **kw):
    m = os.path.join(d, "mirror_" + kind)
    os.makedirs(m, exist_ok=True)
    for s in symbols:
        if kind == "yahoo":
            with open(os.path.join(m, f"{s.upper()}.json"), "wb") as fh:
                fh.write(_yahoo(**kw))
        else:
            with open(os.path.join(m, f"{s.lower()}.us.csv"), "w") as fh:
                fh.write(_csv(**kw))
    return "file://" + m


# --------------------------------------------------------------------------
# 1. URLs and parsing
# --------------------------------------------------------------------------

def test_url_shapes():
    assert stooq_url("SPY") == "https://stooq.com/q/d/l/?s=spy.us&i=d"
    assert stooq_url("BRK.B").count(".us") == 0
    assert stooq_url("SPY", base="file:///tmp/m") == "file:///tmp/m/spy.us.csv"
    assert yahoo_url("spy").endswith("/SPY?interval=1d&range=10y")
    assert yahoo_url("SPY", base="file:///tmp/m") == "file:///tmp/m/SPY.json"


def test_yahoo_parsing_including_the_awkward_cases():
    rows = parse_yahoo(_yahoo(n=50))
    assert len(rows) == 50 and all(v > 0 for v in rows.values())
    assert min(rows) == "2024-01-01"

    # nulls are HOLES, not zeros — a vendor gap must not become a price of 0
    holed = json.loads(_yahoo(n=5).decode())
    holed["chart"]["result"][0]["indicators"]["quote"][0]["close"][2] = None
    holed["chart"]["result"][0]["indicators"].pop("adjclose")
    assert len(parse_yahoo(json.dumps(holed).encode())) == 4

    for bad, why in (
            (b'{"chart":{"result":[],"error":null}}', "unknown symbol"),
            (b'{"chart":{"result":null,"error":{"code":"Not Found"}}}', "vendor error"),
            (b'{"chart":{"result":[{"timestamp":[],"indicators":{}}]}}', "empty"),
            (BOT_CHECK, "an HTML page")):
        try:
            parse_yahoo(bad)
        except (ValueError, json.JSONDecodeError):
            continue
        raise AssertionError(f"{why} should not parse as prices")


# --------------------------------------------------------------------------
# 2. the incident: a 200 that is a web page, not data
# --------------------------------------------------------------------------

def test_a_bot_check_page_is_named_as_such_and_never_cached():
    """The real failure: HTTP 200, an HTML page, and a cache that looks fine."""
    d = tempfile.mkdtemp()
    try:
        for kind, name in (("stooq", "zzzz.us.csv"), ("yahoo", "ZZZZ.json")):
            m = os.path.join(d, "m_" + kind)
            os.makedirs(m, exist_ok=True)
            with open(os.path.join(m, name), "wb") as fh:
                fh.write(BOT_CHECK)
            p = os.path.join(d, "c", kind, "ZZZZ.csv")
            r = fetch_one("ZZZZ", p, source=kind, base="file://" + m)
            assert "BLOCKED" in r["status"], (kind, r)
            assert "bot-check" in r["status"]
            assert not os.path.exists(p)
            assert not os.path.exists(p + ".part")
            assert not os.path.exists(p + ".probe")
            # and the 40kB page must not be dumped into the message
            assert len(r["status"]) < 200, len(r["status"])
    finally:
        shutil.rmtree(d)


def test_a_long_non_html_error_is_truncated_to_one_line():
    """A vendor can fail verbosely without being a web page.

    The BLOCKED branch returns a fixed sentence, so it can never be long. This
    covers the other path: a parse error whose message carries a chunk of the
    response. Untruncated, a single bad symbol floods the terminal and the eleven
    good ones scroll out of sight.
    """
    d = tempfile.mkdtemp()
    try:
        m = os.path.join(d, "m")
        os.makedirs(m)
        # This is what actually reached the terminal: read_series names every
        # column it found, and when the "CSV" is one enormous line that list is
        # thousands of characters long.
        junk = ",".join(f"col{i}_{'x' * 40}" for i in range(200))
        with open(os.path.join(m, "zzzz.us.csv"), "w") as fh:
            fh.write(junk + "\n" + ",".join("0" for _ in range(200)) + "\n")
        r = fetch_one("ZZZZ", os.path.join(d, "c", "ZZZZ.csv"), source="stooq",
                      base="file://" + m)
        assert "UNUSABLE" in r["status"], r
        assert len(r["status"]) < 250, (
            f"the message is {len(r['status'])} chars; one bad symbol would "
            f"scroll the eleven good ones off the screen")
        assert r["status"].endswith("..."), "a truncated message should say so"
        assert "\n" not in r["status"]
    finally:
        shutil.rmtree(d)


def test_nonpositive_prices_are_dropped():
    """A zero or negative close is not a price; it would poison every ratio."""
    payload = json.loads(_yahoo(n=6).decode())
    ind = payload["chart"]["result"][0]["indicators"]
    ind.pop("adjclose")
    ind["quote"][0]["close"][1] = 0.0
    ind["quote"][0]["close"][3] = -5.0
    rows = parse_yahoo(json.dumps(payload).encode())
    assert len(rows) == 4, rows
    assert all(v > 0 for v in rows.values())


def test_one_dead_source_falls_through_to_the_next():
    """A keyless feed is a courtesy and courtesies get withdrawn."""
    d = tempfile.mkdtemp()
    try:
        blocked = os.path.join(d, "m_yahoo")
        os.makedirs(blocked)
        with open(os.path.join(blocked, "SPY.json"), "wb") as fh:
            fh.write(BOT_CHECK)                       # yahoo is down
        good = os.path.join(d, "mirror_stooq")
        os.makedirs(good)
        with open(os.path.join(good, "spy.us.csv"), "w") as fh:
            fh.write(_csv())                          # stooq works

        # base points both sources at the same dir; only stooq finds its file
        r = fetch_one_auto("SPY", os.path.join(d, "c", "SPY.csv"),
                           sources=("yahoo", "stooq"), base="file://" + good)
        assert r["status"].startswith("fetched"), r
        assert r["source"] == "stooq"
        assert "after" in r["status"] and "yahoo" in r["status"], (
            "a source that failed must be named, not silently skipped")
    finally:
        shutil.rmtree(d)


def test_when_every_source_fails_it_says_so_once():
    d = tempfile.mkdtemp()
    try:
        r = fetch_one_auto("SPY", os.path.join(d, "c", "SPY.csv"),
                           base="file:///nonexistent")
        assert r["status"].startswith("ALL SOURCES FAILED")
        for s in SOURCE_ORDER:
            assert s in r["status"], (s, r["status"])
    finally:
        shutil.rmtree(d)


# --------------------------------------------------------------------------
# 3. caching, baskets, and the end-to-end file
# --------------------------------------------------------------------------

def test_a_good_response_is_cached_and_reused():
    d = tempfile.mkdtemp()
    try:
        base = _mirror(d, ["SPY"])
        p = os.path.join(d, "c", "SPY.csv")
        a = fetch_one("SPY", p, source="yahoo", base=base)
        assert a["status"] == "fetched" and a["rows"] == 200, a
        assert a["first"] == "2024-01-01"
        assert fetch_one("SPY", p, source="yahoo", base=base)["status"] == "cached"
        assert fetch_one("SPY", p, source="yahoo", base=base,
                         force=True)["status"] == "fetched"
    finally:
        shutil.rmtree(d)


def test_a_corrupt_cache_file_is_replaced_not_trusted():
    d = tempfile.mkdtemp()
    try:
        base = _mirror(d, ["SPY"])
        p = os.path.join(d, "c", "SPY.csv")
        os.makedirs(os.path.dirname(p))
        with open(p, "w") as fh:
            fh.write("garbage,not,a,price,file\n")
        assert fetch_one("SPY", p, source="yahoo", base=base)["rows"] == 200
    finally:
        shutil.rmtree(d)


def test_a_suspiciously_short_series_is_rejected():
    d = tempfile.mkdtemp()
    try:
        base = _mirror(d, ["NEW"], n=5)
        p = os.path.join(d, "c", "NEW.csv")
        r = fetch_one("NEW", p, source="yahoo", base=base)
        assert "5 row(s)" in r["status"], r
        assert not os.path.exists(p)
    finally:
        shutil.rmtree(d)


def test_one_dead_ticker_does_not_abort_the_basket():
    d = tempfile.mkdtemp()
    try:
        base = _mirror(d, ["SPY", "XLK"])             # XLV absent
        rows = fetch_basket(["SPY", "XLV", "XLK"], os.path.join(d, "c"),
                            sources=("yahoo",), base=base, pause=0)
        by = {r["symbol"]: r for r in rows}
        assert by["SPY"]["status"] == "fetched" and by["XLK"]["status"] == "fetched"
        assert "FAILED" in by["XLV"]["status"], by["XLV"]
        assert len(rows) == 3
    finally:
        shutil.rmtree(d)


def test_end_to_end_produces_the_file_the_dashboard_reads():
    d = tempfile.mkdtemp()
    try:
        base = _mirror(d, DEFAULT_BASKET)
        out = os.path.join(d, "sectors.csv")
        rc = main(["--symbols", ",".join(DEFAULT_BASKET), "--out", out,
                   "--cache", os.path.join(d, "c"), "--source", "yahoo",
                   "--base-url", base, "--pause", "0"])
        assert rc == 0 and os.path.exists(out)

        from tools.rotation_dashboard import load_csv
        series, bench, dates = load_csv(out)
        assert len(series) == len(DEFAULT_BASKET) - 1
        assert len(bench) == len(dates) == 200
    finally:
        shutil.rmtree(d)


def test_a_missing_required_symbol_fails_the_run():
    d = tempfile.mkdtemp()
    try:
        base = _mirror(d, ["XLK", "XLV"])             # no SPY
        rc = main(["--symbols", "XLK,XLV", "--out", os.path.join(d, "o.csv"),
                   "--cache", os.path.join(d, "c"), "--source", "yahoo",
                   "--base-url", base, "--pause", "0", "--require", "SPY"])
        assert rc == 1, "a basket without its benchmark must not be written"
    finally:
        shutil.rmtree(d)


def test_the_two_total_failures_give_DIFFERENT_advice():
    """A blocked vendor and a blocked network need different next steps."""
    import io
    from contextlib import redirect_stdout

    d = tempfile.mkdtemp()
    try:
        # (a) no network at all
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = main(["--symbols", "SPY,XLK", "--out", os.path.join(d, "o.csv"),
                       "--cache", os.path.join(d, "c1"),
                       "--base-url", "file:///nonexistent", "--pause", "0"])
        out = buf.getvalue()
        assert rc == 1 and "nothing fetched" in out
        assert "open outbound access" in out, out

        # (b) the vendor is up and serving a bot-check page
        m = os.path.join(d, "blocked")
        os.makedirs(m)
        for s in ("SPY", "XLK"):
            with open(os.path.join(m, f"{s}.json"), "wb") as fh:
                fh.write(BOT_CHECK)
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = main(["--symbols", "SPY,XLK", "--out", os.path.join(d, "o2.csv"),
                       "--cache", os.path.join(d, "c2"), "--source", "yahoo",
                       "--base-url", "file://" + m, "--pause", "0"])
        out = buf.getvalue()
        assert rc == 1
        assert "bot-check or rate limit" in out, out
        assert "merge_csv" in out, "it must name the manual fallback"
        assert "open outbound access" not in out, (
            "a reachable vendor is not a network problem and must not say so")
    finally:
        shutil.rmtree(d)


def _run_all():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
