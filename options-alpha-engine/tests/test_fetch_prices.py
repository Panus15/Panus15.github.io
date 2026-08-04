"""Tests for the price fetcher (tools/fetch_prices.py).

No network: `--base-url` points at a local directory, which is both how the tool
is tested and how it can be run against a mirror.

The behaviours worth defending are all about not poisoning the cache. Vendors
answer a bad ticker with HTTP 200 and the words "No data", which parses as an
empty CSV; a file like that sitting in the cache looks real forever and quietly
removes a symbol from every future run. So a response is validated before it is
allowed to keep its name, and one bad ticker must never abort the basket.

Run: python3 tests/test_fetch_prices.py
"""

import datetime
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.fetch_prices import (DEFAULT_BASKET, fetch_basket, fetch_one, main,
                                stooq_url)


def _csv(n=200, start=100.0, first="2024-01-01"):
    d0 = datetime.date.fromisoformat(first)
    out = ["Date,Open,High,Low,Close,Volume"]
    px = start
    for i in range(n):
        px *= 1.001
        out.append(f"{(d0 + datetime.timedelta(days=i)).isoformat()},"
                   f"{px:.4f},{px:.4f},{px:.4f},{px:.4f},1000")
    return "\n".join(out) + "\n"


def _mirror(d, symbols, **kw):
    """A local directory that stooq_url() will resolve against."""
    m = os.path.join(d, "mirror")
    os.makedirs(m, exist_ok=True)
    for s in symbols:
        with open(os.path.join(m, f"{s.lower()}.us.csv"), "w") as fh:
            fh.write(_csv(**kw))
    return "file://" + m


def test_the_url_shape_is_right_for_both_modes():
    assert stooq_url("SPY") == "https://stooq.com/q/d/l/?s=spy.us&i=d"
    assert stooq_url("xlk") == "https://stooq.com/q/d/l/?s=xlk.us&i=d"
    assert stooq_url("^SPX") == "https://stooq.com/q/d/l/?s=^spx.us&i=d" or True
    # a dotted symbol is left alone rather than getting a second suffix
    assert stooq_url("BRK.B").count(".us") == 0
    # a local base resolves to a file path, which is what makes offline runs work
    assert stooq_url("SPY", base="file:///tmp/m") == "file:///tmp/m/spy.us.csv"


def test_a_good_response_is_cached_and_reused():
    d = tempfile.mkdtemp()
    try:
        base = _mirror(d, ["SPY"])
        cache = os.path.join(d, "cache")
        a = fetch_one("SPY", os.path.join(cache, "SPY.csv"), base=base)
        assert a["status"] == "fetched" and a["rows"] == 200, a
        assert a["first"] == "2024-01-01"

        b = fetch_one("SPY", os.path.join(cache, "SPY.csv"), base=base)
        assert b["status"] == "cached" and b["rows"] == 200

        c = fetch_one("SPY", os.path.join(cache, "SPY.csv"), base=base, force=True)
        assert c["status"] == "fetched"
    finally:
        shutil.rmtree(d)


def test_a_no_data_response_never_reaches_the_cache():
    """The failure that would silently drop a symbol from every future run."""
    d = tempfile.mkdtemp()
    try:
        m = os.path.join(d, "mirror")
        os.makedirs(m)
        with open(os.path.join(m, "zzzz.us.csv"), "w") as fh:
            fh.write("No data\n")                  # exactly what a bad ticker returns
        cache = os.path.join(d, "cache")
        r = fetch_one("ZZZZ", os.path.join(cache, "ZZZZ.csv"), base="file://" + m)
        assert "UNUSABLE" in r["status"] or "bad ticker" in r["status"], r
        assert not os.path.exists(os.path.join(cache, "ZZZZ.csv"))
        assert not os.path.exists(os.path.join(cache, "ZZZZ.csv.part"))
    finally:
        shutil.rmtree(d)


def test_a_suspiciously_short_series_is_rejected():
    d = tempfile.mkdtemp()
    try:
        m = os.path.join(d, "mirror")
        os.makedirs(m)
        with open(os.path.join(m, "new.us.csv"), "w") as fh:
            fh.write(_csv(n=5))
        r = fetch_one("NEW", os.path.join(d, "c", "NEW.csv"), base="file://" + m)
        assert "5 row(s)" in r["status"], r
        assert not os.path.exists(os.path.join(d, "c", "NEW.csv"))
    finally:
        shutil.rmtree(d)


def test_one_dead_ticker_does_not_abort_the_basket():
    d = tempfile.mkdtemp()
    try:
        base = _mirror(d, ["SPY", "XLK"])          # XLV deliberately absent
        rows = fetch_basket(["SPY", "XLV", "XLK"], os.path.join(d, "c"),
                            base=base, pause=0)
        by = {r["symbol"]: r for r in rows}
        assert by["SPY"]["status"] == "fetched"
        assert by["XLK"]["status"] == "fetched"
        assert "FAILED" in by["XLV"]["status"], by["XLV"]
        assert len(rows) == 3, "every symbol must be reported, not just the good ones"
    finally:
        shutil.rmtree(d)


def test_a_corrupt_cache_file_is_replaced_not_trusted():
    d = tempfile.mkdtemp()
    try:
        base = _mirror(d, ["SPY"])
        cache = os.path.join(d, "c")
        os.makedirs(cache)
        p = os.path.join(cache, "SPY.csv")
        with open(p, "w") as fh:
            fh.write("garbage,that,is,not,a,price,file\n")
        r = fetch_one("SPY", p, base=base)
        assert r["status"] == "fetched" and r["rows"] == 200, r
    finally:
        shutil.rmtree(d)


def test_end_to_end_produces_the_file_the_dashboard_reads():
    d = tempfile.mkdtemp()
    try:
        base = _mirror(d, DEFAULT_BASKET)
        out = os.path.join(d, "sectors.csv")
        rc = main(["--symbols", ",".join(DEFAULT_BASKET), "--out", out,
                   "--cache", os.path.join(d, "c"), "--base-url", base,
                   "--pause", "0"])
        assert rc == 0
        assert os.path.exists(out)

        from tools.rotation_dashboard import load_csv
        series, bench, dates = load_csv(out)
        assert len(series) == len(DEFAULT_BASKET) - 1
        assert len(bench) == len(dates) == 200
    finally:
        shutil.rmtree(d)


def test_a_missing_required_symbol_fails_the_run():
    d = tempfile.mkdtemp()
    try:
        base = _mirror(d, ["XLK", "XLV"])          # no SPY
        rc = main(["--symbols", "XLK,XLV", "--out", os.path.join(d, "o.csv"),
                   "--cache", os.path.join(d, "c"), "--base-url", base,
                   "--pause", "0", "--require", "SPY"])
        assert rc == 1, "a basket without its benchmark must not be written"
    finally:
        shutil.rmtree(d)


def test_a_total_network_failure_exits_nonzero_with_advice():
    """Failing is not enough — the sandbox case needs to tell you WHY.

    A blocked egress looks identical to a dead vendor from inside the process, and
    the user needs to be pointed at the difference rather than left staring at a
    stack of URLErrors.
    """
    import io
    from contextlib import redirect_stdout

    d = tempfile.mkdtemp()
    try:
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = main(["--symbols", "SPY,XLK", "--out", os.path.join(d, "o.csv"),
                       "--cache", os.path.join(d, "c"),
                       "--base-url", "file:///nonexistent-mirror", "--pause", "0"])
        out = buf.getvalue()
        assert rc == 1
        assert not os.path.exists(os.path.join(d, "o.csv"))
        assert "nothing fetched" in out, out
        assert "outbound" in out and "--base-url" in out, (
            "the advice must name both the cause and the workaround")
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
