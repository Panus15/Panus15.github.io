"""Tests for the holdings archive (tools/archive_holdings.py).

The archive's job is to be trustworthy years from now, when nobody remembers what
was downloaded. So the tests are about refusal as much as capture: a file that
does not parse must not be saved, an existing date must not be overwritten, and
above all a book published AFTER a decision date must never reach a backtest that
is trying to measure what was knowable at the time.

Run: python3 tests/test_archive_holdings.py
"""

import datetime
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.archive_holdings import (coverage, fetch_one, load_archive,
                                    supply_at_from_archive)

ROWS = ("underlying,expiry,strike,type,quantity\n"
        "QQQ,2026-08-21,470,call,-8000\n"
        "QQQ,2026-08-21,480,call,-2000\n"
        "QQQ,2026-08-21,400,put,1500\n")


def _archive(root, fund, asof, body=ROWS):
    d = os.path.join(root, fund)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, asof + ".csv"), "w") as fh:
        fh.write(body)


def _serve(body: bytes, tmp):
    """A file:// URL standing in for the vendor, so the test needs no network."""
    p = os.path.join(tmp, "served.csv")
    with open(p, "wb") as fh:
        fh.write(body)
    return "file://" + p


def test_a_good_file_is_archived_under_its_date():
    d = tempfile.mkdtemp()
    try:
        url = _serve(ROWS.encode(), d)
        r = fetch_one("QQQI", url, os.path.join(d, "arc"), asof="2026-08-01")
        assert r["status"] == "archived", r
        assert r["positions"] == 3 and r["shorts"] == 2
        assert os.path.exists(os.path.join(d, "arc", "QQQI", "2026-08-01.csv"))
        # a second run is a no-op, not a re-download
        again = fetch_one("QQQI", url, os.path.join(d, "arc"), asof="2026-08-01")
        assert again["status"] == "already have it"
    finally:
        shutil.rmtree(d)


def test_an_html_error_page_is_never_archived():
    """The failure that would be invisible: a directory of files that are not data."""
    d = tempfile.mkdtemp()
    try:
        page = b"<!doctype html><html><body>403 Forbidden</body></html>"
        r = fetch_one("QQQI", _serve(page, d), os.path.join(d, "arc"),
                      asof="2026-08-01")
        assert "not archived" in r["status"], r
        assert not os.path.exists(os.path.join(d, "arc", "QQQI", "2026-08-01.csv"))
        assert not os.path.exists(os.path.join(d, "arc", "QQQI", "2026-08-01.csv.part"))
    finally:
        shutil.rmtree(d)


def test_a_corrupt_download_leaves_nothing_behind():
    """A file that cannot even be opened must not leave a .part turd in the archive.

    Distinct from the HTML case: that one parses fine and simply has no option
    lines, so it exercises a different branch. This one makes the parser raise, and
    the temp file has to be cleaned up on the way out — otherwise the archive
    accumulates half-downloads that look like nothing and confuse the next run.
    """
    d = tempfile.mkdtemp()
    try:
        arc = os.path.join(d, "arc")
        # a .json endpoint that returns truncated JSON: the parser RAISES here,
        # unlike the HTML case which parses to zero rows
        bad = os.path.join(d, "served.json")
        with open(bad, "wb") as fh:
            fh.write(b'{"fund": "QQQI", "positions": [{"underlying": "QQ')
        r = fetch_one("QQQI", "file://" + bad, arc, asof="2026-08-01")
        assert "UNPARSEABLE" in r["status"], r
        fdir = os.path.join(arc, "QQQI")
        left = os.listdir(fdir) if os.path.isdir(fdir) else []
        assert left == [], f"the archive should be empty, found {left}"

        # ...and a WELL-FORMED json endpoint must archive, which is what proves the
        # temp file keeps its real extension. Writing it to a ".part" name sent
        # every JSON download down the CSV parser and rejected it.
        good = os.path.join(d, "ok.json")
        with open(good, "w") as fh:
            json.dump({"fund": "QQQI", "asof": "2026-08-02", "positions": [
                {"underlying": "QQQ", "expiry": "2026-08-21", "strike": 470,
                 "type": "call", "quantity": -8000}]}, fh)
        r2 = fetch_one("QQQI", "file://" + good, arc, asof="2026-08-02")
        assert r2["status"] == "archived", r2
        assert r2["shorts"] == 1
        assert os.path.exists(os.path.join(arc, "QQQI", "2026-08-02.json"))
    finally:
        shutil.rmtree(d)


def test_a_dead_url_reports_and_does_not_abort():
    d = tempfile.mkdtemp()
    try:
        r = fetch_one("QQQI", "file:///nonexistent/nope.csv",
                      os.path.join(d, "arc"), asof="2026-08-01")
        assert "FETCH FAILED" in r["status"], r
        assert r["fund"] == "QQQI"
    finally:
        shutil.rmtree(d)


def test_coverage_counts_the_gaps_not_just_the_hits():
    d = tempfile.mkdtemp()
    try:
        for day in ("2026-08-03", "2026-08-04", "2026-08-07"):   # Mon Tue ... Fri
            _archive(d, "QQQI", day)
        c = coverage(d)["QQQI"]
        assert c["days"] == 3 and c["first"] == "2026-08-03" and c["last"] == "2026-08-07"
        assert c["weekdays_in_span"] == 5
        assert c["missing"] == 2, "two weekdays were not archived and it must say so"
    finally:
        shutil.rmtree(d)


def test_supply_is_strictly_point_in_time():
    """The mistake that would manufacture the very effect the study measures."""
    d = tempfile.mkdtemp()
    try:
        _archive(d, "QQQI", "2026-08-03")
        _archive(d, "QQQI", "2026-08-10")
        dates = [(datetime.date(2026, 8, 1) + datetime.timedelta(days=i)).isoformat()
                 for i in range(20)]
        prices = [450.0] * 20
        at = supply_at_from_archive(d, "QQQ", dates, prices, lag_days=1)
        assert at is not None

        # 2026-08-02: nothing published yet -> no supply, not a guess
        assert at(1, prices[:2]) == []
        # 2026-08-05: only the 08-03 book is knowable
        assert at(4, prices[:5]), "the 08-03 book should be usable by 08-05"
        # the publication lag is honoured: on 08-03 itself the book is not yet ours
        assert at(2, prices[:3]) == [], "same-day book must not be treated as known"
        # after 08-10 the newer book is used, and it is the only one
        buckets = at(12, prices[:13])
        assert buckets and buckets[0].contracts == 8000
    finally:
        shutil.rmtree(d)


def test_the_archive_feeds_the_crowding_study_end_to_end():
    """Archive -> supply_at -> run_crowding_backtest, with no hand-built fixture."""
    from engine.crowding_backtest import run_crowding_backtest
    from engine.signal_backtest import synthetic_chain_series
    from engine.hedged_backtest import price_path_with_crash

    d = tempfile.mkdtemp()
    try:
        prices = price_path_with_crash(300)
        dates = [(datetime.date(2024, 1, 1) + datetime.timedelta(days=i)).isoformat()
                 for i in range(len(prices))]
        # a book every fortnight, strikes near where the chain quotes
        for i in range(0, 200, 14):
            spot = prices[i]
            rows = ("underlying,expiry,strike,type,quantity\n"
                    f"QQQ,2026-08-21,{round(spot * 1.05, 2)},call,-9000\n")
            _archive(d, "QQQI", dates[i], rows)

        at = supply_at_from_archive(d, "QQQ", dates, prices, lag_days=1)
        res = run_crowding_backtest(prices, synthetic_chain_series(dte=21), at,
                                    dte=21, warmup=63)
        # the point is that it RUNS on archive-derived supply and reports honestly,
        # not that a 300-bar fixture produces a finding
        assert isinstance(res.n_pairs, int)
        assert "NOT ENOUGH DATA" in res.verdict or "EFFECT" in res.verdict
        assert res.skipped or res.n_pairs > 0
    finally:
        shutil.rmtree(d)


def test_an_empty_archive_says_so_rather_than_returning_a_broken_callback():
    d = tempfile.mkdtemp()
    try:
        assert load_archive(d) == []
        assert coverage(d) == {}
        assert supply_at_from_archive(d, "QQQ", ["2026-08-01"], [450.0]) is None
        assert supply_at_from_archive(os.path.join(d, "nope"), "QQQ",
                                      ["2026-08-01"], [450.0]) is None
    finally:
        shutil.rmtree(d)


def test_sources_can_be_overridden_without_editing_code():
    from tools.archive_holdings import KNOWN_SOURCES, fetch_all
    d = tempfile.mkdtemp()
    try:
        url = _serve(ROWS.encode(), d)
        rows = fetch_all(os.path.join(d, "arc"), sources={"MYFUND": url},
                         asof="2026-08-01")
        assert len(rows) == 1 and rows[0]["fund"] == "MYFUND"
        assert rows[0]["status"] == "archived"
        assert set(KNOWN_SOURCES) >= {"QQQI", "JEPQ"}
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
