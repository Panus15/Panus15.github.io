"""Tests for the holdings archive (tools/archive_holdings.py).

The archive's job is to be trustworthy years from now, when nobody remembers what
was downloaded. So the tests are about refusal as much as capture: a file that
does not parse must not be saved, an existing date must not be overwritten, and
above all a book published AFTER a decision date must never reach a backtest that
is trying to measure what was knowable at the time.

Run: python3 tests/test_archive_holdings.py
"""

import datetime
import io
import json
import os
import subprocess
import shutil
import sys
import tempfile
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.archive_holdings import (KNOWN_SOURCES, NEEDS_URL, coverage,
                                    fetch_one, load_archive, main,
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



def test_the_archive_is_not_gitignored():
    """The one dataset here that cannot be re-bought must be under version control.

    `.gitignore` carried a blanket `*.csv` under a "never commit vendor data"
    heading. Fund holdings ARE vendor data, so the rule was defensible in the
    abstract and catastrophic in particular: holdings are published daily and
    overwritten, so a day not captured is gone at any price, and the crowding
    study wants ~18 months of them. The effect was that the only irreplaceable
    data in the repo was the only data with no history and no backup.

    A blanket ignore is also exactly the kind of rule someone re-adds while
    tidying, so this asserts on `git check-ignore` rather than on the file text.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isdir(os.path.join(root, ".git")) and not os.path.isfile(
            os.path.join(root, ".gitignore")):
        return                                   # not a checkout; nothing to assert

    probe = os.path.join(root, "holdings", "QQQI", "2099-01-01.csv")
    os.makedirs(os.path.dirname(probe), exist_ok=True)
    made = not os.path.exists(probe)
    if made:
        with open(probe, "w") as fh:
            fh.write("date,strike\n")
    try:
        # -q, NOT -v: with -v the exit code is 0 whenever any rule MATCHES,
        # including a negation, so `-v` would read "!holdings/**/*.csv matched"
        # as "ignored" and the test would fail on the very fix it defends.
        # Bare check-ignore is the one that means what it says: 0 = ignored.
        q = subprocess.run(["git", "check-ignore", "-q", probe], cwd=root,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        why = subprocess.run(["git", "check-ignore", "-v", probe], cwd=root,
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        assert q.returncode != 0, (
            "the holdings archive is gitignored by "
            + why.stdout.decode("utf-8", "replace").strip()
            + " -- point-in-time fund books cannot be re-fetched, so an ignored "
              "archive is a dataset with no backup")
    finally:
        if made:
            os.remove(probe)
        for d in (os.path.dirname(probe), os.path.join(root, "holdings")):
            try:
                os.rmdir(d)
            except OSError:
                pass


def test_secrets_and_scratch_data_are_still_ignored():
    """The un-ignore must be surgical. A negation broad enough to sweep in .env
    or a downloaded price cache would trade one disaster for a worse one."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isfile(os.path.join(root, ".gitignore")):
        return
    for rel in (".env", "data/vendor.csv", "sectors.csv", "_prices/SPY.csv"):
        path = os.path.join(root, rel)
        p = subprocess.run(["git", "check-ignore", "-q", path], cwd=root,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        assert p.returncode == 0, f"{rel} is no longer ignored"



def _cfg(tmp, mapping):
    p = os.path.join(tmp, "sources.json")
    with open(p, "w") as fh:
        json.dump(mapping, fh)
    return p


def _fetch(tmp, mapping):
    """Run the CLI exactly as the scheduler would, capturing what it prints."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["fetch", "--dir", os.path.join(tmp, "arc"),
                   "--sources", _cfg(tmp, mapping), "--asof", "2026-08-01"])
    return rc, buf.getvalue()


def test_a_run_that_captured_nothing_exits_nonzero():
    """A green exit on an empty run is the most expensive lie this tool can tell.

    Holdings are published daily and overwritten. A scheduled job that reports
    success on a day it captured nothing loses that day permanently, and the
    operator finds out eighteen months later when the study will not run.
    """
    d = tempfile.mkdtemp()
    try:
        rc, out = _fetch(d, {"NOPE": "file:///nonexistent/none.csv"})
        assert rc == 1, f"a run that archived nothing exited 0:\n{out}"
        assert "NOTHING CAPTURED" in out, out
    finally:
        shutil.rmtree(d)


def test_a_day_already_on_disk_is_success_not_failure():
    """`already have it` is the idempotent case: a re-run, or a weekend with no
    new file. Today's data exists, so the run has done its job and must not cry
    wolf — an alarm that fires on normal days gets muted."""
    d = tempfile.mkdtemp()
    try:
        m = {"QQQI": _serve(ROWS.encode(), d)}
        assert _fetch(d, m)[0] == 0
        rc, out = _fetch(d, m)
        assert rc == 0, f"the second, idempotent run reported failure:\n{out}"
    finally:
        shutil.rmtree(d)


def test_a_partial_capture_exits_zero_but_says_what_failed():
    """Two funds have no URL and have not for weeks. Failing daily on a known
    gap trains the operator to ignore the alarm, and then the day QQQI itself
    breaks goes unnoticed. Report it loudly, exit 0."""
    d = tempfile.mkdtemp()
    try:
        rc, out = _fetch(d, {"QQQI": _serve(ROWS.encode(), d),
                             "NOPE": "file:///nonexistent/none.csv"})
        assert rc == 0, f"a partial capture failed the whole run:\n{out}"
        assert "1 fund(s) did not archive" in out, out
    finally:
        shutil.rmtree(d)


def test_a_fund_with_no_url_says_so_instead_of_fetching_a_web_page():
    """JEPI/JEPQ pointed at am.jpmorgan.com product PAGES. The HTML was saved as
    .csv, parsed to zero option lines and rejected — every day, while the run
    still reported success. Now the gap is named before anything is downloaded."""
    d = tempfile.mkdtemp()
    try:
        r = fetch_one("JEPQ", NEEDS_URL, os.path.join(d, "arc"))
        assert "NO URL CONFIGURED" in r["status"], r["status"]
        assert not os.path.exists(os.path.join(d, "arc", "JEPQ")), \
            "it created a directory for a fund it cannot fetch"
    finally:
        shutil.rmtree(d)



def test_every_built_in_url_is_shaped_like_a_data_file():
    """The table said JEPQ was covered when it could not possibly work.

    No network call: a product landing page is identifiable by its shape alone.
    The rule is that a source is either a real data file — an extension the
    parser dispatches on — or an explicit, visible admission that we have none.
    """
    for fund, url in KNOWN_SOURCES.items():
        if url == NEEDS_URL:
            continue
        low = url.lower().split("?")[0]
        assert low.startswith("https://"), (fund, url)
        assert low.endswith((".csv", ".json")), (
            f"{fund} points at {url!r}, which has no data-file extension. "
            f"FundBook.from_file dispatches on the extension, so a landing page "
            f"is saved as .csv and rejected daily while the run looks fine. "
            f"Either use the direct-download URL or set it to NEEDS_URL.")


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
