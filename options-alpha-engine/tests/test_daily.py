"""Tests for the daily job (tools/daily.py).

This is the module whose failure mode is silence. If one vendor is down and the
whole run aborts, the fund-holdings capture is lost for that day and cannot be
recovered later at any price — so the tests are mostly about a step failing
WITHOUT taking the others with it, and about the status report telling the truth
when nothing has been captured yet.

Run: python3 tests/test_daily.py
"""

import datetime
import json
import os
import platform
import io
import shutil
import sys
from contextlib import redirect_stdout
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import daily

from tools.daily import (TRADING_DAYS_NEEDED, _step, install_line, main, run,
                         status)

ROWS = ("underlying,expiry,strike,type,quantity\n"
        "QQQ,2026-08-21,470,call,-8000\n")


def _cfg(**kw):
    base = {"holdings": "", "ledger": "", "prices": "", "events": "",
            "sources": "", "source": "deribit", "dte": 30,
            "currency": "BTC", "symbol": "SPX"}
    base.update(kw)
    return base


def _archive(root, fund, days):
    d = os.path.join(root, fund)
    os.makedirs(d, exist_ok=True)
    d0 = datetime.date(2026, 1, 5)
    for i in range(days):
        day = d0 + datetime.timedelta(days=i)
        if day.weekday() >= 5:
            continue
        with open(os.path.join(d, day.isoformat() + ".csv"), "w") as fh:
            fh.write(ROWS)


def test_a_failing_step_is_caught_and_reported_not_raised():
    def boom():
        raise RuntimeError("vendor down")
    r = _step("holdings", boom)
    assert r["ok"] is False and "vendor down" in r["detail"]

    ok = _step("holdings", lambda: "3/4 books")
    assert ok["ok"] is True and ok["detail"] == "3/4 books"

    # a main() that exits non-zero counts as failure, not as a crash
    def exits():
        raise SystemExit(2)
    assert _step("x", exits)["ok"] is False


def test_a_step_exiting_with_a_message_is_contained_like_any_other_failure():
    """`sys.exit("message")` sets SystemExit.code to a STRING, and int() on it
    raised ValueError out of the handler whose entire job is to contain failures —
    so the one construct it exists to catch was the one that escaped it, taking
    every remaining step with it."""
    def _boom():
        raise SystemExit("benchmark column 'SPY' not found")
    st = daily._step("noisy", _boom)
    assert st["ok"] is False
    assert "SPY" in st["detail"], st["detail"]


def test_a_clean_exit_zero_still_counts_as_success():
    def _fine():
        raise SystemExit(0)
    assert daily._step("quiet", _fine)["ok"] is True


def test_one_broken_step_does_not_cost_the_others():
    """The day's capture must survive one vendor being down."""
    d = tempfile.mkdtemp()
    try:
        # holdings points at a real dir with unreachable sources -> that step fails
        srcs = os.path.join(d, "src.json")
        with open(srcs, "w") as fh:
            json.dump({"NOPE": "file:///definitely/not/here.csv"}, fh)
        rows = run(_cfg(holdings=os.path.join(d, "h"), sources=srcs,
                        prices=os.path.join(d, "p.csv")))
        names = [r["step"] for r in rows]
        assert "fund holdings" in names and "price basket" in names, names
        assert len(rows) == 2, "every configured step must be attempted"
    finally:
        shutil.rmtree(d)


def test_status_is_blunt_when_nothing_has_been_captured():
    d = tempfile.mkdtemp()
    try:
        s = status(_cfg(holdings=os.path.join(d, "empty"),
                        ledger=os.path.join(d, "no.jsonl")))
        assert "NOT STARTED" in s
        assert "unrecoverable" in s, "the cost of waiting must be stated"
        assert "forward-test time only accrues" in s
    finally:
        shutil.rmtree(d)


def test_status_counts_the_clock_in_days_and_gives_a_date():
    d = tempfile.mkdtemp()
    try:
        h = os.path.join(d, "h")
        _archive(h, "QQQI", 40)                       # ~28 weekdays
        s = status(_cfg(holdings=h))
        assert f"/{TRADING_DAYS_NEEDED} trading days" in s
        assert "becomes possible around" in s
        assert "%" in s and "[" in s                  # a progress bar, not a number
        # the projected date must be in the future, not today
        assert str(datetime.date.today().year) in s or str(
            datetime.date.today().year + 1) in s
    finally:
        shutil.rmtree(d)


def test_status_says_when_the_archive_is_finally_deep_enough():
    d = tempfile.mkdtemp()
    try:
        h = os.path.join(d, "h")
        _archive(h, "QQQI", int(TRADING_DAYS_NEEDED * 7 / 5) + 20)
        s = status(_cfg(holdings=h))
        assert "ENOUGH DATA" in s, s
        assert "becomes possible around" not in s
    finally:
        shutil.rmtree(d)


def test_the_ledger_clock_counts_settled_entries_not_rows():
    """30 open entries are not 30 results; only settled ones can be scored."""
    d = tempfile.mkdtemp()
    try:
        p = os.path.join(d, "l.jsonl")
        with open(p, "w") as fh:
            for i in range(12):
                fh.write(json.dumps({"id": i, "status":
                                     "settled" if i < 4 else "open"}) + "\n")
            fh.write("not json\n")                     # must not crash the count
        s = status(_cfg(ledger=p, dte=30))
        assert "12 entries, 4 settled" in s, s
        assert "need 30 settled" in s
        # the junk line is NOT an entry, and is surfaced rather than swallowed
        assert "1 unreadable line" in s, s
    finally:
        shutil.rmtree(d)


def test_install_prints_a_line_for_every_platform():
    """All three branches, not just whichever OS the tests happen to run on.

    A scheduler line is copy-pasted once and never read again, so a broken
    weekday filter on the macOS branch would quietly run the job on Saturdays
    forever — and the Linux-only test that used to live here could not see it.
    """
    import platform as _p
    real = _p.system
    try:
        for osname, marker in (("Linux", "crontab"), ("Darwin", "crontab"),
                               ("Windows", "schtasks")):
            _p.system = lambda o=osname: o
            line = install_line(_cfg(holdings="holdings", ledger="btc.jsonl"),
                                hour=18)
            assert marker in line, (osname, line)
            assert "tools.daily run" in line
            assert "--holdings holdings" in line and "--ledger btc.jsonl" in line
            # weekdays only, however this platform spells it
            if marker == "crontab":
                assert "* * 1-5" in line, (osname, line)
                assert "18" in line
            else:
                assert "/sc daily" in line and "18:00" in line, line
    finally:
        _p.system = real


def test_run_with_nothing_configured_fails_loudly():
    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["run", "--holdings", "", "--ledger", "", "--prices", ""])
    assert rc == 1
    assert "nothing configured" in buf.getvalue()


def test_a_totally_failed_run_exits_nonzero_but_a_partial_one_does_not():
    d = tempfile.mkdtemp()
    try:
        import io
        from contextlib import redirect_stdout
        srcs = os.path.join(d, "src.json")
        with open(srcs, "w") as fh:
            json.dump({"NOPE": "file:///nope.csv"}, fh)

        # the only configured step fails -> non-zero
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = main(["run", "--holdings", os.path.join(d, "h"),
                       "--sources", srcs, "--ledger", "", "--prices", ""])
        # archive_holdings reports a per-fund failure but the STEP itself succeeds,
        # so this asserts the run completed and said what happened rather than died
        assert rc in (0, 1)
        out = buf.getvalue()
        assert "fund holdings" in out
        assert "daily run" in out
    finally:
        shutil.rmtree(d)



# --------------------------------------------------------------------------
# install: the scheduler must get a PATH, and it must carry every flag
# --------------------------------------------------------------------------

def _cfg(**kw):
    base = {"holdings": "holdings", "ledger": "", "prices": "", "events": "",
            "sources": "", "source": "deribit", "dte": 30, "currency": "BTC",
            "symbol": "SPX", "dashboard": "", "chain_json": "", "price_json": "",
            "equity": 100_000.0}
    base.update(kw)
    return base


def test_every_configured_flag_reaches_the_scheduled_command():
    """The defect class, pinned. The old builder listed five flags by hand and
    appended two more, which dropped --symbol: a job configured for SPY
    scheduled itself without it and fell back to SPX at record time, forever,
    silently recording a different underlying than the operator chose.

    This test USED to claim that "a seventh flag added tomorrow and forgotten
    fails here" while asserting on a hardcoded list of eight — so it could not
    have caught the very thing it promised. The expectation is now derived from
    what `main()` accepts, so a flag added to the job and forgotten in `_EMIT`
    fails here by construction.
    """
    cfg = _cfg(ledger="paper.jsonl", prices="sectors.csv", events="ev.json",
               sources="src.json", source="tradier", symbol="SPY", dte=45,
               dashboard="rotation.html", chain_json="c.json",
               price_json="p.json", equity=250_000.0)
    cmd = daily.job_command(cfg, python="/usr/bin/python3")

    # 1. every cfg key the job understands has a rule in the emit table
    missing = sorted(set(cfg) - set(daily._EMIT))
    assert not missing, (f"no _EMIT rule for {missing} — a job configured with "
                         f"them schedules itself without them, silently")

    # 2. and every rule that fires actually puts its flag on the line
    for key, want in daily._EMIT.items():
        flag = "--" + key.replace("_", "-")
        if want(cfg) and cfg.get(key) not in (None, ""):
            assert f"{flag} " in cmd, f"{flag} missing from: {cmd}"
        else:
            assert flag not in cmd, f"{flag} leaked into: {cmd}"

    # 3. spot-check the values, since presence alone would pass on a wrong one
    for flag, value in (("--holdings", "holdings"), ("--ledger", "paper.jsonl"),
                        ("--dte", "45"), ("--symbol", "SPY"),
                        ("--dashboard", "rotation.html")):
        assert f"{flag} {value}" in cmd, f"{flag} {value} missing from: {cmd}"
    assert "--currency" not in cmd, "a deribit-only flag leaked into a tradier job"


def test_the_config_covers_every_flag_the_parser_accepts():
    """The other half of the same leak: a flag added to the parser but not to
    `_cfg` is accepted on the command line and then ignored."""
    import argparse
    seen = {}
    real_add = argparse.ArgumentParser.add_argument

    def spy(self, *args, **kw):
        act = real_add(self, *args, **kw)
        if args and str(args[0]).startswith("--"):
            seen[act.dest] = True
        return act

    argparse.ArgumentParser.add_argument = spy
    try:
        try:
            daily.main(["status", "--holdings", ""])
        except SystemExit:
            pass
    finally:
        argparse.ArgumentParser.add_argument = real_add

    # flags that configure the scheduler itself rather than the job
    NOT_JOB = {"apply", "hour"}
    cfg_keys = set(_cfg())
    orphans = sorted(set(seen) - cfg_keys - NOT_JOB)
    assert not orphans, (f"{orphans} can be passed on the command line but never "
                         f"reach the job config, so they are silently ignored")


def test_the_underlying_flag_matches_the_source():
    assert "--currency BTC" in daily.job_command(_cfg(source="deribit"))
    assert "--symbol" not in daily.job_command(_cfg(source="deribit"))
    assert "--symbol SPY" in daily.job_command(_cfg(source="tradier", symbol="SPY"))
    assert "--currency" not in daily.job_command(_cfg(source="tradier"))


def test_a_space_in_the_interpreter_path_survives():
    """The verified Windows failure. A default install puts python under
    C:\\Program Files\\, cmd split at the space, and the task failed every day
    while schtasks still reported it registered — a clock the operator believed
    was running."""
    cmd = daily.job_command(_cfg(), python=r"C:\Program Files\Python313\python.exe")
    assert cmd.startswith('"C:\\Program Files\\Python313\\python.exe"'), cmd


def test_the_scheduler_gets_one_quoted_path_and_no_compound_command():
    """schtasks takes its payload in /tr as a single double-quoted string with
    no way to escape an inner quote, so `cd X && python ...` was structurally
    unquotable. A wrapper script reduces the payload to one path."""
    real = platform.system
    platform.system = lambda: "Windows"
    try:
        line = daily.scheduler_command(r"C:\Users\First Last\repo\daily-job.bat")
    finally:
        platform.system = real
    assert "&&" not in line, line
    assert '/tr "C:\\Users\\First Last\\repo\\daily-job.bat"' in line, line


def test_the_wrapper_logs_on_every_platform():
    """Windows had no log redirect while Linux and macOS both did — on the one
    OS where the command was broken, there was no record that it ran."""
    real = platform.system
    for osname, suffix in (("Windows", ".bat"), ("Linux", ".sh")):
        platform.system = lambda o=osname: o
        d = tempfile.mkdtemp()
        try:
            path = daily.write_wrapper(_cfg(), python="/usr/bin/python3",
                                       path=os.path.join(d, "job" + suffix))
            body = open(path).read()
            assert "daily.log" in body, (osname, body)
            assert "2>&1" in body, (osname, body)
            assert "tools.daily run" in body, (osname, body)
        finally:
            platform.system = real
            shutil.rmtree(d)


def test_install_writes_the_wrapper_but_schedules_nothing_without_apply():
    """Registering a scheduled task is a side effect on the machine, and on
    Windows it needs elevation this shell may not have. Writing a file is not."""
    called = []
    real_call = daily.subprocess.call
    daily.subprocess.call = lambda *a, **k: called.append(a) or 0
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            rc = daily.main(["install"])
    finally:
        daily.subprocess.call = real_call
        for f in ("daily-job.sh", "daily-job.bat"):
            p = os.path.join(daily.ROOT, f)
            if os.path.exists(p):
                os.remove(p)
    assert rc == 0
    assert not called, "install scheduled a task without --apply"
    out = buf.getvalue()
    assert "daily-job" in out and "--apply" in out, out



def test_a_space_in_a_configured_path_survives():
    """Not only the interpreter. A ledger under "C:\\My Data\\" is ordinary, and
    an unquoted value splits the same way the interpreter path did."""
    cmd = daily.job_command(_cfg(ledger=r"C:\My Data\paper.jsonl",
                                 sources=r"C:\My Data\src.json"),
                            python="/usr/bin/python3")
    assert '--ledger "C:\\My Data\\paper.jsonl"' in cmd, cmd
    assert '--sources "C:\\My Data\\src.json"' in cmd, cmd


def test_the_wrapper_enters_the_repo_before_running():
    """`python -m tools.daily` resolves the module from the working directory,
    and a scheduler starts in its own — C:\\Windows\\System32 for schtasks. A
    wrapper that does not cd fails with ModuleNotFoundError every night, in a
    log nobody reads, while the task itself reports success."""
    real = platform.system
    for osname, suffix in (("Windows", ".bat"), ("Linux", ".sh")):
        platform.system = lambda o=osname: o
        d = tempfile.mkdtemp()
        try:
            body = open(daily.write_wrapper(
                _cfg(), python="/usr/bin/python3",
                path=os.path.join(d, "job" + suffix))).read()
            assert daily.ROOT in body, (osname, body)
            assert body.count("cd") >= 1, (osname, body)
            first = [l for l in body.splitlines()
                     if l.strip() and not l.lstrip().startswith(("#", "@", "REM"))][0]
            assert "cd" in first, f"{osname}: first real line is not a cd: {first}"
        finally:
            platform.system = real
            shutil.rmtree(d)


# --------------------------------------------------------------------------
# The dashboard step — the clocks are only visible on the page
# --------------------------------------------------------------------------

def _no_network():
    """Any test below that reaches a vendor fails loudly instead of depending on
    whether this machine happens to have outbound access.

    This is the guard the three rewritten tests needed. They called `daily.run`,
    which fetches prices and records a live ledger entry; in a blocked sandbox
    those steps failed and the tests passed BY ACCIDENT, and in CI they succeeded
    and clobbered the very fixtures the tests had written.
    """
    import socket
    import urllib.request

    def _boom(*a, **kw):
        raise AssertionError("this test reached the network")

    saved = (socket.socket, urllib.request.urlopen)
    socket.socket, urllib.request.urlopen = _boom, _boom

    class _Restore:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            socket.socket, urllib.request.urlopen = saved
            return False
    return _Restore()


def test_the_run_includes_a_dashboard_step_when_one_is_configured():
    """Only the dashboard is configured, so nothing here touches a vendor."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp, _no_network():
        out = os.path.join(tmp, "page.html")
        steps = daily.run(_cfg(holdings="", ledger="", prices="", dashboard=out))
        assert [st["step"] for st in steps] == ["dashboard"], steps
        assert steps[0]["ok"], steps[0]["detail"]
        page = io.open(out, encoding="utf-8").read()
        assert page.lstrip().startswith("<!doctype html>")
        assert "@@" not in page, "an unfilled template token reached the page"
        assert 'id="fwd"' in page and 'id="opt"' in page
        assert "0 recorded" in steps[0]["detail"], steps[0]["detail"]


def test_the_dashboard_falls_back_to_the_fixture_when_no_basket_exists():
    """A page reading "0 settled" is the honest state and is more use than no
    page. Rendered directly: making the price STEP fail to reach this condition is
    what made the old test depend on a broken network."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp, _no_network():
        out = os.path.join(tmp, "page.html")
        cfg = _cfg(holdings="", ledger="", dashboard=out)
        cfg["prices"] = os.path.join(tmp, "never-fetched.csv")
        detail = daily.render_dashboard(cfg)
        assert "FIXTURE" in detail, detail
        assert os.path.exists(out)


def test_a_real_price_basket_is_used_rather_than_the_fixture():
    """Falling back to the generated world when a real CSV exists would draw
    invented prices while labelling nothing — the most misleading thing this job
    could produce."""
    import tempfile
    from tools import rotation_dashboard as rd
    with tempfile.TemporaryDirectory() as tmp, _no_network():
        px, bench, dates = rd.demo_world(n=700)
        cols = dict(px)
        cols[rd.BENCHMARK] = bench      # load_csv needs the benchmark column too
        syms = sorted(cols)
        csv = os.path.join(tmp, "sectors.csv")
        with io.open(csv, "w", encoding="utf-8") as fh:
            fh.write("date," + ",".join(syms) + "\n")
            for i, d in enumerate(dates):
                fh.write(d + "," + ",".join(f"{cols[sy][i]:.4f}" for sy in syms) + "\n")
        out = os.path.join(tmp, "page.html")
        cfg = _cfg(holdings="", ledger="", dashboard=out)
        cfg["prices"] = csv
        detail = daily.render_dashboard(cfg)
        assert "sectors.csv" in detail, detail
        assert "FIXTURE" not in detail, (
            "a real price basket was ignored in favour of generated data")
        assert dates[-1] in io.open(out, encoding="utf-8").read(), (
            "the page was not drawn from the supplied CSV")


def test_the_dashboard_reports_the_ledger_counts_it_drew():
    """So the operator sees the clock move in the run log, not only in a browser.

    Rendered directly. Going through `run` also ran the RECORDER, which appends a
    live vendor entry — so in CI the ledger had grown between the count this test
    computed and the count the page drew, and the two never matched."""
    import tempfile
    from engine.data import SyntheticAdapter
    from engine.signal_backtest import synthetic_chain_series
    from models.baseline import BaselineDensityForecaster
    from tools.paper_trade import paper_trade_series
    with tempfile.TemporaryDirectory() as tmp:
        ladder = tuple(round(-0.20 + 0.025 * i, 3) for i in range(17))
        prices = SyntheticAdapter(seed=7).price_history("SPY", days=400)
        led = paper_trade_series(prices, synthetic_chain_series(dte=21, ladder=ladder,
                                                               min_px=0.005),
                                 BaselineDensityForecaster(), dte=21, warmup=120,
                                 step=21)
        lp = os.path.join(tmp, "led.jsonl")
        led.save(lp)
        rec = led.report()
        out = os.path.join(tmp, "page.html")
        cfg = _cfg(holdings="", prices="", dashboard=out)
        cfg["ledger"] = lp
        with _no_network():
            detail = daily.render_dashboard(cfg)
        assert f"{rec.n_recorded} recorded" in detail, (detail, rec.n_recorded)
        assert f"{rec.n_settled} settled" in detail, (detail, rec.n_settled)


def test_status_names_the_dashboard_and_whether_it_exists_yet():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "page.html")
        txt = daily.status(_cfg(holdings="", dashboard=out))
        assert "dashboard" in txt and "not rendered yet" in txt
        io.open(out, "w", encoding="utf-8").write("x")
        txt2 = daily.status(_cfg(holdings="", dashboard=out))
        assert "not rendered yet" not in txt2


def test_a_job_with_no_dashboard_configured_does_not_render_one():
    steps = daily.run(_cfg(holdings="", dashboard=""))
    assert "dashboard" not in [st["step"] for st in steps]


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
