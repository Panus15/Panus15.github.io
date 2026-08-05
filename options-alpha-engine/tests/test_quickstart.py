"""Tests for the one-command launcher (tools/quickstart.py).

The launcher exists because of shell mistakes, so the tests are about shell
mistakes: it must work when the process is started from an unrelated working
directory, it must refuse to git-pull onto the wrong branch, and — the one that
actually bit a user — when the data step produces nothing it must stop there
and say so, rather than letting the next step die on a missing file.

Run: python3 tests/test_quickstart.py
"""

import datetime
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import quickstart
from tools.quickstart import BRANCH, main, project_dir, update


def _yahoo(n=400, start=100.0, first="2022-01-01", drift=1.0004):
    d0 = datetime.datetime.fromisoformat(first + "T00:00:00")
    stamps, closes, px = [], [], start
    for i in range(n):
        px *= drift
        stamps.append(int((d0 + datetime.timedelta(days=i)).timestamp()))
        closes.append(round(px, 4))
    return json.dumps({"chart": {"result": [{
        "meta": {}, "timestamp": stamps,
        "indicators": {"quote": [{"close": closes}],
                       "adjclose": [{"adjclose": closes}]}}], "error": None}}).encode()


def _mirror(d):
    """A local stand-in for the vendor, one file per symbol in the basket."""
    m = os.path.join(d, "mirror")
    os.makedirs(m, exist_ok=True)
    from tools.fetch_prices import DEFAULT_BASKET
    for i, s in enumerate(DEFAULT_BASKET):
        with open(os.path.join(m, f"{s.upper()}.json"), "wb") as fh:
            fh.write(_yahoo(start=50.0 + i, drift=1.0002 + i * 0.00004))
    return "file://" + m


def _run(argv, cwd=None):
    """Run the launcher, capturing stdout, from an arbitrary directory."""
    old = os.getcwd()
    if cwd:
        os.chdir(cwd)
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            rc = main(argv)
    finally:
        os.chdir(old)
    return rc, buf.getvalue()


# --------------------------------------------------------------------------
# 1. The working directory cannot be wrong
# --------------------------------------------------------------------------

def test_it_runs_from_an_unrelated_working_directory():
    """The reported failure was `cd` run twice. Starting elsewhere must work."""
    d = tempfile.mkdtemp()
    try:
        rc, out = _run(["--demo", "--no-update", "--no-open",
                        "--out", os.path.join(d, "r.html")], cwd=d)
        assert rc == 0, out
        assert os.path.exists(os.path.join(d, "r.html")), "no dashboard written"
    finally:
        shutil.rmtree(d)


def test_project_dir_is_the_code_not_the_shell():
    d = tempfile.mkdtemp()
    old = os.getcwd()
    try:
        os.chdir(d)
        assert os.path.exists(os.path.join(project_dir(), "tools", "quickstart.py"))
    finally:
        os.chdir(old)
        shutil.rmtree(d)


def test_relative_output_paths_land_next_to_the_code():
    """`--out x.html` must not scatter files wherever the user was standing."""
    d = tempfile.mkdtemp()
    name = "_qs_probe.html"
    target = os.path.join(project_dir(), name)
    try:
        rc, out = _run(["--demo", "--no-update", "--no-open", "--out", name], cwd=d)
        assert rc == 0, out
        assert os.path.exists(target), "relative --out did not resolve to the project"
        assert not os.path.exists(os.path.join(d, name))
    finally:
        for p in (target, os.path.join(project_dir(), quickstart.RESULT)):
            if os.path.exists(p):
                os.remove(p)
        shutil.rmtree(d)


# --------------------------------------------------------------------------
# 2. The update step is allowed to refuse
# --------------------------------------------------------------------------

def test_update_refuses_when_a_different_branch_is_checked_out():
    """`git pull origin <branch>` from elsewhere MERGES. Skipping is correct."""
    calls = []
    real = quickstart._git

    def fake(args, cwd, timeout=120):
        calls.append(args[0])
        if args[0] == "rev-parse":
            return 0, "some-other-branch"
        return 0, "Already up to date."

    quickstart._git = fake
    try:
        msg = update("/nowhere")
    finally:
        quickstart._git = real
    assert "skipped" in msg and "some-other-branch" in msg, msg
    assert "pull" not in calls, "it pulled onto the wrong branch"


def test_update_pulls_on_the_right_branch():
    calls = []
    real = quickstart._git

    def fake(args, cwd, timeout=120):
        calls.append(args[0])
        return 0, (BRANCH if args[0] == "rev-parse" else "Fast-forward 3 files")

    quickstart._git = fake
    try:
        msg = update("/nowhere")
    finally:
        quickstart._git = real
    assert "pull" in calls and "updated" in msg, (calls, msg)


def test_a_missing_git_is_not_fatal():
    real = quickstart._git
    quickstart._git = lambda a, c, timeout=120: (1, "FileNotFoundError: git")
    try:
        msg = update("/nowhere")
    finally:
        quickstart._git = real
    assert "skipped" in msg and "git checkout" in msg, msg


# --------------------------------------------------------------------------
# 3. A dead vendor stops the run where it died
# --------------------------------------------------------------------------

def test_a_blocked_vendor_stops_at_the_download_step():
    """The real incident: no sectors.csv, then FileNotFoundError two steps later.

    The launcher must name the step that failed and must NOT reach the render.
    """
    d = tempfile.mkdtemp()
    try:
        rc, out = _run(["--no-update", "--no-open",
                        "--base-url", "file://" + os.path.join(d, "empty"),
                        "--csv", os.path.join(d, "s.csv"),
                        "--out", os.path.join(d, "r.html")], cwd=d)
        assert rc == 1, "a run with no data reported success"
        assert "STOPPED at step 2" in out, out[-400:]
        assert not os.path.exists(os.path.join(d, "r.html")), \
            "it rendered a chart with no prices"
        assert "Traceback" not in out and "FileNotFoundError" not in out, \
            "the user saw a stack trace instead of the reason"
    finally:
        shutil.rmtree(d)


def test_the_stop_message_blames_the_vendor_not_the_user():
    d = tempfile.mkdtemp()
    try:
        _, out = _run(["--no-update", "--no-open",
                       "--base-url", "file://" + os.path.join(d, "empty"),
                       "--csv", os.path.join(d, "s.csv"),
                       "--out", os.path.join(d, "r.html")], cwd=d)
        tail = out[out.index("STOPPED"):]
        assert "not about your setup" in tail, tail
    finally:
        shutil.rmtree(d)


# --------------------------------------------------------------------------
# 4. The happy path, end to end, against a local mirror
# --------------------------------------------------------------------------

def test_end_to_end_against_a_mirror_writes_both_files_and_the_verdicts():
    d = tempfile.mkdtemp()
    try:
        rc, out = _run(["--no-update", "--no-open", "--base-url", _mirror(d),
                        "--csv", os.path.join(d, "s.csv"),
                        "--out", os.path.join(d, "r.html")], cwd=d)
        assert rc == 0, out[-600:]
        assert os.path.exists(os.path.join(d, "s.csv"))
        html = open(os.path.join(d, "r.html"), encoding="utf-8").read()
        assert html.lstrip().lower().startswith("<!doctype html>"), html[:60]
        assert len(html) > 5000
        assert "panel:" in out and "book :" in out, "the verdicts were not printed"
    finally:
        for p in (os.path.join(project_dir(), quickstart.RESULT),):
            if os.path.exists(p):
                os.remove(p)
        shutil.rmtree(d)


def test_the_page_declares_its_encoding_so_it_reads_on_any_windows():
    """No charset means the browser guesses from the machine's locale.

    The page carries em dashes. Read as cp874 or cp1252 they become mojibake,
    which nobody sees until the file is opened on a non-English desktop.
    """
    d = tempfile.mkdtemp()
    try:
        _run(["--demo", "--no-update", "--no-open",
              "--out", os.path.join(d, "r.html")], cwd=d)
        raw = open(os.path.join(d, "r.html"), "rb").read()
        head = raw[:200].decode("ascii", "replace").lower()
        assert "<!doctype html>" in head, head
        assert 'charset="utf-8"' in head, "the page does not declare an encoding"
        raw.decode("utf-8")               # written as what it claims to be
    finally:
        for p in (os.path.join(project_dir(), quickstart.RESULT),):
            if os.path.exists(p):
                os.remove(p)
        shutil.rmtree(d)


def test_it_survives_a_console_that_cannot_represent_an_em_dash():
    """The only faithful test of encoding is a real interpreter in that locale.

    `open()` resolves its default encoding below the Python level, so patching
    `locale` in-process proves nothing. This runs the launcher in a subprocess
    whose default encoding is ASCII — the situation on any machine whose
    codepage is narrower than the punctuation this tool prints — and requires
    both the page and the run to survive it.
    """
    d = tempfile.mkdtemp()
    out = os.path.join(d, "r.html")
    env = dict(os.environ, PYTHONUTF8="0", LC_ALL="C", LANG="C")
    try:
        p = subprocess.run(
            [sys.executable, "-m", "tools.quickstart", "--demo", "--no-update",
             "--no-open", "--out", out],
            cwd=project_dir(), env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, timeout=300)
        body = p.stdout.decode("utf-8", "replace")
        assert "UnicodeEncodeError" not in body, body[-800:]
        assert p.returncode == 0, body[-800:]
        open(out, encoding="utf-8").read()      # not the machine's codepage
    finally:
        for q in (os.path.join(project_dir(), quickstart.RESULT),):
            if os.path.exists(q):
                os.remove(q)
        shutil.rmtree(d)


def test_the_verdicts_are_saved_where_they_can_be_copied_from():
    """Reading two lines off a scrolled terminal is how results get mistyped."""
    d = tempfile.mkdtemp()
    saved = os.path.join(project_dir(), quickstart.RESULT)
    try:
        _run(["--demo", "--no-update", "--no-open",
              "--out", os.path.join(d, "r.html")], cwd=d)
        assert os.path.exists(saved), f"{quickstart.RESULT} was not written"
        body = open(saved).read()
        assert body.startswith("panel:") and "book :" in body, body
        assert len(body.splitlines()) == 2, "the file should be exactly the result"
    finally:
        if os.path.exists(saved):
            os.remove(saved)
        shutil.rmtree(d)


def test_demo_mode_says_the_numbers_are_not_the_market():
    d = tempfile.mkdtemp()
    try:
        _, out = _run(["--demo", "--no-update", "--no-open",
                       "--out", os.path.join(d, "r.html")], cwd=d)
        assert "not the market" in out, \
            "a fixture verdict printed with nothing marking it as a fixture"
    finally:
        for p in (os.path.join(project_dir(), quickstart.RESULT),):
            if os.path.exists(p):
                os.remove(p)
        shutil.rmtree(d)


def test_demo_mode_never_touches_the_network():
    """--demo is the fallback advice when the vendor is down; it must not fetch."""
    d = tempfile.mkdtemp()
    real = quickstart.fetch_prices.main
    quickstart.fetch_prices.main = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("--demo tried to download"))
    try:
        rc, _ = _run(["--demo", "--no-update", "--no-open",
                      "--out", os.path.join(d, "r.html")], cwd=d)
        assert rc == 0
    finally:
        quickstart.fetch_prices.main = real
        for p in (os.path.join(project_dir(), quickstart.RESULT),):
            if os.path.exists(p):
                os.remove(p)
        shutil.rmtree(d)


# --------------------------------------------------------------------------
# 5. The launcher is the documented experiment, and Windows can reach it
# --------------------------------------------------------------------------

def test_the_launcher_runs_the_pre_registered_parameters():
    """A one-click path that quietly runs a different experiment is worse than
    no one-click path: the numbers would look official and cite a registration
    that does not describe them. So the launcher's parameters must still equal
    the CLI's, and both must equal what PREREGISTRATION.md locked."""
    import argparse
    from tools import rotation_dashboard as rd

    real = argparse.ArgumentParser.parse_args
    seen = {}

    def spy(self, argv=None):
        for act in self._actions:
            seen[act.dest] = act.default
        raise SystemExit(0)

    argparse.ArgumentParser.parse_args = spy
    try:
        rd.main([])
    except SystemExit:
        pass
    finally:
        argparse.ArgumentParser.parse_args = real

    for k, v in quickstart.PARAMS.items():
        assert seen[k] == v, f"{k}: launcher {v}, dashboard default {seen[k]}"

    doc = open(os.path.join(project_dir(), "PREREGISTRATION.md"),
               encoding="utf-8").read()
    for k, v in quickstart.PARAMS.items():
        row = f"| `{k}` | {v} |"
        assert row in doc, f"PREREGISTRATION.md has no row {row!r}"


def test_the_bat_file_pins_its_own_directory_and_holds_the_window():
    bat = os.path.join(project_dir(), "START.bat")
    assert os.path.exists(bat), "START.bat is missing"
    body = open(bat).read()
    assert 'cd /d "%~dp0"' in body, "the bat does not pin its own directory"
    assert "tools.quickstart" in body
    assert "pause" in body, "the window would close before the message is read"


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
