"""Tests for UPDATE.bat, the double-click code updater.

The person this is for runs the engine on Windows from a folder that also holds
their records: the paper ledger, the log, the scheduled job's wrapper, the
dashboard, cached prices, archived fund books. An updater that touched any of
those would destroy the forward test the whole repo is waiting on. So these
tests pin what it copies, that it never deletes, and that it can overwrite
itself without cmd.exe misreading the rest of the file.

No Windows needed: they read the file as text.

Run: python3 tests/test_update_bat.py
"""

import fnmatch
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BAT = os.path.join(HERE, "UPDATE.bat")

#: What lives beside the code on the user's machine and must survive an update.
RECORDS = ["btc.jsonl", "eth.jsonl", "daily.log", "daily-job.bat", "dashboard.html",
           "BTC.prices.json", "basket.csv", "prices", "holdings", "data"]


def _raw():
    with open(BAT, "rb") as fh:
        return fh.read()


def _text():
    return _raw().decode("ascii")


def _code_dirs():
    m = re.search(r"for %%F in \(([^)]*)\) do", _text())
    assert m, "the per-folder copy loop is missing"
    return m.group(1).split()


def _top_patterns():
    m = re.search(r'robocopy "%SRC%" "%DEST%" ([^/\r\n]*?) /', _text())
    assert m, "the top-level copy line is missing"
    return m.group(1).split()


def _tracked():
    try:
        out = subprocess.run(["git", "ls-files"], cwd=HERE, capture_output=True,
                             text=True, check=True).stdout.split()
    except (OSError, subprocess.CalledProcessError):
        out = []
    if not out:            # not a git checkout: what is on disk is the release
        out = []
        for name in os.listdir(HERE):
            out += ([f"{name}/x"] if os.path.isdir(os.path.join(HERE, name)) else [name])
    return out


def test_batch_files_are_crlf_and_ascii():
    """cmd.exe finds labels by scanning in 512-byte chunks; in an LF-only file a
    `goto` can land mid-line. Every .bat here is CRLF, and .gitattributes stops
    git from converting them."""
    for name in ("UPDATE.bat", "START.bat"):
        with open(os.path.join(HERE, name), "rb") as fh:
            raw = fh.read()
        raw.decode("ascii")
        assert raw.count(b"\n") == raw.count(b"\r\n") > 0, name
    with open(os.path.join(HERE, ".gitattributes")) as fh:
        assert "*.bat -text" in fh.read()


def test_it_never_deletes_from_the_users_folder():
    code = "\n".join(l for l in _text().splitlines()
                     if not l.strip().lower().startswith("rem"))
    for flag in ("/MIR", "/PURGE", "/MOV", "/MOVE"):
        assert flag not in code.upper(), flag
    for line in _text().splitlines():
        low = line.strip().lower()
        if low.startswith("rem"):
            continue
        if re.search(r"\b(rmdir|rd|del|erase)\b", low):
            assert "%work%" in low or "%zip%" in low, line
            assert "%dest%" not in low, line


def test_it_copies_exactly_the_code_folders_that_exist():
    dirs = _code_dirs()
    assert sorted(dirs) == sorted(["engine", "models", "tools", "tests", "mutation", "docs"])
    tracked_dirs = {p.split("/")[0] for p in _tracked() if "/" in p}
    assert set(dirs) == tracked_dirs, (set(dirs) ^ tracked_dirs)


def test_every_shipped_top_level_file_is_covered():
    """A new top-level file that no pattern matches would silently never reach
    the user. Dotfiles are git's, not the user's."""
    pats = _top_patterns()
    for p in _tracked():
        if "/" in p or p.startswith("."):
            continue
        assert any(fnmatch.fnmatch(p, pat) for pat in pats), (p, pats)


def test_no_record_matches_anything_it_copies():
    pats = _top_patterns()
    dirs = _code_dirs()
    for r in RECORDS:
        assert not any(fnmatch.fnmatch(r, pat) for pat in pats), r
        assert r not in dirs, r


def test_it_runs_from_a_temp_copy_because_it_overwrites_itself():
    t = _text()
    assert "UPDATE.bat" in _top_patterns()
    head = t.split('set "DEST=%~2"')[0]
    assert '--from-temp' in head and 'copy /y "%~f0" "%TEMP%\\oae_update.bat"' in head
    # no CALL: control must not come back into the file being overwritten
    hop = [l for l in head.splitlines() if l.strip().startswith('"%TEMP%\\oae_update.bat"')]
    assert len(hop) == 1, head
    assert "call " not in head.lower()


def test_the_destination_loses_its_trailing_backslash():
    """%~dp0 ends in a backslash; before a closing quote robocopy reads `\\"` as
    an escaped quote and copies to the wrong place."""
    assert 'if "%DEST:~-1%"=="\\" set "DEST=%DEST:~0,-1%"' in _text()


def test_it_refuses_a_folder_that_is_not_the_engine():
    t = _text()
    assert 'if not exist "%DEST%\\engine\\pricing.py"' in t
    assert 'if not exist "%SRC%\\engine\\pricing.py" goto badsrc' in t


def test_robocopy_failure_is_exit_8_or_more():
    """robocopy exits 1 for "files copied". Only 8+ is a failure."""
    t = _text()
    assert t.count("if errorlevel 8 set \"FAILED=1\"") == 2
    assert "if errorlevel 1" not in t.split("[3/3]")[1].split("code imports")[0]


def test_it_downloads_this_branch_and_pauses():
    t = _text()
    assert 'set "BRANCH=claude/ml-options-trading-engine-b471vm"' in t
    assert "github.com/Panus15/Panus15.github.io/archive/refs/heads/%BRANCH%.zip" in t
    assert t.rstrip().splitlines()[-2:] == ["pause", "exit /b"]


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
