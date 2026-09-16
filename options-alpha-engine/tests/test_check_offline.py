"""Tests for the offline checker (tools/check_offline.py).

A guard that cannot fail is a claim of safety nobody has verified, and this guard
exists because of a defect that already shipped: three tests in test_daily.py
passed on a machine with no outbound access and failed the moment CI ran them with
a working network. So the checker itself is tested both ways — it must catch a file
that reaches an external host, and it must NOT punish one that stands up a local
server, which is the correct way to test a fetcher offline.

Run: python3 tests/test_check_offline.py
"""

import io
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# `test_files` is imported UNDER A NEW NAME on purpose. `_run_all` collects
# everything in globals() starting with "test_", so importing it as-is made the
# runner call the production helper as if it were a test — it took no arguments,
# returned a list, raised nothing, and printed "PASS test_files". A seventh result
# for six tests, and the exact failure this repo keeps finding: work that did not
# happen looking like work that succeeded.
from tools.check_offline import (MARKER, PRINTS_THE_MARKER_ON_PURPOSE, check,
                                main)
from tools.check_offline import test_files as enumerate_test_files

_OFFENDER = '''
import socket, sys
def test_external():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.settimeout(2)
    s.connect(("93.184.216.34", 80)); s.close()
def _run_all():
    try:
        test_external(); print("PASS test_external"); return 0
    except Exception as e:
        print(f"FAIL test_external: {type(e).__name__}: {e}"); return 1
if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
'''

_LOOPBACK_OK = '''
import http.server, socket, sys, threading
def test_local_server():
    srv = http.server.HTTPServer(("127.0.0.1", 0), http.server.BaseHTTPRequestHandler)
    threading.Thread(target=srv.handle_request, daemon=True).start()
    socket.create_connection(("127.0.0.1", srv.server_address[1]), timeout=3).close()
    srv.server_close()
def _run_all():
    try:
        test_local_server(); print("PASS test_local_server"); return 0
    except Exception as e:
        print(f"FAIL test_local_server: {type(e).__name__}: {e}"); return 1
if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
'''


def _write(body):
    """Inside tests/ because the shim runs files by path relative to the repo."""
    fd, path = tempfile.mkstemp(prefix="_probe_", suffix=".py",
                                dir=os.path.join(ROOT, "tests"))
    os.close(fd)
    io.open(path, "w", encoding="utf-8").write(body)
    return os.path.join("tests", os.path.basename(path))


def test_a_file_that_reaches_an_external_host_is_caught_and_named():
    rel = _write(_OFFENDER)
    try:
        bad = check([rel])
        assert len(bad) == 1, bad
        path, rc, hits, lines = bad[0]
        assert path == rel and rc != 0
        assert hits >= 1, f"the refusal was not attributed to the shim: {lines}"
        assert any("93.184.216.34" in l for l in lines), lines
        assert any(MARKER in l for l in lines), lines
    finally:
        os.remove(os.path.join(ROOT, rel))


def test_a_file_that_only_uses_loopback_is_not_punished():
    """Blocking loopback would fail `test_fetch_prices.py`,
    `test_archive_holdings.py` and `test_quickstart.py`, which stand up local HTTP
    servers — the correct way to test a fetcher with no network at all."""
    rel = _write(_LOOPBACK_OK)
    try:
        assert check([rel]) == [], "a local-server test was reported as an offender"
    finally:
        os.remove(os.path.join(ROOT, rel))


def test_the_whole_suite_is_enumerated():
    files = enumerate_test_files()
    assert len(files) >= 40, len(files)
    assert all(f.startswith("tests/test_") and f.endswith(".py") for f in files)
    assert "tests/test_daily.py" in files, "the file the defect shipped in"
    assert "tests/test_check_offline.py" in files, "this file must check itself too"


def test_the_exit_status_is_the_gate():
    rel = _write(_OFFENDER)
    try:
        assert main([rel]) == 1
    finally:
        os.remove(os.path.join(ROOT, rel))
    rel2 = _write(_LOOPBACK_OK)
    try:
        assert main([rel2]) == 0
    finally:
        os.remove(os.path.join(ROOT, rel2))


_SWALLOWS = 'import socket, sys\ndef test_tries_and_shrugs():\n    try:\n        socket.create_connection(("93.184.216.34", 80), timeout=2).close()\n    except Exception:\n        pass                      # the defect: the dependency is hidden\ndef _run_all():\n    test_tries_and_shrugs(); print("PASS test_tries_and_shrugs"); return 0\nif __name__ == "__main__":\n    sys.exit(1 if _run_all() else 0)\n'


def test_the_runner_counts_only_real_tests_in_this_file():
    """The seventh PASS line for six tests. Anything imported under a test_ name is
    collected by `_run_all` and reported as passing, so the count in the output
    stops matching the tests that exist."""
    import ast
    src = io.open(os.path.abspath(__file__), encoding="utf-8").read()
    defined = {n.name for n in ast.parse(src).body
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
               and n.name.startswith("test_")}
    collected = {k for k, v in globals().items()
                 if k.startswith("test_") and callable(v)}
    extra = sorted(collected - defined)
    assert not extra, (f"{extra} would be run as tests but are not defined here — "
                       f"import them under a name that does not start with test_")


_URL_FETCHER = "\n".join([
    "import sys, urllib.request",
    "def test_fetches_a_vendor():",
    "    try:",
    '        urllib.request.urlopen("https://query1.finance.yahoo.com/", timeout=2)',
    "    except Exception:",
    "        pass                      # swallowed, exactly as fetch_prices does",
    "def _run_all():",
    '    test_fetches_a_vendor(); print("PASS test_fetches_a_vendor"); return 0',
    'if __name__ == "__main__":',
    "    sys.exit(1 if _run_all() else 0)",
    "",
])


def test_a_vendor_url_is_refused_even_behind_a_local_proxy():
    """Refusing by socket ADDRESS alone is blind on any machine that routes through
    a local HTTP proxy: the connect target is loopback and the real host never
    appears. That is precisely how a test fetching twelve symbols from Yahoo and
    Stooq looked clean in a proxied sandbox and was caught only on CI. The URL is
    where the intent is, so it is refused there too."""
    rel = _write(_URL_FETCHER)
    try:
        bad = check([rel])
        assert len(bad) == 1, "a vendor URL was not refused"
        path, rc, hits, lines = bad[0]
        assert hits >= 1, lines
        assert any("query1.finance.yahoo.com" in l for l in lines), lines
    finally:
        os.remove(os.path.join(ROOT, rel))


def test_a_file_url_is_not_treated_as_a_vendor():
    """`tools/archive_holdings.py` is tested against `file:///...` sources, which
    touch no network at all. Refusing those would punish the offline fixture."""
    body = "\n".join([
        "import os, sys, tempfile, urllib.request",
        "def test_reads_a_local_file():",
        "    fd, p = tempfile.mkstemp(suffix='.csv'); os.close(fd)",
        "    open(p, 'w').write('a,b\\n1,2\\n')",
        "    urllib.request.urlopen('file://' + p).read()",
        "    os.remove(p)",
        "def _run_all():",
        "    test_reads_a_local_file(); print('PASS test_reads_a_local_file'); return 0",
        "if __name__ == '__main__':",
        "    sys.exit(1 if _run_all() else 0)",
        "",
    ])
    rel = _write(body)
    try:
        assert check([rel]) == [], "a file:// read was reported as a vendor call"
    finally:
        os.remove(os.path.join(ROOT, rel))


def test_a_swallowed_refusal_is_still_caught():
    """The subtler offence: a test that reaches a vendor, catches the error and
    passes anyway. It exits 0, so the exit status alone would miss it, and the
    dependency stays hidden until the day the vendor behaves differently."""
    rel = _write(_SWALLOWS)
    try:
        bad = check([rel])
        assert len(bad) == 1, "a swallowed external attempt went unreported"
        path, rc, hits, lines = bad[0]
        assert rc == 0, "the point of this case is that it exits cleanly"
        assert hits >= 1, lines
    finally:
        os.remove(os.path.join(ROOT, rel))


def test_the_marker_exemption_waives_the_counter_and_nothing_else():
    """This file's child prints the marker on purpose and `main()` relays it, so
    the outer run would otherwise count it as this file's own. The exemption
    waives the COUNT, never the exit status — the file still passes like any
    other, and an offending probe is still caught."""
    assert "tests/test_check_offline.py" in PRINTS_THE_MARKER_ON_PURPOSE
    reason = PRINTS_THE_MARKER_ON_PURPOSE["tests/test_check_offline.py"]
    assert reason.strip(), "every exemption carries a reason"
    rel = _write(_OFFENDER)
    try:
        bad = check([rel])
        assert len(bad) == 1 and bad[0][1] != 0, bad
    finally:
        os.remove(os.path.join(ROOT, rel))


def _run_all():
    tests = [v for k, v in globals().items()
             if k.startswith("test_") and callable(v)]
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
