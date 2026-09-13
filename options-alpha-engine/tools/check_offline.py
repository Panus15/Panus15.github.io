"""Prove the test suite never reaches an external host.

WHY THIS EXISTS. Three tests in `tests/test_daily.py` passed for months of wall
clock on a machine with no outbound access, and failed the moment CI ran them with
a working network: one asserted that a vendor was unreachable, one had its fixture
CSV overwritten by a real download, and one had live vendor entries appended to its
ledger mid-test. The suite was making real API calls and the absence of a network
was the only thing hiding it.

A test that behaves differently depending on whether the machine it runs on has
outbound access is not a test, it is a coin flip with a changelog. So this runs
every test file with EXTERNAL connections blocked and names any that needs one.

WHY LOOPBACK STAYS OPEN. A blunt socket block also breaks several tests here that
stand up a local HTTP server and fetch from 127.0.0.1 — `test_fetch_prices.py`,
`test_archive_holdings.py`, `test_quickstart.py` — and those are exactly the right
way to test a fetcher offline. Blocking them would punish the correct pattern. The
narrower question is the one that matters: does any test reach a host that is not
this machine?

    python3 -m tools.check_offline              # all test files
    python3 -m tools.check_offline tests/test_daily.py

A file that exits non-zero is always an offence. A file that exits 0 but printed the
marker attempted an external connection and SWALLOWED the refusal, which is caught
too — `PRINTS_THE_MARKER_ON_PURPOSE` lists the one file where that is the point.

Exit status is 0 only if every file passed with external access denied.

A NOTE ON WHAT THE COUNTER MEASURES. The shim refuses by ADDRESS, at connect time,
so a urllib call to a hostname is caught after DNS as an IP — except where the
machine routes through a local HTTP proxy, in which case the connect target IS
loopback and the request dies at the proxy instead. Both outcomes fail the file,
which is why the exit status is the gate and the external-attempt count is only a
hint about where the refusal happened.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Injected ahead of the test module. Subclassing the real socket rather than
#: replacing it keeps loopback servers working, and raising by NAME means an
#: offender fails with a message instead of whatever the vendor's timeout does.
SHIM = '''
import ipaddress, runpy, socket, sys
_real_socket, _real_create = socket.socket, socket.create_connection

def _is_local(host):
    if host in (None, "", "localhost", "localhost.localdomain"):
        return True
    try:
        return ipaddress.ip_address(str(host)).is_loopback
    except ValueError:
        return False

def _refuse(how, host):
    # RECORDED BEFORE IT IS RAISED. A test that catches the exception and passes
    # anyway leaves no trace in the exception path, so counting printed messages
    # alone could never see it — the refusal has to be written here, not left to
    # whatever the caller does with the error.
    msg = "EXTERNAL NETWORK: %s to %s" % (how, host)
    print(msg, file=sys.stderr, flush=True)
    raise RuntimeError(msg)

class _Sock(_real_socket):
    def connect(self, addr):
        if isinstance(addr, tuple) and not _is_local(addr[0]):
            _refuse("connect", addr[0])
        return super().connect(addr)
    def connect_ex(self, addr):
        if isinstance(addr, tuple) and not _is_local(addr[0]):
            _refuse("connect_ex", addr[0])
        return super().connect_ex(addr)

def _create(addr, *a, **kw):
    if isinstance(addr, tuple) and not _is_local(addr[0]):
        _refuse("create_connection", addr[0])
    return _real_create(addr, *a, **kw)

socket.socket, socket.create_connection = _Sock, _create
sys.argv = [sys.argv[1]]
runpy.run_path(sys.argv[0], run_name="__main__")
'''

MARKER = "EXTERNAL NETWORK"

#: Files allowed to PRINT the marker while still exiting 0. Exactly one belongs
#: here and it needs a reason, so adding to this list is a decision rather than a
#: shortcut: the checker's own test spawns a child that deliberately reaches an
#: external host to prove the shim refuses it, and `main()` prints that child's
#: refusal — which the outer run then counts as its own. The file still has to
#: exit 0; the exemption waives the marker count, never the exit status.
PRINTS_THE_MARKER_ON_PURPOSE = {
    "tests/test_check_offline.py": "probes an external host to prove the shim works",
}


def check(paths: list) -> list:
    """Run each path with external access denied. Returns a list of offenders as
    ``(path, returncode, n_external_attempts, first_lines)``."""
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    bad = []
    for p in paths:
        r = subprocess.run([sys.executable, "-c", SHIM, p], cwd=ROOT, env=env,
                           capture_output=True, text=True)
        out = (r.stdout or "") + (r.stderr or "")
        hits = out.count(MARKER)
        # A non-zero exit is always an offence. A marker on a file that exited 0
        # means a test ATTEMPTED an external connection and SWALLOWED the refusal —
        # caught because the shim records every refusal on stderr before raising,
        # so catching the exception does not erase the evidence. Exempt only where
        # the attempt is the point.
        exempt = p.replace(os.sep, "/") in PRINTS_THE_MARKER_ON_PURPOSE
        if r.returncode != 0 or (hits and not exempt):
            lines = [l for l in out.splitlines()
                     if l.startswith("FAIL") or MARKER in l][:4]
            bad.append((p, r.returncode, hits, lines))
    return bad


def test_files() -> list:
    d = os.path.join(ROOT, "tests")
    return [os.path.join("tests", f) for f in sorted(os.listdir(d))
            if f.startswith("test_") and f.endswith(".py")]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="*", help="test files (default: all of them)")
    a = ap.parse_args(argv)
    paths = a.paths or test_files()
    print(f"running {len(paths)} file(s) with external network denied "
          f"(loopback still allowed)\n")
    bad = check(paths)
    if not bad:
        print(f"OK — all {len(paths)} file(s) pass with no external access.\n"
              f"   The suite does not depend on, and does not use, a network.")
        return 0
    print(f"{len(bad)} file(s) need a network they should not need:\n")
    for p, rc, hits, lines in bad:
        print(f"  {p}   exit {rc}, {hits} external attempt(s)")
        for l in lines:
            print(f"      {l[:150]}")
    print("\nA test that passes only where outbound access exists (or only where it\n"
          "does NOT) is a coin flip. Use a local server fixture, or deny the network\n"
          "inside the test so the dependency fails by name.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
