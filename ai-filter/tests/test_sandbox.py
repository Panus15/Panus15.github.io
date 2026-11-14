"""Tests for engine/sandbox.py. Offline (run=True tests spawn real local Python
subprocesses; the network test only proves the guard refuses).

Run: python3 tests/test_sandbox.py
"""
import os, sys, time, tempfile, uuid
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.models import Claim, ClaimKind, Status
from engine import sandbox

V, U, C = Status.VERIFIED, Status.UNVERIFIED, Status.CONTRADICTED
_n = [0]


def claim(code, language="python", expected=None, kind=ClaimKind.CODE):
    _n[0] += 1
    return Claim(f"c{_n[0]}", kind, code, 0, len(code),
                 {"language": language, "code": code, "expected_output": expected})


def one(code, language="python", expected=None, run=True, timeout=5.0):
    out = sandbox.check([claim(code, language, expected)], run=run, timeout=timeout)
    assert len(out) == 1 and out[0].module == "sandbox"
    return out[0]


# ---------------------------------------------------------------- no-run mode

def test_run_false_never_spawns_a_subprocess():
    real = sandbox.subprocess.Popen
    calls = []

    def boom(*a, **k):
        calls.append(a)
        raise AssertionError("Popen must not be called when run=False")
    sandbox.subprocess.Popen = boom
    try:
        f = one("print('hi')", run=False)
        assert f.status == U and "not executed" in f.reason and "--run-code" in f.reason
        f = one("def f(:\n  pass", run=False)
        assert f.status == C
        f = one(">>> 1 + 1\n2\n", run=False)
        assert f.status == U
        assert one("x", language="rust", run=False).status == U
    finally:
        sandbox.subprocess.Popen = real
    assert not calls


def test_syntax_error_is_contradicted_with_line():
    for run in (False, True):
        f = one("x = 1\nprint('a'\ny = 2\n", run=run)
        assert f.status == C, f
        assert "parse" in f.reason.lower()
        assert f.evidence and "line" in f.evidence[0]
    assert one("def f(:\n    pass", run=False).status == C
    assert one("print 'hello'", run=False).status == C       # python 2 syntax


def test_parse_ok_is_never_verified_without_running():
    assert one("print(1)", run=False, expected="1").status == U


def test_non_python_language_and_odd_inputs():
    for lang in ("bash", "javascript", "rust", "json"):
        f = one("echo hi", language=lang)
        assert f.status == U and "only python" in f.reason.lower()
    assert one("", run=True).status == U
    assert one("x=1", language="Python", run=False).status == U          # case-insensitive, parses
    assert one("x = 1\x00", run=False).status == U or True               # null byte: no crash
    assert sandbox.check([claim("1+1", kind=ClaimKind.URL)], run=True) == []


def test_empty_language_only_python_if_it_parses():
    f = one("$ ls -la", language="", run=False)
    assert f.status == U and "language" in f.reason.lower()
    f = one("x = 1", language="", run=False)
    assert f.status == U and "not executed" in f.reason
    assert one("print('hi')", language="", run=True).status == V
    # unlabelled blocks are never called wrong, they may not be Python at all
    f = one("true", language="", run=True)
    assert f.status == U, f
    assert one("print('a')", language="", expected="b", run=True).status == U


# ------------------------------------------------------------------- run mode

def test_print_ok_verified_with_caveat():
    f = one("print('OK')")
    assert f.status == V and "does not prove" in f.reason
    assert one("print('OK')", language="py").status == V
    assert one("print('OK')", language="python3").status == V


def test_expected_output_comparison():
    assert one("print('OK')", expected="OK").status == V
    assert one("print('a')\nprint('b')", expected="a  \r\nb\r\n\n").status == V   # whitespace/CRLF normalised
    f = one("print('4')", expected="5")
    assert f.status == C
    assert any("4" in e for e in f.evidence) and any("5" in e for e in f.evidence)
    assert one("pass", expected="").status == V
    assert one("print('x')", expected="").status == C


def test_exception_is_contradicted():
    f = one("x = [1]\nprint(x[3])")
    assert f.status == C and "IndexError" in f.reason
    f = one("raise ValueError('bad thing')")
    assert f.status == C and "ValueError: bad thing" in f.reason
    assert f.evidence and "<snippet>" in f.evidence[0] and "line 1" in f.evidence[0]
    assert one("import os\nos.nothing_here").status == C
    assert one("from os import nothing_here").status == C
    assert one("return 5").status == C       # SyntaxError at compile time, same story


def test_missing_third_party_module_is_unverified():
    f = one("import definitely_not_a_real_pkg_xyz\nprint(1)")
    assert f.status == U and "definitely_not_a_real_pkg_xyz" in f.reason
    f = one("from nopkg_abc.sub import thing")
    assert f.status == U and "nopkg_abc" in f.reason


def test_network_is_refused_and_unverified():
    flag = os.path.join(tempfile.gettempdir(), "aifilter-net-" + uuid.uuid4().hex)
    code = (
        "import socket\n"
        "try:\n"
        f"    open({flag!r}, 'w').write('before')\n"
        "    socket.create_connection(('example.com', 80), timeout=1)\n"
        "except PermissionError:\n"
        "    raise\n"
        "print('connected?!')\n")
    try:
        f = one(code)
        assert f.status == U and "network" in f.reason.lower(), f
        assert "connected" not in "".join(f.evidence)
    finally:
        if os.path.exists(flag):
            os.remove(flag)
    for body in ("import socket; socket.socket()",
                 "import socket; socket.getaddrinfo('example.com', 80)",
                 "import socket; socket.gethostbyname('example.com')",
                 "import urllib.request; urllib.request.urlopen('http://example.com/')",
                 "import http.client; http.client.HTTPConnection('example.com').request('GET', '/')"):
        f = one(body, timeout=4)
        assert f.status == U and "network" in f.reason.lower(), (body, f)
    # swallowing the error does not turn it into VERIFIED
    f = one("import socket\ntry:\n    socket.socket()\nexcept OSError:\n    pass\nprint('ok')")
    assert f.status == U
    # but a socketpair (used by asyncio itself) still works
    assert one("import asyncio\nasync def m():\n    return 3\nprint(asyncio.run(m()))", expected="3").status == V


def test_infinite_loop_is_killed_and_unverified():
    t = time.time()
    f = one("while True:\n    pass", timeout=1.0)
    el = time.time() - t
    assert f.status == U and "did not finish" in f.reason, f
    assert el < 4.0, el
    t = time.time()
    f = one("import time\ntime.sleep(60)", timeout=0.5)
    assert f.status == U and time.time() - t < 4.0


def test_child_processes_are_killed_with_the_group():
    mark = "aifilter-child-" + uuid.uuid4().hex
    code = ("import subprocess, sys, time\n"
            f"subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)  # {mark}'])\n"
            "time.sleep(60)\n")
    f = one(code, timeout=1.0)
    assert f.status == U
    time.sleep(0.3)
    import subprocess
    ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    assert mark not in ps


def test_memory_hog_does_not_take_down_runner():
    f = one("x = 'a' * 10**10\nprint(len(x))", timeout=5)
    assert f.status == U, f
    assert "memory" in f.reason.lower() or "killed" in f.reason.lower()
    f = one("x = []\nwhile True:\n    x.append(bytearray(10**7))", timeout=5)
    assert f.status == U, f


def test_output_flood_is_truncated():
    f = one("import sys\nfor i in range(2000):\n    print('x' * 1000)\nsys.stderr.write('e' * 100000)")
    assert f.status == V, f
    f = one("for i in range(2000):\n    print('x' * 1000)", expected="x")
    assert f.status == U and "10 KB" in f.reason
    f = one("for i in range(2000):\n    print('x' * 1000)\nraise RuntimeError('boom')")
    assert f.status == C
    assert sum(len(e) for e in f.evidence) < 6000
    f = one("print('a'*50000)\nraise SystemExit(0)", expected="b" * 10)
    assert f.status == U


def test_file_writes_stay_out_of_repo_cwd():
    name = "sbx_written_" + uuid.uuid4().hex + ".txt"
    cwd_before = os.getcwd()
    f = one(f"open({name!r}, 'w').write('hello')\nimport os\nprint(os.path.exists({name!r}))", expected="True")
    assert f.status == V, f
    assert not os.path.exists(os.path.join(cwd_before, name))
    assert not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), name))
    # the working directory is a temp dir, not ours
    f = one("import os\nprint(os.getcwd())")
    assert f.status == V
    f = one("import os\nprint(os.getcwd() != " + repr(cwd_before) + ")", expected="True")
    assert f.status == V


def test_environment_is_not_inherited():
    os.environ["AIFILTER_SECRET_SENTINEL"] = "s3cret-" + uuid.uuid4().hex
    try:
        f = one("import os\nprint(os.environ.get('AIFILTER_SECRET_SENTINEL'))", expected="None")
        assert f.status == V, f
        f = one("import os\nprint(sorted(os.environ))")
        assert f.status == V
        f = one("import os\nprint('HOME' in os.environ or 'PATH' in os.environ)", expected="False")
        assert f.status == V, f
    finally:
        del os.environ["AIFILTER_SECRET_SENTINEL"]


def test_stdin_is_empty():
    f = one("name = input('name? ')\nprint(name)")
    assert f.status == U and "input" in f.reason.lower(), f
    assert one("import sys\nprint(repr(sys.stdin.read()))", expected="''").status == V


def test_sys_exit_codes():
    assert one("import sys\nprint('x')\nsys.exit(0)", expected="x").status == V
    f = one("import sys\nsys.exit(3)")
    assert f.status == U and "3" in f.reason
    assert one("import sys\nsys.exit('usage: tool FILE')").status == U


def test_isolated_mode_ignores_cwd_and_pythonpath_shadowing():
    f = one("import json\nprint(json.__file__.endswith('json/__init__.py'))", expected="True")
    assert f.status == V, f
    f = one("open('json.py', 'w').write('raise RuntimeError')\nimport json\nprint(json.dumps([1]))", expected="[1]")
    assert f.status == V, f


def test_main_guard_and_name():
    code = "def f():\n    return 41\nif __name__ == '__main__':\n    print(f() + 1)"
    assert one(code, expected="42").status == V


def test_doctest_style_snippet():
    f = one(">>> 1 + 1\n2\n>>> print('hi')\nhi\n")
    assert f.status == V, f
    f = one(">>> 1 + 1\n3\n")
    assert f.status == C, f
    f = one(">>> for i in range(2):\n...     print(i)\n...\n0\n1\n")
    assert f.status == V, f
    f = one(">>> x = 5\n>>> x * 2\n10\n")
    assert f.status == V
    assert one(">>> 1 +\n2\n", run=False).status == C                 # real syntax error in the REPL line
    assert one(">>> 1/0\nTraceback (most recent call last):\n  ...\nZeroDivisionError: x\n").status == U
    assert one(">>> import random\n>>> random.random()\n0.5\n...").status in (U, V)
    assert one(">>> object()\n<object object at 0x7f>\n").status == V   # unstable output is not compared
    assert one(">>> 1 + 1\n2\n", language="", run=False).status == U


def test_sandbox_internal_failure_is_unverified():
    real = sandbox.subprocess.Popen

    def boom(*a, **k):
        raise OSError("cannot fork")
    sandbox.subprocess.Popen = boom
    try:
        f = one("print(1)")
        assert f.status == U and "sandbox" in f.reason.lower()
    finally:
        sandbox.subprocess.Popen = real
    # crazy parameters do not crash
    assert one("print(1)", timeout=float("nan")).status == V
    assert one("print(1)", timeout=-5).status in (V, U)


def test_one_finding_per_code_claim_in_order():
    a, b, c = claim("print(1)"), claim("print(1)", language="sh"), claim("print(1)", expected="2")
    other = claim("x", kind=ClaimKind.VAGUE)
    out = sandbox.check([a, other, b, c], run=True)
    assert [f.claim.id for f in out] == [a.id, b.id, c.id]
    assert [f.status for f in out] == [V, U, C]


if __name__ == "__main__":
    fails = 0
    t0 = time.time()
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("ok  ", name)
            except Exception as e:
                fails += 1; print("FAIL", name, repr(e))
    print(f"{time.time() - t0:.1f}s")
    sys.exit(1 if fails else 0)
