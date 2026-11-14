"""Sandbox checker: does a Python code block parse, and (opt-in) does it run?

check(claims, *, run=False, timeout=5.0) -> list[Finding], one per CODE claim,
module="sandbox".  Other kinds are ignored.

Without run=True nothing is ever executed and no subprocess is created.  The
code is only parsed: a SyntaxError is positive evidence the block is broken
(CONTRADICTED); code that parses is UNVERIFIED ("syntax OK; not executed"),
because parsing proves very little.

With run=True the code runs in a child interpreter:
  * `python -I -B -X utf8` (isolated mode: no PYTHON* env vars, no user site, no
    cwd / script dir on sys.path).  -S is NOT used, so site-packages that are
    installed (numpy, ...) still import and the code can really run.
  * cwd = a fresh temporary directory (deleted afterwards), stdin = /dev/null,
    environment = {"PYTHONIOENCODING": "utf-8"} only (no inherited secrets).
  * own session / process group; on timeout the whole group gets SIGKILL.
  * rlimits, applied inside the child before user code runs (not via preexec_fn,
    which is unsafe in threaded parents): CPU = timeout+1 s, address space
    512 MB, file size 10 MB, core dumps 0, and (Linux, best effort) NPROC a bit
    above the user's current task count.  NPROC is not enforced for root.
  * socket.socket / create_connection / getaddrinfo / gethostbyname* / create_server
    are replaced by functions raising PermissionError before the code runs.
  * stdout / stderr are read through pipes and kept only up to 10 KB each.
  * The code is exec()'d from a file under the name "<snippet>" so tracebacks
    point at the user's own line numbers.

HONEST LIMITS: this is defence in depth against accidents (runaway loops, huge
allocations, a stray network call, scribbling in the working directory).  It is
NOT a security boundary against hostile code: the child runs as the same OS user,
can read any file that user can read, can `import _socket` or ctypes around the
guard, and can spawn processes that escape the process group.  For untrusted
input run the whole tool inside Docker / gVisor / a VM.

Verdict policy (THE ONE RULE: our own limits are UNVERIFIED, never CONTRADICTED):
  VERIFIED     exited 0 (and, if expected_output was given, stdout matched after
               normalising CRLF and trailing whitespace).  "Ran without error"
               does not prove the code is correct, and the reason says so.
  CONTRADICTED does not parse; raised an ordinary exception; or stdout differs
               from expected_output.
  UNVERIFIED   not Python / language unknown, not executed, timeout or killed by
               a signal, MemoryError (our 512 MB cap), missing third-party
               package, needs network, needs input(), non-zero sys.exit(),
               output over 10 KB with an expected output to compare, any
               internal failure.  A code block with NO language label is only
               treated as Python if it parses, and any bad outcome for it is
               UNVERIFIED (it could be JSON, pseudo-code ...).
REPL transcripts (">>> " prompts) are run statement by statement in
interactive mode; the transcript's own output lines are used as expected output
only when they are simple (no "...", no 0x addresses, no tracebacks).
"""

from __future__ import annotations

import ast
import doctest
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import threading
from typing import Sequence

from .models import Claim, ClaimKind, Finding, Status

MODULE = "sandbox"
PYTHON_LANGS = {"python", "py", "python3", ""}
MAX_CODE_CHARS = 200_000
OUTPUT_CAP = 10 * 1024
EVIDENCE_CHARS = 2000
MEM_LIMIT = 512 * 1024 * 1024
FSIZE_LIMIT = 10 * 1024 * 1024
NETWORK_MESSAGE = "network access is disabled in the sandbox"

# Runs inside the child.  Kept as a separate string; user code never shares a
# source file with it (it is read from spec.json and exec'd as "<snippet>").
_RUNNER = r'''
import json, os, sys, traceback, linecache, socket

spec_path, ctl_dir = sys.argv[1], sys.argv[2]
with open(spec_path, encoding="utf-8") as f:
    spec = json.load(f)

_marker = os.path.join(ctl_dir, "network_denied")

def _note():
    try:
        with open(_marker, "w") as f:
            f.write("1")
    except Exception:
        pass

class _NetworkDenied(PermissionError):
    pass

def _deny(*a, **k):
    _note()
    raise _NetworkDenied(1, spec["network_message"])

_Real = socket.socket
class _GuardedSocket(_Real):
    __slots__ = ()
    def __init__(self, family=-1, type=-1, proto=-1, fileno=None):
        if fileno is None:
            _deny()
        super().__init__(family, type, proto, fileno)

socket.socket = _GuardedSocket
for _name in ("create_connection", "getaddrinfo", "gethostbyname", "gethostbyname_ex",
              "gethostbyaddr", "getnameinfo", "create_server"):
    if hasattr(socket, _name):
        setattr(socket, _name, _deny)

try:
    import resource
    for _name, _val in spec["limits"].items():
        _lim = getattr(resource, _name, None)
        if _lim is not None:
            try:
                resource.setrlimit(_lim, (_val, _val))
            except (ValueError, OSError):
                pass
except ImportError:
    pass

def _write_result(d):
    try:
        with open(os.path.join(ctl_dir, "result.json"), "w", encoding="utf-8") as f:
            json.dump(d, f)
    except Exception:
        pass

sources = spec["sources"]
if spec["mode"] == "exec" and len(sources) == 1:
    lines = sources[0].splitlines(True)
    linecache.cache["<snippet>"] = (len(sources[0]), None, lines, "<snippet>")

ns = {"__name__": "__main__", "__file__": "<snippet>"}
sys.argv = ["<snippet>"]
code = 0
try:
    for src in sources:
        exec(compile(src, "<snippet>", spec["mode"]), ns)
except SystemExit as e:
    c = e.code
    if c is None:
        code = 0
    elif isinstance(c, int):
        code = c
    else:
        try:
            sys.stderr.write(str(c) + "\n")
        except Exception:
            pass
        code = 1
except BaseException as e:
    try:
        msg = str(e)
    except Exception:
        msg = "<unprintable>"
    info = {"type": type(e).__name__, "message": msg[:500],
            "import_name": getattr(e, "name", None) if isinstance(e, ImportError) else None,
            "network": isinstance(e, _NetworkDenied)}
    _write_result(info)
    try:
        traceback.print_exception(type(e), e, e.__traceback__.tb_next if e.__traceback__ else None)
    except Exception:
        pass
    code = 1
try:
    sys.stdout.flush()
    sys.stderr.flush()
except Exception:
    pass
sys.exit(code)
'''


# ------------------------------------------------------------------ helpers --

def _mk(claim: Claim, status: Status, reason: str, evidence: tuple = ()) -> Finding:
    return Finding(claim, status, MODULE, reason, evidence)


def _clip(s: str, n: int = EVIDENCE_CHARS, tail: bool = False) -> str:
    if len(s) <= n:
        return s
    return ("[...] " + s[-n:]) if tail else (s[:n] + " [...]")


def _normalise(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in text.split("\n")).strip()


def _task_count() -> int | None:
    """Best-effort number of tasks (processes + threads) owned by our uid."""
    try:
        uid = os.getuid()
        total = 0
        for d in os.listdir("/proc"):
            if d.isdigit():
                try:
                    if os.stat(f"/proc/{d}").st_uid == uid:
                        total += len(os.listdir(f"/proc/{d}/task"))
                except OSError:
                    pass
        return total or None
    except (OSError, AttributeError):
        return None


class _Capture(threading.Thread):
    """Drain a pipe, keeping only the first OUTPUT_CAP bytes."""

    def __init__(self, stream) -> None:
        super().__init__(daemon=True)
        self.stream = stream
        self.buf = bytearray()
        self.truncated = False

    def run(self) -> None:
        try:
            while True:
                chunk = self.stream.read1(65536)
                if not chunk:
                    return
                room = OUTPUT_CAP - len(self.buf)
                if room > 0:
                    self.buf += chunk[:room]
                if len(chunk) > room:
                    self.truncated = True
        except (OSError, ValueError):
            return

    def text(self) -> str:
        return bytes(self.buf).decode("utf-8", errors="replace")


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        if hasattr(os, "killpg"):
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except (ProcessLookupError, PermissionError, OSError):
        pass


class _RunResult:
    def __init__(self) -> None:
        self.timed_out = False
        self.returncode: int | None = None
        self.stdout = ""
        self.stderr = ""
        self.stdout_truncated = False
        self.exc: dict | None = None
        self.network = False


def _execute(sources: list[str], mode: str, timeout: float) -> _RunResult:
    res = _RunResult()
    cpu = int(math.ceil(timeout)) + 1
    limits = {"RLIMIT_CPU": cpu, "RLIMIT_AS": MEM_LIMIT, "RLIMIT_FSIZE": FSIZE_LIMIT, "RLIMIT_CORE": 0}
    tasks = _task_count()
    if tasks:
        limits["RLIMIT_NPROC"] = tasks + 64
    with tempfile.TemporaryDirectory(prefix="aifilter-sbx-", ignore_cleanup_errors=True) as tmp:
        work = os.path.join(tmp, "work")
        ctl = os.path.join(tmp, "ctl")
        os.mkdir(work)
        os.mkdir(ctl)
        runner = os.path.join(ctl, "runner.py")
        spec_path = os.path.join(ctl, "spec.json")
        with open(runner, "w", encoding="utf-8") as f:
            f.write(_RUNNER)
        with open(spec_path, "w", encoding="utf-8") as f:
            json.dump({"sources": sources, "mode": mode, "limits": limits,
                       "network_message": NETWORK_MESSAGE}, f)
        env = {"PYTHONIOENCODING": "utf-8"}
        if os.name == "nt":  # pragma: no cover - POSIX is the supported target
            env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", "")
        proc = subprocess.Popen(
            [sys.executable, "-I", "-B", "-X", "utf8", runner, spec_path, ctl],
            cwd=work, env=env, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            close_fds=True, start_new_session=True)
        out_t, err_t = _Capture(proc.stdout), _Capture(proc.stderr)
        out_t.start()
        err_t.start()
        try:
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                res.timed_out = True
                _kill_group(proc)
                proc.wait()
            else:
                _kill_group(proc)          # stragglers the code left behind
        finally:
            if proc.poll() is None:
                _kill_group(proc)
                proc.wait()
        out_t.join(2)
        err_t.join(2)
        res.returncode = proc.returncode
        res.stdout, res.stderr = out_t.text(), err_t.text()
        res.stdout_truncated = out_t.truncated
        res.network = os.path.exists(os.path.join(ctl, "network_denied"))
        try:
            with open(os.path.join(ctl, "result.json"), encoding="utf-8") as f:
                data = json.load(f)
            res.exc = data if isinstance(data, dict) else None
        except (OSError, ValueError):
            res.exc = None
    return res


# ------------------------------------------------------------- classification --

def _sig_name(n: int) -> str:
    try:
        return signal.Signals(n).name
    except ValueError:
        return f"signal {n}"


def _classify(claim: Claim, res: _RunResult, expected: str | None, timeout: float,
              unlabelled: bool, note: str) -> Finding:
    def bad(status_if_labelled: Status, reason: str, evidence: tuple = ()) -> Finding:
        # unlabelled blocks are never called wrong: they may not be Python at all
        if unlabelled and status_if_labelled is Status.CONTRADICTED:
            return _mk(claim, Status.UNVERIFIED, reason + " (no language was given, so this is not treated as an error)", evidence)
        return _mk(claim, status_if_labelled, reason, evidence)

    if res.timed_out:
        return _mk(claim, Status.UNVERIFIED, f"The code did not finish within {timeout:g} s, so it was stopped (this may be our limit, not a bug).")
    rc = res.returncode
    if rc is not None and rc < 0:
        return _mk(claim, Status.UNVERIFIED,
                   f"The code was killed by {_sig_name(-rc)}, probably by the sandbox's resource limits, so it could not be checked.")
    exc = res.exc
    if rc not in (0, None) and exc:
        etype, msg = str(exc.get("type")), str(exc.get("message", ""))
        ev = ((_clip(res.stderr, tail=True),) if res.stderr else ())
        if res.network or exc.get("network"):
            return _mk(claim, Status.UNVERIFIED, "The code needs network access, which the sandbox disables, so it could not be checked.", ev)
        if etype in ("ModuleNotFoundError", "ImportError") and exc.get("import_name"):
            name = str(exc["import_name"])
            top = name.split(".")[0]
            stdlib = top in sys.stdlib_module_names
            if not stdlib:
                return _mk(claim, Status.UNVERIFIED, f"The code needs the third-party package '{top}', which is not installed here, so it could not be checked.", ev)
            if etype == "ModuleNotFoundError" and name == top:
                return _mk(claim, Status.UNVERIFIED, f"The code needs the standard-library module '{top}', which is not available in this Python build.", ev)
        if etype == "MemoryError":
            return _mk(claim, Status.UNVERIFIED, "The code ran out of memory under the sandbox's 512 MB limit, so it could not be checked.", ev)
        if etype == "EOFError" and "EOF when reading a line" in msg:
            return _mk(claim, Status.UNVERIFIED, "The code needs interactive input (input()), which is not available, so it could not be checked.", ev)
        if "can't start new thread" in msg or "Resource temporarily unavailable" in msg:
            return _mk(claim, Status.UNVERIFIED, "The code hit the sandbox's process/thread limit, so it could not be checked.", ev)
        if "File too large" in msg or "Disk quota" in msg:
            return _mk(claim, Status.UNVERIFIED, "The code hit the sandbox's file-size limit, so it could not be checked.", ev)
        return bad(Status.CONTRADICTED, f"The code raised {etype}: {_clip(msg, 200)}", ev)
    if rc not in (0, None):
        return _mk(claim, Status.UNVERIFIED, f"The code exited with status {rc} without a Python error, so it could not be checked.",
                   ((_clip(res.stderr, tail=True),) if res.stderr else ()))
    # exit status 0
    if expected is not None:
        if res.stdout_truncated:
            return _mk(claim, Status.UNVERIFIED, "The code ran, but its output is longer than 10 KB, so it was not compared with the expected output.")
        got, want = _normalise(res.stdout), _normalise(expected)
        if got == want:
            return _mk(claim, Status.VERIFIED, "The code ran and printed exactly the expected output." + note)
        if res.network:
            return _mk(claim, Status.UNVERIFIED, "The code tried to use the network, which the sandbox disables, so its output could not be compared.")
        return bad(Status.CONTRADICTED, "The code ran but its output differs from the expected output.",
                   ("expected output:\n" + _clip(want), "actual output:\n" + _clip(got)))
    if res.network:
        return _mk(claim, Status.UNVERIFIED, "The code tried to use the network, which the sandbox disables, so it could not be fully checked.")
    return _mk(claim, Status.VERIFIED,
               "The code ran without error (this does not prove it is correct or does what the text says)." + note)


# ------------------------------------------------------------------- main ----

def _repl_examples(code: str):
    """None if the snippet is not a '>>> ' transcript, else the doctest examples."""
    first = next((ln for ln in code.splitlines() if ln.strip()), "")
    if not first.lstrip().startswith(">>>"):
        return None
    return doctest.DocTestParser().get_examples(code)


def _simple_wants(examples) -> str | None:
    wants = []
    for ex in examples:
        w = ex.want
        if ex.exc_msg is not None or w.lstrip().startswith("Traceback"):
            return None
        wants.append(w.replace("<BLANKLINE>", ""))
    joined = "".join(wants)
    if not joined.strip() or "..." in joined or "0x" in joined:
        return None
    return joined


def _check_one(claim: Claim, run: bool, timeout: float) -> Finding:
    data = claim.data
    lang = str(data.get("language") or "").strip().lower()
    code = data.get("code")
    expected = data.get("expected_output")
    if lang not in PYTHON_LANGS:
        return _mk(claim, Status.UNVERIFIED, "Only Python code is checked; this block is not Python.")
    if not isinstance(code, str) or not code.strip():
        return _mk(claim, Status.UNVERIFIED, "The code block is empty, so there is nothing to check.")
    if len(code) > MAX_CODE_CHARS:
        return _mk(claim, Status.UNVERIFIED, "The code block is too large to check.")
    if expected is not None and not isinstance(expected, str):
        expected = None
    unlabelled = lang == ""

    sources: list[str]
    mode: str
    try:
        examples = _repl_examples(code)
    except ValueError:
        return _mk(claim, Status.UNVERIFIED, "This looks like a Python console transcript but could not be interpreted, so it was not checked.")
    try:
        if examples is not None:
            if not examples:
                return _mk(claim, Status.UNVERIFIED, "This looks like a Python console transcript but could not be interpreted, so it was not checked.")
            if any(ex.exc_msg is not None or ex.want.lstrip().startswith("Traceback") for ex in examples):
                return _mk(claim, Status.UNVERIFIED,
                           "This console transcript expects an exception, which this checker does not compare, so it was not checked.")
            sources, mode = [ex.source for ex in examples], "single"
            for s in sources:
                compile(s, "<snippet>", "single")
            if expected is None:
                expected = _simple_wants(examples)
        else:
            sources, mode = [code], "exec"
            ast.parse(code, filename="<snippet>")
    except SyntaxError as e:
        if unlabelled:
            return _mk(claim, Status.UNVERIFIED, "The code block has no language label and does not parse as Python, so it was not checked.")
        where = f"line {e.lineno}" if e.lineno else "an unknown line"
        text = (e.text or "").strip()
        ev = (f"{type(e).__name__}: {e.msg} ({where})" + (f": {text}" if text else ""),)
        return _mk(claim, Status.CONTRADICTED, f"The Python code does not even parse: {e.msg} ({where}).", ev)
    except (ValueError, RecursionError, MemoryError):
        return _mk(claim, Status.UNVERIFIED, "The code could not be parsed by this checker (internal limit), so it was not checked.")

    if not run:
        return _mk(claim, Status.UNVERIFIED, "Syntax OK; not executed (pass --run-code to run it).")

    try:
        t = float(timeout)
        if not math.isfinite(t):
            t = 5.0
    except (TypeError, ValueError):
        t = 5.0
    t = max(0.1, min(t, 120.0))
    try:
        res = _execute(sources, mode, t)
    except Exception as e:  # our own failure
        return _mk(claim, Status.UNVERIFIED, f"The sandbox itself failed ({type(e).__name__}), so the code could not be run.")
    note = " (no language was given; treated as Python)" if unlabelled else ""
    return _classify(claim, res, expected, t, unlabelled, note)


def check(claims: Sequence[Claim], *, run: bool = False, timeout: float = 5.0) -> list[Finding]:
    out: list[Finding] = []
    for claim in claims:
        if claim.kind != ClaimKind.CODE:
            continue
        try:
            out.append(_check_one(claim, run, timeout))
        except Exception as e:  # never let a checker crash the pipeline
            out.append(_mk(claim, Status.UNVERIFIED,
                           f"The sandbox hit an internal error ({type(e).__name__}), so the code was not checked."))
    return out
