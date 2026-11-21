"""Mutation testing for the options engine, in the repository.

This harness is a VERBATIM COPY of fx-engine/mutation/run.py except for this
docstring. The two engines are deliberately independent trees -- separate
requirements, separate CI steps, stdlib-only, no shared import root -- and
coupling them through a shared package to save 180 lines of a development tool
would be the worse trade. The cost is real and worth stating: a fix to one copy
does not reach the other.

    python3 mutation/run.py                    # every target
    python3 mutation/run.py decision sizing    # named targets only

WHAT IT DOES. For each declared defect it rewrites one exact string in a source
file, runs that module's test file, and expects a FAILURE. A mutant that
survives means the tests do not distinguish the real code from a broken version
of it, which means the line it changed is untested whatever a coverage report
says.

WHY IT IS HERE. This engine prints confidence intervals, says "spans zero" when
a result is not distinguishable from luck, and refuses to let an untested input
manufacture position size. All of that is arithmetic, and arithmetic nobody has
tried to break is a claim rather than a property. On the FX side the same
harness found fifteen real defects in code that passed its whole suite --
including a cost model that charged financing per BAR instead of per night, a
24x overcharge that had been reported in a commit message as a result.

WHY IT REFUSES TO RUN ON A DIRTY FILE. It edits your source in place. A target
with uncommitted changes is skipped by name rather than risked.

AND WHY IT KEEPS A RESTORE COPY. The `finally` that puts the file back does not
run when the process is SIGKILLed, and that is not hypothetical: a sweep of
this harness on the FX tree was killed mid-run twice and left `if False:`
sitting in a source file, once disabling the very guard the next test run was
checking. So the original is written to `mutation/.restore/` BEFORE the first
mutation and removed only after the file is back. A run that finds files there
repairs them and says so, which turns a silent corruption into a loud one.

WHY AN ANCHOR THAT NO LONGER MATCHES IS AN ERROR, NOT A SKIP. Each defect is
keyed to an exact snippet. When the source is edited the snippet stops matching,
and a harness that quietly skipped those would report a shrinking, still-perfect
score -- the count would stay green while the coverage silently drained away.
A miss is counted, printed, and makes the run fail.
"""

from __future__ import annotations

import importlib.util
import io
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(ROOT, "mutation", "mutants")
RESTORE = os.path.join(ROOT, "mutation", ".restore")


def _load(path: str):
    spec = importlib.util.spec_from_file_location("m", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _restore_path(rel: str) -> str:
    return os.path.join(RESTORE, rel.replace(os.sep, "__"))


def repair() -> int:
    """Put back anything a killed run left mutated. Returns how many.

    Called before every sweep. A leftover mutant is a disabled guard sitting in
    the working tree, and it passes its own test suite by construction, so the
    only thing that finds it is looking for it on purpose.
    """
    if not os.path.isdir(RESTORE):
        return 0
    n = 0
    for f in sorted(os.listdir(RESTORE)):
        rel = f.replace("__", os.sep)
        src = os.path.join(RESTORE, f)
        dst = os.path.join(ROOT, rel)
        print(f"REPAIRED {rel} — a previous run was killed before it could put "
              f"this back")
        io.open(dst, "w", encoding="utf-8").write(
            io.open(src, encoding="utf-8").read())
        os.remove(src)
        n += 1
    return n


def _is_dirty(rel: str) -> bool:
    """True when `rel` has uncommitted changes, or git cannot say."""
    try:
        r = subprocess.run(["git", "status", "--porcelain", "--", rel],
                           cwd=ROOT, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return True
    return r.returncode != 0 or bool(r.stdout.strip())


def sweep(name: str, spec, env: dict) -> tuple:
    """Run one target's defects. Returns (killed, survived, missed)."""
    target = os.path.join(ROOT, spec.TARGET)
    tests = os.path.join(ROOT, spec.TESTS)
    rel = os.path.relpath(target, ROOT)
    print(f"\n=== {name}: {spec.TARGET} against {spec.TESTS} "
          f"({len(spec.MUTANTS)} defects)")
    if not os.path.exists(target) or not os.path.exists(tests):
        print(f"  !! missing {spec.TARGET} or {spec.TESTS}")
        return (0, 0, len(spec.MUTANTS))
    if _is_dirty(rel):
        print(f"  SKIPPED — {rel} has uncommitted changes. This rewrites it in "
              f"place and a kill between write and restore would lose them.")
        return (0, 0, 0)

    orig = io.open(target, encoding="utf-8").read()
    os.makedirs(RESTORE, exist_ok=True)
    keep = _restore_path(rel)
    io.open(keep, "w", encoding="utf-8").write(orig)
    killed = survived = missed = 0
    try:
        for label, old, new in spec.MUTANTS:
            n = orig.count(old)
            if n != 1:
                missed += 1
                print(f"  ANCHOR  {label:34} matched {n} times, expected 1")
                continue
            io.open(target, "w", encoding="utf-8").write(orig.replace(old, new))
            try:
                r = subprocess.run([sys.executable, spec.TESTS], cwd=ROOT,
                                   capture_output=True, text=True, env=env,
                                   timeout=600)
                rc, out = r.returncode, r.stdout
            except subprocess.TimeoutExpired:
                rc, out = 1, "FAIL timeout"
            if rc == 0:
                survived += 1
                print(f"  SURVIVED {label}")
            else:
                killed += 1
                who = [l.split(":")[0].replace("FAIL ", "")
                       for l in out.splitlines() if l.startswith("FAIL")]
                print(f"  killed  {label:34} by {who[0] if who else 'error'}")
    finally:
        # restored on KeyboardInterrupt and on any exception. A SIGKILL skips
        # this entirely, which is what the restore copy is for.
        io.open(target, "w", encoding="utf-8").write(orig)
        if os.path.exists(keep):
            os.remove(keep)
    return (killed, survived, missed)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    specs = {}
    for f in sorted(os.listdir(HERE)):
        if f.endswith(".py") and not f.startswith("_"):
            specs[f[:-3]] = _load(os.path.join(HERE, f))
    if argv:
        unknown = [a for a in argv if a not in specs]
        if unknown:
            print(f"no such target: {', '.join(unknown)}\n"
                  f"available: {', '.join(sorted(specs))}", file=sys.stderr)
            return 2
        specs = {k: v for k, v in specs.items() if k in argv}

    # stale bytecode once inverted a sweep in this project, reporting mutants
    # killed that were not; this is not negotiable per-run
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    fixed = repair()
    tot_k = tot_s = tot_m = 0
    for name, spec in specs.items():
        k, s, m = sweep(name, spec, env)
        tot_k += k
        tot_s += s
        tot_m += m

    total = tot_k + tot_s + tot_m
    print(f"\n{'=' * 60}"
          + (f"\n{fixed} file(s) repaired from a killed run before starting\n"
             if fixed else "\n")
          + f"{tot_k}/{total} killed"
          + (f"  |  {tot_s} SURVIVED" if tot_s else "")
          + (f"  |  {tot_m} anchors missed" if tot_m else ""))
    if tot_s or tot_m:
        print("A survivor means the tests cannot tell the real code from a "
              "broken version of it.\nA missed anchor means a defect was never "
              "tried, so the score above is not the\nscore it claims to be.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
