"""Every module must be reachable from something a person actually runs.

This test exists because of a defect that already happened. `models/events.py`
was built, tested and mutation-verified, and then consumed by nothing except a
display card — so every backtest number in the repo was produced with the
scheduled-event gate silently OFF, and no unit test could see it, because each
module passed its own tests in isolation.

A module nobody calls is a module whose output nobody reads and whose bugs nobody
finds. So the wiring itself is now a test: if a new module lands and no CLI, demo
or harness reaches it, this fails and names it.

Run: python3 tests/test_wiring.py
"""

import io
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: What counts as "something a person runs".
ENTRY_POINTS = ["demo.py", "tools"]

#: Modules that are libraries for other modules rather than pipeline stages, and
#: are legitimately reached only from inside engine/ or models/. Each needs a
#: reason, so that adding to this list is a decision rather than a shortcut.
INTERNAL = {
    "density": "the shared output type; every forecaster returns one",
    "objective": "scoring primitives used by the gate and tail_fit",
    "surface": "SVI fit consumed by rnd, not driven directly",
    "neural": "an alternative head behind mdn",
    "gru": "an alternative head behind mdn",
    "news_signal": "composed into the gates by sentiment/paper_trade",
    "sizing": "used by portfolio",
    "iv": "solver used everywhere",
    "pricing": "solver used everywhere",
    "data": "the OptionChain type itself",
    "adapters": "vendor plumbing behind run_live",
    "ibkr": "vendor adapter, reached by symbol name at runtime",
    "backtest": "metrics primitive behind every harness",
    "signal": "scalar scanner behind demo's section 3",
    "volforecast": "used by every forecaster",
    "portfolio": "driven by the backtests",
    "american": "driven by run_live --american",
    "hedged_backtest": "driven by run_live and demo",
    "signal_backtest": "driven by benchmark/stress/demo",
    "deribit": "vendor adapter behind run_live",
    "tradier": "vendor adapter behind run_live",
    "mdn": "driven by the promotion gate",
    "baseline": "the default forecaster",
    "rnd": "driven by run_live",
    "edge": "driven by run_live and the harnesses",
    "strike_scan": "driven by run_live and demo",
    "macro": "gate composed into signal_backtest",
    "sentiment": "gate composed into paper_trade",
}


def _modules():
    out = []
    for pkg in ("engine", "models"):
        for f in sorted(os.listdir(os.path.join(ROOT, pkg))):
            if f.endswith(".py") and not f.startswith("_"):
                out.append((pkg, f[:-3]))
    return out


def _entry_text() -> str:
    """Everything a person can run, concatenated."""
    chunks = []
    for e in ENTRY_POINTS:
        p = os.path.join(ROOT, e)
        if os.path.isfile(p):
            chunks.append(open(p, encoding="utf-8").read())
        elif os.path.isdir(p):
            for f in sorted(os.listdir(p)):
                if f.endswith(".py"):
                    chunks.append(open(os.path.join(p, f), encoding="utf-8").read())
    return "\n".join(chunks)


def test_every_module_is_reachable_from_a_demo_or_a_cli():
    text = _entry_text()
    orphans = []
    for pkg, mod in _modules():
        if mod in INTERNAL:
            continue
        pat = rf"(from\s+{pkg}\.{mod}\s+import|from\s+{pkg}\s+import[^\n]*\b{mod}\b|{pkg}\.{mod}\b)"
        if not re.search(pat, text):
            orphans.append(f"{pkg}/{mod}.py")
    assert not orphans, (
        "built, tested, and reachable from nothing a person runs:\n  "
        + "\n  ".join(orphans)
        + "\n\nThis is how models/events.py ended up switched off in every backtest "
          "in the repo. Wire it into a CLI or demo.py, or add it to INTERNAL with "
          "a reason.")


def test_the_internal_allowlist_has_no_dead_entries():
    """An allowlist that outlives its modules stops meaning anything."""
    names = {m for _, m in _modules()}
    stale = sorted(set(INTERNAL) - names)
    assert not stale, f"INTERNAL names modules that no longer exist: {stale}"


def test_every_internal_exemption_carries_a_reason():
    blank = [k for k, v in INTERNAL.items() if not (v or "").strip()]
    assert not blank, f"exemptions without a reason: {blank}"


def test_no_test_file_imports_a_name_that_its_runner_would_collect():
    """Every file here collects tests with `globals()` filtered on a `test_` prefix,
    so ANY imported name with that prefix is called as if it were a test.

    `test_check_offline.py` imported `tools.check_offline.test_files` — a production
    helper that enumerates the suite. It takes no arguments, returns a list and
    raises nothing, so the runner called it, counted it, and printed
    `PASS test_files`: a seventh result for six tests. That is the failure this repo
    keeps finding in other forms — work that did not happen looking like work that
    succeeded — and here it inflates the very count the docs quote.
    """
    import ast
    bad = []
    for f in sorted(os.listdir(os.path.join(ROOT, "tests"))):
        if not (f.startswith("test_") and f.endswith(".py")):
            continue
        tree = ast.parse(io.open(os.path.join(ROOT, "tests", f),
                                 encoding="utf-8").read())
        for n in ast.walk(tree):
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                for al in n.names:
                    bound = al.asname or al.name.split(".")[0]
                    if bound.startswith("test_"):
                        bad.append(f"tests/{f}:{n.lineno} imports {bound!r}")
    assert not bad, ("; ".join(bad) + " — the runner collects anything named test_*, "
                     "so import it under another name (`as something_else`)")


def test_the_documented_test_count_is_the_real_one():
    """The docs advertise a scale. It was maintained by hand and edited six times
    in one session, which is the reliable signal that it should not be.

    A count that drifts is worse than no count: a reader checking "534 tests, all
    green" against a suite of 400 has no way to tell whether tests were deleted or
    the sentence was simply never updated, so the whole document loses its claim to
    being measured rather than asserted.

    The test FUNCTIONS are counted by parsing each file, not by running it — this
    must stay fast enough to run in the suite it is counting.
    """
    import ast
    files = sorted(f for f in os.listdir(os.path.join(ROOT, "tests"))
                   if f.startswith("test_") and f.endswith(".py"))
    total = optional = 0
    for f in files:
        src = io.open(os.path.join(ROOT, "tests", f), encoding="utf-8").read()
        n_tests = sum(1 for n in ast.parse(src).body
                      if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                      and n.name.startswith("test_"))

        # A file that declares OPTIONAL_DEPENDENCY skips entirely where that
        # dependency is absent, which is the environment this repo advertises. Its
        # tests are real but they are not part of "all green", and counting them
        # there would claim 10 results nobody has seen.
        # Detected as a module-level ASSIGNMENT, not as a substring: this file
        # mentions the name in its own source, so a substring check excluded its
        # own seven tests from the count it was computing.
        tree = ast.parse(src)
        declares = any(
            isinstance(n, (ast.Assign, ast.AnnAssign))
            and any(getattr(t, "id", "") == "OPTIONAL_DEPENDENCY"
                    for t in (n.targets if isinstance(n, ast.Assign) else [n.target]))
            for n in tree.body)
        if declares:
            optional += n_tests
        else:
            total += n_tests

    # Only the CANONICAL phrasings for the whole suite. A loose "N tests" also
    # matches a deliberate historical note ("391 tests passed around it", recording
    # how many existed when sizing.py had none) and a per-file count ("7 correctness
    # tests"), neither of which should be rewritten every time the suite grows.
    PATTERNS = [(r"(\d[\d,]*)\s+tests?,\s*all green", "tests"),
                (r"(\d[\d,]*)\s+tests?,\s*\d+\s+files", "tests"),
                (r"(\d+)\s+test files", "files"),
                (r"tests?,\s*(\d+)\s+files", "files")]
    claims = []
    for doc in ("PIPELINE.md", "QUICKSTART.md", "README.md", "ARCHITECTURE.md",
                "PREREGISTRATION.md"):
        path = os.path.join(ROOT, doc)
        if not os.path.exists(path):
            continue
        for n, line in enumerate(io.open(path, encoding="utf-8"), 1):
            for pat, kind in PATTERNS:
                for m in re.finditer(pat, line):
                    claims.append((doc, n, int(m.group(1).replace(",", "")), kind))

    assert claims, "no documented scale found at all — did the wording change?"
    want = {"tests": total, "files": len(files)}
    wrong = [f"{d}:{n} says {v} {kind}, actual {want[kind]}"
             for d, n, v, kind in claims if v != want[kind]]
    assert not wrong, ("the documented scale has drifted: " + "; ".join(wrong))

    # and the skipped ones are stated rather than quietly excluded, or the headline
    # count becomes a number that happens to be true for a reason nobody can see
    if optional:
        pipeline = io.open(os.path.join(ROOT, "PIPELINE.md"), encoding="utf-8").read()
        assert re.search(rf"{optional}\s+more\b", pipeline), (
            f"{optional} tests are skipped without numpy and PIPELINE.md does not "
            f"say so — write '{optional} more require numpy'")


def test_no_entry_point_hardcodes_the_exit_rule():
    """The rule on the screen and the rule in the harness must be one string. They
    were not: models/ticket.py advised "close at 50% of max profit, or at 7 DTE"
    and engine/spread_backtest.py held every spread to expiry, so every reported
    Sharpe, worst trade and return described a strategy the screen did not
    recommend. A literal copy anywhere lets them drift apart again."""
    bad = []
    for entry in ENTRY_POINTS:
        path = os.path.join(ROOT, entry)
        files = ([path] if os.path.isfile(path)
                 else [os.path.join(path, f) for f in sorted(os.listdir(path))
                       if f.endswith(".py")])
        for f in files:
            for n, line in enumerate(io.open(f, encoding="utf-8"), 1):
                if "exit_rule=" in line and "max profit" in line:
                    bad.append(f"{os.path.relpath(f, ROOT)}:{n}")
    assert not bad, ("exit_rule spelled out literally at " + ", ".join(bad)
                     + " — use engine.spread_backtest.EXIT_RULE_TEXT so the "
                       "recommendation cannot drift from the measurement")


def test_every_order_ticket_is_built_from_the_fused_decision():
    """Import-level wiring was never the problem for models/ticket.py — it was
    reachable from demo.py the whole time. It was reachable and UNINFORMED: it
    read only the variance card, so the size cut that sector rotation and fund
    crowding bought was discarded at the last step, and the shipped order carried
    67-100% more risk than the fused view authorised. Reachability does not imply
    correctness, so the call SHAPE is pinned too."""
    bad = []
    for entry in ENTRY_POINTS:
        path = os.path.join(ROOT, entry)
        files = ([path] if os.path.isfile(path)
                 else [os.path.join(path, f) for f in sorted(os.listdir(path))
                       if f.endswith(".py")])
        for f in files:
            src = io.open(f, encoding="utf-8").read()
            for m in re.finditer(r"build_ticket\s*\(", src):
                # take the balanced argument list, so a nested call cannot end it early
                i, depth = m.end(), 1
                while i < len(src) and depth:
                    depth += (src[i] == "(") - (src[i] == ")")
                    i += 1
                if "decision=" not in src[m.end():i]:
                    line = src[:m.start()].count("\n") + 1
                    bad.append(f"{os.path.relpath(f, ROOT)}:{line}")
    assert not bad, ("build_ticket called without decision= at: " + ", ".join(bad)
                     + " — the ticket would print full size regardless of what "
                       "the fused context cut")


def test_the_demo_actually_runs_and_prints_its_sections():
    """The demo is the repo's front door; a broken section there is invisible."""
    p = subprocess.run([sys.executable, "demo.py"], cwd=ROOT,
                       capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, p.stderr[-2000:]
    out = p.stdout
    for marker in ("Distributional P-vs-Q scan",
                   "Walk-forward delta-hedged",
                   "Forward-test paper ledger",
                   "Benchmarks",
                   "Overnight-gap stress",
                   "Does fund crowding matter?",
                   "Defined-risk spreads vs the delta-hedged naked book",
                   "Seed ensemble",
                   "Correlation-aware sizing vs the per-trade view"):
        assert marker in out, f"demo.py no longer prints {marker!r}"


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
