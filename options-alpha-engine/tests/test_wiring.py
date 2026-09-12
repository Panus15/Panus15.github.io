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
