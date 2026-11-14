"""Tests for the documentation, written because it was empty for nine commits.

SETUP.md and README.md were truncated to zero bytes by a commit whose message
describes editing them, and nothing noticed for nine commits. Every check that
could have caught it failed to:

  - the test suite never opened either file;
  - the mutation harness targets code;
  - and the one script that did read README.md verified the figures inside its
    fenced code blocks, which passes trivially on a file with no blocks in it.

A check that cannot fail on an empty input is not a check, so the first test
here is that these files are not empty, and the rest assert content that an
empty file cannot have.

THE FIGURES ARE MEASURED, NOT TRUSTED. Every count these pages quote is
compared against the tree. That has caught a wrong number in this repo more
than once -- 216 for 223 tests, 236 for 265, 282 for 279 -- every time because
it was written from memory and not one of them was off by enough to look wrong.

Run: python3 tests/test_docs.py
"""

import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

#: Pages that must exist, with the floor each must clear in bytes. The floors
#: are deliberately far below the real sizes: this guards against truncation
#: and accidental overwrite, not against editing.
PAGES = {
    "README.md": 3000,
    "SETUP.md": 4000,
    "mutation/README.md": 1500,
    "example/README.md": 1500,
}


def _read(rel: str) -> str:
    with io.open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
        return fh.read()


def count_tests() -> int:
    """Test functions across every test file.

    Counted from the source rather than by running the suite, which would
    recurse into this file. Verified equal to the running total: each file's
    `_run_all` collects exactly the `test_`-prefixed globals it defines.
    """
    total = 0
    d = os.path.join(ROOT, "tests")
    for name in sorted(os.listdir(d)):
        if name.startswith("test_") and name.endswith(".py"):
            total += len(re.findall(r"^def test_", _read(f"tests/{name}"),
                                    re.M))
    return total


def count_mutants() -> int:
    """Defects across every mutant table, by importing each table."""
    import importlib.util
    total = 0
    d = os.path.join(ROOT, "mutation", "mutants")
    for name in sorted(os.listdir(d)):
        if not name.endswith(".py") or name.startswith("_"):
            continue
        spec = importlib.util.spec_from_file_location(
            f"_mt_{name[:-3]}", os.path.join(d, name))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        total += len(mod.MUTANTS)
    return total


def count_targets() -> int:
    d = os.path.join(ROOT, "mutation", "mutants")
    return len([n for n in os.listdir(d)
                if n.endswith(".py") and not n.startswith("_")])


def test_no_documentation_page_is_empty_or_truncated():
    # The test this file exists for. It must fail on a zero-byte file, which
    # is the state these pages were actually in.
    for rel, floor in sorted(PAGES.items()):
        path = os.path.join(ROOT, rel)
        assert os.path.exists(path), f"{rel} is missing"
        size = os.path.getsize(path)
        assert size >= floor, (
            f"{rel} is {size} bytes, under the {floor}-byte floor — it has "
            f"been truncated or overwritten, which is what happened before")


def test_every_page_has_a_heading_and_prose():
    for rel in sorted(PAGES):
        text = _read(rel)
        assert text.lstrip().startswith("#"), f"{rel} has no heading"
        words = len(re.findall(r"[A-Za-z]{3,}", text))
        assert words > 200, f"{rel} has only {words} words of prose"


def test_the_test_count_the_pages_quote_matches_the_tree():
    want = count_tests()
    assert want > 0, "counted zero tests, so this check proves nothing"
    found = []
    for rel in sorted(PAGES):
        for m in re.finditer(r"(\d[\d,]*)\s+tests\b", _read(rel)):
            found.append((rel, int(m.group(1).replace(",", ""))))
    assert found, "no page states a test count, so nothing is being checked"
    for rel, n in found:
        assert n == want, (
            f"{rel} claims {n} tests; the tree has {want}. Figures written "
            f"from memory are the recurring error here — measure, then write")


def test_the_defect_count_the_pages_quote_matches_the_tables():
    want = count_mutants()
    assert want > 0, "counted zero mutants, so this check proves nothing"
    found = []
    for rel in sorted(PAGES):
        for m in re.finditer(r"(\d[\d,]*)\s+(?:deliberate\s+)?(?:defects|mutants)\b",
                             _read(rel)):
            found.append((rel, int(m.group(1).replace(",", ""))))
    assert found, "no page states a defect count"
    for rel, n in found:
        assert n == want, (
            f"{rel} claims {n} defects; the tables hold {want}")


def test_the_target_count_matches_the_mutant_tables():
    want = count_targets()
    for rel in sorted(PAGES):
        for m in re.finditer(r"(\d+)\s+targets\b", _read(rel)):
            assert int(m.group(1)) == want, (
                f"{rel} claims {m.group(1)} targets; there are {want}")


def test_the_per_target_defect_table_matches_the_tables_themselves():
    # The summary figure and the table drifted apart once already: the total
    # said 348 while the rows still added to 302, and the rows are what a
    # reader checks a single target against.
    import importlib.util
    d = os.path.join(ROOT, "mutation", "mutants")
    real = {}
    for name in sorted(os.listdir(d)):
        if not name.endswith(".py") or name.startswith("_"):
            continue
        spec = importlib.util.spec_from_file_location(
            f"_tt_{name[:-3]}", os.path.join(d, name))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        real[name[:-3]] = (mod.TARGET, len(mod.MUTANTS))

    text = _read("mutation/README.md")
    listed = {m.group(1): (m.group(2), int(m.group(3)))
              for m in re.finditer(
                  r"^\|\s*`(\w+)`\s*\|\s*`([^`]+)`\s*\|\s*(\d+)\s*\|",
                  text, re.M)}
    assert listed, "the per-target table is gone from mutation/README.md"
    assert set(listed) == set(real), (
        f"table lists {sorted(set(listed) - set(real))} that do not exist and "
        f"omits {sorted(set(real) - set(listed))}")
    for name, (path, count) in sorted(real.items()):
        assert listed[name] == (path, count), (
            f"{name}: table says {listed[name]}, tree says {(path, count)}")
    assert sum(c for _, c in listed.values()) == count_mutants(), (
        "the rows do not add up to the total the same page quotes")


def test_every_tool_the_pages_tell_you_to_run_exists():
    # A documented command that cannot run is worse than no documentation:
    # the reader concludes their copy is broken.
    missing = []
    for rel in sorted(PAGES):
        for m in re.finditer(r"python3\s+(?:-m\s+)?([\w./]+)", _read(rel)):
            ref = m.group(1)
            if ref.startswith("-"):
                continue
            path = ref.replace(".", "/") + ".py" if "/" not in ref \
                and not ref.endswith(".py") else ref
            if not os.path.exists(os.path.join(ROOT, path)):
                missing.append((rel, ref, path))
    assert not missing, f"documented commands that do not exist: {missing}"


def test_setup_covers_every_step_a_reader_needs():
    text = _read("SETUP.md")
    for needed in ("What you need", "price data", "costs", "hold-out",
                   "refusal"):
        assert needed.lower() in text.lower(), (
            f"SETUP.md no longer covers {needed!r}")


def test_the_refusal_codes_the_pages_list_are_the_codes_that_exist():
    # A reader who hits a code not on the page has nowhere to go, and a page
    # listing a code that no longer fires sends them looking for a cause that
    # cannot happen.
    import engine.verdict as V
    real = {n for n in dir(V)
            if n.isupper() and isinstance(getattr(V, n), str)
            and getattr(V, n) == n}
    text = _read("SETUP.md") + _read("README.md")
    documented = {c for c in real if c in text}
    assert documented == real, (
        f"refusal codes in the code but not in the pages: "
        f"{sorted(real - documented)}")


def _run_all():
    tests = [v for k, v in sorted(globals().items())
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
