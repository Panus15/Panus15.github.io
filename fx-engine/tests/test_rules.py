"""Tests for loading a rule you wrote — mostly for what it refuses to load.

A homemade rule's defects do not raise. They come out as a P&L figure. So the
loader's job is to find them before the backtest turns them into one, and the
test that matters most is the look-ahead check: a rule that reads one bar into
the future produces an equity curve that slopes up and a code that looks fine.

Measured, on the random walk used elsewhere in this suite, with the check
switched off: a one-bar peek reported 4,312 trades at +3.49 pips each, +15,062
pips total, TRADEABLE — on data where nothing can be true by construction.
That is what this file exists to stop.

Run: python3 tests/test_rules.py
"""

import io
import math
import os
import random
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.rules import (LOOK_AHEAD_BARS, YOUR_OWN_RULE, Rule, check_look_ahead,
                          check_shape, load_rule)
from models.patterns import Bar, Detection

_TMP = tempfile.mkdtemp(prefix="fxrule-")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Every rule file needs the engine on its path to import Detection.
PREAMBLE = f'''import sys
sys.path.insert(0, {ROOT!r})
from models.patterns import Detection
'''


def _rule_file(body: str, name: str = None) -> str:
    path = os.path.join(_TMP, name or f"r{len(os.listdir(_TMP))}.py")
    io.open(path, "w", encoding="utf-8").write(PREAMBLE + body)
    return path


def _bars(n=300, seed=5):
    rng = random.Random(seed)
    out, px = [], 1.1000
    for _ in range(n):
        o, path = px, [px]
        for _ in range(4):
            px *= math.exp(rng.gauss(0.0, 0.0004))
            path.append(px)
        out.append(Bar(o, max(path), min(path), px))
    return out


HONEST = '''
PARAMETER_COUNT = 2

def detect(bars):
    out = []
    for i in range(20, len(bars)):
        if bars[i].c > bars[i - 1].c and bars[i - 1].c > bars[i - 2].c:
            e = bars[i].c
            out.append(Detection("up2", i, 1, e, e - 0.0030, e + 0.0060))
    return out
'''

PEEKS = '''
PARAMETER_COUNT = 1

def detect(bars):
    out = []
    for i in range(1, len(bars) - 1):
        if bars[i + 1].c > bars[i].c:        # bar i+1 has not printed yet
            e = bars[i].c
            out.append(Detection("cheat", i, 1, e, e - 0.0030, e + 0.0060))
    return out
'''


# ---------------------------------------------------------------------------
# the check that justifies the module
# ---------------------------------------------------------------------------

def test_a_rule_that_peeks_one_bar_ahead_is_refused():
    """The whole point. Nothing about the code looks wrong and the backtest
    would simply report an edge."""
    bars = _bars()
    try:
        load_rule(_rule_file(PEEKS), bars)
    except ValueError as e:
        assert "looks ahead" in str(e), e
        assert "had not printed" in str(e), e
    else:
        raise AssertionError("a rule reading future bars was accepted")


def test_the_refusal_names_the_bar_that_gave_it_away():
    """A complaint you cannot locate is a complaint you cannot act on."""
    bars = _bars()
    problems = check_look_ahead(_loaded(PEEKS).detect, bars)
    assert problems, "the peeking rule passed the check"
    assert "truncated at bar" in problems[0], problems[0]
    assert "disappears" in problems[0]


def test_an_honest_rule_passes_the_check():
    """Isolation in the other direction: a check that refused everything would
    be indistinguishable from one that works."""
    bars = _bars()
    assert check_look_ahead(_loaded(HONEST).detect, bars) == []
    r = load_rule(_rule_file(HONEST), bars)
    assert r.parameter_count == 2
    assert r.detect(bars), "the honest fixture must actually fire"


def test_a_series_too_short_to_check_says_so_rather_than_passing():
    short = _bars(LOOK_AHEAD_BARS)
    problems = check_look_ahead(_loaded(HONEST).detect, short)
    assert problems and "too few" in problems[0], problems


def _loaded(body):
    return load_rule(_rule_file(body), None, check=False)


# ---------------------------------------------------------------------------
# the declaration that cannot be guessed
# ---------------------------------------------------------------------------

def test_a_rule_must_declare_its_parameter_count():
    body = HONEST.replace("PARAMETER_COUNT = 2", "")
    try:
        load_rule(_rule_file(body), _bars())
    except ValueError as e:
        assert "PARAMETER_COUNT" in str(e), e
        assert "multiple-testing correction" in str(e), (
            "the refusal did not say WHY the number is needed")
    else:
        raise AssertionError("a rule with no declared parameter count loaded")


def test_a_parameter_count_that_is_not_a_positive_integer_is_refused():
    for bad in ("0", "-1", "'three'", "2.5", "True"):
        body = HONEST.replace("PARAMETER_COUNT = 2", f"PARAMETER_COUNT = {bad}")
        try:
            load_rule(_rule_file(body), _bars())
        except ValueError as e:
            assert "PARAMETER_COUNT" in str(e), e
        else:
            raise AssertionError(f"PARAMETER_COUNT = {bad} was accepted")


def test_a_declared_count_of_one_is_allowed():
    """1 is a legitimate declaration. Only 0 and below are impossible."""
    body = HONEST.replace("PARAMETER_COUNT = 2", "PARAMETER_COUNT = 1")
    assert load_rule(_rule_file(body), _bars()).parameter_count == 1


# ---------------------------------------------------------------------------
# the file itself
# ---------------------------------------------------------------------------

def test_a_file_with_no_detect_is_refused():
    try:
        load_rule(_rule_file("PARAMETER_COUNT = 1\n"), _bars())
    except ValueError as e:
        assert "detect(bars)" in str(e), e
    else:
        raise AssertionError("a file with no detect() loaded")


def test_a_detect_that_is_not_callable_is_refused():
    body = "PARAMETER_COUNT = 1\ndetect = 42\n"
    try:
        load_rule(_rule_file(body), _bars())
    except ValueError as e:
        assert "detect(bars)" in str(e), e
    else:
        raise AssertionError("a non-callable detect loaded")


def test_a_missing_file_is_refused_by_path():
    try:
        load_rule(os.path.join(_TMP, "nope.py"), _bars())
    except ValueError as e:
        assert "no rule file at" in str(e), e
    else:
        raise AssertionError("a nonexistent path loaded")


def test_the_name_comes_from_the_file_unless_the_rule_says_otherwise():
    assert load_rule(_rule_file(HONEST, "my_cross.py"), _bars()).name == "my_cross"
    named = HONEST + '\nNAME = "declared_name"\n'
    assert load_rule(_rule_file(named), _bars()).name == "declared_name"
    assert load_rule(_rule_file(HONEST), _bars(), name="explicit").name == "explicit"


# ---------------------------------------------------------------------------
# shapes that would become a P&L figure rather than an error
# ---------------------------------------------------------------------------

def _d(**kw):
    base = dict(name="x", index=5, direction=1, entry=1.1000, stop=1.0970,
                target=1.1060)
    base.update(kw)
    return Detection(**base)


def test_a_well_formed_detection_passes():
    check_shape([_d(), _d(direction=-1, stop=1.1030, target=1.0940)], 300)


def test_a_direction_that_is_not_plus_or_minus_one_is_refused():
    for bad in (0, 2, -2, 0.5):
        try:
            check_shape([_d(direction=bad)], 300)
        except ValueError as e:
            assert "direction" in str(e), e
        else:
            raise AssertionError(f"direction={bad} was accepted")


def test_a_detection_outside_the_series_is_refused():
    for bad in (-1, 300, 999):
        try:
            check_shape([_d(index=bad)], 300)
        except ValueError as e:
            assert "outside the" in str(e), e
        else:
            raise AssertionError(f"index={bad} was accepted")


def test_a_long_with_its_stop_above_its_entry_is_refused():
    """A guaranteed loss, not a trade — and the backtest would price it."""
    try:
        check_shape([_d(stop=1.1030)], 300)
    except ValueError as e:
        assert "guaranteed loss" in str(e), e
    else:
        raise AssertionError("a long with an inverted stop was accepted")


def test_a_short_with_its_levels_the_wrong_way_round_is_refused():
    try:
        check_shape([_d(direction=-1, stop=1.0970, target=1.1060)], 300)
    except ValueError as e:
        assert "short" in str(e), e
    else:
        raise AssertionError("a short with long levels was accepted")


def test_something_that_is_not_a_detection_is_refused():
    try:
        check_shape([{"index": 5}], 300)
    except ValueError as e:
        assert "not a Detection" in str(e), e
    else:
        raise AssertionError("a dict was accepted as a detection")


def test_load_rule_actually_runs_the_shape_checks():
    """check_shape being correct is not the same as load_rule calling it. The
    tests above exercise the function directly, which leaves "load the rule and
    skip the check" indistinguishable from working."""
    bad = '''
PARAMETER_COUNT = 1

def detect(bars):
    e = bars[50].c
    # a long whose stop sits ABOVE its entry: a guaranteed loss, not a trade
    return [Detection("bad", 50, 1, e, e + 0.0030, e + 0.0060)]
'''
    try:
        load_rule(_rule_file(bad), _bars())
    except ValueError as e:
        assert "guaranteed loss" in str(e), e
    else:
        raise AssertionError("load_rule accepted a malformed detection")

    wrong_type = '''
PARAMETER_COUNT = 1

def detect(bars):
    return "not a list"
'''
    try:
        load_rule(_rule_file(wrong_type), _bars())
    except ValueError as e:
        assert "must return a list" in str(e), e
    else:
        raise AssertionError("load_rule accepted a non-list return")


def test_a_detect_that_returns_the_wrong_type_is_refused():
    for bad in (None, 42, "detections", (_d(),)):
        try:
            check_shape(bad, 300)
        except ValueError as e:
            assert "must return a list" in str(e), e
        else:
            raise AssertionError(f"{type(bad).__name__} was accepted")


# ---------------------------------------------------------------------------
# what the engine says it knows about your rule
# ---------------------------------------------------------------------------

def test_your_own_rule_is_not_labelled_never_tested_in_fx():
    """NEVER_TESTED means the literature did not test a KNOWN pattern. For a
    rule you just wrote there is no literature to have tested anything, and no
    result from your own data either until the hold-out has run."""
    r = load_rule(_rule_file(HONEST), _bars())
    label, detail = r.evidence
    assert label == YOUR_OWN_RULE
    assert "no published evidence" in label
    assert "never tested in FX" not in label
    assert "2 parameter(s)" in detail
    assert "until it has been" in detail and "not built from" in detail


def test_the_rule_carries_where_it_came_from():
    path = _rule_file(HONEST)
    r = load_rule(path, _bars())
    assert isinstance(r, Rule)
    assert r.source == path
    assert path in r.evidence[1]


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
