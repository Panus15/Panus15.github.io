"""Your own rule, put through the same machine.

Everything else here tests six patterns this engine chose. That is the wrong
scope for the question it claims to answer. "Does MY idea clear MY costs" is not
answerable by a fixed list of shapes someone else picked, so a rule you write
goes through the identical pipeline: same costs, same ambiguous-bar rule, same
multiple-testing correction, same refusals, same hold-out.

    python3 tools/analyse.py data.csv --pair EURUSD --side bid --rule my_rule.py

A rule file is a Python module that defines two things:

    PARAMETER_COUNT = 3          # how many knobs you could have swept
    def detect(bars): ...        # -> list[Detection]

WHY PARAMETER_COUNT IS REQUIRED AND NOT GUESSED. It is the denominator of the
multiple-testing correction, and understating it makes the significance test
easier in exactly the direction that passes noise. This engine cannot count the
variants you tried and discarded before settling on the rule you are handing it
— only you know that — so it asks, the way `models/patterns.py` declares its own
four. Declaring 1 when you swept twenty is not a shortcut, it is the thing that
makes `NOT_SIGNIFICANT` stop working.

WHAT IT CHECKS BEFORE IT WILL RUN YOUR RULE.

A homemade rule's most common defect is look-ahead, and it is invisible: the
backtest simply reports an edge. So `check_look_ahead` runs your `detect` on the
series TRUNCATED at successive bars and compares. If a detection appears at bar
i when the data stops at bar i, but not when the data stops one bar earlier,
your rule read a bar that had not printed. Nothing about your code looks wrong;
the equity curve is just a lie. This is the same test `models/patterns.py` is
held to, and two of its own detectors failed it when it was first written.

It also checks the cheap structural things, because each produces a silently
wrong number downstream rather than an error: a detection must carry a direction
of +1 or -1, a stop on the losing side of the entry, a target on the winning
side, and an index inside the series.

A NOTE ON WHAT LOADING A FILE MEANS. `load_rule` imports the path you give it,
which runs the code in it, exactly as `python3 my_rule.py` would. That is the
point — it is your file on your machine. Do not point it at a rule someone sent
you without reading it first, for the same reason you would not run their script.
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass

from models.patterns import Detection

#: Evidence label for a rule this engine has never heard of. Distinct from
#: NEVER_TESTED, which means the literature never tested a KNOWN pattern: here
#: there is no literature to have tested anything, and no result from your own
#: data either until the hold-out has run.
YOUR_OWN_RULE = "your own rule — no published evidence, and none from this file"

#: Bars to truncate across when checking for look-ahead. Each one is a separate
#: re-run of the whole detector, so this is a cost; 40 is enough to catch a rule
#: that reads one bar ahead, which is the overwhelmingly common case.
LOOK_AHEAD_BARS = 40


@dataclass(frozen=True)
class Rule:
    """A loaded rule, with everything the pipeline needs to treat it like a
    built-in pattern."""

    name: str
    detect: object
    parameter_count: int
    source: str

    @property
    def evidence(self) -> tuple:
        return (YOUR_OWN_RULE,
                f"loaded from {self.source}, declaring {self.parameter_count} "
                f"parameter(s). Nothing is known about it until it has been "
                f"measured on data it was not built from")


def _import(path: str):
    if not os.path.exists(path):
        raise ValueError(f"no rule file at {path}")
    spec = importlib.util.spec_from_file_location("fx_user_rule", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"{path} is not an importable Python module")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_shape(detections, n_bars: int) -> None:
    """Refuse a detection that would produce a silently wrong number.

    Each of these would otherwise flow into the backtest and come out as a P&L
    figure rather than as an error.
    """
    if not isinstance(detections, list):
        raise ValueError(
            f"detect() must return a list of Detection, got "
            f"{type(detections).__name__}")
    for i, d in enumerate(detections):
        if not isinstance(d, Detection):
            raise ValueError(
                f"detection {i} is a {type(d).__name__}, not a Detection. "
                f"Import it from models.patterns so the fields are checked")
        if d.direction not in (1, -1):
            raise ValueError(
                f"detection {i} has direction {d.direction!r}; it must be +1 "
                f"for long or -1 for short, since every pip figure downstream "
                f"is multiplied by it")
        if not 0 <= d.index < n_bars:
            raise ValueError(
                f"detection {i} is at bar {d.index}, outside the {n_bars} bars "
                f"it was given")
        if d.direction > 0 and not (d.stop < d.entry < d.target):
            raise ValueError(
                f"detection {i} is long but its levels are not ordered "
                f"stop < entry < target ({d.stop} / {d.entry} / {d.target}). A "
                f"stop above the entry is a guaranteed loss, not a trade")
        if d.direction < 0 and not (d.target < d.entry < d.stop):
            raise ValueError(
                f"detection {i} is short but its levels are not ordered "
                f"target < entry < stop ({d.target} / {d.entry} / {d.stop})")


def check_look_ahead(detect, bars: list, span: int = LOOK_AHEAD_BARS) -> list:
    """Find detections that only exist once future bars are visible.

    Returns a list of complaints, empty when clean. The test: a rule that fires
    at bar i must fire at bar i when the series STOPS at bar i. If truncating
    the data removes a detection that the full series reported at an index
    inside the truncation, the rule used a bar that had not printed yet.
    """
    n = len(bars)
    if n < span + 2:
        return [f"only {n} bars — too few to check for look-ahead across "
                f"{span}; run this on a longer series before trusting it"]
    out = []
    full = {d.index for d in detect(bars)}
    for cut in range(n - span, n + 1):
        visible = {d.index for d in detect(bars[:cut])}
        # a detection the full series puts strictly before the cut must also be
        # found when the data stops at the cut
        for i in sorted(d for d in full if d < cut):
            if i not in visible:
                out.append(
                    f"a detection at bar {i} disappears when the series is "
                    f"truncated at bar {cut}, so finding it needed bars "
                    f"{cut}..{n - 1} — which had not printed at bar {i}")
                return out
    return out


def load_rule(path: str, bars: list = None, *, name: str = None,
              check: bool = True) -> Rule:
    """Import a rule file and refuse it if it cannot be trusted.

    ``bars`` is used for the structural and look-ahead checks. Passing none
    skips them, which is only for testing the loader itself: a rule that has
    not been checked for look-ahead is a rule whose backtest means nothing.
    """
    mod = _import(path)
    if not hasattr(mod, "detect") or not callable(mod.detect):
        raise ValueError(
            f"{path} defines no callable detect(bars). A rule file needs "
            f"detect(bars) -> list[Detection] and PARAMETER_COUNT")
    pc = getattr(mod, "PARAMETER_COUNT", None)
    if not isinstance(pc, int) or isinstance(pc, bool) or pc < 1:
        raise ValueError(
            f"{path} must declare PARAMETER_COUNT as an integer >= 1. It is "
            f"the denominator of the multiple-testing correction, and this "
            f"engine cannot count the variants you tried and discarded before "
            f"settling on this rule — only you can. Declaring it too low is "
            f"what makes the significance test stop working")
    rule = Rule(name=name or getattr(mod, "NAME", None)
                or os.path.basename(path).rsplit(".", 1)[0],
                detect=mod.detect, parameter_count=pc, source=path)
    if check and bars:
        check_shape(rule.detect(bars), len(bars))
        problems = check_look_ahead(rule.detect, bars)
        if problems:
            raise ValueError(
                f"{path} looks ahead: {problems[0]} Nothing about the code "
                f"will look wrong and the backtest will simply report an edge, "
                f"which is why this is checked before anything else runs")
    return rule
