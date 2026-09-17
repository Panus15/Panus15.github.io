"""Tests for the pattern detectors.

A pattern detector is easy to write and easy to fool yourself with, so these test
the two things that actually matter:

  1. NO LOOK-AHEAD. A detection at bar i must not depend on any bar after i. This
     is the defect that makes every pattern backtest look profitable, and it is
     tested by truncating the series at the detection and re-running.
  2. THE EVIDENCE LABEL TRAVELS WITH THE DETECTION. The research found only one of
     these patterns has any peer-reviewed FX test and that it failed; the rest have
     none at all. A detection that arrives without that label is a recommendation
     wearing a measurement's clothes.

Shape tests come from hand-built series where the answer is known by construction.

Run: python3 tests/test_patterns.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.patterns import (DETECTORS, EQUAL_TOL, EVIDENCE, FAILS_RANDOM_NULL,
                             MAX_SPAN, MIN_HEIGHT, MIN_SPAN, NEVER_TESTED,
                             PARAMETER_COUNT,
                             SWING_BARS, TESTED_AND_FAILED, Bar, Detection,
                             double_bottom, double_top, engulfing, flag,
                             head_and_shoulders, scan, swing_highs, swing_lows,
                             triangle)


def _bar(price, spread=0.0005):
    """A flat bar at `price` — the filler between the shapes under test."""
    return Bar(price, price + spread, price - spread, price)


def _ramp(a, b, n):
    """n bars walking linearly from a to b."""
    if n <= 1:
        return [_bar(b)]
    step = (b - a) / (n - 1)
    return [_bar(a + step * i) for i in range(n)]


def _peak(base, top, up, down):
    return _ramp(base, top, up) + _ramp(top, base, down)[1:]


# --------------------------------------------------------------------------
# The primitive
# --------------------------------------------------------------------------

def test_a_swing_high_needs_k_lower_bars_on_both_sides():
    bars = _ramp(1.10, 1.12, 8) + _ramp(1.12, 1.10, 8)[1:]
    highs = swing_highs(bars)
    assert highs, "a clean peak produced no swing high"
    assert all(SWING_BARS <= i < len(bars) - SWING_BARS for i in highs)
    top = max(range(len(bars)), key=lambda i: bars[i].h)
    assert top in highs


def test_a_monotone_series_has_no_interior_swings():
    """The commonest false positive: a trend is not a sequence of patterns."""
    assert swing_highs(_ramp(1.10, 1.20, 60)) == []
    assert swing_lows(_ramp(1.20, 1.10, 60)) == []


def test_swings_ignore_single_bar_noise():
    flat = [_bar(1.10) for _ in range(40)]
    flat[20] = Bar(1.10, 1.1002, 1.0998, 1.10)      # a one-tick blip
    assert 20 not in swing_highs(flat), "a single-tick blip was called a swing"


# --------------------------------------------------------------------------
# The property that decides whether any of this means anything
# --------------------------------------------------------------------------

def _all_detections(bars):
    return scan(bars)


def test_no_detector_looks_at_a_bar_after_the_one_it_fires_on():
    """THE test. A detection at bar i must survive truncating the series at i —
    if it does not, the detector saw the future, and every backtest built on it is
    reporting a profit that was not available at the time.

    Run over several shapes so each detector gets exercised.
    """
    series = [
        # head and shoulders: shoulder, higher head, shoulder, break
        _ramp(1.10, 1.13, 8) + _ramp(1.13, 1.105, 8)[1:]
        + _ramp(1.105, 1.16, 10)[1:] + _ramp(1.16, 1.104, 10)[1:]
        + _ramp(1.104, 1.131, 8)[1:] + _ramp(1.131, 1.08, 14)[1:],
        # double top
        _peak(1.10, 1.14, 10, 10) + _peak(1.10, 1.1402, 10, 14)[1:],
        # double bottom
        _peak(1.14, 1.10, 10, 10) + _peak(1.14, 1.0998, 10, 14)[1:],
        # triangle: converging
        _peak(1.10, 1.16, 8, 8) + _peak(1.115, 1.145, 8, 8)[1:]
        + _ramp(1.13, 1.18, 16)[1:],
        # flag: pole then shallow drift then continuation
        _ramp(1.10, 1.14, 16) + _ramp(1.14, 1.128, 6)[1:] + _ramp(1.128, 1.17, 12)[1:],
    ]
    checked = 0
    for bars in series:
        for d in _all_detections(bars):
            truncated = bars[:d.index + 1]
            again = [x for x in _all_detections(truncated)
                     if x.name == d.name and x.index == d.index]
            assert again, (
                f"{d.name} at bar {d.index} vanished when the series was cut at "
                f"that bar — the detector used a later bar")
            got = again[0]
            for field in ("direction", "entry", "stop", "target"):
                assert abs(getattr(got, field) - getattr(d, field)) < 1e-12 \
                    if isinstance(getattr(d, field), float) \
                    else getattr(got, field) == getattr(d, field), (d.name, field)
            checked += 1
    assert checked >= 4, f"only {checked} detections were exercised; widen the fixtures"


def test_engulfing_fires_only_on_the_bar_that_completes_it():
    bars = [_bar(1.10) for _ in range(5)]
    bars[2] = Bar(1.1020, 1.1025, 1.0995, 1.1000)      # down body
    bars[3] = Bar(1.0995, 1.1035, 1.0990, 1.1030)      # up body engulfing it
    found = engulfing(bars)
    assert [d.index for d in found] == [3], [d.index for d in found]
    assert found[0].direction == +1
    # and it survives truncation at its own bar
    assert [d.index for d in engulfing(bars[:4])] == [3]


# --------------------------------------------------------------------------
# Shapes, where the answer is known by construction
# --------------------------------------------------------------------------

def test_head_and_shoulders_is_short_with_the_stop_above_the_right_shoulder():
    bars = (_ramp(1.10, 1.13, 8) + _ramp(1.13, 1.105, 8)[1:]
            + _ramp(1.105, 1.16, 10)[1:] + _ramp(1.16, 1.104, 10)[1:]
            + _ramp(1.104, 1.131, 8)[1:] + _ramp(1.131, 1.08, 14)[1:])
    found = head_and_shoulders(bars)
    assert found, "the textbook shape produced no detection"
    d = found[0]
    assert d.direction == -1
    assert d.stop > d.entry, "a short's stop must sit above its entry"
    assert d.target < d.entry
    assert d.risk > 0 and d.reward > 0


def test_a_head_that_is_not_the_highest_point_is_not_a_head():
    """Shoulders above the head is not the pattern, however suggestive it looks.

    The shoulders here are deliberately EQUAL to each other, so the only guard
    that can reject this shape is the one being tested. An earlier version had
    mismatched shoulders and was therefore rejected by a different rule, which let
    a broken head check pass a mutation sweep.
    """
    bars = (_peak(1.10, 1.16, 8, 8) + _peak(1.10, 1.13, 8, 8)[1:]
            + _peak(1.10, 1.16, 8, 12)[1:] + _ramp(1.10, 1.06, 12)[1:])
    highs = swing_highs(bars)
    assert len(highs) >= 3, "the fixture must present three peaks to be meaningful"
    assert head_and_shoulders(bars) == [], "a dip between two peaks became a 'head'"


def test_shoulders_that_do_not_match_are_not_a_head_and_shoulders():
    """A real head, but a right shoulder 3% below the left. EQUAL_TOL is 0.15%, so
    this is twenty times outside it — written as a literal rather than as a
    multiple of the constant, because a fixture scaled by the parameter under test
    moves with the bug."""
    bars = (_ramp(1.10, 1.13, 8) + _ramp(1.13, 1.105, 8)[1:]
            + _ramp(1.105, 1.16, 10)[1:] + _ramp(1.16, 1.104, 10)[1:]
            + _ramp(1.104, 1.1261, 8)[1:] + _ramp(1.1261, 1.08, 14)[1:])
    assert head_and_shoulders(bars) == [], (
        "shoulders 3% apart were accepted as a matched pair")


def test_double_top_and_bottom_are_mirror_images():
    # the tail must carry price THROUGH the neckline — a decline that stops at the
    # trough never completes the pattern, which is what the detector is right to
    # insist on and what the first version of this fixture forgot
    up = (_peak(1.10, 1.14, 10, 10) + _peak(1.10, 1.1402, 10, 10)[1:]
          + _ramp(1.10, 1.09, 6)[1:])
    down = (_peak(1.14, 1.10, 10, 10) + _peak(1.14, 1.0998, 10, 10)[1:]
            + _ramp(1.14, 1.15, 6)[1:])
    t, b = double_top(up), double_bottom(down)
    assert t and b, (len(t), len(b))
    assert t[0].direction == -1 and b[0].direction == +1
    assert t[0].stop > t[0].entry and b[0].stop < b[0].entry


def test_two_tops_that_are_not_close_enough_are_not_a_double_top():
    """EQUAL_TOL is a locked parameter and this is what it buys.

    The gap is a LITERAL 3%, not a multiple of EQUAL_TOL. Scaling the fixture by
    the constant under test is self-referential: widening EQUAL_TOL to 0.5 would
    widen the fixture too and the test would keep passing.
    """
    assert EQUAL_TOL < 0.01, "the literal gap below assumes a sub-1% tolerance"
    far = (_peak(1.10, 1.14, 10, 10) + _peak(1.10, 1.1742, 10, 10)[1:]
           + _ramp(1.10, 1.09, 6)[1:])
    assert double_top(far) == [], "tops 3% apart were called equal"


def test_a_pattern_shorter_than_the_floor_is_refused():
    """The peaks must be real swings, so that SPAN is the only thing rejecting it.
    A fixture whose ramps are too short to produce swing points at all is rejected
    by the swing detector instead, and proves nothing about the span limit."""
    assert MIN_SPAN < MAX_SPAN
    tight = (_peak(1.10, 1.115, 5, 5) + _peak(1.10, 1.1151, 5, 5)[1:]
             + _ramp(1.10, 1.09, 6)[1:])
    highs = swing_highs(tight)
    assert len(highs) >= 2, "the fixture must present two real swing highs"
    assert highs[1] - highs[0] < MIN_SPAN, (
        f"the peaks are {highs[1] - highs[0]} bars apart, not under MIN_SPAN")
    assert double_top(tight) == [], "a formation below MIN_SPAN was accepted"


def test_a_head_and_shoulders_packed_into_too_few_bars_is_refused():
    """MIN_SPAN on the H&S path, isolated. All three peaks are genuine swings and
    the shoulders match, so the span limit is the only rule left that can reject
    this — which is what makes it a test of the span limit."""
    bars = (_peak(1.100, 1.115, 4, 3) + _peak(1.100, 1.125, 3, 3)[1:]
            + _peak(1.100, 1.115, 3, 4)[1:] + _ramp(1.100, 1.090, 5)[1:])
    highs = swing_highs(bars)
    assert len(highs) >= 3, "the fixture must present three real swing highs"
    assert highs[2] - highs[0] < MIN_SPAN, (
        f"the formation spans {highs[2] - highs[0]} bars, not under MIN_SPAN")
    assert bars[highs[1]].h > bars[highs[0]].h, "the middle peak must be the head"
    assert head_and_shoulders(bars) == [], (
        f"a {highs[2] - highs[0]}-bar formation cleared MIN_SPAN={MIN_SPAN}")


def test_a_head_and_shoulders_with_no_height_is_refused():
    """MIN_HEIGHT on the H&S path, isolated. Real swings, matched shoulders, a
    genuine head — and a head-to-neckline height of 0.15% against a 0.2% floor."""
    base = 1.10
    bars = (_peak(base, base * 1.00036, 8, 8)
            + _peak(base, base * 1.0006, 8, 8)[1:]
            + _peak(base, base * 1.00036, 8, 8)[1:]
            + _ramp(base, base * 0.9995, 6)[1:])
    highs, lows = swing_highs(bars), swing_lows(bars)
    assert len(highs) >= 3 and lows, "the fixture must present the full shape"
    neck = min(bars[i].l for i in lows)
    height_frac = (bars[highs[1]].h - neck) / bars[highs[1]].h
    assert height_frac < MIN_HEIGHT, (
        f"the fixture is {height_frac:.5f} tall, not under MIN_HEIGHT={MIN_HEIGHT}")
    assert head_and_shoulders(bars) == [], "a formation with no height was accepted"


def test_a_formation_with_real_swings_but_no_height_is_refused():
    """MIN_HEIGHT, isolated. The swings are genuine — so the only rule that can
    reject this is the height floor. A flat noise series is rejected earlier, by
    the swing detector, and therefore tests nothing here."""
    up = _peak(1.1000, 1.1012, 10, 10)          # 0.11% tall, MIN_HEIGHT is 0.2%
    bars = up + _peak(1.1000, 1.1012, 10, 10)[1:] + _ramp(1.1000, 1.0995, 6)[1:]
    assert len(swing_highs(bars)) >= 2, "the fixture must present two real swings"
    assert double_top(bars) == [], "a 0.11%-tall formation cleared MIN_HEIGHT"


def test_a_flat_stretch_of_noise_is_not_a_formation():
    """MIN_HEIGHT exists so that a range of nothing is not reported as structure."""
    flat = []
    for i in range(200):
        p = 1.10 + (0.00002 if i % 2 else -0.00002)
        flat.append(_bar(p))
    assert scan(flat) == [], "noise was reported as patterns"


def test_every_detection_has_a_usable_risk_and_a_planned_payoff():
    bars = (_ramp(1.10, 1.14, 16) + _ramp(1.14, 1.128, 6)[1:]
            + _ramp(1.128, 1.17, 12)[1:])
    for d in scan(bars):
        assert d.risk > 0, f"{d.name} produced a zero-width stop"
        assert d.planned_payoff > 0
        # the target must be on the profitable side of the entry
        assert (d.target - d.entry) * d.direction > 0, d.name


# --------------------------------------------------------------------------
# The evidence label, which is the point
# --------------------------------------------------------------------------

def test_every_detector_has_an_evidence_label():
    for name in DETECTORS:
        assert name in EVIDENCE, f"{name} detects but says nothing about its evidence"
        verdict, detail = EVIDENCE[name]
        assert verdict and detail, name
        assert len(detail) > 40, f"{name}'s evidence detail is too thin to be useful"


def test_the_labels_match_what_the_research_actually_found():
    """Not editorial. Only head-and-shoulders has a substantial peer-reviewed FX
    literature and it is negative; candlesticks fail a randomised null; the rest
    have no FX test at all."""
    assert EVIDENCE["head_and_shoulders"][0] == TESTED_AND_FAILED
    assert EVIDENCE["engulfing"][0] == FAILS_RANDOM_NULL
    for n in ("double_top", "double_bottom", "triangle", "flag"):
        assert EVIDENCE[n][0] == NEVER_TESTED, n
    # the named papers must survive a rewrite of the prose
    hs = EVIDENCE["head_and_shoulders"][1]
    for cite in ("Lucke", "Chang & Osler", "Osler"):
        assert cite in hs, cite
    assert "Marshall" in EVIDENCE["engulfing"][1]
    assert "Bulkowski" in EVIDENCE["double_top"][1], (
        "the provenance of the quoted win-rate tables is the whole point")


def test_the_label_reaches_the_detection_not_just_the_table():
    d = Detection("head_and_shoulders", 10, -1, 1.10, 1.11, 1.09)
    assert d.evidence == TESTED_AND_FAILED
    assert "Lucke" in d.evidence_detail
    unknown = Detection("wedge", 10, 1, 1.10, 1.09, 1.12)
    assert unknown.evidence == NEVER_TESTED, (
        "an unlabelled pattern must default to 'never tested', never to silence")


def test_the_free_parameters_are_declared_and_counted():
    """Free parameters are how this kind of study is overfitted, so the count is
    published for the multiple-testing correction downstream to use."""
    assert PARAMETER_COUNT == 4
    for p in (SWING_BARS, EQUAL_TOL, MIN_SPAN, MAX_SPAN):
        assert p > 0


def test_scan_returns_detections_in_bar_order():
    """The fixture must contain detections from MORE THAN ONE detector, interleaved
    in time. `scan` concatenates per-detector lists, so a fixture whose detections
    happen to already be in order would pass without any sorting at all."""
    bars = (_ramp(1.10, 1.14, 16) + _ramp(1.14, 1.128, 6)[1:]
            + _ramp(1.128, 1.17, 12)[1:] + _peak(1.17, 1.20, 10, 10)[1:]
            + _peak(1.17, 1.2002, 10, 10)[1:] + _ramp(1.17, 1.15, 8)[1:])
    found = scan(bars)
    assert len({d.name for d in found}) >= 2, (
        f"only {[d.name for d in found]} — the fixture cannot detect mis-ordering")
    unsorted = [d.index for sub in
                (fn(bars) for fn in DETECTORS.values()) for d in sub]
    assert unsorted != sorted(unsorted), (
        "the raw per-detector order is already sorted, so this fixture would pass "
        "even with no sorting")
    assert [d.index for d in found] == sorted(d.index for d in found)


def test_scan_can_be_narrowed_to_named_patterns():
    bars = _peak(1.10, 1.14, 10, 10) + _peak(1.10, 1.1402, 10, 14)[1:]
    only = scan(bars, names=["double_top"])
    assert all(d.name == "double_top" for d in only)
    assert len(only) <= len(scan(bars))


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
