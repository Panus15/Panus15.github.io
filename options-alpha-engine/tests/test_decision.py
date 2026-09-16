"""Tests for the fused decision (models/decision.py).

One property matters more than the rest and most of these tests exist to defend
it: an input with no evidence behind it may only REDUCE size. If sector rotation
or fund crowding could ever raise conviction, an unmeasured signal would be
manufacturing position size — the exact way a research codebase turns into a loss.

Run: python3 tests/test_decision.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.decision import (FIXTURE_ONLY, NO_DATA, TESTED_NO_EDGE, VALIDATED,
                             Decision, Input, build_decision)
from models.rotation import RotationPoint


class _Card:
    """Stand-in for a TradeCard: the fusion must not care which one it gets."""

    def __init__(self, side="SELL VOL", vrp=0.013, direction="LONG"):
        self.symbol, self.asof, self.expiry_days = "XLK", "2026-07-31", 30
        self.vol_side, self.vol_reason = side, "market implies more variance"
        self.p_vol, self.q_vol, self.vrp = 0.179, 0.212, vrp
        self.direction, self.direction_score = direction, 0.25
        self.warnings = []


def _pt(quadrant, rs=101.0, mom=101.0):
    return RotationPoint(symbol="XLK", name="Technology", rs_ratio=rs,
                         rs_momentum=mom, quadrant=quadrant)


def test_a_clean_setup_keeps_full_size():
    d = build_decision(_Card(), rotation_point=_pt("Leading"), crowding=(0.02, ""))
    assert d.action == "SELL VOL"
    assert d.size_multiplier == 1.0
    assert all(i.effect == 1.0 for i in d.inputs)


def test_nothing_can_exceed_the_variance_edge_alone():
    """The asymmetry the whole module is built on: every effect is a reduction."""
    best = build_decision(_Card(), rotation_point=_pt("Leading", 130.0, 130.0),
                          crowding=(0.0, ""))
    assert best.size_multiplier == 1.0, "nothing may exceed the variance edge alone"
    for q in ("Leading", "Weakening", "Lagging", "Improving"):
        for share in (0.0, 0.2, 0.9):
            d = build_decision(_Card(), rotation_point=_pt(q), crowding=(share, ""))
            assert d.size_multiplier <= 1.0, (q, share, d.size_multiplier)
            assert all(i.effect <= 1.0 for i in d.inputs)


def test_a_disproven_signal_does_not_move_size_in_either_direction():
    """Rule 2. The quadrant was measured on real data and predicted nothing
    (§7.1, t=+0.71 on a test that charges no costs), so it must not spend size.

    Cutting on a disproven signal is not caution — it is acting on noise while
    printing a sentence that looks like reasoning. A re-introduced quadrant cut
    would sail through every other test in this file, so it is pinned here.
    """
    sizes = {q: build_decision(_Card(), rotation_point=_pt(q),
                               crowding=(0.0, "")).size_multiplier
             for q in ("Leading", "Weakening", "Lagging", "Improving")}
    assert set(sizes.values()) == {1.0}, sizes

    for q in sizes:
        d = build_decision(_Card(), rotation_point=_pt(q), crowding=(0.0, ""))
        rot = [i for i in d.inputs if i.name == "sector rotation"][0]
        assert rot.effect == 1.0, (q, rot.effect)
        assert rot.status == TESTED_NO_EDGE, rot.status
        assert q in rot.reading, "the quadrant is still reported as context"


def test_crowding_is_the_only_input_still_trimming():
    """Crowding stays under rule 1 — untested, so it may still subtract.

    It is the one reduction left, so its multiplier is pinned exactly: a
    silently neutered branch would leave the fusion doing nothing at all while
    still printing three inputs.
    """
    crowd = build_decision(_Card(), rotation_point=_pt("Leading"), crowding=(0.8, ""))
    assert abs(crowd.size_multiplier - 0.6) < 1e-9, crowd.size_multiplier
    assert "size reduced by" in crowd.render()
    assert "fund crowding" in crowd.render()

    # and the quadrant cannot change that number, whichever one it is
    for q in ("Lagging", "Weakening", "Leading", "Improving"):
        d = build_decision(_Card(), rotation_point=_pt(q), crowding=(0.8, ""))
        assert abs(d.size_multiplier - 0.6) < 1e-9, (q, d.size_multiplier)

    quiet = build_decision(_Card(), rotation_point=_pt("Lagging"), crowding=(0.02, ""))
    assert quiet.size_multiplier == 1.0
    assert "sector rotation" in quiet.render(), "the reading is still shown"
    assert "size reduced by" not in quiet.render()


def test_direction_is_carried_but_powerless():
    """It is the least-evidenced thing on the card and must size nothing."""
    long_ = build_decision(_Card(direction="LONG"), rotation_point=_pt("Leading"),
                           crowding=(0.0, ""))
    short = build_decision(_Card(direction="SHORT"), rotation_point=_pt("Leading"),
                           crowding=(0.0, ""))
    assert long_.size_multiplier == short.size_multiplier == 1.0
    d_in = [i for i in long_.inputs if i.name == "direction"][0]
    assert d_in.effect == 1.0 and d_in.status == FIXTURE_ONLY
    assert "does not size anything" in d_in.note


def test_no_trade_from_the_variance_edge_ends_it():
    """The one input with evidence is the only one that can CREATE a trade."""
    d = build_decision(_Card(side="NO TRADE"), rotation_point=_pt("Leading"),
                       crowding=(0.0, ""))
    assert d.action == "NO TRADE" and d.size_multiplier == 0.0
    # ...and no amount of favourable context resurrects it
    for q in ("Leading", "Improving"):
        assert build_decision(_Card(side="NO TRADE"),
                              rotation_point=_pt(q)).action == "NO TRADE"


def test_missing_inputs_are_reported_as_missing_not_as_benign():
    d = build_decision(_Card(), rotation_point=None, crowding=None)
    rot = [i for i in d.inputs if i.name == "sector rotation"][0]
    crd = [i for i in d.inputs if i.name == "fund crowding"][0]
    assert rot.status == NO_DATA and crd.status == NO_DATA
    assert "absence of DATA" in rot.note
    assert "not an empty market" in crd.note
    # missing data does not reduce size either — it is stated, not guessed at
    assert d.size_multiplier == 1.0


def test_the_reduction_line_never_names_nobody():
    """A NO TRADE zeroes the size through a different path than an input cut, so
    keying the line on the multiplier printed 'size reduced by:' with nothing
    after it — a blank accusation against no input at all."""
    d = Decision(symbol="X", asof=None, expiry_days=30, action="NO TRADE",
                 reason="the variance edge did not fire", size_multiplier=0.0,
                 inputs=[Input("variance edge", "no signal", VALIDATED, 1.0),
                         Input("sector rotation", "X Lagging", TESTED_NO_EDGE, 1.0)])
    assert "size reduced by" not in d.render(), d.render()

    cut = Decision(symbol="X", asof=None, expiry_days=30, action="SELL VOL",
                   reason="", size_multiplier=0.6,
                   inputs=[Input("fund crowding", "80%", NO_DATA, 0.6)])
    assert "size reduced by: fund crowding" in cut.render()


def test_a_disproven_input_ranks_below_an_untested_one():
    """Ordering, tested directly because nothing reaches it today.

    Rotation now sits at effect 1.0, so `weakest_evidence` never sees this status
    and a wrong rank would be invisible. It still has to be right: if some future
    input is disproven AND acting, it is the least trustworthy thing in the room
    — worse than untested, because untested might yet be true. Ranking it
    alongside `validated` would print the most reassuring label available next to
    the least supported number.
    """
    def weakest(*statuses):
        return Decision(symbol="X", asof=None, expiry_days=30, action="SELL VOL",
                        reason="", inputs=[Input(f"i{n}", "r", s, 0.5)
                                           for n, s in enumerate(statuses)]
                        ).weakest_evidence

    assert weakest(VALIDATED, TESTED_NO_EDGE) == TESTED_NO_EDGE
    assert weakest(TESTED_NO_EDGE, NO_DATA) == TESTED_NO_EDGE
    assert weakest(TESTED_NO_EDGE, FIXTURE_ONLY) == TESTED_NO_EDGE
    assert weakest(VALIDATED, FIXTURE_ONLY) == FIXTURE_ONLY
    assert weakest(VALIDATED, NO_DATA) == NO_DATA


def test_the_weakest_acting_evidence_is_surfaced():
    # nothing reduced the size, so the decision rests on the variance edge alone
    strong = build_decision(_Card(), rotation_point=_pt("Leading"), crowding=(0.0, ""))
    assert strong.size_multiplier == 1.0
    assert strong.weakest_evidence == VALIDATED, (
        "with no input acting, reporting a weaker status would misdescribe a "
        "decision that rests entirely on the edge")
    assert not strong.warnings

    # a disproven input that does NOT act must not drag the level down either —
    # it is the worst-ranked status in the table, so if `weakest_evidence` ever
    # stopped filtering on effect it would report every decision as no-edge
    for q in ("Lagging", "Weakening", "Leading", "Improving"):
        d = build_decision(_Card(), rotation_point=_pt(q), crowding=(0.0, ""))
        assert d.weakest_evidence == VALIDATED, (q, d.weakest_evidence)

    crowded = build_decision(_Card(), rotation_point=_pt("Leading"),
                             crowding=(0.9, ""))
    assert crowded.weakest_evidence == NO_DATA
    assert any("NO-DATA input is moving this decision" in w
               for w in crowded.warnings), crowded.warnings
    assert "weakest evidence" in crowded.render()


def test_the_variance_edge_is_the_only_validated_input():
    d = build_decision(_Card(), rotation_point=_pt("Leading"), crowding=(0.0, ""))
    validated = [i.name for i in d.inputs if i.status == VALIDATED]
    assert validated == ["variance edge"], validated
    assert "unproven on a live option market" in d.inputs[0].note


def test_card_warnings_survive_the_fusion():
    c = _Card()
    c.warnings = ["the P vol looks ~44% too LOW here"]
    d = build_decision(c, rotation_point=_pt("Leading"), crowding=(0.0, ""))
    assert any("44%" in w for w in d.warnings)
    assert "44%" in d.render()


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
