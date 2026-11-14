"""Tests for verdict roll-up and report rendering. Offline, no checkers involved.

Run: python3 tests/test_score.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.models import Claim, ClaimKind, Finding, Status
from engine.score import EXIT_CODE, Verdict, counts, verdict
from engine import report


def F(status, text="x", kind=ClaimKind.URL, evidence=(), reason="r"):
    return Finding(Claim("c1", kind, text, 0, len(text)), status, "t", reason, tuple(evidence))


def test_one_contradiction_outweighs_everything():
    fs = [F(Status.VERIFIED)] * 50 + [F(Status.UNVERIFIED), F(Status.CONTRADICTED)]
    assert verdict(fs) is Verdict.DO_NOT_USE


def test_any_unverified_means_review_never_pass():
    assert verdict([F(Status.VERIFIED), F(Status.UNVERIFIED)]) is Verdict.REVIEW


def test_nothing_checkable_is_not_a_pass():
    v = verdict([])
    assert v is Verdict.NO_CHECKABLE_CLAIMS and v is not Verdict.NO_PROBLEMS_FOUND


def test_all_verified():
    assert verdict([F(Status.VERIFIED)]) is Verdict.NO_PROBLEMS_FOUND


def test_exit_codes_order_by_severity():
    assert EXIT_CODE[Verdict.NO_PROBLEMS_FOUND] == 0 == EXIT_CODE[Verdict.NO_CHECKABLE_CLAIMS]
    assert EXIT_CODE[Verdict.REVIEW] == 1 and EXIT_CODE[Verdict.DO_NOT_USE] == 2


def test_counts_include_zero_buckets():
    c = counts([F(Status.VERIFIED)])
    assert c == {Status.VERIFIED: 1, Status.UNVERIFIED: 0, Status.CONTRADICTED: 0}


def test_report_puts_contradicted_first_and_has_headline():
    md = report.render([F(Status.VERIFIED, "ok"), F(Status.CONTRADICTED, "bad")])
    assert md.index("bad") < md.index("ok")
    assert "ห้ามใช้" in md


def test_report_neutralises_markdown_injection():
    evil = "](http://evil) ![x](http://evil/p.png) `` ``` \n# heading\x1b[31m"
    md = report.render([F(Status.UNVERIFIED, evil, evidence=[evil], reason="a\n# h")])
    lines = md.splitlines()
    assert not any(l.startswith("# heading") for l in lines)
    assert "\x1b" not in md
    # every occurrence of the hostile text sits inside a code span: no live link syntax outside spans
    stripped = report.code_span(evil)
    assert stripped.startswith("```") or stripped.startswith("`")


def test_code_span_fences_out_backticks():
    s = report.code_span("a``b")
    assert s.startswith("```") and s.endswith("```")
    s = report.code_span("`edge`")
    assert s.startswith("`` ") and s.endswith(" ``")


def test_long_text_is_truncated():
    assert len(report.code_span("a" * 5000)) < 300


def test_to_dict_roundtrips_json():
    import json
    d = report.to_dict([F(Status.CONTRADICTED, "ไทย")])
    assert json.loads(json.dumps(d, ensure_ascii=False))["verdict"] == "do_not_use"


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("ok  ", name)
            except Exception as e:
                fails += 1; print("FAIL", name, repr(e))
    sys.exit(1 if fails else 0)
