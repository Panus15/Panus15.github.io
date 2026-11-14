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


def test_risk_levels_and_weights():
    from engine.score import risk, CUE_WEIGHTS, CONTRADICTED_WEIGHT
    def cue(tag, text="x"):
        return Finding(Claim("q1", ClaimKind.CUE, text, 0, len(text)), Status.UNVERIFIED, "cue", "r", (), tag)
    assert risk([])["level"] == "none"
    assert risk([F(Status.VERIFIED)])["level"] == "none"
    assert risk([cue("time_sensitive")])["level"] == "low"
    assert risk([cue("unsourced_quote")])["level"] == "medium"            # weight 3
    assert risk([cue("unsourced_quote"), cue("placeholder_url"), cue("count_mismatch")])["level"] == "high"
    assert risk([F(Status.CONTRADICTED)])["level"] == "high"              # one proven error is always high
    assert risk([cue("unknown_future_tag")])["score"] == 1                # unknown tag: default weight, never a crash
    assert risk([cue("consistency_unavailable")])["level"] == "none"
    assert CONTRADICTED_WEIGHT > max(CUE_WEIGHTS.values()) * 3


def test_risk_top_is_heaviest_first_and_capped():
    from engine.score import risk
    fs = [Finding(Claim(f"q{i}", ClaimKind.CUE, "x", 0, 1), Status.UNVERIFIED, "cue", "r", (), t)
          for i, t in enumerate(["time_sensitive", "unsourced_quote", "absolute_claim"] * 3)]
    top = risk(fs)["top"]
    assert len(top) == 5 and top[0].tag == "unsourced_quote"


def test_report_shows_risk_line_and_json_risk():
    import json
    md = report.render([F(Status.UNVERIFIED)])
    assert "ระดับความเสี่ยง" in md and "ยังไม่ได้ปรับ" in md
    assert json.loads(json.dumps(report.to_dict([F(Status.CONTRADICTED)]), ensure_ascii=False))["risk"]["level"] == "high"


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("ok  ", name)
            except Exception as e:
                fails += 1; print("FAIL", name, repr(e))
    sys.exit(1 if fails else 0)
