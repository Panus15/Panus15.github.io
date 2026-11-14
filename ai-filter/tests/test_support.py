"""Tests for engine/support.py. Offline; llm / fetcher / parse / judge are stubs.

Run: python3 tests/test_support.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import extract
from engine.evidence import Judgement, Page
from engine.llm import LLMError
from engine.models import Claim, ClaimKind, Finding, Status
from engine.net import FetchResult
from engine.support import assertion_for, is_checkable, refine

NOT_STATED = "The link opens, but we could not find this claim on the page — read it yourself."


# ---------------------------------------------------------------- stubs
class StubFetcher:
    def __init__(self):
        self.calls = []

    def fetch(self, url, *, headers=None, timeout=10.0, max_bytes=0):
        self.calls.append((url, max_bytes))
        return FetchResult(url, url, 200, b"<html>page for " + url.encode() + b"</html>")


class RaisingFetcher(StubFetcher):
    def fetch(self, url, **kw):
        self.calls.append((url, kw.get("max_bytes")))
        raise RuntimeError("boom")


def stub_parse(result):
    return Page(result.final_url, "t", "page text of " + result.url)


class StubJudge:
    """verdict per URL (default 'supports')."""
    def __init__(self, verdicts=None, default="supports", quote="the page quote"):
        self.verdicts = verdicts or {}
        self.default = default
        self.quote = quote
        self.calls = []

    def __call__(self, statement, page, llm):
        self.calls.append((statement, page.url))
        v = self.verdicts.get(page.url, self.default)
        if isinstance(v, Exception):
            raise v
        return Judgement(v, self.quote if v in ("supports", "contradicts") else "", "r")


def citation_findings(text):
    """What the pipeline has before refine: citations findings, VERIFIED for URLs."""
    out = []
    for c in extract.extract(text):
        if c.kind in (ClaimKind.URL, ClaimKind.DOI):
            out.append(Finding(c, Status.VERIFIED, "citations", "The link opens, but ...", (c.text,)))
    return out


def run(text, judge=None, fetcher=None, findings=None, **kw):
    judge = judge or StubJudge()
    fetcher = fetcher or StubFetcher()
    fs = findings if findings is not None else citation_findings(text)
    res = refine(text, fs, llm=object(), fetcher=fetcher, parse_page_fn=kw.pop("parse_page_fn", stub_parse),
                 judge_fn=judge, **kw)
    return fs, res, fetcher, judge


LONG = "Acme Corp reported revenue growth of 12% in 2023 across all regions"


# ---------------------------------------------------------------- outcomes
def test_supports_verifies_with_quote():
    t = LONG + ", see https://acme.example/report for details."
    fs, res, f, j = run(t)
    assert len(res) == 1 and res[0].status == Status.VERIFIED and res[0].module == "support"
    assert res[0].claim is fs[0].claim
    assert res[0].reason == "The page says: “the page quote”"
    assert res[0].evidence == ("https://acme.example/report",)
    assert f.calls == [("https://acme.example/report", 512 * 1024)]
    assert j.calls[0][0].startswith("Acme Corp reported revenue growth of 12%")
    assert "https://" not in j.calls[0][0]


def test_quote_is_capped_at_200():
    t = LONG + " https://acme.example/r"
    _, res, _, _ = run(t, judge=StubJudge(quote="word " * 100))
    q = res[0].reason[len("The page says: “"):-1]
    assert len(q) <= 200 and q.endswith("…")


def test_not_stated_downgrades_to_unverified():
    t = LONG + " https://acme.example/report"
    _, res, _, _ = run(t, judge=StubJudge(default="not_stated"))
    assert res[0].status == Status.UNVERIFIED and res[0].module == "support"
    assert res[0].reason == NOT_STATED


def test_contradicts():
    t = LONG + " https://acme.example/report"
    _, res, _, _ = run(t, judge=StubJudge(default="contradicts", quote="revenue fell 3%"))
    assert res[0].status == Status.CONTRADICTED and res[0].module == "support"
    assert res[0].reason == "The page says: “revenue fell 3%” — this differs from what the answer claims."


def test_error_verdict_passes_through_unchanged():
    t = LONG + " https://acme.example/report"
    fs, res, _, _ = run(t, judge=StubJudge(default="error"))
    assert res[0] is fs[0]


def test_supports_without_quote_passes_through():
    t = LONG + " https://acme.example/report"
    fs, res, _, _ = run(t, judge=StubJudge(quote=""))
    assert res[0] is fs[0]


def test_llmerror_stops_further_checks():
    t = (LONG + " https://a.example/1\n" + LONG.replace("12%", "15%") + " https://b.example/2\n"
         + LONG.replace("12%", "18%") + " https://c.example/3\n")
    judge = StubJudge({"https://a.example/1": "supports", "https://b.example/2": LLMError("budget")})
    fs, res, f, j = run(t, judge=judge)
    assert len(fs) == 3 and res[0].status == Status.VERIFIED and res[0].module == "support"
    assert res[1] is fs[1] and res[2] is fs[2]
    assert [u for u, _ in f.calls] == ["https://a.example/1", "https://b.example/2"]  # c never fetched
    assert len(j.calls) == 2


def test_exception_containment():
    t = (LONG + " https://a.example/1\n" + LONG.replace("12%", "15%") + " https://b.example/2\n")
    judge = StubJudge({"https://a.example/1": ValueError("bug")})
    fs, res, _, _ = run(t, judge=judge)
    assert res[0] is fs[0]                           # exception -> unchanged
    assert res[1].module == "support"                # and the next candidate is still tried
    fs, res, f, _ = run(t, fetcher=RaisingFetcher())
    assert res == fs and all(a is b for a, b in zip(res, fs)) and len(f.calls) == 2

    def bad_parse(r):
        raise RuntimeError("x")
    fs, res, _, _ = run(t, parse_page_fn=bad_parse)
    assert all(a is b for a, b in zip(res, fs))


def test_parse_none_passes_through():
    t = LONG + " https://acme.example/doc.pdf"
    fs, res, f, j = run(t, parse_page_fn=lambda r: None)
    assert res[0] is fs[0] and len(f.calls) == 1 and j.calls == []


# ---------------------------------------------------------------- candidates
def test_non_candidates_untouched_same_objects():
    t = LONG + " https://acme.example/ok\n"
    ok = citation_findings(t)[0]
    c = ok.claim
    other = [
        Finding(c, Status.UNVERIFIED, "citations", "could not check"),
        Finding(c, Status.CONTRADICTED, "citations", "404"),
        Finding(c, Status.VERIFIED, "vague", "x"),
        Finding(c, Status.VERIFIED, "support", "already refined"),
        Finding(Claim("c9", ClaimKind.DOI, "10.1/x", 0, 6, {"doi": "10.1/x"}), Status.VERIFIED, "citations", "registered"),
        Finding(Claim("c8", ClaimKind.REFERENCE, "x", 0, 1, {"title": "t"}), Status.VERIFIED, "citations", "exists"),
    ]
    fs = other[:3] + [ok] + other[3:]
    _, res, f, _ = run(t, findings=fs)
    assert len(res) == len(fs)
    for i, (a, b) in enumerate(zip(fs, res)):
        if i == 3:
            assert b.module == "support"
        else:
            assert a is b, i
    assert len(f.calls) == 1


def test_length_and_order_preserved():
    ts = [LONG.replace("12%", "%d%%" % n) + " https://h.example/%d\n" % n for n in range(1, 5)]
    t = "".join(ts)
    fs = citation_findings(t)
    mid = Finding(Claim("v1", ClaimKind.VAGUE, "x", 0, 1, {}), Status.UNVERIFIED, "vague", "r")
    fs = [fs[0], mid, fs[1], fs[2], fs[3]]
    _, res, _, _ = run(t, findings=fs)
    assert [r.claim.id for r in res] == [x.claim.id for x in fs]
    assert res[1] is mid


def test_max_checks():
    t = "".join(LONG.replace("12%", "%d%%" % n) + " https://h.example/%d\n" % n for n in range(1, 6))
    fs, res, f, _ = run(t, max_checks=2)
    assert len(f.calls) == 2
    assert [r.module for r in res] == ["support", "support", "citations", "citations", "citations"]
    assert all(res[i] is fs[i] for i in (2, 3, 4))
    fs, res, f, _ = run(t, max_checks=0)
    assert f.calls == [] and all(a is b for a, b in zip(fs, res))


def test_thin_assertions_skipped_without_any_call():
    for t in ("See https://x.org", "Source: https://x.org/very/long/path/to/an/article",
              "Read more: https://x.org/a/b/c", "https://x.org/only", "ดู https://x.org/a",
              "Hello there, the weather is nice today https://x.org/w", "ที่มา: https://x.org/a"):
        fs, res, f, j = run(t)
        assert f.calls == [] and j.calls == [], t
        assert res[0] is fs[0], t


def test_dedupe_identical_url_assertion():
    t = LONG + " https://acme.example/r\n"
    fs = citation_findings(t)
    dup = Finding(fs[0].claim, Status.VERIFIED, "citations", "again", ("x",))
    _, res, f, j = run(t, findings=[fs[0], dup])
    assert len(f.calls) == 1 and len(j.calls) == 1
    assert res[0].module == "support" and res[1].module == "support"


def test_two_urls_one_sentence_judged_against_same_assertion():
    t = LONG + " (see https://a.example/1 and https://b.example/2)."
    _, res, f, j = run(t)
    assert [c[0] for c in f.calls] == ["https://a.example/1", "https://b.example/2"]
    assert j.calls[0][0] == j.calls[1][0] and "12%" in j.calls[0][0]
    assert all(r.module == "support" for r in res)


def test_markdown_link_label_carries_assertion():
    t = "[Acme Corp reported revenue growth of 12% in 2023](https://acme.example/r)"
    _, res, f, j = run(t)
    assert j.calls[0][0] == "Acme Corp reported revenue growth of 12% in 2023"
    assert res[0].module == "support"


def test_no_text_or_empty_findings():
    assert refine("", [], llm=None, fetcher=None) == []
    fs = citation_findings(LONG + " https://acme.example/r")
    assert refine("", fs, llm=None, fetcher=None) == fs


# ---------------------------------------------------------------- assertion extraction
def A(text, which=0):
    cl = [c for c in extract.extract(text) if c.kind in (ClaimKind.URL, ClaimKind.DOI)]
    return assertion_for(text, cl[which], cl)


def test_assertion_table():
    S = "Acme Corp grew revenue by 12% in 2023"
    table = [
        # english, parenthetical source
        (S + " (source: https://x.org/r).", 0, S + "."),
        # english, "see" lead-in at the end of the sentence
        (S + ", see https://x.org/r.", 0, S + "."),
        # source on its own line -> preceding sentence fallback
        (S + ".\nSource: https://x.org/r\n", 0, S + "."),
        # <url> autolink
        (S + " <https://x.org/r>.", 0, S + "."),
        # markdown link whose label is the URL
        (S + " [https://x.org/r](https://x.org/r).", 0, S + "."),
        # markdown generic label dropped
        (S + " ([source](https://x.org/r))", 0, S),
        # label carries content, kept
        ("According to [the 2023 Acme report](https://x.org/r), revenue grew 3.5% overall.", 0,
         "According to the 2023 Acme report, revenue grew 3.5% overall."),
        # decimals do not split
        ("Pi is about 3.14159 and e is about 2.71828 per https://x.org/r in tests.", 0,
         "Pi is about 3.14159 and e is about 2.71828 per in tests."),
        # abbreviations do not split
        ("Dr. Smith et al. found 3.5 mg of caffeine, e.g. in tea, helps https://x.org/r", 0,
         "Dr. Smith et al. found 3.5 mg of caffeine, e.g. in tea, helps"),
        ("The U.S. Dept. of Energy said 40 GW were added https://x.org/r", 0,
         "The U.S. Dept. of Energy said 40 GW were added"),
        # initials
        ("J. K. Rowling published 7 Harry Potter books by 2007 https://x.org/r", 0,
         "J. K. Rowling published 7 Harry Potter books by 2007"),
        # CRLF
        ("Intro sentence here.\r\n" + S + " https://x.org/r\r\nNext line.", 0, S),
        # emoji
        ("\U0001F680 " + S + " https://x.org/r \U0001F389", 0, "\U0001F680 " + S + " \U0001F389"),
        # URL at start of text
        ("https://x.org/r says " + S + ".", 0, "says " + S + "."),
        # URL at the end without punctuation
        (S + " https://x.org/r", 0, S),
        # list numbering is not a boundary, and the bullet is removed
        ("1. " + S + " https://x.org/r\n2. Another item entirely.", 0, S),
        ("- " + S + " https://x.org/r", 0, S),
        # preceding-sentence fallback
        ("Overall revenue grew 12% in 2023 for Acme Corp. Details: https://x.org/r", 0,
         "Overall revenue grew 12% in 2023 for Acme Corp."),
        ("First unrelated sentence about nothing. Overall revenue grew 12% in 2023 for Acme Corp. Read more at https://x.org/r", 0,
         "Overall revenue grew 12% in 2023 for Acme Corp."),
        # thai: no punctuation, line is the unit, lead-in removed
        ("รายได้ของบริษัทเติบโต 12% ในปี 2566 ที่มา: https://x.org/r", 0,
         "รายได้ของบริษัทเติบโต 12% ในปี 2566"),
        ("บรรทัดแรกไม่เกี่ยวข้อง\nรายได้ของบริษัทเติบโต 12% ในปี 2566\nดู https://x.org/r\nบรรทัดถัดไป", 0,
         "รายได้ของบริษัทเติบโต 12% ในปี 2566"),
        ("อ่านเพิ่มเติมที่ https://x.org/r สถิติระบุว่าประชากรไทยมี 66 ล้านคนในปี 2566", 0,
         "สถิติระบุว่าประชากรไทยมี 66 ล้านคนในปี 2566"),
        # thai abbreviations with dots do not split
        ("นพ. สมชาย ใจดี แนะนำให้ดื่มน้ำ 8 แก้วต่อวัน https://x.org/r", 0,
         "นพ. สมชาย ใจดี แนะนำให้ดื่มน้ำ 8 แก้วต่อวัน"),
        # two URLs, one sentence: identical assertion
        (S + " (see https://x.org/1 and https://x.org/2).", 0, S + "."),
        (S + " (see https://x.org/1 and https://x.org/2).", 1, S + "."),
    ]
    for text, which, want in table:
        got = A(text, which)
        assert got == want, "\ntext: %r\n got: %r\nwant: %r" % (text, got, want)


def test_assertion_is_capped_at_300():
    long_text = "Acme grew by 12% " * 40 + "https://x.org/r " + "in 2023 " * 40
    got = A(long_text)
    assert 0 < len(got) <= 300 and "https" not in got
    thai = "รายได้เติบโต 12% ในปี 2566 " * 30 + "https://x.org/r"
    got = A(thai)
    assert 0 < len(got) <= 300


def test_checkable_rules():
    assert not is_checkable("See")
    assert not is_checkable("this is a long sentence without anything specific in it at all")
    assert not is_checkable("The weather was nice and everyone enjoyed the long walk")   # only the first word is capital
    assert is_checkable("the company grew 12% in the last year of the long period")      # digit
    assert is_checkable("the report was written by Alice Johnson for the board")         # capitalised words
    assert is_checkable("ประเทศไทยมีภูมิประเทศหลากหลายทั้งภูเขาและชายทะเลยาว")           # >= 20 thai chars
    assert not is_checkable("ไทยดี")


def test_sentence_split_does_not_cut_inside_url():
    t = "Acme Corp reported 12% growth. Details at https://x.org/a.b/c.html?q=1.5 and more words follow here for 2023."
    cl = [c for c in extract.extract(t) if c.kind == ClaimKind.URL]
    assert len(cl) == 1 and cl[0].text == "https://x.org/a.b/c.html?q=1.5"
    assert A(t) == "Details at and more words follow here for 2023."   # URL dots/? did not split the sentence


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("ok  ", name)
            except Exception as e:
                fails += 1; print("FAIL", name, repr(e))
    sys.exit(1 if fails else 0)
