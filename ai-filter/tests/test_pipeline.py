"""End-to-end: a planted-defect answer through pipeline + CLI. Offline.

Each defect is one the filter exists to catch; each control is a true statement
that must NOT be flagged. If a defect stops being caught, or a control starts
being flagged, this file fails and names which.

Run: python3 tests/test_pipeline.py
"""
import io, json, os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cli
from engine import extract, pipeline, recompute
from engine.models import ClaimKind, Status
from engine.net import FakeFetcher, FetchResult, OfflineFetcher
from engine.score import Verdict, verdict

V, U, C = Status.VERIFIED, Status.UNVERIFIED, Status.CONTRADICTED


def ok(url, body=b"<html><title>Real page</title></html>"):
    return FetchResult(url, url, 200, body)


GOOD_URL = "https://www.nature.com/articles/real-paper"
DEAD_URL = "https://www.example.org/reports/2023/made-up-study"
GOOD_DOI = "10.1038/real.2020.1"
FAKE_DOI = "10.9999/totally.fake.2021"

ANSWER = f"""
อัตราดอกเบี้ย 1,200 × 5% = 60 บาทต่อปี และ 2 + 2 = 5 ตามที่ผมคำนวณ

อ่านเพิ่มเติมที่ {GOOD_URL} และ {DEAD_URL}

บทความ doi:{GOOD_DOI} และ doi:{FAKE_DOI} ยืนยันเรื่องนี้

ส่วนวิธีนี้ มีการศึกษาพบว่าได้ผลดีมาก

```python
def f(:
    return 1
```
"""


def routes():
    cr = "https://api.crossref.org/works/"
    hd = "https://doi.org/api/handles/"
    return {
        GOOD_URL: ok(GOOD_URL),
        DEAD_URL: FetchResult(DEAD_URL, DEAD_URL, 404, b""),
        cr + GOOD_DOI: FetchResult("", "", 200, json.dumps({"message": {"title": ["Real"], "issued": {"date-parts": [[2020]]}}}).encode()),
        cr + FAKE_DOI: FetchResult("", "", 404, b""),
        hd + FAKE_DOI: FetchResult("", "", 404, json.dumps({"responseCode": 100}).encode()),
    }


def by_text(findings, needle):
    hits = [f for f in findings if needle in f.claim.text]
    assert len(hits) == 1, (needle, [f.claim.text for f in findings])
    return hits[0]


def test_planted_defects_are_caught_and_controls_pass():
    fs = pipeline.run(ANSWER, FakeFetcher(routes()))
    assert by_text(fs, "1,200 × 5% = 60").status is V          # control: true arithmetic
    assert by_text(fs, "2 + 2 = 5").status is C                # defect: wrong arithmetic
    assert by_text(fs, GOOD_URL).status is V                   # control: live link
    assert by_text(fs, DEAD_URL).status is C                   # defect: dead link
    assert by_text(fs, GOOD_DOI).status is V                   # control: real DOI
    assert by_text(fs, FAKE_DOI).status is C                   # defect: fabricated DOI
    assert by_text(fs, "def f(").status is C                   # defect: code that does not parse
    vag = [f for f in fs if f.module == "vague"]
    assert len(vag) == 1 and vag[0].status is U                # defect: authority with no source
    assert verdict(fs) is Verdict.DO_NOT_USE


def test_every_extracted_claim_gets_exactly_one_finding():
    fs = pipeline.run(ANSWER, FakeFetcher(routes()))
    claims = extract.extract(ANSWER)
    ids = [f.claim.id for f in fs if f.module != "vague"]
    assert sorted(ids) == sorted(c.id for c in claims)
    assert len(ids) == len(set(ids))


def test_findings_are_in_document_order():
    fs = pipeline.run(ANSWER, FakeFetcher(routes()))
    starts = [f.claim.start for f in fs]
    assert starts == sorted(starts)


def test_plain_prose_is_not_a_pass():
    fs = pipeline.run("The sky appears blue because of Rayleigh scattering.", FakeFetcher({}))
    assert fs == [] and verdict(fs) is Verdict.NO_CHECKABLE_CLAIMS


def test_offline_never_produces_contradicted_links():
    fs = pipeline.run(f"See {GOOD_URL} and {DEAD_URL} and doi:{FAKE_DOI}", OfflineFetcher())
    assert fs and all(f.status is U for f in fs), [(f.claim.text, f.status) for f in fs]


def test_a_crashing_checker_degrades_to_unverified_not_a_crash():
    real = recompute.check
    recompute.check = lambda cs: 1 / 0
    try:
        fs = pipeline.run("2 + 2 = 5", FakeFetcher({}))
    finally:
        recompute.check = real
    assert len(fs) == 1 and fs[0].status is U and "could not check" in fs[0].reason


def test_unhandled_claim_kind_is_not_silently_dropped():
    # a claim kind with no checker must surface as UNVERIFIED, never vanish
    real = extract.extract
    from engine.models import Claim
    extract.extract = lambda t: [Claim("c1", ClaimKind.QUOTE, "q", 0, 1, {"quote": "q", "attribution": None})]
    try:
        fs = pipeline.run("q", FakeFetcher({}))
    finally:
        extract.extract = real
    assert [f.status for f in fs if f.module == "pipeline"] == [U]


def _cli(argv, stdin_text=None):
    out = io.StringIO()
    stdin = io.StringIO(stdin_text) if stdin_text is not None else None
    return cli.main(argv, stdin=stdin, stdout=out), out.getvalue()


def test_cli_exit_codes_and_outputs():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "a.md")
        open(p, "w", encoding="utf-8").write("2 + 2 = 5")
        code, out = _cli(["check", p, "--offline"])
        assert code == 2 and "ห้ามใช้" in out
        code, out = _cli(["check", p, "--offline", "--json"])
        assert code == 2 and json.loads(out)["verdict"] == "do_not_use"
        open(p, "w", encoding="utf-8").write("plain words only")
        assert _cli(["check", p, "--offline"])[0] == 0
        open(p, "w", encoding="utf-8").write("see https://example.com/x")
        assert _cli(["check", p, "--offline"])[0] == 1            # unverifiable -> review
        rp = os.path.join(d, "r.md")
        assert _cli(["check", p, "--offline", "-o", rp])[0] == 1 and os.path.getsize(rp) > 0
    assert _cli(["check", "/nonexistent/x.md"])[0] == 3
    assert _cli([])[0] == 3
    with tempfile.TemporaryDirectory() as d:
        bad = os.path.join(d, "b.bin"); open(bad, "wb").write(b"\xff\xfe\x00bad")
        assert _cli(["check", bad])[0] == 3


def test_cli_never_runs_code_unless_asked():
    with tempfile.TemporaryDirectory() as d:
        marker = os.path.join(d, "ran.txt")
        p = os.path.join(d, "a.md")
        open(p, "w").write(f"```python\nopen({marker!r}, 'w').write('x')\n```\n")
        code, out = _cli(["check", p, "--offline"])
        assert not os.path.exists(marker), "code executed without --run-code"
        assert code == 1 and "not executed" in out
        _cli(["check", p, "--offline", "--run-code"])
        # with --run-code it runs in an isolated temp cwd; absolute path write may succeed -- that is
        # exactly why the flag is opt-in and documented as not a security boundary.


def test_a_fake_citation_does_not_back_a_nearby_unsourced_claim():
    from engine.net import FakeFetcher, FetchResult
    import json
    text = "มีการศึกษาพบว่าได้ผลดี doi:10.9999/invented.2021 ยืนยัน\n"
    fake = FakeFetcher({
        "https://api.crossref.org/works/10.9999/invented.2021": FetchResult("", "", 404, b""),
        "https://doi.org/api/handles/10.9999/invented.2021": FetchResult("", "", 404, json.dumps({"responseCode": 100}).encode()),
    })
    fs = pipeline.run(text, fake)
    assert any(f.module == "citations" and f.status is C for f in fs)
    assert any(f.module == "vague" and f.status is U for f in fs), [(f.module, f.status) for f in fs]
    # control: the SAME sentence with a REAL doi is backed, so no vague finding
    ok = FakeFetcher({"https://api.crossref.org/works/10.1038/nature14539": FetchResult("", "", 200, json.dumps({"message": {"title": ["Deep learning"], "issued": {"date-parts": [[2015]]}}}).encode())})
    fs2 = pipeline.run("มีการศึกษาพบว่าได้ผลดี doi:10.1038/nature14539 ยืนยัน\n", ok)
    assert not any(f.module == "vague" for f in fs2)


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("ok  ", name)
            except Exception as e:
                fails += 1; print("FAIL", name, repr(e))
    sys.exit(1 if fails else 0)
