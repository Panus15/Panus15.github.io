"""Tests for engine/citations.py. Offline: FakeFetcher only.

Run: python3 tests/test_citations.py
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.citations import check
from engine.models import Claim, ClaimKind, Status
from engine.citations import _Ctx, _link_dois_to_references
from engine.net import FakeFetcher, FetchResult

V, U, C = Status.VERIFIED, Status.UNVERIFIED, Status.CONTRADICTED

CROSSREF = "https://api.crossref.org/works/"
HANDLES = "https://doi.org/api/handles/"
OPENALEX = "https://api.openalex.org/works?search="

_n = [0]


def claim(kind, **data):
    _n[0] += 1
    text = "x%d" % _n[0]
    start = _n[0] * 1000            # disjoint spans, so no claim is "inside" another by accident
    return Claim("c%d" % _n[0], kind, text, start, start + len(text), data)


def url_claim(u):
    return claim(ClaimKind.URL, url=u)


def doi_claim(d):
    return claim(ClaimKind.DOI, doi=d)


def ref_claim(title, authors=(), year=None):
    return claim(ClaimKind.REFERENCE, title=title, authors=list(authors), year=year)


def R(url, status=200, body=b"", final=None, error=None):
    if isinstance(body, (dict, list)):
        body = json.dumps(body).encode()
    if isinstance(body, str):
        body = body.encode()
    return FetchResult(url, final or url, status, body, error=error)


def one(c, routes):
    f = FakeFetcher(routes)
    out = check([c], f)
    assert len(out) == 1
    return out[0], f


def page(title):
    return ("<html><head><title>%s</title></head><body>hi</body></html>" % title).encode()


# ---------------------------------------------------------------- URL

def test_url_ok_is_verified_with_caveat():
    u = "https://example.org/a/b"
    fd, _ = one(url_claim(u), {u: R(u, 200, page("A real page"))})
    assert fd.status == V and fd.module == "citations"
    assert "does not show" in fd.reason.lower() and u in fd.evidence


def test_url_redirect_to_other_page_still_verified():
    u = "http://example.org/old"
    fd, _ = one(url_claim(u), {u: R(u, 200, page("New"), final="https://example.org/new")})
    assert fd.status == V


def test_url_soft_404_root_redirect():
    u = "https://example.org/some/deep/page"
    fd, _ = one(url_claim(u), {u: R(u, 200, page("Welcome"), final="https://example.org/")})
    assert fd.status == U
    assert "home page" in fd.reason


def test_url_root_to_root_is_fine():
    u = "https://example.org/"
    fd, _ = one(url_claim(u), {u: R(u, 200, page("Home"))})
    assert fd.status == V
    u2 = "https://example.org"
    fd, _ = one(url_claim(u2), {u2: R(u2, 200, page("Home"), final="https://example.org/")})
    assert fd.status == V


def test_url_soft_404_titles():
    for t in ("404 Error", "Page Not Found", "Not found - Site", "ไม่พบหน้าที่คุณต้องการ", "ไม่พบ"):
        u = "https://example.org/p"
        fd, _ = one(url_claim(u), {u: R(u, 200, page(t))})
        assert fd.status == U, t


def test_url_title_with_attributes_and_entities():
    u = "https://example.org/p"
    body = b'<html><TITLE lang="en">\n  Page &amp; Not\n Found </TITLE>'
    fd, _ = one(url_claim(u), {u: R(u, 200, body)})
    # "Page & Not Found" after whitespace collapse contains "not found"
    assert fd.status == U


def test_url_title_only_read_in_first_64kb():
    u = "https://example.org/p"
    body = b"<html><head></head>" + b"x" * 70000 + b"<title>404</title>"
    fd, _ = one(url_claim(u), {u: R(u, 200, body)})
    assert fd.status == V


def test_url_non_utf8_body_does_not_crash():
    u = "https://example.org/p"
    fd, _ = one(url_claim(u), {u: R(u, 200, b"\xff\xfe<title>ok</title>\x80")})
    assert fd.status == V


def test_url_404_and_410_contradicted():
    for st in (404, 410):
        u = "https://example.org/gone"
        fd, _ = one(url_claim(u), {u: R(u, st)})
        assert fd.status == C and "does not exist" in fd.reason


def test_url_dns_contradicted():
    u = "https://no-such-domain-xyz.example/a"
    fd, _ = one(url_claim(u), {u: R(u, None, error="dns")})
    assert fd.status == C and "domain does not exist" in fd.reason


def test_url_bot_blocking_statuses_unverified():
    for st in (401, 403, 429, 500, 502, 503, 999, 418, 400):
        u = "https://example.org/x"
        fd, _ = one(url_claim(u), {u: R(u, st)})
        assert fd.status == U, st


def test_url_unresolved_3xx_unverified():
    for st in (301, 302, 307):
        u = "https://example.org/x"
        fd, _ = one(url_claim(u), {u: R(u, st)})
        assert fd.status == U, st


def test_url_our_failures_never_contradicted():
    for err in ("timeout", "ssl", "network", "other", "blocked"):
        u = "https://example.org/x"
        fd, _ = one(url_claim(u), {u: R(u, None, error=err)})
        assert fd.status == U, err
        assert "could not check" in fd.reason.lower(), err
    u = "https://example.org/x"
    fd, _ = one(url_claim(u), {u: R(u, None)})
    assert fd.status == U


def test_url_missing_data_unverified():
    fd, _ = one(claim(ClaimKind.URL), {})
    assert fd.status == U and "could not check" in fd.reason.lower()


# ---------------------------------------------------------------- DOI

def crossref_ok(doi, title="Deep Residual Learning", year=2016):
    return {"status": "ok", "message": {"title": [title], "issued": {"date-parts": [[year, 5]]}}}


def test_doi_crossref_hit():
    doi = "10.1109/cvpr.2016.90"
    u = CROSSREF + doi
    fd, _ = one(doi_claim(doi), {u: R(u, 200, crossref_ok(doi))})
    assert fd.status == V
    assert "Deep Residual Learning" in fd.reason and "2016" in fd.reason
    assert u in fd.evidence


def test_doi_is_url_quoted_but_slash_kept():
    doi = "10.1002/(sici)1097-4571(199806)49:8<693::aid-asi4>3.0.co;2-o"
    q = "10.1002/%28sici%291097-4571%28199806%2949%3A8%3C693%3A%3Aaid-asi4%3E3.0.co%3B2-o"
    u = CROSSREF + q
    fd, f = one(doi_claim(doi), {u: R(u, 200, crossref_ok(doi, "Quoted", 1998))})
    assert fd.status == V, f.calls
    assert f.calls == [u]


def test_doi_uppercase_normalised():
    u = CROSSREF + "10.1000/abc"
    fd, f = one(doi_claim("10.1000/ABC"), {u: R(u, 200, crossref_ok("x"))})
    assert fd.status == V and f.calls == [u]


def test_doi_crossref_200_without_year_or_title():
    doi = "10.1000/abc"
    u = CROSSREF + doi
    fd, _ = one(doi_claim(doi), {u: R(u, 200, {"message": {"title": []}})})
    assert fd.status == V


def test_doi_crossref_404_then_handle_100_contradicted():
    doi = "10.9999/fake.123"
    cu, hu = CROSSREF + doi, HANDLES + doi
    fd, f = one(doi_claim(doi), {cu: R(cu, 404, b"Resource not found."),
                                 hu: R(hu, 404, {"responseCode": 100})})
    assert fd.status == C and "not registered" in fd.reason
    assert f.calls == [cu, hu]


def test_doi_crossref_404_then_handle_1_verified_not_in_crossref():
    doi = "10.5281/zenodo.123456"
    cu, hu = CROSSREF + doi, HANDLES + doi
    fd, _ = one(doi_claim(doi), {cu: R(cu, 404), hu: R(hu, 200, {"responseCode": 1, "handle": doi})})
    assert fd.status == V and "not in crossref" in fd.reason.lower()


def test_doi_crossref_404_handle_other_or_unavailable_unverified():
    doi = "10.5281/zenodo.1"
    cu, hu = CROSSREF + doi, HANDLES + doi
    for h in (R(hu, 200, {"responseCode": 200}), R(hu, 200, b"not json"), R(hu, 500),
              R(hu, None, error="timeout"), R(hu, 200, {"responseCode": True}),
              R(hu, 404, b""), R(hu, None, error="dns")):
        fd, _ = one(doi_claim(doi), {cu: R(cu, 404), hu: h})
        assert fd.status == U, h


def test_doi_crossref_failures_unverified_and_no_handle_call():
    doi = "10.1000/abc"
    cu = CROSSREF + doi
    for r in (R(cu, 429), R(cu, 503), R(cu, None, error="timeout"), R(cu, None, error="dns"),
              R(cu, 200, b"{not json"), R(cu, 200, {"message": "oops"}), R(cu, 200, b"[1,2]")):
        fd, f = one(doi_claim(doi), {cu: r})
        assert fd.status == U, r
        assert f.calls == [cu]


def test_doi_malformed_unverified_without_fetch():
    fd, f = one(doi_claim("hello"), {})
    assert fd.status == U and f.calls == []


# ---------------------------------------------------------------- REFERENCE

def oa_url(title):
    from_q = title.replace(" ", "%20")
    return OPENALEX + from_q + "&per-page=5"


def oa_work(name, year, authors, **extra):
    w = {"id": "https://openalex.org/W1", "display_name": name, "publication_year": year,
         "authorships": [{"author": {"display_name": a}} for a in authors]}
    w.update(extra)
    return w


ATTN = "Attention Is All You Need"
ATTN_AUTHORS = ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar", "Jakob Uszkoreit",
                "Llion Jones", "Aidan N. Gomez", "Lukasz Kaiser", "Illia Polosukhin"]


def ref_one(title, authors, year, works, status=200):
    u = oa_url(title)
    c = ref_claim(title, authors, year)
    return one(c, {u: R(u, status, {"results": works})})


def test_reference_verified():
    fd, f = ref_one(ATTN, ["Vaswani, A.", "Shazeer, N."], 2017,
                    [oa_work(ATTN, 2017, ATTN_AUTHORS)])
    assert fd.status == V
    assert "exists" in fd.reason and "title/authors/year" in fd.reason


def test_reference_author_matched_beyond_first_author():
    # Claimed author is the LAST author of the record; must still count.
    fd, _ = ref_one(ATTN, ["Polosukhin"], 2017, [oa_work(ATTN, 2017, ATTN_AUTHORS)])
    assert fd.status == V


def test_reference_author_name_styles_and_accents():
    t = "Deep neural networks in practice"
    works = [oa_work(t, 2020, ["José Núñez", "Marie Curie-Sklodowska"])]
    for authors in (["Jose Nunez"], ["Núñez, J."], ["J. NUÑEZ"], ["Nunez JA"], ["Someone Else", "Curie-Sklodowska"]):
        fd, _ = ref_one(t, authors, 2020, works)
        assert fd.status == V, authors


def test_reference_title_hit_wrong_authors_contradicted():
    fd, _ = ref_one(ATTN, ["Smith, J.", "Brown, K."], 2017, [oa_work(ATTN, 2017, ATTN_AUTHORS)])
    assert fd.status == C
    assert "Vaswani" in fd.reason and "Smith" in fd.reason


def test_reference_title_match_tolerates_case_and_punctuation():
    fd, _ = ref_one("attention is all you need.", ["Vaswani"], 2017,
                    [oa_work(ATTN, 2017, ATTN_AUTHORS)])
    assert fd.status == V


def test_reference_year_off_by_one_ok_by_three_unverified():
    works = [oa_work(ATTN, 2017, ATTN_AUTHORS)]
    fd, _ = ref_one(ATTN, ["Vaswani"], 2018, works)
    assert fd.status == V
    fd, _ = ref_one(ATTN, ["Vaswani"], 2016, works)
    assert fd.status == V
    fd, _ = ref_one(ATTN, ["Vaswani"], 2020, works)
    assert fd.status == U and "2017" in fd.reason and "2020" in fd.reason


def test_reference_year_off_and_authors_conflict_contradicted():
    fd, _ = ref_one(ATTN, ["Smith"], 2012, [oa_work(ATTN, 2017, ATTN_AUTHORS)])
    assert fd.status == C
    assert "2012" in fd.reason and "2017" in fd.reason


def test_reference_no_authors_given_year_only():
    works = [oa_work(ATTN, 2017, ATTN_AUTHORS)]
    fd, _ = ref_one(ATTN, [], 2017, works)
    assert fd.status == V
    fd, _ = ref_one(ATTN, [], 2010, works)
    assert fd.status == U
    fd, _ = ref_one(ATTN, [], None, works)
    assert fd.status == V


def test_reference_et_al_only_is_not_an_author_conflict():
    fd, _ = ref_one(ATTN, ["et al."], 2017, [oa_work(ATTN, 2017, ATTN_AUTHORS)])
    assert fd.status == V


def test_reference_record_without_authors_or_truncated_is_not_a_conflict():
    fd, _ = ref_one(ATTN, ["Smith"], 2017, [oa_work(ATTN, 2017, [])])
    assert fd.status == V
    fd, _ = ref_one(ATTN, ["Smith"], 2017, [oa_work(ATTN, 2017, ["A B"], is_authors_truncated=True)])
    assert fd.status == V


def test_reference_second_candidate_with_right_authors_wins():
    works = [oa_work(ATTN, 2017, ["Someone Else"]), oa_work(ATTN, 2017, ATTN_AUTHORS)]
    fd, _ = ref_one(ATTN, ["Vaswani"], 2017, works)
    assert fd.status == V


def test_reference_no_match_unverified_never_contradicted():
    fd, _ = ref_one("A Completely Invented Paper Title Here", ["Smith"], 2020,
                    [oa_work("Something unrelated about geology", 2020, ["Smith"])])
    assert fd.status == U
    assert "not found in openalex" in fd.reason.lower() and "search" in fd.reason.lower()
    fd, _ = ref_one("A Completely Invented Paper Title Here", ["Smith"], 2020, [])
    assert fd.status == U


def test_reference_similar_but_not_close_title_is_no_match():
    fd, _ = ref_one("Attention Is All You Need", ["Vaswani"], 2017,
                    [oa_work("Attention Is Not All You Need: Pure Attention Loses Rank", 2021, ["Dong"])])
    assert fd.status == U


def test_reference_short_or_empty_title():
    for t in ("", "   ", "Short title", None):
        fd, f = one(ref_claim(t, ["Smith"], 2020), {})
        assert fd.status == U and "too short" in fd.reason, t
        assert f.calls == []


def test_reference_bad_responses_unverified():
    for status, body in ((200, b"not json"), (200, {"results": "x"}), (200, {"nope": 1}),
                         (200, [1]), (500, b""), (429, b""), (404, {"results": []})):
        u = oa_url(ATTN)
        fd, _ = one(ref_claim(ATTN, ["Smith"], 2017), {u: R(u, status, body)})
        assert fd.status == U, (status, body)
    u = oa_url(ATTN)
    for err in ("timeout", "dns", "network", "ssl", "blocked"):
        fd, _ = one(ref_claim(ATTN, ["Smith"], 2017), {u: R(u, None, error=err)})
        assert fd.status == U, err


def test_reference_odd_result_entries_do_not_crash():
    works = ["junk", None, {"display_name": None}, {"display_name": ATTN, "authorships": "bad",
                                                    "publication_year": "2017"}]
    fd, _ = ref_one(ATTN, ["Vaswani"], 2017, works)
    assert fd.status == V  # no usable author list -> not a conflict


def test_reference_query_url_encoding():
    title = "Café: a study/of #things & more"
    expected = OPENALEX + "Caf%C3%A9%3A%20a%20study%2Fof%20%23things%20%26%20more&per-page=5"
    f = FakeFetcher({expected: R(expected, 200, {"results": []})})
    out = check([ref_claim(title, [], None)], f)
    assert f.calls == [expected], f.calls
    assert out[0].status == U


# ---------------------------------------------------------------- cross-cutting

def test_dedupe_identical_urls_fetched_once():
    u = "https://example.org/a"
    f = FakeFetcher({u: R(u, 200, page("ok"))})
    out = check([url_claim(u), url_claim(u)], f)
    assert [x.status for x in out] == [V, V]
    assert f.calls == [u]
    assert out[0].claim is not out[1].claim


def test_dedupe_dois_and_references():
    doi = "10.1000/abc"
    cu = CROSSREF + doi
    ru = oa_url(ATTN)
    f = FakeFetcher({cu: R(cu, 200, crossref_ok(doi)),
                     ru: R(ru, 200, {"results": [oa_work(ATTN, 2017, ATTN_AUTHORS)]})})
    cl = [doi_claim(doi), doi_claim("10.1000/ABC"), ref_claim(ATTN, ["Vaswani"], 2017),
          ref_claim(ATTN, ["Smith"], 2017)]
    out = check(cl, f)
    assert [x.status for x in out] == [V, V, V, C]   # same fetch, per-claim judgement
    assert sorted(f.calls) == sorted([cu, ru])


def test_max_lookups():
    urls = ["https://example.org/%d" % i for i in range(4)]
    f = FakeFetcher({u: R(u, 200, page("ok")) for u in urls})
    out = check([url_claim(u) for u in urls], f, max_lookups=2)
    assert [x.status for x in out] == [V, V, U, U]
    assert "lookup limit reached" in out[2].reason and "lookup limit reached" in out[3].reason
    assert len(f.calls) == 2
    # cached repeats don't consume budget
    out = check([url_claim(urls[0]), url_claim(urls[0]), url_claim(urls[1])],
                FakeFetcher({u: R(u, 200, page("ok")) for u in urls}), max_lookups=1)
    assert [x.status for x in out] == [V, V, U]


def test_max_lookups_zero_and_doi_second_lookup():
    u = "https://example.org/a"
    f = FakeFetcher({u: R(u, 200, page("ok"))})
    assert check([url_claim(u)], f, max_lookups=0)[0].status == U and f.calls == []
    doi = "10.9999/x"
    cu, hu = CROSSREF + doi, HANDLES + doi
    f = FakeFetcher({cu: R(cu, 404), hu: R(hu, 404, {"responseCode": 100})})
    fd = check([doi_claim(doi)], f, max_lookups=1)[0]
    assert fd.status == U and "lookup limit reached" in fd.reason   # not CONTRADICTED


def test_fetcher_raising_never_propagates():
    class Boom:
        def fetch(self, url, **kw):
            raise RuntimeError("kaboom")
    cl = [url_claim("https://example.org/a"), doi_claim("10.1000/abc"), ref_claim(ATTN, ["Smith"], 2017)]
    out = check(cl, Boom())
    assert [x.status for x in out] == [U, U, U]
    assert all("could not check" in x.reason.lower() for x in out)


def test_fetcher_returning_junk_never_propagates():
    class Junk:
        def fetch(self, url, **kw):
            return None
    out = check([url_claim("https://example.org/a")], Junk())
    assert out[0].status == U and "could not check" in out[0].reason.lower()


def test_malformed_claim_data_never_propagates():
    bad = [claim(ClaimKind.URL, url=12345), claim(ClaimKind.DOI, doi=None),
           claim(ClaimKind.REFERENCE, title=["x"], authors="Smith", year="soon"),
           claim(ClaimKind.REFERENCE, title=ATTN, authors=[1, None, {}], year=[2017])]
    ru = oa_url(ATTN)
    f = FakeFetcher({ru: R(ru, 200, {"results": [oa_work(ATTN, 2017, ATTN_AUTHORS)]})})
    out = check(bad, f)
    assert len(out) == 4
    assert all(x.status in (U, V) for x in out)
    assert out[0].status == U and out[1].status == U and out[2].status == U


def test_input_order_preserved_and_other_kinds_ignored():
    u1, u2 = "https://example.org/1", "https://example.org/2"
    doi = "10.1000/abc"
    cu = CROSSREF + doi
    f = FakeFetcher({u1: R(u1, 404), u2: R(u2, 200, page("ok")), cu: R(cu, 200, crossref_ok(doi))})
    cl = [claim(ClaimKind.QUOTE, quote="q", attribution=None),
          url_claim(u1),
          claim(ClaimKind.VAGUE, phrase="studies show"),
          doi_claim(doi),
          claim(ClaimKind.ARITHMETIC, expression="1+1", claimed="2"),
          url_claim(u2),
          claim(ClaimKind.CODE, language="py", code="x=1")]
    out = check(cl, f)
    assert [x.claim.id for x in out] == [cl[1].id, cl[3].id, cl[5].id]
    assert [x.status for x in out] == [C, V, V]
    assert all(x.module == "citations" and x.reason for x in out)


def test_empty_and_irrelevant_input():
    assert check([], FakeFetcher({})) == []
    f = FakeFetcher({})
    assert check([claim(ClaimKind.VAGUE, phrase="x")], f) == [] and f.calls == []


def test_timeout_is_forwarded_to_fetcher():
    seen = []

    class Spy:
        def fetch(self, url, *, headers=None, timeout=10.0, max_bytes=0):
            seen.append(timeout)
            return R(url, 200, page("ok"))
    check([url_claim("https://example.org/a")], Spy(), timeout=3.5)
    assert seen == [3.5]


def test_openalex_key_is_sent_but_never_reported():
    import engine.citations as cit
    KEY = "SECRETKEY-abc123"
    title = ATTN
    plain = oa_url(title)
    keyed = plain + "&api_key=" + KEY
    c = ref_claim(title, ["Vaswani"], 2017)
    fd = FakeFetcher({keyed: R(keyed, 200, {"results": [oa_work(ATTN, 2017, ATTN_AUTHORS)]})})
    out = cit.check([c], fd, openalex_key=KEY)
    assert out[0].status == V
    assert fd.calls == [keyed]                                     # key was sent on the wire...
    blob = repr(out)
    assert KEY not in blob and "api_key" not in blob              # ...and is nowhere in the findings


def test_openalex_key_masked_if_a_fetcher_error_echoes_it():
    import engine.citations as cit
    KEY = "SECRETKEY-abc123"
    class Boom:
        def fetch(self, url, **k): raise RuntimeError("failed for " + url)
    out = cit.check([ref_claim(ATTN, ["Vaswani"], 2017)], Boom(), openalex_key=KEY)
    assert out[0].status == U and KEY not in repr(out)


def test_openalex_key_read_from_env_and_no_key_means_no_param():
    import os, engine.citations as cit
    title = ATTN
    plain = oa_url(title)
    old = os.environ.pop("OPENALEX_API_KEY", None)
    try:
        fd = FakeFetcher({plain: R(plain, 200, {"results": [oa_work(ATTN, 2017, ATTN_AUTHORS)]})})
        assert cit.check([ref_claim(title, [], 2017)], fd)[0].status == V and fd.calls == [plain]
        os.environ["OPENALEX_API_KEY"] = "ENVKEY1"
        fd = FakeFetcher({plain + "&api_key=ENVKEY1": R(plain, 200, {"results": [oa_work(ATTN, 2017, ATTN_AUTHORS)]})})
        assert cit.check([ref_claim(title, [], 2017)], fd)[0].status == V
    finally:
        os.environ.pop("OPENALEX_API_KEY", None)
        if old is not None: os.environ["OPENALEX_API_KEY"] = old


def test_openalex_429_explains_the_shared_budget():
    fd, _ = ref_one(ATTN, ["Vaswani"], 2017, [], status=429)
    assert fd.status == U and "OPENALEX_API_KEY" in fd.reason


# ------------------------------------------------- DOI inside a reference must be the same work

def ref_with_doi(title, authors, year, doi):
    """A REFERENCE claim whose span contains a DOI claim, as extract produces for one reference line."""
    base = claim(ClaimKind.REFERENCE, title=title, authors=list(authors), year=year)
    ref = Claim(base.id, base.kind, base.text, base.start, base.start + 200, base.data)
    d = Claim("d" + ref.id, ClaimKind.DOI, "doi:" + doi, ref.start + 100, ref.start + 104 + len(doi), {"doi": doi})
    return ref, d


def cr_record(title, year, families):
    return {"message": {"title": [title], "issued": {"date-parts": [[year]]},
                        "author": [{"family": f, "given": "X"} for f in families]}}


def test_real_doi_borrowed_by_a_fabricated_reference_is_contradicted():
    doi = "10.1038/nature14539"
    ref, d = ref_with_doi("Quantum gravity effects in green tea fermentation", ["Smith"], 2020, doi)
    cu = CROSSREF + doi
    f = FakeFetcher({cu: R(cu, 200, cr_record("Deep learning", 2015, ["LeCun", "Bengio", "Hinton"]))})
    out = check([ref, d], f)
    assert out[0].status == C and "different work" in out[0].reason and "Deep learning" in out[0].reason
    assert out[1].status == V                      # the DOI itself does exist
    assert f.calls == [cu]                         # one lookup serves both, no OpenAlex needed


def test_doi_matching_reference_is_verified_without_openalex():
    doi = "10.1038/nature14539"
    ref, d = ref_with_doi("Deep learning", ["LeCun", "Hinton"], 2015, doi)
    cu = CROSSREF + doi
    f = FakeFetcher({cu: R(cu, 200, cr_record("Deep learning", 2015, ["LeCun", "Bengio", "Hinton"]))})
    out = check([ref, d], f)
    assert [x.status for x in out] == [V, V] and f.calls == [cu]


def test_right_title_and_doi_but_wrong_authors_is_contradicted():
    doi = "10.1038/nature14539"
    ref, d = ref_with_doi("Deep learning", ["Smith", "Chen"], 2015, doi)
    cu = CROSSREF + doi
    f = FakeFetcher({cu: R(cu, 200, cr_record("Deep learning", 2015, ["LeCun", "Bengio", "Hinton"]))})
    assert check([ref, d], f)[0].status == C


def test_doi_cross_check_year_off_is_unverified_and_subtitle_tolerated():
    doi = "10.1038/x"
    ref, d = ref_with_doi("Deep learning: a review", [], 2019, doi)
    cu = CROSSREF + doi
    f = FakeFetcher({cu: R(cu, 200, cr_record("Deep learning", 2015, ["LeCun"]))})
    out = check([ref, d], f)
    assert out[0].status == U and "2015" in out[0].reason and "2019" in out[0].reason


def test_cross_check_falls_through_when_unsure_or_unavailable():
    doi = "10.1038/x"
    # (a) partly similar titles -> neither same nor different: fall through to OpenAlex
    ref, d = ref_with_doi("Deep learning methods for protein structure prediction", [], None, doi)
    cu, ou = CROSSREF + doi, oa_url("Deep learning methods for protein structure prediction")
    f = FakeFetcher({cu: R(cu, 200, cr_record("Deep learning for protein structure", 2015, ["A"])),
                     ou: R(ou, 200, {"results": []})})
    out = check([ref, d], f)
    assert ou in f.calls and out[0].status == U
    # (b) Crossref has no record (e.g. a DataCite DOI) -> fall through, never contradicted
    ref, d = ref_with_doi("Attention Is All You Need", ["Vaswani"], 2017, "10.48550/arxiv.1706.03762")
    cu = CROSSREF + "10.48550/arxiv.1706.03762"; ru = oa_url(ATTN)
    f = FakeFetcher({cu: R(cu, 404, b""), HANDLES + "10.48550/arxiv.1706.03762": R("", 200, {"responseCode": 1}),
                     ru: R(ru, 200, {"results": [oa_work(ATTN, 2017, ATTN_AUTHORS)]})})
    out = check([ref, d], f)
    assert out[0].status == V
    # (c) Crossref lookup fails -> our failure, not a contradiction
    ref, d = ref_with_doi("Quantum gravity effects in green tea fermentation", [], None, "10.1038/y")
    f = FakeFetcher({})
    assert check([ref, d], f)[0].status == U


def test_reference_without_a_doi_in_it_is_unaffected():
    ref = ref_claim(ATTN, ["Vaswani"], 2017)
    other_doi = claim(ClaimKind.DOI, doi="10.1038/nature14539")     # elsewhere in the answer
    ru = oa_url(ATTN)
    f = FakeFetcher({ru: R(ru, 200, {"results": [oa_work(ATTN, 2017, ATTN_AUTHORS)]}),
                     CROSSREF + "10.1038/nature14539": R("", 200, crossref_ok("10.1038/nature14539"))})
    out = check([ref, other_doi], f)
    assert out[0].status == V and (CROSSREF + "10.1038/nature14539") in f.calls


def test_doi_on_the_same_line_but_outside_the_reference_span_is_linked():
    # extract ends a REFERENCE span after the title, so the DOI is usually just OUTSIDE it
    text = "Smith, J. (2020). *Quantum gravity effects in green tea fermentation*. Imaginary Sci. doi:10.1038/nature14539\n"
    doi = "10.1038/nature14539"
    ref = Claim("r1", ClaimKind.REFERENCE, text[:80], 0, 80,
                {"title": "Quantum gravity effects in green tea fermentation", "authors": ["Smith"], "year": 2020})
    at = text.index("doi:")
    d = Claim("d1", ClaimKind.DOI, "doi:" + doi, at, at + 4 + len(doi), {"doi": doi})
    cu = CROSSREF + doi
    f = FakeFetcher({cu: R(cu, 200, cr_record("Deep learning", 2015, ["LeCun"]))})
    assert check([ref, d], f, text=text)[0].status == C
    # without the text it cannot know they belong together and falls back to the title search
    f2 = FakeFetcher({cu: R(cu, 200, cr_record("Deep learning", 2015, ["LeCun"]))})
    assert check([ref, d], f2)[0].status == U


def test_doi_on_another_line_is_not_linked():
    t1 = "Smith, J. (2020). *Quantum gravity effects in green tea fermentation*.\ndoi:10.1038/nature14539\n"
    ref = Claim("r1", ClaimKind.REFERENCE, "x", 0, 70,
                {"title": "Quantum gravity effects in green tea fermentation", "authors": [], "year": 2020})
    at = t1.index("doi:")
    d = Claim("d1", ClaimKind.DOI, "doi", at, at + 23, {"doi": "10.1038/nature14539"})
    ctx = _Ctx(None, 1.0, 5)
    _link_dois_to_references([ref, d], ctx, t1)
    assert ctx.doi_for_ref == {}


def test_doi_goes_to_the_nearer_of_two_references_on_one_line():
    t2 = "A (2020). *First title that is long enough*. B (2021). *Second title that is long enough*. doi:10.1/x\n"
    s2 = t2.index("B (2021)")
    at = t2.index("doi:")
    r1 = Claim("r1", ClaimKind.REFERENCE, "x", 0, s2 - 1, {"title": "First title that is long enough", "authors": [], "year": 2020})
    r2 = Claim("r2", ClaimKind.REFERENCE, "x", s2, at - 1, {"title": "Second title that is long enough", "authors": [], "year": 2021})
    dd = Claim("d1", ClaimKind.DOI, "doi", at, at + 11, {"doi": "10.1/x"})
    ctx = _Ctx(None, 1.0, 5)
    _link_dois_to_references([r1, r2, dd], ctx, t2)
    assert ctx.doi_for_ref == {"r2": "10.1/x"}


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("ok  ", name)
            except Exception as e:
                fails += 1; print("FAIL", name, repr(e))
    sys.exit(1 if fails else 0)
