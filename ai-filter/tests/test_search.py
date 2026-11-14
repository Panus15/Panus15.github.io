"""Tests for engine/search.py (host_matches, BraveSearch, FakeSearch) and engine/sources.py. Offline.

Run: python3 tests/test_search.py
"""
import json, os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.net import FakeFetcher, FetchResult
from engine.search import BraveSearch, FakeSearch, SearchError, SearchResult, host_matches
from engine.sources import KINDS, TIERS, TOPICS, load_registry

KEY = "SECRETKEY123"


# ---------------------------------------------------------------- host_matches

def test_host_matches_accepts():
    D = ["who.int", "gov.uk"]
    for u in ("https://who.int/", "http://who.int", "https://www.who.int/a/b?c=d#e", "https://WHO.INT/",
              "https://Www.Who.Int/x", "https://who.int:443/x", "https://who.int:8080/", "https://who.int./",
              "https://a.b.who.int/", "https://legislation.gov.uk/x", "HTTPS://who.int/", "https://who.int?x=1",
              "https://who.int#frag"):
        assert host_matches(u, D), u
    assert host_matches("https://www.who.int/", ["WHO.INT."])      # domain list is normalised too
    assert host_matches("https://xn--80ak6aa92e.com/", ["xn--80ak6aa92e.com"])   # punycode compared literally


def test_host_matches_attack_table():
    D = ["who.int"]
    attacks = [
        "https://who.int.evil.com/", "https://evil.com/who.int", "https://evil.com?x=who.int",
        "https://who.int@evil.com/", "https://who.int:pw@evil.com/", "https://evil.com#@who.int",
        "https://evil.com/#@who.int", "https://notwho.int/", "https://awho.int/", "https://who.intx/",
        "https://evil.com\\@who.int", "https://evil.com\\.who.int", "https:\\\\who.int", "https://who.int\\@evil.com",
        "https://who.int /", " https://who.int/", "https://who.int/ ", "https://who.int\t.evil.com/",
        "https://who.int\n.evil.com/", "https://who.int\r\n/", "https://who.int\x00.evil.com/", "https://who.int\x7f/",
        "https://who.int​.evil.com/", "https://who.int‮/", "https://who .int/",
        "ftp://who.int/", "file://who.int/", "javascript://who.int/%0aalert(1)", "data:text/html,who.int",
        "//who.int/", "who.int", "https:who.int", "https:/who.int", "https:///who.int", "https://", "https://:443/",
        "https://.who.int/", "https://who..int/", "https://who.int../", "https://-who.int/", "https://who_.int/",
        "https://[::1]/", "https://[::ffff:who.int]/", "https://127.0.0.1/", "https://1.2.3.4/who.int",
        "https://who.int:abc/", "https://who.int:99999/", "https://who.int:/x:y", "https://who.int:80:80/",
        "https://who%2eint.evil.com/", "https://who.int%2f.evil.com/", "https://evil.com%2f@who.int/",
        "https://ｗｈｏ.int/",            # full-width "who" (parser differential risk): refused
        "https://who.int．evil.com/", "https://who。int/", "https://K.who.int/",
        "https://ßa.who.int/", "https://who.int/\x01", "http://who.int /", "https://evil.com/\\who.int",
        "", None, 5, b"https://who.int/",
    ]
    for u in attacks:
        assert not host_matches(u, D), repr(u)


def test_host_matches_domain_list_edge_cases():
    assert not host_matches("https://who.int/", [])
    assert not host_matches("https://who.int/", [""])
    assert not host_matches("https://who.int/", ["."])
    assert not host_matches("https://who.int/", ["int"])           # a bare TLD never counts
    assert not host_matches("https://evil.com/", ["com"])
    assert not host_matches("https://x.evil.com/", ["*.evil.com"])
    assert not host_matches("https://who.int/", ["https://who.int"])
    assert not host_matches("https://who.int/", ["who.int/path"])
    assert not host_matches("https://who.int/", ["1.2.3.4"])
    assert not host_matches("https://1.2.3.4/", ["1.2.3.4"])
    assert not host_matches("https://who.int/", [None, 5])         # junk entries are ignored, not crashes
    assert host_matches("https://who.int/", "who.int")             # bare string accepted as ONE domain
    assert not host_matches("https://i/", "who.int")               # ...not iterated per character


# ---------------------------------------------------------------- BraveSearch

class RecordingFetcher(FakeFetcher):
    def __init__(self, routes_or_result):
        super().__init__({})
        self.result = routes_or_result
        self.headers_seen = []
        self.urls = []

    def fetch(self, url, *, headers=None, timeout=10.0, max_bytes=0):
        self.urls.append(url)
        self.headers_seen.append(dict(headers or {}))
        r = self.result
        return r(url) if callable(r) else r


def _ok(items):
    return FetchResult("u", "u", 200, json.dumps({"web": {"results": items}}).encode())


def _item(title, url, desc=""):
    return {"title": title, "url": url, "description": desc}


def test_brave_success_and_request_shape():
    f = RecordingFetcher(_ok([_item("T", "https://www.who.int/a", "S")]))
    r = BraveSearch(KEY, fetcher=f).search("covid cases", count=3)
    assert r == [SearchResult("T", "https://www.who.int/a", "S")]
    assert f.urls[0].startswith("https://api.search.brave.com/res/v1/web/search?")
    assert "q=covid+cases" in f.urls[0] and "count=3" in f.urls[0]
    assert f.headers_seen[0] == {"Accept": "application/json", "X-Subscription-Token": KEY}
    assert KEY not in f.urls[0]


def test_brave_strips_html_and_unescapes():
    f = RecordingFetcher(_ok([_item("<strong>Hello</strong> &amp; bye", "https://who.int/",
                                    "a <b>bold</b> claim&#39;s &quot;text&quot;\n  here")]))
    r = BraveSearch(KEY, fetcher=f).search("q")
    assert r[0].title == "Hello & bye"
    assert r[0].snippet == 'a bold claim\'s "text" here'


def test_brave_domain_filter_drops_offlist_even_if_returned():
    items = [_item("evil", "https://who.int.evil.com/"), _item("evil2", "https://evil.com/who.int"),
             _item("ok", "https://www.who.int/x"), _item("evil3", "https://who.int@evil.com/"),
             _item("js", "javascript:alert(1)"), _item("ok2", "https://who.int/y")]
    f = RecordingFetcher(_ok(items))
    r = BraveSearch(KEY, fetcher=f).search("q", count=5, domains=["who.int"])
    assert [x.title for x in r] == ["ok", "ok2"]


def test_brave_site_filter_and_request_count():
    f = RecordingFetcher(_ok([]))
    BraveSearch(KEY, fetcher=f).search("rates", count=5, domains=["bot.or.th", "mof.go.th"])
    from urllib.parse import urlsplit, parse_qs
    qs = parse_qs(urlsplit(f.urls[0]).query)
    assert qs["q"] == ["(site:bot.or.th OR site:mof.go.th) rates"]
    assert qs["count"] == ["15"]
    f.urls.clear()
    BraveSearch(KEY, fetcher=f).search("rates", count=10, domains=["bot.or.th"])
    assert parse_qs(urlsplit(f.urls[0]).query)["count"] == ["20"]          # capped at 20
    f.urls.clear()
    BraveSearch(KEY, fetcher=f).search("rates", count=2, domains=["a.com", "b.com", "c.com", "d.com", "e.com"])
    qs = parse_qs(urlsplit(f.urls[0]).query)
    assert qs["q"] == ["rates"] and qs["count"] == ["6"]                   # >4 domains: no site: filter, client filter only
    f.urls.clear()
    BraveSearch(KEY, fetcher=f).search("x", count=99)
    assert parse_qs(urlsplit(f.urls[0]).query)["count"] == ["20"]


def test_brave_empty_domain_list_returns_nothing_without_network():
    f = RecordingFetcher(_ok([_item("a", "https://who.int/")]))
    assert BraveSearch(KEY, fetcher=f).search("q", domains=[]) == []
    assert BraveSearch(KEY, fetcher=f).search("q", domains=["not a domain"]) == []
    assert f.urls == []


def test_brave_count_truncation():
    items = [_item(f"t{i}", f"https://who.int/{i}") for i in range(10)]
    r = BraveSearch(KEY, fetcher=RecordingFetcher(_ok(items))).search("q", count=4, domains=["who.int"])
    assert [x.title for x in r] == ["t0", "t1", "t2", "t3"]
    r = BraveSearch(KEY, fetcher=RecordingFetcher(_ok(items))).search("q", count=3)
    assert len(r) == 3


def test_brave_query_is_cleaned_and_capped():
    f = RecordingFetcher(_ok([]))
    BraveSearch(KEY, fetcher=f).search("a\x00b\r\nc\x1b" + "x" * 500)
    from urllib.parse import urlsplit, parse_qs
    q = parse_qs(urlsplit(f.urls[0]).query)["q"][0]
    assert len(q) == 300 and q.startswith("a b c ") and not any(ord(c) < 32 for c in q)
    try:
        BraveSearch(KEY, fetcher=f).search(" \x00\n ")
        assert False
    except SearchError:
        pass


def _err(res, expect):
    try:
        BraveSearch(KEY, fetcher=RecordingFetcher(res)).search("q")
    except SearchError as e:
        assert expect in str(e), str(e)
        assert KEY not in str(e) and KEY not in repr(e)
        return
    assert False, "no SearchError"


def test_brave_error_statuses():
    _err(FetchResult("u", "u", 401), "rejected the key")
    _err(FetchResult("u", "u", 403), "rejected the key")
    _err(FetchResult("u", "u", 429), "quota/rate limit")
    _err(FetchResult("u", "u", 500), "500")
    _err(FetchResult("u", "u", 404), "404")
    _err(FetchResult("u", "u", None, error="timeout", detail="took " + KEY), "timeout")
    _err(FetchResult("u", "u", None, error="blocked"), "blocked")


def test_brave_malformed_responses():
    for body in (b"not json", b"", b"\xff\xfe", b"[]", b'"x"', b"{}", b'{"web": []}', b'{"web": {}}',
                 b'{"web": {"results": "x"}}', b'{"web": null}'):
        _err(FetchResult("u", "u", 200, body), "unreadable search response")
    # Brave omits "web" when there are no hits
    r = BraveSearch(KEY, fetcher=RecordingFetcher(FetchResult("u", "u", 200, b'{"type": "search"}'))).search("q")
    assert r == []
    # junk items inside a good list are skipped
    f = RecordingFetcher(_ok([None, 5, {"title": "no url"}, {"url": 5}, _item("ok", "https://who.int/")]))
    assert [x.title for x in BraveSearch(KEY, fetcher=f).search("q")] == ["ok"]


def test_brave_missing_key(monkeypatch=None):
    old = os.environ.pop("BRAVE_API_KEY", None)
    try:
        for k in (None, "", "   "):
            try:
                BraveSearch(k, fetcher=RecordingFetcher(_ok([]))).search("q")
                assert False
            except SearchError as e:
                assert str(e) == "no search API key: set BRAVE_API_KEY"
        os.environ["BRAVE_API_KEY"] = KEY
        f = RecordingFetcher(_ok([_item("a", "https://who.int/")]))
        assert len(BraveSearch(fetcher=f).search("q")) == 1
        assert f.headers_seen[0]["X-Subscription-Token"] == KEY
    finally:
        os.environ.pop("BRAVE_API_KEY", None)
        if old is not None:
            os.environ["BRAVE_API_KEY"] = old


def test_brave_key_never_leaks():
    b = BraveSearch(KEY, fetcher=RecordingFetcher(_ok([_item("t", "https://who.int/", "s")])))
    assert KEY not in repr(b) and KEY not in str(b)
    for r in b.search("q"):
        assert KEY not in repr(r)
    try:
        BraveSearch("bad\nkey" + KEY, fetcher=RecordingFetcher(_ok([]))).search("q")
        assert False
    except SearchError as e:
        assert KEY not in str(e)


# ---------------------------------------------------------------- FakeSearch

def test_fake_search():
    a = SearchResult("a", "https://who.int/a")
    b = SearchResult("b", "https://evil.com/who.int")
    c = SearchResult("c", "https://www.who.int/c")
    fs = FakeSearch({"q": [a, b, c], "boom": SearchError("down")})
    assert fs.search("q") == [a, b, c]
    assert fs.search("q", domains=["who.int"]) == [a, c]            # same filter as the real client
    assert fs.search("q", count=1) == [a]
    assert fs.search("unknown") == []
    try:
        fs.search("boom")
        assert False
    except SearchError:
        pass
    assert [h.query for h in fs.history] == ["q", "q", "q", "unknown", "boom"]
    assert fs.history[1].domains == ("who.int",) and fs.history[0].domains is None and fs.history[2].count == 1
    q, n, d = fs.history[1]                                          # tuple-unpackable
    fs2 = FakeSearch(lambda q, n, d: [b, a] if q == "x" else SearchError("no"))
    assert fs2.search("x", domains=["who.int"]) == [a]
    try:
        fs2.search("y")
        assert False
    except SearchError:
        pass


# ---------------------------------------------------------------- registry

def test_real_registry_invariants():
    reg = load_registry()
    ds = reg.all_domains()                                           # default: official tier only
    everything = reg.all_domains(TIERS)
    assert len(ds) >= 70 and len(set(ds)) == len(ds)
    for s in reg.sources:
        assert s.domain == s.domain.lower() and "." in s.domain and s.domain.isascii()
        assert not any(c in s.domain for c in "/:*@ \\?#%_")
        assert s.kind in KINDS and set(s.topics) <= set(TOPICS) and s.topics
        assert s.name.strip() and s.country.isupper()
        assert host_matches(f"https://{s.domain}/", [s.domain])      # round-trips
        assert host_matches(f"https://www.{s.domain}/x", [s.domain])
        assert s.tier in TIERS and (s.kind == "reference") == (s.tier == "reference")
        assert reg.is_trusted(f"https://{s.domain}/", TIERS)
        assert reg.is_trusted(f"https://{s.domain}/") == (s.tier == "official")   # default never widens
        assert reg.tier_of(f"https://www.{s.domain}/x") == s.tier
        assert not host_matches(f"https://{s.domain}.evil.com/", [s.domain])
        assert reg.tier_of(f"https://{s.domain}.evil.com/") is None
    for must in ("krisdika.go.th", "ratchakitcha.soc.go.th", "bot.or.th", "who.int", "worldbank.org", "reuters.com"):
        assert must in ds
    assert not reg.is_trusted("https://who.int.evil.com/") and not reg.is_trusted("https://evil.com/who.int")
    assert not reg.is_trusted("https://example.com/")
    assert everything == ds + [d for d in everything if d not in ds]
    refs = reg.all_domains(("reference",))
    assert 1 <= len(refs) <= 8 and "wikipedia.org" in refs and not set(refs) & set(ds)
    assert not reg.is_trusted("https://en.wikipedia.org/wiki/X")          # reference is not "official"
    assert reg.is_trusted("https://th.wikipedia.org/wiki/X", ("reference",))
    assert reg.tier_of("https://th.wikipedia.org/wiki/X") == "reference"
    assert reg.tier_of("https://who.int/x") == "official"
    assert reg.tier_of("https://wikipedia.org.evil.example/x") is None
    assert reg.tier_of("not a url") is None


def test_domains_for_tiers():
    reg = load_registry()
    official = reg.domains_for("other")
    assert reg.domains_for("other", (), ("official",)) == official
    assert reg.domains_for("law") == reg.domains_for("law", tiers=("official",))
    refs = reg.domains_for("other", tiers=("reference",))
    assert set(refs) == set(reg.all_domains(("reference",))) and "wikipedia.org" in refs
    assert not set(refs) & set(official) and not set(official) & set(refs)
    # topic hits in the official tier never leak reference domains, and vice versa
    law_ref = reg.domains_for("law", ["health"], tiers=("reference",))
    assert set(law_ref) == set(refs)                    # no reference matched the topics -> fallback stays in tier
    both = reg.domains_for("other", tiers=("official", "reference"))
    assert set(both) == set(official) | set(refs)
    assert reg.domains_for("other", tiers=()) == []
    assert reg.domains_for("other", tiers="reference") == refs
    for bad in (("bogus",), ("official", "news"), "Official"):
        try:
            reg.domains_for("other", tiers=bad); assert False, bad
        except ValueError as e:
            assert "tier" in str(e)
    for call in (lambda: reg.all_domains(("x",)), lambda: reg.is_trusted("https://who.int/", ("x",)),
                 lambda: reg.domains_for("other", tiers=5)):
        try:
            call(); assert False
        except ValueError:
            pass


def test_domains_for_ordering_and_fallbacks():
    reg = load_registry()
    law = reg.domains_for("law")
    assert law and law[0] == "ratchakitcha.soc.go.th"
    assert "krisdika.go.th" in law and "legislation.gov.uk" in law
    th = [d for d in law if d.endswith(".th")]
    assert law[:len(th)] == th                                       # Thai first
    assert "reuters.com" not in law
    price = reg.domains_for("price")
    assert "bot.or.th" in price and "who.int" not in price
    num = reg.domains_for("number")
    assert "nso.go.th" in num and "who.int" in num
    # news is always last
    allnews = {s.domain for s in reg.sources if s.kind == "news"}
    mixed = reg.domains_for("other", ["news", "economy"])
    first_news = min(i for i, d in enumerate(mixed) if d in allnews)
    assert all(d in allnews for d in mixed[first_news:])
    # fallback: no topic matched -> all non-news
    other = reg.domains_for("other")
    assert set(other) == set(reg.all_domains()) - allnews
    assert set(reg.domains_for("other", ["nonexistent-topic"])) == set(other)
    assert reg.domains_for("name", ["health"])[0] == "moph.go.th"
    # given topics union with implied ones
    assert {"who.int", "krisdika.go.th"} <= set(reg.domains_for("law", ["health"]))


def test_describe_longest_match_wins():
    reg = load_registry()
    assert reg.describe("https://www.legislation.gov.uk/ukpga/1") == "UK legislation (The National Archives)"
    assert reg.describe("https://www.gov.uk/x") == "UK Government"
    assert reg.describe("https://ddc.moph.go.th/a") == "Department of Disease Control (Thailand)"
    assert reg.describe("https://www.moph.go.th/") == "Ministry of Public Health (Thailand)"
    assert reg.describe("https://who.int.evil.com/") is None
    assert reg.describe("not a url") is None


def test_registry_is_immutable():
    reg = load_registry()
    try:
        reg.sources = ()
        assert False
    except Exception as e:
        assert type(e).__name__ == "FrozenInstanceError"


def _write(obj, raw=None):
    fd, p = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(raw if raw is not None else json.dumps(obj))
    return p


def _entry(**kw):
    e = {"domain": "example.gov", "name": "Example", "kind": "government", "tier": "official",
         "country": "US", "topics": ["law"]}
    e.update(kw)
    return e


def test_bad_registry_files_raise_valueerror():
    good = {"sources": [_entry()]}
    p = _write(good)
    assert load_registry(p).all_domains() == ["example.gov"]
    os.unlink(p)
    p = _write({"sources": [_entry(), _entry(domain="wiki.example", kind="reference", tier="reference",
                                             topics=["general"])]})
    r = load_registry(p)
    os.unlink(p)
    assert r.all_domains() == ["example.gov"] and r.all_domains(("reference",)) == ["wiki.example"]
    assert r.domains_for("law") == ["example.gov"] and r.domains_for("law", tiers=("reference",)) == ["wiki.example"]
    assert r.tier_of("https://wiki.example/x") == "reference" and r.tier_of("https://example.gov/") == "official"
    assert not r.is_trusted("https://wiki.example/x")
    bads = [
        ({"sources": [_entry(domain="Example.gov")]}, "Example.gov"),
        ({"sources": [_entry(domain="example")]}, "example"),
        ({"sources": [_entry(domain="*.example.gov")]}, "*.example.gov"),
        ({"sources": [_entry(domain="https://example.gov")]}, "https"),
        ({"sources": [_entry(domain="example.gov/path")]}, "example.gov/path"),
        ({"sources": [_entry(domain="example.gov:80")]}, "example.gov:80"),
        ({"sources": [_entry(domain="1.2.3.4")]}, "1.2.3.4"),
        ({"sources": [_entry(domain="exämple.gov")]}, "entry #0"),
        ({"sources": [_entry(domain=5)]}, "entry #0"),
        ({"sources": [_entry(), _entry()]}, "duplicate"),
        ({"sources": [_entry(kind="blog")]}, "kind"),
        ({"sources": [_entry(tier="gold")]}, "tier"),
        ({"sources": [_entry(tier=None)]}, "tier"),
        ({"sources": [_entry(tier="news")]}, "tier"),
        ({"sources": [_entry(kind="reference")]}, "reference"),       # reference kind needs reference tier
        ({"sources": [_entry(tier="reference")]}, "reference"),       # and vice versa
        ({"sources": [{k: v for k, v in _entry().items() if k != "tier"}]}, "keys"),
        ({"sources": [_entry(country="us")]}, "country"),
        ({"sources": [_entry(country=5)]}, "country"),
        ({"sources": [_entry(topics=["gossip"])]}, "gossip"),
        ({"sources": [_entry(topics=[])]}, "topics"),
        ({"sources": [_entry(topics="law")]}, "topics"),
        ({"sources": [_entry(name="")]}, "name"),
        ({"sources": [{"domain": "a.gov"}]}, "keys"),
        ({"sources": [_entry(extra=1)]}, "keys"),
        ({"sources": ["x"]}, "entry #0"),
        ({"sources": {}}, "sources"),
        ([], "sources"),
        ({"nope": 1}, "sources"),
    ]
    for obj, frag in bads:
        p = _write(obj)
        try:
            load_registry(p)
            assert False, obj
        except ValueError as e:
            assert frag in str(e), (frag, str(e))
        finally:
            os.unlink(p)
    p = _write(None, raw="{not json")
    try:
        load_registry(p); assert False
    except ValueError:
        pass
    finally:
        os.unlink(p)
    try:
        load_registry(os.path.join(tempfile.gettempdir(), "definitely-missing-registry.json")); assert False
    except ValueError:
        pass


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("ok  ", name)
            except Exception as e:
                fails += 1; print("FAIL", name, repr(e))
    sys.exit(1 if fails else 0)
