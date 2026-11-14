"""The web-search seam: protocol, Brave implementation, fake, and `host_matches`.

`host_matches` is SECURITY-CRITICAL: it decides which sites count as "official".
It fails closed. Anything odd about a URL (backslash, whitespace, control or
format characters, userinfo, non-ASCII host, IP literal, bad port...) makes it
return False rather than guess, because different parsers (this one, a
browser, the fetcher) disagree about such URLs and an attacker lives in the
disagreement.

Rules our code keeps: our failures (no key, quota, HTTP error, bad JSON) raise
SearchError -- callers turn that into UNVERIFIED. The API key never appears in
an exception message, a return value or a log line.
"""

from __future__ import annotations

import html
import json
import os
import re
import unicodedata
import urllib.parse
from dataclasses import dataclass
from typing import Callable, NamedTuple, Optional, Protocol, Sequence, Union

from engine.net import Fetcher


class SearchError(Exception):
    """Search could not be performed (no key, HTTP error, quota, bad response)."""


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str = ""


class SearchClient(Protocol):
    def search(self, query: str, *, count: int = 5,
               domains: Optional[Sequence[str]] = None) -> list[SearchResult]:
        """Web results for `query`. If `domains` is given, ONLY results whose host is
        one of those domains or a subdomain of one are returned. Raises SearchError."""
        ...


# --------------------------------------------------------------------------
# host_matches
# --------------------------------------------------------------------------

_LABEL = re.compile(r"^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?$")


def _valid_hostname(host: str) -> bool:
    """Strict LDH hostname, ASCII, lower-case, >= 2 labels, not an IP address."""
    if not host or len(host) > 253 or not host.isascii():
        return False
    labels = host.split(".")
    if len(labels) < 2 or not all(_LABEL.match(l) for l in labels):
        return False
    if labels[-1].isdigit():          # 1.2.3.4, 0x-less numeric hosts: never a name
        return False
    return True


def _has_forbidden_char(s: str) -> bool:
    # Backslash (browsers read it as "/", urllib does not), and every character
    # in a Control (C*) or Separator (Z*) category: spaces, tabs, newlines, NUL,
    # DEL, zero-width and bidi-override characters, line separators...
    if "\\" in s:
        return True
    return any(unicodedata.category(c)[0] in "CZ" for c in s)


def _normalise_domain(d: object) -> Optional[str]:
    if not isinstance(d, str) or _has_forbidden_char(d):
        return None
    d = d.strip().lower()
    if d.endswith("."):
        d = d[:-1]
    return d if _valid_hostname(d) else None


def _host_of(url: object) -> Optional[str]:
    """The hostname of a plain http(s) URL, or None if the URL is anything but
    boringly well-formed. Never raises."""
    if not isinstance(url, str) or len(url) > 4096 or _has_forbidden_char(url):
        return None
    if not url[:8].lower().startswith(("http://", "https://")):
        return None
    try:
        parts = urllib.parse.urlsplit(url)
        if parts.scheme not in ("http", "https"):
            return None
        netloc = parts.netloc
        if not netloc or not netloc.isascii() or "@" in netloc or "%" in netloc or "[" in netloc or "]" in netloc:
            return None               # userinfo, percent-encoded or IPv6-literal hosts: refuse
        host_port = netloc
        host, sep, port = host_port.partition(":")
        if sep:
            if not (port.isascii() and port.isdigit() and 0 < int(port) <= 65535):
                return None
        if parts.hostname is None or parts.hostname != host.lower():
            return None               # parser and our own split disagree: refuse
        host = host.lower()
        if host.endswith("."):
            host = host[:-1]
        # Non-ASCII hosts are refused outright: IDNA2003 vs IDNA2008 (e.g. "ß")
        # map to different names in different programs. Punycode ("xn--...") is
        # ASCII and is compared literally.
        return host if _valid_hostname(host) else None
    except Exception:
        return None


def host_matches(url: str, domains: Sequence[str]) -> bool:
    """True only if `url` is http(s) and its hostname equals one of `domains`
    or is a subdomain of one. Fails closed on anything unusual."""
    if isinstance(domains, str):
        domains = [domains]           # a bare string would otherwise iterate by character
    host = _host_of(url)
    if host is None:
        return False
    for d in domains or ():
        nd = _normalise_domain(d)
        if nd is not None and (host == nd or host.endswith("." + nd)):
            return True
    return False


# --------------------------------------------------------------------------
# helpers shared by the implementations
# --------------------------------------------------------------------------

_TAG = re.compile(r"<[^>]*>")
_WS = re.compile(r"\s+")
_CTRL = re.compile(r"[\x00-\x1f\x7f-\x9f]")
MAX_QUERY_CHARS = 300
MAX_COUNT = 20


def _clean_text(s: object) -> str:
    if not isinstance(s, str):
        return ""
    s = html.unescape(_TAG.sub("", s))
    return _WS.sub(" ", _CTRL.sub(" ", s)).strip()


def _clean_query(q: str) -> str:
    q = _WS.sub(" ", _CTRL.sub(" ", q if isinstance(q, str) else "")).strip()
    return q[:MAX_QUERY_CHARS].strip()


def _clamp_count(count: int) -> int:
    try:
        return max(1, min(int(count), MAX_COUNT))
    except (TypeError, ValueError):
        return 5


def _filter_domains(domains: Optional[Sequence[str]]) -> Optional[list[str]]:
    """None stays None (no restriction). Otherwise the list of valid, normalised domains."""
    if domains is None:
        return None
    if isinstance(domains, str):
        domains = [domains]
    out: list[str] = []
    for d in domains:
        nd = _normalise_domain(d)
        if nd and nd not in out:
            out.append(nd)
    return out


# --------------------------------------------------------------------------
# Brave
# --------------------------------------------------------------------------

BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"


class BraveSearch:
    def __init__(self, api_key: Optional[str] = None, *, fetcher: Optional[Fetcher] = None,
                 timeout: float = 10.0):
        self._key = api_key
        self._fetcher = fetcher
        self._timeout = timeout

    def __repr__(self) -> str:        # never show the key
        return "BraveSearch(api_key=<hidden>)"

    def _api_key(self) -> str:
        key = self._key if self._key is not None else os.environ.get("BRAVE_API_KEY")
        key = key.strip() if isinstance(key, str) else ""
        if not key:
            raise SearchError("no search API key: set BRAVE_API_KEY")
        if _has_forbidden_char(key):
            raise SearchError("search API key is malformed")
        return key

    def search(self, query: str, *, count: int = 5,
               domains: Optional[Sequence[str]] = None) -> list[SearchResult]:
        key = self._api_key()
        q = _clean_query(query)
        if not q:
            raise SearchError("empty search query")
        count = _clamp_count(count)
        allowed = _filter_domains(domains)
        request_count = count
        if allowed is not None:
            if not allowed:
                return []             # nothing is allowed, so nothing can match
            request_count = min(count * 3, MAX_COUNT)
            if len(allowed) <= 4:
                q = "(" + " OR ".join(f"site:{d}" for d in allowed) + ") " + q
        url = BRAVE_ENDPOINT + "?" + urllib.parse.urlencode({"q": q, "count": request_count})

        fetcher = self._fetcher
        if fetcher is None:
            from engine.net import UrllibFetcher
            fetcher = self._fetcher = UrllibFetcher()
        res = fetcher.fetch(url, headers={"Accept": "application/json", "X-Subscription-Token": key},
                            timeout=self._timeout)
        if res.status is None:
            raise SearchError(f"search request failed ({res.error or 'no response'})")
        if res.status in (401, 403):
            raise SearchError("search API rejected the key")
        if res.status == 429:
            raise SearchError("search quota/rate limit")
        if res.status != 200:
            raise SearchError(f"search API returned HTTP {res.status}")
        try:
            data = json.loads(res.body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError, AttributeError):
            raise SearchError("unreadable search response") from None
        raw = self._results_of(data)

        out: list[SearchResult] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            u = item.get("url")
            if not isinstance(u, str) or _host_of(u) is None:
                continue              # not a plain http(s) URL: never hand it onward
            if allowed is not None and not host_matches(u, allowed):
                continue
            out.append(SearchResult(_clean_text(item.get("title")), u, _clean_text(item.get("description"))))
            if len(out) >= count:
                break
        return out

    @staticmethod
    def _results_of(data: object) -> list:
        if not isinstance(data, dict):
            raise SearchError("unreadable search response")
        web = data.get("web")
        if web is None and data.get("type") == "search":
            return []                 # Brave omits "web" when there are no web hits
        if not isinstance(web, dict) or not isinstance(web.get("results"), list):
            raise SearchError("unreadable search response")
        return web["results"]


# --------------------------------------------------------------------------
# Fake, for other modules' tests
# --------------------------------------------------------------------------

class SearchCall(NamedTuple):
    query: str
    count: int
    domains: Optional[tuple]


class FakeSearch:
    """`table`: dict query -> list[SearchResult] (or an Exception instance to raise),
    or a callable (query, count, domains) -> list / Exception. Applies the same
    `host_matches` filter as the real client when `domains` is given, then
    truncates to `count`. Unknown query -> []. `.history` records every call."""

    def __init__(self, table: Union[dict, Callable]):
        self.table = table
        self.history: list[SearchCall] = []

    def search(self, query: str, *, count: int = 5,
               domains: Optional[Sequence[str]] = None) -> list[SearchResult]:
        dom = None if domains is None else tuple([domains] if isinstance(domains, str) else domains)
        self.history.append(SearchCall(query, count, dom))
        if callable(self.table):
            value = self.table(query, count, dom)
        else:
            value = self.table.get(query, [])
        if isinstance(value, BaseException):
            raise value
        results = list(value or [])
        if dom is not None:
            results = [r for r in results if host_matches(r.url, dom)]
        return results[:max(0, count)]
