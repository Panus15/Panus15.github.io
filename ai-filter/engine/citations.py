"""Citation checker: do the links, DOIs and references in an AI answer exist?

Handles ClaimKind.URL, DOI and REFERENCE only. One Finding per such claim, in
input order. All network access goes through the injected Fetcher.

What each VERIFIED means (and does not mean):
  * URL       the link opens. It does NOT show the page supports the claim.
  * DOI       the DOI is registered. It does NOT show the work says what the AI says.
  * REFERENCE a record with that title/authors/year exists. Same caveat.

CONTRADICTED is reserved for positive evidence: a 404/410, a domain that does
not exist, a DOI that doi.org says is unregistered, or a real paper whose
title matches but whose authors do not (the classic hallucinated citation).
Absence from a database (OpenAlex, Crossref) is never evidence by itself, and
any failure of our own machinery is UNVERIFIED (see engine/models.py).
"""

from __future__ import annotations

import difflib
import html
import json
import re
import unicodedata
from typing import Any, Optional, Sequence

from .models import Claim, ClaimKind, Finding, Status
from .net import Fetcher, FetchResult, UrllibFetcher

MODULE = "citations"

CROSSREF = "https://api.crossref.org/works/"
HANDLES = "https://doi.org/api/handles/"
OPENALEX = "https://api.openalex.org/works?search="

TITLE_SIMILARITY = 0.90
MIN_TITLE_CHARS = 12
URL_BODY_BYTES = 64 * 1024
_KINDS = (ClaimKind.URL, ClaimKind.DOI, ClaimKind.REFERENCE)


# --------------------------------------------------------------------------
# small string helpers (hand-rolled so that only net.py touches the
# network-related stdlib modules)
# --------------------------------------------------------------------------

_UNRESERVED = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-~")


def _quote(s: str, safe: str = "") -> str:
    """Percent-encode (UTF-8; letters, digits and _.-~ kept, plus `safe`)."""
    out = []
    for b in s.encode("utf-8"):
        ch = chr(b)
        if b < 128 and (ch in _UNRESERVED or ch in safe):
            out.append(ch)
        else:
            out.append("%%%02X" % b)
    return "".join(out)


_PATH_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]*://[^/?#]*([^?#]*)")


def _path(url: str) -> str:
    m = _PATH_RE.match(url or "")
    return m.group(1) if m else ""


def _short(s: str, n: int = 90) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def _fold(s: str) -> str:
    """Lower-case and strip accents (for author names)."""
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c)).casefold()


def _norm_title(s: str) -> str:
    """Lower-case, drop punctuation/whitespace. Keeps letters, digits and
    combining marks (so Thai vowels/tone marks survive)."""
    s = unicodedata.normalize("NFKC", s).casefold()
    return "".join(c for c in s if unicodedata.category(c)[0] in "LNM")


def _as_year(v: Any) -> Optional[int]:
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, str) and v.strip().isdigit():
        return int(v.strip())
    return None


def _json_body(r: FetchResult) -> Any:
    try:
        return json.loads(r.body.decode("utf-8", errors="replace"))
    except Exception:
        return None


# --------------------------------------------------------------------------
# per-call context: cache + lookup budget
# --------------------------------------------------------------------------

class _LimitReached(Exception):
    pass


class _Ctx:
    def __init__(self, fetcher: Optional[Fetcher], timeout: float, max_lookups: int):
        self.fetcher = fetcher
        self.timeout = timeout
        self.max_lookups = max_lookups
        self.used = 0
        self.cache: dict[str, FetchResult] = {}

    def fetch(self, url: str, *, api: bool = False, max_bytes: Optional[int] = None) -> FetchResult:
        """Cached fetch. Never raises except _LimitReached. A fetcher that raises
        or returns junk becomes error='other' (our failure => UNVERIFIED)."""
        if url in self.cache:
            return self.cache[url]
        if self.used >= self.max_lookups:
            raise _LimitReached()
        self.used += 1
        kwargs: dict[str, Any] = {"timeout": self.timeout}
        if api:
            kwargs["headers"] = {"Accept": "application/json"}
        if max_bytes:
            kwargs["max_bytes"] = max_bytes
        try:
            if self.fetcher is None:
                raise RuntimeError("no fetcher available")
            r = self.fetcher.fetch(url, **kwargs)
            if not isinstance(r, FetchResult):
                raise TypeError("fetcher returned %r" % type(r).__name__)
        except Exception as e:
            r = FetchResult(url, url, None, error="other", detail="fetcher failed: %r" % (e,))
        self.cache[url] = r
        return r


def _finding(claim: Claim, status: Status, reason: str, *evidence: str) -> Finding:
    return Finding(claim, status, MODULE, reason, tuple(e for e in evidence if e))


def _unverified(claim: Claim, reason: str, *evidence: str) -> Finding:
    return _finding(claim, Status.UNVERIFIED, reason, *evidence)


# --------------------------------------------------------------------------
# URL
# --------------------------------------------------------------------------

_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.I | re.S)
_SOFT404_WORDS = ("404", "not found", "page not found", "ไม่พบ", "ไม่พบหน้า")


def _soft_404_title(body: bytes) -> Optional[str]:
    text = body[:URL_BODY_BYTES].decode("utf-8", errors="replace")
    m = _TITLE_RE.search(text)
    if not m:
        return None
    title = " ".join(html.unescape(m.group(1)).split())
    low = title.casefold()
    if any(w in low for w in _SOFT404_WORDS):
        return title
    return None


def _check_url(claim: Claim, ctx: _Ctx) -> Finding:
    url = claim.data.get("url")
    if not isinstance(url, str) or not url.strip():
        return _unverified(claim, "We could not check this link because no address was extracted.")
    url = url.strip()
    r = ctx.fetch(url, max_bytes=URL_BODY_BYTES)
    ev = [url]
    if r.final_url and r.final_url != url:
        ev.append("redirected to " + r.final_url)

    if r.error == "dns":
        return _finding(claim, Status.CONTRADICTED,
                        "The domain does not exist, so this link cannot be real; check for a typo or an invented address.",
                        *ev)
    if r.error or r.status is None:
        what = r.error or "no response"
        if r.error == "blocked":
            what = "blocked by our own safety guard"
        return _unverified(claim, "We could not check this link (%s); open it yourself to see if it works." % what, *ev)

    s = r.status
    if 200 <= s < 300:
        if _path(url).strip("/") and _path(r.final_url or url) in ("", "/"):
            return _unverified(claim, "The link redirects to the site's home page, which usually means the original page no longer exists; open it yourself.", *ev)
        title = _soft_404_title(r.body or b"")
        if title is not None:
            return _unverified(claim, 'The link opens but the page title says "%s", which suggests the page is missing; open it yourself.' % _short(title, 60), *ev)
        return _finding(claim, Status.VERIFIED,
                        "The link opens, but that does NOT show the page supports what the AI said; read the page yourself.",
                        *ev)
    if s in (404, 410):
        return _finding(claim, Status.CONTRADICTED,
                        "This link does not exist (the site answers HTTP %d, page gone)." % s, *ev)
    if 300 <= s < 400:
        return _unverified(claim, "We could not check this link: it ended on an unresolved redirect (HTTP %d)." % s, *ev)
    if s in (401, 403, 429, 999) or s >= 500:
        return _unverified(claim, "The site did not let us in or failed to answer (HTTP %d); many sites block automated checks, so open the link yourself." % s, *ev)
    return _unverified(claim, "The site answered HTTP %d, which does not show whether the page exists; open the link yourself." % s, *ev)


# --------------------------------------------------------------------------
# DOI
# --------------------------------------------------------------------------

def _crossref_year(msg: dict) -> Optional[int]:
    for key in ("issued", "published", "published-print", "published-online", "created"):
        block = msg.get(key)
        if not isinstance(block, dict):
            continue
        try:
            y = block["date-parts"][0][0]
        except Exception:
            continue
        if isinstance(y, int) and not isinstance(y, bool):
            return y
    return None


def _check_doi(claim: Claim, ctx: _Ctx) -> Finding:
    doi = claim.data.get("doi")
    doi = doi.strip().lower() if isinstance(doi, str) else ""
    if not doi.startswith("10.") or "/" not in doi:
        return _unverified(claim, "We could not check this DOI because it is not in the usual 10.xxxx/... form; check it for typos.")
    q = _quote(doi, safe="/")
    cr_url = CROSSREF + q
    r = ctx.fetch(cr_url, api=True)

    if r.error or r.status is None:
        return _unverified(claim, "We could not check this DOI (Crossref lookup failed: %s); try again later." % (r.error or "no response"), cr_url)

    if r.status == 200:
        data = _json_body(r)
        msg = data.get("message") if isinstance(data, dict) else None
        if not isinstance(msg, dict):
            return _unverified(claim, "We could not check this DOI (Crossref sent an unreadable answer).", cr_url)
        titles = msg.get("title")
        title = titles[0] if isinstance(titles, list) and titles and isinstance(titles[0], str) else ""
        year = _crossref_year(msg)
        desc = '"%s"' % _short(title, 120) if title else "(no title on record)"
        if year:
            desc += " (%d)" % year
        return _finding(claim, Status.VERIFIED,
                        "The DOI is registered in Crossref as %s; this shows the work exists, not that it says what the AI claims." % desc,
                        cr_url)

    if r.status != 404:
        return _unverified(claim, "We could not check this DOI (Crossref answered HTTP %d); try again later." % r.status, cr_url)

    # Crossref does not know it. Other registries (DataCite etc.) may, so ask the DOI handle system.
    h_url = HANDLES + q
    h = ctx.fetch(h_url, api=True)
    ev = (cr_url, h_url)
    if h.error or h.status is None:
        return _unverified(claim, "Crossref has no record of this DOI and we could not check the DOI system (%s); search for it manually." % (h.error or "no response"), *ev)
    data = _json_body(h)
    code = data.get("responseCode") if isinstance(data, dict) else None
    if isinstance(code, int) and not isinstance(code, bool):
        if code == 1:
            return _finding(claim, Status.VERIFIED,
                            "The DOI is registered with doi.org (not in Crossref, so we cannot read its title); this shows it exists, not that it says what the AI claims.",
                            *ev)
        if code == 100:
            return _finding(claim, Status.CONTRADICTED,
                            "This DOI is not registered anywhere (doi.org has no record of it), so it is probably invented or mistyped.",
                            *ev)
    return _unverified(claim, "Crossref has no record of this DOI and the DOI system gave no clear answer; search for it manually.", *ev)


# --------------------------------------------------------------------------
# REFERENCE
# --------------------------------------------------------------------------

_INITIALS_RE = re.compile(r"[A-Z]{1,3}\.?|(?:[A-Z]\.){1,3}")


def _word_tokens(s: str) -> list[str]:
    return [t for t in re.split(r"[^\w]+", _fold(s)) if t]


def _surname_token(entry: str) -> Optional[str]:
    """The claimed surname (one token) from 'Vaswani, A.', 'Ashish Vaswani',
    'A. Vaswani', 'Smith JK'. None if nothing usable ('et al.')."""
    entry = re.sub(r"\bet\.?\s*al\.?", " ", entry, flags=re.I).strip()
    if not entry:
        return None
    if "," in entry:                       # "Surname, Given"
        toks = _word_tokens(entry.split(",", 1)[0])
        return toks[-1] if toks else None
    words = entry.split()
    pick = words[-1]
    if len(words) >= 2 and _INITIALS_RE.fullmatch(pick):   # Vancouver "Smith JK"
        pick = words[-2]
    toks = _word_tokens(pick)
    return toks[-1] if toks else None


def _claimed_surnames(authors: Any) -> list[str]:
    """One surname token per claimed author."""
    if not isinstance(authors, (list, tuple)):
        return []
    out: list[str] = []
    for a in authors:
        if not isinstance(a, str):
            continue
        for piece in re.split(r"\s+(?:and|&)\s+|;", a):
            tok = _surname_token(piece)
            if tok:
                out.append(tok)
    return out


def _record_authors(work: dict) -> tuple[list[str], bool]:
    names: list[str] = []
    auths = work.get("authorships")
    if isinstance(auths, list):
        for a in auths:
            try:
                n = a["author"]["display_name"]
            except Exception:
                continue
            if isinstance(n, str) and n.strip():
                names.append(n.strip())
    return names, work.get("is_authors_truncated") is True


def _authors_consistent(surnames: list[str], names: list[str], truncated: bool) -> bool:
    """True unless we have positive evidence that NONE of the claimed surnames
    is among the record's authors. Unknown (no authors on record, truncated
    list) counts as consistent: absence of data is not a conflict."""
    if not surnames or not names or truncated:
        return True
    tokens = set()
    for n in names:
        tokens.update(t for t in re.split(r"[^\w]+", _fold(n)) if t)
    return any(s in tokens for s in surnames)


def _check_reference(claim: Claim, ctx: _Ctx) -> Finding:
    d = claim.data
    title = d.get("title")
    title = " ".join(title.split()) if isinstance(title, str) else ""
    if len(title) < MIN_TITLE_CHARS:
        return _unverified(claim, "The title is too short to search reliably; look the work up manually.")
    surnames = _claimed_surnames(d.get("authors"))
    claimed_authors = [a for a in (d.get("authors") or []) if isinstance(a, str)] if isinstance(d.get("authors"), (list, tuple)) else []
    year = _as_year(d.get("year"))

    url = OPENALEX + _quote(title[:300]) + "&per-page=5"
    r = ctx.fetch(url, api=True)
    if r.error or r.status != 200:
        why = r.error or ("HTTP %s" % r.status)
        return _unverified(claim, "We could not check this reference (OpenAlex lookup failed: %s); search for it manually." % why, url)
    data = _json_body(r)
    results = data.get("results") if isinstance(data, dict) else None
    if not isinstance(results, list):
        return _unverified(claim, "We could not check this reference (OpenAlex sent an unreadable answer).", url)

    want = _norm_title(title)
    matches = []
    for w in results:
        if not isinstance(w, dict):
            continue
        name = w.get("display_name") or w.get("title")
        if not isinstance(name, str) or not name.strip():
            continue
        have = _norm_title(name)
        if not have or not want:
            continue
        ratio = difflib.SequenceMatcher(None, want, have).ratio()
        if ratio >= TITLE_SIMILARITY:
            names, trunc = _record_authors(w)
            py = w.get("publication_year")
            py = py if isinstance(py, int) and not isinstance(py, bool) else None
            matches.append({
                "work": w, "name": name.strip(), "ratio": ratio, "authors": names, "py": py,
                "a_ok": _authors_consistent(surnames, names, trunc),
                "y_ok": year is None or py is None or abs(year - py) <= 1,
            })
    if not matches:
        return _unverified(claim, "Not found in OpenAlex; books, Thai-language and non-indexed works are often missing, so search for it manually.", url)
    matches.sort(key=lambda m: -m["ratio"])

    def ev(m):
        wid = m["work"].get("id")
        return (url, "%s %s" % (wid, _short(m["name"], 80)) if isinstance(wid, str) else _short(m["name"], 80))

    full = [m for m in matches if m["a_ok"] and m["y_ok"]]
    if full:
        m = full[0]
        parts = ["title"] + (["authors"] if surnames else []) + (["year"] if year is not None else [])
        shown = '"%s"' % _short(m["name"], 80) + (" (%d)" % m["py"] if m["py"] else "")
        return _finding(claim, Status.VERIFIED,
                        "A record matching the %s exists in OpenAlex: %s; this shows the work exists, not that it says what the AI claims." % ("/".join(parts), shown),
                        *ev(m))

    author_ok = [m for m in matches if m["a_ok"]]
    if author_ok:  # authors fine, year off by more than 1
        m = author_ok[0]
        who = "title and authors" if surnames else "title"
        return _unverified(claim,
                           "A work with this %s exists, but OpenAlex dates it %d while the AI said %d; preprints can differ, so check the citation." % (who, m["py"], year),
                           *ev(m))

    # Every title match has authors that do not include any claimed surname.
    m = matches[0]
    rec = ", ".join(m["authors"][:4]) + (", …" if len(m["authors"]) > 4 else "")
    claimed = ", ".join(claimed_authors[:4]) + (", …" if len(claimed_authors) > 4 else "")
    reason = ('A real paper titled "%s" exists, but its authors are %s, not %s; a real title with the wrong authors is a common sign of an invented citation.'
              % (_short(m["name"], 80), rec, claimed))
    if not m["y_ok"]:
        reason += " The year also differs (%s vs %d)." % (year, m["py"])
    return _finding(claim, Status.CONTRADICTED, reason, *ev(m))


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def _check_one(claim: Claim, ctx: _Ctx) -> Finding:
    try:
        if claim.kind == ClaimKind.URL:
            return _check_url(claim, ctx)
        if claim.kind == ClaimKind.DOI:
            return _check_doi(claim, ctx)
        return _check_reference(claim, ctx)
    except _LimitReached:
        return _unverified(claim, "Not checked: lookup limit reached for this run; verify it manually.")
    except Exception as e:  # a bug or odd response must never break the run
        return _unverified(claim, "We could not check this citation (internal error: %s); verify it manually." % type(e).__name__)


def check(claims: Sequence[Claim], fetcher: Optional[Fetcher] = None, *,
          timeout: float = 10.0, max_lookups: int = 40) -> list[Finding]:
    """Check URL / DOI / REFERENCE claims. One Finding each, input order; other kinds ignored."""
    relevant = [c for c in claims if getattr(c, "kind", None) in _KINDS]
    if not relevant:
        return []
    if fetcher is None:
        try:
            fetcher = UrllibFetcher()
        except Exception:
            fetcher = None
    ctx = _Ctx(fetcher, timeout, max_lookups)
    return [_check_one(c, ctx) for c in relevant]
