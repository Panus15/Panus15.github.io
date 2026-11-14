"""Support check: does the page behind a link actually say what the answer says?

    refine(text, findings, *, llm, fetcher) -> list[Finding]

citations.py can only say "this link opens". A link that opens but does not
contain the claim is the typical hallucinated citation. This module closes that
gap for URL findings that came back VERIFIED from citations.py:

  1. Derive the ASSERTION the author attaches to the URL from the original text
     (the sentence / line holding the URL, minus the URL, markdown link syntax
     and "source:" style lead-ins; the preceding sentence is added when the
     part before the URL is too thin to carry a claim).
  2. Skip when the assertion is too short or has nothing checkable in it
     ("See https://x.org"): no fetch, no LLM call.
  3. Fetch the page, parse it (engine.pagetext) and ask the judge (engine.judge,
     an LLM whose answer is mechanically verified there).
  4. Replace the finding (module="support", same Claim):
        supports     -> VERIFIED     with a short verbatim quote
        not_stated   -> UNVERIFIED   (deliberately downgrades "link opens")
        contradicts  -> CONTRADICTED with a short verbatim quote
        error / any exception / empty quote -> the ORIGINAL finding, untouched.
     An LLMError means the LLM budget is gone: stop trying further candidates.

The returned list has the same length and order as the input; everything that
is not a candidate is the very same object. Page text never enters a Finding
beyond the short quote. Our own failures never turn into CONTRADICTED.

Sentence splitting here is deliberately local to this module (offsets are into
the ORIGINAL text, CRLF / emoji / Thai combining marks included).
"""

from __future__ import annotations

import re
from bisect import bisect_right
from typing import Any, Callable, Optional, Sequence

from .llm import LLMError
from .models import Claim, ClaimKind, Finding, Status

__all__ = ["refine", "assertion_for", "is_checkable"]

MODULE = "support"
MAX_PAGE_BYTES = 512 * 1024
ASSERTION_CAP = 300
MIN_ASSERTION_CHARS = 25
MIN_THAI_CHARS = 20
QUOTE_CAP = 200

NOT_STATED_REASON = "The link opens, but we could not find this claim on the page — read it yourself."


# --------------------------------------------------------------------------
# sentence splitting (offsets into the original text)
# --------------------------------------------------------------------------

_CLOSERS = "\"'”’)]»）」』"
_ALWAYS_ABBR = frozenset((
    "e.g i.e cf vs viz dr mr mrs ms prof sr jr st mt gen gov sen rep hon rev lt col capt sgt "
    "approx dept").split())
_NUM_ABBR = frozenset("fig figs eq eqs sec ch vol pp p no nos art ed eds al".split())
_STARTERS = frozenset(
    "the it this that he she we they in on at a i there these those and but as if for however then".split())
_DOTTED = re.compile(r"(?:[A-Za-z]\.)*[A-Za-z]")
_THAI_ABBR = re.compile(r"(?:[฀-๿]{1,3}\.)*[฀-๿]{1,4}")
_LISTNUM = re.compile(r"\d{1,3}(?:\.\d{1,3})*")


def _line_start(text: str, t: int) -> bool:
    """True if only indentation / bullet characters precede position t on its line."""
    ln = text.rfind("\n", 0, t)
    seg = text[ln + 1:t]
    return not seg.strip(" \t\r-*+•>#")


def _hard_wrapped(text: str, i: int) -> bool:
    """Newline at i in the middle of a wrapped line: previous line ends in a
    word/comma and the next one starts with a lower-case ASCII letter."""
    j = i - 1
    while j >= 0 and text[j] in " \t\r":
        j -= 1
    k = i + 1
    while k < len(text) and text[k] in " \t\r":
        k += 1
    if j < 0 or k >= len(text):
        return False
    prev, nxt = text[j], text[k]
    return (prev.isalnum() or prev in ",;") and nxt.isascii() and nxt.islower()


def _dot_is_boundary(text: str, i: int, j: int) -> bool:
    """text[i] == '.', text[j:] is what follows it (after closing quotes/brackets)."""
    n = len(text)
    k = j
    while k < n and text[k] in " \t ":
        k += 1
    if k >= n or text[k] in "\r\n":
        return True
    nxt = text[k]
    if nxt.isascii() and nxt.islower():
        return False
    t = i
    while t > 0 and not text[t - 1].isspace() and text[t - 1] not in "([\"'“‘<{（":
        t -= 1
    tok = text[t:i]
    if not tok:
        return True
    low = tok.lower()
    if low in _ALWAYS_ABBR:
        return False
    if low in _NUM_ABBR and (nxt.isdigit() or nxt in "([,"):
        return False
    if _LISTNUM.fullmatch(tok) and _line_start(text, t):
        return False                                   # "1. Item", "2.1. Item"
    if len(tok) == 1 and tok.isalpha() and _line_start(text, t):
        return False                                   # "a. Item"
    if _THAI_ABBR.fullmatch(tok):
        return False                                   # นพ. ดร. พ.ศ. ค.ศ.
    if _DOTTED.fullmatch(tok) and (len(tok) > 1 or tok.isupper()):
        # initials / acronyms: "J. K. Rowling", "U.S. policy", "9 a.m. Monday"
        m = re.match(r"[A-Za-z]+", text[k:])
        if m and m.group(0).lower() in _STARTERS:
            return True
        if nxt.isupper() or nxt.isdigit():
            return False
    return True


def _cuts(text: str) -> list[int]:
    n = len(text)
    cuts: list[int] = []
    i = 0
    while i < n:
        c = text[i]
        if c == "\n" or (c == "\r" and text[i + 1:i + 2] != "\n"):
            if not _hard_wrapped(text, i):
                cuts.append(i + 1)
        elif c in "。！？｡":
            j = i + 1
            while j < n and text[j] in _CLOSERS:
                j += 1
            cuts.append(j)
            i = j - 1
        elif c in ".?!":
            j = i + 1
            while j < n and text[j] in _CLOSERS:
                j += 1
            if j >= n or text[j].isspace():
                if c != ".":
                    k = j
                    while k < n and text[k] in " \t":
                        k += 1
                    ok = not (k < n and text[k].isascii() and text[k].islower())
                else:
                    ok = _dot_is_boundary(text, i, j)
                if ok:
                    cuts.append(j)
            i = j - 1
        i += 1
    return cuts


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    points = sorted(set([0] + _cuts(text) + [len(text)]))
    out: list[tuple[int, int]] = []
    for a, b in zip(points, points[1:]):
        seg = text[a:b]
        if not any(ch.isalnum() for ch in seg):
            continue
        lead = len(seg) - len(seg.lstrip())
        trail = len(seg) - len(seg.rstrip())
        out.append((a + lead, b - trail))
    return out


# --------------------------------------------------------------------------
# cleaning a sentence into an assertion
# --------------------------------------------------------------------------

_M0, _M1 = "\x00", "\x01"          # other URL / the URL being checked
_MK = "[\x00\x01]"
_THAI_RANGE = "฀-๿"

_URL_FALLBACK = re.compile(
    r"(?<![A-Za-z0-9_.@/-])(?:https?://|www\.)"
    r"[^\s<>\[\]\x00\x01" + _THAI_RANGE + "⺀-鿿＀-￯‘-”«»]+", re.I)
_TRAIL_PUNCT = ".,;:!?)'\"*"

_LEAD_EN = (
    r"(?:for\s+)?(?:more|further)\s+(?:info(?:rmation)?|details?|reading)(?:\s+(?:is\s+)?(?:available\s+)?(?:at|here|on))?"
    r"|read\s+(?:more|the\s+full\s+(?:article|report|story))(?:\s+(?:at|here|on))?"
    r"|learn\s+more(?:\s+(?:at|here|on))?"
    r"|click\s+here(?:\s+(?:for\s+more|to\s+read))?"
    r"|see\s+also|sources?|references?|ref|via|see|link|url|available\s+(?:at|online\s+at)|found\s+(?:at|here)"
    r"|cited\s+(?:in|at)|full\s+(?:text|report)|details"
)
_LEAD_TH = (
    "อ่านเพิ่มเติมที่|อ่านเพิ่มเติม|ดูเพิ่มเติมที่|ดูเพิ่มเติม|แหล่งที่มา|แหล่งอ้างอิง|ที่มา|อ้างอิง|"
    "ลิงก์|ลิงค์|ดูที่|อ่านที่|ดู"
)
_LEAD_BEFORE_MARK = re.compile(
    r"(?:(?<![A-Za-z])(?:" + _LEAD_EN + r")|(?:" + _LEAD_TH + r"))[ \t]*[:：\-–—]?[ \t]*(?=" + _MK + r")", re.I)
_PAREN_MARK = re.compile(
    r"[(\[（<«]\s*(?:(?:(?<![A-Za-z])(?:" + _LEAD_EN + r")|(?:" + _LEAD_TH + r"))\s*[:：]?\s*)?"
    r"(" + _MK + r"(?:[\s,;]*" + _MK + r")*)\s*[)\]）>»]", re.I)
_LEAD_AT_START = re.compile(
    r"^\s*(?:sources?|references?|ref|ที่มา|แหล่งที่มา|แหล่งอ้างอิง|อ้างอิง)\s*[:：]\s*", re.I)
_LIST_PREFIX = re.compile(r"^\s*(?:[-*+•>]+\s+|\d{1,3}[.)]\s+|#{1,6}\s+)")
_MD_LINK = re.compile(
    r"!?\[([^\]]*)\]\(\s*(" + _MK + r")\s*(?:\"[^\"]*\"|'[^']*')?\s*\)")
_GENERIC_LABELS = frozenset((
    "here", "link", "source", "sources", "this", "this link", "click here", "read more", "more", "website",
    "site", "page", "this page", "url", "ที่นี่", "ลิงก์", "ลิงค์", "แหล่งที่มา", "ที่มา", "อ้างอิง", "ต้นฉบับ"))
_MARK_JOIN = re.compile(r"(" + _MK + r")\s*(?:[,;&]|\b(?:and|or)\b|และ|หรือ)\s*(?=" + _MK + r")", re.I)
_EMPTY_BRACKETS = re.compile(r"[(\[（]\s*[,;:]?\s*[)\]）]")


def _clean(marked: str) -> str:
    """Remove link syntax and lead-ins; keep the URL markers; collapse whitespace."""
    def link(m: re.Match) -> str:
        label = re.sub(_MK, "", m.group(1)).strip()
        if not label or label.casefold() in _GENERIC_LABELS or label.casefold().startswith(("http://", "https://", "www.")):
            label = ""
        return (label + m.group(2)) if label else m.group(2)

    s = _MD_LINK.sub(link, marked)
    s = _MARK_JOIN.sub(r"\1", s)
    s = _PAREN_MARK.sub(lambda m: m.group(1), s)
    for _ in range(3):
        t = _LEAD_BEFORE_MARK.sub("", s)
        if t == s:
            break
        s = t
    s = _LIST_PREFIX.sub("", s, count=1)
    s = _LEAD_AT_START.sub("", s, count=1)
    s = s.replace("**", "").replace("__", "")
    for _ in range(2):
        s = _EMPTY_BRACKETS.sub(" ", s)
    return " ".join(s.split())


def _finish(s: str) -> str:
    s = re.sub(_MK, " ", s)
    s = " ".join(s.split())
    s = re.sub(r"\s+([.,;:!?。])", r"\1", s)
    s = re.sub(r"[,;:]+([.!?。])$", r"\1", s)
    return s.strip(" :;,-–—|")


def _nonspace(s: str) -> int:
    return sum(1 for ch in s if not ch.isspace())


def _window(marked: str, cap: int) -> str:
    """Cap a marked string at `cap` chars, keeping what is nearest the \\x01 marker."""
    if len(marked) <= cap:
        return marked
    idx = marked.find(_M1)
    if idx < 0:
        idx = len(marked) // 2
    before, after = marked[:idx], marked[idx:]
    na = min(len(after), 100)
    nb = min(len(before), cap - na)
    na = min(len(after), cap - nb)
    b = before[len(before) - nb:] if nb else ""
    a = after[:na]
    # prefer cutting at a space when the cut falls inside a spaced (non-Thai) word
    if nb < len(before) and b and b[0] != " " and before[len(before) - nb - 1] != " ":
        sp = b.find(" ")
        if 0 <= sp < 30:
            b = b[sp + 1:]
    if na < len(after) and a and a[-1] != " " and after[na] != " ":
        sp = a.rfind(" ")
        if sp > len(a) - 30 and sp > 0:
            a = a[:sp]
    return b + a


# --------------------------------------------------------------------------
# the document: sentences + URL spans, assertion per claim
# --------------------------------------------------------------------------

class _Doc:
    def __init__(self, text: str, claims: Sequence[Claim]):
        self.text = text
        self.sents = _sentence_spans(text)
        self.starts = [a for a, _ in self.sents]
        spans = set()
        for c in claims:
            if c.kind in (ClaimKind.URL, ClaimKind.DOI) and 0 <= c.start < c.end <= len(text):
                spans.add((c.start, c.end))
        self.spans = sorted(spans)

    def _marked(self, a: int, b: int, cur: Optional[tuple[int, int]]) -> str:
        parts: list[str] = []
        pos = a

        def plain(seg: str) -> str:
            def sub(m: re.Match) -> str:
                u = m.group(0)
                k = len(u)
                while k and u[k - 1] in _TRAIL_PUNCT:
                    k -= 1
                return _M0 + u[k:]
            return _URL_FALLBACK.sub(sub, seg)

        for s, e in self.spans:
            if e <= a or s >= b:
                continue
            s, e = max(s, pos), min(e, b)
            if s < pos or s >= e:
                continue
            parts.append(plain(self.text[pos:s]))
            parts.append(_M1 if cur is not None and (s, e) == cur else _M0)
            pos = e
        parts.append(plain(self.text[pos:b]))
        return "".join(parts)

    def _index(self, pos: int) -> int:
        i = bisect_right(self.starts, pos) - 1
        return i if i >= 0 and self.sents[i][0] <= pos < self.sents[i][1] + 1 else -1

    def assertion(self, claim: Claim) -> str:
        i = self._index(claim.start)
        if i < 0:
            return ""
        a, b = self.sents[i]
        cur = (claim.start, min(claim.end, b))
        cur_marked = _clean(self._marked(a, b, cur if cur[0] < cur[1] else None))
        if _M1 not in cur_marked:                          # e.g. link syntax swallowed it
            cur_marked += _M1
        idx = cur_marked.find(_M1)
        before = _finish(cur_marked[:idx])
        after = _finish(cur_marked[idx:])
        cur_marked = _window(cur_marked, ASSERTION_CAP)
        current = _finish(cur_marked)
        if _nonspace(before) < MIN_ASSERTION_CHARS and _nonspace(after) < MIN_ASSERTION_CHARS and i > 0:
            pa, pb = self.sents[i - 1]
            prev = _finish(_clean(self._marked(pa, pb, None)))
            room = ASSERTION_CAP - len(current) - 1
            if prev and room >= 20:
                if len(prev) > room:
                    prev = prev[len(prev) - room:]
                    sp = prev.find(" ")
                    if 0 <= sp < 30:
                        prev = prev[sp + 1:]
                current = (prev + " " + current).strip()
        return current[:ASSERTION_CAP].strip()


def assertion_for(text: str, claim: Claim, claims: Sequence[Claim] = ()) -> str:
    """The assertion the author attaches to `claim` (a URL claim) in `text`."""
    return _Doc(text, list(claims) + [claim]).assertion(claim)


# --------------------------------------------------------------------------
# is there anything worth checking?
# --------------------------------------------------------------------------

_WORD = re.compile(r"[^\W\d_][\w'’\-]*")


def is_checkable(assertion: str) -> bool:
    """Long enough and containing a digit, a capitalised word that is not the
    first of its sentence, or a Thai passage of at least 20 characters."""
    if _nonspace(assertion) < MIN_ASSERTION_CHARS:
        return False
    if any(ch.isdigit() for ch in assertion):
        return True
    first = True
    for m in _WORD.finditer(assertion):
        tok = m.group(0)
        if first:
            first = False
            continue
        if len(tok) >= 2 and tok[0].isupper() and not re.search(r"[.!?。]\s*$", assertion[:m.start()]):
            return True
    return sum(1 for ch in assertion if "฀" <= ch <= "๿") >= MIN_THAI_CHARS


# --------------------------------------------------------------------------
# refine
# --------------------------------------------------------------------------

def _cap_quote(q: Any) -> str:
    q = " ".join(str(q).split())
    return q if len(q) <= QUOTE_CAP else q[:QUOTE_CAP - 1].rstrip() + "…"


def _outcome(j: Any) -> Optional[tuple[Status, str]]:
    verdict = getattr(j, "verdict", None)
    quote = _cap_quote(getattr(j, "quote", "") or "")
    if verdict == "supports" and quote:
        return Status.VERIFIED, "The page says: “%s”" % quote
    if verdict == "not_stated":
        return Status.UNVERIFIED, NOT_STATED_REASON
    if verdict == "contradicts" and quote:
        return Status.CONTRADICTED, "The page says: “%s” — this differs from what the answer claims." % quote
    return None                                            # "error", junk, or a verdict without its quote


def _is_candidate(f: Any) -> bool:
    c = getattr(f, "claim", None)
    if not (getattr(f, "module", None) == "citations" and c is not None
            and getattr(c, "kind", None) == ClaimKind.URL
            and getattr(f, "status", None) == Status.VERIFIED):
        return False
    try:
        u = c.data.get("url")
    except Exception:
        return False
    return isinstance(u, str) and bool(u.strip())


def refine(text: str, findings: Sequence[Finding], *, llm: Any, fetcher: Any,
           max_checks: int = 10,
           parse_page_fn: Optional[Callable[..., Any]] = None,
           judge_fn: Optional[Callable[..., Any]] = None) -> list[Finding]:
    """Replace VERIFIED "the link opens" findings by "the page says / does not say".

    Same length and order as `findings`; non-candidates are the same objects.
    """
    out = list(findings)
    if not isinstance(text, str) or not text or max_checks <= 0:
        return out
    cand = [i for i, f in enumerate(out) if _is_candidate(f)]
    if not cand:
        return out
    try:
        if parse_page_fn is None:
            from .pagetext import parse_page as parse_page_fn
        if judge_fn is None:
            from .judge import judge as judge_fn
        doc = _Doc(text, [f.claim for f in out if getattr(f, "claim", None) is not None
                          and isinstance(getattr(f.claim, "start", None), int)
                          and isinstance(getattr(f.claim, "end", None), int)])
    except Exception:
        return out

    decided: dict[tuple[str, str], Optional[tuple[Status, str]]] = {}
    pages: dict[str, Any] = {}
    checks = 0
    for i in cand:
        f = out[i]
        claim = f.claim
        url = claim.data["url"].strip()
        try:
            assertion = doc.assertion(claim)
        except Exception:
            continue
        if not is_checkable(assertion):
            continue
        key = (url, assertion)
        if key in decided:
            res = decided[key]
        else:
            if checks >= max_checks:
                continue
            checks += 1
            try:
                if url not in pages:
                    try:
                        result = fetcher.fetch(url, max_bytes=MAX_PAGE_BYTES)
                        pages[url] = parse_page_fn(result)
                    except Exception:
                        pages[url] = None
                page = pages[url]
                res = None if page is None else _outcome(judge_fn(assertion, page, llm))
            except LLMError:
                break                                      # budget gone: leave the rest as they are
            except Exception:
                res = None
            decided[key] = res
        if res is not None:
            out[i] = Finding(claim, res[0], MODULE, res[1], (url,))
    return out
