"""Rule-based claim extraction: answer text -> [Claim].

Purely regex / state machine. No AI, no network. Emits URL, DOI, REFERENCE,
ARITHMETIC and CODE claims (QUOTE and VAGUE belong to other modules).

Design notes
------------
* Offsets are Python str indices into the ORIGINAL text. Every working copy of
  the text (fences masked, Thai digits normalised, URLs masked...) is
  length-preserving, so a match position in a copy is a position in the source,
  and ``claim.text == source[claim.start:claim.end]`` always holds.
* Masking is how overlap is avoided: fenced code is blanked before anything
  else looks at the text; inline code is blanked before URL/DOI/reference
  scanning; URL and DOI spans are blanked before arithmetic/reference scanning.
* Precision over recall: when a shape is ambiguous, nothing is emitted.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from .models import Claim, ClaimKind

__all__ = ["extract"]

_THAI = "฀-๿"
_THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")

# (start, end, kind, data)
_Cand = tuple[int, int, ClaimKind, dict]


def _mask(s: str, spans) -> str:
    """Blank out spans with spaces (newlines kept), preserving length."""
    if not spans:
        return s
    chars = list(s)
    for a, b in spans:
        for i in range(a, b):
            if chars[i] not in "\r\n":
                chars[i] = " "
    return "".join(chars)


def _overlaps(a: int, b: int, spans) -> bool:
    return any(a < y and x < b for x, y in spans)


# --------------------------------------------------------------------------
# CODE
# --------------------------------------------------------------------------

_FENCE_OPEN = re.compile(r"^([ \t]*)(`{3,}|~{3,})[ \t]*([^\r\n]*)$")
_OUTPUT_LANGS = {"output", "text", "console", "stdout", "result"}
_OUTPUT_LABEL = re.compile(
    r"^[ \t]*(?:#{1,6}[ \t]*)?\**[ \t]*(?:expected[ \t]+output|output|ผลลัพธ์(?:ที่ได้)?)[ \t]*\**[ \t]*(?:[:：]\**[ \t]*.*)?$",
    re.I,
)


def _split_lines(text: str):
    """[(start, content_end, next_start)] handling \\n, \\r\\n and \\r."""
    out = []
    pos, n = 0, len(text)
    while pos < n:
        m = re.compile(r"[^\r\n]*").match(text, pos)
        end = m.end()
        if text.startswith("\r\n", end):
            nxt = end + 2
        elif end < n:
            nxt = end + 1
        else:
            nxt = end
        out.append((pos, end, nxt))
        pos = nxt
    return out


def _fence_info(text: str, lines, i: int):
    """If line i opens a fence return (indent, marker, info) else None."""
    s, e, _ = lines[i]
    m = _FENCE_OPEN.match(text[s:e])
    if not m:
        return None
    indent, marker, info = m.group(1), m.group(2), m.group(3)
    if marker[0] == "`" and "`" in info:
        return None  # ```inline``` on one line is inline code, not a fence
    return indent, marker, info


def _parse_block(text: str, lines, i: int):
    """Parse the fenced block opening at line i.

    Returns (start, end, language, info, body, next_line_index) or None.
    """
    fi = _fence_info(text, lines, i)
    if fi is None:
        return None
    indent, marker, info = fi
    ch, n = marker[0], len(marker)
    close_re = re.compile(r"^[ \t]*" + re.escape(ch) + "{" + str(n) + r",}[ \t]*$")
    start = lines[i][0] + len(indent)
    body: list[str] = []
    j = i + 1
    end = None
    while j < len(lines):
        s, e, _ = lines[j]
        raw = text[s:e]
        if close_re.match(raw):
            end = s + len(raw.rstrip())
            j += 1
            break
        # strip up to len(indent) columns of indentation, like CommonMark
        k = 0
        while k < len(indent) and k < len(raw) and raw[k] in " \t":
            k += 1
        body.append(raw[k:])
        j += 1
    if end is None:  # unterminated: runs to end of text
        last = len(lines) - 1
        # trim trailing blank lines from the span
        while last > i and not text[lines[last][0]:lines[last][1]].strip():
            last -= 1
        end = lines[last][1]
        # drop the blank lines we trimmed from the body too
        keep = last - i
        body = body[:keep]
        j = len(lines)
        while body and not body[-1].strip():
            body.pop()
    lang_tok = info.split()[0] if info.split() else ""
    language = lang_tok.strip("{}.").lower()
    return start, end, language, info, "\n".join(body), j


def _extract_code(text: str):
    """Returns (claims, fence_spans). fence_spans covers every fenced block,
    including output blocks consumed as expected_output."""
    lines = _split_lines(text)
    claims: list[_Cand] = []
    spans: list[tuple[int, int]] = []
    i = 0
    while i < len(lines):
        blk = _parse_block(text, lines, i)
        if blk is None:
            i += 1
            continue
        start, end, language, _info, code, nxt = blk
        spans.append((start, end))
        expected: Optional[str] = None
        # look ahead past blank lines
        k = nxt
        while k < len(lines) and not text[lines[k][0]:lines[k][1]].strip():
            k += 1
        if k < len(lines):
            nb = _parse_block(text, lines, k) if _fence_info(text, lines, k) else None
            if nb is not None and nb[2] in _OUTPUT_LANGS:
                expected = nb[4]
                spans.append((nb[0], nb[1]))
                nxt = nb[5]
            elif nb is None and _OUTPUT_LABEL.match(text[lines[k][0]:lines[k][1]]):
                k2 = k + 1
                while k2 < len(lines) and not text[lines[k2][0]:lines[k2][1]].strip():
                    k2 += 1
                if k2 < len(lines) and _fence_info(text, lines, k2):
                    nb = _parse_block(text, lines, k2)
                    if nb is not None:
                        expected = nb[4]
                        spans.append((nb[0], nb[1]))
                        nxt = nb[5]
        claims.append((start, end, ClaimKind.CODE,
                       {"language": language, "code": code, "expected_output": expected}))
        i = nxt
    return claims, spans


# --------------------------------------------------------------------------
# inline code
# --------------------------------------------------------------------------

_INLINE_CODE = re.compile(r"(?<!`)(`+)(?!`)([^\r\n]+?)(?<!`)\1(?!`)")


def _inline_code_spans(s: str):
    return [m.span() for m in _INLINE_CODE.finditer(s)]


# --------------------------------------------------------------------------
# URL / DOI
# --------------------------------------------------------------------------

_URL_RE = re.compile(
    r"(?<![A-Za-z0-9_.@/-])(?:https?://|www\.)"
    r"[^\s<>" + _THAI + r"⺀-鿿＀-￯‘-”«»]+",
    re.I,
)
_OPENERS = {")": "(", "]": "[", "}": "{"}
_DOI_URL_RE = re.compile(r"^(?:https?://|www\.)(?:www\.)?(?:dx\.)?doi\.org/(10\.\d{4,9}/\S+)$", re.I)
_DOI_RE = re.compile(
    r"(?P<prefix>(?:(?<![A-Za-z0-9])doi[ \t]*:[ \t]*)?(?:(?:https?://)?(?:dx\.)?doi\.org/)?)"
    r"(?<![\w.])(?P<doi>10\.\d{4,9}/[^\s<>" + _THAI + r"\"“”‘’]+)",
    re.I,
)


def _strip_trailing(s: str) -> str:
    while s:
        c = s[-1]
        if c in ".,;:!?'\"*":
            s = s[:-1]
        elif c in _OPENERS and s.count(c) > s.count(_OPENERS[c]):
            s = s[:-1]
        else:
            break
    return s


def _valid_url(u: str) -> bool:
    if u[:4].lower() == "www.":
        return re.match(r"www\.[A-Za-z0-9-]+\.[A-Za-z]{2,}", u, re.I) is not None
    m = re.match(r"https?://([^\s/?#]+)", u, re.I)
    return bool(m) and bool(re.match(r"[A-Za-z0-9\[]", m.group(1)))


def _extract_urls(s: str):
    """s: text with code masked. Returns (url claims, doi claims, spans)."""
    out: list[_Cand] = []
    spans: list[tuple[int, int]] = []
    pos = 0
    while True:
        m = _URL_RE.search(s, pos)
        if not m:
            break
        a = m.start()
        raw = m.group(0)
        cut = raw.find("](")
        skip_link_text = False
        if cut != -1:
            raw = raw[:cut]
            nxt_pos = a + cut + 2
            pos_after = nxt_pos
            # [https://x](https://x): the first is just link text, skip it
            tail = _strip_trailing(raw)
            if tail and s.startswith(tail, nxt_pos):
                skip_link_text = True
        else:
            pos_after = m.end()
        url = _strip_trailing(raw)
        pos = max(pos_after, a + 1)
        if not url or not _valid_url(url) or skip_link_text:
            continue
        b = a + len(url)
        dm = _DOI_URL_RE.match(url)
        if dm:
            doi = _strip_trailing(dm.group(1)).lower()
            out.append((a, b, ClaimKind.DOI, {"doi": doi}))
        else:
            data_url = url if url[:4].lower() != "www." else "https://" + url
            out.append((a, b, ClaimKind.URL, {"url": data_url}))
        spans.append((a, b))
    return out, spans


def _extract_dois(s: str):
    """s: text with code AND urls masked."""
    out: list[_Cand] = []
    spans: list[tuple[int, int]] = []
    for m in _DOI_RE.finditer(s):
        doi = _strip_trailing(m.group("doi"))
        if not re.match(r"10\.\d{4,9}/.+", doi):
            continue
        a = m.start()
        b = m.start("doi") + len(doi)
        out.append((a, b, ClaimKind.DOI, {"doi": doi.lower()}))
        spans.append((a, b))
    return out, spans


# --------------------------------------------------------------------------
# ARITHMETIC
# --------------------------------------------------------------------------

_NUM = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
_TOK_RE = re.compile(
    r"(?P<num>(?:" + _NUM + r")%?)|(?P<op>[+\-−×xX*/÷^])|(?P<lp>\()|(?P<rp>\))|(?P<ws>[ \t ]+)"
)
_WALK = set("0123456789.,%+-−×xX*/÷^() \t ")
_OPMAP = {"−": "-", "×": "*", "x": "*", "X": "*", "÷": "/"}
_RHS_RE = re.compile(r"([-−]?)(" + _NUM + r")(%?)")
_RHS_TAIL = re.compile(r"[ \t ]*[+\-−×xX*/÷^][ \t ]*[\d(]")
_DATEISH = re.compile(r"^(?:\d{4}-\d{1,2}-\d{1,2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})$")
_PCT_OF = re.compile(
    r"(?<![A-Za-z0-9_.,])(?P<n>" + _NUM + r")[ \t]*%[ \t]*"
    r"(?:(?<![A-Za-z])of(?![A-Za-z])|ของ)[ \t]*(?P<m>" + _NUM + r")[ \t]*"
    r"(?:(?<![A-Za-z])(?:is|equals?)(?![A-Za-z])|=|คือ|เท่ากับ)[ \t]*(?P<k>" + _NUM + r")(?P<pct>%)?"
    r"(?![A-Za-z0-9_])",
    re.I,
)


def _parses(tokens) -> bool:
    n = len(tokens)
    ops = [0]

    def term(i):
        if i >= n:
            return None
        k = tokens[i][0]
        if k == "num":
            return i + 1
        if k == "lp":
            j = expr(i + 1)
            if j is None or j >= n or tokens[j][0] != "rp":
                return None
            return j + 1
        return None

    def expr(i):
        j = term(i)
        if j is None:
            return None
        while j < n and tokens[j][0] == "op":
            ops[0] += 1
            j2 = term(j + 1)
            if j2 is None:
                return None
            j = j2
        return j

    return expr(0) == n and ops[0] >= 1


def _normalise(tokens) -> str:
    out = ""
    for kind, val, *_ in tokens:
        if kind == "rp":
            out += ")"
            continue
        if kind == "lp":
            v = "("
        elif kind == "num":
            v = val.replace(",", "")
        else:
            v = _OPMAP.get(val, val)
        out += ("" if out == "" or out.endswith("(") else " ") + v
    return out


def _extract_arith(a: str):
    """a: text with fenced code / urls / dois masked and Thai digits normalised."""
    out: list[_Cand] = []
    taken: list[tuple[int, int]] = []

    # verbal "N% of M is K"
    for m in _PCT_OF.finditer(a):
        expr = f"{m.group('n').replace(',', '')}% * {m.group('m').replace(',', '')}"
        claimed = m.group("k").replace(",", "") + (m.group("pct") or "")
        out.append((m.start(), m.end(), ClaimKind.ARITHMETIC,
                    {"expression": expr, "claimed": claimed}))
        taken.append(m.span())
    a = _mask(a, taken)

    n = len(a)
    for m in re.finditer("=", a):
        i = m.start()
        if i > 0 and a[i - 1] in "=<>!:≠+-*/%&|^":
            continue
        if i + 1 < n and a[i + 1] in "=>":
            continue
        # ---- right-hand side
        j = i + 1
        while j < n and a[j] in " \t *_":
            j += 1
        rm = _RHS_RE.match(a, j)
        if not rm:
            continue
        rend = rm.end()
        if rend < n and (a[rend].isascii() and a[rend].isalpha() or a[rend] == "_"):
            continue
        if rend + 1 < n and a[rend] in ".," and a[rend + 1].isdigit():
            continue
        if _RHS_TAIL.match(a, rend):
            continue  # the right side is itself an expression
        claimed = (rm.group(1).replace("−", "-") + rm.group(2).replace(",", "") + rm.group(3))
        # ---- left-hand side: walk back over expression characters
        k = i
        lim = max(0, i - 300)
        while k > lim and a[k - 1] in _WALK:
            k -= 1
        region = a[k:i]
        toks: list[tuple] = []
        pos = 0
        while pos < len(region):
            mt = _TOK_RE.match(region, pos)
            if not mt:
                toks = []
                pos += 1
                continue
            kind = mt.lastgroup
            if kind != "ws":
                toks.append((kind, mt.group(), k + mt.start(), k + mt.end()))
            pos = mt.end()
        if not toks:
            continue
        pick = None
        for s0 in range(len(toks)):
            if _parses([(t[0], t[1]) for t in toks[s0:]]):
                pick = s0
                break
        if pick is None:
            continue
        dropped = toks[:pick]
        start = toks[pick][2]
        if dropped:
            rest = [t for t in dropped if t[1] != "*"]
            ok = False
            if not rest:
                ok = True
            elif len(rest) == 1 and rest[0][1] in "-+−" and rest[0][3] < n and a[rest[0][3]] in " \t ":
                ok = True  # list bullet "- 5 + 3 = 8"
            elif (len(rest) == 1 and rest[0][1] in "xX" and rest[0][2] > 0
                  and a[rest[0][2] - 1].isascii() and a[rest[0][2] - 1].isalpha()):
                ok = True  # the x of a word like "box"/"max"
            if not ok:
                continue
        if start > 0:
            prev = a[start - 1]
            if prev.isascii() and (prev.isalnum() or prev in "._,-+−()/^%×÷"):
                continue
        lhs_end = toks[-1][3]
        if _DATEISH.match(a[start:lhs_end].replace(" ", "")):
            continue
        end = rm.end()
        out.append((start, end, ClaimKind.ARITHMETIC,
                    {"expression": _normalise(toks[pick:]), "claimed": claimed}))
    return out


# --------------------------------------------------------------------------
# REFERENCE
# --------------------------------------------------------------------------

_YEAR = r"(?:19\d{2}|20\d{2}|2[45]\d{2})"
_YEAR_FIND = re.compile(r"(?<!\d)(" + _YEAR + r")(?!\d)")
_MARK_RE = re.compile(r"^[ \t>]*(?P<bul>(?:[-*+•–]|\d{1,3}[.)]|\[\d{1,3}\])[ \t]+)?[*_]*")
_HEADING_REFS = re.compile(
    r"(references?|bibliography|sources?|citations?|works[ \t]+cited|further[ \t]+reading|"
    r"เอกสารอ้างอิง|อ้างอิง|บรรณานุกรม|แหล่งที่มา|แหล่งอ้างอิง)",
    re.I,
)
_TITLE_PAT = (
    r"(?:\"(?P<t1>[^\"\r\n]{3,200})\""
    r"|“(?P<t2>[^”\r\n]{3,200})”"
    r"|(?<![*\w])\*(?!\*)(?P<t3>[^*\r\n]{3,200}?)(?<!\*)\*(?![*\w])"
    r"|(?<![\w_])_(?P<t4>[^_\r\n]{3,200}?)_(?![\w_]))"
)
_TITLE_RE = re.compile(_TITLE_PAT)
_NAME = r"[A-Z][\w'’\-]+"
_AUTHS = (
    rf"{_NAME}(?:(?:[ \t]*,[ \t]*|[ \t]+(?:&|and)[ \t]+|[ \t]*,[ \t]*(?:&|and)[ \t]+){_NAME})*"
    r"(?:[ \t]+et[ \t]+al\.?)?"
)
_B1 = re.compile(
    rf"(?P<auth>{_AUTHS})[ \t]*\([ \t]*(?P<year>{_YEAR})[a-z]?[ \t]*\)[ \t]*[,:;]?[ \t]*(?P<q>"
    + _TITLE_PAT.replace("t1", "b1").replace("t2", "b2").replace("t3", "b3").replace("t4", "b4") + ")"
)
_B2 = re.compile(
    rf"\([ \t]*(?P<auth>{_AUTHS})[ \t]*,?[ \t]*(?P<year>{_YEAR})[a-z]?[ \t]*\)[ \t]*[,:;]?[ \t]*(?P<q>"
    + _TITLE_PAT.replace("t1", "c1").replace("t2", "c2").replace("t3", "c3").replace("t4", "c4") + ")"
)
_B3 = re.compile(
    r"(?P<q>" + _TITLE_PAT.replace("t1", "d1").replace("t2", "d2").replace("t3", "d3").replace("t4", "d4")
    + rf")[ \t]*\([ \t]*(?P<auth>{_AUTHS})[ \t]*,?[ \t]*(?P<year>{_YEAR})[a-z]?[ \t]*\)"
)
_APA_RE = re.compile(
    r"^(?P<auth>[A-Z" + _THAI + r"][^()\r\n\"“”]{1,200}?)[ \t]*"
    rf"\([ \t]*(?P<year>{_YEAR})(?P<suf>[a-z])?(?:[ \t]*,[ \t]*[A-Za-z]{{3,9}}\.?(?:[ \t]+\d{{1,2}})?)?[ \t]*\)"
    r"(?P<dot>[ \t]*[.,:])?[ \t]*(?P<rest>\S.*)$"
)
_STRONG_AUTH = re.compile(r"[A-Z][\w'’\-]+,[ \t]*[A-Z]\.|\bet[ \t]+al\b")
_SURNAME_INIT = re.compile(
    r"((?:[a-z]{2,4}[ \t]+)?[A-Z][\w'’\-]+(?:[ \t]+[A-Z][\w'’\-]+)*)[ \t]*,[ \t]*[A-Z]\."
)
_TITLE_END = re.compile(r"(?<!\b[A-Z])(?<!\bet al)(?<!\bvs)(?<!\bNo)(?<!\bpp)(?<!\bVol)[.?!](?=\s|$)")


def _auth_script_ok(auth: str) -> bool:
    """Thai authors must be pure Thai and short (Thai has no spaces, so a long
    run is prose, not a name); Latin authors must not contain Thai."""
    has_thai = re.search("[" + _THAI + "]", auth) is not None
    has_latin = re.search(r"[A-Za-z]", auth) is not None
    if has_thai and has_latin:
        return False
    if has_thai:
        return len(auth) <= 40 and _nwords(auth) <= 4
    return True


def _year_ce(s: str) -> int:
    y = int(s)
    return y - 543 if y >= 2400 else y


def _clean_title(t: str) -> str:
    return t.strip().strip("*_\"“”'‘’.,;: \t").strip()


def _nwords(t: str) -> int:
    return len(t.split())


def _surnames(auth: str) -> list[str]:
    auth = auth.strip(" \t*_,;") + " "
    if _SURNAME_INIT.search(auth):
        names = [m.group(1).strip() for m in _SURNAME_INIT.finditer(auth)]
    else:
        a = re.sub(r"\bet[ \t]+al\.?", "", auth)
        names = []
        for chunk in re.split(r"[,;&]|\band\b|และ", a):
            chunk = chunk.strip(" \t.*_")
            if not chunk:
                continue
            m = re.match(r"^(?:[A-Z]\.[ \t]*)+(.+)$", chunk)
            if m:
                chunk = m.group(1).strip()
            names.append(chunk)
    seen: set[str] = set()
    out = []
    for n in names:
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out


def _first_group(m: re.Match, prefix: str) -> Optional[str]:
    for i in range(1, 5):
        g = m.group(f"{prefix}{i}")
        if g is not None:
            return g
    return None


def _extract_refs(line_src: str, r: str, line_ranges):
    """line_src: text with only fences/inline code masked (for line extents);
    r: same with urls/dois masked and Thai digits normalised (for matching)."""
    out: list[_Cand] = []
    in_refs = False
    for (ls, le) in line_ranges:
        line = r[ls:le]
        stripped = line.strip()
        if not stripped:
            continue
        # heading / "References:" label detection
        hm = re.match(r"^[ \t]{0,3}#{1,6}[ \t]*(.*)$", line)
        label_c = (hm.group(1) if hm else stripped).strip(" \t*_:：#")
        is_label = bool(hm) or (len(label_c) <= 40 and (
            stripped.rstrip("*_ \t").endswith((":", "："))
            or (stripped.startswith("**") and stripped.endswith("**"))))
        if is_label:
            if len(label_c) <= 40 and _HEADING_REFS.search(label_c):
                in_refs = True
                continue
            if hm:
                in_refs = False
                continue
        if hm:
            continue
        mk = _MARK_RE.match(line)
        bs = mk.end()
        has_marker = mk.group("bul") is not None
        lead_marks = bool(re.search(r"[*_]", line[mk.start():bs]))
        body = line[bs:]
        if not body.strip():
            continue
        raw = line_src[ls:le]
        end = le
        while end > ls and raw[end - ls - 1] in " \t":
            end -= 1
        if lead_marks:
            while end > ls + bs and raw[end - ls - 1] in "*_ \t":
                end -= 1
        span_start = ls + bs

        # ---- (a) APA-style line
        am = _APA_RE.match(body)
        if am:
            auth = am.group("auth").strip(" \t*_,")
            rest = am.group("rest")
            rest_off = bs + am.start("rest")
            ok = (
                _nwords(auth) <= 25
                and not re.search(r"\d", auth)
                and not re.search(r"[:;]", auth)
                and not re.search(r"[a-z]{2}[.!?][ \t]+[A-Z]", auth)
                and _auth_script_ok(auth)
                and re.match(r"[A-Z" + _THAI + r"0-9\"“*_(]", rest) is not None
            )
            if ok:
                tm = _TITLE_RE.match(rest)
                if tm:
                    title = next(g for g in tm.groups() if g)
                    fancy = True
                else:
                    em = _TITLE_END.search(rest)
                    if em:
                        title = rest[: em.end()] if rest[em.start()] in "?!" else rest[: em.start()]
                    else:
                        title = rest
                    fancy = False
                title = _clean_title(title)
                strong = _STRONG_AUTH.search(auth) is not None
                has_dot = bool(am.group("dot") and "." in am.group("dot"))
                title_ok = (
                    len(title) >= 3
                    and (_nwords(title) >= 2 or fancy or strong
                         or re.search("[" + _THAI + "]", title) is not None and len(title) >= 6)
                )
                bare_ok = strong or ((has_marker or in_refs or fancy) and has_dot)
                if title_ok and bare_ok:
                    out.append((span_start, end, ClaimKind.REFERENCE,
                                {"title": title, "authors": _surnames(auth),
                                 "year": _year_ce(am.group("year"))}))
                    continue

        # ---- (c) bullet / numbered line with a quoted or italic title and a year
        if has_marker:
            done = False
            for tm in _TITLE_RE.finditer(body):
                title = _clean_title(next(g for g in tm.groups() if g))
                if _nwords(title) < 2 or title.endswith((":", "：")) or re.search(r"[:：]\s*$", tm.group(0)[1:-1]):
                    continue
                ym = _YEAR_FIND.search(body)
                if not ym:
                    break
                prefix = body[: tm.start()].strip()
                if prefix:
                    if not re.match(r"[A-Z" + _THAI + r"]", prefix) or len(prefix) > 160:
                        break
                    signal = (re.search(r"\b[A-Z]\.", prefix) or re.search(r"\bet[ \t]+al\b", prefix)
                              or "&" in prefix or _YEAR_FIND.search(prefix)
                              or prefix[-1] in ",.:;(–—-")
                    if not signal:
                        break
                auth_part = re.sub(r"\(?[ \t]*" + _YEAR + r"[a-z]?[ \t]*\)?", "", prefix)
                out.append((span_start, end, ClaimKind.REFERENCE,
                            {"title": title, "authors": _surnames(auth_part) if auth_part.strip(" \t,.;:()") else [],
                             "year": _year_ce(ym.group(1))}))
                done = True
                break
            if done:
                continue

        # ---- (b) inline cites with an attached title
        taken: list[tuple[int, int]] = []
        cands = []
        for rx, pre in ((_B1, "b"), (_B2, "c"), (_B3, "d")):
            for bm in rx.finditer(line):
                title = _clean_title(_first_group(bm, pre) or "")
                if len(title) < 3:
                    continue
                s0, e0 = bm.start(), bm.end()
                cands.append((s0, e0, title, bm.group("auth"), bm.group("year")))
        cands.sort(key=lambda c: (c[0], -c[1]))
        for s0, e0, title, auth, year in cands:
            if _overlaps(s0, e0, taken):
                continue
            taken.append((s0, e0))
            out.append((ls + s0, ls + e0, ClaimKind.REFERENCE,
                        {"title": title, "authors": _surnames(auth), "year": _year_ce(year)}))
    return out


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def extract(text: str) -> list[Claim]:
    """Split an answer into Claims (URL, DOI, REFERENCE, ARITHMETIC, CODE)."""
    if not text:
        return []

    cands: list[_Cand] = []

    code_claims, fence_spans = _extract_code(text)
    cands += code_claims

    fenced = _mask(text, fence_spans)                       # code blocks blanked
    inline = _mask(fenced, _inline_code_spans(fenced))      # + inline code blanked

    url_claims, url_spans = _extract_urls(inline)
    cands += url_claims
    after_urls = _mask(inline, url_spans)
    doi_claims, doi_spans = _extract_dois(after_urls)
    cands += doi_claims
    cite_spans = url_spans + doi_spans

    # arithmetic: inline code is allowed (explicit equations), fences/urls are not
    arith_src = _mask(fenced, cite_spans).translate(_THAI_DIGITS)
    cands += _extract_arith(arith_src)

    # references: line-based over text with code blanked
    ref_src = _mask(inline, cite_spans).translate(_THAI_DIGITS)
    line_ranges = [(s, e) for s, e, _ in _split_lines(ref_src) if e > s]
    cands += _extract_refs(inline, ref_src, line_ranges)

    cands.sort(key=lambda c: (c[0], c[1]))
    claims = []
    for n, (a, b, kind, data) in enumerate(cands, 1):
        claims.append(Claim(id=f"c{n}", kind=kind, text=text[a:b], start=a, end=b, data=data))
    return claims
