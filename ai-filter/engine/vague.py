"""Vague-authority scanner: flags *unsourced appeals to authority* and
suspiciously precise unsourced statistics.

    scan(text) -> list[Finding]

This module can NOT know whether a statement is true, so it NEVER returns
VERIFIED or CONTRADICTED.  Every Finding is UNVERIFIED with module="vague": it
only says "this needs a source".  Phrases that are *backed* (see below) yield
nothing at all.

How it works
------------
1. Fenced code blocks and inline code are blanked out (offsets preserved) and
   take no part in anything: no triggers, no backing, no sentence boundaries.
2. The blanked text is NFC-normalised for matching only; an offset map takes
   every match back to the ORIGINAL string, so `text[c.start:c.end] == c.text`.
3. Triggers come from the PHRASE TABLE below (extend it!) plus the
   "suspicious precision" number rules.  Overlaps resolve leftmost-longest, so
   "many experts say" is ONE finding, not two.
4. A trigger is dropped if it is BACKED (BACKING TABLE below).

Backing-window policy
---------------------
A trigger is backed when a backing signal lies inside its WINDOW:
  * the sentence containing the trigger, plus
  * the NEXT sentence,
  * additionally capped to 120 chars before the trigger start and 300 chars
    after the trigger end (so an unpunctuated Thai paragraph, where words are
    not separated and spaces are not reliable clause boundaries, does not turn
    into one giant "sentence").
Sentence boundaries: `.` `?` `!` (plus closing quotes/brackets) followed by
whitespace or end; full-width `。！？`; and every newline, EXCEPT a newline in
the middle of a hard-wrapped line (previous line ends in a word/comma and the
next starts with a lowercase ASCII letter).  Not boundaries: abbreviations
("et al.", "Dr.", "e.g."), single-letter initials, decimals, list numbers
("1."), the Thai abbreviation mark ฯ and repetition mark ๆ, dots inside URLs.
Thai spaces are NOT treated as boundaries (Thai uses spaces inconsistently);
the char caps above do that job instead.  Two sentences later never backs.
"""

from __future__ import annotations

import re
import unicodedata
from bisect import bisect_left, bisect_right

from .models import Claim, ClaimKind, Finding, Status

__all__ = ["scan"]

MODULE = "vague"
_PRE_WINDOW = 120   # chars before the trigger that may back it (same sentence only)
_POST_WINDOW = 300  # chars after the trigger end (same + next sentence only)

# ---------------------------------------------------------------------------
# PHRASE TABLE  --  extend here.
#   (family, language, regex, options)
# * In English patterns a single space means "flexible whitespace" (spaces,
#   tabs, nbsp, at most one line break).  Do not put a space inside [...].
# * No capturing groups inside a pattern (the table is compiled into one big
#   alternation and the matching row is found via m.lastindex).
# * English rows are case-insensitive and word-bounded automatically; Thai
#   rows are plain substrings (Thai has no word separators) - guard false
#   positives with lookbehinds such as (?<!ไม่มี) ("there is no ...").
# * options: "nofig"  skip if a digit follows in the same sentence (the claim
#                     then carries its own figure: "data shows sales rose 12%")
#            "det"    skip if preceded by the/this/our/... ("the research
#                     shows" usually points back at something already named)
# ---------------------------------------------------------------------------
_TH = r"[ก-๎]"

_PHRASES: list[tuple[str, str, str, tuple[str, ...]]] = [
    # --- English: studies / research ------------------------------------
    # "studies show", "recent research suggests", "surveys indicate", ...
    ("study", "en",
     r"(?:(?:recent|new|several|many|numerous|multiple|various|some|countless|scientific|clinical) )*"
     r"(?:studies|research|surveys|trials) "
     r"(?:show|shows|have shown|has shown|suggest|suggests|indicate|indicates|find|finds|found|have found|"
     r"has found|reveal|reveals|have revealed|demonstrate|demonstrates|confirm|confirms)",
     ("det",)),
    # "a study found", "a recent survey showed", "a 2019 study found" (a year is not a source)
    ("study", "en",
     r"(?:a|one|another) "
     r"(?:(?:recent|new|large|small|major|landmark|famous|well-known|(?:19|20)[0-9]{2}|national|international|"
     r"clinical|scientific|peer-reviewed|systematic|meta) )*"
     r"(?:study|survey|analysis|meta-analysis|trial|experiment|review|report) "
     r"(?:found|shows|showed|suggests|suggested|indicates|indicated|revealed|reveals|has shown|has found|"
     r"demonstrated|concluded|confirmed)",
     ()),
    # "evidence suggests", "growing evidence indicates"
    ("study", "en",
     r"(?:(?:scientific|empirical|growing|mounting|emerging|clinical|strong|some|substantial) )?"
     r"evidence (?:suggests|indicates|shows|points to)",
     ("det",)),
    # "according to studies / research / surveys / statistics"
    ("study", "en",
     r"according to (?:(?:some|many|several|recent|various|most|multiple|numerous|certain|leading|other) )?"
     r"(?:studies|research|surveys|statistics)",
     ()),
    # "according to reports / sources"
    ("report", "en",
     r"according to (?:(?:some|many|several|various|multiple|numerous|unnamed|other) )?(?:sources|reports)",
     ()),
    # "according to experts / scientists / some experts"
    ("expert", "en",
     r"according to (?:(?:some|many|several|recent|various|most|multiple|numerous|certain|leading|other) )?"
     r"(?:experts|scientists|researchers|doctors|analysts|economists|specialists|scholars|historians)",
     ()),
    # --- English: experts / scientists -----------------------------------
    # "experts say", "many experts agree", "scientists have found", "doctors recommend"
    ("expert", "en",
     r"(?:(?:many|some|most|several|leading|top|numerous|various|certain|other) )?"
     r"(?:experts|scientists|researchers|doctors|physicians|economists|analysts|nutritionists|dietitians|"
     r"specialists|historians|scholars) "
     r"(?:(?:widely|generally|often|commonly|all) )?"
     r"(?:say|agree|believe|recommend|suggest|warn|claim|think|advise|argue|maintain|"
     r"have found|have shown|have said|have concluded)",
     ()),
    # --- English: common belief ------------------------------------------
    # "it is widely known", "it's generally accepted", "it is well known", "common knowledge"
    ("belief", "en",
     r"it(?: is|['’]s) (?:(?:widely|commonly|generally|universally|broadly) "
     r"(?:known|believed|accepted|established|recognized|recognised|understood|agreed|documented|"
     r"acknowledged|held|thought)|well(?:-| )(?:known|established|documented|accepted|understood)|"
     r"common knowledge)",
     ()),
    # "many people believe", "most people think"
    ("belief", "en",
     r"(?:many|most|countless|numerous) (?:people|folks) (?:believe|say|think|claim|assume)",
     ()),
    # --- English: bare statistics / data ---------------------------------
    # "statistics show"
    ("stats", "en",
     r"(?:statistics|stats) (?:show|shows|indicate|indicates|suggest|suggests|reveal|reveals)",
     ()),
    # "data shows / suggests" - only when no figure follows in the sentence
    ("data", "en",
     r"(?:data|figures) (?:show|shows|suggest|suggests|indicate|indicates|reveal|reveals)",
     ("nofig", "det")),
    # --- English: "proven" -----------------------------------------------
    ("proven", "en",
     r"it has been (?:(?:scientifically|clinically|mathematically) )?"
     r"(?:proven|proved|shown|demonstrated|established)(?: that)?",
     ()),
    ("proven", "en",
     r"(?:is|are|has been|have been) (?:(?:scientifically|clinically|medically) )?(?:proven|proved) to",
     ()),
    # --- Thai: studies / research / surveys ------------------------------
    # มีการศึกษาพบว่า, มีงานวิจัยชี้ว่า, งานวิจัยพบว่า, จากการสำรวจพบว่า, ผลสำรวจชี้ว่า ...
    ("study", "th",
     r"(?:มี|จาก)?(?<!ไม่มี)"
     r"(?:ผลการศึกษา|ผลการวิจัย|ผลวิจัย|ผลสำรวจ|การศึกษา|งานวิจัย|การวิจัย|การสำรวจ)"
     r"(?:พบ|ชี้|ระบุ|บ่งชี้|เผย|ยืนยัน|สรุป|แสดงให้เห็น)ว่า",
     ()),
    # นักวิจัยพบว่า, นักวิทยาศาสตร์พบว่า
    ("expert", "th",
     r"(?<!ไม่มี)(?:นักวิจัย|นักวิทยาศาสตร์|นักวิชาการ|ทีมวิจัย)"
     r"(?:พบ|ค้นพบ|ชี้|ระบุ|เชื่อ|เห็น|กล่าว)ว่า",
     ()),
    # --- Thai: experts ------------------------------------------------------
    # ผู้เชี่ยวชาญระบุว่า / กล่าวว่า / แนะนำว่า / เห็นพ้อง, มีผู้เชี่ยวชาญ(บางท่าน)เชื่อว่า
    # (bare "มีผู้เชี่ยวชาญ" is only taken when a verb of saying follows)
    ("expert", "th",
     r"(?:มี)?(?<!ไม่มี)ผู้เชี่ยวชาญ(?:ส่วนใหญ่|หลายคน|หลายท่าน|บางคน|บางท่าน|หลายราย)?"
     r"(?:ด้าน" + _TH + r"{2,25}?)?"
     r"(?:(?:ระบุ|กล่าว|เชื่อ|บอก|ชี้|เตือน|ยืนยัน|เห็น|พบ)ว่า|แนะนำ|เห็นพ้อง|เห็นด้วย)",
     ()),
    # --- Thai: statistics / data / reports ---------------------------------
    ("stats", "th", r"สถิติ(?:ชี้|ระบุ|แสดง(?:ให้เห็น)?|บ่งชี้|พบ|เผย)ว่า", ()),
    ("data", "th",
     r"(?<!ฐาน)(?<!ชุด)(?<!คลัง)ข้อมูล(?:ระบุ|ชี้|แสดง(?:ให้เห็น)?|บ่งชี้)ว่า",
     ("nofig",)),
    ("report", "th", r"(?<!ไม่)มีรายงาน(?:ออกมา)?(?:ว่า|ระบุว่า|ชี้ว่า)", ()),
    # --- Thai: common belief / proven / evidence ---------------------------
    ("belief", "th",
     r"เป็นที่(?:ทราบ|รู้)กันดีว่า|เป็นที่ยอมรับกัน(?:ทั่วไป)?ว่า|ว่ากันว่า|(?:หลายคน|คนส่วนใหญ่)เชื่อว่า",
     ()),
    ("proven", "th",
     r"(?<!ไม่)ได้รับการพิสูจน์(?:ทางวิทยาศาสตร์|ทางการแพทย์)?แล้วว่า",
     ()),
    ("study", "th",
     r"(?<!ไม่)มีหลักฐาน(?:ทางวิทยาศาสตร์|ทางการแพทย์)?(?:ชี้|บ่งชี้|แสดง(?:ให้เห็น)?|ระบุ|พบ)ว่า",
     ()),
]

_REASONS = {
    "study": "Says '{p}' but names no study, author, institution or year — ask for the source.",
    "expert": "Cites '{p}' but names no expert, institution or publication — ask who exactly said it.",
    "report": "Says '{p}' but does not say who reported it — ask for the source.",
    "belief": "Says '{p}' as if it were settled but gives no source — ask for evidence.",
    "stats": "Says '{p}' but names no dataset, survey or institution — ask for the source.",
    "data": "Says '{p}' but does not say which data or who collected it — ask for the source.",
    "proven": "Says '{p}' but cites no proof or source — ask for the source.",
}
_NUMBER_REASON = "Very precise figure ('{p}') with no source — ask where it comes from."

# ---------------------------------------------------------------------------
# BACKING TABLE  --  anything here, inside the window, means "a source is
# named", so the trigger is NOT reported.  (name, regex, case-insensitive?)
# A bare year is deliberately NOT here: "in 2020" backs nothing by itself.
# ---------------------------------------------------------------------------
_YEAR = r"(?:(?:19|20)[0-9]{2}|2[45][0-9]{2})[a-z]?"            # 1900-2099, Thai 2400-2599
_NAME_STOP = (r"(?!(?:In|On|Since|By|Updated|Released|Published|Version|Year|Circa|Around|About|Early|Late|Mid|"
              r"Approx|Est|Founded|Established|Born|Died|Created|Fiscal|FY|Q[1-4]|Between|From|Until|Before|"
              r"After|During|Only|Over|Under|Last|Next|This|That)\b)")
_CAP = r"[A-Z][A-Za-z'’\-]+"
_ACRONYMS = (r"WHO|CDC|NASA|NIH|FDA|IMF|UN|UNESCO|UNICEF|OECD|NHS|EPA|USDA|NOAA|IPCC|EU|ECB|ADB|WTO|ILO|MIT|NSF|"
             r"DARPA|CERN|ESA|BBC|CNN|AMA|AHA|APA|ACS|NEJM|BMJ|JAMA|PNAS|IEEE|ACM|BLS|CBO|SEC")
_NAMED_ORGS = (r"Harvard|Stanford|Oxford|Cambridge|Yale|Princeton|Berkeley|Caltech|Cornell|Johns Hopkins|"
               r"Mayo Clinic|Cleveland Clinic|World Bank|World Health Organization|World Economic Forum|"
               r"Federal Reserve|European Union|United Nations|Pew Research|Gallup|McKinsey|Gartner|Statista|"
               r"Reuters|Bloomberg|The Lancet|Lancet|New England Journal|Nature Medicine|Nature Communications|"
               r"Science Advances|Scientific Reports|PLOS ONE|Cochrane|Eurostat|Brookings|Statistics Bureau")
_THAI_ORGS = (
    r"ธนาคารแห่งประเทศไทย|องค์การอนามัยโลก|ธนาคารโลก|สหประชาชาติ|ราชวิทยาลัย|"
    r"มหาวิทยาลัย|กระทรวง|โรงพยาบาล|"
    r"สถาบัน" + _TH + r"{2,}|"
    r"องค์การ" + _TH + r"{2,}|"
    r"สำนักงาน(?!ใหญ่|สาขา|ทนายความ|บัญชี|ให้เช่า)" + _TH + r"{2,}|"
    # "กรม" is also the tail of โปรแกรม etc.: only at a word start or after a particle
    r"(?:(?<![฀-๿])|(?<=จาก)|(?<=ของ)|(?<=โดย)|(?<=ว่า)|(?<=และ)|(?<=ใน))กรม" + _TH + r"{2,}"
)
_ORG_NOUN = (r"University|Institute|Journal|Organization|Organisation|Agency|Bureau|Foundation|Association|Society|"
             r"Council|Commission|Laboratory|Laboratories|Hospital|Clinic|Academy|Ministry|Department|"
             r"Administration")
_BDRY_L = r"(?<![A-Za-z0-9])"
_BDRY_R = r"(?![A-Za-z0-9])"

_BACKING: list[tuple[str, str, bool]] = [
    # URL, DOI, arXiv
    ("url", r"(?:https?://|www\.)[^\s<>\"')\]]+", True),
    ("doi", r"(?<![0-9A-Za-z])(?:doi:?\s*)?10\.[0-9]{4,9}/[^\s\"<>]+", True),
    ("arxiv", r"arXiv:\s*[0-9]{4}\.[0-9]{4,5}", True),
    # numbered citation [1] [12] [1,2] [3-5], footnote [^1]
    ("numbered", r"\[\^?[0-9]{1,3}(?:\s*[,–\-]\s*[0-9]{1,3})*\]", False),
    # (Smith, 2020)  (Smith et al., 2020)  (Smith & Jones 2020)  (WHO, 2020)  (สมชาย, 2563)
    ("author-year", r"\(\s*" + _NAME_STOP +
     r"[A-Z฀-๿][^()\n0-9]{0,49}?(?:,|\bet al\.?|\b(?:&|and) " + _CAP + r")\s*,?\s*" + _YEAR +
     r"(?:[,;:][^()\n]{0,40})?\s*\)", False),
    ("author-year", r"\(\s*" + _NAME_STOP + _CAP + r"\s+" + _YEAR + r"\s*\)", False),   # (Smith 2020)
    # Smith et al. (2020)   Smith and Jones (2020)   Smith (2020)
    ("author-year", _BDRY_L + _NAME_STOP + _CAP + r"(?:\s+et al\.?|\s+(?:&|and)\s+" + _CAP + r")?\s*\(\s*" +
     _YEAR + r"\s*\)", False),
    # named institutions: case-sensitive acronyms and names
    ("org", _BDRY_L + r"(?:" + _ACRONYMS + r"|" + _NAMED_ORGS + r")" + _BDRY_R, False),
    ("org-th", _THAI_ORGS, False),
    # "in Nature", "published in Science" - only with a preposition, not bare words
    ("journal", r"\b(?:in|by|via|per|journal)\s+(?:the\s+)?(?:journal\s+)?(?:Nature|Science)\b"
                r"(?!\s+(?:of|and|is|are|was|were|class|fair|fiction|teacher|can|will))", False),
    # Capitalised multi-word proper name + University/Institute/...,  "University of X"
    ("org-generic", r"(?:" + _CAP + r"\s+){1,4}(?:" + _ORG_NOUN + r")\b", False),
    ("org-of", r"\b(?:University|Institute|College|Ministry|Department|Bureau|Office|Academy|Journal) of [A-Z][A-Za-z]+",
     False),
    # named person: Dr. Smith, Prof. Jane Doe, ดร.สมชาย, ศ.นพ.สมหมาย, รศ.ดร.สมศรี
    ("person", r"\b(?:Dr|Prof|Professor)\.?\s+[A-Z][A-Za-z'’\-]+", False),
    ("person-th",
     r"(?:(?<![฀-๿.])(?:ศ|รศ|ผศ)|ดร|นพ|พญ|ภก|ภญ)\.(?:ดร\.|นพ\.|พญ\.)?\s*[ก-ฮ]{2,}", False),
]

# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------
_WS = r"(?:[ \t ]+|[ \t]*\r?\n[ \t]*)"


def _n(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def _compile_phrases():
    groups = []
    for _fam, lang, pat, _opt in _PHRASES:
        if lang == "en":
            groups.append("(" + r"(?<![A-Za-z])" + pat.replace(" ", _WS) + r"(?![A-Za-z]))")
        else:
            groups.append("(" + _n(pat) + ")")
    return re.compile("|".join(groups), re.I)


_PHRASE_RE = _compile_phrases()
_BACKING_RES = [re.compile(_n(p), re.I if ignore_case else 0) for _name, p, ignore_case in _BACKING]
_DET_BEFORE = re.compile(r"(?<![A-Za-z])(?:the|this|that|these|those|our|my|your|his|her|its|their|such|which)\s+\Z", re.I)
_AUX_BEFORE = re.compile(
    r"(?<![A-Za-z])(?:do|does|did|can|could|would|should|will|may|might|if|whether|unless|what|which|how|why|"
    r"where|when)\s+\Z", re.I)
_FIGURE = re.compile(r"[0-9๐-๙]")
_URL_MASK = re.compile(r"(?:https?://|ftp://|www\.)\S+")

# precise percentages: 73.42%, 12.345 percent, ร้อยละ 73.42
_PCT = re.compile(
    r"(?<![0-9A-Za-z_.,/:$€£฿])(?<![0-9]-)"
    r"(?:ร้อยละ\s?(?P<th>[0-9]+\.[0-9]{2,})(?![0-9])"
    r"|(?P<en>[0-9]+\.[0-9]{2,})(?![0-9])\s?(?:%|percent\b|per cent\b|เปอร์เซ็นต์|เปอร์เซนต์))")
_COUNT_NOUNS_EN = (r"users|people|customers|patients|participants|respondents|visitors|downloads|views|subscribers|"
                   r"cases|deaths|residents|citizens|employees|students|members|accounts|installs|victims|voters|"
                   r"households|workers|children|women|men|individuals")
_COUNT_NOUNS_TH = r"คน|ราย(?![ก-๎])|ผู้ใช้|ผู้ป่วย|ผู้ติดเชื้อ|ผู้เสียชีวิต|ครั้ง|ดาวน์โหลด|บัญชี"
_COUNT = re.compile(
    r"(?<![0-9A-Za-z_.,/:$€£฿¥₹])(?<![0-9]-)"
    r"(?P<num>(?:[0-9]{1,3}(?:,[0-9]{3}){2,}|[0-9]{7,}))(?![0-9,]|\.[0-9])"
    r"(?P<unit>\s+(?:" + _COUNT_NOUNS_EN + r")\b|\s*(?:" + _COUNT_NOUNS_TH + r"))", re.I)
_PCT_OF = re.compile(r"\s*(?:of|ของ)\s*[0-9$€£฿]", re.I)
_ILLUSTRATIVE = re.compile(
    r"for example|for instance|e\.g\.|\bexample\b|suppose|assum(?:e|ing)|hypothetic|imagine|let's say|"
    r"ตัวอย่าง|สมมติ|สมมุติ", re.I)
_HAS_ALPHA = re.compile(r"[^\W\d_]")
_LISTNUM = re.compile(r"(?:\A|(?<=\n))[ \t>*#\-]*[0-9]{1,3}\Z")
_ABBR_TOKEN = re.compile(r"([A-Za-z.]+)\Z")
_ABBREVIATIONS = {
    "et al", "al", "e.g", "i.e", "vs", "dr", "prof", "mr", "mrs", "ms", "fig", "figs", "approx", "inc", "ltd", "co",
    "jr", "sr", "st", "cf", "ca", "vol", "pp", "ed", "eds", "no", "u.s", "u.k", "etc", "est",
}
_CAND = re.compile(r"[.!?]+[\"'”’)\]]*(?=\s|\Z)|[。！？]+|\n")
_FENCE_OPEN = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
_BACKTICKS = re.compile(r"`+")


# ---------------------------------------------------------------------------
# Masking code, normalising, sentences
# ---------------------------------------------------------------------------
def _blank(s: str) -> str:
    return "".join(c if c in "\r\n" else "\x00" for c in s)


def _mask_inline(line: str) -> str:
    runs = [(m.start(), m.end()) for m in _BACKTICKS.finditer(line)]
    if len(runs) < 2:
        return line
    by_len: dict[int, list[int]] = {}
    for i, (a, b) in enumerate(runs):
        by_len.setdefault(b - a, []).append(i)
    out, last, i = [], 0, 0
    while i < len(runs):
        a, b = runs[i]
        lst = by_len[b - a]
        j = bisect_right(lst, i)
        if j < len(lst):
            k = lst[j]
            out.append(line[last:a])
            out.append("\x00" * (runs[k][1] - a))
            last = runs[k][1]
            i = k + 1
        else:
            i += 1
    out.append(line[last:])
    return "".join(out)


def _mask_code(text: str) -> str:
    """Blank fenced + inline code with NULs, keeping every offset and newline."""
    if "`" not in text and "~~~" not in text:
        return text
    out: list[str] = []
    fence: tuple[str, int] | None = None
    for line in text.split("\n"):
        core = line.rstrip("\r")
        if fence is not None:
            ch, n = fence
            s = core.lstrip(" ")
            if len(core) - len(s) <= 3 and s.startswith(ch * n) and not s.strip(ch).strip():
                fence = None
            out.append(_blank(line))
            continue
        m = _FENCE_OPEN.match(core)
        if m and not (m.group(1)[0] == "`" and "`" in m.group(2)):
            fence = (m.group(1)[0], len(m.group(1)))
            out.append(_blank(line))
            continue
        out.append(_mask_inline(line) if "`" in line else line)
    return "\n".join(out)


def _nfc(s: str):
    """NFC-normalise for matching.  Returns (norm, omap) where omap is None when
    the string was already NFC (identity), else (starts, ends) per norm char."""
    if s.isascii() or unicodedata.is_normalized("NFC", s):
        return s, None
    chars: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    i, n = 0, len(s)
    while i < n:
        j = i + 1
        while j < n and unicodedata.category(s[j])[0] == "M":
            j += 1
        piece = unicodedata.normalize("NFC", s[i:j])
        for ch in piece:
            chars.append(ch)
            starts.append(i)
            ends.append(j)
        i = j
    return "".join(chars), (starts, ends)


def _sentences(t: str) -> list[tuple[int, int]]:
    cuts = [0]
    for m in _CAND.finditer(t):
        a, e = m.start(), m.end()
        if t[a] == "\n":
            before = t[max(0, a - 40):a].rstrip(" \t\r")
            after = t[a + 1:a + 41].lstrip(" \t\r")
            if (before and after and after[0].isascii() and after[0].islower()
                    and (before[-1].isalnum() or before[-1] in ",;")):
                continue                                   # hard-wrapped line, not a boundary
        elif t[a] == "." and t[a + 1:a + 2] != ".":
            if a > 0 and t[a - 1].isdigit():               # list number "1. Item"
                ln = t.rfind("\n", max(0, a - 20), a)
                start = ln + 1 if ln >= 0 else (0 if a - 20 <= 0 else None)
                if start is not None and _LISTNUM.search(t, start, a):
                    continue
            tok = _ABBR_TOKEN.search(t, max(0, a - 10), a)
            if tok:
                raw = tok.group(1).strip(".")
                if raw.lower() in _ABBREVIATIONS or (len(raw) == 1 and raw.isupper()):
                    continue
        cuts.append(e)
    cuts.append(len(t))
    sents: list[tuple[int, int]] = []
    for a, b in zip(cuts, cuts[1:]):
        seg = t[a:b]
        if not _HAS_ALPHA.search(seg):
            continue
        lead = len(seg) - len(seg.lstrip())
        trail = len(seg) - len(seg.rstrip())
        sents.append((a + lead, b - trail))
    return sents


# ---------------------------------------------------------------------------
# scan
# ---------------------------------------------------------------------------
def scan(text: str) -> list[Finding]:
    """Flag unsourced appeals to authority and unsourced over-precise figures."""
    if not isinstance(text, str) or not text:
        return []
    norm, omap = _nfc(_mask_code(text))
    scan_txt = _URL_MASK.sub(lambda m: "\x00" * len(m.group()), norm)

    sents = _sentences(norm)
    if not sents:
        return []
    s_starts = [s for s, _ in sents]
    s_ends = [e for _, e in sents]
    signals = sorted((m.start(), m.end()) for rx in _BACKING_RES for m in rx.finditer(norm))
    sig_starts = [s for s, _ in signals]

    def sent_index(pos: int) -> int:
        return max(0, bisect_right(s_starts, pos) - 1)

    def backed(ts: int, te: int) -> bool:
        i, j = sent_index(ts), sent_index(max(ts, te - 1))
        lo = max(s_starts[i], ts - _PRE_WINDOW)
        hi = min(s_ends[min(j + 1, len(sents) - 1)], te + _POST_WINDOW)
        k = bisect_left(sig_starts, lo)
        while k < len(signals) and sig_starts[k] < hi:
            if signals[k][1] <= hi:
                return True
            k += 1
        return False

    hits: list[tuple[int, int, str, str]] = []     # start, end, phrase, reason

    # -- phrase triggers ----------------------------------------------------
    for m in _PHRASE_RE.finditer(scan_txt):
        fam, _lang, _pat, opts = _PHRASES[m.lastindex - 1]
        ts, te = m.start(), m.end()
        lo = max(0, ts - 16)
        if _PHRASES[m.lastindex - 1][1] == "en" and _AUX_BEFORE.search(norm, lo, ts):
            continue                                   # a question / condition, not an assertion
        if "det" in opts and _DET_BEFORE.search(norm, lo, ts):
            continue                                   # "the research shows": refers back to a named thing
        if "nofig" in opts and _FIGURE.search(norm, te, min(s_ends[sent_index(te - 1)], te + 200)):
            continue                                   # carries its own figure
        if backed(ts, te):
            continue
        phrase = re.sub(r"\s+", " ", norm[ts:te])
        hits.append((ts, te, phrase, _REASONS[fam].format(p=phrase)))

    # -- suspiciously precise numbers --------------------------------------------
    def number_ok(ts: int, te: int) -> bool:
        ls = norm.rfind("\n", 0, ts) + 1
        if norm[ls:ts].lstrip(" \t").startswith("|"):
            return False                               # markdown table row
        i = sent_index(ts)
        sent = norm[s_starts[i]:s_ends[i]]
        if "=" in sent or _ILLUSTRATIVE.search(sent):
            return False                               # arithmetic / worked example
        return not backed(ts, te)

    for m in _PCT.finditer(scan_txt):
        num = m.group("th") or m.group("en")
        if float(num) >= 99.9:
            continue                                   # "99.99% uptime" style
        if _PCT_OF.match(scan_txt, m.end()):
            continue                                   # "12.50% of 80": a calculation
        if number_ok(m.start(), m.end()):
            phrase = num + "%"
            hits.append((m.start(), m.end(), phrase, _NUMBER_REASON.format(p=phrase)))

    for m in _COUNT.finditer(scan_txt):
        digits = m.group("num").replace(",", "")
        if digits.endswith("000") or digits.startswith("0"):
            continue                                   # round number / identifier
        if number_ok(m.start(), m.end()):
            hits.append((m.start(), m.end(), m.group("num"), _NUMBER_REASON.format(p=m.group("num"))))

    # -- resolve overlaps (leftmost, then longest), number in document order ------
    hits.sort(key=lambda h: (h[0], -h[1]))
    kept: list[tuple[int, int, str, str]] = []
    for h in hits:
        if kept and h[0] < kept[-1][1]:
            continue
        kept.append(h)

    findings: list[Finding] = []
    for n, (ts, te, phrase, reason) in enumerate(kept, 1):
        if omap is None:
            s, e = ts, te
        else:
            s, e = omap[0][ts], omap[1][te - 1]
        claim = Claim(id=f"v{n}", kind=ClaimKind.VAGUE, text=text[s:e], start=s, end=e,
                      data={"phrase": phrase})
        findings.append(Finding(claim=claim, status=Status.UNVERIFIED, module=MODULE, reason=reason))
    return findings
