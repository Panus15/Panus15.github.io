"""Computable claims: statements in ANY topic that pure calculation can prove or refute.

scan(text, *, today=None) -> list[Finding]   (kind COMPUTED, ids "m1","m2",... in document order,
module "computable").  No database, no network, no locale: only datetime and Fractions.

What it looks for (Claim.data["type"])
--------------------------------------
calendar_date   a date that does not exist (30 Feb, 31 Apr, 29 Feb 2023, day 0/32, month 13).  CONTRADICTED.
                A valid standalone date produces NOTHING.  Thai months and Buddhist-era years are understood
                (a bare year >= 2400, or an explicit BE / พ.ศ., has 543 subtracted).
weekday         a date with a stated weekday ("Monday, 15 March 2021", "วันจันทร์ที่ 15 มีนาคม 2564").
span_years / span_days   "from 1999 to 2010 (11 years)", "between 3 March 2020 and 9 March 2020 is 6 days".
lifespan        "Name (1912-1954)" or "(born 1912, died 1954)" plus a stated age at death.
century         "1750 (18th century)", "ศตวรรษที่ 19 (พ.ศ. 2300)", "1857 (1850s)".
unit_conversion "5 km (3.1 mi)", "100 °C, about 212 °F", "5-10 km (3-6 mi)", compound "5 ft 9 in (175 cm)".
percent_remaining   "30% ... and the remaining 60%" not adding to 100; "12 out of 10"; "130% of the total".
date_order      "from 15 June 2021 to 3 June 2021", "from 2010 to 2005": the end is before the start.

Status policy
-------------
* CONTRADICTED only with arithmetic proof.  A claim that is ambiguous, depends on a convention (Julian vs
  Gregorian before 1583, US vs imperial gallon, nautical vs statute mile, "1900 is 20th century?"),
  is only partly parsed, or is hedged, is UNVERIFIED or produces nothing.
* "Everything parsed" rule: a half-parsed date / quantity / range emits nothing.  Dates whose month name is
  lower-case and have no year ("may 5", "march 31") are skipped (the words are ordinary English).
* Ambiguous numeric dates (03/04/2020) are never findings, unless NEITHER day/month reading exists.
* Code (fenced blocks and inline `code`) is ignored.  Offsets are exact: text == source[start:end].

Policy details
--------------
* Unit conversion tolerance = rounding at the precision STATED for the converted value (half a unit of its last
  digit) plus 1 % slack.  The claim passes if EITHER direction (A->B at B's precision, B->A at A's precision)
  fits, because either number may be the rounded one.  A whole number with trailing zeros ("60", "1,000") may
  have been rounded to that place: matching only under that looser reading is UNVERIFIED, not CONTRADICTED.
  Temperatures also try the "difference" reading (5 C = 9 F as a change); that alone is UNVERIFIED unless the
  sentence says rise/drop/by/...  Gallon / fluid ounce / pint default to US with an imperial fallback
  (UNVERIFIED, convention named); "mile" defaults to statute with a nautical fallback.
* Span claims accept inclusive and exclusive counting (and completed years for full dates).  A hedge such as
  "about" widens the tolerance to max(1, 10 %) and a miss is UNVERIFIED; inequality words ("over", "at least")
  make the span claim skipped.  An end before the start is CONTRADICTED unless the sentence says
  back/reverse/countdown.  A yearless "from 15 June to 3 June" can legitimately wrap New Year (fiscal and
  school years), so it is UNVERIFIED (and silent when the sentence says annual/fiscal/...).
* A year that ends in 00 stated as the start of the next century (1900 -> "20th century") is UNVERIFIED (strictly
  it is the last year of the 19th), any other mismatch is CONTRADICTED.
* A lifespan over 125 years is UNVERIFIED ("implausible"), never CONTRADICTED.
* Gregorian rules are applied exactly; before 15 Oct 1582 weekday and 29-Feb statements depend on the calendar
  in use and are UNVERIFIED.
"""

from __future__ import annotations

import datetime
import math
import re
from fractions import Fraction
from typing import Optional

from .models import Claim, ClaimKind, Finding, Status

__all__ = ["scan"]

MODULE = "computable"
V, U, C = Status.VERIFIED, Status.UNVERIFIED, Status.CONTRADICTED

# ------------------------------------------------------------------ text prep ---

def _digit_map() -> dict:
    m = {}
    for base in (0x0E50, 0x0660, 0x06F0, 0xFF10):          # Thai, Arabic-Indic, Persian, full-width
        for i in range(10):
            m[base + i] = ord("0") + i
    m[0x00A0] = ord(" ")
    return m


_NORMALIZE = _digit_map()
_CODE = "\x00"
_FENCE_OPEN = re.compile(r"[ \t]*(`{3,}|~{3,})")
_INLINE_CODE = re.compile(r"(?<!`)(`+)(?!`)([^\r\n]+?)(?<!`)\1(?!`)")
_LINES = re.compile(r"[^\r\n]*(?:\r\n|\r|\n)|[^\r\n]+")


def _blank(s: str, a: int, b: int) -> str:
    return s[:a] + re.sub(r"[^\r\n]", _CODE, s[a:b]) + s[b:]


def _mask_code(text: str) -> str:
    """Same-length copy with fenced and inline code replaced by NUL (newlines kept)."""
    spans = []
    fence = None
    fstart = 0
    for m in _LINES.finditer(text):
        line = m.group(0)
        body = line.rstrip("\r\n")
        if fence is None:
            fm = _FENCE_OPEN.match(body)
            if fm and not (fm.group(1)[0] == "`" and "`" in body[fm.end():]):
                fence = (fm.group(1)[0], len(fm.group(1)))
                fstart = m.start()
        else:
            fm = _FENCE_OPEN.match(body)
            if fm and fm.group(1)[0] == fence[0] and len(fm.group(1)) >= fence[1] \
                    and not body[fm.end():].strip():
                spans.append((fstart, m.start() + len(body)))
                fence = None
    if fence is not None:
        spans.append((fstart, len(text)))
    out, pos = [], 0
    for a, b in spans:
        out.append(text[pos:a])
        out.append(re.sub(r"[^\r\n]", _CODE, text[a:b]))
        pos = b
    out.append(text[pos:])
    t = "".join(out)
    parts, pos = [], 0
    for m in _INLINE_CODE.finditer(t):
        parts.append(t[pos:m.start()])
        parts.append(re.sub(r"[^\r\n]", _CODE, m.group(0)))
        pos = m.end()
    parts.append(t[pos:])
    return "".join(parts)


_BOUND = re.compile(r"[!?\n\r。]|\.(?=[ \t\r\n]|\Z)")


def _sent_start(s: str, a: int) -> int:
    lo = max(0, a - 400)
    last = lo
    for m in _BOUND.finditer(s, lo, a):
        last = m.end()
    return last


def _sent_end(s: str, b: int) -> int:
    hi = min(len(s), b + 400)
    m = _BOUND.search(s, b, hi)
    return m.start() if m else hi


# ------------------------------------------------------------------ shared bits ---

S0 = r"[ \t]{0,4}"
S1 = r"[ \t]{1,4}"
_NB = r"0-9A-Za-z.,/:\-_=#@%&"        # characters that may not directly precede a date

MONTH_NAMES = ["", "January", "February", "March", "April", "May", "June", "July", "August",
               "September", "October", "November", "December"]
_DIM = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

_EN_MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3, "april": 4, "apr": 4,
    "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7, "august": 8, "aug": 8, "september": 9,
    "sept": 9, "sep": 9, "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}
_TH_MONTH_LIST = [
    (1, ["มกราคม", "ม.ค."]), (2, ["กุมภาพันธ์", "ก.พ."]), (3, ["มีนาคม", "มี.ค."]),
    (4, ["เมษายน", "เม.ย."]), (5, ["พฤษภาคม", "พ.ค."]), (6, ["มิถุนายน", "มิ.ย."]),
    (7, ["กรกฎาคม", "ก.ค."]), (8, ["สิงหาคม", "ส.ค."]), (9, ["กันยายน", "ก.ย."]),
    (10, ["ตุลาคม", "ต.ค."]), (11, ["พฤศจิกายน", "พ.ย."]), (12, ["ธันวาคม", "ธ.ค."]),
]
_TH_MONTHS = {}
for _n, _names in _TH_MONTH_LIST:
    for _nm in _names:
        _TH_MONTHS[_nm] = _n
        _TH_MONTHS[_nm.rstrip(".")] = _n


def _alt(words) -> str:
    return "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))


_EN_MON_RX = r"(?P<mon>" + _alt(_EN_MONTHS) + r")(?![A-Za-z])\.?"
_TH_MON_RX = r"(?P<mon>" + _alt(_TH_MONTHS) + r")"

_ERA1 = r"(?-i:(?:พ\.ศ\.|ค\.ศ\.|B\.E\.|C\.E\.|A\.D\.|BE|CE|AD)(?![A-Za-z]))"
_ERA2 = (r"(?-i:(?:พ\.ศ\.|ค\.ศ\.|B\.C\.E\.|B\.C\.|B\.E\.|C\.E\.|A\.D\.|BCE|BC|BE|CE|AD)(?![A-Za-z]))")
_YEAR_TAIL = (r"(?:,?" + S0 + r"(?:(?P<e1>" + _ERA1 + r")" + S0 + r"|ปี" + S0 + r")?"
              r"(?P<yr>\d{4})(?!\d)(?:" + S0 + r"(?P<e2>" + _ERA2 + r"))?)?")
_ORD = r"(?:st|nd|rd|th)?"

_DATE_SRC = {
    "dmy_en": (r"(?<![" + _NB + r"])(?P<d>\d{1,2})" + _ORD + r"(?:" + S1 + r"of)?" + S0 + _EN_MON_RX + _YEAR_TAIL),
    "mdy_en": (r"(?<![A-Za-z])" + _EN_MON_RX + r"(?:" + S1 + r"(?:the" + S1 + r")?|" + S0 + r")(?P<d>\d{1,2})"
               + _ORD + r"(?!\d)(?![:.]\d)(?![/\-]\d)" + _YEAR_TAIL),
    "dmy_th": (r"(?<![" + _NB + r"])(?P<d>\d{1,2})" + S0 + _TH_MON_RX + _YEAR_TAIL),
    "iso": (r"(?<![" + _NB + r"])(?P<y>\d{4})(?P<sep>[-/])(?P<m>\d{1,2})(?P=sep)(?P<d>\d{1,2})"
            r"(?![0-9A-Za-z_@/]|-\d|\.\w)"),
    "num": (r"(?<![" + _NB + r"])(?P<a>\d{1,2})(?P<sep>[/.\-])(?P<b>\d{1,2})(?P=sep)(?P<y>\d{4})"
            r"(?![0-9A-Za-z_@/]|[/\-.]\d)"),
}
_DATE_RX = {k: re.compile(v, re.I) for k, v in _DATE_SRC.items()}


def _nocap(p: str) -> str:
    p = re.sub(r"\(\?P<\w+>", "(?:", p)
    return p.replace("(?P=sep)", "[-/.]")


_DATE_EP = "(?:" + "|".join(_nocap(_DATE_SRC[k]) for k in ("dmy_en", "mdy_en", "dmy_th", "iso", "num")) + ")"
_YEAR_EP = (r"(?<![0-9A-Za-z.,/:\-_=#@%&])(?:(?:" + _ERA1 + S0 + r"|ปี" + S0 + r")?\d{4}(?!\d)(?:" + S0 + _ERA2 + r")?)")
_EP = "(?:" + _DATE_EP + "|" + _YEAR_EP + ")"
_YEAR_FULL = re.compile(r"(?:(?P<e1>" + _ERA1 + r")" + S0 + r"|ปี" + S0 + r")?(?P<yr>\d{4})(?!\d)(?:" + S0
                        + r"(?P<e2>" + _ERA2 + r"))?", re.I)


def _leap(y: int) -> bool:
    return y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)


def _year_ce(yr: int, e1, e2):
    """-> (ce_year, was_buddhist) or None (BC / out of range / contradictory markers)."""
    be = ce = bc = False
    for x in (e1, e2):
        if not x:
            continue
        u = x.replace(".", "").replace(" ", "")
        if u in ("พศ", "BE"):
            be = True
        elif u in ("คศ", "CE", "AD"):
            ce = True
        elif u.startswith("BC") or u.startswith("ก่อน"):
            bc = True
    if bc or (be and ce):
        return None
    if be:
        y, b = yr - 543, True
    elif ce:
        y, b = yr, False
    elif yr >= 2400:
        y, b = yr - 543, True
    else:
        y, b = yr, False
    if y < 1 or y > 9999:
        return None
    return y, b


def _check_ymd(y: Optional[int], m: int, d: int):
    """-> ("ok"|"bad"|"conv", message).  y None = year not stated."""
    if m < 1 or m > 12:
        return "bad", f"there is no month {m} (months run from 1 to 12)"
    if d < 1:
        return "bad", f"there is no day {d} in a month"
    if d > 31:
        return "bad", f"no month has {d} days"
    name = MONTH_NAMES[m]
    if y is None:
        top = 29 if m == 2 else _DIM[m]
        if d > top:
            return "bad", f"{name} never has more than {top} days"
        return "ok", ""
    dim = _DIM[m] + (1 if (m == 2 and _leap(y)) else 0)
    if d <= dim:
        return "ok", ""
    if m == 2 and d == 29:
        if y < 1583 and y % 4 == 0:
            return "conv", (f"{y} is a leap year only in the Julian calendar (in use before 1582), "
                            "not in the Gregorian calendar")
        why = "divisible by 100 but not by 400" if y % 100 == 0 else "not divisible by 4"
        return "bad", f"{y} is not a leap year ({why}), so February has only 28 days"
    return "bad", f"{name} {y} has only {dim} days"


class _Date:
    __slots__ = ("start", "end", "y", "readings", "status", "msg", "be", "be_year", "kind")

    def __init__(self):
        self.y = None
        self.readings = []      # [(month, day)] valid readings
        self.status = "ok"
        self.msg = ""
        self.be = False
        self.be_year = None
        self.kind = ""


def _date_from_match(kind: str, m) -> Optional[_Date]:
    g = m.groupdict()
    dt = _Date()
    dt.start, dt.end, dt.kind = m.start(), m.end(), kind
    y = None
    if kind in ("dmy_en", "mdy_en", "dmy_th"):
        monraw = g["mon"]
        key = monraw.rstrip(".")
        mon = _EN_MONTHS.get(key.lower()) if kind != "dmy_th" else _TH_MONTHS.get(monraw) or _TH_MONTHS.get(key)
        if mon is None:
            return None
        day = int(g["d"])
        if g.get("yr"):
            r = _year_ce(int(g["yr"]), g.get("e1"), g.get("e2"))
            if r is None:
                return None
            y, dt.be = r
            dt.be_year = int(g["yr"]) if dt.be else None
        elif kind != "dmy_th" and (key[:1].islower() and not key.isupper()):
            return None            # "may 5", "march 31" without a year: ordinary words
        elif kind == "mdy_en" and not _yearless_follow_ok(m):
            return None
        pairs = [(mon, day)]
    elif kind == "iso":
        r = _year_ce(int(g["y"]), None, None)
        if r is None:
            return None
        y, dt.be = r
        dt.be_year = int(g["y"]) if dt.be else None
        mon, day = int(g["m"]), int(g["d"])
        if mon > 12 or mon == 0:
            if not (1 <= day <= 31) or not (1000 <= int(g["y"]) <= 2999):
                return None
        pairs = [(mon, day)]
    else:                       # num
        r = _year_ce(int(g["y"]), None, None)
        if r is None:
            return None
        y, dt.be = r
        dt.be_year = int(g["y"]) if dt.be else None
        a, b = int(g["a"]), int(g["b"])
        pairs = [(b, a), (a, b)] if a != b else [(b, a)]       # (month, day): day/month first, then month/day
    dt.y = y
    results = [(p, _check_ymd(y, p[0], p[1])) for p in pairs]
    oks = [p for p, (st, _) in results if st == "ok"]
    convs = [(p, msg) for p, (st, msg) in results if st == "conv"]
    if oks:
        dt.readings = oks
        dt.status = "ok"
    elif convs:
        dt.readings = []
        dt.status = "conv"
        dt.msg = convs[0][1]
        dt.readings = [convs[0][0]]
    else:
        dt.status = "bad"
        if kind == "num":
            a, b = int(g["a"]), int(g["b"])
            if a > 12 and b > 12:
                dt.msg = f"neither {a} nor {b} can be a month (months run from 1 to 12)"
            elif a > 12:
                dt.msg = results[0][1][1]
            elif b > 12:
                dt.msg = results[-1][1][1]
            else:
                dt.msg = results[0][1][1]
        else:
            dt.msg = results[0][1][1]
        dt.readings = [pairs[0]]
    return dt


_FOLLOW_OK = re.compile(r"[ \t]{0,3}(?:\Z|[.,;:!?)\]\"'”’\r\n]|[-–—]|(?:and|or|to|till|until|through|at|in|on|by|is|was)(?![A-Za-z]))",
                        re.I)


def _yearless_follow_ok(m) -> bool:
    return _FOLLOW_OK.match(m.string, m.end()) is not None


def _find_dates(s: str) -> list:
    cands = []
    for kind, rx in _DATE_RX.items():
        for m in rx.finditer(s):
            d = _date_from_match(kind, m)
            if d is not None:
                cands.append(d)
    cands.sort(key=lambda d: (d.start, -(d.end - d.start)))
    out, last = [], -1
    for d in cands:
        if d.start >= last:
            out.append(d)
            last = d.end
    return out


def _fmt_date(y, m: int, d: int, dt: Optional[_Date] = None) -> str:
    base = f"{d} {MONTH_NAMES[m]}" + (f" {y}" if y is not None else "")
    if dt is not None and dt.be and dt.be_year:
        base += f" (Buddhist Era {dt.be_year})"
    return base


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        return f"{n}th"
    return f"{n}" + {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


# candidate: (start, end, type, status, reason, priority, evidence)
def _cand(start, end, typ, status, reason, prio, evidence=()):
    return (start, end, typ, status, reason, prio, tuple(evidence))


# ------------------------------------------------------------- 1+2: dates, weekdays ---

_WD_EN_MAP = {
    "monday": 0, "mon": 0, "tuesday": 1, "tues": 1, "tue": 1, "wednesday": 2, "wed": 2,
    "thursday": 3, "thurs": 3, "thur": 3, "thu": 3, "friday": 4, "fri": 4, "saturday": 5, "sat": 5,
    "sunday": 6, "sun": 6,
}
_WD_TH_MAP = {"จันทร์": 0, "อังคาร": 1, "พุธ": 2, "พฤหัสบดี": 3, "พฤหัส": 3, "ศุกร์": 4, "เสาร์": 5, "อาทิตย์": 6}
_WD_EN = r"(?:" + _alt(_WD_EN_MAP) + r")"
_WD_TH = r"(?:วัน)?(?:" + _alt(_WD_TH_MAP) + r")"
_WD_ANY = r"(?:" + _WD_EN + r"(?![A-Za-z])|" + _WD_TH + r")"

_PRE_WD = re.compile(r"(?<![A-Za-z])(?P<wd>" + _WD_ANY + r")\.?" + S0 + r",?" + S0 + r"(?:ที่" + S0 + r")?"
                     r"(?:(?:the|on)" + S1 + r")?\Z", re.I)
_SUF_WD = re.compile(
    S0 + r"(?:\(" + S0 + r"(?:on" + S1 + r")?(?P<w1>" + _WD_ANY + r")\.?" + S0 + r"\)"
    r"|,?" + S0 + r"(?:(?:which|that|and)" + S1 + r")?(?:was|is|fell" + S1 + r"on|falls" + S1 + r"on|will" + S1 + r"be|"
    r"would" + S1 + r"be|being|landed" + S1 + r"on|lands" + S1 + r"on)" + S1 +
    r"(?:(?:on" + S1 + r")?(?:a|an)" + S1 + r"|on" + S1 + r"|the" + S1 + r"day" + S1 + r")?"
    r"(?P<w2>" + _WD_ANY + r")(?![A-Za-z])"
    r"|" + S0 + r"(?:ตรงกับ|เป็น|คือ)" + S0 + r"(?P<w3>" + _WD_TH + r"))", re.I)


def _wd_index(tok: str) -> Optional[int]:
    t = tok.strip().rstrip(".")
    if t.startswith("วัน"):
        t = t[3:]
    if t in _WD_TH_MAP:
        return _WD_TH_MAP[t]
    return _WD_EN_MAP.get(t.lower())


def _check_dates(s: str, today: datetime.date) -> list:
    out = []
    for dt in _find_dates(s):
        label = s[dt.start:dt.end].strip()
        if dt.status == "bad":
            out.append(_cand(dt.start, dt.end, "calendar_date", C,
                             f"\"{label}\" is not a real date: {dt.msg}.", 0, ("calendar: Gregorian rules",)))
            continue
        if dt.status == "conv":
            out.append(_cand(dt.start, dt.end, "calendar_date", U,
                             f"\"{label}\" is valid only in the Julian calendar: {dt.msg}.", 0))
            continue
        if dt.y is None:
            continue
        # weekday: prefix first, then suffix
        wd_idx = None
        a, b = dt.start, dt.end
        lo = max(0, dt.start - 60)
        pm = _PRE_WD.search(s, lo, dt.start)
        if pm:
            wd_idx = _wd_index(pm.group("wd"))
            a = pm.start()
        else:
            sm = _SUF_WD.match(s, dt.end, min(len(s), dt.end + 90))
            if sm:
                tok = sm.group("w1") or sm.group("w2") or sm.group("w3")
                wd_idx = _wd_index(tok)
                b = sm.end()
        if wd_idx is None:
            continue
        actual = []
        for (mo, da) in dt.readings:
            try:
                actual.append((datetime.date(dt.y, mo, da), mo, da))
            except ValueError:
                pass
        if not actual:
            continue
        claimed = WEEKDAYS[wd_idx]
        if len({x[0].weekday() for x in actual}) > 1:
            # ambiguous numeric date: proven wrong only if it matches under neither reading
            if all(x[0].weekday() != wd_idx for x in actual):
                alts = " and ".join(f"{_fmt_date(dt.y, x[1], x[2])} is a {WEEKDAYS[x[0].weekday()]}" for x in actual)
                out.append(_cand(a, b, "weekday", C,
                                 f"Whether read day-first or month-first, the date is not a {claimed}: {alts}.", 0))
            continue
        d0, mo, da = actual[0]
        shown = _fmt_date(dt.y, mo, da, dt)
        if d0 < datetime.date(1582, 10, 15):
            out.append(_cand(a, b, "weekday", U,
                             f"The weekday of {shown} depends on whether the Julian or Gregorian calendar is meant "
                             f"(the Gregorian proleptic calendar gives a {WEEKDAYS[d0.weekday()]}).", 0))
            continue
        real = WEEKDAYS[d0.weekday()]
        verb = "will be" if d0 > today else "was"
        if d0.weekday() == wd_idx:
            out.append(_cand(a, b, "weekday", V, f"{shown} {verb} indeed a {real}.", 0, (f"computed: {real}",)))
        else:
            out.append(_cand(a, b, "weekday", C, f"{shown} {verb} a {real}, not a {claimed}.", 0,
                             (f"computed: {real}", f"claimed: {claimed}")))
    return out


# ---------------------------------------------------- 3+8: spans and date order ---

_HEDGE_APPROX = r"(?:about|around|roughly|approximately|approx\.?|circa|~|≈|some|ประมาณ|ราวๆ|ราว)"
_HEDGE_INEQ = (r"(?:nearly|almost|over|more[ \t]+than|at[ \t]+least|at[ \t]+most|up[ \t]+to|under|less[ \t]+than|"
               r"เกือบ|กว่า|มากกว่า|ไม่ถึง|อย่างน้อย)")
_COUNT_EN = re.compile(
    r"(?:" + S0 + r"[(,—–-]" + S0 + r"(?:(?:a" + S1 + r")?(?:span|period|total|duration|difference|gap)" + S1 + r"of" + S1 + r")?"
    r"|" + S1 + r"(?:is|was|are|were|spans?|spanned|covers?|covered|represents?|represented|makes?|equals?|gives?|"
    r"totals?|totalled|totaled|lasted|lasts|=|:)" + S1 +
    r"(?:(?:a" + S1 + r")?(?:span|period|total|duration|difference|gap|time)" + S1 + r"of" + S1 + r"|exactly" + S1 + r")?)"
    r"(?:(?P<hedge>" + _HEDGE_APPROX + r"|" + _HEDGE_INEQ + r")" + S1 + r")?"
    r"(?P<n>\d{1,6})(?!\d)" + S0 + r"-?" + S0 + r"(?P<unit>years?|yrs?|days?)(?![A-Za-z])"
    r"(?!" + S0 + r"(?:later|after|before|ago|old|earlier|prior|apart))", re.I)
_COUNT_TH = re.compile(
    r"(?:" + S0 + r"[(,]" + S0 + r"|" + S0 + r"(?:รวมเป็น|รวม|เป็นเวลา|นาน|ห่างกัน|เท่ากับ|คือ|ได้)" + S0 + r")"
    r"(?P<hedge>" + _HEDGE_APPROX + r"|" + _HEDGE_INEQ + r")?" + S0 +
    r"(?P<n>\d{1,6})(?!\d)" + S0 + r"(?P<unit>ปี(?!ที่|ก่อน|นี้)|วัน(?!ที่|นี้))")

_RANGE_FROM = re.compile(
    r"(?<![A-Za-z])(?P<kw>from|between)" + S1 + r"(?P<e1>" + _EP + r")"
    r"(?P<sep>" + S1 + r"(?:to|until|till|through|thru|and)" + S1 + r"|" + S0 + r"[-–—]" + S0 + r")(?P<e2>" + _EP + r")",
    re.I)
_RANGE_BARE = re.compile(r"(?P<e1>" + _EP + r")" + S0 + r"(?P<sep>[-–—])" + S0 + r"(?P<e2>" + _EP + r")", re.I)
_RANGE_TH = re.compile(
    r"(?P<kw>ตั้งแต่|ระหว่าง)" + S0 + r"(?:ปี" + S0 + r")?(?P<e1>" + _EP + r")" + S0 +
    r"(?P<sep>ไปจนถึง|จนกระทั่ง|ไปถึง|จนถึง|ถึง|กับ|และ|[-–—])" + S0 + r"(?P<e2>" + _EP + r")")

_GUARD_ORDER = re.compile(
    r"\b(?:back(?:ward|wards)?|reverse[d]?|in reverse|countdown|count(?:ing)? down|descending|newest|oldest|"
    r"before the present|ago|BC|BCE)\b|ย้อนหลัง|ย้อนกลับ|นับถอยหลัง|ก่อนคริสตกาล|ปีที่แล้ว|เรียงจาก", re.I)
_GUARD_CYCLE = re.compile(
    r"\b(?:annual|annually|yearly|every year|each year|fiscal|academic|school year|season|seasonal|calendar year|"
    r"overnight|per year)\b|ทุกปี|ปีงบประมาณ|ปีการศึกษา|ภาคเรียน|ฤดู", re.I)
_AFTER_BC = re.compile(r"[ \t]{0,3}(?:BCE?\b|B\.C\.|ก่อน|years?[ \t]+ago|ปีที่แล้ว|ปีก่อน|ago\b)", re.I)


def _parse_ep(txt: str):
    for kind, rx in _DATE_RX.items():
        m = rx.fullmatch(txt)
        if m:
            d = _date_from_match(kind, m)
            if d is None:
                return None
            return ("date", d)
    m = _YEAR_FULL.fullmatch(txt)
    if m:
        r = _year_ce(int(m.group("yr")), m.group("e1"), m.group("e2"))
        if r is None:
            return None
        return ("year", r[0])
    return None


def _ep_value(ep):
    """(sortable tuple, shown text, has_year) or None if not a single usable value."""
    kind, v = ep
    if kind == "year":
        return (v,), str(v), True
    if v.status != "ok" or len(v.readings) != 1:
        return None
    mo, da = v.readings[0]
    if v.y is None:
        return (mo, da), f"{da} {MONTH_NAMES[mo]}", False
    return (v.y, mo, da), _fmt_date(v.y, mo, da), True


def _ep_kind(ep) -> str:
    return "year" if ep[0] == "year" else ("fulldate" if ep[1].y is not None else "yearless")


def _check_ranges(s: str) -> list:
    out = []
    seen = set()
    for rx, th in ((_RANGE_FROM, False), (_RANGE_TH, True), (_RANGE_BARE, False)):
        bare = rx is _RANGE_BARE
        for m in rx.finditer(s):
            key = (m.start(), m.end())
            if key in seen:
                continue
            seen.add(key)
            try:
                c = _one_range(s, m, th, bare)
            except Exception:
                c = None
            if c:
                out.append(c)
    return out


def _one_range(s: str, m, th: bool, bare: bool):
    e1, e2 = _parse_ep(m.group("e1")), _parse_ep(m.group("e2"))
    if e1 is None or e2 is None:
        return None
    k1, k2 = _ep_kind(e1), _ep_kind(e2)
    if k1 != k2:
        return None
    v1, v2 = _ep_value(e1), _ep_value(e2)
    if v1 is None or v2 is None:
        return None
    if _AFTER_BC.match(s, m.end()):
        return None
    sepw = m.group("sep").strip().lower()
    kw = (m.groupdict().get("kw") or "").lower()
    symmetric = (kw in ("between", "ระหว่าง") and sepw in ("and", "กับ", "และ")) or sepw in ("and", "กับ", "และ")
    cm = (_COUNT_TH if th else _COUNT_EN).match(s, m.end(), min(len(s), m.end() + 160))
    if bare and cm is None:
        return None
    if bare and not (s[m.end():cm.start("hedge") if cm.group("hedge") else cm.start("n")].lstrip().startswith("(")):
        return None
    end = cm.end() if cm else m.end()
    # closing paren belongs to the claim when we opened one
    if cm and s[m.end():cm.start("n")].count("(") and s[end:end + 6].lstrip(" \t").startswith(")"):
        end = end + (len(s[end:end + 6]) - len(s[end:end + 6].lstrip(" \t"))) + 1
    # sentence-level guards
    ss, se = _sent_start(s, m.start()), _sent_end(s, end)
    sentence = s[ss:se]
    start = m.start()
    shown1, shown2 = v1[1], v2[1]
    order_bad = (not symmetric) and v2[0] < v1[0]

    if cm is None:
        # date_order only (no count): directed ranges only
        if not order_bad or symmetric:
            return None
        if _GUARD_ORDER.search(sentence):
            return None
        if k1 == "yearless":
            if _GUARD_CYCLE.search(sentence):
                return None
            return _cand(start, end, "date_order", U,
                         f"The range ends ({shown2}) before it starts ({shown1}), which is fine only if it wraps "
                         "into the next year.", 1)
        return _cand(start, end, "date_order", C,
                     f"The range ends before it starts: {shown2} is earlier than {shown1}.", 1)

    hedge = (cm.group("hedge") or "").strip().lower()
    if hedge and re.fullmatch(_HEDGE_INEQ, hedge, re.I):
        return None
    n = int(cm.group("n"))
    unit_raw = cm.group("unit").lower()
    unit = "years" if unit_raw.startswith(("y", "ปี")) else "days"
    typ = "span_years" if unit == "years" else "span_days"
    if k1 == "yearless":
        return None
    if k1 == "year" and unit == "days":
        return None
    if order_bad:
        if _GUARD_ORDER.search(sentence):
            return None
        return _cand(start, end, typ, C,
                     f"The span ends before it starts: {shown2} is earlier than {shown1}, so it cannot last {n} {unit}.", 0)
    # accepted counts
    if k1 == "year":
        y1, y2 = abs(v2[0][0] - v1[0][0]), 0
        base = y1
        accept = {base: "exclusive count", base + 1: "inclusive count"}
    else:
        da, db = datetime.date(*v1[0]), datetime.date(*v2[0])
        if db < da:
            da, db = db, da
        if unit == "days":
            base = (db - da).days
            accept = {base: "exclusive count", base + 1: "inclusive count"}
        else:
            full = db.year - da.year - (1 if (db.month, db.day) < (da.month, da.day) else 0)
            diff = db.year - da.year
            accept = {full: "completed years"}
            accept.setdefault(diff, "difference of calendar years")
            accept.setdefault(diff + 1, "inclusive count of calendar years")
            base = full
    where = f"{shown1} to {shown2}"
    if n in accept:
        if len(accept) == 2 and n == base:
            how = f"counting {unit} exclusively"
        else:
            how = accept[n]
        return _cand(start, end, typ, V, f"From {where} is {n} {unit} ({how}), as stated.", 0,
                     (f"computed: {sorted(accept)}",))
    if hedge:
        tol = max(1, int(0.1 * max(accept)))
        if any(abs(n - a) <= tol for a in accept):
            return _cand(start, end, typ, V, f"From {where} is about {base} {unit}, consistent with the hedged {n}.", 0)
        return _cand(start, end, typ, U,
                     f"From {where} is {base} {unit}, which differs from the hedged \"about {n}\".", 0)
    opts = " or ".join(str(a) for a in sorted(accept))
    return _cand(start, end, typ, C,
                 f"From {where} is {opts} {unit} (depending on inclusive or exclusive counting), not {n}.", 0,
                 (f"computed: {sorted(accept)}", f"claimed: {n}"))


# --------------------------------------------------------------------- 4: lifespan ---

_Y = lambda name: r"(?:(?:" + _ERA1 + r")" + S0 + r")?(?P<" + name + r">\d{4})(?!\d)"
_L_DASH = re.compile(
    r"\(" + S0 + r"(?:(?:b\.|born|เกิด)" + S0 + r"(?:(?:in|ปี)" + S0 + r")?)?" + _Y("y1") + S0 + r"[–—-]" + S0 +
    r"(?:(?:d\.|died|เสียชีวิต)" + S0 + r"(?:(?:in|ปี)" + S0 + r")?)?" + _Y("y2") + S0 + r"\)", re.I)
_L_BORN_EN = re.compile(
    r"(?:\(" + S0 + r")?(?:born|b\.)" + S0 + r"(?:in" + S1 + r")?" + _Y("y1") + S0 + r"(?:[,;]|and|[–—-])?" + S0 +
    r"(?:and" + S1 + r")?(?:died|d\.|death)" + S0 + r"(?:in" + S1 + r")?" + _Y("y2") + r"(?:" + S0 + r"\))?", re.I)
_L_BORN_TH = re.compile(
    r"(?:\(" + S0 + r")?(?:เกิด(?:เมื่อ)?)" + S0 + r"(?:ปี|พ\.ศ\.|ค\.ศ\.)?" + S0 + _Y("y1") + S0 + r"(?:[,;]|และ)?" + S0 +
    r"(?:เสียชีวิต|ถึงแก่กรรม|สิ้นพระชนม์|สิ้นชีพ|ตาย)" + S0 + r"(?:ปี|เมื่อ|พ\.ศ\.|ค\.ศ\.)?" + S0 + _Y("y2") +
    r"(?:" + S0 + r"\))?", re.I)
_AGE = re.compile(
    r"(?:(?:died|dying|passed" + S1 + r"away|death)" + S1 + r"(?:at|aged)" + S1 + r"(?:(?:the" + S1 + r")?age" + S1 + r"(?:of" + S1 + r")?)?"
    r"|(?<![A-Za-z])(?:aged|age" + S1 + r"of|at" + S1 + r"the" + S1 + r"age" + S1 + r"of|at" + S1 + r"age)" + S1 + r")"
    r"(?P<age>\d{1,3})(?:" + S0 + r"(?:years?|yrs?)(?:" + S1 + r"(?:old|of" + S1 + r"age))?)?(?![A-Za-z0-9%])"
    r"|อายุ" + S0 + r"(?:ได้" + S0 + r"|รวม" + S0 + r")?(?P<tage>\d{1,3})(?!\d)" + S0 + r"ปี", re.I)


def _lifespan_year(txt: str, g):
    r = _year_ce(int(g), None, None)
    return r[0] if r else None


def _check_lifespans(s: str, today: datetime.date) -> list:
    out = []
    items = []
    for m in _L_DASH.finditer(s):
        j = m.start() - 1
        while j >= 0 and s[j] in " \t":
            j -= 1
        if j < 0 or not s[j].isalpha():
            continue
        items.append((m, False))
    for rx in (_L_BORN_EN, _L_BORN_TH):
        for m in rx.finditer(s):
            items.append((m, True))
    # an item may not overlap another
    items.sort(key=lambda t: t[0].start())
    kept = []
    for m, explicit in items:
        if kept and m.start() < kept[-1][0].end():
            continue
        kept.append((m, explicit))
    for idx, (m, explicit) in enumerate(kept):
        y1 = _lifespan_year(m.group("y1"), m.group("y1"))
        y2 = _lifespan_year(m.group("y2"), m.group("y2"))
        if y1 is None or y2 is None:
            continue
        ss, se = _sent_start(s, m.start()), _sent_end(s, m.end())
        others = [k for k in kept if k[0] is not m and ss <= k[0].start() < se]
        if others:
            continue
        ages = [a for a in _AGE.finditer(s, ss, se)]
        age = None
        a_start, a_end = m.start(), m.end()
        if len(ages) == 1:
            a = ages[0]
            if a.start() >= m.end() or a.end() <= m.start():
                age = int(a.group("age") or a.group("tage"))
                a_start, a_end = min(a_start, a.start()), max(a_end, a.end())
        elif len(ages) > 1:
            continue
        if age is None and not explicit:
            continue                          # a bare "(1954-1912)" is not necessarily a person's lifespan
        life = y2 - y1
        span_txt = f"born {y1}, died {y2}"
        if y2 < y1:
            out.append(_cand(a_start, a_end, "lifespan", C,
                             f"The death year {y2} is before the birth year {y1}.", 2))
            continue
        if y2 > today.year:
            out.append(_cand(a_start, a_end, "lifespan", C,
                             f"A death in {y2} has not happened yet (the current year is {today.year}).", 2))
            continue
        if age is not None:
            if age not in (life, life - 1):
                opts = f"{life - 1} or {life}" if life >= 1 else "0"
                out.append(_cand(a_start, a_end, "lifespan", C,
                                 f"Someone {span_txt} would have been {opts} at death, not {age}.", 2,
                                 (f"computed: {opts}", f"claimed: {age}")))
                continue
            if life > 125:
                out.append(_cand(a_start, a_end, "lifespan", U,
                                 f"A lifespan of {life} years ({y1}-{y2}) is implausible, so the age {age} was not accepted.", 2))
                continue
            out.append(_cand(a_start, a_end, "lifespan", V,
                             f"{y2} - {y1} = {life}, so an age of {age} at death fits (birthday unknown, so {life - 1} or {life}).", 2))
            continue
        if life > 125:
            out.append(_cand(a_start, a_end, "lifespan", U,
                             f"A lifespan of {life} years ({y1}-{y2}) is implausible.", 2))
    return out


# --------------------------------------------------------------------- 5: century ---

_ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50}


def _roman(r: str) -> Optional[int]:
    tot = 0
    for i, ch in enumerate(r):
        v = _ROMAN.get(ch)
        if v is None:
            return None
        nxt = _ROMAN.get(r[i + 1]) if i + 1 < len(r) else None
        tot += -v if (nxt and nxt > v) else v
    return tot if 1 <= tot <= 30 else None


_YC = (r"(?:(?P<e1>" + _ERA1 + r")" + S0 + r"|ปี" + S0 + r")?(?P<y>\d{3,4})(?!\d)(?:" + S0 + r"(?P<e2>" + _ERA2 + r"))?")
_CEN_EN = (r"(?:the" + S1 + r")?(?:(?P<c>\d{1,2})(?:st|nd|rd|th)|(?-i:(?P<r>[IVXL]{1,6}))(?:th)?)(?:[ \t-]{1,3})century"
           r"(?:" + S1 + r"(?-i:(?P<ce>AD|CE)))?")
_CEN_TH = r"(?P<b>พุทธ)?ศตวรรษ(?:ที่)?" + S0 + r"(?P<c>\d{1,2})(?!\d)"
_PRE_APPROX = r"(?:(?:in|around|circa|c\.|ca\.|about)" + S1 + r")?"
_RX_C1 = re.compile(r"(?<![0-9A-Za-z.,])" + _YC + S0 + r"(?:\(" + S0 + _PRE_APPROX + r"(?P<cen>" + _CEN_EN + r")" + S0 + r"\)|,"
                    + S0 + r"(?P<cen2>(?!the" + S1 + r")" + _CEN_EN.replace("(?P<c>", "(?P<c2>").replace("(?P<r>", "(?P<r2>").replace("(?P<ce>", "(?P<ce2>")
                    + r")(?=" + S0 + r"(?:[.,;)]|\Z|\n|\r)))", re.I)
_RX_C2 = re.compile(r"(?<![0-9A-Za-z])(?P<cen>" + _CEN_EN + r")" + S0 + r"\(" + S0 + _PRE_APPROX + _YC + S0 + r"\)", re.I)
_RX_C1T = re.compile(r"(?<![0-9A-Za-z.,])" + _YC + S0 + r"[(,]" + S0 + _CEN_TH + S0 + r"\)?", re.I)
_RX_C2T = re.compile(_CEN_TH + S0 + r"[(,]" + S0 + _YC + S0 + r"\)?", re.I)
_DEC = r"(?P<dy>\d{4})(?:'s|’s|s)(?![A-Za-z])"
_RX_D1 = re.compile(r"(?<![0-9A-Za-z.,])" + _YC + S0 + r"\(" + S0 + r"(?:the" + S1 + r")?" + _DEC + S0 + r"\)", re.I)
_RX_D2 = re.compile(r"(?<![0-9A-Za-z.,])(?:the" + S1 + r")?" + _DEC + S0 + r"\(" + S0 + _YC + S0 + r"\)", re.I)


def _century_of(y: int) -> int:
    return (y - 1) // 100 + 1


def _check_century(s: str) -> list:
    out = []
    for rx, thai, rev in ((_RX_C1, False, False), (_RX_C2, False, True), (_RX_C1T, True, False), (_RX_C2T, True, True)):
        for m in rx.finditer(s):
            g = m.groupdict()
            if _AFTER_BC.match(s, m.end()):
                continue
            if thai:
                c = int(g["c"])
                buddh_only = bool(g.get("b"))
            else:
                ce_txt = g.get("cen") or g.get("cen2")
                suffix_c = g.get("c") or g.get("c2")
                rom = g.get("r") or g.get("r2")
                c = int(suffix_c) if suffix_c else (_roman(rom) if rom else None)
                buddh_only = False
            if c is None or not (1 <= c <= 30):
                continue
            yr = int(g["y"])
            if yr < 1:
                continue
            r = _year_ce(yr, g.get("e1"), g.get("e2")) if yr >= 100 else (yr, False)
            if r is None:
                continue
            ce_year, was_be = r
            if ce_year < 1:
                continue
            ce_c = _century_of(ce_year)
            be_c = _century_of(yr)
            shown = f"{yr}" + (f" (= {ce_year} CE)" if was_be else "")
            accepted = {ce_c: "the CE century"}
            if was_be or thai:
                if was_be:
                    accepted[be_c] = "the Buddhist-Era century"
            if buddh_only and was_be:
                accepted = {be_c: "the Buddhist-Era century"}
            if c in accepted:
                out.append(_cand(m.start(), m.end(), "century", V,
                                 f"The year {shown} is in the {_ordinal(c)} century ({accepted[c]}).", 2,
                                 (f"computed: {ce_c}",)))
                continue
            if (not was_be) and ce_year % 100 == 0 and c == ce_c + 1:
                out.append(_cand(m.start(), m.end(), "century", U,
                                 f"{ce_year} is strictly the last year of the {_ordinal(ce_c)} century, but some count it "
                                 f"as the start of the {_ordinal(c)}.", 2))
                continue
            rng = f"{(ce_c - 1) * 100 + 1}-{ce_c * 100}"
            extra = f" ({_ordinal(be_c)} century in the Buddhist Era)" if was_be else ""
            out.append(_cand(m.start(), m.end(), "century", C,
                             f"The year {shown} falls in the {_ordinal(ce_c)} century ({rng}){extra}, not the {_ordinal(c)}.", 2,
                             (f"computed: {ce_c}", f"claimed: {c}")))
    for rx, rev in ((_RX_D1, False), (_RX_D2, True)):
        for m in rx.finditer(s):
            g = m.groupdict()
            dy = int(g["dy"])
            yr = int(g["y"])
            r = _year_ce(yr, g.get("e1"), g.get("e2"))
            if r is None:
                continue
            ce_year = r[0]
            was_be = r[1]
            if dy % 10:
                continue
            if was_be:
                continue                 # decades in BE: too convention-heavy
            if dy % 100 == 0:
                if dy <= ce_year <= dy + 9:
                    out.append(_cand(m.start(), m.end(), "century", V,
                                     f"{ce_year} falls in the {dy}s.", 2))
                elif dy + 10 <= ce_year <= dy + 99:
                    out.append(_cand(m.start(), m.end(), "century", U,
                                     f"{ce_year} is in the {dy}s only if \"{dy}s\" means the whole century {dy}-{dy + 99}, "
                                     f"not the decade {dy}-{dy + 9}.", 2))
                else:
                    out.append(_cand(m.start(), m.end(), "century", C,
                                     f"{ce_year} is outside the {dy}s ({dy}-{dy + 99}).", 2))
            else:
                if dy <= ce_year <= dy + 9:
                    out.append(_cand(m.start(), m.end(), "century", V, f"{ce_year} falls in the {dy}s.", 2))
                else:
                    out.append(_cand(m.start(), m.end(), "century", C,
                                     f"{ce_year} is outside the {dy}s ({dy}-{dy + 9}).", 2))
    return out


# ------------------------------------------------------------- 6: unit conversion ---

F = Fraction
_NUM = r"[-−+]?(?:\d{1,3}(?:,\d{3}){1,8}|\d{1,30})(?:\.\d{1,30})?"


class _Unit:
    def __init__(self, uid, dim, factor, disp, pats, alts=(), quote=False):
        self.id, self.dim, self.factor, self.disp = uid, dim, factor, disp
        self.pats, self.alts, self.quote = pats, list(alts), quote


def _u(uid, dim, factor, disp, pats, alts=(), quote=False):
    return _Unit(uid, dim, F(factor), disp, pats, alts, quote)


_LEN, _MASS, _VOL, _AREA, _SPEED, _TEMP = "length", "mass", "volume", "area", "speed", "temperature"
_GAL_IMP = ("the imperial gallon (4.546 L)", F("4.54609"))
_FLOZ_IMP = ("the imperial fluid ounce (28.41 mL)", F("0.0284130625"))
_PINT_IMP = ("the imperial pint (0.568 L)", F("0.56826125"))
_NMI = ("the nautical mile (1.852 km)", F(1852))

_UNITS = [
    _u("km", _LEN, 1000, "km", [(r"kilomet(?:er|re)s?", 1), (r"km", 1), (r"กิโลเมตร", 0), (r"กม\.", 0)]),
    _u("m", _LEN, 1, "m", [(r"met(?:er|re)s?", 1), (r"m", 0), (r"เมตร", 0)]),
    _u("cm", _LEN, F(1, 100), "cm", [(r"centimet(?:er|re)s?", 1), (r"cm", 1), (r"เซนติเมตร", 0), (r"ซม\.", 0)]),
    _u("mm", _LEN, F(1, 1000), "mm", [(r"millimet(?:er|re)s?", 1), (r"mm", 1), (r"มิลลิเมตร", 0), (r"มม\.", 0)]),
    _u("mi", _LEN, F("1609.344"), "mi", [(r"miles?", 1), (r"mi", 0), (r"ไมล์", 0)], [_NMI]),
    _u("mi_s", _LEN, F("1609.344"), "mi", [(r"statute[ \t]miles?", 1)]),
    _u("nmi", _LEN, 1852, "nautical miles", [(r"nautical[ \t]miles?", 1), (r"nmi", 1), (r"ไมล์ทะเล", 0)]),
    _u("ft", _LEN, F("0.3048"), "ft", [(r"feet", 1), (r"foot", 1), (r"ft", 1), (r"ฟุต", 0), (r"[′']", 0)], quote=True),
    _u("in", _LEN, F("0.0254"), "in", [(r"inch(?:es)?", 1), (r"in", 0), (r"นิ้ว", 0), (r"[″\"]", 0)], quote=True),
    _u("yd", _LEN, F("0.9144"), "yd", [(r"yards?", 1), (r"yd", 1)]),
    _u("kg", _MASS, 1, "kg", [(r"kilogram(?:me)?s?", 1), (r"kgs?", 1), (r"กิโลกรัม", 0), (r"กก\.", 0)]),
    _u("g", _MASS, F(1, 1000), "g", [(r"gram(?:me)?s?", 1), (r"g", 0), (r"กรัม", 0)]),
    _u("lb", _MASS, F("0.45359237"), "lb", [(r"pounds?", 1), (r"lbs?\.?", 1), (r"ปอนด์", 0)]),
    _u("oz", _MASS, F("0.028349523125"), "oz", [(r"ounces?", 1), (r"oz\.?", 1), (r"ออนซ์", 0)]),
    _u("L", _VOL, 1, "L", [(r"lit(?:er|re)s?", 1), (r"[Ll]", 0), (r"ลิตร", 0)]),
    _u("mL", _VOL, F(1, 1000), "mL", [(r"millilit(?:er|re)s?", 1), (r"ml", 1), (r"มิลลิลิตร", 0), (r"มล\.", 0)]),
    _u("gal", _VOL, F("3.785411784"), "US gal", [(r"gal(?:lons?)?\.?", 1), (r"แกลลอน", 0)], [_GAL_IMP]),
    _u("gal_us", _VOL, F("3.785411784"), "US gal", [(r"(?:US|U\.S\.)[ \t]gal(?:lons?)?\.?", 0)]),
    _u("gal_imp", _VOL, F("4.54609"), "imperial gal",
       [(r"(?:imp(?:erial)?\.?|UK|U\.K\.)[ \t]gal(?:lons?)?\.?", 1)]),
    _u("floz", _VOL, F("0.0295735295625"), "US fl oz", [(r"fl\.?[ \t]?oz\.?", 1), (r"fluid[ \t]ounces?", 1)], [_FLOZ_IMP]),
    _u("pint", _VOL, F("0.473176473"), "US pt", [(r"pints?", 1)], [_PINT_IMP]),
    _u("pint_us", _VOL, F("0.473176473"), "US pt", [(r"(?:US|U\.S\.)[ \t]pints?", 0)]),
    _u("pint_imp", _VOL, F("0.56826125"), "imperial pt", [(r"(?:imp(?:erial)?\.?|UK)[ \t]pints?", 1)]),
    _u("m2", _AREA, 1, "m²", [(r"m2", 0), (r"m²", 0), (r"sq\.?[ \t]?m", 1), (r"square[ \t]met(?:er|re)s?", 1),
                               (r"ตารางเมตร", 0), (r"ตร\.ม\.", 0)]),
    _u("ft2", _AREA, F("0.09290304"), "ft²", [(r"ft2", 1), (r"ft²", 1), (r"sq\.?[ \t]?ft", 1), (r"square[ \t]f(?:eet|oot)", 1),
                                               (r"ตารางฟุต", 0)]),
    _u("km2", _AREA, 1000000, "km²", [(r"km2", 1), (r"km²", 1), (r"sq\.?[ \t]?km", 1), (r"square[ \t]kilomet(?:er|re)s?", 1)]),
    _u("mi2", _AREA, F("2589988.110336"), "mi²", [(r"mi2", 1), (r"mi²", 1), (r"sq\.?[ \t]?mi", 1), (r"square[ \t]miles?", 1)]),
    _u("ha", _AREA, 10000, "ha", [(r"hectares?", 1), (r"ha", 1), (r"เฮกตาร์", 0)]),
    _u("acre", _AREA, F("4046.8564224"), "acres", [(r"acres?", 1), (r"เอเคอร์", 0)]),
    _u("kmh", _SPEED, F(1000, 3600), "km/h", [(r"km/h", 1), (r"km/hr", 1), (r"kph", 1), (r"kmh", 1),
                                              (r"kilomet(?:er|re)s?[ \t]per[ \t]hour", 1), (r"กม\./ชม\.", 0),
                                              (r"กิโลเมตรต่อชั่วโมง", 0)]),
    _u("mph", _SPEED, F("0.44704"), "mph", [(r"mph", 1), (r"mi/h", 1), (r"mi/hr", 1), (r"miles?[ \t]per[ \t]hour", 1),
                                            (r"ไมล์ต่อชั่วโมง", 0)]),
    _u("ms", _SPEED, 1, "m/s", [(r"m/s", 1), (r"met(?:er|re)s?[ \t]per[ \t]second", 1), (r"เมตรต่อวินาที", 0)]),
    _u("C", _TEMP, 1, "°C", [(r"[°º˚][ \t]?C(?:elsius)?", 0), (r"℃", 0), (r"degrees?[ \t]+(?:C(?:elsius)?|centigrade)", 1),
                              (r"deg\.?[ \t]?C", 0), (r"องศาเซลเซียส", 0), (r"องศา[ \t]?C", 0)]),
    _u("F", _TEMP, 1, "°F", [(r"[°º˚][ \t]?F(?:ahrenheit)?", 0), (r"℉", 0), (r"degrees?[ \t]+F(?:ahrenheit)?", 1),
                              (r"deg\.?[ \t]?F", 0), (r"องศาฟาเรนไฮต์", 0), (r"องศา[ \t]?F", 0)]),
]
_UNIT_BY_ID = {u.id: u for u in _UNITS}


def _build_unit_rx():
    entries = []
    for u in _UNITS:
        for pat, ci in u.pats:
            entries.append((u.id, pat, ci))
    entries.sort(key=lambda e: -len(re.sub(r"\\.|[\[\]()?:]", "", e[1])))
    parts = []
    for i, (uid, pat, ci) in enumerate(entries):
        parts.append(f"(?P<u{i}>(?{'i' if ci else '-i'}:{pat}))")
    return "(?:" + "|".join(parts) + ")", {f"u{i}": e[0] for i, e in enumerate(entries)}


_UNIT_RX, _UNIT_GROUPS = _build_unit_rx()
_QTY = re.compile(
    r"(?<![0-9A-Za-z.,])(?P<a1>" + _NUM + r")(?:" + S0 + r"(?P<op>-|–|—|to)" + S0 + r"(?P<a2>" + _NUM + r"))?"
    r"(?P<gap>[ \t]{0,3}-?[ \t]{0,3})" + _UNIT_RX + r"(?![A-Za-z0-9²³/])", re.I)
_HEDGE = (r"(?:about|approximately|approx\.?|roughly|around|circa|c\.|~|≈|=|or|i\.e\.|equal[ \t]+to|"
          r"equivalent[ \t]+to|that[ \t]+is|that's|which[ \t]+is)")
_INEQ = (r"(?:nearly|almost|over|under|more[ \t]+than|less[ \t]+than|at[ \t]+least|at[ \t]+most|up[ \t]+to|"
         r"just[ \t]+(?:over|under)|exceeding)")
_SEP_PAREN = re.compile(r"[ \t]{0,3}\([ \t]{0,3}(?:(?P<h>" + _HEDGE + r"|" + _INEQ + r")[ \t]{0,3})?", re.I)
_SEP_COMMA = re.compile(r"[ \t]{0,3},[ \t]{1,3}(?:or[ \t]{1,3})?(?P<h>" + _HEDGE + r"|" + _INEQ + r")[ \t]{0,3}", re.I)
_SEP_WORD = re.compile(r"[ \t]{1,3}(?P<c>or|which[ \t]+is|is|are|was|equals|equal[ \t]+to)[ \t]{1,3}"
                       r"(?:(?P<h>" + _HEDGE + r"|" + _INEQ + r")[ \t]{0,3})?", re.I)
_SEP_SYM = re.compile(r"[ \t]{0,3}(?:=|≈|≅)[ \t]{0,3}")
_INEQ_BEFORE = re.compile(r"(?:" + _INEQ + r")[ \t]{0,3}\Z", re.I)
_MULT_BEFORE = re.compile(r"\d[ \t]{0,3}(?:x|×|\*|by)[ \t]{0,3}\Z", re.I)
_DELTA = re.compile(r"(?:rise|rose|risen|rising|increase[ds]?|increasing|drop(?:ped|s)?|fall|fell|falling|decrease[ds]?|"
                    r"differ(?:ence|ent|s)?|change[ds]?|warmer|cooler|colder|hotter|higher|lower|swing|gain(?:ed)?|"
                    r"lose|lost|by|up|down|variation|within|±|plus or minus)", re.I)


class _Q:
    __slots__ = ("start", "end", "unit", "vals", "ds", "tz", "texts", "gap", "raw_unit", "is_range", "compound")


def _num(txt: str):
    t = txt.replace("−", "-")
    neg = t.startswith("-")
    t = t.lstrip("+-")
    if "," in t:
        t = t.replace(",", "")
    ip, _, fp = t.partition(".")
    val = F(int(ip + fp), 10 ** len(fp))
    tz = 0
    if not fp:
        stripped = ip.rstrip("0")
        tz = len(ip) - len(stripped) if stripped else 0
    return (-val if neg else val), len(fp), tz


def _quantities(s: str) -> list:
    qs = []
    for m in _QTY.finditer(s):
        q = _Q()
        q.start, q.end = m.start(), m.end()
        uname = m.lastgroup
        if uname not in _UNIT_GROUPS:
            uname = next((k for k in _UNIT_GROUPS if m.group(k) is not None), None)
            if uname is None:
                continue
        q.unit = _UNIT_BY_ID[_UNIT_GROUPS[uname]]
        q.raw_unit = m.group(uname)
        q.gap = m.group("gap")
        if q.unit.quote and q.raw_unit in ("'", "′", '"', "″") and q.gap:
            continue
        if q.unit.id == "L" and q.raw_unit in ("l", "L") and False:
            continue
        vals, ds, tzs, texts = [], [], [], []
        for g in ("a1", "a2"):
            t = m.group(g)
            if t is None:
                continue
            v, d, tz = _num(t)
            vals.append(v)
            ds.append(d)
            tzs.append(tz)
            texts.append(t.lstrip("+"))
        q.vals, q.ds, q.tz, q.texts = vals, ds, tzs, texts
        q.is_range = len(vals) == 2
        q.compound = False
        qs.append(q)
    # merge compound quantities: 5 ft 9 in, 7 lb 4 oz
    merged = []
    for q in qs:
        if merged:
            p = merged[-1]
            if (not p.is_range and not q.is_range and not p.compound
                    and ((p.unit.id == "ft" and q.unit.id == "in") or (p.unit.id == "lb" and q.unit.id == "oz"))
                    and re.fullmatch(r"[ \t]{0,3}(?:,|and)?[ \t]{0,3}", s[p.end:q.start])):
                mult = 12 if p.unit.id == "ft" else 16
                p.vals = [p.vals[0] * mult + q.vals[0]]
                p.ds, p.tz = q.ds, [0]
                p.texts = [f"{p.texts[0]} {p.unit.disp} {q.texts[0]}"]
                p.unit = q.unit
                p.end = q.end
                p.compound = True
                continue
        merged.append(q)
    return merged


def _fmt_num(x: Fraction) -> str:
    if x == 0:
        return "0"
    ax = abs(x)
    mag = math.floor(math.log10(float(ax))) if ax > 0 else 0
    if mag >= 15 or mag < -6:
        return f"{float(x):.3g}"
    if mag >= 3:
        r = int(round(float(x)))
        return f"{r:,}"
    decimals = max(0, 2 - mag)
    txt = f"{float(x):.{decimals}f}"
    if "." in txt:
        txt = txt.rstrip("0").rstrip(".")
    return txt


def _half(d: int) -> Fraction:
    return F(1, 2 * 10 ** d)


def _fits(stated: Fraction, expected: Fraction, half: Fraction) -> bool:
    return abs(stated - expected) <= half + abs(expected) / 100


def _conv_linear(v: Fraction, ffrom: Fraction, fto: Fraction) -> Fraction:
    return v * ffrom / fto


def _conv_temp(v: Fraction, uf: str, delta: bool) -> Fraction:
    if uf == "C":
        return v * F(9, 5) if delta else v * F(9, 5) + 32
    return v * F(5, 9) if delta else (v - 32) * F(5, 9)


def _variants(u: _Unit):
    yield None, u.factor
    for label, f in u.alts:
        yield label, f
