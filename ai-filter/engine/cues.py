"""Topic-independent risk cues: signals in the TEXT of an answer that raise the
chance it contains an error the reader should check, on ANY subject (history,
cooking, medicine, law, sport, code ...) where there is no link, DOI or sum to verify.

    scan(text, *, question=None, today=None) -> list[Finding]

A cue NEVER says the answer is wrong.  Every Finding is UNVERIFIED, module="cue",
claim.kind == ClaimKind.CUE, claim.data == {"cue": tag}, Finding.tag == tag, ids
"q1", "q2", ... in document order, `text == source[start:end]` exactly.

Method (shared by all cues)
---------------------------
1. Fenced and inline code are blanked with NULs (offsets and newlines preserved) and
   take no part in anything.  URLs are blanked for every cue except the URL cue and the
   "is there a source nearby" test.  Matching runs on the ORIGINAL string (no
   normalisation), so offsets are exact by construction, also after emoji, Thai combining
   marks and CRLF.
2. A small sentence splitter (see _sentences) gives "same sentence" its meaning.  Thai
   spaces are NOT boundaries, so a Thai trigger only looks +-100 characters around itself
   (English: +-300) when it needs a second signal in "the same sentence".
3. Precision over recall: when unsure a cue applies, it does not fire.  time_sensitive and
   absolute_claim fire at most once per sentence (the first qualifying trigger).  A single
   cue never fires twice on overlapping spans; different cues may overlap.

The cues (tag: what, why it signals risk, KNOWN false positive)
---------------------------------------------------------------
time_sensitive   "currently", "as of", "latest", "ปัจจุบัน", or a year >= today.year-1,
                 in a sentence that also states a changeable fact (a number, a capitalised
                 name, an office/price/record/version word).  Risk: a model has a training
                 cutoff, so "who is CEO now" / "latest version" answers go stale.
                 FP: "Today we have 3 people in the kitchen" (a number, nothing that
                 changes), "Today's menu is Pad Thai" (capitalised dish name).
absolute_claim   always / never / all / every / none / guaranteed / 100% / cures /
                 completely safe ... but ONLY in a sentence with a health, drug, legal, tax,
                 money or safety word.  Risk: absolutes in high-stakes domains are almost
                 never exactly true and are where a wrong answer hurts.
                 FP: "All taxes must be filed by April" (a correct, sourced-elsewhere rule);
                 Thai "รับประกัน" also means an ordinary product warranty.
ai_self_disclosure
                 the answer admits its own limits ("as an AI", "my knowledge cutoff",
                 "please verify").  Risk: the model flagged its own uncertainty, so the
                 facts around it deserve care.
                 FP: "As an AI company we ..." in a business answer; boilerplate "please
                 verify your email".
placeholder_url  a URL that is a template (example.com, your-site, xxx, ..., /path/to/,
                 lorem, 12345, localhost, [link], <url>).  Risk: a placeholder address
                 cannot be a source, so the claim it "backs" is unsourced.
                 FP: a tutorial that legitimately says "open http://localhost:3000".
unsourced_quote  a long quoted passage (>= 8 words / >= 30 Thai chars) attributed to a
                 named person or organisation with no source in the same/next sentence.
                 Risk: AI often fabricates plausible-sounding quotations.
                 FP: a genuine, widely known quotation whose source the author left out.
named_expert_claim
                 "Dr./Prof./ดร./ศ./รศ./ผศ. <Name>" plus an affiliation or a finding verb,
                 with no citation nearby.  Risk: invented experts are a classic AI error.
                 (vague.py counts a named person as BACKING; this cue says "unverifiable"
                 - a different purpose.)  FP: a real, famous professor named in passing.
count_mismatch   the answer announces N items ("3 ways", "top 10", "3 ข้อ") and the list
                 directly after it has a different length (a clearly labelled trailing
                 conclusion item is allowed).  Risk: something was dropped or invented.
                 FP: an announcement about a subset ("3 key points" above a longer list).
question_asks_for_sources
                 the QUESTION asks for sources/statistics/studies/quotes/links/"latest
                 figures" but the answer contains no URL, DOI, [n] or author-year citation.
                 Risk: models invent citations when asked for them without search.
                 FP: the user wanted a plain explanation and only used the word "sources".

Interplay with vague.py (do not duplicate it): vague flags "studies show", "experts say",
"it has been proven that", "is proven to" and over-precise figures.  absolute_claim skips
those exact "proven" shapes and only takes the adjective-style ones ("100% proven",
"proven cure").  Nothing else here overlaps.
"""

from __future__ import annotations

import datetime as _dt
import re
from bisect import bisect_right
from typing import Optional

from .models import Claim, ClaimKind, Finding, Status

__all__ = ["scan"]

MODULE = "cue"

# ===========================================================================
# WORD TABLE  --  every word list lives here.  English entries are regex fragments
# (case-insensitive, word-bounded by the compiled pattern); Thai entries are plain
# substrings (Thai has no word separators) guarded with lookarounds where a
# fragment is also part of common words.
# ===========================================================================

# -- time_sensitive ---------------------------------------------------------
_TIME_EN = [r"currently", r"as\s+of(?!\s+yet)", r"at\s+present", r"right\s+now", r"today", r"this\s+year",
            r"latest", r"newest", r"most\s+recent", r"up[-\s]to[-\s]date"]
_TIME_TH = ["ล่าสุด", "ปัจจุบัน", "ขณะนี้", "ตอนนี้", "ปีนี้"]
# a "changeable fact" is a number, a capitalised name, or one of these office/price words
_OFFICE_EN = [r"ceo", r"cto", r"cfo", r"president", r"vice[-\s]president", r"prime\s+minister", r"chancellor",
              r"mayor", r"governor", r"minister", r"king", r"queen", r"leader", r"chairman", r"chairwoman",
              r"director", r"coach", r"manager", r"champions?", r"holder", r"records?", r"prices?", r"costs?",
              r"fees?", r"rates?", r"versions?", r"released?", r"ranking", r"ranked", r"population",
              r"market\s+cap", r"valuation", r"salary", r"winner", r"inflation", r"gdp", r"unemployment"]
_OFFICE_TH = ["ประธานาธิบดี", "นายกรัฐมนตรี", "นายก", "รัฐมนตรี", "ซีอีโอ", "ผู้บริหาร", "แชมป์", "สถิติ", "ราคา",
              "เวอร์ชัน", "เวอร์ชั่น", "รุ่น", "อันดับ", "อัตรา", "ประชากร", "ผู้ว่า", "กษัตริย์", "ผู้นำ", "ผู้จัดการ",
              "โค้ช", "ดอกเบี้ย", "อัตราแลกเปลี่ยน", "ผู้ชนะ", "ยอดขาย", "มูลค่า", "เงินเฟ้อ", "ค่าเงิน", "ค่าธรรมเนียม"]
# capitalised words that are NOT names (so "Today is Monday" is not a "changeable fact")
_NOT_NAMES = {"I", "I'm", "I'll", "I've", "I'd", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
              "Sunday", "January", "February", "March", "April", "May", "June", "July", "August", "September",
              "October", "November", "December", "English", "Thai", "French", "German", "Chinese", "Japanese",
              "Spanish", "Italian", "American", "British", "Please", "Note", "Yes", "No", "OK"}

# -- absolute_claim ---------------------------------------------------------
# (the trigger regex itself is _ABS_RE below; the families need per-family skip rules)
_DOMAIN_EN = [r"health\w*", r"medic\w*", r"drugs?", r"disease\w*", r"illness\w*", r"cancer\w*", r"diabet\w*",
              r"diet\w*", r"dos(?:e|es|age)", r"symptoms?", r"treatments?", r"therap(?:y|ies|ist|ists|eutic)",
              r"vaccin\w+", r"supplements?", r"pregnan\w+", r"surgery", r"surgical", r"infect\w+", r"allerg\w+",
              r"antibiotics?", r"doctors?", r"patients?", r"legal\w*", r"illegal", r"laws?", r"lawful", r"lawsuits?",
              r"lawyers?", r"attorneys?", r"courts?", r"tax", r"taxes", r"taxation", r"taxable",
              r"invest(?:s|ed|ing|ment|ments|or|ors)?", r"(?:rate\s+of|annual|expected|investment|average|stock|market)\s+returns?",
              r"return\s+on", r"stocks", r"stock\s+market", r"crypto\w*", r"portfolios?", r"loans?", r"mortgages?",
              r"interest\s+rates?", r"safety", r"unsafe", r"safe", r"dangerous", r"toxic", r"poison\w*", r"hazard\w*"]
_DOMAIN_TH = ["สุขภาพ", "โรค", "วัคซีน", "มะเร็ง", "เบาหวาน", "ความดัน", "อาหารเสริม", "ผลข้างเคียง", "ลดน้ำหนัก",
              "ตั้งครรภ์", "ติดเชื้อ", "ผ่าตัด", "แพทย์", "การรักษา", "รักษาโรค", "กฎหมาย", "ภาษี", "ลงทุน",
              "ผลตอบแทน", "หุ้น", "กองทุน", "เงินกู้", "ดอกเบี้ย", "ปลอดภัย", "อันตราย", "สารพิษ", "เป็นพิษ", "คดี",
              "ศาล(?!า)",
              # "ยา" (medicine) is also inside ความยาว, พยาบาล, วิทยา ...: only after a verb or a non-letter
              r"(?:(?<=กิน)|(?<=ทาน)|(?<=ใช้)|(?<=รับประทาน)|(?<=ฉีด)|(?<=ซื้อ)|(?<![ก-ฺเ-ๆ]))ยา(?![วก])"]

# -- ai_self_disclosure -----------------------------------------------------
_SELF_EN = [
    r"as\s+an?\s+(?:ai|artificial\s+intelligence)\b(?!\s+(?:researcher|engineer|expert|developer|enthusiast|startup|"
    r"company|scientist|practitioner|user|professional|consultant|lab|team|writer|artist))",
    r"i\s+(?:don['’]t|do\s+not)\s+have\s+(?:direct\s+)?(?:access\s+to\s+)?(?:real[-\s]?time|current|up[-\s]to[-\s]date|live)",
    r"i\s+(?:can(?:not|['’]t)|am\s+(?:unable|not\s+able)\s+to)\s+(?:browse|access\s+the\s+(?:internet|web)|"
    r"search\s+the\s+(?:internet|web)|look\s+up)",
    r"my\s+(?:training|knowledge)(?:\s+data)?(?:\s+cut-?\s?off|\s+is\s+limited|\s+only\s+(?:goes|extends)|\s+ends)",
    r"(?:knowledge|training)(?:\s+data)?\s+cut-?\s?off",
    r"i\s+(?:may|might|could)\s+(?:be\s+)?(?:inaccurate|wrong|mistaken|incorrect|out\s+of\s+date|outdated)",
    r"i(?:['’]m|\s+am)\s+not\s+(?:certain|sure|100%\s+sure|completely\s+sure)",
    r"please\s+(?:verify|double[-\s]check|fact[-\s]check)",
    r"i\s+(?:would\s+)?(?:recommend|suggest|advise)\s+(?:verifying|double[-\s]checking)",
]
_SELF_TH = [
    r"ในฐานะ\s?(?:AI|เอไอ|ปัญญาประดิษฐ์|โมเดลภาษา|แชทบอท)",
    r"ข้อมูล(?:การฝึก)?(?:ที่|ของ)?(?:ฉัน|ผม|ดิฉัน)(?:มี)?(?:อัปเดต|อัพเดต)?(?:ล่าสุด)?ถึง(?:ปี|เดือน)",
    r"ไม่สามารถ(?:เข้าถึง|ค้นหา|เรียกดู|ท่อง)(?:ข้อมูล)?(?:แบบ|ที่เป็น)?(?:เรียลไทม์|เรียลไทม์|ปัจจุบัน|ล่าสุด|อินเทอร์เน็ต|อินเตอร์เน็ต|เว็บ)",
    r"(?:ฉัน|ผม|ดิฉัน)(?:อาจ|อาจจะ)(?:ผิดพลาด|ไม่ถูกต้อง|ผิด|คลาดเคลื่อน)",
    r"(?:ฉัน|ผม|ดิฉัน)ไม่(?:แน่ใจ|มั่นใจ)",
    r"(?:โปรด|กรุณา)(?:ตรวจสอบ|ยืนยัน)",
]

# -- placeholder_url --------------------------------------------------------
_PLACEHOLDER_HOSTS = (r"localhost", r"127\.0\.0\.1", r"0\.0\.0\.0", r"(?:[a-z0-9-]+\.)*example\.(?:com|org|net)")
_PLACEHOLDER_ANY = (r"your[-_]?(?:site|domain|website|company|url|link|username|name|repo|project)", r"yoursite",
                    r"yourdomain", r"lorem", r"link-to-", r"insert[-_]", r"/path/to/")
_PLACEHOLDER_SEGMENT = r"x{3,}|\.{2,}|…+|12345|lorem|\{[^/]*\}|<[^/]*>|\[[^/]*\]"
_PLACEHOLDER_TOKENS = [
    r"\[(?:link|url|insert\s+(?:a\s+)?(?:link|url|source)[^\]\n]{0,20}|your\s+(?:link|url)|link\s+here|url\s+here|"
    r"source\s+link|citation\s+link)\](?!\()",
    r"<(?:url|link|your[-_\s]?(?:url|link|domain|site|website)[^<>\n]{0,20}|insert[^<>\n]{0,30}|"
    r"link[-_\s]?here|url[-_\s]?here|source[-_\s]?url)>",
    r"\{\{?\s*(?:url|link|source)\s*\}\}?",
    r"https?://<[^<>\s]{1,40}>[^\s<>\"'`]*",
]

# -- unsourced_quote / named_expert_claim -----------------------------------
_SAY_VERBS = (r"said|says|wrote|writes|stated|states|argued|argues|told|noted|added|explained|declared|claimed|"
              r"remarked|insisted|announced|warned|continued|observed|admitted|asserted|put\s+it")
_SAY_VERBS_TH = ["กล่าวไว้ว่า", "เขียนไว้ว่า", "ระบุไว้ว่า", "กล่าวว่า", "เขียนว่า", "ระบุว่า", "พูดว่า", "บอกว่า",
                 "ประกาศว่า", "เปิดเผยว่า", "ย้ำว่า"]
_THAI_GENERIC_SUBJECTS = ("เขา", "เธอ", "เค้า", "มัน", "ท่าน", "พวกเขา", "ผู้", "นักวิจัย", "นักวิทยาศาสตร์", "แพทย์",
                          "หมอ", "เจ้าหน้าที่", "โฆษก", "รายงาน", "ผลวิจัย", "งานวิจัย", "คนที่", "ใคร", "ซึ่ง", "ที่",
                          "ก็", "จึง", "ยัง", "ได้", "เคย", "เอง", "นั้น", "นี้", "ไว้", "แล้ว", "และ", "หรือ")
_GENERIC_EN = {"he", "she", "they", "i", "we", "it", "you", "this", "that", "who", "one", "someone", "many", "some",
               "others", "everyone", "the", "a", "an", "experts", "scientists", "researchers", "doctors", "critics",
               "officials", "sources", "analysts", "economists", "historians", "authorities", "teachers", "reports",
               "studies", "people", "users", "customers", "investors", "residents", "witnesses", "police",
               "spokesperson", "nobody", "everybody", "however", "then", "later", "also", "he/she", "yes", "no"}
_FIND_VERBS_EN = (r"found|finds|says|said|argues|argued|claims|claimed|reports|reported|showed|shows|discovered|"
                  r"concluded|recommends|recommended|warns|warned|stated|states|explains|explained|notes|noted|"
                  r"believes|believed|suggests|suggested|proved|proposed|demonstrated|published|developed|"
                  r"invented|coined|pioneered|observed|revealed")
_FIND_VERBS_TH = ("พบว่า", "ค้นพบ", "ระบุ", "กล่าว", "เปิดเผย", "แนะนำ", "เตือน", "ยืนยัน", "อธิบาย", "สรุป", "เผย",
                  "ชี้", "เห็นว่า", "เชื่อว่า", "ให้สัมภาษณ์")
_AFFIL_TH = ("จาก", "แห่ง", "ประจำ", "สังกัด")

# -- count_mismatch ---------------------------------------------------------
_COUNT_NOUNS_EN = (r"ways|reasons|steps|tips|tricks|things|types|kinds|benefits|factors|ideas|methods|rules|signs|"
                   r"tools|items|points|principles|examples|features|options|mistakes|habits|strategies|stages|"
                   r"phases|causes|ingredients|differences|advantages|disadvantages|pros|cons|lessons|takeaways|"
                   r"questions|myths|facts|characteristics|qualities|elements|components|categories|levels|risks|"
                   r"symptoms|requirements|considerations|challenges|resources|techniques|approaches|"
                   r"recommendations|key\s+points|main\s+points|exercises|foods|places|books|movies|apps|skills|goals|"
                   r"countries|cities|languages|dishes|recipes|destinations|activities")
_COUNT_NOUNS_TH = ("อย่าง", "ข้อ", "ขั้นตอน", "วิธี", "เหตุผล", "ประการ", "เคล็ดลับ", "ประเภท", "ชนิด", "แบบ",
                   "ปัจจัย", "แนวทาง", "สัญญาณ", "ตัวเลือก", "ประเด็น", "เทคนิค", "สิ่ง", "ข้อดี", "ข้อเสีย", "ขั้น",
                   "หลักการ", "ข้อควรระวัง", "ข้อควรรู้", "รายการ", "กลยุทธ์", "แนวคิด", "อาการ", "สาเหตุ", "ตัวอย่าง")
_NUM_WORDS_EN = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
                 "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
                 "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20}
_NUM_WORDS_TH = {"หนึ่ง": 1, "สอง": 2, "สาม": 3, "สี่": 4, "ห้า": 5, "หก": 6, "เจ็ด": 7, "แปด": 8, "เก้า": 9,
                 "สิบ": 10, "สิบเอ็ด": 11, "สิบสอง": 12, "สิบสาม": 13, "สิบสี่": 14, "สิบห้า": 15, "สิบหก": 16,
                 "สิบเจ็ด": 17, "สิบแปด": 18, "สิบเก้า": 19, "ยี่สิบ": 20}
_CONCLUSION_LABEL = (r"conclusion|in\s+conclusion|summary|in\s+summary|to\s+summari[sz]e|to\s+sum\s+up|bonus(?:\s+tip)?|"
                     r"final\s+thoughts?|key\s+takeaway|takeaway|in\s+short|overall|สรุป|โบนัส|บทสรุป|ข้อสรุป")

# -- question_asks_for_sources ----------------------------------------------
_ASK_EN = [
    r"\b(?:cite|cites|citing|citations?|bibliograph\w+|footnotes?)\b",
    r"\breferences\b",
    r"\bwith\s+(?:sources|links|citations|references|evidence|proof|quotes|studies|statistics)\b",
    r"\b(?:provide|include|including|give|list|add|attach|share|show|need|want|find|get|send|any)\b[^.?!\n]{0,40}?"
    r"\b(?:sources?|references?|links?|urls?|quotes?|quotations?|statistics|stats|studies|papers|evidence|dois?)\b",
    r"\b(?:what|which)\s+(?:sources|studies|research|papers|statistics|stats)\b",
    r"\bwhere\s+(?:did|do)\s+you\s+(?:get|find|read)\b",
    r"\b(?:peer[-\s]reviewed|evidence[-\s]based)\b",
    r"\b(?:latest|most\s+recent|current|up[-\s]to[-\s]date)\s+(?:statistics|stats|figures|numbers|data|studies|research)\b",
    r"\b(?:statistics|stats)\s+(?:on|about|for)\b",
]
_ASK_TH = [
    r"อ้างอิง", r"แหล่งที่มา", r"แหล่งข้อมูล", r"ลิ(?:้|)ง(?:ก์|ค์|ก|ค)", r"สถิติ", r"เอกสารอ้างอิง", r"ยกคำพูด",
    r"(?:ตัวเลข|ข้อมูล)ล่าสุด",
    r"(?:ขอ|หา|มี|ให้|อ้าง|ยก|ช่วย|แนะนำ|พร้อม|ไหน|อะไร)[^\n]{0,25}(?:งานวิจัย|ผลวิจัย|ผลการวิจัย|ผลการศึกษา)",
    r"(?:งานวิจัย|ผลวิจัย|ผลการวิจัย|ผลการศึกษา)[^\n]{0,8}(?:ไหน|อะไร|บ้าง)",
]

# ===========================================================================
# Compiled patterns
# ===========================================================================
_L = r"(?<![A-Za-z])"
_R = r"(?![A-Za-z])"

_TIME_RE = re.compile(_L + "(?:" + "|".join(_TIME_EN) + ")" + _R + "|" + "|".join(_TIME_TH), re.I)
_KEEP_BEFORE = re.compile(r"(?:keep|keeps|kept|keeping|stay|stays|staying|get|remain|remains|make\s+sure|is|be)\s+(?:\w+\s+){0,3}\Z", re.I)
_YEAR_RE = re.compile(
    r"(?<![\d.,/:$€£฿#-])(?P<ce>(?:20|21)\d\d)(?![\d%]|[.,]\d|[-/]\d)"
    r"(?!\s?(?:calories|kcal|mg|ml|kg|km|baht|usd|dollars|people|words|users|บาท|คน|กรัม|แคล))"
    r"|(?:พ\.ศ\.|ปี)\s?(?P<be>25\d\d)(?![\d%]|[.,]\d)")
_CITE_YEAR = re.compile(r"[,(]\s*\Z")
_OFFICE_RE = re.compile(_L + "(?:" + "|".join(_OFFICE_EN) + ")" + _R + "|" + "|".join(_OFFICE_TH), re.I)
_NUM_RE = re.compile(r"(?<![A-Za-z])[0-9๐-๙]+")
_CAPTOK = re.compile(r"(?<![A-Za-z0-9'’-])[A-Z][A-Za-z0-9'’\-]*")
_LEADMARK = re.compile(r"[ \t>*#\-]*[0-9๐-๙]{1,3}[.)]\s")
_THAI_CH = re.compile(r"[ก-๛]")

_ABS_RE = re.compile("|".join([
    r"(?P<alwaysnever>" + _L + r"(?:always|never)" + _R + ")",
    r"(?P<all>(?<![\w-])all(?![\w-]))",
    r"(?P<every>" + _L + r"every" + _R + ")",
    r"(?P<none>" + _L + r"none" + _R + ")",
    r"(?P<guar>" + _L + r"guaranteed" + _R + ")",
    r"(?P<proven>" + _L + r"proven(?:\s+to)?" + _R + ")",
    r"(?P<phr>" + _L + r"(?:(?:completely|totally|absolutely|entirely|100%)\s+(?:safe|harmless|risk[-\s]free)|"
    r"no\s+(?:known\s+|serious\s+|significant\s+|adverse\s+)?side[\s-]+effects?|impossible|risk[-\s]free|foolproof)" + _R + ")",
    r"(?P<pct>(?<![\w.,$€£])100\s?%)",
    r"(?P<cure>" + _L + r"(?:(?:can|will|to|completely|fully|permanently|instantly|quickly|miracle)\s+cure|cures|cure-all)(?![A-Za-z-]))",
    # เสมอ is also in สม่ำเสมอ (regularly), เสมอภาค (equality), เสมอกัน (a draw)
    r"(?P<th>ทุกคน|(?<!ม่ำ)เสมอ(?!ภาค|กัน|ต้น|ปลาย|หน้า)|ไม่เคย|(?<!ไม่)(?<!ไม่ได้)รับประกัน|หายขาด|ทุกกรณี|ไม่มีผลข้างเคียง|ร้อยเปอร์เซ็นต์)",
]), re.I)
_ABS_NEG_BEFORE = re.compile(r"(?:not|n['’]t)\s+\Z", re.I)
_ABS_IMPERATIVE = re.compile(r"(?:you|should|must|please|to|can|could|let['’]s)\s+\Z", re.I)
_VAGUE_PROVEN_BEFORE = re.compile(
    r"(?:it\s+has\s+been|is|are|has\s+been|have\s+been)\s+(?:(?:scientifically|clinically|mathematically|medically)\s+)?\Z", re.I)
_PROVEN_ADV = re.compile(r"(?:100%|absolutely|definitively|totally|completely|scientifically|clinically|medically)\s+\Z", re.I)
_PROVEN_NEXT = re.compile(r"\s+(?:cure|safe|remedy|treatment|method|effective|fact)\b", re.I)
_ALL_BEFORE = re.compile(r"(?:after|at|above|of|not|first\s+of|most\s+of)\s+\Z", re.I)
_ALL_AFTER = re.compile(r"\s+(?:right|set|in|but)\b", re.I)
_EVERY_AFTER = re.compile(
    r"\s+(?:[0-9]|few|other|second|couple|day|night|morning|evening|afternoon|week|weekend|month|year|hour|minute|"
    r"one|two|three|four|six|eight|twelve)\b", re.I)
_NEVER_AFTER = re.compile(r"\s+(?:be\s+)?(?:guaranteed|certain|assured|promised)\b", re.I)
_GUAR_BEFORE = re.compile(r"(?:not|n['’]t|no|never)\s+\Z", re.I)
_DOMAIN_RE = re.compile(r"(?<![A-Za-z-])(?:" + "|".join(_DOMAIN_EN) + r")(?![A-Za-z])|" + "|".join(_DOMAIN_TH), re.I)

_SELF_RE = re.compile("|".join(_SELF_EN) + "|" + "|".join(_SELF_TH), re.I)

_URL_RE = re.compile(r"(?:https?://|ftp://|www\.)[^\s<>\"'`“”«»「」]+", re.I)
_BARE_EXAMPLE_RE = re.compile(r"(?<![@\w./-])(?:[a-z0-9-]+\.)*example\.(?:com|org|net)(?:/[^\s<>\"'`]*)?", re.I)
_TOKEN_RE = re.compile("|".join(_PLACEHOLDER_TOKENS), re.I)
_PH_ANY_RE = re.compile("|".join(_PLACEHOLDER_ANY), re.I)
_PH_HOST_RE = re.compile("(?:" + "|".join(_PLACEHOLDER_HOSTS) + ")$", re.I)
_PH_SEG_RE = re.compile("(?:" + _PLACEHOLDER_SEGMENT + ")$", re.I)

_Q_STRAIGHT = re.compile(r'(?<![A-Za-z0-9"])"(?=[^\s"])([^"\n]{20,2000}?)(?<=[^\s"])"(?![A-Za-z0-9])')
_Q_CURLY = re.compile(r"“([^“”\n]{20,2000}?)”")
_Q_ANGLE = re.compile(r"«([^«»\n]{20,2000}?)»")
_Q_CORNER = re.compile(r"「([^「」\n]{15,2000}?)」")
_WORD_RE = re.compile(r"[A-Za-z0-9À-ɏ]+(?:['’\-][A-Za-z0-9]+)*")

_NAME_EN = (r"(?:(?:Dr|Prof|Mr|Mrs|Ms|Sir|Professor|President|Senator|Sen|Rep|Gen|Gov)\.?\s+)?"
            r"(?:[A-Z][A-Za-z'’\-]+|[A-Z]\.)"
            r"(?:\s+(?:(?:of|the|de|van|von|der|bin|al|el|la|and|&)\s+)?(?:[A-Z][A-Za-z'’\-]+|[A-Z]\.)){0,4}")
_ATTR_PRE_EN = re.compile(
    r"(?P<name>" + _NAME_EN + r")\s+(?:[a-z]+\s+)?(?:" + _SAY_VERBS + r")\b[^\"“”«»「」\n]{0,60}\Z")
_ATTR_PRE_ACCORDING = re.compile(r"(?:according\s+to|as)\s+(?P<name>" + _NAME_EN + r")\s*[,:]?\s*\Z", re.I)
_ATTR_PRE_TH = re.compile(r"(?P<verb>" + "|".join(_SAY_VERBS_TH) + r")\s*[:：,\-]?\s*\Z")
_ATTR_POST_EN = re.compile(
    r"\A[\s,.;:]*[—–-]*\s*(?:(?:" + _SAY_VERBS + r"|according\s+to)\s+(?P<n1>" + _NAME_EN + r")|(?P<n2>" + _NAME_EN +
    r")\s+(?:" + _SAY_VERBS + r")\b)")
_ATTR_POST_DASH = re.compile(r"\A\s*(?:[—–]|\s-\s)\s*(?P<n3>" + _NAME_EN + r")")
_ATTR_POST_TH = re.compile(
    r"\A\s*[,.]?\s*(?P<subj>[ก-๿]{2,30}|[A-Z][A-Za-z.'’\-]+(?:\s[A-Z][A-Za-z.'’\-]+){0,3})\s?(?:กล่าว|เขียน|ระบุ|พูด|บอก|เปิดเผย|ย้ำ)")
_THAI_TAIL_TOKEN = re.compile(r"(?P<tok>[ก-๿]+|[A-Za-z][A-Za-z.'’\-]*)\Z")

_SRC_URL = re.compile(r"(?:https?://|www\.)\S+", re.I)
_SRC_DOI = re.compile(r"(?<![0-9A-Za-z])(?:doi:?\s*)?10\.[0-9]{4,9}/\S+", re.I)
_SRC_ARXIV = re.compile(r"arXiv:\s*[0-9]{4}\.[0-9]{4,5}", re.I)
_SRC_NUM = re.compile(r"\[\^?[0-9]{1,3}(?:\s*[,–\-]\s*[0-9]{1,3})*\]")
_YEAR_ANY = r"(?:(?:19|20|25)[0-9]{2})[a-z]?"
_NAME_STOP = r"(?!(?:In|On|Since|By|Updated|Released|Published|Version|Year|Circa|Around|About|Early|Late|Mid|Founded|Born|Died)\b)"
_SRC_AUTHOR_YEAR = [
    re.compile(r"\(\s*" + _NAME_STOP + r"[A-Z฀-๿][^()\n0-9]{0,49}?,?\s*" + _YEAR_ANY + r"(?:[,;:][^()\n]{0,40})?\s*\)"),
    re.compile(_NAME_STOP + r"(?<![A-Za-z0-9])[A-Z][A-Za-z'’\-]+(?:\s+et\s+al\.?|\s+(?:&|and)\s+[A-Z][A-Za-z'’\-]+)?\s*\(\s*"
               + _YEAR_ANY + r"\s*\)"),
]
_SRC_ITALIC = [
    re.compile(r"(?<![*\w])\*(?![*\s])([A-Z0-9][^*\n]{2,79}?)(?<![\s*])\*(?![*\w])"),
    re.compile(r"(?<![_\w])_(?![_\s])([A-Z0-9][^_\n]{2,79}?)(?<![\s_])_(?![_\w])"),
]
_SRC_QUOTED_TITLE = re.compile(r"[\"“«]([A-Z][^\"“”«»\n]{2,70})[\"”»]")

_EXPERT_EN = re.compile(
    r"(?<![A-Za-z])(?:Dr|Prof|Professor)(?:\.\s*|\s+)((?:[A-Z]\.\s*)?[A-Z][A-Za-z'’\-]+(?:\s+[A-Z][A-Za-z'’\-]+){0,2})")
_TH_STOPWORDS = "|".join(sorted(set(_AFFIL_TH + _FIND_VERBS_TH + ("ให้", "ได้", "พบ")), key=len, reverse=True))
_TH_NAME_END = r"(?=\s|" + _TH_STOPWORDS + r"|[^ก-๿]|\Z)"
_EXPERT_TH = re.compile(
    r"(?<![ก-๿.])(?:(?:ศ|รศ|ผศ)\.\s?(?:ดร|นพ|พญ)\.\s?|(?:ศ|รศ|ผศ|ดร|นพ|พญ)\.\s?)"
    r"[ก-ฮเ-ไ][ก-๎]{1,24}?" + _TH_NAME_END +
    r"(?:\s(?!" + _TH_STOPWORDS + r")[ก-ฮเ-ไ][ก-๎]{1,24}?" + _TH_NAME_END + r")?")
_AFFIL_EN = re.compile(r"\b(?:of|from|at)\s+(?:the\s+)?[A-Z][A-Za-z]")
_FIND_EN = re.compile(r"\b(?:" + _FIND_VERBS_EN + r")\b")
_FIND_TH = re.compile("|".join(_AFFIL_TH + _FIND_VERBS_TH))

_NUM_EN_ALT = r"\d{1,2}|" + "|".join(sorted(_NUM_WORDS_EN, key=len, reverse=True))
_NUM_TH_ALT = r"[0-9๐-๙]{1,2}|" + "|".join(sorted(_NUM_WORDS_TH, key=len, reverse=True))
_ANNOUNCE_EN = re.compile(
    r"(?<![\w.,$€£/-])(?:top\s+(?P<tn>" + _NUM_EN_ALT + r")(?![\d%/]|[.,]\d)(?!\s*(?:%|percent|per\s+cent))"
    r"|(?P<n>" + _NUM_EN_ALT + r")(?![\d%/]|[.,]\d)\s+(?:[A-Za-z][A-Za-z'’-]*\s+){0,2}?(?:" + _COUNT_NOUNS_EN.replace(" ", r"\s+") + r")(?![A-Za-z]))",
    re.I)
_ANNOUNCE_TH = re.compile(
    r"(?<![0-9๐-๙.,])(?<![ก-ฮ])(?P<n>" + _NUM_TH_ALT + r")\s?(?:" + "|".join(sorted(_COUNT_NOUNS_TH, key=len, reverse=True)) + ")")
_ANNOUNCE_BEFORE = re.compile(
    r"(?:at\s+least|about|around|roughly|approximately|more\s+than|over|up\s+to|nearly|almost|some\s+of|a\s+few\s+of|only|just)\s+\Z",
    re.I)
_LIST_ITEM = re.compile(
    r"^(?P<indent>[ \t]*)(?:(?P<bul>[-*+•·▪●○])|\**(?P<num>[0-9๐-๙]{1,3})[.)]\**|ข้อ\s?(?P<tnum>[0-9๐-๙]{1,3})[.):])\s+(?=\S)")
_CONCLUSION_RE = re.compile(r"\**(?:" + _CONCLUSION_LABEL + r")\b[\s:*—–-]", re.I)
_THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")

_ASK_RE = re.compile("|".join(_ASK_EN + _ASK_TH), re.I)

_ABBREVIATIONS = {"et al", "al", "e.g", "i.e", "vs", "dr", "prof", "mr", "mrs", "ms", "fig", "figs", "approx", "inc",
                  "ltd", "co", "jr", "sr", "st", "cf", "ca", "vol", "pp", "ed", "eds", "u.s", "u.k", "etc", "est",
                  "ref", "refs", "sec", "eq", "ch", "p", "no", "a.m", "p.m", "gen", "sen", "gov", "rep", "mt"}
_CAND = re.compile(r"[.!?]+[\"'”’)\]]*(?=\s|\Z)|[。！？]+|\n")
_ABBR_TOKEN = re.compile(r"([A-Za-z.]+)\Z")
_INITIAL_CTX = re.compile(r"(?:\b[A-Z][a-z]+ |\b[A-Z]\. ?)\Z")
_HAS_ALPHA = re.compile(r"[^\W_]")
_LISTNUM_PREFIX = re.compile(r"[ \t>*#\-]*[0-9๐-๙]{1,3}\Z")
_FENCE_OPEN = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
_BACKTICKS = re.compile(r"`+")


# ===========================================================================
# Text preparation
# ===========================================================================
def _blank(s: str) -> str:
    return "".join(c if c in "\r\n" else "\x00" for c in s)


def _mask_inline(line: str) -> str:
    runs = [(m.start(), m.end()) for m in _BACKTICKS.finditer(line)]
    if len(runs) < 2:
        return line
    out: list[str] = []
    last = 0
    i = 0
    while i < len(runs):
        a, b = runs[i]
        width = b - a
        k = next((j for j in range(i + 1, len(runs)) if runs[j][1] - runs[j][0] == width), None)
        if k is None:
            i += 1
            continue
        out.append(line[last:a])
        out.append("\x00" * (runs[k][1] - a))
        last = runs[k][1]
        i = k + 1
    out.append(line[last:])
    return "".join(out)


def _mask_code(text: str) -> str:
    """Blank fenced + inline code with NULs, keeping every offset and newline."""
    if "`" not in text and "~~~" not in text:
        return text
    out: list[str] = []
    fence: Optional[tuple[str, int]] = None
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


def _sentences(t: str) -> list[tuple[int, int]]:
    cuts = [0]
    for m in _CAND.finditer(t):
        a, e = m.start(), m.end()
        c = t[a]
        if c == "\n":
            before = t[max(0, a - 40):a].rstrip(" \t\r")
            after = t[a + 1:a + 41].lstrip(" \t\r")
            if (before and after and after[0].isascii() and after[0].islower()
                    and (before[-1].isalnum() or before[-1] in ",;")):
                continue                                         # hard-wrapped line
        elif c == "." and t[a + 1:a + 2] != ".":
            ls = t.rfind("\n", 0, a) + 1
            if _LISTNUM_PREFIX.fullmatch(t[ls:a]):
                continue                                         # list number "1. Item"
            tok = _ABBR_TOKEN.search(t, max(0, a - 10), a)
            if tok:
                raw = tok.group(1).strip(".")
                if raw.lower() in _ABBREVIATIONS:
                    continue
                if len(raw) == 1 and raw.isupper() and (
                        tok.start() == 0 or _INITIAL_CTX.search(t, max(0, tok.start() - 20), tok.start())):
                    continue                                     # "John F. Kennedy"
            j = e
            while j < len(t) and t[j] in " \t":
                j += 1
            if j < len(t) and t[j].isascii() and t[j].islower():
                continue                                         # "e.g. something", "vs. the"
        cuts.append(e)
    cuts.append(len(t))
    out: list[tuple[int, int]] = []
    for a, b in zip(cuts, cuts[1:]):
        seg = t[a:b]
        if not _HAS_ALPHA.search(seg):
            continue
        lead = len(seg) - len(seg.lstrip())
        trail = len(seg) - len(seg.rstrip())
        out.append((a + lead, b - trail))
    return out


def _trim_url_end(raw: str, start: int) -> int:
    """End offset of a URL match: drops trailing sentence punctuation, but keeps a '...' placeholder."""
    u = raw.rstrip(",;:!?}*_>")
    while u.endswith(")") and u.count(")") > u.count("("):
        u = u[:-1].rstrip(",;:!?}*_>")
    body = re.sub(r"^[a-z]+://", "", u, flags=re.I)
    if body and not re.fullmatch(r"[.…]+", body) and not re.search(r"/(?:\.{3}|…)$", u):
        u = u.rstrip(".…")
    return start + len(u)


def _find_urls(t: str) -> list[tuple[int, int]]:
    out = []
    for m in _URL_RE.finditer(t):
        e = _trim_url_end(m.group(), m.start())
        if e > m.start():
            out.append((m.start(), e))
    return out


def _mask_spans(t: str, spans) -> str:
    if not spans:
        return t
    parts, last = [], 0
    for a, b in spans:
        parts.append(t[last:a])
        parts.append("\x00" * (b - a))
        last = b
    parts.append(t[last:])
    return "".join(parts)


def _find_quotes(t: str) -> list[tuple[int, int, int, int]]:
    """(start, end, inner_start, inner_end) of quoted passages, non-overlapping, in order."""
    found = []
    for rx in (_Q_STRAIGHT, _Q_CURLY, _Q_ANGLE, _Q_CORNER):
        for m in rx.finditer(t):
            found.append((m.start(), m.end(), m.start(1), m.end(1)))
    found.sort(key=lambda q: (q[0], -q[1]))
    out: list[tuple[int, int, int, int]] = []
    for q in found:
        if out and q[0] < out[-1][1]:
            continue
        out.append(q)
    return out


def _neutralise_quote_punct(t: str, quotes) -> str:
    """Inside a quoted passage ". ! ?" do not end the surrounding sentence."""
    if not quotes:
        return t
    parts, last = [], 0
    for _a, _b, ia, ib in quotes:
        parts.append(t[last:ia])
        parts.append(re.sub(r"[.!?。！？]", ",", t[ia:ib]))
        last = ib
    parts.append(t[last:])
    return "".join(parts)


# ===========================================================================
# Shared context helpers
# ===========================================================================
class _Ctx:
    def __init__(self, text: str):
        self.masked = _mask_code(text)
        self.urls = _find_urls(self.masked)
        self.prose = _mask_spans(self.masked, self.urls)
        self.quotes = _find_quotes(self.prose)
        self.sent_txt = _neutralise_quote_punct(self.prose, self.quotes)
        self.sents = _sentences(self.sent_txt)
        self.starts = [s for s, _ in self.sents]
        self._signals: Optional[list[tuple[int, int, str]]] = None

    def sidx(self, pos: int) -> int:
        return max(0, bisect_right(self.starts, pos) - 1)

    def is_question(self, i: int) -> bool:
        s, e = self.sents[i]
        return self.prose[s:e].rstrip().endswith(("?", "？"))

    def first_word_pos(self, i: int) -> int:
        s, e = self.sents[i]
        m = re.search(r"[A-Za-z]", self.prose[s:e])
        return s + m.start() if m else s

    def cap(self, a: int, b: int) -> int:
        """Look-around radius for 'the same sentence': tight for Thai (no reliable spaces)."""
        return 100 if _THAI_CH.search(self.prose, max(0, a - 20), b + 20) else 300

    # -- source signals (URL, DOI, [n], author-year, italic title, quoted title) ---------------
    def signals(self) -> list[tuple[int, int, str]]:
        if self._signals is None:
            t = self.masked
            sig: list[tuple[int, int, str]] = []
            for rx, kind in ((_SRC_URL, "url"), (_SRC_DOI, "doi"), (_SRC_ARXIV, "arxiv"), (_SRC_NUM, "num")):
                sig += [(m.start(), m.end(), kind) for m in rx.finditer(t)]
            for rx in _SRC_AUTHOR_YEAR:
                sig += [(m.start(), m.end(), "ay") for m in rx.finditer(t)]
            for rx in _SRC_ITALIC:
                sig += [(m.start(), m.end(), "italic") for m in rx.finditer(t)]
            sig += [(m.start(), m.end(), "qtitle") for m in _SRC_QUOTED_TITLE.finditer(t)]
            sig.sort()
            self._signals = sig
        return self._signals

    def has_source(self, a: int, b: int, exclude: tuple[int, int] | None = None, *, titles: bool = True) -> bool:
        """Is a source signal inside [a, b) (and not inside the excluded span)?"""
        sig = self.signals()
        from bisect import bisect_left
        k = bisect_left(sig, (a, -1, ""))
        while k < len(sig) and sig[k][0] < b:
            s, e, kind = sig[k]
            k += 1
            if e > b:
                continue
            if exclude and s < exclude[1] and exclude[0] < e:
                continue
            if not titles and kind in ("italic", "qtitle"):
                continue
            return True
        return False


Hit = tuple[int, int, str, str]      # start, end, tag, reason


def _short(s: str, n: int = 40) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s if len(s) <= n else s[:n - 1] + "…"


# ===========================================================================
# 1. time_sensitive
# ===========================================================================
def _has_changeable_fact(cx: _Ctx, si: int, a: int, b: int) -> bool:
    s, e = cx.sents[si]
    cap = cx.cap(a, b)
    cs, ce = max(s, a - cap), min(e, b + cap)
    seg = cx.prose[cs:a] + " " * (b - a) + cx.prose[b:ce]          # same length; the trigger itself is blanked
    lead = _LEADMARK.match(cx.prose, s)
    lead_end = lead.end() if lead else s
    for m in _NUM_RE.finditer(seg):
        if cs + m.start() >= lead_end:
            return True
    if _OFFICE_RE.search(seg):
        return True
    thai_ctx = bool(_THAI_CH.search(seg))
    first = cx.first_word_pos(si)
    for m in _CAPTOK.finditer(seg):
        if cs + m.start() == first and not thai_ctx:
            continue
        if m.group() in _NOT_NAMES:
            continue
        return True
    return False


def _time_cue(cx: _Ctx, today: _dt.date, hits: list[Hit]) -> None:
    min_year = today.year - 1
    trig: list[tuple[int, int, bool]] = []
    for m in _TIME_RE.finditer(cx.prose):
        a, b = m.span()
        g = m.group().lower()
        if g.startswith("up") and _KEEP_BEFORE.search(cx.prose, max(0, a - 30), a):
            continue                                            # "keep it up to date"
        trig.append((a, b, False))
    for m in _YEAR_RE.finditer(cx.prose):
        g = "ce" if m.group("ce") else "be"
        y = int(m.group(g))
        if g == "be":
            y -= 543
        if y < min_year:
            continue
        a, b = m.span(g)
        if _CITE_YEAR.search(cx.prose, max(0, a - 6), a) and re.match(r"\s*[)\],;]", cx.prose[b:b + 3]):
            continue                                            # "(Smith, 2025)"
        trig.append((a, b, True))
    trig.sort()
    by_sent: dict[int, list[tuple[int, int, bool]]] = {}
    for a, b, is_year in trig:
        by_sent.setdefault(cx.sidx(a), []).append((a, b, is_year))
    for si, lst in by_sent.items():
        if cx.is_question(si):
            continue
        for a, b, is_year in lst[:8]:
            if _has_changeable_fact(cx, si, a, b):
                phrase = _short(cx.prose[a:b])
                what = f"the year {phrase}" if is_year else f"'{phrase}'"
                hits.append((a, b, "time_sensitive",
                             f"Describes how things stand now or soon ({what}) — AI answers can be out of date, "
                             f"so check this against a current source."))
                break


# ===========================================================================
# 2. absolute_claim
# ===========================================================================
def _absolute_cue(cx: _Ctx, hits: list[Hit]) -> None:
    by_sent: dict[int, list[tuple[int, int, str]]] = {}
    for m in _ABS_RE.finditer(cx.prose):
        fam = m.lastgroup
        a, b = m.span()
        before = cx.prose[max(0, a - 28):a]
        after = cx.prose[b:b + 24]
        si = cx.sidx(a)
        if fam == "alwaysnever":
            word = m.group().lower()
            if a == cx.first_word_pos(si) or _ABS_IMPERATIVE.search(before):
                continue                                        # "Always consult a doctor": advice, not a claim
            if word == "always" and _ABS_NEG_BEFORE.search(before):
                continue                                        # "not always"
            if word == "never" and _NEVER_AFTER.match(after):
                continue                                        # "returns are never guaranteed": a caution
        elif fam == "all":
            if _ALL_BEFORE.search(before) or _ALL_AFTER.match(after):
                continue                                        # "after all", "at all", "not all", "all right"
        elif fam == "every":
            if _EVERY_AFTER.match(after) or _ABS_NEG_BEFORE.search(before):
                continue                                        # "every 4 hours", "not every"
        elif fam == "guar":
            if _GUAR_BEFORE.search(before):
                continue                                        # "no guarantee ... not guaranteed" is a caution
        elif fam == "proven":
            has_to = m.group().lower().rstrip().endswith("to")
            if _VAGUE_PROVEN_BEFORE.search(before):
                continue                                        # vague.py already flags these shapes
            if not has_to and not (_PROVEN_ADV.search(before) or _PROVEN_NEXT.match(after)):
                continue
        by_sent.setdefault(si, []).append((a, b, fam or ""))
    for si, lst in by_sent.items():
        if cx.is_question(si):
            continue
        for a, b, _fam in lst[:8]:
            cap = cx.cap(a, b)
            s, e = cx.sents[si]
            if _DOMAIN_RE.search(cx.prose, max(s, a - cap), min(e, b + cap)):
                phrase = _short(cx.prose[a:b])
                hits.append((a, b, "absolute_claim",
                             f"Absolute wording ('{phrase}') in a health, legal, money or safety statement is rarely "
                             f"fully true — check it against a reliable source."))
                break


# ===========================================================================
# 3. ai_self_disclosure
# ===========================================================================
def _self_cue(cx: _Ctx, hits: list[Hit]) -> None:
    for m in _SELF_RE.finditer(cx.prose):
        phrase = _short(m.group())
        hits.append((m.start(), m.end(), "ai_self_disclosure",
                     f"The answer itself says it may be limited or wrong ('{phrase}') — treat the facts near it "
                     f"with care and check them."))


# ===========================================================================
# 4. placeholder_url
# ===========================================================================
def _is_placeholder_url(u: str) -> bool:
    low = u.lower()
    m = re.match(r"(?:[a-z][a-z0-9+.-]*://)?([^/?#]*)", low)
    authority = m.group(1) if m else ""
    host = authority.rsplit("@", 1)[-1]
    host = re.sub(r":\d*$", "", host)
    rest = low[m.end():] if m else ""
    if _PH_HOST_RE.match(host) or host.endswith(".localhost"):
        return True
    if _PH_ANY_RE.search(low):
        return True
    if host in ("...", "…") or "..." in host or any(re.fullmatch(r"x{3,}", lab) for lab in host.split(".")):
        return True
    path = re.split(r"[?#]", rest, 1)[0]
    return any(_PH_SEG_RE.match(seg) for seg in path.split("/") if seg)


def _placeholder_cue(cx: _Ctx, hits: list[Hit]) -> None:
    t = cx.masked
    cands: list[tuple[int, int]] = [(m.start(), m.end()) for m in _TOKEN_RE.finditer(t)]
    cands += [(a, b) for a, b in cx.urls if _is_placeholder_url(t[a:b])]
    url_starts = [a for a, _ in cx.urls]
    for m in _BARE_EXAMPLE_RE.finditer(t):
        k = bisect_right(url_starts, m.start()) - 1
        if k >= 0 and m.start() < cx.urls[k][1]:
            continue                                            # already inside a scheme URL
        cands.append((m.start(), _trim_url_end(m.group(), m.start())))
    cands.sort(key=lambda c: (c[0], -c[1]))
    last_end = -1
    for a, b in cands:
        if a < last_end:
            continue
        last_end = b
        hits.append((a, b, "placeholder_url",
                     f"'{_short(t[a:b], 50)}' is a template address, not a real source — a placeholder cannot back a claim."))


# ===========================================================================
# 5. unsourced_quote
# ===========================================================================
def _named(name: str) -> bool:
    toks = name.replace("&", " ").split()
    if not toks:
        return False
    if len(toks) == 1 and toks[0].lower().strip(".,") in _GENERIC_EN and not (toks[0].isupper() and len(toks[0]) > 1):
        return False
    return not all(tk.lower() in _GENERIC_EN and not (tk.isupper() and len(tk) > 1) for tk in toks)


def _thai_subject_named(before: str) -> bool:
    m = _THAI_TAIL_TOKEN.search(before.rstrip())
    if not m:
        return False
    tok = m.group("tok")
    if re.match(r"[A-Za-z]", tok):
        return tok[0].isupper() and tok.lower().strip(".") not in _GENERIC_EN
    if len(tok) < 2:
        return False
    return not any(tok.startswith(g) or tok.endswith(g) for g in _THAI_GENERIC_SUBJECTS)


def _has_attribution(cx: _Ctx, qs: int, qe: int) -> bool:
    s0, _e0 = cx.sents[cx.sidx(qs)]
    _s1, e1 = cx.sents[cx.sidx(max(qs, qe - 1))]
    pre = cx.prose[max(s0, qs - 160):qs]
    post = cx.prose[qe:min(e1, qe + 120)]
    m = _ATTR_PRE_EN.search(pre)
    if m and _named(m.group("name")):
        return True
    m = _ATTR_PRE_ACCORDING.search(pre)
    if m and _named(m.group("name")):
        return True
    m = _ATTR_PRE_TH.search(pre)
    if m and _thai_subject_named(pre[:m.start()]):
        return True
    m = _ATTR_POST_EN.match(post)
    if m and _named(m.group("n1") or m.group("n2") or ""):
        return True
    m = _ATTR_POST_DASH.match(post)
    if m and _named(m.group("n3")) and (len(m.group("n3").split()) >= 2 or m.group("n3").isupper()):
        return True
    m = _ATTR_POST_TH.match(post)
    if m and _thai_subject_named(m.group("subj")):
        return True
    return False


def _quote_cue(cx: _Ctx, hits: list[Hit]) -> None:
    for qs, qe, ia, ib in cx.quotes:
        inner = cx.prose[ia:ib]
        if len(_WORD_RE.findall(inner)) < 8 and len(_THAI_CH.findall(inner)) < 30:
            continue
        if not _has_attribution(cx, qs, qe):
            continue
        i, j = cx.sidx(qs), cx.sidx(max(qs, qe - 1))
        lo = cx.sents[i][0]
        hi = cx.sents[min(j + 1, len(cx.sents) - 1)][1]
        if cx.has_source(lo, hi, exclude=(qs, qe)):
            continue
        hits.append((qs, qe, "unsourced_quote",
                     "A long quotation is attributed to a named source but no source is given — AI often invents "
                     "plausible quotes, so find the original before using it."))


# ===========================================================================
# 6. named_expert_claim
# ===========================================================================
def _expert_cue(cx: _Ctx, hits: list[Hit]) -> None:
    cands: list[tuple[int, int, bool]] = []
    for m in _EXPERT_EN.finditer(cx.prose):
        cands.append((m.start(), m.end(), False))
    for m in _EXPERT_TH.finditer(cx.prose):
        cands.append((m.start(), m.end(), True))
    for a, b, th in cands:
        si = cx.sidx(a)
        _s, e = cx.sents[si]
        tail = cx.prose[b:min(e, b + (100 if th else 120))]
        if th:
            ok = bool(_FIND_TH.search(tail))
        else:
            ok = bool(_AFFIL_EN.search(tail[:80]) or _FIND_EN.search(tail))
        if not ok:
            continue
        i, j = cx.sidx(a), cx.sidx(max(a, b - 1))
        lo = cx.sents[i][0]
        hi = cx.sents[min(j + 1, len(cx.sents) - 1)][1]
        if cx.has_source(lo, hi):
            continue
        hits.append((a, b, "named_expert_claim",
                     f"Names '{_short(cx.prose[a:b], 50)}' as an expert but gives no citation — check that this "
                     f"person exists and really said or found this."))


# ===========================================================================
# 7. count_mismatch
# ===========================================================================
class _Line:
    __slots__ = ("start", "text", "kind", "indent", "num", "item_text")

    def __init__(self, start: int, text: str):
        self.start = start
        self.text = text
        self.num: Optional[int] = None
        self.item_text = ""
        stripped = text.strip()
        if not stripped:
            self.kind, self.indent = "blank", 0
            return
        self.indent = len(text) - len(text.lstrip(" \t"))
        self.indent += 3 * text[:self.indent].count("\t")
        if "\x00" in text and not stripped.strip("\x00"):
            self.kind = "code"
            return
        m = _LIST_ITEM.match(text)
        if m:
            self.kind = "item"
            n = m.group("num") or m.group("tnum")
            self.num = int(n.translate(_THAI_DIGITS)) if n else None
            self.item_text = text[m.end():]
        else:
            self.kind = "text"


def _num_value(tok: str) -> Optional[int]:
    tok = tok.strip().lower()
    if tok.translate(_THAI_DIGITS).isdigit():
        return int(tok.translate(_THAI_DIGITS))
    return _NUM_WORDS_EN.get(tok) or _NUM_WORDS_TH.get(tok)


def _announcements(line: str) -> list[tuple[int, int, int]]:
    """Announcements (start, end, N) in the LAST sentence of `line`."""
    sl = _sentences(line)
    a0, b0 = sl[-1] if sl else (0, len(line))
    seg = line[a0:b0]
    out = []
    for m in _ANNOUNCE_EN.finditer(seg):
        n = _num_value(m.group("tn") or m.group("n"))
        if n is not None and not _ANNOUNCE_BEFORE.search(seg, max(0, m.start() - 20), m.start()):
            out.append((a0 + m.start(), a0 + m.end(), n))
    for m in _ANNOUNCE_TH.finditer(seg):
        n = _num_value(m.group("n"))
        if n is not None:
            out.append((a0 + m.start(), a0 + m.end(), n))
    return out


def _count_cue(cx: _Ctx, hits: list[Hit]) -> None:
    t = cx.masked
    lines: list[_Line] = []
    pos = 0
    for raw in t.split("\n"):
        lines.append(_Line(pos, raw.rstrip("\r")))
        pos += len(raw) + 1
    n_lines = len(lines)
    for k, L in enumerate(lines):
        if L.kind != "item":
            continue
        p = k - 1
        if p >= 0 and lines[p].kind == "blank":
            p -= 1
        if p < 0 or lines[p].kind != "text":
            continue
        ann = _announcements(lines[p].text)
        if len(ann) != 1:
            continue
        a, b, n_claimed = ann[0]
        if not 2 <= n_claimed <= 30:
            continue
        # --- collect the contiguous top-level list that starts at line k
        base = L.indent
        kind0 = "num" if L.num is not None else "bul"
        items: list[_Line] = []
        j = k
        ambiguous = False
        while j < n_lines:
            M = lines[j]
            if M.kind == "item":
                if M.indent == base:
                    if ("num" if M.num is not None else "bul") != kind0:
                        ambiguous = True
                        break
                    items.append(M)
                elif M.indent < base:
                    break
            elif M.kind == "blank":
                if j + 1 < n_lines and lines[j + 1].kind == "blank":
                    break
                q = j + 1
                if q >= n_lines:
                    break
                nxt = lines[q]
                if nxt.kind == "item" and nxt.indent >= base:
                    pass
                elif nxt.kind == "text" and nxt.indent > base:
                    pass
                elif nxt.kind == "code":
                    pass
                else:
                    break
            j += 1
        if ambiguous or not items:
            continue
        # a same-shaped list resuming right after an interjected paragraph is ambiguous
        q, seen_text, cnt = j, None, 0
        while q < n_lines and cnt < 4:
            if lines[q].kind == "blank":
                q += 1
                continue
            cnt += 1
            R = lines[q]
            if R.kind == "item" and R.indent == base and ("num" if R.num is not None else "bul") == kind0:
                if not (seen_text is not None and seen_text.rstrip().endswith(":")):
                    ambiguous = True
                break
            if R.kind == "text":
                seen_text = R.text
            q += 1
        if ambiguous:
            continue
        if kind0 == "num" and [it.num for it in items] != list(range(1, len(items) + 1)):
            continue
        m_total = len(items)
        concl = bool(_CONCLUSION_RE.match(items[-1].item_text.lstrip()))
        if n_claimed == m_total or (concl and n_claimed == m_total - 1):
            continue
        m_eff = m_total - 1 if concl else m_total
        base_off = lines[p].start
        hits.append((base_off + a, base_off + b, "count_mismatch",
                     f"Says '{_short(t[base_off + a:base_off + b])}' but the list right after it has {m_eff} "
                     f"item{'s' if m_eff != 1 else ''} — something may be missing or extra."))


# ===========================================================================
# 8. question_asks_for_sources
# ===========================================================================
def _question_cue(cx: _Ctx, question: str, hits: list[Hit]) -> None:
    if not isinstance(question, str) or not question.strip():
        return
    if not _ASK_RE.search(question):
        return
    if cx.has_source(0, len(cx.masked) + 1, titles=False):
        return
    if not cx.sents:
        return
    s, e = cx.sents[0]
    hits.append((s, e, "question_asks_for_sources",
                 "You asked for sources or figures but the answer gives no link or citation — AI tools often invent "
                 "references when asked, so treat its figures and quotes as unchecked."))


# ===========================================================================
# scan
# ===========================================================================
_TAG_ORDER = {t: i for i, t in enumerate((
    "time_sensitive", "absolute_claim", "ai_self_disclosure", "placeholder_url", "unsourced_quote",
    "named_expert_claim", "count_mismatch", "question_asks_for_sources"))}


def scan(text: str, *, question: Optional[str] = None, today: Optional[_dt.date] = None) -> list[Finding]:
    """Flag topic-independent risk cues.  Never VERIFIED / CONTRADICTED: only 'check this'."""
    if not isinstance(text, str) or not text.strip():
        return []
    today = today or _dt.date.today()
    cx = _Ctx(text)
    if not cx.masked.strip("\x00 \t\r\n"):
        return []
    hits: list[Hit] = []
    if cx.sents:
        _time_cue(cx, today, hits)
        _absolute_cue(cx, hits)
        _quote_cue(cx, hits)
        _expert_cue(cx, hits)
    _self_cue(cx, hits)
    _placeholder_cue(cx, hits)
    _count_cue(cx, hits)
    if question is not None:
        _question_cue(cx, question, hits)

    # one cue never fires twice on overlapping spans (leftmost, then longest, within a tag)
    hits.sort(key=lambda h: (_TAG_ORDER[h[2]], h[0], -h[1]))
    kept: list[Hit] = []
    for h in hits:
        if kept and kept[-1][2] == h[2] and h[0] < kept[-1][1]:
            continue
        kept.append(h)
    kept.sort(key=lambda h: (h[0], h[1], _TAG_ORDER[h[2]]))

    findings: list[Finding] = []
    for n, (a, b, tag, reason) in enumerate(kept, 1):
        claim = Claim(id=f"q{n}", kind=ClaimKind.CUE, text=text[a:b], start=a, end=b, data={"cue": tag})
        findings.append(Finding(claim=claim, status=Status.UNVERIFIED, module=MODULE, reason=reason, tag=tag))
    return findings
