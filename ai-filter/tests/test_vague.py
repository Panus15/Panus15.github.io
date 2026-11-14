"""Tests for engine/vague.py (unsourced authority appeals + suspicious precision). Offline.

Run: python3 tests/test_vague.py
"""
import os, sys, time, unicodedata
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.models import ClaimKind, Status
from engine.vague import scan


def run(text):
    """scan + the invariants every Finding must satisfy, in every test."""
    fs = scan(text)
    last_start = -1
    for n, f in enumerate(fs, 1):
        c = f.claim
        assert c.id == f"v{n}", (c.id, n)
        assert c.kind is ClaimKind.VAGUE
        assert f.status is Status.UNVERIFIED and f.module == "vague"
        assert text[c.start:c.end] == c.text and c.text, (c, text[c.start:c.end])
        assert c.start > last_start, "findings must be in document order"
        last_start = c.start
        assert isinstance(c.data["phrase"], str) and c.data["phrase"]
        assert f.reason and f.reason.isascii() is False or f.reason  # one non-empty sentence
        assert "\n" not in f.reason
    return fs


def phrases(text):
    return [f.claim.data["phrase"] for f in run(text)]


# --- every phrase family triggers (English) ---------------------------------
def test_english_studies_family():
    for t in ("Studies show that it works.", "Studies have shown it works.", "Studies suggest it works.",
              "Studies indicate it works.", "Studies find it works.", "Research shows it works.",
              "Research suggests it works.", "Research indicates it works.", "Research has found it works.",
              "A study found it works.", "A recent study found it works.", "Surveys show it works.",
              "Surveys indicate it works."):
        assert len(run(t)) == 1, t


def test_english_experts_family():
    for t in ("Experts say it works.", "Experts agree it works.", "Experts believe it works.",
              "Experts recommend it.", "Scientists say it works.", "Scientists have found it works.",
              "Scientists believe it works.", "According to experts, it works.",
              "According to some experts, it works.", "According to studies, it works.",
              "According to research, it works.", "According to scientists, it works."):
        assert len(run(t)) == 1, t


def test_english_belief_evidence_proven_family():
    for t in ("It is widely known that it works.", "It is well known that it works.",
              "It is commonly believed that it works.", "It is generally accepted that it works.",
              "It's well established that it works.", "Many people believe it works.",
              "Many researchers think it works.", "Evidence suggests it works.", "Statistics show it works.",
              "Data shows it works.", "It has been proven that it works.", "It has been shown that it works.",
              "This is proven to work."):
        assert len(run(t)) == 1, t


def test_case_insensitive_and_flexible_spacing():
    assert len(run("STUDIES SHOW it works.")) == 1
    assert len(run("studies   show it works.")) == 1
    assert len(run("studies\nshow it works.")) == 1
    assert len(run("studies show".replace(" ", " ") + " it works.")) == 1
    # word boundaries: "studies shown" is not "studies show"
    assert run("The prestudies shown here are fine.") == []
    assert run("Studies shown above are fine.") == []


# --- every phrase family triggers (Thai) ------------------------------------
def test_thai_phrase_families():
    for t in ("มีการศึกษาพบว่ากาแฟดี", "มีงานวิจัยพบว่ากาแฟดี", "มีงานวิจัยชี้ว่ากาแฟดี", "งานวิจัยชี้ว่ากาแฟดี",
              "งานวิจัยพบว่ากาแฟดี", "การศึกษาพบว่ากาแฟดี", "การศึกษาชี้ว่ากาแฟดี", "นักวิจัยพบว่ากาแฟดี",
              "นักวิทยาศาสตร์พบว่ากาแฟดี", "ผู้เชี่ยวชาญระบุว่ากาแฟดี", "ผู้เชี่ยวชาญกล่าวว่ากาแฟดี",
              "ผู้เชี่ยวชาญแนะนำว่ากาแฟดี", "ผู้เชี่ยวชาญเห็นพ้องกันว่ากาแฟดี", "จากการสำรวจพบว่ากาแฟดี",
              "ผลสำรวจชี้ว่ากาแฟดี", "สถิติชี้ว่ากาแฟดี", "ข้อมูลระบุว่ากาแฟดี", "มีรายงานว่ากาแฟดี",
              "มีผู้เชี่ยวชาญบางท่านเชื่อว่ากาแฟดี", "เป็นที่ทราบกันดีว่ากาแฟดี", "ว่ากันว่ากาแฟดี",
              "หลายคนเชื่อว่ากาแฟดี", "ได้รับการพิสูจน์แล้วว่ากาแฟดี", "มีหลักฐานชี้ว่ากาแฟดี"):
        assert len(run(t)) == 1, t


def test_thai_false_positives_avoided():
    assert run("ไม่มีงานวิจัยพบว่ากาแฟอันตราย") == []          # "no study found ..." is not an appeal
    assert run("ไม่มีผู้เชี่ยวชาญระบุว่ากาแฟอันตราย") == []
    assert run("ยังไม่ได้รับการพิสูจน์แล้วว่ากาแฟดี") == []
    assert run("บริษัทมีผู้เชี่ยวชาญด้านการเงินให้บริการ") == []   # bare "have experts", no verb of saying
    assert run("ฐานข้อมูลระบุว่ากาแฟดี") == []                    # "the database says"
    assert run("การศึกษาของไทยกำลังเปลี่ยนไป") == []              # "education" in general
    assert run("ข้อมูลระบุว่ายอดขายเพิ่มขึ้น 12 หน่วย") == []     # carries its own figure


def test_thai_unbacked_vs_backed():
    assert len(run("มีงานวิจัยพบว่ากาแฟช่วยลดความเสี่ยงโรคหัวใจ")) == 1
    assert run("มีงานวิจัยพบว่ากาแฟช่วยลดความเสี่ยงโรคหัวใจ ตามรายงานขององค์การอนามัยโลก") == []
    assert run("มีงานวิจัยพบว่ากาแฟดี จากมหาวิทยาลัยมหิดล") == []
    assert run("มีงานวิจัยพบว่ากาแฟดี (สมชาย, 2563)") == []
    assert run("ผู้เชี่ยวชาญระบุว่ากาแฟดี ดร.สมชาย ใจดี กล่าวไว้") == []
    assert run("มีงานวิจัยพบว่ากาแฟดี ดูที่ https://example.org/paper") == []
    # "โปรแกรม" contains กรม but is not a department
    assert len(run("มีงานวิจัยพบว่าโปรแกรมนี้ดี")) == 1
    assert run("มีงานวิจัยพบว่าโปรแกรมนี้ดี ตามที่กรมวิทยาศาสตร์บริการระบุ") == []


# --- backed vs unbacked, English --------------------------------------------
def test_each_backing_kind_backs_a_claim():
    unbacked = "Studies show that coffee is healthy."
    assert len(run(unbacked)) == 1
    for tail in (" (Smith, 2020).", " (Smith et al., 2020).", " [1].", " [12].", " See https://example.org/x.",
                 " See www.example.org/x.", " doi:10.1000/xyz123.", " See 10.1038/nature12373.",
                 " Per the WHO.", " Reported by Harvard.", " Smith et al. (2020) found this.",
                 " Dr. Smith confirmed it.", " Published in Nature.", " Stanford University did it.",
                 " Per the Mayo Clinic."):
        t = "Studies show that coffee is healthy." + tail
        assert run(t) == [], t


def test_backing_in_same_sentence_and_before_trigger():
    assert run("According to the WHO, studies show that coffee is healthy.") == []
    assert run("Studies show (Smith, 2020) that coffee is healthy.") == []
    assert run("Experts at Johns Hopkins say it works.") == []


def test_year_alone_is_not_enough():
    assert len(run("Studies show that coffee is healthy (2020).")) == 1
    assert len(run("Studies show in 2020 that coffee is healthy.")) == 1
    assert len(run("A 2019 study found that coffee is healthy.")) == 1
    assert len(run("มีการศึกษาพบว่ากาแฟดี ในปี 2563")) == 1
    assert len(run("มีการศึกษาพบว่ากาแฟดี (พ.ศ. 2563)")) == 1
    # ... but a year attached to a name is a citation
    assert run("Studies show that coffee is healthy (Smith 2020).") == []


def test_not_every_capitalised_word_is_an_institution():
    assert len(run("Studies show that Science is hard. Nature is lovely.")) == 1
    assert len(run("Studies show that it works. WHOLE grains help.")) == 1   # not the WHO


def test_next_sentence_backs_but_third_sentence_does_not():
    assert run("Studies show that coffee is healthy. See https://example.org/paper.") == []
    assert run("Studies show that coffee is healthy. The WHO agrees.") == []
    t = "Studies show that coffee is healthy. It tastes good. It is cheap. See https://example.org/paper."
    assert len(run(t)) == 1
    t = "Studies show that coffee is healthy.\n\nIt tastes good.\n\nIt is cheap. Per the WHO."
    assert len(run(t)) == 1
    # a source in the PREVIOUS sentence does not back it either
    assert len(run("The WHO publishes data. Studies show that coffee is healthy.")) == 1


def test_newline_is_a_boundary_but_hard_wrap_is_not():
    assert run("- Studies show that coffee is healthy\n- Per the WHO") == []
    assert len(run("- Studies show that coffee is healthy\n- It tastes good\n- It is cheap\n- Per the WHO")) == 1
    assert run("Studies show that coffee is\nhealthy according to the WHO.") == []   # wrapped line, same sentence


def test_abbreviations_do_not_split_sentences():
    assert run("Studies show this, e.g. in Smith et al. (2020).") == []
    assert run("Studies show this, as Dr. Jones said.") == []
    assert run("Studies show this. See Fig. 3 and ref. [4].") == []      # [4] is in the next sentence


def test_overlapping_phrases_yield_one_finding_longest_leftmost():
    fs = run("Many experts say that coffee is healthy.")
    assert len(fs) == 1 and fs[0].claim.text == "Many experts say"
    fs = run("According to some experts, many experts say that coffee is healthy.")
    assert [f.claim.text for f in fs] == ["According to some experts", "many experts say"]
    fs = run("มีการศึกษาพบว่ากาแฟดี")
    assert len(fs) == 1 and fs[0].claim.text == "มีการศึกษาพบว่า"            # not just การศึกษาพบว่า
    fs = run("มีผู้เชี่ยวชาญเห็นพ้องกันว่ากาแฟดี")
    assert len(fs) == 1


def test_questions_conditionals_and_back_references_are_not_flagged():
    assert run("Do studies show that coffee is healthy?") == []
    assert run("If experts say so, we will check.") == []
    assert run("The research shows the following results.") == []         # points back at named research
    assert run("This data shows the trend.") == []
    assert run("The data shows it works.") == []


def test_data_shows_depends_on_figure():
    assert len(run("Data shows that coffee is healthy.")) == 1
    assert run("Data shows that sales rose 12% in March.") == []


# --- precision rule ---------------------------------------------------------
def test_precise_numbers_flagged_and_backed():
    fs = run("The churn rate is 73.42% this quarter.")
    assert len(fs) == 1 and fs[0].claim.text == "73.42%" and fs[0].claim.data["phrase"] == "73.42%"
    assert fs[0].reason.startswith("Very precise figure")
    fs = run("We now have 1,284,903 users on the platform.")
    assert len(fs) == 1 and fs[0].claim.text == "1,284,903 users" and fs[0].claim.data["phrase"] == "1,284,903"
    fs = run("ร้อยละ 73.42 ของผู้ใช้เลือกแบบนี้")
    assert len(fs) == 1 and fs[0].claim.text == "ร้อยละ 73.42"
    assert run("The churn rate is 73.42% this quarter (WHO, 2021).") == []
    assert run("The churn rate is 73.42% this quarter [3].") == []
    assert run("The churn rate is 73.42% this quarter. Source: https://example.org/report") == []
    assert run("1,284,903 users joined, according to the OECD.") == []


def test_non_precise_things_not_flagged():
    for t in ("It was released in 2024.", "Use version 3.10.2 or 1.2.30 now.", "It costs $1,284.50 or 12.99 USD.",
              "The meeting is at 10:45.", "1. First step\n2. Second step\n3. Third step",
              "Growth was 12% and 7.5% last year.", "We have 1,284,000 users.", "Visit https://x.org/a/73.42%25/b",
              "Price: ฿1,284,903 baht.", "Call 0812345678 users.", "Uptime was 99.99% last year.",
              "For example, 12.50% of 80 is 10.", "Compute 73.42% * 3 = 220.26.",
              "Pages 1,284,903 of the book."):
        assert run(t) == [], t
    assert run("| metric | value |\n|---|---|\n| churn | 73.42% |\n") == []   # table rows are left alone


# --- code is ignored ----------------------------------------------------------
def test_code_fence_and_inline_code_ignored():
    assert run("```\nStudies show that 73.42% of users\n```\n") == []
    assert run("~~~python\n# experts say 1,284,903 users\n~~~") == []
    assert run("Use `studies show` and `73.42%` literally.") == []
    assert run("``studies ` show`` is odd") == []
    t = "```\nx\n```\nStudies show it works."
    fs = run(t)
    assert len(fs) == 1 and fs[0].claim.start == t.index("Studies")
    # a URL or citation inside code does not back anything
    assert len(run("Studies show it works. `[1]`")) == 1
    assert len(run("Studies show it works.\n```\nhttps://example.org\n```\n")) == 1
    # unterminated fence swallows the rest, like CommonMark
    assert run("Intro.\n```\nStudies show it works.") == []


# --- offsets, newlines, ids ---------------------------------------------------
def test_offsets_after_emoji_and_combining_marks():
    t = "😀 Studies show it works. élan: experts say so. 👍🏽 Research suggests more."
    fs = run(t)
    assert [f.claim.text for f in fs] == ["Studies show", "experts say", "Research suggests"]
    t = "ก็จริงนะ น้ำ ผู้ปุ่ นี่ มีการศึกษาพบว่าดี และ 😀 ผู้เชี่ยวชาญระบุว่าดี"
    fs = run(t)
    assert [f.claim.text for f in fs] == ["มีการศึกษาพบว่า", "ผู้เชี่ยวชาญระบุว่า"]


def test_offsets_exact_when_text_is_not_nfc():
    # decomposed accents and Thai marks typed in non-canonical order: matching normalises,
    # offsets and text still come from the ORIGINAL string
    t = "café ผู้ studies show x. " + "มีการศึกษาพบว่า ok"
    assert not unicodedata.is_normalized("NFC", t)
    fs = run(t)
    assert [f.claim.text for f in fs] == ["studies show", "มีการศึกษาพบว่า"]
    # tone mark before vowel (non-canonical order) still matches the Thai phrase
    t2 = "ผู้เชี่ยวชาญระบุว่า ok"
    assert len(run(t2)) == 1


def test_crlf_and_blank_input():
    t = "Studies show it works.\r\nIt is nice.\r\nIt is cheap.\r\nPer the WHO.\r\n"
    fs = run(t)
    assert len(fs) == 1 and fs[0].claim.text == "Studies show"
    assert run("Studies show it works.\r\nPer the WHO.\r\n") == []
    assert run("Studies\r\nshow it works.")[0].claim.text == "Studies\r\nshow"
    assert scan("") == [] and scan("   \n ") == [] and run("😀") == []


def test_ids_increase_in_document_order():
    t = "Studies show A. Research suggests B.\n\nExperts say C.\n\nThe rate is 73.42%.\n\nมีงานวิจัยพบว่า D"
    fs = run(t)
    assert [f.claim.id for f in fs] == ["v1", "v2", "v3", "v4", "v5"]
    assert [f.claim.start for f in fs] == sorted(f.claim.start for f in fs)


def test_phrase_and_reason_shape():
    f = run("Studies show that coffee is healthy.")[0]
    assert f.claim.data == {"phrase": "Studies show"} and f.claim.text == "Studies show"
    assert f.reason.startswith("Says 'Studies show' but names no study") and "ask for the source" in f.reason
    assert f.evidence == ()
    f = run("studies   show it")[0]
    assert f.claim.data["phrase"] == "studies show"      # whitespace collapsed in data, not in text
    assert f.claim.text == "studies   show"


# --- performance / adversarial ------------------------------------------------
def _fast(text, limit=1.0):
    t0 = time.perf_counter()
    fs = run(text)
    dt = time.perf_counter() - t0
    assert dt < limit, f"{dt:.2f}s for {len(text)} chars"
    return fs


def test_very_long_input_is_fast():
    fs = _fast(("Studies show that it works. " * 8000)[:200_000])
    assert len(fs) == 8000 or len(fs) > 7000
    _fast(("มีการศึกษาพบว่า ดีมาก " * 12000)[:200_000])
    _fast(("The rate is 12.34% and 1,284,903 users. " * 5000)[:200_000])
    _fast("Plain filler text with no triggers at all, just words. " * 4000)


def test_adversarial_inputs_do_not_backtrack():
    for s in ("a" * 200_000, " " * 200_000, "(" * 200_000, "(A, " * 50_000, "A " * 100_000, "`" * 200_000,
              "`x ``y " * 25_000, "1" * 200_000, "1.00 " * 40_000, "```x\n" * 40_000, "a\n" * 100_000,
              ". " * 100_000, "http://x " * 25_000, "Foo " * 50_000, "1,234,567 users " * 12_000,
              "studies" + " " * 200_000 + "show", "it is" + " " * 100_000 + "x", "1." * 100_000):
        _fast(s)


def test_never_verified_or_contradicted():
    texts = ["Studies show it works.", "73.42% of 1,284,903 users agree.", "มีงานวิจัยพบว่าดี",
             "Studies show it works (Smith, 2020)."]
    for t in texts:
        for f in run(t):
            assert f.status is Status.UNVERIFIED and f.module == "vague"


# --- realistic paragraphs -----------------------------------------------------
def test_realistic_english_paragraphs():
    p1 = ("Intermittent fasting has become popular. Studies show that it can improve insulin sensitivity. "
          "Many experts recommend starting with a 12-hour window. You should talk to your doctor first.")
    assert phrases(p1) == ["Studies show", "Many experts recommend"]
    p2 = ("A 2021 randomized trial in the New England Journal of Medicine found a 4.3 kg difference. "
          "Researchers at Stanford University have found similar effects. Dr. Lee (2022) agrees.")
    assert run(p2) == []
    p3 = ("Python 3.12 was released in October 2023. The function runs in O(n log n) time. "
          "Use `pip install requests` to get started. Version 2.31.0 is the latest.")
    assert run(p3) == []
    p4 = ("The company grew 12% last year. It is widely believed that remote work boosts productivity, "
          "but the numbers vary: revenue reached $1,284,903 while headcount stayed at 250.")
    assert phrases(p4) == ["It is widely believed"]


def test_realistic_thai_paragraphs():
    p1 = ("การนอนหลับให้เพียงพอเป็นสิ่งสำคัญ มีงานวิจัยพบว่าผู้ที่นอนน้อยกว่า 6 ชั่วโมงมีความเสี่ยงโรคหัวใจสูงขึ้น "
          "ผู้เชี่ยวชาญแนะนำว่าควรนอนวันละ 7-9 ชั่วโมง")
    assert phrases(p1) == ["มีงานวิจัยพบว่า", "ผู้เชี่ยวชาญแนะนำ"]
    p2 = ("องค์การอนามัยโลกแนะนำให้ผู้ใหญ่ออกกำลังกายอย่างน้อย 150 นาทีต่อสัปดาห์ "
          "ข้อมูลจากกระทรวงสาธารณสุขปี 2565 ระบุว่าคนไทยออกกำลังกายน้อยลง")
    assert run(p2) == []
    p3 = ("ราคาน้ำมันวันนี้ลิตรละ 35.50 บาท ปี พ.ศ. 2567 ตามที่ ดร.สมชาย กล่าว "
          "ใช้ฟังก์ชัน `scan()` เพื่อตรวจสอบ")
    assert run(p3) == []
    p4 = "ว่ากันว่าการดื่มน้ำมะนาวตอนเช้าช่วยลดน้ำหนัก ทั้งที่ ผลสำรวจชี้ว่าคนไทยร้อยละ 73.42 ไม่เคยลอง"
    assert phrases(p4) == ["ว่ากันว่า", "ผลสำรวจชี้ว่า", "ร้อยละ 73.42"]


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("ok  ", name)
            except Exception as e:
                import traceback; traceback.print_exc()
                fails += 1; print("FAIL", name, repr(e))
    sys.exit(1 if fails else 0)
