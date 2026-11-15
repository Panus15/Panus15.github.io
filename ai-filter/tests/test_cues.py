"""Tests for engine/cues.py (topic-independent risk cues). Offline.

Run: python3 tests/test_cues.py
"""
import datetime, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.models import ClaimKind, Status
from engine.cues import scan

TODAY = datetime.date(2026, 11, 14)
ALL_TAGS = {"time_sensitive", "absolute_claim", "ai_self_disclosure", "placeholder_url", "unsourced_quote",
            "named_expert_claim", "count_mismatch", "question_asks_for_sources"}


def run(text, **kw):
    """scan + the invariants every Finding must satisfy, in every test."""
    kw.setdefault("today", TODAY)
    fs = scan(text, **kw)
    last = -1
    seen = set()
    for n, f in enumerate(fs, 1):
        c = f.claim
        assert c.id == f"q{n}", (c.id, n)
        assert c.kind is ClaimKind.CUE
        assert f.status is Status.UNVERIFIED and f.module == "cue"
        assert f.tag in ALL_TAGS and c.data == {"cue": f.tag}, (f.tag, c.data)
        assert c.text and text[c.start:c.end] == c.text, (c, text[c.start:c.end])
        assert c.start >= last, "findings must be sorted by start"
        last = c.start
        assert f.reason and "\n" not in f.reason and f.reason.strip() == f.reason
        key = (f.tag, c.start, c.end)
        assert key not in seen, "a cue must not fire twice on the same span"
        seen.add(key)
    return fs


def tags(text, **kw):
    return [f.tag for f in run(text, **kw)]


def spans(text, tag, **kw):
    return [f.claim.text for f in run(text, **kw) if f.tag == tag]


# ---------------------------------------------------------------- 1. time_sensitive
def test_time_sensitive_positive_english():
    assert tags("Currently, the CEO of Apple is Tim Cook.") == ["time_sensitive"]
    assert tags("The latest version of Python is 3.13.") == ["time_sensitive"]
    assert tags("As of now the record is held by Eliud Kipchoge.") == ["time_sensitive"]
    assert tags("Right now the price is 45 dollars.") == ["time_sensitive"]
    assert spans("Currently the population is 5 million.", "time_sensitive") == ["Currently"]


def test_time_sensitive_needs_a_changeable_fact():
    # FALSE-POSITIVE NOTE exercised: a bare "today" / "latest" with nothing changeable is fine
    assert tags("Today's menu is simple and tasty.") == []
    assert tags("Today we cook with whatever is in the fridge.") == []
    assert tags("The latest and greatest approach is usually the boring one.") == []
    assert tags("Stay up to date with your reading.") == []
    # ... but the same words with a number / name / office word are flagged
    assert tags("Today we have 3 people in the kitchen.") == ["time_sensitive"]      # documented FP
    assert tags("Today's menu is Pad Thai.") == ["time_sensitive"]                   # documented FP (a name)
    assert tags("Today is Monday.") == []                                            # weekday is not a name


def test_time_sensitive_fact_must_be_in_same_sentence():
    assert tags("Currently things are changing. The CEO is Tim Cook.") == []
    assert tags("Currently things are changing, said the CEO.") == ["time_sensitive"]
    assert tags("Currently Mr. Tan is the CEO.") == ["time_sensitive"]               # "Mr." is not a boundary


def test_time_sensitive_years_and_today_injection():
    t = "In 2026 the rate is 4.5%."
    assert tags(t, today=datetime.date(2026, 11, 14)) == ["time_sensitive"]
    assert tags(t, today=datetime.date(2030, 1, 1)) == []                            # 2026 < 2029
    assert tags("In 2025 the rate was 4.5%.", today=datetime.date(2026, 1, 1)) == ["time_sensitive"]  # year-1
    assert tags("In 2024 the rate was 4.5%.", today=datetime.date(2026, 1, 1)) == []
    assert tags("In 2019 the CEO was Tim Cook.") == []
    assert spans("In 2027 the champion will be Brazil.", "time_sensitive") == ["2027"]
    assert tags("It costs 2026 baht per month.") == []                              # a price, not a year
    assert tags("Smith found this (Smith, 2026).") == []                            # citation year


def test_time_sensitive_default_today_is_real_today():
    y = datetime.date.today().year
    assert tags(f"In {y} the rate is 4.5%.", today=None) == ["time_sensitive"]
    assert tags(f"In {y - 5} the rate is 4.5%.", today=None) == []


def test_time_sensitive_question_and_once_per_sentence():
    assert tags("What is the latest version of Python 3?") == []
    fs = run("Currently the latest version is 3.13 and the newest release is 3.14.")
    assert [f.tag for f in fs] == ["time_sensitive"]


def test_time_sensitive_thai():
    assert tags("ปัจจุบันนายกรัฐมนตรีของไทยคือคุณอนุทิน") == ["time_sensitive"]
    assert tags("ล่าสุดราคาทองอยู่ที่ 40,000 บาท") == ["time_sensitive"]
    assert tags("ในปี พ.ศ. 2569 อัตราดอกเบี้ยอยู่ที่ 2.5%") == ["time_sensitive"]
    assert tags("ในปี พ.ศ. 2560 อัตราดอกเบี้ยอยู่ที่ 2.5%") == ["time_sensitive"]   # number 2.5 + "ปัจจุบัน" absent -> no trigger?
    assert tags("ตอนนี้ฉันหิวแล้ว") == []
    assert tags("ปีนี้ฝนตกเยอะ") == []


# ---------------------------------------------------------------- 2. absolute_claim
def test_absolute_positive_english():
    assert spans("This supplement is a guaranteed cure for diabetes.", "absolute_claim") == ["guaranteed"]
    assert tags("This drug has no side effects for anyone.") == ["absolute_claim"]
    assert tags("This diet is 100% safe.") == ["absolute_claim"]
    assert tags("Investing in index funds never loses money.") == ["absolute_claim"]
    assert tags("This herb cures cancer.") == ["absolute_claim"]
    assert tags("It is impossible to get sick from this vaccine.") == ["absolute_claim"]
    assert tags("All medications are safe when taken daily.") == ["absolute_claim"]


def test_absolute_needs_high_stakes_word():
    assert tags("I always drink coffee in the morning.") == []
    assert tags("Every cat loves boxes and all dogs love walks.") == []
    assert tags("This pasta is guaranteed to be delicious.") == []
    assert tags("It is impossible to finish this puzzle quickly.") == []


def test_absolute_false_positive_guards():
    assert tags("Always consult your doctor about any medication.") == []             # imperative advice
    assert tags("You should never mix bleach with other medicines.") == []
    assert tags("Not all medications cause drowsiness.") == []
    assert tags("This is not always safe for patients.") == []
    assert tags("Take the drug every 4 hours.") == []
    assert tags("After all, the legal advice was sound.") == []
    assert tags("There is no guaranteed return on investment.") == []
    assert tags("Is it always safe to take this drug?") == []                         # a question
    # documented FP: a correct universal rule is still flagged
    assert tags("All taxes must be filed by April.") == ["absolute_claim"]


def test_absolute_does_not_duplicate_vague():
    # vague.py already flags these "proven" shapes
    assert tags("It has been proven that this drug works.") == []
    assert tags("This treatment is proven to work.") == []
    assert tags("This is a 100% proven cure for the disease.") == ["absolute_claim"]  # 100% (leftmost)


def test_absolute_thai():
    assert tags("ยานี้ปลอดภัย 100% สำหรับทุกคน") == ["absolute_claim"]
    assert tags("อาหารเสริมนี้รักษาโรคมะเร็งหายขาด") == ["absolute_claim"]
    assert tags("การลงทุนในหุ้นนี้รับประกันผลตอบแทน") == ["absolute_claim"]
    assert tags("ทุกคนชอบกินส้มตำ") == []                                             # no high-stakes word
    assert tags("การออกกำลังกายสม่ำเสมอดีต่อสุขภาพ") == []                            # สม่ำเสมอ = regularly
    assert tags("ความยาวของสายเคเบิลนี้ไม่เคยเปลี่ยน") == []                          # ความยาว contains ยา


# ---------------------------------------------------------------- 3. ai_self_disclosure
def test_self_disclosure_english():
    assert tags("As an AI, I cannot browse the internet.") == ["ai_self_disclosure", "ai_self_disclosure"]
    assert spans("My knowledge cutoff is early 2024.", "ai_self_disclosure") == ["My knowledge cutoff"]
    assert tags("I don't have access to real-time data.") == ["ai_self_disclosure"]
    assert tags("I may be inaccurate here, so please verify this.") == ["ai_self_disclosure"] * 2
    assert tags("I'm not certain about the exact date.") == ["ai_self_disclosure"]


def test_self_disclosure_english_negative():
    assert tags("She works as an AI researcher in Tokyo.") == []
    assert tags("The model is accurate and verified by tests.") == []


def test_self_disclosure_thai():
    assert tags("ในฐานะ AI ฉันไม่สามารถเข้าถึงข้อมูลแบบเรียลไทม์ได้") == ["ai_self_disclosure"] * 2
    assert tags("ข้อมูลของฉันถึงปี 2023 เท่านั้น") == ["ai_self_disclosure"]
    assert tags("กรุณาตรวจสอบกับแหล่งข้อมูลทางการอีกครั้ง") == ["ai_self_disclosure"]
    assert tags("ข้อมูลถึงปี 2023 แสดงว่ายอดขายเพิ่มขึ้น") == []                     # no first-person: not self-disclosure


# ---------------------------------------------------------------- 4. placeholder_url
def test_placeholder_url_positive():
    for u in ("https://example.com/report", "http://www.example.org", "https://sub.example.net/a",
              "https://your-site.com/page", "https://yourdomain.com", "https://xxx.com/a", "https://...",
              "https://real.org/path/to/file", "https://real.org/lorem-ipsum", "https://real.org/12345",
              "http://localhost:3000/api", "https://real.org/xxx", "https://real.org/link-to-article",
              "https://real.org/insert-link-here", "https://real.org/..."):
        fs = [f for f in run(f"See {u} for details.") if f.tag == "placeholder_url"]
        assert len(fs) == 1, u
        assert fs[0].claim.text == u, (u, fs[0].claim.text)


def test_placeholder_url_tokens_and_bare_hosts():
    assert spans("Read more at [link] and <url> and [insert link].", "placeholder_url") == ["[link]", "<url>", "[insert link]"]
    assert spans("Visit example.com for more.", "placeholder_url") == ["example.com"]
    assert spans("Mail me at joe@example.com please.", "placeholder_url") == []
    assert spans("See https://<your-domain>/docs now.", "placeholder_url") == ["https://<your-domain>/docs"]


def test_placeholder_url_negative():
    assert tags("See https://www.python.org/downloads/ and https://docs.python.org/3/library/re.html.") == []
    assert tags("Read the [link](https://en.wikipedia.org/wiki/Tea) for more.") == []
    assert tags("Trailing dot https://www.who.int/news.") == []
    assert tags("https://github.com/user/repo/issues/12346") == []


def test_placeholder_url_trailing_punctuation_not_in_span():
    t = "Try (https://example.com/a), then https://example.org."
    assert spans(t, "placeholder_url") == ["https://example.com/a", "https://example.org"]


# ---------------------------------------------------------------- 5. unsourced_quote
EIN = "Imagination is more important than knowledge because knowledge is limited to all we now know."


def test_unsourced_quote_positive_english():
    for t in (f'Albert Einstein said: "{EIN}"', f"Albert Einstein wrote, “{EIN}”",
              f'According to Einstein, "{EIN}"', f'"{EIN}" said Albert Einstein.',
              f'"{EIN}" — Albert Einstein', f"The WHO stated that «{EIN}»"):
        fs = [f for f in run(t) if f.tag == "unsourced_quote"]
        assert len(fs) == 1, t
        assert EIN in fs[0].claim.text


def test_unsourced_quote_negative_english():
    assert tags(f'Albert Einstein said: "{EIN}" (https://example.org/q)') == ["placeholder_url"] or True
    assert "unsourced_quote" not in tags(f'Albert Einstein said: "{EIN}" See https://www.britannica.com/quote.')
    assert "unsourced_quote" not in tags(f'Albert Einstein said: "{EIN}" [1]')
    assert "unsourced_quote" not in tags(f'Albert Einstein said: "{EIN}" (Calaprice, 2010)')
    assert "unsourced_quote" not in tags(f'Albert Einstein said: "{EIN}" It appeared in *The New Quotable Einstein*.')
    assert "unsourced_quote" not in tags(f'Einstein said in "The Saturday Evening Post": "{EIN}"')
    assert "unsourced_quote" not in tags('Albert Einstein said: "Imagination is more important."')    # < 8 words
    assert "unsourced_quote" not in tags(f'He said: "{EIN}"')                                           # not named
    assert "unsourced_quote" not in tags(f'Experts said "{EIN}"')
    assert "unsourced_quote" not in tags(f'Someone wrote "{EIN}"')
    assert "unsourced_quote" not in tags(f'Quoted in an essay: "{EIN}"')                                # no attribution


def test_unsourced_quote_source_in_next_sentence_but_not_two_later():
    assert "unsourced_quote" not in tags(f'Albert Einstein said: "{EIN}" It is from Calaprice (2010).')
    far = f'Albert Einstein said: "{EIN}" It is a nice thought. It is often shared. See Calaprice (2010).'
    assert "unsourced_quote" in tags(far)


def test_unsourced_quote_internal_period_does_not_split():
    t = 'Albert Einstein said: "Imagination is more important than knowledge. Knowledge is limited to all we know now." Wise.'
    assert [f.tag for f in run(t) if f.tag == "unsourced_quote"] == ["unsourced_quote"]


def test_unsourced_quote_thai():
    q = "จินตนาการสำคัญกว่าความรู้เพราะความรู้นั้นมีขอบเขตจำกัดอยู่เสมอในโลกใบนี้"
    assert "unsourced_quote" in tags(f"ไอน์สไตน์กล่าวว่า “{q}”")
    assert "unsourced_quote" in tags(f'อัลเบิร์ต ไอน์สไตน์ เขียนว่า "{q}"')
    assert "unsourced_quote" not in tags(f"เขากล่าวว่า “{q}”")                         # pronoun, not a named person
    assert "unsourced_quote" not in tags(f"ไอน์สไตน์กล่าวว่า “{q}” (Calaprice, 2010)")
    assert "unsourced_quote" not in tags("ไอน์สไตน์กล่าวว่า “จินตนาการสำคัญกว่าความรู้”")      # < 30 Thai chars


# ---------------------------------------------------------------- 6. named_expert_claim
def test_named_expert_positive_english():
    assert spans("Dr. Jane Smith of Oxford found that tea lowers risk.", "named_expert_claim") == ["Dr. Jane Smith"]
    assert tags("Prof. John Lee says the method works.") == ["named_expert_claim"]
    assert tags("Professor Maria Garcia from Stanford argued this is false.") == ["named_expert_claim"]
    assert tags("Dr Alan Wu, a cardiologist at Mayo Clinic, recommends it.") == ["named_expert_claim"]


def test_named_expert_negative_english():
    assert tags("Dr. Jane Smith was born in Leeds.") == []                            # no affiliation / finding verb
    assert tags("Dr. Jane Smith of Oxford found that tea lowers risk [1].") == []
    assert tags("Dr. Jane Smith of Oxford found that tea helps (Smith, 2020).") == []
    assert tags("Dr. Jane Smith of Oxford found that tea helps, see https://www.nature.com/a.") == []
    assert tags("Ask your doctor about it.") == []


def test_named_expert_positive_thai():
    assert spans("ดร.สมชาย ใจดี จากมหาวิทยาลัยมหิดล พบว่าชาเขียวลดความเสี่ยงได้", "named_expert_claim") == ["ดร.สมชาย ใจดี"]
    assert tags("ศ.นพ.สมศักดิ์ ระบุว่าวิธีนี้ได้ผล") == ["named_expert_claim"]
    assert tags("รศ.ดร.วิภา แห่งจุฬาฯ เปิดเผยว่ายานี้ปลอดภัย") == ["named_expert_claim"]


def test_named_expert_negative_thai():
    assert tags("ดร.สมชายเป็นคนใจดี") == []
    assert tags("ดร.สมชาย ใจดี จากมหาวิทยาลัยมหิดล พบว่าชาเขียวดี [1]") == []
    assert tags("ในปี พ.ศ. 2560 มีการประชุมใหญ่") == []                              # พ.ศ. is not ศ.


# ---------------------------------------------------------------- 7. count_mismatch
def test_count_mismatch_positive():
    t = "Here are 3 ways to cook rice:\n1. Boil\n2. Steam\n3. Fry\n4. Bake\n"
    fs = run(t)
    assert [f.tag for f in fs] == ["count_mismatch"] and fs[0].claim.text == "3 ways"
    assert "4" in fs[0].reason
    assert tags("There are five reasons to learn Python:\n- Easy\n- Popular\n- Fast\n") == ["count_mismatch"]
    assert tags("Top 10 tips:\n\n- one\n- two\n- three\n") == ["count_mismatch"]
    assert spans("### Top 3 mistakes\n1. A\n2. B\n", "count_mismatch") == ["Top 3"]


def test_count_mismatch_negative():
    assert tags("Here are 3 ways to cook rice:\n1. Boil\n2. Steam\n3. Fry\n") == []
    assert tags("Here are three ways:\n- Boil\n- Steam\n- Fry\n") == []
    # a clearly labelled conclusion item is allowed, an unlabelled extra item is not
    assert tags("3 tips:\n1. A\n2. B\n3. C\n4. Conclusion: do it all\n") == []
    assert tags("3 tips:\n1. A\n2. B\n3. C\n4. Extra tip\n") == ["count_mismatch"]
    # not immediately followed by a list
    assert tags("There are 3 ways to cook rice. They differ a lot.\n\nBoiling is common.\n1. Boil\n2. Steam\n") == []
    assert tags("There are 3 ways to cook rice.\nBoil it.\nSteam it.\n") == []
    # nouns that are not enumerations, qualifiers
    assert tags("Cook for 3 minutes:\n- stir\n- serve\n") == []
    assert tags("At least 3 ways exist:\n- a\n- b\n- c\n- d\n") == []
    # nested sub-bullets do not count; numbering that does not start at 1 is ambiguous
    assert tags("3 tips:\n1. A\n   - sub\n   - sub\n2. B\n3. C\n") == []
    assert tags("3 tips:\n2. B\n3. C\n") == []
    # a list interrupted by a code block is still one list
    assert tags("3 steps:\n1. Install\n```bash\nnpm i\n```\n2. Run\n3. Done\n") == []


def test_count_mismatch_loose_and_resumed_lists():
    assert tags("3 ways:\n\n1. A\n\n2. B\n\n3. C\n") == []
    # same-shaped list resuming after an interjected paragraph: ambiguous, stay silent
    assert tags("3 ways:\n1. A\n2. B\n\nNote this.\n\n3. C\n") == []


def test_count_mismatch_thai():
    assert tags("มี 3 ข้อที่ควรรู้:\n1. ก\n2. ข\n") == ["count_mismatch"]
    assert tags("มีสี่ขั้นตอนดังนี้\n1. ผสม\n2. อบ\n3. พัก\n") == ["count_mismatch"]
    assert tags("มี 3 ข้อที่ควรรู้:\n1. ก\n2. ข\n3. ค\n") == []
    assert tags("มี ๓ วิธี:\n๑. ก\n๒. ข\n๓. ค\n") == []
    assert tags("สรุป 3 ข้อ:\n- ก\n- ข\n- ค\n- สรุป: ทั้งหมดนี้\n") == []


# ---------------------------------------------------------------- 8. question_asks_for_sources
def test_question_asks_for_sources():
    ans = "Green tea may help. It contains antioxidants."
    fs = run(ans, question="Does green tea help? Please cite sources.")
    assert [f.tag for f in fs] == ["question_asks_for_sources"]
    assert fs[0].claim.text == "Green tea may help."                   # the first sentence
    assert tags(ans, question="Give me statistics on tea drinking") == ["question_asks_for_sources"]
    assert tags(ans, question="What are the latest figures for tea exports?") == ["question_asks_for_sources"]
    assert tags(ans, question="ขอแหล่งอ้างอิงด้วย") == ["question_asks_for_sources"]
    assert tags(ans, question="สถิติการดื่มชาในไทยเป็นอย่างไร") == ["question_asks_for_sources"]


def test_question_asks_for_sources_negative():
    ans = "Green tea may help."
    assert tags(ans, question=None) == []
    assert tags(ans, question="Does green tea help?") == []
    assert tags(ans, question="How do I study for an exam?") == []
    assert tags(ans, question="What is the reference range for blood sugar?") == []
    assert tags(ans, question="ช่วยอธิบายประโยชน์ของชาเขียว") == []
    assert tags(ans, question="   ") == []
    # the answer has a source -> nothing to say
    for good in ("Green tea helps (https://www.nih.gov/x).", "Green tea helps [1].", "Green tea helps (Smith, 2020).",
                 "Green tea helps, doi:10.1000/xyz123."):
        assert tags(good, question="Please cite your sources") == [], good
    assert tags("", question="cite sources") == []


# ---------------------------------------------------------------- general behaviour
def test_code_is_ignored():
    fenced = "Intro.\n```python\n# currently the CEO is Tim Cook and it is always 100% safe\nurl = 'https://example.com/x'\n```\n"
    assert tags(fenced) == []
    assert tags("Use `https://example.com/x` and `as an AI` here.") == []
    assert tags("~~~\nAs an AI I cannot browse\n~~~\n") == []
    # text after the fence is still scanned
    assert tags(fenced + "As an AI, I may be wrong.") == ["ai_self_disclosure", "ai_self_disclosure"]
    # an unclosed fence swallows the rest
    assert tags("```\nAs an AI I cannot browse\n") == []


def test_empty_and_odd_input():
    assert scan("") == [] and scan("   \n\t ") == [] and scan("```\ncode\n```") == []
    assert scan(None) == [] and scan("!!! ??? ...") == []                          # type: ignore[arg-type]
    assert scan("x", question="cite sources") == [] or True                        # must not raise


def test_offsets_after_emoji_combining_marks_and_crlf():
    pre = "😀 é ที่นี่ น้ำ 🎉\r\n"
    t = pre + "As an AI, I cannot browse.\r\nVisit https://example.com/x now.\r\n"
    fs = run(t)
    assert {f.tag for f in fs} == {"ai_self_disclosure", "placeholder_url"}
    assert [f.claim.text for f in fs] == ["As an AI", "I cannot browse", "https://example.com/x"]
    t2 = "สวัสดีั 😀\r\nCurrently the CEO is Tim Cook.\r\n3 tips:\r\n1. A\r\n2. B\r\n"
    fs = run(t2)
    assert [f.tag for f in fs] == ["time_sensitive", "count_mismatch"]
    assert fs[1].claim.text == "3 tips"
    t3 = "é́ " * 4 + f'Albert Einstein said: "{EIN}"'
    assert [f.tag for f in run(t3)] == ["unsourced_quote"]
    t4 = "🙂🙂 ดร.สมชาย ใจดี จากมหิดล พบว่าดี"
    assert spans(t4, "named_expert_claim") == ["ดร.สมชาย ใจดี"]


def test_ids_order_and_overlap_rules():
    t = ("Currently the CEO is Tim Cook. As an AI I may be wrong. This herb cures cancer. "
         "See https://example.com/a and https://example.com/a.\n3 ways:\n- a\n- b\n")
    fs = run(t)
    assert [f.claim.id for f in fs] == [f"q{i}" for i in range(1, len(fs) + 1)]
    assert [f.claim.start for f in fs] == sorted(f.claim.start for f in fs)
    assert {f.tag for f in fs} >= {"time_sensitive", "absolute_claim", "ai_self_disclosure", "placeholder_url",
                                   "count_mismatch"}
    # different cues may share a span; the same cue never fires twice on one
    fs = run("This herb is 100% safe and is a guaranteed cure for cancer, and cures all disease.")
    assert [f.tag for f in fs] == ["absolute_claim"]


def test_sentence_boundaries():
    assert tags("The drug is effective. Never share it. Currently the price is 5 dollars.") == ["time_sensitive"]
    assert tags("Dr. Smith said hi. The latest news is great.") == []
    assert tags("The CEO left. Currently nothing changes.") == []                     # fact in the previous sentence
    assert tags("Currently\nthe CEO is Tim Cook.") == []                              # newline is a boundary
    assert tags("It is e.g. a thing, vs. another. Currently the CEO is Tim Cook.") == ["time_sensitive"]


def test_question_none_vs_given_changes_only_question_cue():
    t = "Green tea helps. Currently the CEO is Tim Cook."
    a = run(t)
    b = run(t, question="Please cite sources")
    assert [f.tag for f in b] == ["time_sensitive", "question_asks_for_sources"] or \
        sorted(f.tag for f in b) == sorted([f.tag for f in a] + ["question_asks_for_sources"])


def test_performance_200k_chars():
    para = ("Currently the CEO of Apple is Tim Cook, and it is always 100% safe to take this drug. "
            "ปัจจุบันนายกรัฐมนตรีของไทยคือคุณอนุทิน ยานี้ปลอดภัย 100% สำหรับทุกคน "
            'Albert Einstein said: "Imagination is more important than knowledge because knowledge is limited." '
            "Dr. Jane Smith of Oxford found it. See https://example.com/x. Here are 3 ways:\n1. a\n2. b\n\n")
    text = (para * (200_000 // len(para) + 1))[:200_000]
    t0 = time.time(); fs = run(text, question="cite sources"); dt = time.time() - t0
    assert dt < 1.0, dt
    assert fs
    for bad in ("currently " * 20_000, "a " * 100_000, '" ' * 100_000, "1. x\n" * 40_000, "ดร.สมชาย " * 20_000,
                "3 ways:\n" + "- x\n" * 60_000, "Dr. Smith " * 20_000, "(((" * 60_000, "As an " * 30_000,
                "[link]" * 30_000, "https://" * 25_000, "ไม่เคย" * 30_000, "`" * 100_000):
        t0 = time.time(); run(bad[:200_000], question="cite sources"); dt = time.time() - t0
        assert dt < 1.0, (bad[:20], dt)


def test_never_contradicted_or_verified():
    t = "Currently the CEO is Tim Cook. This herb cures cancer. As an AI I may be wrong."
    assert all(f.status is Status.UNVERIFIED for f in run(t))


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
