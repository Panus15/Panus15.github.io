"""Tests for engine/extract.py (rule-based claim splitter). Offline.

Run: python3 tests/test_extract.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.extract import extract
from engine.models import ClaimKind

URL, DOI, REF, ARITH, CODE = (ClaimKind.URL, ClaimKind.DOI, ClaimKind.REFERENCE,
                              ClaimKind.ARITHMETIC, ClaimKind.CODE)


def ex(text):
    """extract() plus the invariants that must hold for EVERY claim."""
    claims = extract(text)
    for i, c in enumerate(claims, 1):
        assert c.id == f"c{i}", (c.id, i)
        assert text[c.start:c.end] == c.text, (c, text[c.start:c.end])
        assert 0 <= c.start < c.end <= len(text)
    assert [c.start for c in claims] == sorted(c.start for c in claims), "not in document order"
    return claims


def of(claims, kind):
    return [c for c in claims if c.kind == kind]


def urls(text):
    return [c.data["url"] for c in of(ex(text), URL)]


def ariths(text):
    return [(c.data["expression"], c.data["claimed"]) for c in of(ex(text), ARITH)]


# ---------------------------------------------------------------- basics

def test_empty_and_trivial():
    assert ex("") == []
    assert ex("   \n\n  ") == []
    assert ex("hello world") == []


def test_only_thai_no_claims():
    assert ex("สวัสดีครับ วันนี้อากาศดีมาก ประเทศไทยมีสี่ภาค") == []


# ---------------------------------------------------------------- URL

def test_url_basic_and_trailing_period():
    cs = ex("See https://example.com/a/b. Then more.")
    assert [c.text for c in cs] == ["https://example.com/a/b"]
    assert ex("(https://example.com/x), ok;")[0].text == "https://example.com/x"
    assert ex('"https://example.com/x"!')[0].text == "https://example.com/x"


def test_url_glued_to_thai():
    cs = ex("ดูที่ https://example.com/pathครับ และต่อ")
    assert [c.text for c in cs] == ["https://example.com/path"]
    cs = ex("ลิงก์https://example.com/a?b=1ซึ่งดี")
    assert [c.text for c in cs] == ["https://example.com/a?b=1"]


def test_url_wikipedia_parens():
    cs = ex("อ่าน https://en.wikipedia.org/wiki/Foo_(bar). ได้")
    assert cs[0].data["url"] == "https://en.wikipedia.org/wiki/Foo_(bar)"
    cs = ex("(see https://en.wikipedia.org/wiki/Foo_(bar))")
    assert cs[0].text == "https://en.wikipedia.org/wiki/Foo_(bar)"


def test_url_markdown_link_once():
    cs = ex("Read [the docs](https://docs.python.org/3/) now")
    assert len(cs) == 1 and cs[0].text == "https://docs.python.org/3/"
    # link text that is itself the URL: still one claim
    cs = ex("[https://x.org](https://x.org)")
    assert len(cs) == 1 and cs[0].start == len("[https://x.org](")
    cs = ex("[t](https://a.com/x_(y))")
    assert cs[0].text == "https://a.com/x_(y)"
    cs = ex('[t](https://a.com/x "title")')
    assert cs[0].text == "https://a.com/x"


def test_url_angle_brackets():
    cs = ex("go <https://example.com/a> now")
    assert [c.text for c in cs] == ["https://example.com/a"]


def test_url_bare_www():
    cs = ex("ไปที่ www.example.com/x, ครับ")
    assert cs[0].text == "www.example.com/x"
    assert cs[0].data["url"] == "https://www.example.com/x"
    assert ex("see www.nonsense and www.") == []


def test_url_incomplete_not_emitted():
    assert ex("just https:// alone") == []
    assert ex("ftp://example.com/a") == []


def test_url_duplicates_are_two_claims():
    cs = ex("https://a.com/x and https://a.com/x")
    assert len(cs) == 2


def test_url_in_code_not_emitted():
    assert ex("```\nhttps://evil.example/x\n```") [0].kind == CODE
    assert of(ex("```\nhttps://evil.example/x\n```"), URL) == []
    assert of(ex("run `curl https://evil.example/x` now"), URL) == []
    assert of(ex("run ``curl https://e.example/x`` now"), URL) == []


def test_url_and_bold_markup():
    assert urls("**https://example.com/a**") == ["https://example.com/a"]


# ---------------------------------------------------------------- DOI

def test_doi_forms():
    for t, want in [
        ("doi:10.1000/ABC.1", "10.1000/abc.1"),
        ("DOI: 10.1038/nature123.", "10.1038/nature123"),
        ("see 10.1145/3292500.3330701, ok", "10.1145/3292500.3330701"),
        ("(10.1016/S0140-6736(20)30183-5)", "10.1016/s0140-6736(20)30183-5"),
        ("doi.org/10.1000/xyz", "10.1000/xyz"),
    ]:
        cs = of(ex(t), DOI)
        assert len(cs) == 1 and cs[0].data == {"doi": want}, (t, cs)


def test_doi_url_is_one_doi_claim_not_url():
    cs = ex("https://doi.org/10.1000/ABC(2)v3). x")
    assert len(cs) == 1 and cs[0].kind == DOI
    assert cs[0].data["doi"] == "10.1000/abc(2)v3"
    assert cs[0].text == "https://doi.org/10.1000/ABC(2)v3)"[:-1]
    cs = ex("[paper](https://dx.doi.org/10.1000/xyz)")
    assert [c.kind for c in cs] == [DOI]


def test_doi_inside_other_url_is_just_url():
    cs = ex("https://link.springer.com/article/10.1007/s11222-020-09987-4")
    assert [c.kind for c in cs] == [URL]


def test_doi_glued_to_thai_and_not_doi_shapes():
    cs = ex("ดู 10.1000/xyz123ครับ")
    assert cs[0].data["doi"] == "10.1000/xyz123"
    assert of(ex("Not 10.5/abc and 110.1234/abc"), DOI) == []


# ---------------------------------------------------------------- ARITHMETIC

def test_arith_basic_ops():
    assert ariths("2 + 3 = 5") == [("2 + 3", "5")]
    assert ariths("7 × 8 = 54") == [("7 * 8", "54")]
    assert ariths("100 ÷ 4 = 25") == [("100 / 4", "25")]
    assert ariths("10 − 4 = 6") == [("10 - 4", "6")]
    assert ariths("3 x 4 = 12") == [("3 * 4", "12")]
    assert ariths("3 X 4 = 12") == [("3 * 4", "12")]
    assert ariths("2^10 = 1024") == [("2 ^ 10", "1024")]
    assert ariths("1+2+3+4=10") == [("1 + 2 + 3 + 4", "10")]
    assert ariths("(1 + 2) * 3 = 9") == [("(1 + 2) * 3", "9")]


def test_arith_thousands_decimals_percent():
    assert ariths("1,200 + 300 = 1,500") == [("1200 + 300", "1500")]
    assert ariths("3.5 × 2 = 7.0") == [("3.5 * 2", "7.0")]
    assert ariths("50% + 20% = 70%") == [("50% + 20%", "70%")]
    assert ariths("3 - 5 = -2") == [("3 - 5", "-2")]


def test_arith_thai_digits_and_text():
    cs = ex("ผลรวมคือ ๑๒๓ + ๔๕๖ = ๕๗๙ ถูกต้อง")
    a = of(cs, ARITH)
    assert len(a) == 1 and a[0].text == "๑๒๓ + ๔๕๖ = ๕๗๙"
    assert a[0].data == {"expression": "123 + 456", "claimed": "579"}
    assert ariths("ราคารวม 1,200 + 300 = 1,500 บาท") == [("1200 + 300", "1500")]


def test_arith_verbal_percent_of():
    assert ariths("20% of 1,200 is 240.") == [("20% * 1200", "240")]
    assert ariths("15% of 80 equals 12") == [("15% * 80", "12")]
    assert ariths("25% of 200 = 50") == [("25% * 200", "50")]
    assert ariths("20% ของ 50 คือ 10") == [("20% * 50", "10")]
    assert ariths("20% ของ 50 เท่ากับ 10") == [("20% * 50", "10")]
    assert ariths("๒๐% ของ ๕๐ = ๑๐") == [("20% * 50", "10")]
    assert len(ariths("20% of 50 = 10")) == 1


def test_arith_markdown_emphasis_and_bullets():
    assert ariths("**9 × 9 = 81**") == [("9 * 9", "81")]
    assert ariths("9 × 9 = **81**") == [("9 * 9", "81")]
    assert ariths("- 10 - 4 = 6") == [("10 - 4", "6")]
    assert ariths("1. 5 + 3 = 8") == [("5 + 3", "8")]
    assert ariths("`2 + 2 = 4`") == [("2 + 2", "4")]


def test_arith_multiple_per_line():
    assert ariths("1 + 1 = 2, 2 + 2 = 4") == [("1 + 1", "2"), ("2 + 2", "4")]


def test_arith_false_positive_shapes():
    for t in [
        "On 2024-01-05 we met.",            # date, no '='
        "Call 081-234-5678 now",            # phone
        "Between 5-10 people came",         # range
        "Version 1.2.3 and v2.0.1 shipped", # versions
        "1. one\n2. two\n3. three",         # list numbering
        "ปี 2020-2024 และ 5 - 3 คน",        # hyphen between numbers, no '='
        "x = 5 and y = 3",                  # assignment
        "2x + 3 = 13",                      # algebra
        "f(2) = 4",
        "g(3+4) = 7",
        "x2 + 3 = 7",
        "v1.2 + 3 = 4",
        "a == b and 1 <= 2 and 3 >= 1",
        "x += 1",
        "ratio 3:1 = 75%",
        "10 kg x 9.8 = ",
        "2024-01-05 = Friday",
        "12/25/2024 = Christmas",
        "1.2.3 + 4 = 7",
        "-3 - 5 = -8",                      # unary minus: refuse rather than guess
        "5 = 5",                            # no operator
        "3 + 4 = 7 + 1 = ",                 # right side is an expression (first '=')
    ]:
        got = ariths(t)
        assert got == [] or t == "3 + 4 = 7 + 1 = ", (t, got)
    # right side being an expression: only the second equation may survive
    assert ariths("1 + 1 = 2 + 2 = 4") == [("2 + 2", "4")]


def test_arith_url_number_not_arithmetic():
    cs = ex("see https://example.com/a?x=1+2=3 for more")
    assert [c.kind for c in cs] == [URL]
    cs = ex("https://example.com/calc/2+3=5")
    assert [c.kind for c in cs] == [URL]


def test_arith_in_code_fence_not_emitted():
    cs = ex("```python\nx = 1 + 2 = 3\nprint(2*3=6)\n```")
    assert [c.kind for c in cs] == [CODE]
    cs = ex("```\n20% of 50 is 10\n```")
    assert [c.kind for c in cs] == [CODE]


def test_arith_inside_doi_not_emitted():
    assert of(ex("doi:10.1000/1+2=3"), ARITH) == []


# ---------------------------------------------------------------- CODE

def test_code_basic_and_language():
    cs = ex("before\n```Python\nprint(1)\n```\nafter")
    assert len(cs) == 1 and cs[0].kind == CODE
    assert cs[0].data == {"language": "python", "code": "print(1)", "expected_output": None}
    assert cs[0].text == "```Python\nprint(1)\n```"
    cs = ex("```\nno lang\n```")
    assert cs[0].data["language"] == ""
    cs = ex("~~~js\nconsole.log(1)\n~~~")
    assert cs[0].data["language"] == "js" and cs[0].data["code"] == "console.log(1)"
    cs = ex("```python title=x.py\nx=1\n```")
    assert cs[0].data["language"] == "python"


def test_code_multiple_blocks_and_empty_block():
    cs = ex("```a\n1\n```\ntext\n```b\n2\n```\n```\n```")
    assert [c.data["language"] for c in cs] == ["a", "b", ""]
    assert cs[2].data["code"] == ""


def test_code_expected_output_fenced():
    for lang in ("output", "text", "console", "stdout", "result"):
        cs = ex(f"```python\nprint(2+2)\n```\n```{lang}\n4\n```\n")
        assert len(cs) == 1, lang
        assert cs[0].data["expected_output"] == "4", lang
    # blank line allowed between
    cs = ex("```python\nprint(2+2)\n```\n\n```output\n4\n```")
    assert len(cs) == 1 and cs[0].data["expected_output"] == "4"
    # a non-output language is a separate block
    cs = ex("```python\na\n```\n```python\nb\n```")
    assert len(cs) == 2 and cs[0].data["expected_output"] is None


def test_code_expected_output_prose_label():
    cs = ex("```python\nprint(2+2)\n```\nOutput:\n```\n4\n```")
    assert len(cs) == 1 and cs[0].data["expected_output"] == "4"
    cs = ex("```python\nprint(2+2)\n```\nผลลัพธ์:\n\n```\n4\n```\n")
    assert len(cs) == 1 and cs[0].data["expected_output"] == "4"
    cs = ex("```python\nprint(2+2)\n```\n# Output\n```\n4\n```")
    assert len(cs) == 1 and cs[0].data["expected_output"] == "4"
    # a label with no fenced block after it gives nothing
    cs = ex("```python\nprint(2+2)\n```\nOutput: 4\nand more prose")
    assert cs[0].data["expected_output"] is None
    # an unrelated prose line does not capture the next block
    cs = ex("```python\na\n```\nHere is another one:\n```python\nb\n```")
    assert len(cs) == 2 and cs[0].data["expected_output"] is None


def test_code_unterminated_fence():
    t = "intro\n```python\nx = 1\ny = 2\n"
    cs = ex(t)
    assert len(cs) == 1
    assert cs[0].data["code"] == "x = 1\ny = 2"
    assert cs[0].end == len(t.rstrip("\n"))
    cs = ex("```")
    assert len(cs) == 1 and cs[0].data["code"] == ""


def test_code_longer_fences_nest():
    t = "````md\n```python\nnested\n```\n````\nafter https://a.com/x"
    cs = ex(t)
    assert [c.kind for c in cs] == [CODE, URL]
    assert cs[0].data["code"] == "```python\nnested\n```"
    assert cs[0].data["language"] == "md"
    # a shorter closer does not close; closing needs >= opening length
    cs = ex("```` \nx\n```\ny\n````")
    assert len(cs) == 1 and cs[0].data["code"] == "x\n```\ny"
    # a tilde fence is not closed by backticks
    cs = ex("~~~\nx\n```\ny\n~~~")
    assert cs[0].data["code"] == "x\n```\ny"


def test_code_inline_triple_backticks_is_not_a_fence():
    assert of(ex("use ```x``` inline"), CODE) == []


def test_code_indented_fence_in_list():
    t = "1. step\n   ```bash\n   ls -la\n   ```\n2. next"
    cs = ex(t)
    assert len(cs) == 1 and cs[0].data["code"] == "ls -la"


# ---------------------------------------------------------------- REFERENCE

def test_ref_apa_line():
    t = "Smith, J. A., & Jones, B. (2020). Deep learning for everyone. Journal of AI, 12(3), 45-67."
    cs = ex(t)
    assert len(cs) == 1 and cs[0].kind == REF
    assert cs[0].data == {"title": "Deep learning for everyone", "authors": ["Smith", "Jones"], "year": 2020}
    assert cs[0].text == t


def test_ref_apa_many_authors_and_et_al():
    t = "Vaswani, A., Shazeer, N., Parmar, N. (2017). Attention is all you need. NeurIPS."
    c = ex(t)[0]
    assert c.data["authors"] == ["Vaswani", "Shazeer", "Parmar"] and c.data["year"] == 2017
    c = ex("Brown, T. et al. (2020). Language models are few-shot learners. NeurIPS.")[0]
    assert c.data["authors"] == ["Brown"] and c.data["title"] == "Language models are few-shot learners"


def test_ref_in_list_with_doi_and_url_emits_all():
    t = ("## References\n"
         "1. Smith, J. (2020). Deep learning for everyone. *Journal of AI*. "
         "https://doi.org/10.1000/jai.2020.123\n"
         "2. Lee, K. (2019). Notes on graphs. See https://example.com/graphs.")
    cs = ex(t)
    kinds = [c.kind for c in cs]
    assert kinds.count(REF) == 2 and kinds.count(DOI) == 1 and kinds.count(URL) == 1
    r1 = of(cs, REF)[0]
    assert r1.data["title"] == "Deep learning for everyone"
    assert "https://doi.org/10.1000/jai.2020.123" in r1.text  # span covers the whole entry


def test_ref_bare_author_needs_context():
    # bare "Author (Year). Title." is accepted in a list / references section
    cs = ex("- World Health Organization (2021). Global report on health. Geneva: WHO.")
    assert len(cs) == 1 and cs[0].data["authors"] == ["World Health Organization"]
    cs = ex("## References\nOECD (2019). Education at a glance. Paris.")
    assert len(cs) == 1 and cs[0].data["year"] == 2019
    # ...but not in running prose
    assert ex("Pandemic (2020). Things happened quickly in that year.") == []
    assert ex("Smith et al. (2020) found that it works.") == []


def test_ref_inline_needs_title():
    assert ex("As shown by Smith (2020), the market grew.") == []
    assert ex("This is vague (Smith & Jones, 2020) and unsourced.") == []
    assert ex("Kim et al. (2018) reported gains.") == []
    cs = ex('Vaswani et al. (2017) "Attention Is All You Need" shows this.')
    assert len(cs) == 1 and cs[0].kind == REF
    assert cs[0].data == {"title": "Attention Is All You Need", "authors": ["Vaswani"], "year": 2017}
    assert cs[0].text == 'Vaswani et al. (2017) "Attention Is All You Need"'
    cs = ex("As in (Smith & Jones, 2020) *Graph Theory Basics*, we see")
    assert cs[0].data == {"title": "Graph Theory Basics", "authors": ["Smith", "Jones"], "year": 2020}
    cs = ex("the book *Thinking, Fast and Slow* (Kahneman, 2011) is good")
    assert cs[0].data["authors"] == ["Kahneman"] and cs[0].data["title"] == "Thinking, Fast and Slow"
    cs = ex("the paper “Deep Residual Learning” (He et al., 2016) shows")
    assert cs[0].data["title"] == "Deep Residual Learning" and cs[0].data["authors"] == ["He"]


def test_ref_numbered_line_with_quoted_or_italic_title():
    cs = ex('2. Brown, T. et al., "Language Models are Few-Shot Learners", NeurIPS, 2020.')
    assert len(cs) == 1 and cs[0].kind == REF
    assert cs[0].data == {"title": "Language Models are Few-Shot Learners", "authors": ["Brown"], "year": 2020}
    assert cs[0].text.startswith("Brown")  # marker is not part of the span
    cs = ex("- Goodfellow, I., Bengio, Y. *Deep Learning*. MIT Press, 2016.")
    assert cs[0].data["year"] == 2016 and cs[0].data["authors"] == ["Goodfellow", "Bengio"]
    # no year or no title-ish thing -> nothing
    assert ex('- Brown, T. "Language Models are Few-Shot Learners", NeurIPS.') == []
    assert ex("- Brown, T. Language Models are Few-Shot Learners, 2020.") == []
    # prose bullets that merely contain a quote and a year are not references
    assert ex('- He said "this is a great idea" in 2020.') == []
    assert ex('- The report "Global Economic Outlook" published in 2020 says growth is slow.') == []
    assert ex("- *Note*: revenue grew in 2020.") == []


def test_ref_thai_buddhist_year():
    cs = ex("- สมชาย ใจดี (2563). การเรียนรู้ของเครื่อง. กรุงเทพ: สำนักพิมพ์.")
    assert len(cs) == 1 and cs[0].data["year"] == 2020
    assert cs[0].data["authors"] == ["สมชาย ใจดี"]
    assert cs[0].data["title"] == "การเรียนรู้ของเครื่อง"
    assert "(2563)" in cs[0].text  # span keeps the year as written
    cs = ex('ตามที่ Chomsky (๒๕๐๐) *Syntactic Structures* กล่าวไว้')
    assert cs[0].data["year"] == 1957 and "๒๕๐๐" in cs[0].text


def test_ref_thai_prose_not_swallowed_as_author():
    t = 'จากการศึกษาของ Vaswani et al. (2017) "Attention Is All You Need" พบว่าดี'
    cs = ex(t)
    assert len(cs) == 1
    assert cs[0].text == 'Vaswani et al. (2017) "Attention Is All You Need"'
    assert cs[0].data["authors"] == ["Vaswani"]


def test_ref_not_inside_code():
    assert ex('```\nVaswani et al. (2017) "Attention Is All You Need"\n```')[0].kind == CODE
    assert len(ex('```\nVaswani et al. (2017) "Attention Is All You Need"\n```')) == 1


# ---------------------------------------------------------------- misc / robustness

def test_crlf_line_endings():
    t = "intro\r\n```python\r\nprint(1)\r\n```\r\n```output\r\n1\r\n```\r\nsee https://a.com/x\r\nand 2 + 2 = 4\r\n"
    cs = ex(t)
    assert [c.kind for c in cs] == [CODE, URL, ARITH]
    assert cs[0].data == {"language": "python", "code": "print(1)", "expected_output": "1"}
    assert "\r" not in cs[0].data["code"]
    assert cs[1].text == "https://a.com/x"
    t = "- Smith, J. (2020). Deep learning for everyone. Journal.\r\n- Lee, K. (2019). Notes on graphs again.\r\n"
    cs = ex(t)
    assert len(cs) == 2 and all("\r" not in c.text for c in cs)
    cs = ex("a\r\n```\r\nunterminated\r\n")
    assert cs[0].data["code"] == "unterminated"


def test_offsets_exact_after_thai_combining_chars():
    t = "ก็ได้ ที่นี่ ผู้ใช้ น้ำ ๒ + ๓ = ๕ https://example.com/ก็ กิ่ง doi:10.1000/x1"
    cs = ex(t)
    assert [c.kind for c in cs] == [ARITH, URL, DOI]
    assert cs[0].text == "๒ + ๓ = ๕"
    assert t[cs[1].start:cs[1].end] == "https://example.com/"
    assert cs[2].data["doi"] == "10.1000/x1"
    # same with CJK / emoji (astral) before
    t = "😀 日本語 https://a.com/x 😀 1 + 1 = 2"
    assert [c.kind for c in ex(t)] == [URL, ARITH]


def test_ids_and_document_order_mixed():
    t = ("Intro 1 + 1 = 2 then https://a.com/x and doi:10.1000/abc\n"
         "```py\nx\n```\n"
         "- Smith, J. (2020). Deep learning for everyone. Journal.\n")
    cs = ex(t)
    assert [c.kind for c in cs] == [ARITH, URL, DOI, CODE, REF]
    assert [c.id for c in cs] == ["c1", "c2", "c3", "c4", "c5"]


def test_realistic_mixed_paragraph():
    t = ("จากการศึกษาของ Vaswani et al. (2017) \"Attention Is All You Need\" พบว่า Transformer ดีกว่า RNN "
         "ดูเพิ่มเติมที่ https://arxiv.org/abs/1706.03762ครับ อัตราการเติบโต 15% ของ 2,000 คือ 300 บาท "
         "ดังนั้น 2,000 + 300 = 2,300 บาท\n\n```python\nprint(15/100*2000)\n```\nผลลัพธ์:\n```\n300.0\n```\n")
    cs = ex(t)
    assert [c.kind for c in cs] == [REF, URL, ARITH, ARITH, CODE]
    assert cs[2].data == {"expression": "15% * 2000", "claimed": "300"}
    assert cs[3].data == {"expression": "2000 + 300", "claimed": "2300"}
    assert cs[4].data["expected_output"] == "300.0"


def test_pathological_input_is_fast():
    t0 = time.time()
    big = ("Smith (2020) " * 3000) + ("1 + 2 + 3 + 4 + 5 = " * 2000) + ("https://a.com/" * 2000) + ("(" * 5000) + ("*" * 5000)
    ex(big)
    ex("= " * 20000)
    ex("`" * 10000)
    ex("A" * 50000 + " (2020). " + "B " * 20000)
    assert time.time() - t0 < 20


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("ok  ", name)
            except Exception as e:
                import traceback
                fails += 1; print("FAIL", name, repr(e)); traceback.print_exc()
    sys.exit(1 if fails else 0)
