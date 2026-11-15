"""Tests for engine/computable.py (calendar, duration, unit and percentage claims proved by calculation). Offline.

Run: python3 tests/test_computable.py
"""
import datetime, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.models import ClaimKind, Status
from engine.computable import scan

V, U, C = Status.VERIFIED, Status.UNVERIFIED, Status.CONTRADICTED
TODAY = datetime.date(2026, 11, 15)


def run(text):
    """scan + the invariants every Finding must satisfy, in every test."""
    fs = scan(text, today=TODAY)
    last = (-1, -1)
    for n, f in enumerate(fs, 1):
        c = f.claim
        assert c.id == f"m{n}", (c.id, n)
        assert c.kind is ClaimKind.COMPUTED and f.module == "computable"
        assert text[c.start:c.end] == c.text and c.text, (c, text[c.start:c.end])
        assert (c.start, c.end) >= last, "document order"
        assert c.start >= last[1], "findings do not overlap"
        last = (c.start, c.end)
        assert isinstance(c.data["type"], str) and f.reason
        assert "\n" not in f.reason
    return fs


def one(text, typ=None):
    fs = run(text)
    assert len(fs) == 1, [(f.claim.text, f.claim.data["type"], f.status.value, f.reason) for f in fs]
    if typ:
        assert fs[0].claim.data["type"] == typ, fs[0].claim.data
    return fs[0]


def st(text, typ=None):
    return one(text, typ).status


def none(text):
    fs = run(text)
    assert fs == [], [(f.claim.text, f.claim.data["type"], f.reason) for f in fs]


# ------------------------------------------------------------ calendar_date ---

def test_nonexistent_dates_english():
    for t in ("30 February 2024", "February 30, 2024", "31 April 2021", "31 June 2020", "31 September 2019",
              "31 November 2018", "Feb 30 2024", "32 March 2021", "0 March 2021", "March 32, 2021"):
        assert st(f"It happened on {t}.", "calendar_date") is C, t


def test_leap_years_exact():
    assert st("29 February 1900", "calendar_date") is C        # divisible by 100, not 400
    assert st("29 February 2100", "calendar_date") is C
    assert st("29 February 2023", "calendar_date") is C
    none("29 February 2000")                                    # divisible by 400
    none("29 February 2024")
    none("29 February 1904")
    assert st("February 29, 2100", "calendar_date") is C
    assert st("2100-02-29", "calendar_date") is C
    none("2000-02-29")


def test_numeric_and_iso_formats():
    assert st("on 2023-02-29 the", "calendar_date") is C
    assert st("on 2023/02/30 the", "calendar_date") is C
    assert st("on 29/02/2023 the", "calendar_date") is C
    assert st("on 02/29/2023 the", "calendar_date") is C        # month-first reading forced by 29
    assert st("on 31/04/2023 the", "calendar_date") is C
    assert st("on 30/30/2020 the", "calendar_date") is C        # neither reading exists
    assert st("on 2023-13-05 the", "calendar_date") is C        # month 13
    none("on 29/02/2024 the")
    none("on 2024-02-29 the")


def test_ambiguous_dates_are_not_findings():
    none("It was 03/04/2020.")
    none("It was 12/12/2020 and 1/2/2021 and 3.5.2020.")
    none("Friday 03/04/2020")                       # fits the 3 April reading: ambiguous, not proof
    none("Wednesday 03/04/2020")                    # fits the 4 March reading


def test_valid_standalone_dates_say_nothing():
    none("On 15 March 2021 and March 15, 2021 and 2021-03-15 and 15/03/2021 the thing happened.")
    none("15 มีนาคม 2564 หรือ 15 ม.ค. 2564")
    none("It was in March 2021, around 31 March and in May.")


def test_thai_months_and_buddhist_era():
    assert st("วันที่ 30 กุมภาพันธ์ 2567", "calendar_date") is C
    assert st("31 เมษายน 2564", "calendar_date") is C
    assert st("31 ก.ย. 2564", "calendar_date") is C
    assert st("29 กุมภาพันธ์ 2566", "calendar_date") is C       # BE 2566 = 2023: not leap
    none("29 กุมภาพันธ์ 2567")                                  # BE 2567 = 2024: leap
    none("29 ก.พ. 2567")
    f = one("29 กุมภาพันธ์ พ.ศ. 2566")
    assert f.status is C and "2023" in f.reason
    assert st("29 February 2566 BE", "calendar_date") is C      # explicit BE
    none("29 February 2567 BE")
    none("29 February 2400 CE")                                 # explicit CE is not converted: 2400 is leap
    assert st("29 February 2500 CE", "calendar_date") is C      # 2500 is not leap
    assert st("30 ก.พ. ค.ศ. 2020", "calendar_date") is C


def test_half_parsed_dates_produce_nothing():
    none("The 30 of the month; February is short; 2024.")
    none("may 5 and march 31 are ordinary words")
    none("Feb 30 people came")                                  # month-first, no year, followed by a noun
    none("Born in March 15")                                    # valid anyway
    none("32 June")                                             # yearless day > 31: not enough to prove anything


def test_yearless_impossible_day_is_proof():
    assert st("on 30 February the", "calendar_date") is C
    assert st("from February 30 to March 2", "calendar_date") is C
    none("on 29 February the")                                  # may be a leap year


def test_julian_era_is_convention():
    f = one("29 February 1500", "calendar_date")
    assert f.status is U and "Julian" in f.reason
    assert st("30 February 1500", "calendar_date") is C


# ------------------------------------------------------------------ weekday ---

def test_weekday_verified_and_contradicted():
    assert st("Monday, 15 March 2021", "weekday") is V
    f = one("It was Friday, 15 March 2021.", "weekday")
    assert f.status is C and "Monday" in f.reason and "Friday" in f.reason
    assert st("Fri 3 Jan 2020", "weekday") is V
    assert st("Sat 3 Jan 2020", "weekday") is C
    assert st("March 15, 2021 (Monday)", "weekday") is V
    assert st("15 March 2021 was a Tuesday", "weekday") is C
    assert st("15 March 2021 was a Monday", "weekday") is V
    assert st("Monday 2021-03-15", "weekday") is V
    assert st("Tuesday 2021-03-15", "weekday") is C
    assert st("1 January 2000 was a Saturday", "weekday") is V
    assert st("29 February 2024 fell on a Thursday", "weekday") is V


def test_weekday_future_wording():
    f = one("1 January 2030 will be a Monday", "weekday")        # 2030-01-01 is a Tuesday
    assert f.status is C and "will be" in f.reason
    assert st("Tuesday, 1 January 2030", "weekday") is V


def test_weekday_thai():
    assert st("วันจันทร์ที่ 15 มีนาคม 2564", "weekday") is V
    assert st("วันศุกร์ที่ 15 มีนาคม 2564", "weekday") is C
    assert st("จันทร์ที่ 15 มีนาคม 2564", "weekday") is V        # without วัน
    assert st("วันที่ 15 มีนาคม 2564 ตรงกับวันจันทร์", "weekday") is V
    assert st("วันที่ 15 มีนาคม 2564 ตรงกับวันอาทิตย์", "weekday") is C
    assert st("วันพุธที่ 1 มกราคม 2563", "weekday") is V          # 1 Jan 2020 was a Wednesday
    assert st("วันพฤหัสบดีที่ 1 มกราคม 2563", "weekday") is C


def test_weekday_ambiguous_numeric():
    f = one("Monday 03/04/2020", "weekday")                     # neither 3 Apr (Fri) nor 4 Mar (Wed)
    assert f.status is C
    assert st("Friday 15/03/2021", "weekday") is C              # unambiguous day-first reading: a Monday


def test_weekday_before_gregorian_is_unverified():
    assert st("Friday, 4 October 1500", "weekday") is U
    assert st("Friday, 15 October 1582", "weekday") is V


def test_invalid_date_beats_weekday():
    f = one("Monday, 30 February 2021", "calendar_date")
    assert f.status is C and f.claim.text == "30 February 2021"


# --------------------------------------------------------------------- spans ---

def test_span_years_english():
    f = one("The war lasted from 1999 to 2010 (11 years).", "span_years")
    assert f.status is V and "exclusive" in f.reason
    assert st("from 1999 to 2010 (12 years)", "span_years") is V          # inclusive counting
    assert st("from 1999 to 2010 (13 years)", "span_years") is C
    assert st("from 1999 to 2010 (10 years)", "span_years") is C
    assert st("between 1999 and 2010 is 11 years", "span_years") is V
    assert st("1999–2010 (11 years)", "span_years") is V
    assert st("from 1999 until 2010, 11 years", "span_years") is V
    assert st("from 1999 to 2010 (about 11 years)", "span_years") is V


def test_span_days_and_dates():
    assert st("between 3 March 2020 and 9 March 2020 is 6 days", "span_days") is V
    assert st("between 3 March 2020 and 9 March 2020 is 7 days", "span_days") is V      # inclusive
    assert st("between 3 March 2020 and 9 March 2020 is 9 days", "span_days") is C
    assert st("from 28 February 2020 to 1 March 2020 (2 days)", "span_days") is V        # leap year: 2 exclusive
    assert st("from 28 February 2021 to 1 March 2021 (3 days)", "span_days") is C        # 1 or 2 only
    assert st("from 3 March 2020 to 3 March 2025 (5 years)", "span_years") is V
    assert st("from 3 March 2020 to 3 March 2025 (9 years)", "span_years") is C


def test_span_thai():
    assert st("ตั้งแต่ปี 2540 ถึง 2550 (10 ปี)", "span_years") is V
    assert st("ตั้งแต่ปี 2540 ถึง 2550 (11 ปี)", "span_years") is V      # inclusive
    assert st("ตั้งแต่ปี 2540 ถึง 2550 (15 ปี)", "span_years") is C
    assert st("ระหว่างปี 2540 และ 2550 รวม 10 ปี", "span_years") is V
    assert st("ตั้งแต่ 3 มีนาคม 2563 ถึง 9 มีนาคม 2563 (6 วัน)", "span_days") is V
    assert st("ตั้งแต่ 3 มีนาคม 2563 ถึง 9 มีนาคม 2563 (20 วัน)", "span_days") is C


def test_span_ambiguous_or_partial_is_silent():
    none("from 1999 to 2010 (over 11 years)")                 # inequality hedge
    none("from 1999 to 2010 (11 years later)")                # not a duration claim
    none("from 3 March to 9 March (6 days)")                  # no years
    none("from 1999 to 3 March 2010 (11 years)")              # mixed endpoints
    none("from 1999 to 2010")                                 # nothing to compare


def test_span_hedge_is_not_proof():
    f = one("from 1999 to 2010 (about 30 years)", "span_years")
    assert f.status is U


def test_span_end_before_start():
    f = one("from 2010 to 1999 (11 years)", "span_years")
    assert f.status is C and "before" in f.reason


# ---------------------------------------------------------------- date_order ---

def test_date_order():
    assert st("from 2010 to 2005", "date_order") is C
    assert st("from 15 June 2021 to 3 June 2021", "date_order") is C
    none("from 2005 to 2010")
    none("between 2010 and 2005")                              # symmetric wording
    assert st("ระหว่างปี 2550 ถึง 2545", "date_order") is C
    assert st("ตั้งแต่ปี 2550 ถึง 2545", "date_order") is C
    none("ระหว่างปี 2545 ถึง 2550")


def test_date_order_exceptions():
    none("counting back from 2010 to 2005")
    none("a countdown from 2010 to 2005")
    none("listed in reverse, from 2010 to 2005")
    none("from 500 BC to 400 BC")


def test_yearless_wrap_is_not_proof():
    f = one("from 15 June to 3 June", "date_order")
    assert f.status is U
    none("The fiscal year runs from 1 October to 30 September.")


# ------------------------------------------------------------------ lifespan ---

def test_lifespan_age():
    assert st("Alan Turing (1912–1954) died at 41.", "lifespan") is V       # birthday unknown: 41 or 42
    assert st("Alan Turing (1912–1954) died at 42.", "lifespan") is V
    f = one("Alan Turing (1912–1954) died at 52.", "lifespan")
    assert f.status is C and "41 or 42" in f.reason
    assert st("Marie (1867-1934) was aged 66 when she died", "lifespan") is V
    assert st("Marie (1867-1934), who died aged 70", "lifespan") is C


def test_lifespan_born_died_forms():
    assert st("Smith (born 1912, died 1900)", "lifespan") is C
    assert st("Smith, born in 1954 and died in 1912, was", "lifespan") is C
    none("Smith (born 1912, died 1954)")                      # fine, no age stated: nothing to check
    none("The war (1954–1912) ended")                         # bare range, not obviously a person


def test_lifespan_implausible_is_unverified():
    f = one("Smith (born 1800, died 1950)", "lifespan")
    assert f.status is U and "implausible" in f.reason
    assert st("Smith (1800–1950) died at 150", "lifespan") is U


def test_lifespan_future_death():
    assert st("Smith (born 1980, died 2090) aged 110", "lifespan") is C


def test_lifespan_thai():
    assert st("นายสมชาย (พ.ศ. 2455–2497) อายุ 42 ปี", "lifespan") is V
    assert st("นายสมชาย (2455-2497) อายุ 52 ปี", "lifespan") is C
    assert st("นายสมชาย (เกิด พ.ศ. 2497 เสียชีวิต พ.ศ. 2455)", "lifespan") is C


def test_lifespan_two_people_is_silent():
    none("Marx (1818–1883) and Engels (1820–1895) died at 64 and 74.")


# ------------------------------------------------------------------- century ---

def test_century_english():
    assert st("in 1750 (18th century)", "century") is V
    assert st("in 1750 (19th century)", "century") is C
    assert st("in the 19th century (1750)", "century") is C
    assert st("in the 18th century (1750)", "century") is V
    assert st("In 2001 (21st century)", "century") is V
    assert st("In 1801 (19th century)", "century") is V
    assert st("In 1800 (18th century)", "century") is V               # 1800 is the last year of the 18th
    assert st("In 1801 (18th century)", "century") is C
    assert st("In 1750 (XVIII century)", "century") is V
    assert st("In 1750 (XIX century)", "century") is C


def test_century_year_boundary_is_convention():
    f = one("In 1900 (20th century)", "century")
    assert f.status is U
    assert st("In 1900 (19th century)", "century") is V
    assert st("In 2100 (19th century)", "century") is C


def test_century_thai_and_buddhist_era():
    assert st("ศตวรรษที่ 19 (พ.ศ. 2300)", "century") is C           # 2300 BE = 1757 CE: 18th (BE 23rd)
    assert st("ศตวรรษที่ 18 (พ.ศ. 2300)", "century") is V
    assert st("ศตวรรษที่ 23 (พ.ศ. 2300)", "century") is V           # Buddhist-Era century
    assert st("พุทธศตวรรษที่ 25 (พ.ศ. 2450)", "century") is V
    assert st("พุทธศตวรรษที่ 18 (พ.ศ. 2450)", "century") is C       # explicitly BE century
    assert st("ปี 2450 (ศตวรรษที่ 20)", "century") is V


def test_century_decades():
    assert st("1857 (1850s)", "century") is V
    assert st("1867 (1850s)", "century") is C
    assert st("In the 1850s (1857)", "century") is V
    assert st("1805 (1800s)", "century") is V
    assert st("1850 (1800s)", "century") is U                       # decade or century?
    assert st("1950 (1800s)", "century") is C


def test_century_needs_adjacency():
    none("In 1750, the 19th century was far away.")
    none("The 19th century began. Later, 1750 came up.")
    none("1750 BC (18th century BC)")


# ---------------------------------------------------------------------- units ---

def test_unit_pairs_both_directions():
    pairs = [("5 km", "3.1 mi"), ("5 mi", "8 km"), ("10 m", "32.8 ft"), ("10 ft", "3.05 m"),
             ("10 cm", "3.94 in"), ("10 in", "25.4 cm"), ("10 mm", "0.39 in"), ("2 in", "50.8 mm"),
             ("10 kg", "22.05 lb"), ("10 lb", "4.54 kg"), ("100 g", "3.53 oz"), ("4 oz", "113 g"),
             ("10 L", "2.64 gal"), ("10 gal", "37.9 L"), ("100 mL", "3.38 fl oz"), ("4 fl oz", "118 mL"),
             ("100 °C", "212 °F"), ("212 °F", "100 °C"), ("100 km/h", "62 mph"), ("60 mph", "97 km/h"),
             ("10 m/s", "36 km/h"), ("36 km/h", "10 m/s"), ("10 ha", "24.7 acres"), ("10 acres", "4.05 ha"),
             ("100 m2", "1076 ft2"), ("1076 ft²", "100 m²")]
    for a, b in pairs:
        f = one(f"It is {a} ({b}) in total.", "unit_conversion")
        assert f.status is V, (a, b, f.reason)
        f = one(f"It is {a} ({b}) in total.", "unit_conversion")
        f2 = one(f"It is {b} ({a}) in total.", "unit_conversion")
        assert f2.status is V, (b, a, f2.reason)


def test_unit_wrong_values_contradicted_with_right_value():
    f = one("5 miles is about 6 km", "unit_conversion")
    assert f.status is C and "8.05" in f.reason and "6 km" in f.reason
    assert st("5 km (4 mi)", "unit_conversion") is C
    assert st("100 °C (200 °F)", "unit_conversion") is C
    assert st("10 kg (10 lb)", "unit_conversion") is C
    assert st("60 mph (60 km/h)", "unit_conversion") is C
    assert st("1 ha (1 acre)", "unit_conversion") is C


def test_unit_forms_and_digits():
    assert st("The route is 5 km, about 3 miles long", "unit_conversion") is V
    assert st("5 km or roughly 3 miles", "unit_conversion") is V
    assert st("5 km ≈ 3.1 mi", "unit_conversion") is V
    assert st("1,609 m (5,279 ft)", "unit_conversion") is V
    assert st("๕ กม. (๓.๑ ไมล์)", "unit_conversion") is V
    assert st("٥ km (٣.١ mi)", "unit_conversion") is V        # Arabic-Indic digits
    assert st("5 กิโลเมตร (3.1 ไมล์)", "unit_conversion") is V
    assert st("5 กิโลเมตร (6 ไมล์)", "unit_conversion") is C
    assert st("−40 °C (−40 °F)", "unit_conversion") is V
    assert st("37℃ (98.6℉)", "unit_conversion") is V
    assert st("30 องศาเซลเซียส (86 องศาฟาเรนไฮต์)", "unit_conversion") is V
    assert st("100 kph (62 mph)", "unit_conversion") is V
    assert st('6" (15.24 cm)', "unit_conversion") is V


def test_unit_ranges_check_both_ends():
    assert st("5-10 km (3-6 mi)", "unit_conversion") is V
    assert st("5-10 km (3-9 mi)", "unit_conversion") is C
    assert st("5 to 10 km (3 to 6 mi)", "unit_conversion") is V
    none("5-10 km (3 mi)")                                      # shape mismatch: not parsed fully


def test_unit_compound_feet_inches():
    assert st("He is 5 ft 9 in (175 cm) tall", "unit_conversion") is V
    assert st("He is 5 ft 9 in (190 cm) tall", "unit_conversion") is C
    assert st("She weighs 7 lb 4 oz (3.3 kg)", "unit_conversion") is V
    none("He is 5 ft 9 in (tall)")


def test_unit_precision_tolerance_edges():
    assert st("100 km (62 mi)", "unit_conversion") is V          # 62.137 rounds to 62
    assert st("100 km (62.14 mi)", "unit_conversion") is V
    assert st("100 km (62.3 mi)", "unit_conversion") is V        # 62.137 vs 62.3: inside rounding + 1 % slack
    assert st("100 km (63.0 mi)", "unit_conversion") is C        # outside it
    assert st("5 km (3.2 mi)", "unit_conversion") is C           # 3.107 -> 3.1 ; 1% slack not enough
    assert st("5 km (3.1 mi)", "unit_conversion") is V
    assert st("26.2 mi (42.2 km)", "unit_conversion") is V
    assert st("26.2 mi (45 km)", "unit_conversion") is C


def test_unit_trailing_zero_rounding_is_unverified():
    f = one("100 km (60 mi)", "unit_conversion")                 # 62.1: fine only if rounded to tens
    assert f.status is U and "nearest 10" in f.reason
    assert st("100 km (50 mi)", "unit_conversion") is C


def test_unit_convention_fallbacks():
    f = one("10 gallons (45.5 L)", "unit_conversion")            # imperial gallon: 45.46 L
    assert f.status is U and "imperial" in f.reason
    assert st("10 US gallons (37.9 L)", "unit_conversion") is V
    assert st("10 imperial gallons (45.5 L)", "unit_conversion") is V
    assert st("10 imperial gallons (37.9 L)", "unit_conversion") is C
    f = one("10 miles (18.52 km)", "unit_conversion")            # nautical mile: 1.852 km
    assert f.status is U and "nautical" in f.reason
    assert st("10 nautical miles (18.52 km)", "unit_conversion") is V
    assert st("10 fl oz (284 mL)", "unit_conversion") is U       # imperial fluid ounce
    assert st("10 fl oz (296 mL)", "unit_conversion") is V


def test_unit_temperature_difference():
    assert st("a rise of 5 °C (9 °F)", "unit_conversion") is V
    assert st("It is 5 °C (9 °F)", "unit_conversion") is U       # only as a difference
    assert st("5 °C (41 °F)", "unit_conversion") is V


def test_unit_unparsed_or_inequality_is_silent():
    none("It is 5 km (3 mi north of town)")
    none("over 5 km (3 mi)")
    none("nearly 5 km (almost 3 mi)")
    none("5 km, 3 mi")                                          # no relation words
    none("3 x 5 m (16 ft)")
    none("5 km (3 min)")                                        # different dimension


# -------------------------------------------------------------------- percent ---

def test_percent_remaining():
    f = one("30% were X and the remaining 60% were Y.", "percent_remaining")
    assert f.status is C and "90%" in f.reason
    assert st("30% were X and the remaining 70% were Y.", "percent_remaining") is V
    assert st("33% voted yes and the remaining 66% no", "percent_remaining") is V     # rounding
    assert st("33.3% voted yes and the rest 66.6% no", "percent_remaining") is V
    assert st("40% voted yes and the remaining 50% no", "percent_remaining") is C
    assert st("ร้อยละ 30 เป็น X และส่วนที่เหลืออีก 60% เป็น Y", "percent_remaining") is C
    assert st("30% เป็น X ส่วนที่เหลืออีก 70% เป็น Y", "percent_remaining") is V


def test_percent_remaining_ambiguous_is_silent():
    none("20% did A, 30% did B and the remaining 10% did C.")   # three-part split: not checked
    none("The remaining 60% were Y.")


def test_out_of_and_of_total():
    assert st("12 out of 10 people", "percent_remaining") is C
    assert st("Only 1,500 out of 1,000 voters", "percent_remaining") is C
    none("3 out of 10 people")
    none("It scored 4.5 out of 4")                                # decimals are not plain counts
    assert st("120% of the total", "percent_remaining") is C
    assert st("They are 130% of all respondents.", "percent_remaining") is C
    none("It was 50% of the total")
    none("sales rose to 150% of all previous records")
    none("an increase of 150% of the total")
    assert st("10 จากทั้งหมด 5 คน", "percent_remaining") is C
    assert st("ร้อยละ 120 ของทั้งหมด", "percent_remaining") is C


# ---------------------------------------------------- offsets, code, robustness ---

def test_offsets_with_thai_combining_marks_emoji_and_crlf():
    pre = "กำเนิดนี้ที่ 🎉 ผู้ที่ถูก\r\nสวัสดี\r\n"
    for body in ("Monday, 30 February 2021", "Friday, 15 March 2021", "วันศุกร์ที่ 15 มีนาคม ๒๕๖๔",
                 "5 miles (6 km)", "from 2010 to 2005", "12 out of 10"):
        text = pre + body + " 👍 จบ\r\n" + pre
        fs = run(text)
        assert len(fs) == 1 and fs[0].claim.start >= len(pre), (body, fs)
        assert text[fs[0].claim.start:fs[0].claim.end] == fs[0].claim.text


def test_thai_digits_arabic_digits_normalised_with_exact_offsets():
    text = "วันศุกร์ที่ ๑๕ มีนาคม ๒๕๖๔ และ ٢٠٢٣-٠٢-٢٩"
    fs = run(text)
    assert [f.status for f in fs] == [C, C]
    assert fs[0].claim.text == "วันศุกร์ที่ ๑๕ มีนาคม ๒๕๖๔"
    assert fs[1].claim.text == "٢٠٢٣-٠٢-٢٩"


def test_ids_and_document_order():
    text = "5 miles (6 km). Then 30 February 2024. Then 12 out of 10. Then Monday, 15 March 2021."
    fs = run(text)
    assert [f.claim.id for f in fs] == ["m1", "m2", "m3", "m4"]
    assert [f.claim.data["type"] for f in fs] == ["unit_conversion", "calendar_date", "percent_remaining", "weekday"]
    assert [f.status for f in fs] == [C, C, C, V]


def test_code_is_ignored():
    none("```python\nd = '30 February 2024'  # 5 miles (6 km)\n```")
    none("Use `30 February 2024` and `12 out of 10` here.")
    none("~~~\n12 out of 10\n~~~")
    none("```\n30 February 2024")                                # unterminated fence runs to the end
    fs = run("```\n30 February 2024\n```\nBut 31 April 2021 is real text.")
    assert len(fs) == 1 and fs[0].claim.text == "31 April 2021"
    # a masked region never joins two halves of a claim
    none("5 km `x` (3 mi)")
    none("Monday, `15 March 2021`")


def test_empty_and_odd_inputs():
    assert scan("") == []
    assert scan("   \n\t ") == []
    assert scan(None) == []                                     # type: ignore[arg-type]
    none("No numbers here at all, just words.")


def test_adversarial_inputs_are_fast_and_safe():
    cases = [
        "99999999999 km (3 mi)",
        "9" * 5000 + " km (3 mi)",
        "5" + " " * 100000 + "km",
        "from " + "1" * 100000 + " to 2",
        "(" * 50000,
        "30 February " * 15000,
        "5 km (3 mi) " * 15000,
        "1,234," * 40000,
        "a" * 200000,
        "Monday, 15 March 2021 " * 8000,
        "12 out of 10, " * 14000,
        "จันทร์ที่ " * 20000,
        "from 1999 to 2010 (11 years) " * 6000,
        "- " * 100000,
        "0" * 200000,
    ]
    for text in cases:
        t0 = time.time()
        fs = scan(text, today=TODAY)
        dt = time.time() - t0
        assert dt < 1.0, (text[:30], dt)
        for f in fs:
            assert text[f.claim.start:f.claim.end] == f.claim.text
    big = "99999999999 km (3 mi)"
    assert st(big, "unit_conversion") is C


def test_huge_numbers_do_not_crash():
    none("1" * 400 + " out of " + "1" * 400)
    fs = run("1" * 300 + " km (3 mi)")
    assert all(f.status is not V for f in fs)
    run("from 99999 to 100000 (1 years)")
    run("29 February 9999 and 0000-01-01 and 9999-12-31")


def test_mixed_document_end_to_end():
    text = ("Marie Curie (1867–1934) died at 66. She was born on Thursday, 7 November 1867 in Warsaw "
            "(about 1,300 km, roughly 808 mi from Paris). Roughly 30% of it was X and the remaining 60% Y. "
            "In 1867 (19th century) it began.")
    fs = run(text)
    got = {(f.claim.data["type"], f.status) for f in fs}
    assert ("lifespan", V) in got
    assert ("weekday", V) in got                    # 7 November 1867 was a Thursday
    assert ("percent_remaining", C) in got
    assert ("century", V) in got


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("ok  ", name)
            except Exception as e:
                import traceback
                fails += 1; print("FAIL", name, repr(e))
    sys.exit(1 if fails else 0)
