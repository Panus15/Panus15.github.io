"""Tests for the command-line entry point — mostly about when it REFUSES to run.

The tool's value is not the table it prints. It is that it stops before printing
one when the data cannot support it, and that it will not let you skip saying
what your costs are. Both are easy to weaken by accident and neither shows up as
a crash, so they are pinned here:

  - unusable data must stop with a non-zero exit code, not warn and continue;
  - costs must stay unconfirmed until you say otherwise, on every verdict;
  - the bar-to-night conversion must come from the file, since getting it wrong
    is a 24x error on hourly data and nothing downstream can detect it.

Run: python3 tests/test_analyse.py
"""

import io
import math
import os
import random
import sys
import tempfile
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.costs import CostModel
from engine.holdout import SPLIT_DEFAULT
from models.patterns import PARAMETER_COUNT
from tools.analyse import analyse, main

_TMP = tempfile.mkdtemp(prefix="fxcli-")
COSTS = CostModel(pair="EURUSD", round_turn_pips=1.3, swap_markup_annual=0.012)


def _csv(n=1200, start=datetime(2023, 1, 2), step=timedelta(hours=1), spoil=None,
         drift=None):
    """A clean hourly file, optionally with one row spoiled.

    With ``drift`` it becomes a POSITIVE CONTROL: a series that really does
    trend, so patterns really are profitable in it. A refusal engine that only
    ever refuses is untested in the direction that matters — you cannot tell a
    working filter from a stuck one without watching it pass something.
    """
    path = os.path.join(_TMP, f"s{len(os.listdir(_TMP))}.csv")
    rows, px = [], 1.1000
    rng = random.Random(1)
    for i in range(n):
        o = px
        if drift is None:
            c = px + (0.0003 if i % 3 else -0.0002)
            h, l = max(o, c) + 0.0002, min(o, c) - 0.0002
        else:
            path_px = [o]
            for _ in range(4):
                px *= math.exp(rng.gauss(drift / 4, 0.0004))
                path_px.append(px)
            c, h, l = px, max(path_px), min(path_px)
        rows.append([start + i * step, o, h, l, c])
        px = c
    if spoil:
        spoil(rows)
    with io.open(path, "w", encoding="utf-8") as fh:
        fh.write("Date,Open,High,Low,Close\n")
        for r in rows:
            fh.write(f"{r[0]:%Y-%m-%d %H:%M:%S}," +
                     ",".join(f"{v:.5f}" for v in r[1:]) + "\n")
    return path


def _run(path, **kw):
    lines = []
    code = analyse(path, "EURUSD", kw.pop("side", "bid"), kw.pop("costs", COSTS),
                   out=lines.append, **kw)
    return code, "\n".join(lines)


# ---------------------------------------------------------------------------
# it runs
# ---------------------------------------------------------------------------

def test_a_clean_file_produces_a_verdict_for_every_pattern():
    code, text = _run(_csv(), costs_measured=True)
    assert code == 0, text
    for name in ("head_and_shoulders", "double_top", "double_bottom",
                 "triangle", "flag", "engulfing"):
        assert name in text, f"{name} missing from the table"
    assert "8,734" not in text     # guard against a stale copy-paste fixture


def test_it_says_which_pair_side_and_span_it_read():
    _, text = _run(_csv(300), costs_measured=True)
    assert "EURUSD" in text and "BID" in text and "2023-01-02" in text


def _closing(text):
    survived = "survived one IN-SAMPLE test" in text
    nothing = "Nothing survived" in text
    assert survived != nothing, "both or neither closing message was printed"
    return survived


def test_the_closing_message_matches_what_the_table_said():
    """Checked in BOTH directions. A fixture where nothing passes cannot tell
    'always say nothing survived' apart from working, and one where something
    passes cannot tell the message apart from its own inverse."""
    # drift=0 rather than the default fixture: that one steps +3,+3,-2 pips on
    # a cycle, which is a strong uptrend, and trend patterns pass in it fairly.
    _, none = _run(_csv(3000, drift=0.0), costs_measured=True)
    assert "TRADEABLE" not in none, none
    assert _closing(none) is False

    _, some = _run(_csv(3000, drift=0.0006), costs_measured=True)
    assert "TRADEABLE" in some, some
    assert _closing(some) is True
    assert "not a finding" in some, "a pass was reported without its caveat"


def test_a_pattern_that_passes_gets_its_full_verdict_printed():
    _, text = _run(_csv(3000, drift=0.0006), costs_measured=True)
    assert "needs a win rate of" in text and "t-stat" in text
    assert "evidence:" in text
    assert "risk at most" in text, "a tradeable verdict carried no size"


# ---------------------------------------------------------------------------
# it refuses
# ---------------------------------------------------------------------------

def test_unusable_data_stops_before_any_verdict():
    """Warning and continuing would produce a table of arithmetic on data that
    is not what it claims to be — which is worse than no table."""
    def break_a_bar(rows):
        rows[500][1:] = [1.1000, 1.1002, 1.0999, 1.1005]   # high under the body
    code, text = _run(_csv(spoil=break_a_bar), costs_measured=True)
    assert code == 2, text
    assert "STOPPED" in text and "OHLC_VIOLATION" in text
    assert "pips/trade" not in text, "a verdict table was printed anyway"


def test_a_dead_feed_also_stops_it():
    def freeze(rows):
        for i in range(100, 400):        # well past the 5% limit
            rows[i][1:] = [1.1000] * 4
    code, text = _run(_csv(spoil=freeze), costs_measured=True)
    assert code == 2, text
    assert "FROZEN_QUOTES" in text


def test_a_missing_file_is_an_error_not_a_crash():
    assert main(["/nonexistent/nope.csv", "--pair", "EURUSD",
                 "--side", "bid"]) == 1


def test_the_side_of_the_book_cannot_be_omitted_or_invented():
    for argv in ([_csv(50), "--pair", "EURUSD"],
                 [_csv(50), "--pair", "EURUSD", "--side", "middle"]):
        try:
            main(argv)
        except SystemExit as e:
            assert e.code == 2, e.code
        else:
            raise AssertionError(f"{argv} was accepted")


# ---------------------------------------------------------------------------
# costs stay unconfirmed until you say so
# ---------------------------------------------------------------------------

def test_costs_are_unconfirmed_by_default_on_every_verdict():
    default = CostModel(pair="EURUSD")
    _, text = _run(_csv(), costs=default)
    assert "NOT confirmed" in text, text
    assert text.count("COST_NOT_MEASURED") >= 1, text


def test_confirming_the_costs_removes_it():
    default = CostModel(pair="EURUSD")
    _, text = _run(_csv(), costs=default, costs_measured=True)
    assert "COST_NOT_MEASURED" not in text, text
    assert "NOT confirmed" not in text


# ---------------------------------------------------------------------------
# the conversion nothing downstream can check
# ---------------------------------------------------------------------------

def test_the_bar_size_comes_from_the_file():
    _, hourly = _run(_csv(300), costs_measured=True)
    assert "24 bars per rollover" in hourly, hourly
    _, daily = _run(_csv(300, step=timedelta(days=1)), costs_measured=True)
    assert "1 bars per rollover" in daily, daily


def _pips(text, pattern):
    for line in text.splitlines():
        if line.startswith(pattern):
            return float(line.split()[-2])
    raise AssertionError(f"{pattern} not in table")


def test_the_bar_size_actually_reaches_the_backtest():
    """Identical bars, identical prices, only the timestamps differ — so any
    difference in the result is financing, and no difference means the
    conversion was computed, printed, and then dropped on the floor."""
    _, hourly = _run(_csv(1200), costs_measured=True)
    _, daily = _run(_csv(1200, step=timedelta(days=1)), costs_measured=True)
    h, d = _pips(hourly, "engulfing"), _pips(daily, "engulfing")
    assert h > d, (h, d)      # 24x less swap per bar held


def test_a_file_with_no_measurable_interval_says_what_it_assumed():
    code, text = _run(_csv(1), costs_measured=True)
    assert code == 0, text
    assert "assuming 1 bar per night" in text, text


def test_a_short_history_is_called_out():
    _, short = _run(_csv(120), costs_measured=True)
    assert "very short history" in short, short
    _, long_ = _run(_csv(1200), costs_measured=True)
    assert "very short history" not in long_


# ---------------------------------------------------------------------------
# the search width the correction is told about
# ---------------------------------------------------------------------------

def test_restricting_to_one_pattern_narrows_the_table():
    _, text = _run(_csv(), costs_measured=True, patterns=["double_top"])
    assert "double_top" in text
    assert "engulfing" not in text and "triangle" not in text


def test_the_correction_denominator_is_stated_and_scales_with_the_search():
    """The most fudged number in the field, so it is printed rather than kept
    to itself: six patterns is a wider search than one and must say so."""
    _, all_ = _run(_csv(300), costs_measured=True)
    assert "24 hypotheses" in all_, all_
    _, one = _run(_csv(300), costs_measured=True, patterns=["flag"])
    assert "4 hypotheses" in one, one
    assert "FLOOR" in all_ and "tried and dropped" in all_


# ---------------------------------------------------------------------------
# what to do about a refusal
# ---------------------------------------------------------------------------

def test_an_arithmetic_refusal_comes_with_what_you_could_change():
    _, text = _run(_csv(3000, drift=0.0), costs_measured=True)
    assert "CLOSEST TO WORKING" in text, text
    assert "[CAN]" in text or "[cannot]" in text, text
    assert "VERDICT:" in text


def test_a_sample_size_refusal_is_not_answered_with_broker_advice():
    """Running the levers on a rule refused for TOO_FEW_TRADES prints "it
    already clears its costs" directly under a table saying it was refused.
    That reads as a contradiction because it is one: the answer to too little
    data is more data, and no account change affects it."""
    # restricted to the structural patterns, which fire rarely: on a short file
    # they run out of trades before they run out of edge. engulfing fires often
    # enough to lose to costs on any series, so it always supplies an
    # arithmetic refusal and would mask this branch entirely.
    _, text = _run(_csv(800, drift=0.0), costs_measured=True,
                   patterns=["head_and_shoulders", "triangle"])
    rows = [l for l in text.splitlines()
            if l.split() and l.split()[0] in ("head_and_shoulders", "triangle")]
    codes = [r.split()[-1] for r in rows]
    assert len(codes) == 2 and all(c == "TOO_FEW_TRADES" for c in codes), codes
    assert "CLOSEST TO WORKING" not in text, text
    assert "binding constraint is data" in text, text


def test_no_advice_is_offered_when_something_passed():
    """drift=0.0001 is the fixture that can tell the difference: two patterns
    pass AND engulfing is refused on arithmetic with 724 trades. At a stronger
    drift nothing has an arithmetic refusal left, so "show the advice whether
    or not something passed" would have nothing to show and look correct."""
    _, text = _run(_csv(3000, drift=0.0001), costs_measured=True)
    rows = [l.split() for l in text.splitlines()
            if l.split() and l.split()[0] in
            ("double_bottom", "flag", "engulfing")]
    assert any(r[-1] == "TRADEABLE" for r in rows), rows
    assert any(r[-1] == "NEGATIVE_NET_EXPECTANCY" for r in rows), rows
    assert "CLOSEST TO WORKING" not in text, text
    assert "binding constraint is data" not in text, text


def test_the_trade_rate_is_annualised_from_the_span_not_the_raw_count():
    """"at N trades a year" has to mean a year. The sample here is 3,000 hourly
    bars = 125 days, so the annual rate is the sample count x 365/125 = 2.92x,
    and printing the raw count would understate the yearly bleed threefold."""
    _, text = _run(_csv(3000, drift=0.0), costs_measured=True)
    name = [l for l in text.splitlines() if l.startswith("CLOSEST TO WORKING")]
    assert name, text
    pattern = name[0].split("— ")[1].split(",")[0]
    row = [l.split() for l in text.splitlines()
           if l.split() and l.split()[0] == pattern][0]
    n = int(row[1].replace(",", ""))
    rate = [l for l in text.splitlines() if "trades a year" in l][0]
    printed = int(rate.split("at ")[1].split(" trades")[0].replace(",", ""))
    assert printed != n, f"the raw sample count was printed as a yearly rate ({n})"
    assert abs(printed - n * 365.0 / (3000 / 24.0)) <= 1, (printed, n)


# ---------------------------------------------------------------------------
# a rule you wrote
# ---------------------------------------------------------------------------

_RULE_PREAMBLE = """import sys
sys.path.insert(0, %r)
from models.patterns import Detection
""" % os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _rule(body, name=None):
    path = os.path.join(_TMP, name or f"rule{len(os.listdir(_TMP))}.py")
    io.open(path, "w", encoding="utf-8").write(_RULE_PREAMBLE + body)
    return path


_UP2 = """
PARAMETER_COUNT = 2

def detect(bars):
    out = []
    for i in range(20, len(bars)):
        if bars[i].c > bars[i - 1].c and bars[i - 1].c > bars[i - 2].c:
            e = bars[i].c
            out.append(Detection("up2", i, 1, e, e - 0.0030, e + 0.0060))
    return out
"""

_PEEKS = """
PARAMETER_COUNT = 1

def detect(bars):
    out = []
    for i in range(1, len(bars) - 1):
        if bars[i + 1].c > bars[i].c:
            e = bars[i].c
            out.append(Detection("cheat", i, 1, e, e - 0.0030, e + 0.0060))
    return out
"""


def test_a_rule_you_wrote_runs_instead_of_the_built_ins():
    code, text = _run(_csv(1200, drift=0.0), costs_measured=True,
                      rule_paths=[_rule(_UP2, "up2.py")])
    assert code == 0, text
    assert "up2" in text, text
    assert "engulfing" not in text, "the built-ins ran as well, unasked"
    assert "no look-ahead found" in text


def test_the_correction_sums_what_each_rule_declares():
    """Not patterns x 4. A rule declaring 2 contributes 2, and the denominator
    has to follow, or the significance test is measuring the wrong search."""
    _, one = _run(_csv(1200, drift=0.0), costs_measured=True,
                  rule_paths=[_rule(_UP2)])
    assert "2 hypotheses across 1 rule(s)" in one, one
    _, both = _run(_csv(1200, drift=0.0), costs_measured=True,
                   rule_paths=[_rule(_UP2)], patterns=["flag"])
    assert f"{2 + PARAMETER_COUNT} hypotheses across 2 rule(s)" in both, both


def test_a_peeking_rule_stops_the_run_before_any_number_is_printed():
    path = _rule(_PEEKS, "cheat.py")
    assert main([_csv(1200), "--pair", "EURUSD", "--side", "bid",
                 "--costs-measured", "--rule", path]) == 1


def test_a_rule_can_be_taken_through_the_holdout_too():
    """A hold-out that can only validate the six patterns this engine shipped
    validates the wrong thing."""
    code, text = _run(_csv(3000, drift=0.0006), costs_measured=True,
                      rule_paths=[_rule(_UP2, "up2_ho.py")], holdout=0.7)
    assert code == 0, text
    assert "HOLD-OUT" in text, text
    # the name comes from the FILE, so it has to be pinned or this asserts
    # nothing about which rule was actually split
    assert "up2_ho" in text, text
    assert "engulfing" not in text
    assert "out-of-sample" in text, "the rule never reached the second half"


def test_a_rule_that_passes_is_not_labelled_never_tested_in_fx():
    code, text = _run(_csv(3000, drift=0.0006), costs_measured=True,
                      rule_paths=[_rule(_UP2)])
    assert code == 0, text
    if "TRADEABLE" in text:
        assert "no published evidence" in text, text
        assert "never tested in FX" not in text, text


# ---------------------------------------------------------------------------
# the HTML report
# ---------------------------------------------------------------------------

def test_the_report_is_written_only_when_asked():
    path = os.path.join(_TMP, "r1.html")
    _run(_csv(1200), costs_measured=True)
    assert not os.path.exists(path)
    code, text = _run(_csv(1200), costs_measured=True, report_path=path)
    assert code == 0, text
    assert os.path.exists(path), text
    assert "report written to" in text and path in text


def test_the_written_report_is_a_whole_page_with_charts():
    path = os.path.join(_TMP, "r2.html")
    _run(_csv(1200), costs_measured=True, report_path=path)
    page = io.open(path, encoding="utf-8").read()
    assert page.startswith("<!doctype html>"), page[:40]
    assert page.rstrip().endswith("</html>"), page[-40:]
    assert page.count("<polyline") >= 2, "no equity curve was drawn"
    assert "<script" not in page.lower()


def test_the_report_carries_the_same_costs_the_run_used():
    path = os.path.join(_TMP, "r3.html")
    _run(_csv(1200), costs=CostModel(pair="EURUSD", round_turn_pips=2.4,
                                     swap_markup_annual=0.021),
         costs_measured=True, report_path=path)
    page = io.open(path, encoding="utf-8").read()
    assert "2.4 pip round turn" in page, "the report invented its own costs"
    assert "2.10%/yr" in page


# ---------------------------------------------------------------------------
# the hold-out path
# ---------------------------------------------------------------------------

def test_the_holdout_replaces_the_table_with_a_split_test():
    _, text = _run(_csv(3000, drift=0.0006), costs_measured=True, holdout=0.7)
    assert "HOLD-OUT" in text, text
    assert "in-sample" in text and "out-of-sample" in text
    assert "pips/trade  verdict" not in text, "the in-sample table ran as well"


def test_the_holdout_fraction_reaches_the_split():
    _, seventy = _run(_csv(3000, drift=0.0006), costs_measured=True, holdout=0.7)
    _, fifty = _run(_csv(3000, drift=0.0006), costs_measured=True, holdout=0.5)
    assert "found on 2,100 bars, tested on 900" in seventy, seventy
    assert "found on 1,500 bars, tested on 1,500" in fifty, fifty


def test_the_bare_flag_uses_the_declared_default():
    _, text = _run(_csv(3000, drift=0.0006), costs_measured=True,
                   holdout=SPLIT_DEFAULT)
    assert f"{1 - SPLIT_DEFAULT:.0%} held back" in text, text


def test_the_holdout_still_refuses_to_run_on_broken_data():
    """The split is not a way around the data check."""
    def freeze(rows):
        for i in range(100, 400):
            rows[i][1:] = [1.1000] * 4
    code, text = _run(_csv(spoil=freeze), costs_measured=True, holdout=0.7)
    assert code == 2, text
    assert "HOLD-OUT" not in text


def test_an_unspent_holdout_is_not_warned_about():
    """Nothing passed in-sample means the held-back half was never touched, and
    telling someone they have spent something they still have is a lie."""
    _, text = _run(_csv(3000, drift=0.0), costs_measured=True, holdout=0.7)
    assert "still untouched" in text, text
    assert "YOU GET ONE" not in text


# ---------------------------------------------------------------------------
# Charging the spread the file recorded, rather than one flat guess.
#
# A file with a Spread column lets the run price each trade on what its own
# bars cost. All three of these survived the suite when the wiring was added:
# ignoring the column, ignoring --flat-cost, and dropping the commission.
# ---------------------------------------------------------------------------

def _csv_with_spread(n=1200, points=3, wide_every=None):
    """The clean file plus a Spread column, in POINTS as MetaTrader writes it.

    At five decimals a point is 0.00001 and a pip is 0.0001, so `points=3` is
    0.3 pips -- well under the 1.5-pip flat round turn in COSTS, which is what
    makes the two paths tell apart.
    """
    path = os.path.join(_TMP, f"sp{len(os.listdir(_TMP))}.csv")
    px, rows = 1.1000, []
    start = datetime(2023, 1, 2)
    for i in range(n):
        o = px
        c = px + (0.0003 if i % 3 else -0.0002)
        rows.append([start + i * timedelta(hours=1), o,
                     max(o, c) + 0.0002, min(o, c) - 0.0002, c,
                     points * 100 if (wide_every and i % wide_every == 0)
                     else points])
        px = c
    with io.open(path, "w", encoding="utf-8") as fh:
        fh.write("Date,Open,High,Low,Close,Spread\n")
        for r in rows:
            fh.write(f"{r[0]:%Y-%m-%d %H:%M:%S}," +
                     ",".join(f"{v:.5f}" for v in r[1:5]) + f",{r[5]}\n")
    return path


def test_a_recorded_spread_column_is_charged_per_trade():
    code, text = _run(_csv_with_spread(), costs_measured=True)
    assert code == 0, text
    assert "charging:" in text, (
        f"the run did not say it was charging the recorded spread:\n{text}")
    assert "each trade's own bars recorded" in text, text


def test_flat_cost_ignores_the_recorded_spread_column():
    # The escape hatch has to actually switch the path off, or a reader who
    # asks for the old behaviour silently gets the new one.
    code, text = _run(_csv_with_spread(), costs_measured=True,
                      use_recorded_spread=False)
    assert code == 0, text
    assert "charging:" not in text, (
        f"--flat-cost did not switch off the measured path:\n{text}")


def test_a_file_with_no_spread_column_charges_flat_without_being_asked():
    code, text = _run(_csv(), costs_measured=True)
    assert code == 0, text
    assert "charging:" not in text, (
        f"a file with no spread column reported charging a measured one:\n{text}")


def test_a_zero_commission_is_called_out_as_an_under_estimate():
    # Charging a measured 0.3-pip spread with no commission models a broker
    # that works for nothing. The run must say so rather than quietly produce
    # the most flattering cost in the file.
    code, text = _run(_csv_with_spread(), costs_measured=True)
    assert "commission is ZERO" in text, (
        f"a zero commission passed without comment:\n{text}")

    priced = CostModel(pair="EURUSD", round_turn_pips=1.5,
                       swap_markup_annual=0.012, commission_pips=0.7)
    code, text2 = _run(_csv_with_spread(), costs=priced, costs_measured=True)
    assert "commission is ZERO" not in text2, (
        "a declared commission was still called zero")
    assert "0.70 pip commission" in text2, text2


def test_the_commission_flag_reaches_the_cost_model():
    # The three tests above call `analyse()` directly, so none of them touches
    # the argv plumbing -- `--commission` could be parsed and thrown away and
    # they would all still pass. This goes through main() for that reason.
    import contextlib
    from tools.analyse import main
    path = _csv_with_spread()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = main([path, "--pair", "EURUSD", "--side", "bid",
                     "--round-turn", "1.5", "--swap-markup", "0.012",
                     "--commission", "0.7", "--costs-measured"])
    text = buf.getvalue()
    assert code == 0, text
    assert "0.70 pip commission" in text, (
        f"--commission 0.7 did not reach the cost model:\n{text}")
    assert "commission is ZERO" not in text, text


def _run_all():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
