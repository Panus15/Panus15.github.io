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
