"""Tests for the loader, written against data that is wrong but does not error.

Every fixture here is the SAME clean series with exactly one thing changed, so a
detector that fires cannot be firing for another reason. The defects chosen are
the ones that pass silently through a naive loader and then change the answer:

  - a run of o==h==l==c, which is a dead feed written as a flat market;
  - a high below the body, which produces a fill that could not have happened;
  - a hole on a Tuesday, which a backtest reads as two adjacent bars;
  - 03/04/2024, which is April in London and March in New York.

The last one has no detector. It has a REFUSAL, and the test that matters is
that the refusal happens rather than a choice being made quietly.

Run: python3 tests/test_data.py
"""

import io
import os
import sys
import tempfile
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.data import (DUPLICATE_TIMESTAMP, FROZEN_QUOTES, IRREGULAR_INTERVAL,
                         MISSING_BARS, NON_POSITIVE_PRICE, OHLC_VIOLATION,
                         OUT_OF_ORDER, Series, inspect, load_csv)
from models.patterns import Bar

MONDAY = datetime(2024, 1, 1)          # 2024-01-01 is a Monday
_TMP = tempfile.mkdtemp(prefix="fxdata-")


def _rows(n=100, start=MONDAY, step=timedelta(hours=1)):
    """A clean hourly series that never crosses a weekend (100h ends Friday)."""
    out, px = [], 1.1000
    for i in range(n):
        o = px
        c = px + (0.0002 if i % 3 else -0.0002)
        out.append([start + i * step, o, max(o, c) + 0.0001,
                    min(o, c) - 0.0001, c])
        px = c
    return out


def _write(rows, header=True, delim=",", stamp="%Y-%m-%d %H:%M:%S"):
    path = os.path.join(_TMP, f"s{len(os.listdir(_TMP))}.csv")
    lines = []
    if header:
        lines.append(delim.join(("Date", "Open", "High", "Low", "Close")))
    for r in rows:
        t = r[0].strftime(stamp) if isinstance(r[0], datetime) else str(r[0])
        lines.append(delim.join([t] + [f"{v:.5f}" for v in r[1:]]))
    with io.open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def _load(rows, **kw):
    return load_csv(_write(rows, **{k: v for k, v in kw.items()
                                    if k in ("header", "delim", "stamp")}),
                    "EURUSD", side="bid",
                    **{k: v for k, v in kw.items() if k == "time_format"})


# ---------------------------------------------------------------------------
# the control
# ---------------------------------------------------------------------------

def test_a_clean_file_loads_with_nothing_to_report():
    s = _load(_rows())
    assert s.n == 100, s.n
    assert s.problems == [], [str(p) for p in s.problems]
    assert s.usable
    assert s.interval == timedelta(hours=1), s.interval
    assert s.pair == "EURUSD" and s.side == "bid"
    assert s.times[0] == MONDAY and s.times[-1] == MONDAY + timedelta(hours=99)


def test_the_control_is_not_passing_because_nothing_is_checked():
    """If the clean series were empty, or the bars never reached inspect(),
    every isolation test below would pass against a detector that never runs."""
    s = _load(_rows())
    assert s.bars[0].o == 1.1000 and s.bars[0].h > s.bars[0].o
    assert len({b.c for b in s.bars}) > 10, "the fixture is not varying"


def test_headerless_and_alternative_delimiters_load():
    assert _load(_rows(20), header=False).n == 20
    assert _load(_rows(20), delim=";").n == 20
    assert _load(_rows(20), delim="\t").n == 20


# ---------------------------------------------------------------------------
# which side of the book — undetectable, therefore required
# ---------------------------------------------------------------------------

def test_the_side_of_the_book_must_be_declared():
    path = _write(_rows(10))
    for bad in ("BID", "midpoint", "", None):
        try:
            load_csv(path, "EURUSD", side=bad)
        except ValueError as e:
            assert "side must be one of" in str(e), e
        else:
            raise AssertionError(f"side={bad!r} was accepted")
    for ok in ("bid", "ask", "mid"):
        assert load_csv(path, "EURUSD", side=ok).side == ok


# ---------------------------------------------------------------------------
# frozen quotes — the absence of ticks, written as data
# ---------------------------------------------------------------------------

def test_a_run_of_flat_bars_is_reported():
    rows = _rows()
    for i in (10, 11, 12):
        rows[i][1] = rows[i][2] = rows[i][3] = rows[i][4] = 1.1000
    s = _load(rows)
    assert FROZEN_QUOTES in s.codes, s.codes
    assert s.count(FROZEN_QUOTES) == 3, s.count(FROZEN_QUOTES)
    assert s.usable, "3 flat bars in 100 is a note, not a disqualification"


def test_a_pair_of_flat_bars_is_not_a_dead_feed():
    """Isolation in the other direction: the threshold must actually bind, or
    the detector is just 'report every flat bar' wearing a constant."""
    rows = _rows()
    for i in (10, 11):
        rows[i][1] = rows[i][2] = rows[i][3] = rows[i][4] = 1.1000
    assert FROZEN_QUOTES not in _load(rows).codes


def test_a_series_mostly_made_of_dead_feed_is_not_usable():
    rows = _rows()
    for i in range(10, 25):            # 15 of 100, past the 5% limit
        rows[i][1] = rows[i][2] = rows[i][3] = rows[i][4] = 1.1000
    s = _load(rows)
    assert s.count(FROZEN_QUOTES) == 15, s.count(FROZEN_QUOTES)
    assert not s.usable
    assert "NOT USABLE" in s.report()


def test_a_feed_that_dies_at_the_end_of_the_file_is_still_caught():
    """The common case: you downloaded up to "now" and the last bars are stale.
    A loop that only closes a run when a live bar follows never sees it."""
    rows = _rows()
    for i in range(96, 100):
        rows[i][1] = rows[i][2] = rows[i][3] = rows[i][4] = 1.1000
    s = _load(rows)
    assert FROZEN_QUOTES in s.codes, s.codes
    assert s.count(FROZEN_QUOTES) == 4, s.count(FROZEN_QUOTES)


def test_runs_are_counted_separately_not_merged():
    rows = _rows()
    for i in (10, 11, 12, 40, 41, 42):
        rows[i][1] = rows[i][2] = rows[i][3] = rows[i][4] = 1.1000
    assert _load(rows).count(FROZEN_QUOTES) == 6


# ---------------------------------------------------------------------------
# impossible bars
# ---------------------------------------------------------------------------

def test_a_high_below_the_body_is_fatal():
    # the high must sit UNDER the body but still above the low, or the h >= l
    # clause rejects the bar and this test never exercises the one it names.
    rows = _rows()
    rows[42][1:] = [1.1000, 1.1002, 1.0999, 1.1005]        # o, h, l, c
    s = _load(rows)
    assert OHLC_VIOLATION in s.codes, s.codes
    assert not s.usable
    assert [p.first_index for p in s.problems if p.code == OHLC_VIOLATION] == [42]


def test_a_low_above_the_body_is_fatal():
    # likewise: above the body, below the high, so only the low clause fails.
    rows = _rows()
    rows[7][1:] = [1.1000, 1.1008, 1.1002, 1.1005]         # o, h, l, c
    assert OHLC_VIOLATION in _load(rows).codes


def test_a_zero_price_is_fatal():
    rows = _rows()
    rows[5][1] = 0.0
    s = _load(rows)
    assert NON_POSITIVE_PRICE in s.codes, s.codes
    assert not s.usable


# ---------------------------------------------------------------------------
# the clock
# ---------------------------------------------------------------------------

def test_duplicate_timestamps_are_fatal():
    rows = _rows()
    rows[50][0] = rows[49][0]
    s = _load(rows)
    assert DUPLICATE_TIMESTAMP in s.codes, s.codes
    assert not s.usable


def test_timestamps_running_backwards_are_fatal():
    rows = _rows()
    rows[60][0] = rows[60][0] - timedelta(hours=5)
    s = _load(rows)
    assert OUT_OF_ORDER in s.codes, s.codes
    assert not s.usable


def test_a_hole_on_a_weekday_is_reported():
    rows = _rows()
    del rows[30:35]                      # Tue 06:00-10:00 simply absent
    s = _load(rows)
    assert MISSING_BARS in s.codes, s.codes
    assert s.usable, "a hole is a fact about your data, not a broken assumption"


def test_the_weekend_is_not_reported_as_missing_data():
    """52 false alarms a year is how a loader teaches you to ignore it."""
    fri, sun = datetime(2024, 1, 5, 18), datetime(2024, 1, 7, 22)
    rows = _rows(3, start=fri) + _rows(3, start=sun)
    s = _load(rows)
    assert MISSING_BARS not in s.codes, [str(p) for p in s.problems]
    # and the same-sized hole NOT spanning a Saturday is reported
    wed = datetime(2024, 1, 3, 18)
    rows2 = _rows(3, start=wed) + _rows(3, start=wed + timedelta(hours=52))
    assert MISSING_BARS in _load(rows2).codes
    # SATURDAY specifically, not "the weekend": the market is shut all Saturday
    # but only part of Sunday, so a Sunday-evening hole is missing data. A test
    # whose gap spans both days cannot tell the two rules apart.
    sun = datetime(2024, 1, 7, 21)
    rows3 = _rows(3, start=sun) + _rows(3, start=datetime(2024, 1, 8, 5))
    assert MISSING_BARS in _load(rows3).codes, [str(p) for p in _load(rows3).problems]


def test_a_series_that_is_not_fixed_interval_says_so():
    rows = _rows(40)
    for i in range(1, 40, 2):            # most gaps differ from the mode
        rows[i][0] += timedelta(minutes=17 * i)
    codes = _load(rows).codes
    assert IRREGULAR_INTERVAL in codes, codes


# ---------------------------------------------------------------------------
# the timestamp layout that cannot be guessed
# ---------------------------------------------------------------------------

def test_an_ambiguous_date_order_is_refused_rather_than_chosen():
    """03/04/2024 is April in London and March in New York. Picking one shifts
    every bar in the file, with no error and a chart that still looks right."""
    rows = _rows(4, start=datetime(2024, 4, 3))
    try:
        _load(rows, stamp="%d/%m/%Y %H:%M")
    except ValueError as e:
        assert "ambiguous" in str(e) and "time_format=" in str(e), e
    else:
        raise AssertionError("an ambiguous date order was silently resolved")


def test_one_unambiguous_row_settles_the_whole_column():
    # daily bars, so the series reaches the 13th: a day past the twelfth can
    # only be a day, which rules %m/%d out for the whole column at once.
    rows = _rows(15, start=datetime(2024, 4, 3), step=timedelta(days=1))
    s = _load(rows, stamp="%d/%m/%Y %H:%M")
    assert s.times[0] == datetime(2024, 4, 3), s.times[0]
    assert s.times[-1] == datetime(2024, 4, 17), s.times[-1]


def test_an_explicit_format_ends_the_argument():
    rows = _rows(4, start=datetime(2024, 4, 3))
    s = _load(rows, stamp="%d/%m/%Y %H:%M", time_format="%d/%m/%Y %H:%M")
    assert s.times[0] == datetime(2024, 4, 3), s.times[0]
    other = _load(rows, stamp="%d/%m/%Y %H:%M", time_format="%m/%d/%Y %H:%M")
    assert other.times[0] == datetime(2024, 3, 4), other.times[0]


def test_an_unreadable_row_names_the_row():
    path = _write(_rows(5))
    with io.open(path, encoding="utf-8") as fh:
        text = fh.read().splitlines()
    text[3] = "2024-01-01 02:00:00,1.1,notanumber,1.0,1.05"
    with io.open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(text) + "\n")
    try:
        load_csv(path, "EURUSD", side="bid")
    except ValueError as e:
        assert "row 3" in str(e), e
    else:
        raise AssertionError("a bad row loaded without complaint")


# ---------------------------------------------------------------------------
# what it refuses to do
# ---------------------------------------------------------------------------

def test_nothing_is_repaired_or_dropped():
    """A loader that quietly drops bad rows hands back a clean series with an
    unknown relationship to the market."""
    rows = _rows()
    rows[42][2] = 0.5
    rows[43][1] = 0.0
    s = _load(rows)
    assert s.n == 100, f"rows were dropped: {s.n}"
    assert s.bars[42].h == 0.5 and s.bars[43].o == 0.0


def test_the_series_converts_its_own_bars_into_financing_nights():
    """backtest.py counts a hold in BARS and charges swap in NIGHTS. Without
    this conversion hourly data pays 24 times the financing it should, so the
    number comes from the interval the file actually has."""
    assert _load(_rows(50)).bars_per_night() == 24.0
    daily = _rows(20, step=timedelta(days=1))
    assert _load(daily).bars_per_night() == 1.0
    m15 = _rows(20, step=timedelta(minutes=15))
    assert _load(m15).bars_per_night() == 96.0


def test_an_irregular_series_refuses_to_guess_its_own_bar_size():
    s = Series(pair="EURUSD", side="bid", bars=[Bar(1.1, 1.2, 1.0, 1.15)],
               times=[], interval=None)
    try:
        s.bars_per_night()
    except ValueError as e:
        assert "no regular interval" in str(e), e
    else:
        raise AssertionError("an irregular series invented a bar size")


def test_inspect_works_without_any_timestamps():
    bars = [Bar(1.1, 1.2, 1.0, 1.15)] * 5
    problems, interval = inspect(bars, [])
    assert interval is None
    assert FROZEN_QUOTES not in [p.code for p in problems]
    assert inspect([], [])[0] == []


def test_the_report_states_the_pair_the_side_and_the_span():
    text = _load(_rows()).report()
    assert "EURUSD" in text and "BID" in text and "100 bars" in text
    assert "2024-01-01" in text
    assert "not the same as none" in text, "a clean report overclaimed"


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
