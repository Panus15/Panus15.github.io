"""Tests for the report — written against the ways a CHART lies.

A chart is harder to test than a number and easier to be wrong in public. The
failures pinned here are the ones that produce a page which looks fine:

  - the net line drawn ABOVE the gross line, which inverts the entire argument
    and still renders beautifully;
  - a y-axis fitted to the data, so a curve that never got above water is
    framed as though it did;
  - a non-finite coordinate, which makes an SVG polyline vanish with no error
    anywhere, reading as "no trades";
  - patterns ordered by return, which turns six hypotheses into one winner;
  - unescaped text from the caller.

Every SVG is parsed as XML rather than grepped, because a malformed tag is the
one defect a substring check cannot see.

Run: python3 tests/test_report.py
"""

import io
import math
import os
import re
import sys
from datetime import datetime, timedelta
from xml.etree import ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.backtest import BacktestResult, Trade
from engine.costs import CostModel
from engine.data import Series
from engine.verdict import judge
from models.patterns import Bar
from tools.report import (H, PAD_B, PAD_T, SHAPE_MEANINGLESS_BELOW, W,
                          _points, equity_svg, render, winrate_svg)

COSTS = CostModel(pair="EURUSD", round_turn_pips=1.3, swap_markup_annual=0.012)


def _trade(gross, cost=1.5, name="double_top", i=0):
    return Trade(name=name, entry_index=i, exit_index=i + 1, direction=1,
                 entry=1.1000, exit=1.1000 + gross * 0.0001, pips=gross,
                 cost_pips=cost, outcome="target" if gross > 0 else "stop")


def _result(wins, win_pips, losses, loss_pips, *, cost=1.5, name="double_top"):
    res = BacktestResult()
    for k in range(wins):
        res.trades.append(_trade(win_pips, cost, name, k))
    for k in range(losses):
        res.trades.append(_trade(-loss_pips, cost, name, wins + k))
    return res


def _series(n=5000):
    bars, times, px = [], [], 1.1000
    t = datetime(2023, 1, 2)
    for i in range(n):
        o = px
        c = px + (0.0003 if i % 3 else -0.0002)
        bars.append(Bar(o, max(o, c) + 0.0002, min(o, c) - 0.0002, c))
        times.append(t + i * timedelta(hours=1))
        px = c
    return Series(pair="EURUSD", side="bid", bars=bars, times=times,
                  interval=timedelta(hours=1), source="fixture.csv")


def _ys(svg, cls):
    """The y-coordinates of a named polyline, in draw order."""
    m = re.search(rf'<polyline class="{cls}" points="([^"]*)"', svg)
    assert m, f"no polyline with class {cls!r}"
    return [float(p.split(",")[1]) for p in m.group(1).split() if p]


# ---------------------------------------------------------------------------
# the argument, drawn the right way round
# ---------------------------------------------------------------------------

def test_the_net_line_is_never_drawn_above_the_gross_line():
    """Costs are positive, so cumulative net can never exceed cumulative gross.
    Swap the two series and the page still renders perfectly while asserting the
    opposite of the truth — the one error a chart hides best."""
    svg = equity_svg(_result(30, 20.0, 20, 10.0).trades, "x")
    net, gross = _ys(svg, "net"), _ys(svg, "gross")
    assert len(net) == len(gross) == 50
    # SVG y grows downward, so "net below gross in value" is "net y >= gross y"
    assert all(n >= g - 1e-9 for n, g in zip(net, gross)), (net[:4], gross[:4])
    assert net[-1] > gross[-1], "the gap should be widest at the end"


def test_the_gap_between_the_lines_is_the_cost():
    """Two identical runs, one free. With no cost the lines must coincide."""
    free = equity_svg(_result(30, 20.0, 20, 10.0, cost=0.0).trades, "x")
    assert _ys(free, "net") == _ys(free, "gross")
    paid = equity_svg(_result(30, 20.0, 20, 10.0, cost=4.0).trades, "x")
    assert _ys(paid, "net") != _ys(paid, "gross")


def test_zero_is_always_on_the_axis():
    """A rule that never got above water must not be framed as though it did:
    an all-losing series still shows the zero line inside the frame."""
    svg = equity_svg(_result(0, 0.0, 40, 10.0).trades, "x")
    m = re.search(r'<line class="axis"[^>]*y1="([\d.]+)"', svg)
    assert m, svg[:200]
    zero_y = float(m.group(1))
    # INSIDE the drawing frame. The first version of this assertion ended in
    # "or zero_y < min(ys)", which is satisfied by zero falling off the top of
    # the chart -- i.e. by the exact failure it was meant to catch.
    assert PAD_T <= zero_y <= H - PAD_B, (zero_y, PAD_T, H - PAD_B)
    assert ">0<" in svg, "the zero tick label is missing"
    # and with a winning series, zero stays in frame from the other side
    up = equity_svg(_result(40, 20.0, 0, 0.0).trades, "x")
    zy = float(re.search(r'<line class="axis"[^>]*y1="([\d.]+)"', up).group(1))
    assert PAD_T <= zy <= H - PAD_B, zy

    # The case that actually needs zero pinned. A long cumulative series starts
    # near zero anyway, so the 8% padding reaches it whatever the scale does --
    # which hid this for a whole sweep. Two large same-signed trades do not:
    # the series runs 98.5 to 197, and a range fitted to the data leaves the
    # baseline off the bottom of the chart entirely.
    far = equity_svg(_result(2, 100.0, 0, 0.0).trades, "x")
    fy = float(re.search(r'<line class="axis"[^>]*y1="([\d.]+)"', far).group(1))
    assert PAD_T <= fy <= H - PAD_B, (
        fy, "zero fell outside the frame: the axis was fitted to the data")


# ---------------------------------------------------------------------------
# a blank chart must never be mistaken for an empty one
# ---------------------------------------------------------------------------

def test_a_non_finite_point_is_refused_by_name():
    """One nan makes an SVG polyline disappear with no error in any renderer,
    which reads exactly like having no trades."""
    for bad in (float("nan"), float("inf"), -float("inf")):
        try:
            _points([0.0, bad, 1.0], -1.0, 2.0)
        except ValueError as e:
            assert "non-finite" in str(e), e
        else:
            raise AssertionError(f"{bad!r} was written into path data")


def test_a_degenerate_range_is_refused_rather_than_divided_by():
    for lo, hi in ((1.0, 1.0), (2.0, 1.0), (float("nan"), 1.0)):
        try:
            _points([0.0, 1.0], lo, hi)
        except ValueError as e:
            assert "y-range" in str(e), e
        else:
            raise AssertionError(f"range [{lo}, {hi}] was accepted")


def test_no_trades_says_so_instead_of_drawing_an_empty_frame():
    out = equity_svg([], "x")
    assert "<svg" not in out, out
    assert "no trades" in out


def test_one_trade_cannot_make_a_line():
    assert _points([5.0], 0.0, 10.0) == ""


# ---------------------------------------------------------------------------
# a small sample must not look like a big one
# ---------------------------------------------------------------------------

def test_a_tiny_sample_is_labelled_over_the_top_of_the_line():
    """Beside the chart is not enough: the eye believes a line long after the
    reader has read the warning."""
    svg = equity_svg(_result(8, 20.0, 4, 10.0).trades, "x")
    assert "means nothing" in svg, svg[-300:]
    assert "12 trades" in svg


def test_a_large_sample_is_not_labelled():
    svg = equity_svg(_result(30, 20.0, 20, 10.0).trades, "x")
    assert "means nothing" not in svg
    assert len(_ys(svg, "net")) >= SHAPE_MEANINGLESS_BELOW


# ---------------------------------------------------------------------------
# the win-rate bars
# ---------------------------------------------------------------------------

def test_the_shortfall_is_the_picture_not_a_footnote():
    losing = judge(_result(36, 10.0, 24, 20.0), COSTS, costs_confirmed=True)
    svg = winrate_svg(losing)
    assert 'class="bar need short"' in svg, svg
    assert "+" in svg, "the gap was not signed"
    winning = judge(_result(36, 20.0, 24, 10.0), COSTS, costs_confirmed=True)
    assert 'class="bar need short"' not in winrate_svg(winning)
    # The measured bar is the GROSS win rate. A fixture where gross and net
    # agree cannot tell which one was plotted: here 40 winners of +1.0 pip are
    # all eaten by a 1.5 pip cost, so gross is 66.7% and net is 0.0%.
    flipped = judge(_result(40, 1.0, 20, 30.0), COSTS, costs_confirmed=True)
    assert abs(flipped.gross.win_rate - 2 / 3) < 1e-9
    assert flipped.net.win_rate == 0.0
    assert "66.7%" in winrate_svg(flipped), winrate_svg(flipped)
    assert "0.0%" not in winrate_svg(flipped)


# ---------------------------------------------------------------------------
# the page
# ---------------------------------------------------------------------------

def _page(names=("double_top", "flag"), **kw):
    results = {n: _result(36, 20.0, 24, 10.0, name=n) for n in names}
    judged = {n: judge(r, COSTS, name=n, n_hypotheses=24, costs_confirmed=True)
              for n, r in results.items()}
    return render(_series(), results, judged, COSTS, 24.0, 24, **kw)


def test_every_svg_on_the_page_parses_as_xml():
    """Parsed, not grepped: a malformed tag is the one defect a substring check
    cannot see, and it is what blanked the options dashboard."""
    page = _page()
    svgs = re.findall(r"<svg\b.*?</svg>", page, re.S)
    assert len(svgs) == 4, len(svgs)        # two patterns x (equity + win rate)
    for s in svgs:
        ET.fromstring(s)                    # raises on malformed markup


def test_the_page_carries_no_javascript_at_all():
    """Enforced, not merely intended. The whole reason the SVG is built in
    Python is that a template-escaping slip in generated JS blanked a page while
    every test passed."""
    page = _page().lower()
    for token in ("<script", "javascript:", "onload=", "onclick=", "onerror="):
        assert token not in page, token


def test_patterns_are_in_detector_order_not_sorted_by_outcome():
    """Sorting by return is how a report turns six hypotheses into one winner."""
    results = {"double_top": _result(5, 1.0, 40, 30.0, name="double_top"),
               "flag": _result(40, 30.0, 5, 1.0, name="flag")}
    judged = {n: judge(r, COSTS, name=n, n_hypotheses=24, costs_confirmed=True)
              for n, r in results.items()}
    page = render(_series(), results, judged, COSTS, 24.0, 24)
    # the <h3> sections, not the table: the table is built by a separate loop,
    # so checking the first occurrence of each name only ever tested the table.
    heads = re.findall(r"<h3>([a-z_]+)</h3>", page)
    assert heads == ["double_top", "flag"], heads
    rows = re.findall(r"<tr><td>([a-z_]+)</td>", page)
    assert rows == ["double_top", "flag"], rows


def test_a_mismatch_between_results_and_verdicts_is_refused():
    results = {"flag": _result(30, 20.0, 20, 10.0, name="flag")}
    judged = {n: judge(r, COSTS, name=n, costs_confirmed=True)
              for n, r in results.items()}
    judged["triangle"] = judged["flag"]
    try:
        render(_series(), results, judged, COSTS, 24.0, 24)
    except ValueError as e:
        assert "disagree about which patterns" in str(e), e
    else:
        raise AssertionError("a verdict with no result rendered anyway")


def test_text_from_the_caller_is_escaped():
    results = {"flag": _result(30, 20.0, 20, 10.0, name="flag")}
    judged = {n: judge(r, COSTS, name=n, costs_confirmed=True)
              for n, r in results.items()}
    s = _series()
    s.pair = '<img src=x onerror="boom">'
    s.source = "</title><script>evil()</script>"
    page = render(s, results, judged, COSTS, 24.0, 24)
    assert "<img" not in page and "<script" not in page, page[:400]
    assert "&lt;img" in page


def test_the_table_figures_match_the_verdicts_they_came_from():
    results = {"flag": _result(36, 20.0, 24, 10.0, name="flag")}
    judged = {n: judge(r, COSTS, name=n, n_hypotheses=24, costs_confirmed=True)
              for n, r in results.items()}
    v = judged["flag"]
    page = render(_series(), results, judged, COSTS, 24.0, 24)
    assert f"{v.n_trades:,}" in page
    assert f"{v.gross.win_rate:.1%}" in page
    assert f"{v.win_rate_needed:.1%}" in page
    assert f"{v.net.net:+.2f}" in page


def test_a_refused_verdict_is_not_coloured_as_a_pass():
    results = {"double_top": _result(36, 10.0, 24, 20.0, name="double_top"),
               "flag": _result(36, 20.0, 24, 10.0, name="flag")}
    judged = {n: judge(r, COSTS, name=n, n_hypotheses=4, costs_confirmed=True)
              for n, r in results.items()}
    assert not judged["double_top"].tradeable and judged["flag"].tradeable
    page = render(_series(), results, judged, COSTS, 24.0, 24)
    assert '<td class="fail">NEGATIVE_NET_EXPECTANCY</td>' in page, page
    assert '<td class="pass">TRADEABLE</td>' in page, page


def test_the_page_states_the_correction_denominator_and_the_costs():
    page = _page()
    assert "24 hypotheses" in page and "FLOOR" in page
    assert "1.3 pip round turn" in page and "1.20%/yr" in page
    assert "24 bars per rollover" in page


def test_the_footer_refuses_to_let_a_pass_read_as_a_prediction():
    page = _page()
    assert "Nothing here is a prediction" in page, page[-400:]
    assert "survived one in-sample test" in page
    assert "--holdout" in page
    assert "does not bound the loss" in page
    assert "2015-01-15" in page, "the gap-risk example was dropped"


def test_data_problems_are_shown_when_there_are_any():
    from engine.data import Problem
    results = {"flag": _result(30, 20.0, 20, 10.0, name="flag")}
    judged = {n: judge(r, COSTS, name=n, costs_confirmed=True)
              for n, r in results.items()}
    s = _series()
    s.problems = [Problem("FROZEN_QUOTES", 42, "a dead feed written as a flat "
                          "market", 100)]
    page = render(s, results, judged, COSTS, 24.0, 24)
    assert "FROZEN_QUOTES" in page and "42" in page
    assert "dead feed" in page
    assert "FROZEN_QUOTES" not in _page(), "a clean series invented a problem"


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
