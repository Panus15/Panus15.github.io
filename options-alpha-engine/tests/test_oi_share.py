"""Tests for the share-of-open-interest screen (models/oi_share.py).

The screen exists to spend an afternoon instead of eighteen months, so the way it
fails matters more than the way it succeeds. Two failure modes would be expensive
in opposite directions:

  - reading missing open interest as zero, which makes every fund look like it
    owns 100% of its strikes and sends the operator off to archive for a year and
    a half on an artefact;
  - silently dropping the fund lines that did not match a listed quote, which
    makes a fund holding OTC notes look like a tiny participant in listed options
    rather than an absent one.

Both are tested directly. The decision bands are asserted against the values the
module registered in advance, so they cannot drift toward a result.

Run: python3 tests/test_oi_share.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.data import OptionChain, OptionQuote
from models.fund_flow import FundBook, FundOptionPosition
from models.oi_share import (PLAUSIBLE_AT, REFUTED_BELOW, share_of_open_interest)

ASOF = "2026-09-11"
SPOT = 600.0


def _book(lines, asof=ASOF):
    return FundBook(fund="TESTFUND", asof=asof, positions=[
        FundOptionPosition(underlying=u, expiry=e, strike=k, kind=kind,
                           contracts=c) for u, e, k, kind, c in lines])


def _chain(rows, asof=ASOF):
    return OptionChain(symbol="QQQ", spot=SPOT, r=0.04, q=0.0, asof=asof, quotes=[
        OptionQuote(expiry_days=d, strike=k, kind=kind, bid=1.0, ask=1.1,
                    open_interest=oi) for d, k, kind, oi in rows])


def test_a_dominant_fund_reads_as_plausible():
    book = _book([("QQQ", "2026-10-11", 620.0, "call", -8_000)])
    chain = _chain([(30, 620.0, "call", 20_000)])
    rep = share_of_open_interest(book, chain, "QQQ")
    assert abs(rep.max_share - 0.40) < 1e-9
    assert "MECHANISM PLAUSIBLE" in rep.verdict()


def test_a_marginal_fund_refutes_the_premise():
    """The outcome that saves eighteen months, and the one the screen is for."""
    book = _book([("QQQ", "2026-10-11", 620.0, "call", -500)])
    chain = _chain([(30, 620.0, "call", 200_000)])
    rep = share_of_open_interest(book, chain, "QQQ")
    assert rep.max_share < REFUTED_BELOW
    v = rep.verdict()
    assert "PREMISE REFUTED" in v and "§2.2" in v


def test_the_middle_band_says_inconclusive_rather_than_picking_a_side():
    book = _book([("QQQ", "2026-10-11", 620.0, "call", -10_000)])
    chain = _chain([(30, 620.0, "call", 100_000)])       # 10%
    rep = share_of_open_interest(book, chain, "QQQ")
    assert REFUTED_BELOW <= rep.max_share < PLAUSIBLE_AT
    assert "INCONCLUSIVE" in rep.verdict()


def test_unknown_open_interest_is_never_read_as_zero():
    """The expensive error. OI 0 means the feed did not carry it, and dividing by
    it would report every fund as owning 100% of every strike."""
    book = _book([("QQQ", "2026-10-11", 620.0, "call", -8_000)])
    chain = _chain([(30, 620.0, "call", 0)])
    rep = share_of_open_interest(book, chain, "QQQ")
    assert rep.rows and rep.rows[0].share is None
    assert rep.max_share is None
    assert not rep.known_rows
    assert "NO OPEN INTEREST DATA" in rep.verdict()
    assert "REFUTED" not in rep.verdict(), (
        "a missing feed was reported as a refutation - that closes a study on an "
        "absence of evidence")


def test_fund_lines_with_no_listed_quote_are_counted_not_discarded():
    """A fund whose overwriting runs through OTC notes holds no listed options at
    all. Dropping those lines would report it as a small participant in a market
    it is not in - the opposite of the truth, and quieter."""
    book = _book([("QQQ", "2026-10-11", 620.0, "call", -8_000),
                  ("QQQ", "2026-10-11", 999.0, "call", -50_000),   # not listed
                  ("QQQ", "2027-01-15", 620.0, "call", -40_000)])  # wrong expiry
    chain = _chain([(30, 620.0, "call", 20_000)])
    rep = share_of_open_interest(book, chain, "QQQ")
    assert len(rep.rows) == 1
    assert rep.unmatched == 2, rep.unmatched
    assert "2 fund line(s) had no matching quote" in rep.summary()


def test_the_pooled_share_and_the_max_answer_different_questions():
    """A fund can top out at 40% on one illiquid strike and still be 2% of its own
    book. Reporting only the max would oversell it."""
    book = _book([("QQQ", "2026-10-11", 620.0, "call", -4_000),
                  ("QQQ", "2026-10-11", 610.0, "call", -2_000)])
    chain = _chain([(30, 620.0, "call", 10_000),      # 40%
                    (30, 610.0, "call", 400_000)])    # 0.5%
    rep = share_of_open_interest(book, chain, "QQQ")
    assert abs(rep.max_share - 0.40) < 1e-9
    assert rep.weighted_share < 0.02, rep.weighted_share
    assert f"{rep.weighted_share:.1%}" in rep.verdict()


def test_strikes_are_matched_with_a_tolerance_because_holdings_round():
    book = _book([("QQQ", "2026-10-11", 620.004, "call", -8_000)])
    chain = _chain([(30, 620.0, "call", 20_000)])
    assert share_of_open_interest(book, chain, "QQQ").unmatched == 0
    far = _chain([(30, 640.0, "call", 20_000)])
    assert share_of_open_interest(book, far, "QQQ").unmatched == 1


def test_another_underlying_in_the_same_book_is_ignored():
    book = _book([("QQQ", "2026-10-11", 620.0, "call", -8_000),
                  ("SPY", "2026-10-11", 620.0, "call", -99_000)])
    chain = _chain([(30, 620.0, "call", 20_000)])
    rep = share_of_open_interest(book, chain, "QQQ")
    assert len(rep.rows) == 1 and abs(rep.max_share - 0.40) < 1e-9


def test_the_bands_are_the_ones_registered_in_advance():
    """A threshold chosen after the answer is known is not a threshold. If these
    move, the amendment discipline in PREREGISTRATION.md §2 applies."""
    assert REFUTED_BELOW == 0.05
    assert PLAUSIBLE_AT == 0.20
    import models.oi_share as m
    assert "before any data was seen" in (m.__doc__ or "").lower()


def test_a_short_and_a_long_line_both_count_as_presence_at_the_strike():
    """Share of open interest is about SIZE at a strike, not direction. A sign
    error here would report a fund's own hedges as negative participation."""
    short = _book([("QQQ", "2026-10-11", 620.0, "call", -8_000)])
    long_ = _book([("QQQ", "2026-10-11", 620.0, "call", +8_000)])
    chain = _chain([(30, 620.0, "call", 20_000)])
    a = share_of_open_interest(short, chain, "QQQ").max_share
    b = share_of_open_interest(long_, chain, "QQQ").max_share
    assert a == b == 0.40


def _run_all():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
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
