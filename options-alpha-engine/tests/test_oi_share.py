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

THE PATH THE OPERATOR ACTUALLY RUNS WAS UNTESTED. Everything above builds a
`FundBook` in memory, while `tools/oi_share.py` reads a CSV through
`FundBook.from_file` and a chain through `JsonFileAdapter`. Three defects were
stacked in that gap and each one presented as "the chain carries no open interest":

  - `save_chain_json` — the dump command the docs recommend — did not WRITE open
    interest, and `JsonFileAdapter` did not READ it, so a replayed chain never had
    the field the screen needs;
  - a row whose contract sits in a column called `symbol` had the whole OSI string
    kept as its underlying, so every line was filtered out against "QQQ" before it
    could even be counted as unmatched;
  - the CLI passed no as-of date, so days-to-expiry came out None for every line.

The file path is tested from here on.

Run: python3 tests/test_oi_share.py
"""

import io
import json
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


# --------------------------------------------------------------------------
# The path the operator actually runs: a file on disk, a dumped chain
# --------------------------------------------------------------------------

_OSI_BOOK = (
    "symbol,description,quantity\n"
    'QQQ   261016C00570000,"QQQ 10/16/2026 570.00 C",-9500\n'
    'QQQ   261016C00580000,"QQQ 10/16/2026 580.00 C",-15000\n'
)


def _tmpdir():
    import tempfile
    return tempfile.mkdtemp(prefix="oi-share-")


def test_a_contract_in_a_column_called_symbol_still_yields_its_root():
    """`pick` reads "symbol" as a ticker column and issuer files routinely put the
    CONTRACT there. "QQQ   261016C00570000" is truthy, so `underlying or root` kept
    the whole symbol — and the screen then filtered every line out against "QQQ",
    reporting zero rows AND zero unmatched, which reads as a chain problem."""
    import os as _os
    d = _tmpdir()
    path = _os.path.join(d, "2026-09-13.csv")
    io.open(path, "w", encoding="utf-8").write(_OSI_BOOK)
    book = FundBook.from_file(path, fund="QQQI", asof="2026-09-13")
    assert len(book.positions) == 2, book.positions
    assert {p.underlying for p in book.positions} == {"QQQ"}, (
        [p.underlying for p in book.positions])
    assert {p.strike for p in book.positions} == {570.0, 580.0}
    assert all(p.kind == "call" for p in book.positions)
    assert all(p.expiry == "2026-10-16" for p in book.positions)


def test_an_explicit_ticker_column_still_wins_over_the_symbol():
    """The fix must not go the other way: a file that tabulates its underlying is
    more trustworthy than a symbol string, which may be stale."""
    import os as _os
    d = _tmpdir()
    path = _os.path.join(d, "2026-09-13.csv")
    io.open(path, "w", encoding="utf-8").write(
        "ticker,symbol,quantity\n"
        'NDX,QQQ   261016C00570000,-100\n')
    book = FundBook.from_file(path, fund="F", asof="2026-09-13")
    assert book.positions[0].underlying == "NDX", book.positions[0].underlying


def test_open_interest_survives_the_dump_and_the_replay():
    """The documented capture route. It lost the field at BOTH ends, so the screen
    could never be answered from a replay however good the chain was."""
    import os as _os
    from engine.adapters import JsonFileAdapter
    from engine.deribit import save_chain_json
    d = _tmpdir()
    path = _os.path.join(d, "chain.json")
    ch = OptionChain(symbol="QQQ", spot=555.0, r=0.03, q=0.0, asof="2026-09-13",
                     quotes=[OptionQuote(expiry_days=33, strike=570.0, kind="call",
                                         bid=2.1, ask=2.2, open_interest=38_000),
                             OptionQuote(expiry_days=33, strike=580.0, kind="call",
                                         bid=1.1, ask=1.2, open_interest=0)])
    save_chain_json(ch, path)
    payload = json.load(io.open(path, encoding="utf-8"))
    assert all("open_interest" in row for row in payload["quotes"]), payload["quotes"]
    back = JsonFileAdapter(chain_json=path).option_chain("QQQ")
    assert [q.open_interest for q in back.quotes] == [38_000, 0], (
        "a 0 must survive as 0 — it means UNKNOWN downstream, not missing")


def test_a_chain_file_without_the_field_still_loads_as_unknown():
    """Older dumps have no `open_interest` key at all. They must read as UNKNOWN
    rather than crash, so an old fixture keeps working and says nothing."""
    import os as _os
    from engine.adapters import JsonFileAdapter
    d = _tmpdir()
    path = _os.path.join(d, "old.json")
    io.open(path, "w", encoding="utf-8").write(json.dumps({
        "symbol": "QQQ", "spot": 555.0, "r": 0.03, "q": 0.0, "asof": "2026-09-13",
        "quotes": [{"expiry_days": 33, "strike": 570.0, "kind": "call",
                    "bid": 2.1, "ask": 2.2}]}))
    back = JsonFileAdapter(chain_json=path).option_chain("QQQ")
    assert back.quotes[0].open_interest == 0


def test_the_asof_comes_from_the_archivers_own_filename():
    from tools.oi_share import asof_from_path
    assert asof_from_path("holdings/QQQI/2026-09-13.csv") == "2026-09-13"
    assert asof_from_path("/a/b/QQQI/2025-01-02.json") == "2025-01-02"
    # and it is NEVER guessed: a name with no date yields nothing, so the caller
    # reports an empty as-of instead of silently substituting today
    assert asof_from_path("holdings/QQQI/latest.csv") == ""
    assert asof_from_path("QQQI_holdings.csv") == ""


def test_the_whole_cli_path_reaches_a_verdict():
    """File on disk, chain dumped to JSON, a real share printed. This is the run
    that could close PREREGISTRATION §2.2 in an afternoon, and before these fixes
    it reported NO OPEN INTEREST DATA no matter what it was given."""
    import datetime
    import os as _os
    from engine.deribit import save_chain_json
    from tools.oi_share import main as oi_main
    d = _tmpdir()
    today = datetime.date.today()
    dte = (datetime.date(2026, 10, 16) - today).days
    book_path = _os.path.join(d, today.isoformat() + ".csv")
    io.open(book_path, "w", encoding="utf-8").write(_OSI_BOOK)
    chain_path = _os.path.join(d, "chain.json")
    save_chain_json(OptionChain(
        symbol="QQQ", spot=555.0, r=0.03, q=0.0, asof=today.isoformat(),
        quotes=[OptionQuote(expiry_days=dte, strike=570.0, kind="call", bid=2.1,
                            ask=2.2, open_interest=38_000),
                OptionQuote(expiry_days=dte, strike=580.0, kind="call", bid=1.1,
                            ask=1.2, open_interest=52_000)]), chain_path)
    rc = oi_main(["--holdings", book_path, "--underlying", "QQQ", "--fund", "QQQI",
                  "--source", "replay", "--chain-json", chain_path])
    assert rc == 0, "the screen could not reach a verdict from a complete input"


def test_nothing_matching_is_reported_as_such_and_not_blamed_on_the_chain():
    """Two failures used to print the same sentence. "No strike matched" and
    "matched but no open interest" have different causes and different fixes, and
    blaming the chain for a date mismatch sends the operator to re-dump a chain
    that was fine."""
    import contextlib
    import datetime
    import io as _io
    import os as _os
    from engine.deribit import save_chain_json
    from tools.oi_share import main as oi_main
    d = _tmpdir()
    today = datetime.date.today()
    book_path = _os.path.join(d, today.isoformat() + ".csv")
    io.open(book_path, "w", encoding="utf-8").write(_OSI_BOOK)
    chain_path = _os.path.join(d, "chain.json")
    # a chain at a DIFFERENT expiry: nothing can match, and the open interest is
    # perfectly good
    save_chain_json(OptionChain(
        symbol="QQQ", spot=555.0, r=0.03, q=0.0, asof=today.isoformat(),
        quotes=[OptionQuote(expiry_days=3, strike=570.0, kind="call", bid=2.1,
                            ask=2.2, open_interest=38_000)]), chain_path)
    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = oi_main(["--holdings", book_path, "--underlying", "QQQ",
                      "--source", "replay", "--chain-json", chain_path])
    out = buf.getvalue()
    assert rc == 1
    assert "NOTHING MATCHED" in out, out
    assert "no open interest" not in out.lower(), (
        "a date mismatch was blamed on the chain's open interest:\n" + out)


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
