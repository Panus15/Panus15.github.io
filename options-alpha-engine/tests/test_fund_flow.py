"""Tests for the fund option footprint (models/fund_flow.py).
Run: python3 tests/test_fund_flow.py
"""

import csv
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.fund_flow import (FundBook, FundOptionPosition, crowding_score,
                              iv_dent, roll_activity, supply_map)

SPOT = 450.0


def _book(fund, asof, lines):
    return FundBook(fund, asof, [FundOptionPosition(*ln) for ln in lines])


QQQI = _book("QQQI", "2026-08-01", [
    ("QQQ", "2026-08-21", 470.0, "call", -8000.0),      # the big short
    ("QQQ", "2026-08-21", 480.0, "call", -2000.0),
    ("QQQ", "2026-09-18", 475.0, "call", -3000.0),
    ("QQQ", "2026-08-21", 400.0, "put", 1500.0),        # LONG protective put
])
JEPQ = _book("JEPQ", "2026-08-01", [
    ("QQQ", "2026-08-21", 470.0, "call", -5000.0),      # same strike as QQQI
    ("QQQ", "2026-08-21", 465.0, "call", -1000.0),
])


def test_short_and_long_positions_are_distinguished():
    shorts = QQQI.shorts("QQQ")
    assert len(shorts) == 3                              # the long put is excluded
    assert all(p.is_short and p.contracts < 0 for p in shorts)
    assert all(p.kind == "call" for p in shorts)


def test_notional_and_moneyness():
    p = FundOptionPosition("QQQ", "2026-08-21", 470.0, "call", -8000.0)
    assert p.notional(SPOT) == 8000 * 100 * SPOT
    assert abs(p.moneyness(SPOT) - (470.0 / 450.0 - 1.0)) < 1e-12
    assert p.days_to_expiry("2026-08-01") == 20
    assert p.days_to_expiry("bad-date") is None


def test_supply_map_aggregates_across_funds_and_ranks_by_notional():
    buckets = supply_map([QQQI, JEPQ], "QQQ", SPOT)
    top = buckets[0]
    assert (top.expiry, top.strike, top.kind) == ("2026-08-21", 470.0, "call")
    assert top.contracts == 13000                        # 8000 QQQI + 5000 JEPQ
    assert set(top.funds) == {"QQQI", "JEPQ"}
    assert all(buckets[i].notional >= buckets[i + 1].notional
               for i in range(len(buckets) - 1))
    assert not any(b.kind == "put" for b in buckets)     # only SHORT supply mapped
    assert SPOT * 100 * 13000 == top.notional


def test_crowding_flags_our_strike_and_tolerates_near_misses():
    buckets = supply_map([QQQI, JEPQ], "QQQ", SPOT)
    share, note = crowding_score(470.0, "call", 20, "2026-08-01", buckets)
    assert share > 0.5 and "QQQI" in note and "joining their flow" in note
    # a 0.5% strike difference and a few days of expiry drift still match
    near, _ = crowding_score(472.0, "call", 23, "2026-08-01", buckets)
    assert near > 0.5
    # a genuinely different strike does not
    far, note_far = crowding_score(430.0, "call", 20, "2026-08-01", buckets)
    assert far == 0.0 and "no mapped fund supply" in note_far
    # a put is not matched against short-call supply
    assert crowding_score(470.0, "put", 20, "2026-08-01", buckets)[0] == 0.0
    # no data is stated as no data
    assert crowding_score(470.0, "call", 20, "2026-08-01", [])[0] == 0.0


def test_roll_activity_shows_where_new_supply_landed():
    later = _book("QQQI", "2026-08-22", [
        ("QQQ", "2026-09-18", 480.0, "call", -8000.0),   # rolled out and up
        ("QQQ", "2026-09-18", 475.0, "call", -3000.0),   # unchanged
    ])
    r = roll_activity(QQQI, later, "QQQ", SPOT)
    assert r["from"] == "2026-08-01" and r["to"] == "2026-08-22"
    assert 480.0 in r["opened_strikes"]
    assert 470.0 in r["closed_strikes"]                  # the August short is gone
    assert r["opened_contracts"] > 0 and r["closed_contracts"] > 0


def test_book_from_csv_and_json_round_trip():
    rows = [{"underlying": "QQQ", "expiry": "2026-08-21", "strike": "470",
             "type": "Call", "quantity": "-8000"},
            {"underlying": "QQQ", "expiry": "2026-08-21", "strike": "400",
             "type": "Put", "quantity": "1500"},
            {"underlying": "QQQ", "expiry": "", "strike": "0", "type": "call",
             "quantity": "5"}]                            # junk row -> dropped
    d = tempfile.mkdtemp()
    try:
        cpath = os.path.join(d, "h.csv")
        with open(cpath, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        b = FundBook.from_file(cpath, fund="QQQI", asof="2026-08-01")
        assert len(b.positions) == 2                      # strike=0 row dropped
        assert len(b.shorts("QQQ")) == 1
        assert b.shorts("QQQ")[0].kind == "call"

        jpath = os.path.join(d, "h.json")
        with open(jpath, "w") as fh:
            json.dump({"fund": "JEPQ", "asof": "2026-08-01", "positions": rows}, fh)
        bj = FundBook.from_file(jpath)
        assert bj.fund == "JEPQ" and bj.asof == "2026-08-01"
        assert len(bj.shorts("QQQ")) == 1
    finally:
        import shutil
        shutil.rmtree(d)


def test_iv_dent_finds_a_suppressed_strike():
    from engine import pricing
    from engine.data import OptionChain, OptionQuote
    from models import surface
    dte, T, spot = 30, 30 / 365.0, 100.0

    def true_iv(k):
        m = k / spot - 1.0
        return 0.22 - 0.4 * m + 0.5 * m * m

    quotes = []
    for k in range(85, 116, 2):
        iv = true_iv(k)
        if k == 105:
            iv -= 0.03                                    # a supply dent
        for kind in ("call", "put"):
            px = pricing.price(spot, float(k), T, 0.03, 0.0, iv, kind)
            if px < 0.02:
                continue
            quotes.append(OptionQuote(dte, float(k), kind, round(px, 4), round(px, 4)))
    chain = OptionChain("X", spot, 0.03, 0.0, quotes)
    fit = surface.fit_chain(chain, T, dte)
    dents = iv_dent(chain, fit, dte, min_dent=0.01)
    assert dents and abs(dents[0][0] - 105.0) < 1e-9      # the dented strike is found
    assert dents[0][3] > 0.01                             # and it is a real gap



# --------------------------------------------------------------------------
# Reading a REAL issuer file — the archive is worthless if this fails
# --------------------------------------------------------------------------

def test_osi_round_trips_against_the_spec():
    """OSI is a fixed-width spec, so this is an oracle: root, YYMMDD, C/P, and
    the strike as price x 1000 in 8 digits. The answers are known before the
    code runs."""
    from models.fund_flow import parse_osi
    assert parse_osi("QQQ   261016C00620000") == ("QQQ", "2026-10-16", "call", 620.0)
    assert parse_osi("AAPL  240119C00190000") == ("AAPL", "2024-01-19", "call", 190.0)
    assert parse_osi("SPY   251219P00450500") == ("SPY", "2025-12-19", "put", 450.5)
    # a fractional strike must survive the x1000 integer encoding exactly
    assert parse_osi("XSP   260320P00067825")[3] == 67.825
    # unpadded roots appear in the wild too
    assert parse_osi("QQQ 261016C00620000") == ("QQQ", "2026-10-16", "call", 620.0)


def test_osi_declines_everything_that_is_not_an_option():
    """A holdings file is mostly equities and cash. A parser that guesses here
    would invent option positions out of ticker strings."""
    from models.fund_flow import parse_osi
    for junk in ("", "AAPL", "CASH USD", "-", "US912797KL553",
                 "QQQ   261016X00620000",      # not C or P
                 "QQQ   261399C00620000",      # month 13
                 "QQQ   2610160062000",        # too short
                 "      261016C00620000"):     # no root
        assert parse_osi(junk) is None, junk


def test_a_real_issuer_file_parses_even_though_its_columns_are_unguessable():
    """The failure this prevents is silent and expensive.

    Real funds name the contract in ONE column - StockTicker, Identifier,
    SecurityID, depending on the issuer - instead of tabulating strike, expiry
    and type. The old parser needed those columns, found none, dropped every
    line, and tools/archive_holdings then rejected the day with 'the URL
    probably returned a web page'. The download was fine. The operator would
    have banked nothing while being told the wrong reason.
    """
    d = tempfile.mkdtemp()
    try:
        neos = os.path.join(d, "neos.csv")
        with open(neos, "w") as fh:
            fh.write("Account,StockTicker,SecurityName,Shares,MarketValue\n"
                     "QQQI,QQQ   261016C00620000,QQQ OCT 26 620 CALL,-8000,-1200000\n"
                     "QQQI,QQQ   261016C00610000,QQQ OCT 26 610 CALL,-2000,-410000\n"
                     "QQQI,NVDA,NVIDIA CORP,15000,2700000\n"
                     "QQQI,-,CASH,0,140000\n")
        b = FundBook.from_file(neos, fund="QQQI", asof="2026-09-12")
        assert len(b.positions) == 2, [p.__dict__ for p in b.positions]
        assert {p.strike for p in b.positions} == {620.0, 610.0}
        assert all(p.underlying == "QQQ" and p.kind == "call" for p in b.positions)
        assert len(b.shorts("QQQ")) == 2, "the short calls are the whole point"

        # a different issuer, a different column name, same outcome
        gx = os.path.join(d, "gx.csv")
        with open(gx, "w") as fh:
            fh.write("Ticker,Identifier,Name,Quantity,Weight\n"
                     "XYLD,SPY   251219P00450500,SPY DEC25 450.5 PUT,-1200,0.4\n"
                     "XYLD,AAPL,APPLE INC,9000,3.1\n")
        b2 = FundBook.from_file(gx, fund="XYLD", asof="2026-09-12")
        assert len(b2.positions) == 1
        assert b2.positions[0].strike == 450.5 and b2.positions[0].kind == "put"
    finally:
        shutil.rmtree(d)


def test_explicit_columns_win_over_a_symbol_that_disagrees():
    """A file carrying both is more likely to have a stale symbol than a stale
    column, and silently preferring the symbol would move a strike."""
    d = tempfile.mkdtemp()
    try:
        # the symbol disagrees with the columns on strike, KIND and expiry, so
        # every field is checked - overriding any one of them moves a contract
        p = os.path.join(d, "both.csv")
        with open(p, "w") as fh:
            fh.write("symbol,underlying,expiry,strike,kind,quantity\n"
                     "SPY   991231P00999000,QQQ,2026-10-16,620,call,-8000\n")
        b = FundBook.from_file(p, fund="F", asof="2026-09-12")
        assert len(b.positions) == 1
        got = b.positions[0]
        assert got.strike == 620.0, "the OSI symbol overrode the strike column"
        assert got.kind == "call", "the OSI symbol overrode the kind column"
        assert got.expiry == "2026-10-16", "the OSI symbol overrode the expiry column"
        assert got.underlying == "QQQ", "the OSI symbol overrode the underlying"
    finally:
        shutil.rmtree(d)


def test_an_equity_only_fund_yields_no_option_lines_without_inventing_any():
    """JEPI and JEPQ run their overwriting through OTC equity-linked notes and
    hold no listed options. Zero is the right answer for them and must not
    become non-zero by accident."""
    d = tempfile.mkdtemp()
    try:
        p = os.path.join(d, "eq.csv")
        with open(p, "w") as fh:
            fh.write("Ticker,Name,Shares\n"
                     "MSFT,MICROSOFT CORP,120000\n"
                     "GS ELN 09/30/26,GOLDMAN SACHS ELN,50000\n")
        assert FundBook.from_file(p, fund="JEPQ", asof="2026-09-12").positions == []
    finally:
        shutil.rmtree(d)


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
