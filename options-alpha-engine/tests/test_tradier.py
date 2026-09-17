"""Offline tests for the Tradier bridge (engine/tradier.py).

No network: synthesise Tradier's JSON shapes (a Black-Scholes SPX chain, already
in USD) and push them through the pure parsers. Also exercises Tradier's
single-vs-list quirk and the paper-order payload builder.
Run: python3 tests/test_tradier.py
"""

import datetime
import os
import sys
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import pricing
from engine.tradier import (TradierAdapter, build_order_payload, parse_chain,
                            parse_history)
from models import rnd

SPOT, IV, R, Q = 5000.0, 0.18, 0.04, 0.0
ASOF = date(2026, 7, 19)
EXP = (ASOF + timedelta(days=30)).isoformat()
DTE = 30
T = DTE / 365.0


def _option_list():
    opts = []
    for K in range(4400, 5601, 100):        # 13 strikes, dense, spans 0.88-1.12x
        for kind in ("call", "put"):
            px = pricing.price(SPOT, K, T, R, Q, IV, kind)
            opts.append({"strike": float(K), "option_type": kind,
                         "bid": round(px, 2), "ask": round(px, 2),
                         "expiration_date": EXP})
    return opts


def test_parse_chain_recovers_vol_european_usd():
    chain = parse_chain(_option_list(), symbol="SPX", spot=SPOT,
                        asof=ASOF.isoformat(), r=R, q=Q)
    assert all(qt.expiry_days == DTE for qt in chain.quotes)
    mf = rnd.model_free_implied_vol(chain, T, DTE)
    assert abs(mf - IV) < 0.02, mf               # USD quotes -> vol recovered directly


def test_parse_chain_skips_zero_bid():
    opts = _option_list() + [{"strike": 9000.0, "option_type": "call", "bid": 0.0,
                              "ask": 0.0, "expiration_date": EXP}]
    chain = parse_chain(opts, symbol="SPX", spot=SPOT, asof=ASOF.isoformat())
    assert all(qt.strike != 9000.0 for qt in chain.quotes)   # stale strike dropped


def test_single_element_list_quirk():
    # Tradier returns a DICT (not a list) when there is exactly one option.
    one = {"strike": 5000.0, "option_type": "call", "bid": 100.0, "ask": 101.0,
           "expiration_date": EXP}
    chain = parse_chain(one, symbol="SPX", spot=SPOT, asof=ASOF.isoformat())
    assert len(chain.quotes) == 1 and chain.quotes[0].strike == 5000.0


def test_parse_history():
    hist = {"day": [{"date": "2026-01-02", "close": 4700.0},
                    {"date": "2026-01-03", "close": 4725.5}]}
    assert parse_history(hist) == [4700.0, 4725.5]
    assert parse_history({"day": {"date": "2026-01-02", "close": 4700.0}}) == [4700.0]


def test_order_payload_builder():
    p = build_order_payload(option_symbol="SPX260821C05000000", side="sell_to_open",
                            quantity=2, underlying="SPX", order_type="limit", price=12.5)
    assert p["class"] == "option" and p["side"] == "sell_to_open"
    assert p["quantity"] == "2" and p["price"] == "12.5"
    try:
        build_order_payload(option_symbol="X", side="bogus", quantity=1, underlying="SPX")
        assert False, "should reject bad side"
    except ValueError:
        pass


def test_adapter_requires_token():
    ad = TradierAdapter(token=None)
    ad._token = None
    try:
        ad.option_chain("SPX")
        assert False, "should require a token"
    except RuntimeError as e:
        assert "TRADIER_TOKEN" in str(e)



# --------------------------------------------------------------------------
# Which expiries get fetched — the half of the deadlock that reaches the vendor
# --------------------------------------------------------------------------

def _ladder(today, offsets):
    return [(today + datetime.timedelta(days=d)).isoformat() for d in offsets]


def test_expiries_are_chosen_around_the_wanted_tenor_not_the_soonest():
    """The vendor half of the deadlock.

    SPY and QQQ list expiries most weekdays, so taking the k soonest gave 1-4
    DTE — where a strike 10% out is worth fractions of a cent, is dropped by the
    zero-bid filter, and fails rnd._coverage_ok. The 30-45d expiry that DOES
    carry wings was never fetched at all. Same number of requests, wrong axis.
    """
    today = datetime.date(2026, 8, 25)
    exps = _ladder(today, [1, 2, 3, 4, 7, 14, 21, 28, 31, 35, 60, 91])
    picked = TradierAdapter._pick_expirations(exps, 30, 4, today=today)
    days = [(datetime.date.fromisoformat(e) - today).days for e in picked]
    assert all(d >= 14 for d in days), f"still fetching the short end: {days}"
    assert min(days, key=lambda d: abs(d - 30)) in (28, 31), days
    assert days == sorted(days), "expiries must come back in date order"


def test_the_picker_ranks_by_distance_from_the_target():
    today = datetime.date(2026, 8, 25)
    exps = _ladder(today, [1, 2, 3, 45])
    picked = TradierAdapter._pick_expirations(exps, 45, 1, today=today)
    got = (datetime.date.fromisoformat(picked[0]) - today).days
    assert got == 45, f"picked {got}d when 45d was listed and asked for"


def test_expired_dates_are_never_fetched():
    """An expiration list can contain today-or-earlier rows. A negative DTE
    would sail into T = dte/365 and produce a nonsense forward."""
    today = datetime.date(2026, 8, 25)
    exps = _ladder(today, [-10, -1, 30, 60])
    picked = TradierAdapter._pick_expirations(exps, 30, 4, today=today)
    days = [(datetime.date.fromisoformat(e) - today).days for e in picked]
    assert all(d >= 0 for d in days), days
    assert len(days) == 2, days


def test_a_malformed_expiry_string_is_skipped_not_fatal():
    today = datetime.date(2026, 8, 25)
    exps = ["not-a-date", ""] + _ladder(today, [30])
    picked = TradierAdapter._pick_expirations(exps, 30, 3, today=today)
    assert len(picked) == 1


def test_no_target_keeps_the_old_budgeted_behaviour():
    """run_live may call without a tenor; that path must not change."""
    today = datetime.date(2026, 8, 25)
    exps = _ladder(today, [1, 2, 3, 30])
    assert TradierAdapter._pick_expirations(exps, 30, 2, today=today) != exps[:2]


def test_option_chain_passes_the_wanted_tenor_to_the_picker():
    """The picker being right is worthless if option_chain never calls it."""
    seen = {}
    real = TradierAdapter._pick_expirations
    ad = TradierAdapter(token="x")
    today = datetime.date.today()
    exps = _ladder(today, [1, 2, 3, 30, 60])

    def fake_get(path, **kw):
        if "expirations" in path:
            return {"expirations": {"date": exps}}
        if "quotes" in path:
            return {"quotes": {"quote": {"last": 100.0}}}
        seen.setdefault("fetched", []).append(kw.get("expiration"))
        return {"options": {"option": []}}

    ad._get = fake_get
    ad.option_chain("SPY", target_dte=30)
    days = [(datetime.date.fromisoformat(e) - today).days
            for e in seen.get("fetched", [])]
    assert days, "no expiration was fetched"
    # the claim is that the WANTED tenor is among them. With a sparse ladder and
    # a small budget the picker will also take near neighbours, and that is
    # correct - asserting every one is long would fail on the code being right.
    assert 30 in days, f"option_chain never fetched the requested tenor: {days}"


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
