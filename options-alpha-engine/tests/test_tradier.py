"""Offline tests for the Tradier bridge (engine/tradier.py).

No network: synthesise Tradier's JSON shapes (a Black-Scholes SPX chain, already
in USD) and push them through the pure parsers. Also exercises Tradier's
single-vs-list quirk and the paper-order payload builder.
Run: python3 tests/test_tradier.py
"""

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
