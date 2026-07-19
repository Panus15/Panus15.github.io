"""Offline tests for the Interactive Brokers bridge (engine/ibkr.py).

No network: the live Client Portal flow needs a user-authenticated local
gateway (and this sandbox blocks egress), so the TESTED surface is the pure
parse helpers. We synthesise IBKR's marketdata-snapshot shapes from a
Black-Scholes SPX chain — European, already in USD, so NO coin conversion —
push them through parse_snapshot_chain, and prove models/rnd recovers the vol.

Also covered: history-bar parsing, the order-payload builder (+ bad side), the
zero/missing bid-ask skip, and the clear errors when the gateway is unreachable
or an order is placed with no account_id.
Run: python3 tests/test_ibkr.py
"""

import os
import sys
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import pricing
from engine.ibkr import (IBKRAdapter, _days_to_expiry, build_order_payload,
                         parse_history, parse_snapshot_chain)
from models import rnd

SPOT, IV, R, Q = 5000.0, 0.18, 0.04, 0.0
ASOF = date(2026, 7, 19)
EXP = ASOF + timedelta(days=30)             # 2026-08-18
EXP_YYYYMMDD = EXP.strftime("%Y%m%d")       # '20260818' (IBKR maturityDate form)
DTE = 30
T = DTE / 365.0


def _snapshot_chain_ibkr():
    """A dense BSM SPX chain shaped as IBKR (snapshots + secdef `meta`).

    Realistic join: each snapshot carries a conid + FIELD-NUMBER price keys
    (84=bid, 86=ask); the contract terms live in `meta` keyed by conid, the way
    /iserver/secdef/info supplies them. This exercises the multi-step join.
    """
    snapshots, meta = [], {}
    conid = 100000
    for K in range(4400, 5601, 100):         # 13 strikes, dense, spans 0.88-1.12x
        for right, kind in (("C", "call"), ("P", "put")):
            conid += 1
            px = pricing.price(SPOT, float(K), T, R, Q, IV, kind)
            snapshots.append({"conid": conid, "84": round(px, 2), "86": round(px, 2),
                              "31": round(px, 2)})
            meta[conid] = {"strike": float(K), "right": right,
                           "maturityDate": EXP_YYYYMMDD}
    return snapshots, meta


def test_parse_snapshot_chain_recovers_vol_european_usd():
    snapshots, meta = _snapshot_chain_ibkr()
    chain = parse_snapshot_chain(snapshots, meta, symbol="SPX", spot=SPOT,
                                 asof=ASOF.isoformat(), r=R, q=Q)
    assert len(chain.quotes) == 26
    assert all(qt.expiry_days == DTE for qt in chain.quotes)   # YYYYMMDD parsed
    assert chain.asof == ASOF.isoformat()
    mf = rnd.model_free_implied_vol(chain, T, DTE)
    assert abs(mf - IV) < 0.02, mf           # USD quotes -> vol recovered directly


def test_parse_snapshot_chain_skips_zero_and_missing():
    # Inline shape (contract terms on the snapshot, meta={}): supported too.
    good = {"strike": 5000.0, "right": "C", "expiry": EXP_YYYYMMDD,
            "bid": 100.0, "ask": 101.0}
    zero_bid = {"strike": 5100.0, "right": "C", "expiry": EXP_YYYYMMDD,
                "84": 0.0, "86": 5.0}
    missing_ask = {"strike": 4900.0, "right": "P", "expiry": EXP_YYYYMMDD,
                   "84": 4.0}                # no ask at all
    chain = parse_snapshot_chain([good, zero_bid, missing_ask], {}, symbol="SPX",
                                 spot=SPOT, asof=ASOF.isoformat())
    assert len(chain.quotes) == 1
    assert chain.quotes[0].strike == 5000.0 and chain.quotes[0].kind == "call"


def test_parse_history_oldest_first():
    hist = {"data": [{"c": 4725.5, "t": 1_700_100_000_000},   # deliberately out of order
                     {"c": 4700.0, "t": 1_700_000_000_000}]}
    assert parse_history(hist) == [4700.0, 4725.5]            # sorted by t (oldest first)
    assert parse_history({"data": []}) == []
    assert parse_history({}) == []


def test_days_to_expiry_accepts_both_formats():
    assert _days_to_expiry(EXP_YYYYMMDD, ASOF.isoformat()) == 30   # IBKR YYYYMMDD
    assert _days_to_expiry(EXP.isoformat(), ASOF.isoformat()) == 30  # ISO
    assert _days_to_expiry(EXP, ASOF) == 30                          # date objects


def test_order_payload_builder():
    mkt = build_order_payload(conid=265598, side="buy", quantity=2)
    assert mkt == {"conid": 265598, "orderType": "MKT", "side": "BUY",
                   "quantity": 2, "tif": "DAY"}
    lmt = build_order_payload(conid=265598, side="SELL", quantity=1,
                              order_type="LMT", price=12.5)
    assert lmt["orderType"] == "LMT" and lmt["side"] == "SELL" and lmt["price"] == 12.5
    for bad in ("bogus", "sell_to_open", "hold"):
        try:
            build_order_payload(conid=1, side=bad, quantity=1)
            assert False, f"should reject side {bad!r}"
        except ValueError:
            pass
    try:
        build_order_payload(conid=1, side="BUY", quantity=1, order_type="LMT")
        assert False, "LMT without price should raise"
    except ValueError:
        pass


def test_place_paper_order_requires_account_id():
    ad = IBKRAdapter(account_id=None)
    ad.account_id = None                     # also clear any env fallback
    payload = build_order_payload(conid=265598, side="BUY", quantity=1)
    try:
        ad.place_paper_order(payload)         # no network: guard fires first
        assert False, "should require an account_id"
    except RuntimeError as e:
        assert "account_id" in str(e)


def test_unreachable_gateway_raises_clear_error():
    # An unknown URL scheme makes urllib.urlopen raise URLError LOCALLY (no
    # socket opened), standing in for an unreachable gateway. Proves _open wraps
    # transport failures into a clear, actionable RuntimeError — fully offline.
    ad = IBKRAdapter(base="unreachable://localhost:5000/v1/api")
    try:
        ad.price_history("SPX", 5)
        assert False, "unreachable gateway should raise"
    except RuntimeError as e:
        assert "gateway" in str(e).lower()


def _run_all():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
