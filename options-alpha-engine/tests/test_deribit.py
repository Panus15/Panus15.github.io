"""Offline tests for the Deribit bridge (engine/deribit.py).

No network: we synthesise a realistic `get_book_summary_by_currency` response
from a Black-Scholes chain, quoted in COIN units the way Deribit does, then push
it through book_summary_to_chain and recover the vol. If the coin->USD
conversion were dropped, the recovered vol would be off by ~a factor of spot and
these tests would fail loudly — which is the whole point.
Run: python3 tests/test_deribit.py
"""

import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import pricing
from engine.deribit import book_summary_to_chain, parse_instrument_name, save_chain_json
from models import rnd

SPOT = 60000.0
IV = 0.60
ASOF = date(2026, 7, 19)
EXPIRY_TOK = "18AUG26"          # 30 days after asof
DTE = (date(2026, 8, 18) - ASOF).days   # == 30
T = DTE / 365.0


def _coin_book():
    """A dense BSM chain quoted in COIN units, as Deribit returns it."""
    summary = []
    for K in (48000, 51000, 54000, 57000, 60000, 63000, 66000, 69000, 72000):
        for kind, cp in (("call", "C"), ("put", "P")):
            usd = pricing.price(SPOT, K, T, 0.0, 0.0, IV, kind)
            coin = usd / SPOT                      # Deribit quotes premium in coin
            summary.append({
                "instrument_name": f"BTC-{EXPIRY_TOK}-{K}-{cp}",
                "bid_price": coin, "ask_price": coin,
                "underlying_price": SPOT, "mark_iv": IV * 100,
            })
    return summary


def test_parse_instrument_name():
    assert parse_instrument_name("BTC-18AUG26-60000-C") == (date(2026, 8, 18), 60000.0, "call")
    assert parse_instrument_name("ETH-27JUN25-3000-P")[2] == "put"
    assert parse_instrument_name("BTC-PERPETUAL") is None


def test_coin_to_usd_conversion_recovers_vol():
    chain = book_summary_to_chain(_coin_book(), SPOT, asof=ASOF, symbol="BTC")
    assert abs(chain.spot - SPOT) < 1e-6
    assert all(q.expiry_days == DTE for q in chain.quotes)
    # The clincher: model-free Q vol must come back ~0.60. It only does if the
    # coin premiums were multiplied back to USD.
    mf = rnd.model_free_implied_vol(chain, T, DTE)
    assert abs(mf - IV) < 0.03, mf
    m = rnd.bkm_moments(chain, T, DTE)
    assert m.coverage_ok and abs(m.skew) < 0.1


def test_dropping_conversion_would_break_it():
    # Sanity that the test is meaningful: quotes left in COIN units do NOT
    # recover 0.60 (they recover a wildly different / degenerate number).
    coin_chain = book_summary_to_chain(
        [{**it, "underlying_price": 1.0} for it in _coin_book()],  # pretend px=1 (no conversion)
        1.0, asof=ASOF, symbol="BTC")
    mf_wrong = rnd.model_free_implied_vol(coin_chain, T, DTE)
    assert abs(mf_wrong - IV) > 0.1     # conversion genuinely matters


def test_save_chain_json_roundtrips_via_file_adapter():
    import tempfile
    from engine.adapters import JsonFileAdapter
    chain = book_summary_to_chain(_coin_book(), SPOT, asof=ASOF, symbol="BTC")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "btc.json")
        save_chain_json(chain, path)
        reloaded = JsonFileAdapter(chain_json=path).option_chain("BTC")
    assert abs(reloaded.spot - SPOT) < 1e-6
    assert len(reloaded.quotes) == len(chain.quotes)
    assert reloaded.asof == ASOF.isoformat()


def _run_all():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except (AssertionError, Exception) as e:
            failed += 1
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
