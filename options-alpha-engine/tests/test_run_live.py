"""Offline test for the live runner's pure `analyze` core (tools/run_live.py).

No network: build a dense European (BSM) chain + a real-length price history and
run the whole pipeline, asserting it produces a coherent report. Also checks the
replay path (dump chain -> JsonFileAdapter -> analyze).
Run: python3 tests/test_run_live.py
"""

import math
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import pricing
from engine.data import OptionChain, OptionQuote, SyntheticAdapter
from tools import run_live
from tools.run_live import analyze

SPOT, IV = 100.0, 0.20


def _chain(dtes=(7, 30, 60)):
    quotes = []
    for dte in dtes:
        T = dte / 365.0
        for K in range(80, 121, 5):                 # dense, spans 0.8-1.2x
            for kind in ("call", "put"):
                px = pricing.price(SPOT, K, T, 0.0, 0.0, IV, kind)
                if px < 0.01:
                    continue
                quotes.append(OptionQuote(dte, float(K), kind, round(px, 3), round(px, 3)))
    return OptionChain("TEST", SPOT, 0.0, 0.0, quotes, asof=date(2026, 7, 19).isoformat())


def test_analyze_produces_full_report():
    prices = SyntheticAdapter(seed=4).price_history("X", 400)
    rep = analyze(_chain(), prices, dte=30, label="offline", run_backtest=True)
    assert rep["n_quotes"] > 0
    assert 0.10 < rep["q_vol"] < 0.30              # model-free recovers ~0.20
    assert rep["coverage_ok"] is True
    assert rep["signals"] and all("verdict" in s for s in rep["signals"])
    assert "backtest" in rep and "sharpe" in rep["backtest"]


def test_analyze_handles_short_history():
    # Too little history -> scan + backtest skipped gracefully, Q still computed.
    rep = analyze(_chain(), [100.0, 101.0, 99.5], dte=30, run_backtest=True)
    assert "q_vol" in rep
    assert "signals" not in rep and "backtest" not in rep


def test_replay_roundtrip_via_json_adapter():
    import tempfile
    from engine.adapters import JsonFileAdapter
    from engine.deribit import save_chain_json
    chain = _chain()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "chain.json")
        save_chain_json(chain, path)
        reloaded = JsonFileAdapter(chain_json=path).option_chain("TEST")
    prices = SyntheticAdapter(seed=4).price_history("X", 300)
    rep = analyze(reloaded, prices, dte=30, label="replay")
    assert rep["n_quotes"] == len(chain.quotes) and "q_vol" in rep


def test_analyze_snaps_to_the_nearest_listed_expiry():
    # Vendors list their own expiries (Deribit: 5/12/19/33/...); a requested 30d
    # matches nothing and Q extraction used to fail on an empty slice.
    import contextlib
    import io
    from engine import pricing
    from engine.data import OptionChain, OptionQuote
    spot = 100.0
    quotes = []
    for dte, iv in ((19, 0.30), (33, 0.32)):
        t = dte / 365.0
        for k in (90.0, 95.0, 100.0, 105.0, 110.0):
            for kind in ("call", "put"):
                px = pricing.price(spot, k, t, 0.03, 0.0, iv, kind)
                if px < 0.02:
                    continue
                quotes.append(OptionQuote(dte, k, kind, round(px * 0.99, 4),
                                          round(px * 1.01, 4)))
    chain = OptionChain("X", spot, 0.03, 0.0, quotes, asof="2026-01-02")
    prices = [100.0 * math.exp(0.01 * math.sin(i / 4.0)) for i in range(80)]

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rep = run_live.analyze(chain, prices, dte=30, run_backtest=False)
    out = buf.getvalue()
    assert "snapping to the nearest: 33d" in out          # 30 -> 33 (nearer than 19)
    assert rep["dte"] == 33                               # the report records what was USED
    assert "q_vol" in rep                                 # Q extraction now succeeds

    # an exact match must NOT snap or announce anything
    buf2 = io.StringIO()
    with contextlib.redirect_stdout(buf2):
        rep2 = run_live.analyze(chain, prices, dte=19, run_backtest=False)
    assert "snapping" not in buf2.getvalue() and rep2["dte"] == 19


def test_replayed_crypto_chain_keeps_its_unit_contract_size():
    # The dump-then-replay path is documented; a replayed BTC chain must keep a
    # 1-coin contract, not silently revert to 100 shares (a 100x overstatement).
    from engine.data import OptionChain
    btc = OptionChain("BTC", 64000.0, 0.03, 0.0, [], asof="2026-01-02")
    spy = OptionChain("SPY", 500.0, 0.03, 0.0, [], asof="2026-01-02")
    assert run_live.default_contract_mult("deribit", btc) == 1.0
    assert run_live.default_contract_mult("replay", btc) == 1.0      # the defect
    assert run_live.default_contract_mult("replay", spy) == 100.0
    assert run_live.default_contract_mult("tradier", spy) == 100.0


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
