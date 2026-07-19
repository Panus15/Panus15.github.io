"""Offline test for the live runner's pure `analyze` core (tools/run_live.py).

No network: build a dense European (BSM) chain + a real-length price history and
run the whole pipeline, asserting it produces a coherent report. Also checks the
replay path (dump chain -> JsonFileAdapter -> analyze).
Run: python3 tests/test_run_live.py
"""

import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import pricing
from engine.data import OptionChain, OptionQuote, SyntheticAdapter
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
