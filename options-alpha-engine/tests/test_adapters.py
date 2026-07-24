"""Tests for the market-data adapters (engine/adapters.py).

OFFLINE ONLY — no network, no API key. The file adapters (CsvFileAdapter /
JsonFileAdapter) are exercised end-to-end against tiny fixtures written to a temp
path, proving they produce OptionChain/OptionQuote objects that are a drop-in for
SyntheticAdapter and plug straight into the downstream Q-extractors and pricer.
The vendor adapters (Polygon/ORATS) are checked only for graceful-offline
behaviour: a CLEAR error when the credential env var is missing — never a network
call, never an import crash.

Run: python3 tests/test_adapters.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.adapters import (
    CsvFileAdapter,
    JsonFileAdapter,
    ORATSAdapter,
    PolygonAdapter,
)
from engine.data import MarketDataAdapter, OptionChain
from engine import pricing
from models import rnd

# --- Fixtures (literal values -> the parse round-trip is non-circular) --------

# A dense, BSM-consistent 30-DTE chain: matched call+put at 9 strikes spanning
# 0.85*S .. 1.15*S, so the model-free / BKM extractors have clean coverage.
SYMBOL = "SPY"
SPOT = 100.0
R = 0.03
Q = 0.0
ASOF = "2024-01-15"

CHAIN_ROWS = [
    # (expiry_days, strike, kind, bid, ask)
    (30, 85, "call", 14.90, 15.53),
    (30, 85, "put", 0.01, 0.05),
    (30, 88, "call", 11.99, 12.50),
    (30, 88, "put", 0.01, 0.06),
    (30, 92, "call", 8.22, 8.58),
    (30, 92, "put", 0.15, 0.21),
    (30, 96, "call", 4.86, 5.08),
    (30, 96, "put", 0.71, 0.77),
    (30, 100, "call", 2.35, 2.47),
    (30, 100, "put", 2.11, 2.22),
    (30, 104, "call", 0.90, 0.96),
    (30, 104, "put", 4.57, 4.78),
    (30, 108, "call", 0.26, 0.32),
    (30, 108, "put", 7.85, 8.20),
    (30, 112, "call", 0.05, 0.11),
    (30, 112, "put", 11.56, 12.05),
    (30, 115, "call", 0.01, 0.06),
    (30, 115, "put", 14.44, 15.05),
]

PRICE_ROWS = [
    ("2024-01-08", 97.5),
    ("2024-01-09", 98.2),
    ("2024-01-10", 97.9),
    ("2024-01-11", 99.1),
    ("2024-01-12", 100.4),
    ("2024-01-15", 100.0),
]


def _write(dirpath, name, text):
    path = os.path.join(dirpath, name)
    with open(path, "w") as fh:
        fh.write(text)
    return path


def _write_csv_fixtures(dirpath):
    price_csv = _write(dirpath, "prices.csv",
                       "date,close\n" +
                       "".join(f"{d},{c}\n" for d, c in PRICE_ROWS))
    body = ["# symbol=%s" % SYMBOL, "# spot=%s" % SPOT, "# r=%s" % R,
            "# q=%s" % Q, "# asof=%s" % ASOF, "expiry_days,strike,kind,bid,ask"]
    body += [f"{dte},{k},{kind},{b},{a}" for dte, k, kind, b, a in CHAIN_ROWS]
    chain_csv = _write(dirpath, "chain.csv", "\n".join(body) + "\n")
    return price_csv, chain_csv


def _write_json_fixtures(dirpath):
    import json
    price_json = _write(dirpath, "prices.json",
                        json.dumps([{"date": d, "close": c} for d, c in PRICE_ROWS]))
    chain = {
        "symbol": SYMBOL, "spot": SPOT, "r": R, "q": Q, "asof": ASOF,
        "quotes": [
            {"expiry_days": dte, "strike": k, "kind": kind, "bid": b, "ask": a}
            for dte, k, kind, b, a in CHAIN_ROWS
        ],
    }
    chain_json = _write(dirpath, "chain.json", json.dumps(chain))
    return price_json, chain_json


# --- Tests --------------------------------------------------------------------

def test_csv_adapter_parses_metadata_and_quotes():
    with tempfile.TemporaryDirectory() as d:
        price_csv, chain_csv = _write_csv_fixtures(d)
        ad = CsvFileAdapter(price_csv, chain_csv)
        chain = ad.option_chain(SYMBOL)

        assert isinstance(chain, OptionChain)
        assert chain.symbol == SYMBOL
        assert chain.spot == SPOT
        assert chain.r == R and chain.q == Q
        assert chain.asof == ASOF                       # look-ahead contract set
        assert len(chain.quotes) == len(CHAIN_ROWS)

        # A known strike's quote parsed correctly (literal expected, not derived).
        c100 = [q for q in chain.quotes if q.strike == 100 and q.kind == "call"][0]
        assert c100.bid == 2.35 and c100.ask == 2.47
        assert c100.kind == "call" and c100.expiry_days == 30


def test_json_adapter_parses_metadata_and_quotes():
    with tempfile.TemporaryDirectory() as d:
        price_json, chain_json = _write_json_fixtures(d)
        ad = JsonFileAdapter(price_json, chain_json)
        chain = ad.option_chain(SYMBOL)

        assert isinstance(chain, OptionChain)
        assert chain.symbol == SYMBOL and chain.spot == SPOT
        assert chain.r == R and chain.q == Q and chain.asof == ASOF
        assert len(chain.quotes) == len(CHAIN_ROWS)

        p100 = [q for q in chain.quotes if q.strike == 100 and q.kind == "put"][0]
        assert p100.bid == 2.11 and p100.ask == 2.22


def test_optionquote_derived_properties():
    with tempfile.TemporaryDirectory() as d:
        _, chain_json = _write_json_fixtures(d)
        chain = JsonFileAdapter(chain_json=chain_json).option_chain(SYMBOL)
        q = [x for x in chain.quotes if x.strike == 100 and x.kind == "call"][0]
        assert abs(q.mid - 0.5 * (2.35 + 2.47)) < 1e-12
        assert abs(q.spread - (2.47 - 2.35)) < 1e-12
        assert abs(q.T - 30 / 365.0) < 1e-12


def test_price_history_oldest_to_newest():
    with tempfile.TemporaryDirectory() as d:
        price_csv, chain_csv = _write_csv_fixtures(d)
        closes = CsvFileAdapter(price_csv, chain_csv).price_history(SYMBOL)
        assert closes == [c for _, c in PRICE_ROWS]     # already ascending by date
        assert closes[0] == 97.5 and closes[-1] == 100.0

        price_json, _ = _write_json_fixtures(d)
        jcloses = JsonFileAdapter(price_json=price_json).price_history(SYMBOL)
        assert jcloses == [c for _, c in PRICE_ROWS]
        # `days` truncation keeps the most recent tail
        assert CsvFileAdapter(price_csv, chain_csv).price_history(SYMBOL, days=3) \
            == [c for _, c in PRICE_ROWS][-3:]


def test_loaded_chain_plugs_into_downstream_engine():
    """Shape-compatibility: the parsed chain feeds the Q-extractors AND the
    pricer/greeks with no adaptation — same contract as SyntheticAdapter."""
    with tempfile.TemporaryDirectory() as d:
        _, chain_csv = _write_csv_fixtures(d)
        chain = CsvFileAdapter(chain_csv=chain_csv).option_chain(SYMBOL)

        T = 30 / 365.0
        # (a) model-free (VIX-style) implied vol runs and is a sane number
        iv = rnd.model_free_implied_vol(chain, T, 30)
        assert iv == iv and 0.05 < iv < 1.0             # finite & plausible

        # (b) BKM moments run with clean coverage on the dense chain
        moments = rnd.bkm_moments(chain, T, 30)
        assert moments.coverage_ok
        assert moments.n_strikes >= 5

        # (c) BSM pricer + greeks accept a parsed quote directly
        q = [x for x in chain.quotes if x.strike == 100 and x.kind == "call"][0]
        px = pricing.price(chain.spot, q.strike, q.T, chain.r, chain.q, iv, q.kind)
        g = pricing.greeks(chain.spot, q.strike, q.T, chain.r, chain.q, iv, q.kind)
        assert px > 0 and g.vega > 0 and g.gamma > 0


def test_file_adapters_conform_to_protocol():
    # Duck-typed structural conformance to MarketDataAdapter. The protocol is not
    # @runtime_checkable (and we must not edit engine/data.py), so we verify the
    # callable methods it declares are present on every adapter.
    required = [m for m in ("price_history", "option_chain")
                if callable(getattr(MarketDataAdapter, m, None))]
    assert set(required) == {"price_history", "option_chain"}
    for ad in (CsvFileAdapter(), JsonFileAdapter(),
               PolygonAdapter(), ORATSAdapter()):
        for method in required:
            assert callable(getattr(ad, method, None)), (type(ad).__name__, method)


def test_polygon_raises_clear_error_without_key():
    saved = os.environ.pop("POLYGON_API_KEY", None)
    try:
        ad = PolygonAdapter()                           # must NOT crash on construct
        for call in (lambda: ad.price_history("SPY", 30),
                     lambda: ad.option_chain("SPY")):
            raised = False
            try:
                call()
            except RuntimeError as e:
                raised = True
                assert "POLYGON_API_KEY" in str(e)      # actionable message
            assert raised, "expected a RuntimeError when POLYGON_API_KEY is unset"
    finally:
        if saved is not None:
            os.environ["POLYGON_API_KEY"] = saved


def test_orats_raises_clear_error_without_token():
    saved = os.environ.pop("ORATS_TOKEN", None)
    try:
        ad = ORATSAdapter()
        for call in (lambda: ad.price_history("SPY", 30),
                     lambda: ad.option_chain("SPY")):
            raised = False
            try:
                call()
            except RuntimeError as e:
                raised = True
                assert "ORATS_TOKEN" in str(e)
            assert raised, "expected a RuntimeError when ORATS_TOKEN is unset"
    finally:
        if saved is not None:
            os.environ["ORATS_TOKEN"] = saved


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
