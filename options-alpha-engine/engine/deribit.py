"""Deribit bridge — run the engine on REAL European options today, keyless.

Why Deribit is the lowest-friction real-data test bed for THIS engine:
  * BTC/ETH options are EUROPEAN and cash-settled, so models/rnd.py's BKM /
    VIX-style / Breeden-Litzenberger extraction is valid with NO de-Americanizing
    (US single-name equity options are American and would bias the Q moments).
  * The WHOLE chain — bid/ask + mark IV for every strike/expiry — comes back in
    ONE unauthenticated GET (`get_book_summary_by_currency`). No API key, no cost.
  * Deribit also publishes DVOL, a VIX-style index, to cross-check
    `rnd.model_free_implied_vol` against.

⚠️  THE SILENT CORRUPTOR: Deribit inverse-option premiums are quoted in COIN
    (BTC/ETH) while strike and spot are in USD. Feed `bid_price`/`ask_price`
    straight into an OptionQuote and the BKM moments are wrong by ~a factor of
    spot — and nothing errors, the mids just look plausible. `book_summary_to_chain`
    converts premiums to USD (x index price) BEFORE building quotes. The offline
    test round-trips a Black-Scholes chain through coin units to prove it.

Usage (in an environment with outbound network to www.deribit.com):
    python3 -m engine.deribit BTC 30            # pull, save JSON, print Q moments
    # then replay offline forever via the existing JsonFileAdapter.

Note: this session's sandbox blocks market-data hosts; run it where egress is
open (your machine, or an environment whose network policy allows deribit.com).
Pure stdlib (urllib/json/datetime).
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from datetime import date, datetime, timezone

from engine.data import OptionChain, OptionQuote

MAINNET = "https://www.deribit.com/api/v2"
TESTNET = "https://test.deribit.com/api/v2"


def parse_instrument_name(name: str):
    """'BTC-18AUG26-60000-C' -> (expiry_date, strike, kind). None on non-option."""
    parts = name.split("-")
    if len(parts) != 4:
        return None
    _, expiry_tok, strike_tok, cp = parts
    try:
        expiry = datetime.strptime(expiry_tok, "%d%b%y").date()
        strike = float(strike_tok)
    except ValueError:
        return None
    kind = "call" if cp.upper() == "C" else "put" if cp.upper() == "P" else None
    if kind is None:
        return None
    return expiry, strike, kind


def book_summary_to_chain(
    summary: list, index_price: float, *, asof: date, symbol: str = "BTC",
    r: float = 0.0, q: float = 0.0,
) -> OptionChain:
    """Map a `get_book_summary_by_currency` response into an OptionChain.

    PURE (no network) so it is unit-testable. `summary` is the API's ``result``
    list; each item has instrument_name, bid_price, ask_price (COIN units),
    underlying_price. Premiums are converted to USD here (x index_price).
    """
    quotes: list[OptionQuote] = []
    for item in summary:
        parsed = parse_instrument_name(item.get("instrument_name", ""))
        if parsed is None:
            continue
        expiry, strike, kind = parsed
        bid_coin = item.get("bid_price")
        ask_coin = item.get("ask_price")
        if bid_coin is None or ask_coin is None:
            continue                      # stale/empty quote -> skip (starves rnd otherwise)
        expiry_days = (expiry - asof).days
        if expiry_days <= 0:
            continue
        # COIN -> USD. underlying_price per instrument is preferred; fall back to index.
        px = item.get("underlying_price") or index_price
        quotes.append(OptionQuote(
            expiry_days=expiry_days,
            strike=strike,
            kind=kind,
            bid=round(bid_coin * px, 4),
            ask=round(ask_coin * px, 4),
        ))
    return OptionChain(symbol=symbol, spot=index_price, r=r, q=q,
                       quotes=quotes, asof=asof.isoformat())


def save_chain_json(chain: OptionChain, path: str) -> None:
    """Dump to the JsonFileAdapter schema — freeze a live pull as a CI fixture."""
    obj = {
        "symbol": chain.symbol, "spot": chain.spot, "r": chain.r, "q": chain.q,
        "asof": chain.asof,
        "quotes": [{"expiry_days": qt.expiry_days, "strike": qt.strike,
                    "kind": qt.kind, "bid": qt.bid, "ask": qt.ask}
                   for qt in chain.quotes],
    }
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


# --------------------------------------------------------------------------- #
# Live adapter (needs outbound network to deribit.com)
# --------------------------------------------------------------------------- #
def _get(base: str, method: str, **params) -> dict:
    url = f"{base}/public/{method}?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "options-alpha-engine"})
    with urllib.request.urlopen(req, timeout=20) as resp:   # honours HTTPS_PROXY env
        payload = json.load(resp)
    if "error" in payload and payload["error"]:
        raise RuntimeError(f"Deribit error for {method}: {payload['error']}")
    return payload["result"]


class DeribitAdapter:
    """MarketDataAdapter over Deribit's keyless public API (mainnet or testnet)."""

    def __init__(self, currency: str = "BTC", base: str = MAINNET):
        self.currency = currency.upper()
        self.base = base

    def index_price(self) -> float:
        return _get(self.base, "get_index_price",
                    index_name=f"{self.currency.lower()}_usd")["index_price"]

    def option_chain(self, symbol: str | None = None) -> OptionChain:
        cur = (symbol or self.currency).upper()
        summary = _get(self.base, "get_book_summary_by_currency", currency=cur, kind="option")
        spot = self.index_price()
        asof = datetime.now(timezone.utc).date()
        return book_summary_to_chain(summary, spot, asof=asof, symbol=cur)

    def price_history(self, symbol: str | None = None, days: int = 400) -> list[float]:
        cur = (symbol or self.currency).upper()
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        start_ms = now_ms - days * 86_400_000
        res = _get(self.base, "get_tradingview_chart_data",
                   instrument_name=f"{cur}-PERPETUAL", resolution="1D",
                   start_timestamp=start_ms, end_timestamp=now_ms)
        return [float(c) for c in res.get("close", [])]


def main(argv=None):
    import sys
    argv = argv if argv is not None else sys.argv[1:]
    currency = argv[0] if argv else "BTC"
    dte = int(argv[1]) if len(argv) > 1 else 30
    out = argv[2] if len(argv) > 2 else f"{currency.lower()}_chain.json"

    from models import edge, rnd
    from models.baseline import BaselineDensityForecaster

    ad = DeribitAdapter(currency)
    chain = ad.option_chain()
    save_chain_json(chain, out)
    print(f"pulled {len(chain.quotes)} quotes; spot={chain.spot:,.0f}; saved -> {out}")

    T = dte / 365.0
    print(f"model-free Q vol ({dte}d): {rnd.model_free_implied_vol(chain, T, dte):.1%}"
          "   (cross-check vs Deribit DVOL)")
    m = rnd.bkm_moments(chain, T, dte)
    print(f"BKM: vol={m.vol:.1%} skew={m.skew:+.2f} exkurt={m.kurtosis:+.2f} "
          f"coverage_ok={m.coverage_ok}")
    prices = ad.price_history(days=400)
    for s in edge.scan_distribution(chain, BaselineDensityForecaster(), prices,
                                    actionable_only=False)[:5]:
        print(f"  {s.expiry_days:>4}d  P={s.p_vol:.1%} Q={s.q_vol:.1%} "
              f"VRP={s.variance_risk_premium:+.4f}  {s.verdict}")


if __name__ == "__main__":
    main()
