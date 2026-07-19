"""Tradier bridge — free-key EUROPEAN equity/index options + paper execution.

Tradier's Sandbox gives a FREE token (no brokerage account) that unlocks three
things this engine wants in one venue:
  * SPX / XSP option CHAINS — these are EUROPEAN cash-settled index options, so
    models/rnd.py (BKM / VIX / Breeden-Litzenberger) is valid with NO
    de-Americanizing, and quotes are already in USD (no coin conversion).
  * daily price HISTORY (/markets/history) for the hedged backtest + gate.
  * PAPER ORDER routing (/accounts/{id}/orders) — the execution/sandbox lens.

Caveats (from review): sandbox quotes are ~15-min delayed and paper fills never
touch a real book — fine for plumbing, order lifecycle, and the promotion gate,
useless for latency/fill realism (that needs IBKR). One chain call is billed PER
EXPIRATION, so `option_chain` fetches only the nearest ``max_expirations``.

Set TRADIER_TOKEN (sandbox token). Base defaults to the sandbox host. Pure
stdlib. This sandbox blocks tradier.com; run where egress is open. The parse
helpers are pure and unit-tested offline.
"""

from __future__ import annotations

import datetime
import json
import os
import urllib.parse
import urllib.request

from engine.data import OptionChain, OptionQuote

SANDBOX = "https://sandbox.tradier.com/v1"
PROD = "https://api.tradier.com/v1"


def _as_list(x):
    """Tradier returns a dict for one element, a list for many, None for zero."""
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


def _days_to_expiry(expiration: str, asof: str) -> int:
    e = datetime.date.fromisoformat(expiration)
    a = datetime.date.fromisoformat(asof)
    return (e - a).days


def parse_chain(options: list, *, symbol: str, spot: float, asof: str,
                r: float = 0.04, q: float = 0.0) -> OptionChain:
    """Map a /markets/options/chains ``options.option`` list -> OptionChain (PURE).

    Skips strikes with a missing/zero bid or ask (stale quotes starve rnd).
    """
    quotes: list[OptionQuote] = []
    for o in _as_list(options):
        bid, ask = o.get("bid"), o.get("ask")
        if not bid or not ask or bid <= 0 or ask <= 0:
            continue
        kind = o.get("option_type")            # 'call' / 'put'
        if kind not in ("call", "put"):
            continue
        dte = _days_to_expiry(o["expiration_date"], asof)
        if dte <= 0:
            continue
        quotes.append(OptionQuote(expiry_days=dte, strike=float(o["strike"]),
                                  kind=kind, bid=float(bid), ask=float(ask)))
    return OptionChain(symbol=symbol, spot=spot, r=r, q=q, quotes=quotes, asof=asof)


def parse_history(history: dict) -> list[float]:
    """Map /markets/history ``history.day`` -> list[float] closes, oldest first."""
    days = _as_list(history.get("day")) if history else []
    return [float(d["close"]) for d in days if d.get("close") is not None]


def build_order_payload(*, option_symbol: str, side: str, quantity: int,
                        underlying: str, order_type: str = "market",
                        duration: str = "day", price: float | None = None) -> dict:
    """OCC-symbol option order payload for POST /accounts/{id}/orders (PURE).

    side in {buy_to_open, sell_to_open, buy_to_close, sell_to_close}.
    """
    if side not in ("buy_to_open", "sell_to_open", "buy_to_close", "sell_to_close"):
        raise ValueError(f"bad side {side!r}")
    payload = {"class": "option", "symbol": underlying, "option_symbol": option_symbol,
               "side": side, "quantity": str(quantity), "type": order_type,
               "duration": duration}
    if order_type in ("limit", "stop_limit") and price is not None:
        payload["price"] = str(price)
    return payload


class TradierAdapter:
    """MarketDataAdapter over Tradier (sandbox by default) + paper execution."""

    def __init__(self, token: str | None = None, *, base: str = SANDBOX,
                 r: float = 0.04, q: float = 0.0, max_expirations: int = 3,
                 timeout: float = 15.0):
        self._token = token or os.environ.get("TRADIER_TOKEN")
        self.base = base
        self.r, self.q = r, q
        self.max_expirations = max_expirations
        self.timeout = timeout

    def _require_token(self):
        if not self._token:
            raise RuntimeError("Tradier needs a token: set TRADIER_TOKEN "
                               "(free sandbox token from developer.tradier.com)")

    def _get(self, path: str, **params) -> dict:
        self._require_token()
        url = f"{self.base}{path}?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={
            "Authorization": f"Bearer {self._token}", "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.load(resp)

    def _spot(self, symbol: str) -> float:
        q = self._get("/markets/quotes", symbols=symbol)["quotes"]["quote"]
        q = _as_list(q)[0]
        return float(q.get("last") or q.get("close"))

    def option_chain(self, symbol: str) -> OptionChain:
        exps = _as_list(self._get("/markets/options/expirations", symbol=symbol,
                                  includeAllRoots="true")["expirations"].get("date"))
        asof = datetime.date.today().isoformat()
        spot = self._spot(symbol)
        quotes: list[OptionQuote] = []
        for exp in exps[:self.max_expirations]:      # billed per expiration -> budget it
            data = self._get("/markets/options/chains", symbol=symbol,
                             expiration=exp, greeks="true")
            opts = (data.get("options") or {}).get("option")
            quotes.extend(parse_chain(opts, symbol=symbol, spot=spot, asof=asof,
                                      r=self.r, q=self.q).quotes)
        return OptionChain(symbol=symbol, spot=spot, r=self.r, q=self.q,
                           quotes=quotes, asof=asof)

    def price_history(self, symbol: str, days: int = 400) -> list[float]:
        start = (datetime.date.today() - datetime.timedelta(days=int(days * 1.5))).isoformat()
        h = self._get("/markets/history", symbol=symbol, interval="daily", start=start)
        closes = parse_history(h.get("history") or {})
        return closes[-days:] if days else closes

    def place_paper_order(self, account_id: str, payload: dict) -> dict:
        """POST a paper order to the sandbox (execution lens). Live-only."""
        self._require_token()
        url = f"{self.base}/accounts/{account_id}/orders"
        data = urllib.parse.urlencode(payload).encode()
        req = urllib.request.Request(url, data=data, headers={
            "Authorization": f"Bearer {self._token}", "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.load(resp)
