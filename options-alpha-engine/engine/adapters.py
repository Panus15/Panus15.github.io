"""Real & file-based market-data adapters (drop-in for ``SyntheticAdapter``).

Every adapter here implements the engine's ``MarketDataAdapter`` protocol
(``engine/data.py``) and returns the engine's native ``OptionChain`` /
``OptionQuote`` objects, so switching from synthetic to real data — or from a
live vendor to a replayed file snapshot — requires **no downstream change**.
The Q-extractors (``models.rnd``), the P/Q edge scanner (``models.edge``), the
pricer and the backtester all consume these objects unchanged.

Adapters
--------
``PolygonAdapter``    live polygon.io v2 aggregates + v3 options snapshot
``ORATSAdapter``      live ORATS ``datav2`` strikes + historical dailies
``CsvFileAdapter``    OFFLINE — price history CSV + option-chain CSV (tested path)
``JsonFileAdapter``   OFFLINE — price history + option-chain JSON  (tested path)

The two file adapters need no network and no API key; they let you replay a
historical, timestamped snapshot deterministically and are the path exercised by
``tests/test_adapters.py``. The two vendor adapters share a tiny ``_get_json``
helper built on ``urllib`` (stdlib only — it honours ``HTTPS_PROXY`` / env
proxies automatically), read their credential lazily, and raise a clear,
actionable error only when a network method is actually called without a key —
importing this module never touches the network and never crashes.

HOW TO GO LIVE
--------------
1. Set the vendor credential in the environment:
       export POLYGON_API_KEY=...        # for PolygonAdapter
       export ORATS_TOKEN=...            # for ORATSAdapter
2. Swap the adapter at the single construction site, e.g.::

       # was:  adapter = SyntheticAdapter()
       adapter = PolygonAdapter()        # or ORATSAdapter(), or a file adapter

   Nothing else changes: ``adapter.price_history(sym, days)`` still returns a
   ``list[float]`` (oldest->newest) and ``adapter.option_chain(sym)`` still
   returns an ``OptionChain`` with ``asof`` populated (the look-ahead contract).
3. For research/backtests with zero vendor dependency, dump a snapshot to CSV or
   JSON once and replay it forever with ``CsvFileAdapter`` / ``JsonFileAdapter``.

FILE SCHEMAS (offline adapters)
-------------------------------
Price history CSV  — header ``date,close``; one row per day; any order (rows are
    sorted ascending by date on load so the returned list is oldest->newest)::

        date,close
        2024-01-02,98.10
        2024-01-03,98.72

Option-chain CSV  — optional ``#``-comment meta lines of ``key=value`` for
    ``symbol,spot,r,q,asof`` (constructor kwargs are the fallback), followed by a
    header ``expiry_days,strike,kind,bid,ask`` and one quote per row::

        # symbol=SPY
        # spot=100.0
        # r=0.03
        # q=0.0
        # asof=2024-01-15
        expiry_days,strike,kind,bid,ask
        30,100,call,2.35,2.47
        30,100,put,2.11,2.22

Option-chain JSON — a single object::

        {"symbol": "SPY", "spot": 100.0, "r": 0.03, "q": 0.0,
         "asof": "2024-01-15",
         "quotes": [{"expiry_days": 30, "strike": 100, "kind": "call",
                     "bid": 2.35, "ask": 2.47}, ...]}

Price history JSON — either a bare list of closes, a list of
    ``{"date": ..., "close": ...}`` objects, or ``{"closes": [...]}``.
"""

from __future__ import annotations

import csv
import datetime
import json
import os
import urllib.parse
import urllib.request

from .data import OptionChain, OptionQuote

__all__ = [
    "PolygonAdapter",
    "ORATSAdapter",
    "CsvFileAdapter",
    "JsonFileAdapter",
]

_USER_AGENT = "options-alpha-engine/0.1 (+stdlib-urllib)"


def _today_iso() -> str:
    return datetime.date.today().isoformat()


def _days_to_expiry(expiration: str, asof: str | None = None) -> int:
    """Calendar days from ``asof`` (default today) to an ISO expiration date."""
    exp = datetime.date.fromisoformat(str(expiration)[:10])
    ref = datetime.date.fromisoformat(asof) if asof else datetime.date.today()
    return (exp - ref).days


def _get_json(url: str, timeout: float = 15.0) -> dict:
    """Minimal GET -> parsed-JSON helper (stdlib ``urllib``, proxy-friendly).

    ``urllib`` transparently respects ``HTTPS_PROXY`` / ``HTTP_PROXY`` from the
    environment, so this works behind the agent proxy with no extra config.
    Network / HTTP / decode failures are re-raised as a ``RuntimeError`` that
    names the URL (host only) so callers get an actionable message instead of a
    raw traceback.
    """
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = resp.read().decode("utf-8")
        return json.loads(payload)
    except Exception as exc:  # urllib.error.URLError, HTTPError, JSONDecodeError...
        host = urllib.parse.urlsplit(url).netloc or url
        raise RuntimeError(f"market-data request to {host} failed: {exc}") from exc


class PolygonAdapter:
    """Live market data from polygon.io (v2 aggregates + v3 options snapshot).

    Credential: environment variable ``POLYGON_API_KEY``. It is read lazily —
    constructing the adapter never touches the network or the environment; the
    key is required (with a clear error) only when a data method is called.

    Rate/dividend note: the options snapshot carries no risk-free rate or
    dividend yield, so ``r`` and ``q`` default to constructor values
    (``r=0.045``, ``q=0.0``) and can be overridden per instance.

    Endpoints
    ---------
    price_history -> GET /v2/aggs/ticker/{sym}/range/1/day/{from}/{to}
        Daily OHLCV aggregates. ``sort=asc`` -> oldest->newest; close is ``c``.
        https://polygon.io/docs/rest/stocks/aggregates/custom-bars
    option_chain  -> GET /v3/snapshot/options/{underlyingAsset}
        Full options snapshot, paginated via ``next_url``. Per contract:
        ``details.{expiration_date,strike_price,contract_type}`` and
        ``last_quote.{bid,ask}``; underlying spot from ``underlying_asset.price``.
        https://polygon.io/docs/rest/options/snapshots/option-chain-snapshot
    """

    BASE = "https://api.polygon.io"

    def __init__(self, api_key: str | None = None, *, r: float = 0.045,
                 q: float = 0.0, timeout: float = 15.0):
        self._api_key = api_key            # None -> read from env lazily
        self.r = r
        self.q = q
        self.timeout = timeout

    def _key(self) -> str:
        key = self._api_key or os.environ.get("POLYGON_API_KEY")
        if not key:
            raise RuntimeError(
                "PolygonAdapter needs a polygon.io API key. Set the "
                "POLYGON_API_KEY environment variable (or pass api_key=...) "
                "before calling price_history()/option_chain()."
            )
        return key

    def price_history(self, symbol: str, days: int = 260) -> list[float]:
        """Daily close prices, oldest->newest, for the last ``days`` calendar days."""
        key = self._key()
        end = datetime.date.today()
        start = end - datetime.timedelta(days=max(days, 1))
        path = (f"/v2/aggs/ticker/{urllib.parse.quote(symbol)}/range/1/day/"
                f"{start.isoformat()}/{end.isoformat()}")
        query = urllib.parse.urlencode(
            {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": key})
        data = _get_json(f"{self.BASE}{path}?{query}", self.timeout)
        results = data.get("results") or []
        closes = [float(bar["c"]) for bar in results if bar.get("c") is not None]
        if not closes:
            raise RuntimeError(f"Polygon returned no aggregates for {symbol!r}")
        return closes

    def option_chain(self, symbol: str) -> OptionChain:
        """Full option-chain snapshot -> OptionChain (asof = today)."""
        key = self._key()
        url = (f"{self.BASE}/v3/snapshot/options/{urllib.parse.quote(symbol)}"
               f"?limit=250&apiKey={urllib.parse.quote(key)}")
        quotes: list[OptionQuote] = []
        spot: float | None = None
        asof = _today_iso()
        pages = 0
        while url and pages < 50:            # hard page cap — defensive
            data = _get_json(url, self.timeout)
            for item in data.get("results") or []:
                details = item.get("details") or {}
                quote = item.get("last_quote") or {}
                exp = details.get("expiration_date")
                strike = details.get("strike_price")
                kind = details.get("contract_type")           # "call" / "put"
                bid, ask = quote.get("bid"), quote.get("ask")
                if None in (exp, strike, kind, bid, ask):
                    continue
                if spot is None:
                    ua = item.get("underlying_asset") or {}
                    if ua.get("price") is not None:
                        spot = float(ua["price"])
                quotes.append(OptionQuote(
                    expiry_days=_days_to_expiry(exp, asof),
                    strike=float(strike),
                    kind=str(kind).lower(),
                    bid=float(bid),
                    ask=float(ask),
                ))
            nxt = data.get("next_url")
            url = f"{nxt}&apiKey={urllib.parse.quote(key)}" if nxt else None
            pages += 1
        if spot is None:
            spot = self.price_history(symbol, 5)[-1]      # fallback to last close
        if not quotes:
            raise RuntimeError(f"Polygon returned no option quotes for {symbol!r}")
        return OptionChain(symbol=symbol, spot=spot, r=self.r, q=self.q,
                           quotes=quotes, asof=asof)


class ORATSAdapter:
    """Live market data from ORATS (``datav2`` strikes + historical dailies).

    Credential: environment variable ``ORATS_TOKEN`` (read lazily, same
    graceful-offline behaviour as PolygonAdapter).

    ORATS is convenient: each ``strikes`` row carries BOTH the call and put
    quote, the underlying spot, the interpolated rate and the dividend yield, so
    ``r``/``q`` are read straight off the data (median across strikes) rather
    than assumed.

    Endpoints
    ---------
    price_history -> GET /datav2/hist/dailies?token=..&ticker=..
        Historical daily underlying; close is ``clsPx`` (fallback ``stockPrice``),
        oldest->newest by ``tradeDate``.
    option_chain  -> GET /datav2/strikes?token=..&ticker=..
        One row per (expiry, strike) with ``callBidPrice/callAskPrice/
        putBidPrice/putAskPrice``, ``dte``, ``spotPrice``, ``iRate``, ``yieldRate``.
        https://docs.orats.io/datav2-api-guide/data.html#strikes
    """

    BASE = "https://api.orats.io/datav2"

    def __init__(self, token: str | None = None, *, timeout: float = 15.0):
        self._token = token
        self.timeout = timeout

    def _tok(self) -> str:
        tok = self._token or os.environ.get("ORATS_TOKEN")
        if not tok:
            raise RuntimeError(
                "ORATSAdapter needs an ORATS token. Set the ORATS_TOKEN "
                "environment variable (or pass token=...) before calling "
                "price_history()/option_chain()."
            )
        return tok

    def price_history(self, symbol: str, days: int = 260) -> list[float]:
        tok = self._tok()
        query = urllib.parse.urlencode({"token": tok, "ticker": symbol})
        data = _get_json(f"{self.BASE}/hist/dailies?{query}", self.timeout)
        rows = data.get("data") or []
        rows.sort(key=lambda rr: rr.get("tradeDate", ""))
        closes = []
        for rr in rows:
            px = rr.get("clsPx", rr.get("stockPrice"))
            if px is not None:
                closes.append(float(px))
        if not closes:
            raise RuntimeError(f"ORATS returned no dailies for {symbol!r}")
        return closes[-max(days, 1):]

    def option_chain(self, symbol: str) -> OptionChain:
        tok = self._tok()
        query = urllib.parse.urlencode({"token": tok, "ticker": symbol})
        data = _get_json(f"{self.BASE}/strikes?{query}", self.timeout)
        rows = data.get("data") or []
        if not rows:
            raise RuntimeError(f"ORATS returned no strikes for {symbol!r}")
        quotes: list[OptionQuote] = []
        spots, rates, yields = [], [], []
        asof = None
        for rr in rows:
            asof = asof or rr.get("tradeDate")
            dte = rr.get("dte")
            strike = rr.get("strike")
            if dte is None or strike is None:
                continue
            if rr.get("spotPrice") is not None:
                spots.append(float(rr["spotPrice"]))
            if rr.get("iRate") is not None:
                rates.append(float(rr["iRate"]))
            if rr.get("yieldRate") is not None:
                yields.append(float(rr["yieldRate"]))
            cb, ca = rr.get("callBidPrice"), rr.get("callAskPrice")
            pb, pa = rr.get("putBidPrice"), rr.get("putAskPrice")
            if cb is not None and ca is not None:
                quotes.append(OptionQuote(int(dte), float(strike), "call",
                                          float(cb), float(ca)))
            if pb is not None and pa is not None:
                quotes.append(OptionQuote(int(dte), float(strike), "put",
                                          float(pb), float(pa)))
        if not quotes:
            raise RuntimeError(f"ORATS returned no usable quotes for {symbol!r}")
        spot = _median(spots) if spots else quotes[len(quotes) // 2].strike
        r = _median(rates) if rates else 0.045
        q = _median(yields) if yields else 0.0
        return OptionChain(symbol=symbol, spot=spot, r=r, q=q, quotes=quotes,
                           asof=asof or _today_iso())


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])


class _FileAdapterBase:
    """Shared meta -> OptionChain assembly for the offline file adapters."""

    def _build_chain(self, symbol, spot, r, q, asof, quote_rows) -> OptionChain:
        quotes = [
            OptionQuote(
                expiry_days=int(row["expiry_days"]),
                strike=float(row["strike"]),
                kind=str(row["kind"]).strip().lower(),
                bid=float(row["bid"]),
                ask=float(row["ask"]),
            )
            for row in quote_rows
        ]
        for qt in quotes:
            if qt.kind not in ("call", "put"):
                raise ValueError(f"kind must be 'call' or 'put', got {qt.kind!r}")
        if spot is None:
            raise ValueError("option chain is missing 'spot' (set it in the file "
                             "meta or pass spot=... to the adapter)")
        return OptionChain(
            symbol=symbol,
            spot=float(spot),
            r=float(r),
            q=float(q),
            quotes=quotes,
            asof=asof or _today_iso(),
        )


class CsvFileAdapter(_FileAdapterBase):
    """OFFLINE adapter backed by CSV files (see module docstring for schema).

    Parameters
    ----------
    price_csv : path to a ``date,close`` CSV (needed for ``price_history``).
    chain_csv : path to an option-chain CSV with optional ``#`` meta lines and a
        ``expiry_days,strike,kind,bid,ask`` body (needed for ``option_chain``).
    symbol/spot/r/q/asof : fallback meta used when the chain CSV omits them.
    """

    def __init__(self, price_csv: str | None = None, chain_csv: str | None = None,
                 *, symbol: str | None = None, spot: float | None = None,
                 r: float = 0.03, q: float = 0.0, asof: str | None = None):
        self.price_csv = price_csv
        self.chain_csv = chain_csv
        self.symbol = symbol
        self.spot = spot
        self.r = r
        self.q = q
        self.asof = asof

    def price_history(self, symbol: str, days: int | None = None) -> list[float]:
        if not self.price_csv:
            raise RuntimeError("CsvFileAdapter has no price_csv configured")
        rows = []
        with open(self.price_csv, newline="") as fh:
            for row in csv.DictReader(fh):
                if row.get("close") in (None, ""):
                    continue
                rows.append((row.get("date", ""), float(row["close"])))
        rows.sort(key=lambda dc: dc[0])                 # ascending date -> oldest first
        closes = [c for _, c in rows]
        if days is not None:
            closes = closes[-days:]
        return closes

    def option_chain(self, symbol: str) -> OptionChain:
        if not self.chain_csv:
            raise RuntimeError("CsvFileAdapter has no chain_csv configured")
        meta = {"symbol": self.symbol or symbol, "spot": self.spot,
                "r": self.r, "q": self.q, "asof": self.asof}
        data_lines = []
        with open(self.chain_csv) as fh:
            for raw in fh:
                line = raw.strip()
                if not line:
                    continue
                if line.startswith("#"):
                    body = line[1:].strip()
                    if "=" in body:
                        k, v = body.split("=", 1)
                        meta[k.strip()] = v.strip()
                    continue
                data_lines.append(raw)
        quote_rows = list(csv.DictReader(data_lines))
        return self._build_chain(meta["symbol"], meta["spot"], meta["r"],
                                 meta["q"], meta["asof"], quote_rows)


class JsonFileAdapter(_FileAdapterBase):
    """OFFLINE adapter backed by JSON files (see module docstring for schema).

    Parameters
    ----------
    price_json : path to a price-history JSON (bare list of closes, a list of
        ``{"date","close"}`` objects, or ``{"closes":[...]}``).
    chain_json : path to an option-chain JSON object with ``symbol,spot,r,q,asof``
        and a ``quotes`` list of ``{expiry_days,strike,kind,bid,ask}`` objects.
    """

    def __init__(self, price_json: str | None = None,
                 chain_json: str | None = None):
        self.price_json = price_json
        self.chain_json = chain_json

    def price_history(self, symbol: str, days: int | None = None) -> list[float]:
        if not self.price_json:
            raise RuntimeError("JsonFileAdapter has no price_json configured")
        with open(self.price_json) as fh:
            payload = json.load(fh)
        if isinstance(payload, dict):
            payload = payload.get("closes", payload.get("prices", []))
        closes = []
        for item in payload:
            if isinstance(item, dict):
                val = item.get("close", item.get("c"))
                if val is not None:
                    closes.append(float(val))
            else:
                closes.append(float(item))
        if days is not None:
            closes = closes[-days:]
        return closes

    def option_chain(self, symbol: str) -> OptionChain:
        if not self.chain_json:
            raise RuntimeError("JsonFileAdapter has no chain_json configured")
        with open(self.chain_json) as fh:
            obj = json.load(fh)
        return self._build_chain(
            obj.get("symbol", symbol),
            obj.get("spot"),
            obj.get("r", 0.03),
            obj.get("q", 0.0),
            obj.get("asof"),
            obj.get("quotes", []),
        )
