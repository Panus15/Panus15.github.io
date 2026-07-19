"""Interactive Brokers bridge — the ONLY source with real latency + real fills.

Why IBKR closes a lens no other venue in this engine can:
  * Tradier / Alpaca sandboxes are ~15-min DELAYED and fill against a fake book —
    fine for order-lifecycle plumbing, useless for execution realism.
  * IBKR's PAPER account routes through the same matching path as live: real
    latency, real (paper) FILLS, real queue behaviour. That is the
    execution-realism lens of the roadmap.

This adapter targets the **Client Portal Web API** — the REST/JSON gateway you
can reach with urllib — NOT the TWS socket API. The gateway is a small Java
process the USER runs and AUTHENTICATES locally:

    1. Download the Client Portal Gateway from IBKR.
    2. `bin/run.sh root/conf.yaml`  (Windows: `bin\\run.bat`).
    3. Browse to https://localhost:5000 and log in (2FA). The session stays
       alive while the gateway keeps-alive pings; re-auth when it drops.

Because that login is interactive and the gateway is local, THIS adapter cannot
be exercised live in CI (and this sandbox blocks network besides). The TESTED
surface is therefore the PURE parse helpers (`parse_history`,
`parse_snapshot_chain`, `build_order_payload`, `_days_to_expiry`) — see
tests/test_ibkr.py. The live methods are structured one-helper-per-endpoint and
documented, but only run against a real, authenticated local gateway.

TLS note: the gateway serves a SELF-SIGNED cert on localhost. We build an SSL
context that skips verification ONLY when the host is loopback
(localhost/127.0.0.1/::1) — never for a remote host. See `_ssl_context`.

Instrument choice: prefer SPX (or another cash-settled INDEX) in docs and tests.
SPX options are EUROPEAN, so models/rnd.py (BKM / VIX-style / Breeden-Litzenberger)
is valid with NO de-Americanizing, and IBKR quotes are already in USD — there is
no coin conversion to get wrong (unlike Deribit).

Client Portal endpoints used (all under the `base` = /v1/api):
    GET  /iserver/secdef/search?symbol=SPX                 -> underlying conid
    GET  /iserver/secdef/strikes?conid=&sectype=OPT&month= -> {call:[],put:[]}
    GET  /iserver/secdef/info?conid=&sectype=OPT&month=&strike=&right=C|P
                                                           -> option contract(s)
    GET  /iserver/marketdata/snapshot?conids=&fields=31,84,86
                                                           -> last / bid / ask
    GET  /iserver/marketdata/history?conid=&period=&bar=1d -> daily bars
    POST /iserver/account/{accountId}/orders               -> paper order (fills)

Pure stdlib only (urllib, json, ssl, datetime, os).
"""

from __future__ import annotations

import datetime
import json
import os
import ssl
import urllib.error
import urllib.parse
import urllib.request

from engine.data import OptionChain, OptionQuote

DEFAULT_BASE = "https://localhost:5000/v1/api"

# IBKR market-data snapshot field numbers (returned as STRING keys in JSON).
FIELD_LAST = "31"
FIELD_BID = "84"
FIELD_ASK = "86"
SNAPSHOT_FIELDS = f"{FIELD_LAST},{FIELD_BID},{FIELD_ASK}"

_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}


# --------------------------------------------------------------------------- #
# Pure helpers (OFFLINE-testable — this is the surface tests/test_ibkr.py hits)
# --------------------------------------------------------------------------- #
def _num(v):
    """Best-effort float. Accepts numbers or IBKR marketdata strings.

    IBKR sometimes decorates a price string with a leading marker letter
    (e.g. ``'C123.4'`` = prior close, ``'H'``/``'L'`` = session hi/lo). We keep
    the numeric tail. Returns None for anything unparseable / empty / bool.
    """
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if not s:
        return None
    j = 0
    while j < len(s) and s[j] not in "0123456789+-.":
        j += 1
    s = s[j:]
    try:
        return float(s)
    except ValueError:
        return None


def _normalize_kind(v):
    """'C'/'call' -> 'call', 'P'/'put' -> 'put', else None."""
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in ("c", "call"):
        return "call"
    if s in ("p", "put"):
        return "put"
    return None


def _parse_date(v) -> datetime.date:
    """Accept a date/datetime, IBKR 'YYYYMMDD', or ISO 'YYYY-MM-DD'."""
    if isinstance(v, datetime.datetime):
        return v.date()
    if isinstance(v, datetime.date):
        return v
    s = str(v).strip()
    if len(s) == 8 and s.isdigit():          # IBKR maturityDate form
        return datetime.date(int(s[:4]), int(s[4:6]), int(s[6:8]))
    return datetime.date.fromisoformat(s)    # ISO form


def _days_to_expiry(expiry, asof) -> int:
    """Calendar days from ``asof`` to ``expiry``. Both accept YYYYMMDD or ISO."""
    return (_parse_date(expiry) - _parse_date(asof)).days


def parse_history(history_json) -> list[float]:
    """Map an /iserver/marketdata/history response into closes, OLDEST FIRST.

    Shape: ``{"data": [{"o":.., "c": <close>, "h":.., "l":.., "t": <ms epoch>}, ...]}``.
    We read each bar's ``c`` and order by ``t`` so the result is chronological
    regardless of the server's ordering (rnd/backtests want oldest-first). Bars
    missing a numeric close are skipped.
    """
    bars = (history_json or {}).get("data") or []
    rows = []
    for i, b in enumerate(bars):
        if not isinstance(b, dict):
            continue
        c = _num(b.get("c"))
        if c is None:
            continue
        t = b.get("t")
        rows.append((t if t is not None else i, c))
    rows.sort(key=lambda r: r[0])
    return [c for _, c in rows]


def _pick(snap: dict, info: dict, *keys):
    """First non-None value for any of ``keys``, snapshot taking precedence."""
    for src in (snap, info):
        if not src:
            continue
        for k in keys:
            v = src.get(k)
            if v is not None:
                return v
    return None


def parse_snapshot_chain(snapshots, meta, *, symbol, spot, asof,
                         r: float = 0.04, q: float = 0.0) -> OptionChain:
    """Map IBKR option snapshots into an OptionChain (PURE, no network).

    Snapshot dict shape (this adapter's chosen contract)
    ----------------------------------------------------
    Each element of ``snapshots`` is one option's marketdata snapshot. IBKR's
    /iserver/marketdata/snapshot returns FIELD-NUMBER string keys and a conid,
    but NOT the contract terms — those come from the secdef steps. So we accept
    the CONTRACT TERMS either inline on the snapshot OR joined in from ``meta``:

        {
          "conid": 265598,        # IBKR contract id (join key into `meta`)
          "84": 12.3,             # BID   (field 84)  -- or "bid": 12.3
          "86": 12.5,             # ASK   (field 86)  -- or "ask": 12.5
          "31": 12.4,             # LAST  (field 31, optional/unused)
          # contract terms, inline OR supplied via meta[conid]:
          "strike": 5000.0,
          "right":  "C",          # or "kind": "call"
          "expiry": "20260818",   # YYYYMMDD or ISO; or "maturityDate"
        }

    ``meta`` is a dict ``{conid: {strike, right/kind, expiry}}`` — the contract
    metadata gathered from /iserver/secdef/info — merged in when the snapshot
    carries only a conid + price fields. Pass ``{}`` if every snapshot is
    self-contained.

    Both field-number keys ('84'/'86') and named keys ('bid'/'ask') are honored.
    A quote is SKIPPED when strike/right/expiry can't be resolved, when the
    expiry is already past, or when bid or ask is missing / zero (a stale quote
    starves the rnd integrals and biases the recovered vol low).
    """
    meta = meta or {}
    asof_iso = asof.isoformat() if isinstance(asof, datetime.date) else str(asof)
    quotes: list[OptionQuote] = []
    for snap in snapshots:
        conid = snap.get("conid")
        info = meta.get(conid, {}) if conid is not None else {}

        strike = _pick(snap, info, "strike")
        kind = _normalize_kind(_pick(snap, info, "right", "kind"))
        expiry = _pick(snap, info, "expiry", "maturityDate", "expiration")
        if strike is None or kind is None or expiry is None:
            continue

        bid = _num(_pick(snap, info, FIELD_BID, "bid"))
        ask = _num(_pick(snap, info, FIELD_ASK, "ask"))
        if not bid or not ask or bid <= 0 or ask <= 0:
            continue

        dte = _days_to_expiry(expiry, asof)
        if dte <= 0:
            continue

        quotes.append(OptionQuote(expiry_days=dte, strike=float(strike),
                                  kind=kind, bid=float(bid), ask=float(ask)))
    return OptionChain(symbol=symbol, spot=spot, r=r, q=q, quotes=quotes,
                       asof=asof_iso)


def build_order_payload(conid, side, quantity, order_type: str = "MKT",
                        price=None) -> dict:
    """Build ONE order dict for POST /iserver/account/{id}/orders (PURE).

    ``side`` in {'BUY','SELL'} (case-insensitive). ``order_type`` 'MKT' or 'LMT'
    (a limit order requires ``price``). The endpoint expects an envelope
    ``{"orders": [ <this dict> ]}`` — `place_paper_order` wraps it for you.
    """
    s = str(side).upper()
    if s not in ("BUY", "SELL"):
        raise ValueError(f"bad side {side!r}; expected 'BUY' or 'SELL'")
    otype = str(order_type).upper()
    payload = {
        "conid": int(conid),
        "orderType": otype,
        "side": s,
        "quantity": int(quantity),
        "tif": "DAY",
    }
    if otype == "LMT":
        if price is None:
            raise ValueError("limit order ('LMT') requires a price")
        payload["price"] = float(price)
    return payload


# --------------------------------------------------------------------------- #
# Live adapter (needs a running + AUTHENTICATED local Client Portal gateway)
# --------------------------------------------------------------------------- #
class IBKRAdapter:
    """MarketDataAdapter over IBKR's Client Portal Web API + paper execution.

    Live methods require the local gateway to be running and logged in; see the
    module docstring. All network access flows through ``_open``, which turns an
    unreachable / un-authenticated gateway into a CLEAR RuntimeError.
    """

    def __init__(self, base: str = DEFAULT_BASE, account_id: str | None = None,
                 r: float = 0.04, q: float = 0.0, timeout: float = 15.0):
        self.base = base.rstrip("/")
        self.account_id = account_id or os.environ.get("IBKR_ACCOUNT_ID")
        self.r, self.q = r, q
        self.timeout = timeout

    # -- transport ---------------------------------------------------------- #
    def _ssl_context(self):
        """Unverified TLS for LOOPBACK ONLY (self-signed gateway cert).

        The Client Portal gateway presents a self-signed certificate on
        localhost, which the default context rejects. We disable verification
        *only* when the host is loopback; any remote host keeps full
        verification (returns None -> urllib's default verifying context).
        """
        host = (urllib.parse.urlparse(self.base).hostname or "").lower()
        if host in _LOOPBACK_HOSTS:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            return ctx
        return None

    def _open(self, req):
        try:
            with urllib.request.urlopen(req, timeout=self.timeout,
                                        context=self._ssl_context()) as resp:
                return json.load(resp)
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"IBKR Client Portal gateway unreachable at {self.base!r}: {e}. "
                "Start the gateway (bin/run.sh) and authenticate at "
                "https://localhost:5000 before using the live adapter."
            ) from e

    def _get(self, path: str, **params) -> object:
        url = f"{self.base}{path}"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={
            "Accept": "application/json", "User-Agent": "options-alpha-engine"})
        return self._open(req)

    def _post(self, path: str, body: dict) -> object:
        url = f"{self.base}{path}"
        data = json.dumps(body).encode()
        req = urllib.request.Request(url, data=data, method="POST", headers={
            "Content-Type": "application/json", "Accept": "application/json",
            "User-Agent": "options-alpha-engine"})
        return self._open(req)

    # -- one helper per Client Portal endpoint ------------------------------ #
    def _underlying_conid(self, symbol: str) -> int:
        """GET /iserver/secdef/search?symbol=SPX -> the underlying's conid."""
        res = self._get("/iserver/secdef/search", symbol=symbol)
        if not res:
            raise RuntimeError(f"no IBKR contract found for symbol {symbol!r}")
        return int((res[0] if isinstance(res, list) else res)["conid"])

    def _secdef_strikes(self, conid: int, month: str, sectype: str = "OPT") -> dict:
        """GET /iserver/secdef/strikes -> {'call': [...], 'put': [...]}."""
        return self._get("/iserver/secdef/strikes", conid=conid,
                          sectype=sectype, month=month)

    def _secdef_info(self, conid: int, month: str, strike: float, right: str,
                     sectype: str = "OPT") -> list:
        """GET /iserver/secdef/info -> option contract(s) (each has a conid)."""
        res = self._get("/iserver/secdef/info", conid=conid, sectype=sectype,
                        month=month, strike=strike, right=right)
        return res if isinstance(res, list) else [res]

    def _snapshot(self, conids: list, fields: str = SNAPSHOT_FIELDS) -> list:
        """GET /iserver/marketdata/snapshot?conids=..&fields=31,84,86."""
        res = self._get("/iserver/marketdata/snapshot",
                        conids=",".join(str(c) for c in conids), fields=fields)
        return res if isinstance(res, list) else [res]

    @staticmethod
    def _front_month() -> str:
        """IBKR month token for the current month, e.g. 'AUG26'."""
        return datetime.date.today().strftime("%b%y").upper()

    # -- MarketDataAdapter protocol ----------------------------------------- #
    def price_history(self, conid_or_symbol, days: int = 400) -> list[float]:
        """Daily closes, oldest first, via /iserver/marketdata/history.

        ``conid_or_symbol`` may be a numeric conid or a symbol (resolved through
        /iserver/secdef/search). ``period`` is derived from ``days``; bars are
        daily (``bar=1d``).
        """
        conid = conid_or_symbol
        if not str(conid_or_symbol).isdigit():
            conid = self._underlying_conid(str(conid_or_symbol))
        data = self._get("/iserver/marketdata/history", conid=conid,
                         period=f"{int(days)}d", bar="1d")
        closes = parse_history(data)
        return closes[-days:] if days else closes

    def option_chain(self, symbol: str, month: str | None = None,
                     max_strikes: int | None = None) -> OptionChain:
        """Build an OptionChain via IBKR's multi-step secdef -> snapshot flow.

        Elaborate by nature (four endpoint families) and NOT runnable offline;
        each step is factored into its own helper above. Steps:
          1. secdef/search -> underlying conid (+ its snapshot for spot).
          2. secdef/strikes -> the strike ladder for ``month``.
          3. secdef/info per (strike, right) -> option conid + maturityDate,
             accumulated into ``meta`` keyed by conid.
          4. marketdata/snapshot for every option conid -> bid/ask.
          5. parse_snapshot_chain(snapshots, meta) -> OptionChain.
        """
        month = month or self._front_month()
        und_conid = self._underlying_conid(symbol)

        # spot from the underlying's own snapshot (field 31 = last).
        und_snap = self._snapshot([und_conid], fields=FIELD_LAST)
        spot = _num((und_snap[0] if und_snap else {}).get(FIELD_LAST)) or 0.0

        ladder = self._secdef_strikes(und_conid, month)
        strikes = sorted(set(ladder.get("call", [])) | set(ladder.get("put", [])))
        if max_strikes:
            # keep the strikes nearest spot when the caller wants to budget calls
            strikes = sorted(strikes, key=lambda k: abs(k - spot))[:max_strikes]

        meta: dict = {}
        for strike in strikes:
            for right in ("C", "P"):
                for info in self._secdef_info(und_conid, month, strike, right):
                    oc = int(info["conid"])
                    meta[oc] = {"strike": strike, "right": right,
                                "expiry": info.get("maturityDate")}

        asof = datetime.date.today().isoformat()
        if not meta:
            return OptionChain(symbol=symbol, spot=spot, r=self.r, q=self.q,
                               quotes=[], asof=asof)

        snapshots = self._snapshot(list(meta.keys()))
        return parse_snapshot_chain(snapshots, meta, symbol=symbol, spot=spot,
                                    asof=asof, r=self.r, q=self.q)

    # -- execution lens: real latency + real paper fills -------------------- #
    def place_paper_order(self, payload: dict, account_id: str | None = None) -> object:
        """POST a paper order to /iserver/account/{accountId}/orders.

        This is the whole point of the IBKR integration: the order routes
        through the real matching path, so latency and fills are realistic.
        ``payload`` may be a single order dict (from `build_order_payload`) or a
        pre-wrapped ``{"orders": [...]}`` envelope. Raises a clear error if no
        account_id is available (constructor arg or IBKR_ACCOUNT_ID env).
        """
        acct = account_id or self.account_id
        if not acct:
            raise RuntimeError(
                "IBKR order needs an account_id: pass IBKRAdapter(account_id=..) "
                "or set IBKR_ACCOUNT_ID (your paper account, e.g. 'DU1234567').")
        body = payload if isinstance(payload, dict) and "orders" in payload \
            else {"orders": [payload]}
        return self._post(f"/iserver/account/{acct}/orders", body)


if __name__ == "__main__":  # pragma: no cover - offline-safe usage note only
    print(__doc__)
    print("This adapter needs a running, authenticated local Client Portal "
          "gateway; it makes no network call when imported.")
