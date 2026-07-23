"""Paper-trading forward-test ledger — the bridge from backtest to a real track record.

`tools/run_live.analyze` is a one-shot spot check: it looks at a single chain
snapshot and prints a verdict. That can never answer the only question that
matters — *is the edge real out of sample?* — because a single snapshot has no
future to be graded against.

This module closes that gap. It runs the honest scientific loop of a forecaster:

    RECORD   at each decision date, freeze the physical (P) forecast, the market
             risk-neutral (Q) moments, the edge verdict, and the intended trade —
             using ONLY information available then (no look-ahead).
    SETTLE   once a forecast's horizon has elapsed, grade it against the realised
             terminal price with proper scoring rules, and realise the delta-hedged
             P&L of any trade it triggered.
    REPORT   accumulate: did P beat the market-implied Q density OOS (NLL / CRPS /
             left-tail)?  and did the signal-selected trades actually make money
             net of costs?

The ledger is an append-only JSON-lines file, so you record snapshots live on your
own machine (where market-data egress is open) and settle them later from just a
price series — reproducible, offline, no re-fetch. Everything here is pure stdlib.

Two measures, graded separately and honestly:
  * CALIBRATION (density) — scored on EVERY recorded date, traded or not. The P
    forecast is compared head-to-head with the Q log-normal the market prints. A
    win means our physical density is a better description of what happened than
    the risk-neutral one — necessary, not sufficient, for edge.
  * PROFIT (trades) — scored only on dates the signal actually fired. Delta-hedged,
    cost-inclusive, non-overlapping. This is the equity curve that decides it.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field

from engine.signal_backtest import _atm_iv, _realize_short_straddle
from models import edge, objective, rnd
from models.baseline import BaselineDensityForecaster
from models.density import (LogNormalComponent, MixtureLogNormal,
                            single_lognormal_riskneutral)

CAL_DAYS = 365.0        # calendar convention for the forecast horizon T (matches rnd/edge)
TRAD_DAYS = 252         # trading-bar convention for the realised hold (matches hedged_backtest)


# --------------------------------------------------------------------------- #
# Serialisation of the forecast density (so settlement is offline + reproducible)
# --------------------------------------------------------------------------- #
def _density_to_list(dist: MixtureLogNormal):
    return [[c.weight, c.mu, c.sigma] for c in dist.components]


def _density_from_list(rows) -> MixtureLogNormal:
    return MixtureLogNormal([LogNormalComponent(w, mu, sg) for w, mu, sg in rows])


# --------------------------------------------------------------------------- #
# The ledger
# --------------------------------------------------------------------------- #
@dataclass
class PaperReport:
    n_recorded: int
    n_settled: int
    n_open: int
    # calibration (density) — over ALL settled entries
    p_nll: float
    q_nll: float
    nll_win_rate: float          # fraction of dates P scored better than Q
    p_left_tail: float
    q_left_tail: float
    left_tail_win_rate: float
    # profit (trades) — over settled entries the signal actually traded
    n_trades: int
    total_pnl: float
    trade_hit_rate: float
    pnl_ratio: float             # mean/std per trade (NOT annualised)
    ann_sharpe: float            # approx, assumes non-overlapping trades

    def summary(self) -> str:
        def _f(x):
            return "n/a" if x != x else f"{x:.4f}"
        lines = [
            f"recorded={self.n_recorded}  settled={self.n_settled}  open={self.n_open}",
            "",
            "CALIBRATION  (physical P vs market-implied Q, out of sample)",
            f"  mean NLL        P={_f(self.p_nll)}  Q={_f(self.q_nll)}  "
            f"-> P wins {self.nll_win_rate:.0%} of dates",
            f"  left-tail loss  P={_f(self.p_left_tail)}  Q={_f(self.q_left_tail)}  "
            f"-> P wins {self.left_tail_win_rate:.0%} of dates",
            "",
            "PROFIT  (signal-selected, delta-hedged, cost-inclusive)",
            f"  trades={self.n_trades}  total P&L={self.total_pnl:+,.2f}  "
            f"hit-rate={self.trade_hit_rate:.0%}",
            f"  per-trade ratio={_f(self.pnl_ratio)}  ~annualised Sharpe={_f(self.ann_sharpe)}",
        ]
        verdict = self._verdict()
        lines += ["", verdict]
        return "\n".join(lines)

    def _verdict(self) -> str:
        if self.n_settled == 0:
            return "VERDICT: nothing matured yet — keep recording."
        beats_q = self.p_nll < self.q_nll
        profitable = self.n_trades > 0 and self.total_pnl > 0
        if beats_q and profitable:
            return ("VERDICT: P beats Q OOS *and* trades are net-positive — the "
                    "strongest evidence a forward test can give (widen the sample).")
        if beats_q and self.n_trades == 0:
            return ("VERDICT: P calibrates better than Q, but the signal never fired "
                    "in-sample — no profit claim yet.")
        if beats_q:
            return "VERDICT: P calibrates better than Q, but trades did not profit — costs/timing."
        return "VERDICT: P does NOT beat the market's Q density OOS — no edge shown yet."


@dataclass
class PaperLedger:
    """Append-only forward-test ledger. One dict per recorded decision date."""
    entries: list = field(default_factory=list)
    r: float = 0.03
    q: float = 0.0
    hedge_bps: float = 5e-4
    spread_frac: float = 0.015

    # --- record --------------------------------------------------------------
    def record(self, chain, prices, entry_index: int, forecaster, *,
               dte: int = 30, asof: str | None = None, min_vrp: float = 0.0,
               news_feature=None):
        """Freeze one decision. Sees ONLY prices[:entry_index+1] — no look-ahead.

        ``news_feature`` (models.sentiment.SentimentFeature) composes the
        FORWARD-looking news gate with the price regime gate via
        models.news_signal.event_risk — a burst of negative/dispersed news
        vetoes the sell before the realised-vol spike shows up in prices.

        Returns the entry dict (appended), or None if the chain can't yield Q
        moments at this expiry (nothing to grade).
        """
        if entry_index >= len(prices):
            raise ValueError("entry_index beyond price series")
        trailing = prices[:entry_index + 1]
        spot = chain.spot
        T = dte / CAL_DAYS
        try:
            q_vol = rnd.model_free_implied_vol(chain, T, dte)
            q_full = rnd.bkm_moments(chain, T, dte)
        except ValueError:
            return None

        p = forecaster.forecast(trailing, T, r=self.r, q=self.q, spot=spot)
        p_vol = p.log_return_vol(spot, T)
        vrp = q_vol ** 2 - p_vol ** 2
        stressed = edge.regime_stressed(trailing)
        gate_reason = None
        if news_feature is not None:
            from models.news_signal import event_risk
            stressed, gate_reason = event_risk(stressed, news_feature)
        iv, strike = _atm_iv(chain, dte, T)

        # The strategy: sell vol only when the market's Q variance is richer than
        # our P forecast AND the regime is calm AND we have an ATM to sell.
        traded = bool(vrp > min_vrp and not stressed and iv is not None and strike is not None
                      and q_full.coverage_ok)

        entry = {
            "id": len(self.entries),
            "status": "open",
            "symbol": chain.symbol,
            "asof": asof if asof is not None else chain.asof,
            "entry_index": entry_index,
            "entry_spot": spot,
            "dte": dte,
            "r": self.r,
            "q": self.q,
            "hedge_bps": self.hedge_bps,     # frozen per-entry so save->load settles
            "spread_frac": self.spread_frac,  # with the SAME costs, not defaults
            "q_vol": q_vol,
            "q_skew": q_full.skew,
            "coverage_ok": q_full.coverage_ok,
            "p_vol": p_vol,
            "vrp": vrp,
            "regime_stressed": stressed,
            "gate_reason": gate_reason,
            "strike": strike,
            "entry_iv": iv,
            "traded": traded,
            "p_density": _density_to_list(p),
        }
        self.entries.append(entry)
        return entry

    # --- settle --------------------------------------------------------------
    def settle(self, prices) -> int:
        """Grade every matured, still-open entry against ``prices``. Idempotent:
        already-settled entries are untouched; immature ones stay open. Returns
        the number newly settled."""
        n_new = 0
        for e in self.entries:
            if e["status"] != "open":
                continue
            mat = e["entry_index"] + e["dte"]
            if mat >= len(prices):
                continue                                  # not matured yet
            T = e["dte"] / CAL_DAYS
            s_real = prices[mat]
            p = _density_from_list(e["p_density"])
            q_dist = single_lognormal_riskneutral(
                e["entry_spot"], T, e["r"], e["q"], e["q_vol"])

            e["s_realized"] = s_real
            e["p_nll"] = objective.nll(p, s_real)
            e["q_nll"] = objective.nll(q_dist, s_real)
            e["p_crps"] = objective.crps(p, s_real)
            e["q_crps"] = objective.crps(q_dist, s_real)
            e["p_left_tail"] = objective.left_tail_pinball(p, s_real)
            e["q_left_tail"] = objective.left_tail_pinball(q_dist, s_real)

            if e["traded"]:
                day_pnl = _realize_short_straddle(
                    prices, e["entry_index"], e["dte"], e["strike"], e["entry_iv"],
                    e["r"], e["q"], e.get("hedge_bps", self.hedge_bps),
                    e.get("spread_frac", self.spread_frac))
                e["trade_pnl"] = sum(day_pnl.values())
            else:
                e["trade_pnl"] = None

            e["status"] = "settled"
            n_new += 1
        return n_new

    # --- report --------------------------------------------------------------
    def report(self) -> PaperReport:
        settled = [e for e in self.entries if e["status"] == "settled"]
        n_open = sum(1 for e in self.entries if e["status"] == "open")
        nan = float("nan")
        if not settled:
            return PaperReport(len(self.entries), 0, n_open, nan, nan, nan,
                               nan, nan, nan, 0, 0.0, nan, nan, nan)

        p_nll = _mean(e["p_nll"] for e in settled)
        q_nll = _mean(e["q_nll"] for e in settled)
        nll_win = _mean(1.0 if e["p_nll"] < e["q_nll"] else 0.0 for e in settled)
        p_lt = _mean(e["p_left_tail"] for e in settled)
        q_lt = _mean(e["q_left_tail"] for e in settled)
        lt_win = _mean(1.0 if e["p_left_tail"] < e["q_left_tail"] else 0.0 for e in settled)

        pnls = [e["trade_pnl"] for e in settled if e["traded"] and e["trade_pnl"] is not None]
        n_tr = len(pnls)
        total = sum(pnls)
        hit = _mean(1.0 if x > 0 else 0.0 for x in pnls) if pnls else nan
        ratio, ann = _pnl_ratios(pnls, settled[0]["dte"] if settled else 21)

        return PaperReport(
            n_recorded=len(self.entries), n_settled=len(settled), n_open=n_open,
            p_nll=p_nll, q_nll=q_nll, nll_win_rate=nll_win,
            p_left_tail=p_lt, q_left_tail=q_lt, left_tail_win_rate=lt_win,
            n_trades=n_tr, total_pnl=total, trade_hit_rate=hit,
            pnl_ratio=ratio, ann_sharpe=ann,
        )

    # --- persistence ---------------------------------------------------------
    def save(self, path: str):
        with open(path, "w") as fh:
            for e in self.entries:
                fh.write(json.dumps(e) + "\n")

    @classmethod
    def load(cls, path: str, **kw) -> "PaperLedger":
        entries = []
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return cls(entries=entries, **kw)


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def _pnl_ratios(pnls, dte: int):
    """Per-trade mean/std ratio and an APPROXIMATE annualised Sharpe.

    The annualisation assumes NON-OVERLAPPING trades of ``dte`` trading days each,
    so trades-per-year = TRAD_DAYS/dte. It is a rough guide, not a daily Sharpe."""
    nan = float("nan")
    if len(pnls) < 2:
        return nan, nan
    m = sum(pnls) / len(pnls)
    var = sum((x - m) ** 2 for x in pnls) / (len(pnls) - 1)
    sd = math.sqrt(var)
    if sd <= 0:
        return nan, nan
    ratio = m / sd
    trades_per_year = TRAD_DAYS / max(dte, 1)
    return ratio, ratio * math.sqrt(trades_per_year)


# --------------------------------------------------------------------------- #
# Offline driver — walk a price series + a chain_at() into a full ledger
# --------------------------------------------------------------------------- #
def paper_trade_series(prices, chain_at, forecaster, *, dte: int = 21, warmup: int = 63,
                       step: int | None = None, **ledger_kw) -> PaperLedger:
    """Record a forecast every ``step`` bars (non-overlapping by default), then
    settle everything that matured. This is the offline analogue of running the
    live recorder daily and settling at the end — deterministic, no network."""
    step = step or dte
    led = PaperLedger(**ledger_kw)
    t = warmup
    while t < len(prices):
        chain = chain_at(t, prices[:t + 1])
        if chain is not None:
            led.record(chain, prices, t, forecaster, dte=dte)
        t += step
    led.settle(prices)
    return led


# --------------------------------------------------------------------------- #
# CLI — record live daily, settle + report offline.  Turnkey forward test.
#
#   python3 -m tools.paper_trade record --source deribit --currency BTC \
#       --dte 30 --ledger btc.jsonl                  # run once per trading day
#   python3 -m tools.paper_trade report --ledger btc.jsonl
#
# A companion append-only PRICE JOURNAL (<ledger>.prices.json) gives every entry a
# STABLE index: seeded from history on the first record, then +1 close per day, so
# prices[entry_index+dte] is unambiguous however the vendor's window slides. The
# first settled score therefore appears only after `dte` further daily records —
# that lag is the honest cost of a real out-of-sample test, not a bug.
# --------------------------------------------------------------------------- #
def _load_json(path, default):
    if path and os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return default


def _journal_path(args):
    return args.prices or (args.ledger + ".prices.json")


def _open_ledger(args, *, create: bool):
    kw = {}
    for k in ("r", "q", "hedge_bps", "spread_frac"):
        if getattr(args, k, None) is not None:
            kw[k] = getattr(args, k)
    if os.path.exists(args.ledger):
        return PaperLedger.load(args.ledger, **kw)
    if not create:
        raise SystemExit(f"ledger not found: {args.ledger} (run `record` first)")
    return PaperLedger(**kw)


def _news_feature(args, chain):
    """--news-json -> SentimentFeature dated at the chain snapshot (or None)."""
    if not getattr(args, "news_json", None):
        return None
    import datetime
    from models.sentiment import (FileNewsAdapter, LexiconSentimentScorer,
                                  aggregate_sentiment)
    items = FileNewsAdapter(args.news_json).load(chain.symbol or None)
    try:
        asof = datetime.date.fromisoformat(str(chain.asof)[:10])
    except (TypeError, ValueError):
        asof = datetime.date.today()
    return aggregate_sentiment(items, LexiconSentimentScorer(), asof=asof)


def _cmd_record(args):
    from tools.run_live import fetch
    chain, hist = fetch(args.source, args)
    jpath = _journal_path(args)
    journal = _load_json(jpath, [])
    if not journal:
        journal = list(hist)                      # seed once from vendor history
    elif hist:
        journal.append(float(hist[-1]))           # +1 close: today's bar
    led = _open_ledger(args, create=True)
    feat = _news_feature(args, chain)
    if feat is not None:
        print(f"news gate: score={feat.score:+.2f} dispersion={feat.dispersion:.2f} "
              f"volume={feat.volume:.1f} ({feat.n_items} items)")
    e = led.record(chain, journal, len(journal) - 1, BaselineDensityForecaster(),
                   dte=args.dte, min_vrp=args.min_vrp, news_feature=feat)
    with open(jpath, "w") as fh:
        json.dump(journal, fh)
    if e is None:
        print(f"record: chain gave no usable Q at {args.dte}d — nothing added "
              f"(journal={len(journal)} bars)")
    else:
        print(f"record: id={e['id']} asof={e['asof']} spot={e['entry_spot']:,.2f} "
              f"dte={e['dte']} traded={e['traded']} vrp={e['vrp']:+.4f} q_vol={e['q_vol']:.1%} "
              f"(journal={len(journal)} bars; matures at index {e['entry_index'] + e['dte']})")
    n = led.settle(journal)
    led.save(args.ledger)
    if n:
        print(f"record: settled {n} newly-matured entr{'y' if n == 1 else 'ies'}")


def _cmd_settle(args):
    journal = _load_json(_journal_path(args), [])
    led = _open_ledger(args, create=False)
    n = led.settle(journal)
    led.save(args.ledger)
    n_open = sum(1 for x in led.entries if x["status"] == "open")
    print(f"settle: {n} newly matured; {n_open} still open; "
          f"{len(led.entries) - n_open} settled total")


def _cmd_report(args):
    journal = _load_json(_journal_path(args), None)
    led = _open_ledger(args, create=False)
    if journal:
        led.settle(journal)
        led.save(args.ledger)
    print(led.report().summary())


def main(argv=None):
    import argparse

    p = argparse.ArgumentParser(
        prog="tools.paper_trade",
        description="Forward-test paper-trading ledger: record -> settle -> score.")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--ledger", required=True, help="ledger .jsonl path")
        sp.add_argument("--prices", help="price-journal .json (default <ledger>.prices.json)")

    r = sub.add_parser("record", help="freeze today's forecast + market Q, append an entry")
    common(r)
    r.add_argument("--source", required=True, choices=["deribit", "tradier", "replay"])
    r.add_argument("--currency", default="BTC")
    r.add_argument("--symbol", default="SPX")
    r.add_argument("--dte", type=int, default=30)
    r.add_argument("--days", type=int, default=400)
    r.add_argument("--chain-json", dest="chain_json")
    r.add_argument("--price-json", dest="price_json")
    r.add_argument("--r", type=float, default=0.03)
    r.add_argument("--q", type=float, default=0.0)
    r.add_argument("--hedge-bps", dest="hedge_bps", type=float, default=5e-4)
    r.add_argument("--spread-frac", dest="spread_frac", type=float, default=0.015)
    r.add_argument("--min-vrp", dest="min_vrp", type=float, default=0.0)
    r.add_argument("--news-json", dest="news_json",
                   help="news file (JSON/CSV of date,symbol,headline[,body]) -> "
                        "forward-looking event-risk gate on today's entry")
    r.set_defaults(func=_cmd_record)

    s = sub.add_parser("settle", help="grade every matured entry against the price journal")
    common(s)
    s.set_defaults(func=_cmd_settle, r=None, q=None, hedge_bps=None, spread_frac=None)

    rp = sub.add_parser("report", help="settle (if a journal exists) then print the scoreboard")
    common(rp)
    rp.set_defaults(func=_cmd_report, r=None, q=None, hedge_bps=None, spread_frac=None)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
