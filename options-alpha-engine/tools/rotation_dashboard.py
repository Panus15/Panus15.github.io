"""Build the sector-rotation dashboard — chart, leaderboard, and the verdict.

    python3 -m tools.rotation_dashboard --csv sectors.csv --out rotation.html
    python3 -m tools.rotation_dashboard --demo --out rotation.html

The CSV wants one row per trading day, a ``date`` column, and one price column per
symbol including the benchmark:

    date,SPY,XLK,XLV,XLF,XLY,XLP,XLE,XLI,XLB,XLRE,XLU,XLC
    2025-01-02,589.39,231.02,...

What makes this different from the rotation dashboards you have seen: it renders
the chart AND the result of testing whether the chart predicts anything, on the
same page, from the same data. A quadrant panel that says NO SIGNIFICANT EFFECT
is displayed just as prominently as one that says the opposite, because a reader
deciding what to do with the picture needs both.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.rotation_backtest import (preregistered_verdict, quadrant_panel,
                                      run_rotation_backtest)
from models.rotation import (BENCHMARK, SECTOR_ETFS, leaderboard,
                             quadrant_history, relative_strength, rotation_map,
                             transitions)


def load_csv(path: str, benchmark: str = BENCHMARK):
    import csv
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"{path} is empty")
    cols = [c for c in rows[0] if c and c.lower() not in ("date", "")]
    series = {c: [] for c in cols}
    dates = []
    for r in rows:
        dates.append(r.get("date") or r.get("Date") or "")
        for c in cols:
            try:
                series[c].append(float(r[c]))
            except (TypeError, ValueError):
                series[c].append(float("nan"))
    if benchmark not in series:
        raise SystemExit(f"benchmark column {benchmark!r} not found in {path}; "
                         f"columns are {', '.join(cols)}")
    bench = series.pop(benchmark)
    return series, bench, dates


def demo_world(n: int = 1300, seed: int = 20260731):
    """Sectors whose leadership genuinely rotates, so the chart has something to show."""
    rng = random.Random(seed)
    bench = [100.0]
    for _ in range(n):
        bench.append(bench[-1] * math.exp(rng.gauss(0.0004, 0.0085)))
    px = {}
    for i, sym in enumerate(SECTOR_ETFS):
        phase = 2 * math.pi * i / len(SECTOR_ETFS)
        p = [100.0]
        for t in range(n):
            a = 0.0011 * math.sin(2 * math.pi * t / 230.0 + phase)
            p.append(p[-1] * math.exp(math.log(bench[t + 1] / bench[t]) + a
                                      + rng.gauss(0, 0.005)))
        px[sym] = p
    import datetime
    d0 = datetime.date(2021, 6, 1)
    return px, bench, [(d0 + datetime.timedelta(days=i)).isoformat()
                       for i in range(n + 1)]


def options_panel(chain_json: str, price_json: str, *, symbol: str = "",
                  dte: int = 30, equity: float = 100_000.0,
                  max_risk_frac: float = 0.02, ledger: str = "") -> dict:
    """The OPTIONS decision, as JSON for the page — or a named reason it is absent.

    Why this lives on the rotation dashboard at all. The one study this repo has
    run on real market data answered NO (PREREGISTRATION.md §7.1): the sector
    quadrant predicted nothing. So the page a human actually opens was leading
    with its DISPROVEN signal while the half backed by oracle-tested machinery —
    the variance edge, and the order it implies — printed only into a terminal.
    That ordering teaches the reader to act on the wrong number.

    Returns ``{"available": False, "reason": ..., "how": ...}`` when no chain was
    supplied. Absence is rendered, never hidden: a panel that disappears when the
    data is missing looks identical to one that had nothing to say.
    """
    if not chain_json:
        return {"available": False,
                "reason": "no option chain was supplied, so the variance edge — "
                          "the only half of this engine backed by oracle-tested "
                          "machinery — is not on this page",
                "how": "python3 -m tools.run_live tradier --symbol SPX --dte 30 "
                       "--dump spx.json, then pass --chain-json spx.json"}
    if not os.path.exists(chain_json):
        return {"available": False,
                "reason": f"the chain file {chain_json!r} does not exist",
                "how": "check the path, or re-dump it with run_live --dump"}
    try:
        from engine import spread_backtest
        from engine.adapters import JsonFileAdapter
        from models.decision import build_decision
        from models.spreads import scan_spreads
        from models.ticket import build_ticket
        from models.trade_card import build_card
        from models.baseline import BaselineDensityForecaster
        from tools.run_live import _settled_count

        ad = JsonFileAdapter(price_json=price_json or None, chain_json=chain_json)
        chain = ad.option_chain(symbol)
        prices = ad.price_history(symbol) if price_json else []
        if len(prices) < 30:
            return {"available": False,
                    "reason": f"only {len(prices)} price bars reached the forecaster "
                              f"and the P density needs 30; a chain without history "
                              f"gives a Q with nothing to compare it against",
                    "how": "pass --price-json as well as --chain-json"}
        forecaster = BaselineDensityForecaster()
        listed = sorted({q.expiry_days for q in chain.quotes})
        use_dte = min(listed, key=lambda d: abs(d - dte)) if listed else dte
        card = build_card(chain, forecaster, prices, dte=use_dte)
        decision = build_decision(card)
        spreads = scan_spreads(chain, forecaster, prices, dte=use_dte)
        best = max(spreads, key=lambda sp: sp.ev) if spreads else None
        ticket = build_ticket(card, best, equity=equity,
                              max_risk_frac=max_risk_frac, decision=decision,
                              settled_trades=_settled_count(ledger),
                              exit_rule=spread_backtest.EXIT_RULE_TEXT,
                              exit_split_note=spread_backtest.EXIT_SPLIT_NOTE)
    except (ValueError, KeyError, OSError, ZeroDivisionError, ArithmeticError,
            IndexError) as e:
        # A chain too sparse or too short-dated to price is a DATA outcome. The
        # page says which, rather than rendering an empty box.
        return {"available": False,
                "reason": f"the chain could not be turned into a decision ({e})",
                "how": "sparse or very short-dated chains fail coverage; try a "
                       "30-45 DTE expiry on a liquid index"}
    return {
        "available": True,
        "symbol": ticket.symbol or chain.symbol,
        "asof": str(getattr(chain, "asof", "") or getattr(card, "asof", "") or ""),
        "dte": use_dte,
        "requestedDte": dte,
        "volSide": card.vol_side,
        "volReason": getattr(card, "vol_reason", ""),
        "pVol": round(float(getattr(card, "p_vol", 0.0)), 4),
        "qVol": round(float(getattr(card, "q_vol", 0.0)), 4),
        "vrp": round(float(getattr(card, "vrp", 0.0)), 4),
        "action": decision.action,
        "sizeMultiplier": decision.size_multiplier,
        "weakest": decision.weakest_evidence,
        "inputs": [{"name": i.name, "reading": getattr(i, "reading", ""),
                    "status": i.status, "effect": getattr(i, "effect", 1.0),
                    "note": getattr(i, "note", "")} for i in decision.inputs],
        "warnings": list(getattr(decision, "warnings", []) or []),
        "placeable": ticket.placeable,
        "structure": ticket.structure,
        "expiryDate": ticket.expiry_date,
        "contracts": ticket.contracts,
        "uncappedContracts": ticket.uncapped_contracts,
        # NOT rounded. Rounding per-contract and total independently makes the
        # two numbers on screen fail to multiply out, and rounding the total up
        # from a rounded per-contract can print a loss a cent over its own budget.
        # The payload stays exact; the page formats to 2dp at display time.
        "limitCredit": ticket.limit_credit,
        "creditTotal": ticket.credit_total,
        "maxLossPerContract": ticket.max_loss_per_contract,
        "maxLossTotal": ticket.max_loss_total,
        "equity": equity,
        "maxRiskFrac": max_risk_frac,
        "probProfit": round(float(ticket.prob_profit), 4),
        "evPerContract": ticket.ev_per_contract,
        "exitRule": ticket.exit_rule,
        "exitSplitNote": ticket.exit_split_note,
        "evidence": ticket.evidence,
        "legs": [{"side": l.side, "kind": l.kind, "strike": l.strike,
                  "price": l.price} for l in ticket.legs],
        "refusals": [{"code": c, "detail": d} for c, d in ticket.refusals],
    }


def ledger_panel(ledger_path: str, *, dte_hint: int = 30) -> dict:
    """The FORWARD TEST over time — or a named reason it is empty.

    Why this is on the page. Every other panel here is one snapshot: today's
    decision, today's quadrant. Neither can answer the only question that matters,
    which is whether the edge PERSISTS. `tools/paper_trade.py` already records a
    dated (P, Q) pair every day it is run, so the trend exists the moment the clock
    starts — it was simply never rendered, and an invisible clock is one nobody
    winds. The largest risk to this project is not a modelling error, it is that
    the daily job stops being run.

    An empty ledger is reported as EMPTY with the command that starts it, never as
    a blank panel. 0 recorded and 0 settled is the honest current state and it is
    what a reader most needs to see before trusting anything else on this page.
    """
    if not ledger_path:
        return {"available": False, "recorded": 0, "settled": 0,
                "reason": "no forward-test ledger was supplied, so nothing on this "
                          "page has been graded against an outcome that had not "
                          "happened yet",
                "how": "python3 -m tools.paper_trade record --source tradier "
                       "--symbol SPX --dte 30 --ledger spx.jsonl   (once per "
                       "trading day), then pass --ledger spx.jsonl"}
    if not os.path.exists(ledger_path):
        return {"available": False, "recorded": 0, "settled": 0,
                "reason": f"the ledger {ledger_path!r} does not exist yet",
                "how": "python3 -m tools.paper_trade record ... --ledger "
                       + ledger_path}
    try:
        from tools.paper_trade import PaperLedger
        led = PaperLedger.load(ledger_path)
        rep = led.report()
    except (OSError, ValueError, KeyError, ZeroDivisionError) as e:
        return {"available": False, "recorded": 0, "settled": 0,
                "reason": f"the ledger could not be read ({e})",
                "how": "a half-written line is the usual cause; the file is "
                       "append-only JSON lines"}
    if not led.entries:
        return {"available": False, "recorded": 0, "settled": 0,
                "reason": "the ledger exists but holds 0 recorded dates — the "
                          "clock has not started",
                "how": "python3 -m tools.paper_trade record ... --ledger "
                       + ledger_path}

    series = []
    for e in led.entries:
        series.append({
            "date": str(e.get("asof") or ""),
            "pVol": e.get("p_vol"), "qVol": e.get("q_vol"), "vrp": e.get("vrp"),
            "traded": bool(e.get("traded")),
            "settled": e.get("status") == "settled",
            # A date whose wings did not cover +/-10% has a Q biased LOW, so it is
            # marked rather than quietly averaged in with the trustworthy ones. A
            # date that never RECORDED the flag is unknown, not fine: an unmarked
            # point on the chart reads as "checked and good", so it is marked too,
            # with its own reason.
            "coverage": (bool(e["coverage_ok"]) if "coverage_ok" in e else False),
            "coverageKnown": "coverage_ok" in e,
            "pnl": e.get("trade_pnl"),
        })
    dtes = [int(e.get("dte") or dte_hint) for e in led.entries] or [dte_hint]
    return {
        "available": True,
        "recorded": rep.n_recorded, "settled": rep.n_settled, "open": rep.n_open,
        "firstDate": series[0]["date"], "lastDate": series[-1]["date"],
        "dte": max(dtes),
        "series": series,
        "pNll": rep.p_nll, "qNll": rep.q_nll, "nllWinRate": rep.nll_win_rate,
        "pLeftTail": rep.p_left_tail, "qLeftTail": rep.q_left_tail,
        "leftTailWinRate": rep.left_tail_win_rate,
        "nTrades": rep.n_trades, "totalPnl": rep.total_pnl,
        "hitRate": rep.trade_hit_rate, "annSharpe": rep.ann_sharpe,
        "verdict": rep._verdict(),
        # the honest lag: a score exists only once a horizon has elapsed
        "needMoreRecords": max(max(dtes) - rep.n_recorded, 0) if rep.n_settled == 0
                           else 0,
    }


def build_payload(px: dict, bench: list, dates: list, *, window: int, mom_lag: int,
                  tail: int, horizon: int, spark: int = 120, run_test: bool = True,
                  options: dict | None = None, ledger: dict | None = None):
    pts = rotation_map(px, bench, window=window, mom_lag=mom_lag, tail=tail)
    if not pts:
        raise SystemExit(f"no symbol has the {2 * window + mom_lag + tail} bars this "
                         f"chart needs (longest series: "
                         f"{max((len(v) for v in px.values()), default=0)})")
    board = leaderboard(pts)
    rank = {p.symbol: i + 1 for i, p in enumerate(board)}

    points = []
    for p in pts:
        rs_line = relative_strength(px[p.symbol], bench)[-spark:]
        hist = quadrant_history(px[p.symbol], bench, window=window,
                                mom_lag=mom_lag, bars=tail * 6)
        flips = transitions(hist)
        bars_in = 0
        for _, q in reversed(hist):
            if q != p.quadrant:
                break
            bars_in += 1
        points.append({
            "symbol": p.symbol, "name": p.name,
            "rs": round(p.rs_ratio, 2), "mom": round(p.rs_momentum, 2),
            "quadrant": p.quadrant, "excess": round(p.excess_return, 4),
            "rank": rank[p.symbol], "barsIn": bars_in,
            "counterClockwise": sum(1 for _, _, _, cw in flips if not cw),
            "tail": [[round(a, 2), round(b, 2)] for a, b in p.tail],
            "spark": [round(x, 3) for x in rs_line],
        })

    payload = {
        "asof": dates[-1] if dates else "",
        "benchmark": BENCHMARK,
        "benchLast": round(bench[-1], 2),
        "bars": len(bench),
        "window": window, "momLag": mom_lag, "tail": tail, "horizon": horizon,
        "points": points,
        "test": None,
        # Always a dict, never None: the options panel renders its own ABSENCE, so
        # "no chain supplied" must reach the page rather than be falsy and skipped.
        "options": options if options is not None else options_panel("", ""),
        "ledger": ledger if ledger is not None else ledger_panel(""),
    }

    if run_test:
        panel = quadrant_panel(px, bench, window=window, mom_lag=mom_lag,
                               horizon=horizon)
        book = run_rotation_backtest(px, bench, window=window, mom_lag=mom_lag,
                                     horizon=horizon)
        payload["test"] = {
            "panelVerdict": panel.verdict(),
            "dates": panel.n_dates,
            "quadrants": {q: {"n": s["n"], "mean": round(s["mean"], 5),
                              "t": round(s["t"], 2)}
                          for q, s in panel.stats.items()},
            "bookVerdict": book.verdict(),
            "bookReturn": round(book.metrics.total_return, 4),
            "bookSharpe": round(book.metrics.sharpe, 2),
            "rebalances": book.n_rebalances,
            "placeboReturn": (round(book.placebo.total_return, 4)
                              if book.placebo else None),
            "placeboDraws": len(book.placebo_totals),
            "permutationP": (round(book.permutation_p(), 4)
                             if book.permutation_p() is not None else None),
            "turnover": round(book.mean_turnover, 3),
            "prereg": preregistered_verdict(panel, book),
        }
    return payload


TEMPLATE = """<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sector Rotation — @@ASOF@@</title>
<style>
:root{
  --ground:#fbfaf7; --panel:#ffffff; --edge:#e3e0d8; --ink:#1c2026; --ink-2:#5b6069;
  --ink-3:#8b9099; --grid:#e8e5dd; --accent:#0e6b62;
  --lead:#12776a; --weak:#a5761b; --lag:#a04236; --improve:#3a5f95;
  --lead-bg:#12776a1f; --weak-bg:#a5761b1f; --lag-bg:#a042361f; --improve-bg:#3a5f951f;
  --shadow:0 1px 2px #1c202610, 0 8px 24px #1c202608;
}
@media (prefers-color-scheme:dark){:root{
  --ground:#101316; --panel:#171b20; --edge:#272c33; --ink:#eceef1; --ink-2:#a2a8b2;
  --ink-3:#6d747e; --grid:#232830; --accent:#4fd0be;
  --lead:#4fd0be; --weak:#e0aa4e; --lag:#e0705f; --improve:#7ba6e8;
  --lead-bg:#4fd0be1c; --weak-bg:#e0aa4e1c; --lag-bg:#e0705f1c; --improve-bg:#7ba6e81c;
  --shadow:0 1px 2px #0006, 0 12px 32px #0004;
}}
:root[data-theme="dark"]{
  --ground:#101316; --panel:#171b20; --edge:#272c33; --ink:#eceef1; --ink-2:#a2a8b2;
  --ink-3:#6d747e; --grid:#232830; --accent:#4fd0be;
  --lead:#4fd0be; --weak:#e0aa4e; --lag:#e0705f; --improve:#7ba6e8;
  --lead-bg:#4fd0be1c; --weak-bg:#e0aa4e1c; --lag-bg:#e0705f1c; --improve-bg:#7ba6e81c;
  --shadow:0 1px 2px #0006, 0 12px 32px #0004;
}
:root[data-theme="light"]{
  --ground:#fbfaf7; --panel:#ffffff; --edge:#e3e0d8; --ink:#1c2026; --ink-2:#5b6069;
  --ink-3:#8b9099; --grid:#e8e5dd; --accent:#0e6b62;
  --lead:#12776a; --weak:#a5761b; --lag:#a04236; --improve:#3a5f95;
  --lead-bg:#12776a1f; --weak-bg:#a5761b1f; --lag-bg:#a042361f; --improve-bg:#3a5f951f;
  --shadow:0 1px 2px #1c202610, 0 8px 24px #1c202608;
}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  font-size:15px;line-height:1.55;-webkit-font-smoothing:antialiased}
.wrap{max-width:1240px;margin:0 auto;padding:32px 20px 72px}
h1,h2,h3{font-family:ui-serif,Georgia,"Iowan Old Style","Times New Roman",serif;
  font-weight:600;text-wrap:balance;margin:0}
h1{font-size:31px;letter-spacing:-.015em}
h2{font-size:19px;letter-spacing:-.01em}
.num{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
  font-variant-numeric:tabular-nums}
.eyebrow{font-size:11px;letter-spacing:.13em;text-transform:uppercase;
  color:var(--ink-3);font-weight:600}
header{display:flex;flex-wrap:wrap;gap:20px;align-items:flex-end;
  justify-content:space-between;border-bottom:1px solid var(--edge);
  padding-bottom:20px;margin-bottom:28px}
header p{margin:6px 0 0;color:var(--ink-2);max-width:60ch;font-size:14px}
.meta{display:flex;gap:26px}
.meta div{text-align:right}
.meta .k{font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:var(--ink-3)}
.meta .v{font-size:21px;font-weight:600}
.grid{display:grid;grid-template-columns:minmax(0,1.35fr) minmax(0,1fr);gap:22px}
@media(max-width:940px){.grid{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--edge);border-radius:10px;
  padding:18px 20px;box-shadow:var(--shadow)}
.card>header{border:0;padding:0;margin:0 0 14px;display:flex;align-items:baseline;
  justify-content:space-between;gap:12px}
.hint{color:var(--ink-3);font-size:12.5px;margin:12px 0 0}
svg{display:block;width:100%;height:auto;overflow:visible}
.q-label{font-size:10px;letter-spacing:.12em;text-transform:uppercase;font-weight:700}
.dot{cursor:pointer}
.dot:focus{outline:2px solid var(--accent);outline-offset:3px}
table{width:100%;border-collapse:collapse;font-size:13.5px}
th{text-align:left;font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;
  color:var(--ink-3);font-weight:600;padding:0 6px 8px;border-bottom:1px solid var(--edge);
  white-space:nowrap}
th.r,td.r{text-align:right}
td{padding:8px 6px;border-bottom:1px solid var(--edge)}
tr:last-child td{border-bottom:0}
tbody tr{transition:background .12s}
tbody tr:hover,tbody tr.on{background:var(--grid)}
.pill{display:inline-block;padding:2px 8px;border-radius:99px;font-size:10.5px;
  font-weight:700;letter-spacing:.05em;white-space:nowrap}
.Leading{color:var(--lead);background:var(--lead-bg)}
.Weakening{color:var(--weak);background:var(--weak-bg)}
.Lagging{color:var(--lag);background:var(--lag-bg)}
.Improving{color:var(--improve);background:var(--improve-bg)}
.bar{height:7px;border-radius:2px;display:block}
.barwrap{display:flex;align-items:center;gap:7px;justify-content:flex-end;
  white-space:nowrap}
td.excess{white-space:nowrap}
.sector b{display:block;line-height:1.25}
.sector span{display:block;font-size:11.5px;color:var(--ink-2);line-height:1.25}
.sparks{display:grid;grid-template-columns:repeat(auto-fill,minmax(168px,1fr));gap:14px}
.spark{border:1px solid var(--edge);border-radius:8px;padding:10px 12px 6px;
  background:var(--panel)}
.spark .top{display:flex;justify-content:space-between;align-items:baseline;gap:8px}
.spark b{font-size:13px;letter-spacing:.02em}
.verdict{border-left:3px solid var(--accent);padding:2px 0 2px 15px;margin:0 0 16px}
.verdict p{margin:5px 0 0;color:var(--ink-2);font-size:14px}
.verdict strong{color:var(--ink)}
.tworow{display:grid;grid-template-columns:1fr 1fr;gap:22px}
@media(max-width:760px){.tworow{grid-template-columns:1fr}}
.note{margin-top:34px;padding:16px 20px;border:1px dashed var(--edge);border-radius:10px;
  color:var(--ink-2);font-size:13px}
.note b{color:var(--ink)}
.section{margin-top:26px}
.scroll{overflow-x:auto}
@media(prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}

/* --- options panel: the validated half, so it leads the page --- */
#opt{margin-bottom:22px}
#opt .verdict{display:flex;flex-wrap:wrap;align-items:baseline;gap:10px 16px;
  margin:2px 0 14px}
#opt .side{font:700 26px/1.1 ui-sans-serif,system-ui,sans-serif;letter-spacing:-.02em}
#opt .side.sell{color:var(--lead)}
#opt .side.no{color:var(--ink-3)}
#opt .side.buy{color:var(--improve)}
#opt .mult{font-variant-numeric:tabular-nums;color:var(--ink-2);font-size:14px}
#opt .why{color:var(--ink-2);font-size:13.5px;margin:0 0 16px;max-width:72ch}
#opt .order{border:1px solid var(--edge);border-radius:10px;overflow:hidden;
  background:var(--ground)}
#opt .order > .hd{display:flex;flex-wrap:wrap;gap:6px 14px;align-items:baseline;
  padding:10px 14px;border-bottom:1px solid var(--edge);background:var(--panel)}
#opt .order > .hd b{font-size:15px}
#opt table.legs{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
#opt table.legs td{padding:7px 14px;border-bottom:1px solid var(--edge);font-size:13.5px}
#opt table.legs tr:last-child td{border-bottom:0}
#opt .sideTag{display:inline-block;min-width:48px;font-weight:700;font-size:11.5px;
  letter-spacing:.06em;padding:2px 7px;border-radius:5px}
#opt .sideTag.short{background:var(--lag-bg);color:var(--lag)}
#opt .sideTag.long{background:var(--improve-bg);color:var(--improve)}
#opt .nums{display:grid;gap:1px;background:var(--edge);
  grid-template-columns:repeat(auto-fit,minmax(160px,1fr));margin-top:14px}
#opt .nums div{background:var(--panel);padding:10px 13px}
#opt .nums .k{font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;
  color:var(--ink-3)}
#opt .nums .v{font:600 17px/1.3 ui-sans-serif,system-ui,sans-serif;
  font-variant-numeric:tabular-nums;margin-top:3px}
#opt .refuse{border:1px solid var(--edge);border-left:3px solid var(--lag);
  border-radius:8px;padding:12px 14px;background:var(--lag-bg)}
#opt .refuse .code{font:700 11.5px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace;
  letter-spacing:.04em;color:var(--lag)}
#opt .refuse p{margin:2px 0 12px;font-size:13px;color:var(--ink-2);max-width:78ch}
#opt .refuse p:last-child{margin-bottom:0}
#opt .absent{border:1px dashed var(--edge);border-radius:8px;padding:14px;
  color:var(--ink-2);font-size:13.5px;max-width:80ch}
#opt .absent code{background:var(--ground);border:1px solid var(--edge);
  border-radius:5px;padding:1px 6px;font-size:12.5px;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace;word-break:break-all}
#opt ul.inp{list-style:none;margin:16px 0 0;padding:0;font-size:13px}
#opt ul.inp li{display:flex;flex-wrap:wrap;gap:4px 10px;padding:7px 0;
  border-top:1px solid var(--edge)}
#opt ul.inp .nm{min-width:118px;font-weight:600}
#opt ul.inp .rd{color:var(--ink-2);flex:1 1 220px}
#opt ul.inp .st{font-size:11px;letter-spacing:.05em;text-transform:uppercase;
  color:var(--ink-3);white-space:nowrap}
#opt .ev{margin:14px 0 0;font-size:12.5px;color:var(--ink-3);max-width:80ch}

/* --- forward-test ledger: the only panel that can show PERSISTENCE --- */
#fwd{margin-bottom:22px}
#fwd .clocks{display:grid;gap:1px;background:var(--edge);
  grid-template-columns:repeat(auto-fit,minmax(130px,1fr));margin:2px 0 16px}
#fwd .clocks div{background:var(--panel);padding:10px 13px}
#fwd .clocks .k{font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;
  color:var(--ink-3)}
#fwd .clocks .v{font:600 20px/1.2 ui-sans-serif,system-ui,sans-serif;
  font-variant-numeric:tabular-nums;margin-top:3px}
#fwd .vq{display:block;width:100%;height:150px;border:1px solid var(--edge);
  border-radius:8px;background:var(--ground)}
#fwd .key{display:flex;flex-wrap:wrap;gap:6px 18px;margin:10px 0 0;font-size:12.5px;
  color:var(--ink-2)}
#fwd .key i{display:inline-block;width:18px;height:3px;border-radius:2px;
  vertical-align:middle;margin-right:6px}
#fwd .score{display:grid;gap:1px;background:var(--edge);margin-top:16px;
  grid-template-columns:repeat(auto-fit,minmax(190px,1fr))}
#fwd .score div{background:var(--panel);padding:10px 13px;font-size:13px}
#fwd .verdict-line{margin:16px 0 0;padding:12px 14px;border:1px solid var(--edge);
  border-left:3px solid var(--accent);border-radius:8px;font-size:13.5px;
  color:var(--ink);max-width:86ch}
#fwd .absent{border:1px dashed var(--edge);border-radius:8px;padding:14px;
  color:var(--ink-2);font-size:13.5px;max-width:84ch}
#fwd .absent code{background:var(--ground);border:1px solid var(--edge);
  border-radius:5px;padding:1px 6px;font-size:12.5px;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace;word-break:break-all}
</style>

<div class="wrap">
<header>
  <div>
    <div class="eyebrow">Relative rotation &middot; @@N@@ S&amp;P 500 sectors vs @@BENCH@@</div>
    <h1>Where the market is paying, and whether that tells you anything</h1>
    <p>Both axes are z-scores against each sector&rsquo;s <em>own</em> recent history,
       so 100 means &ldquo;in line with its norm&rdquo;, not &ldquo;in line with
       @@BENCH@@&rdquo;. This is a transform of relative price &mdash; not fund flow.</p>
  </div>
  <div class="meta">
    <div><div class="k">As of</div><div class="v num">@@ASOF@@</div></div>
    <div><div class="k">@@BENCH@@</div><div class="v num">@@BENCHLAST@@</div></div>
    <div><div class="k">Bars</div><div class="v num">@@BARS@@</div></div>
  </div>
</header>

<section class="card section" id="opt"></section>

<section class="card section" id="fwd"></section>

<div class="grid">
  <section class="card">
    <header><h2>Rotation graph</h2>
      <span class="eyebrow">@@WINDOW@@-bar window &middot; @@TAIL@@-bar tail</span></header>
    <div id="rrg"></div>
    <p class="hint">Rotation runs clockwise: Improving &rarr; Leading &rarr; Weakening
      &rarr; Lagging. Tails show the last @@TAIL@@ bars. Hover a dot or a row to isolate it.</p>
  </section>

  <section class="card">
    <header><h2>Leaderboard</h2><span class="eyebrow">strength + momentum</span></header>
    <div class="scroll"><table id="board">
      <thead><tr><th>#</th><th>Sector</th><th class="r">RS</th><th class="r">Mom</th>
        <th>Quadrant</th><th class="r">Excess</th></tr></thead>
      <tbody></tbody></table></div>
    <p class="hint">Excess return is measured over the @@TAIL@@-bar tail window, not the year.
       Magnitudes are easier to read in the sparklines below.</p>
  </section>
</div>

<section class="card section" id="testcard"></section>

<section class="section">
  <header style="margin-bottom:14px"><h2>Relative strength vs @@BENCH@@</h2></header>
  <div class="sparks" id="sparks"></div>
</section>

<div class="note">
  <b>Read this before acting on it.</b> A rotation chart describes where price has
  already gone; it cannot see fund flows, creations, or anyone&rsquo;s order book.
  The panel above is this dashboard&rsquo;s own out-of-sample test on the same data,
  and it is shown whether or not the answer flatters the chart. Use it with
  fundamentals, position sizing, and a stop &mdash; not instead of them.
</div>
</div>

<script>
const DATA = @@JSON@@;
const Q = ["Leading","Weakening","Lagging","Improving"];
const cssv = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

/* ---------- rotation graph ---------- */
function drawRRG(){
  const pts = DATA.points, S = 560, pad = 46;
  const all = pts.flatMap(p => p.tail.concat([[p.rs,p.mom]]));
  let lo = 100, hi = 100;
  all.forEach(([a,b]) => { lo = Math.min(lo,a,b); hi = Math.max(hi,a,b); });
  const m = Math.max(hi-100, 100-lo) * 1.18 || 1;
  const x = v => pad + (v-(100-m))/(2*m)*(S-2*pad);
  const y = v => S-pad - (v-(100-m))/(2*m)*(S-2*pad);
  const mid = x(100), midY = y(100);
  const q = (a,b) => a>=100 ? (b>=100?"Leading":"Weakening") : (b>=100?"Improving":"Lagging");
  const col = {Leading:"--lead",Weakening:"--weak",Lagging:"--lag",Improving:"--improve"};

  let s = `<svg viewBox="0 0 ${S} ${S}" role="img" aria-label="Relative rotation graph">`;
  s += `<rect x="${mid}" y="${pad}" width="${S-pad-mid}" height="${midY-pad}" fill="var(--lead-bg)"/>`;
  s += `<rect x="${mid}" y="${midY}" width="${S-pad-mid}" height="${S-pad-midY}" fill="var(--weak-bg)"/>`;
  s += `<rect x="${pad}" y="${midY}" width="${mid-pad}" height="${S-pad-midY}" fill="var(--lag-bg)"/>`;
  s += `<rect x="${pad}" y="${pad}" width="${mid-pad}" height="${midY-pad}" fill="var(--improve-bg)"/>`;
  for(let i=1;i<8;i++){const g=pad+i*(S-2*pad)/8;
    s+=`<line x1="${g}" y1="${pad}" x2="${g}" y2="${S-pad}" stroke="var(--grid)"/>`;
    s+=`<line x1="${pad}" y1="${g}" x2="${S-pad}" y2="${g}" stroke="var(--grid)"/>`;}
  s += `<line x1="${mid}" y1="${pad}" x2="${mid}" y2="${S-pad}" stroke="var(--ink-3)" stroke-dasharray="3 4"/>`;
  s += `<line x1="${pad}" y1="${midY}" x2="${S-pad}" y2="${midY}" stroke="var(--ink-3)" stroke-dasharray="3 4"/>`;
  const lbl=[["Improving",pad+10,pad+18,"start","--improve"],["Leading",S-pad-10,pad+18,"end","--lead"],
             ["Lagging",pad+10,S-pad-8,"start","--lag"],["Weakening",S-pad-10,S-pad-8,"end","--weak"]];
  lbl.forEach(([t,px,py,an,c])=>{s+=`<text class="q-label" x="${px}" y="${py}" text-anchor="${an}" fill="var(${c})">${t}</text>`;});
  s += `<text class="q-label" x="${S/2}" y="${S-10}" text-anchor="middle" fill="var(--ink-3)">RS-Ratio &#8594;</text>`;
  s += `<text class="q-label" x="14" y="${S/2}" text-anchor="middle" fill="var(--ink-3)" transform="rotate(-90 14 ${S/2})">RS-Momentum &#8594;</text>`;

  pts.forEach(p => {
    const c = `var(${col[p.quadrant]})`;
    const path = p.tail.map(([a,b],i)=>`${i?"L":"M"}${x(a).toFixed(1)},${y(b).toFixed(1)}`).join(" ");
    s += `<g class="sym" data-sym="${p.symbol}">`;
    s += `<path d="${path}" fill="none" stroke="${c}" stroke-width="1.6" opacity=".45"
           stroke-linejoin="round"/>`;
    p.tail.forEach(([a,b],i)=>{ if(i%3===0&&i<p.tail.length-1)
      s+=`<circle cx="${x(a).toFixed(1)}" cy="${y(b).toFixed(1)}" r="1.9" fill="${c}" opacity=".5"/>`;});
    s += `<circle class="dot" tabindex="0" cx="${x(p.rs).toFixed(1)}" cy="${y(p.mom).toFixed(1)}"
           r="7" fill="${c}" stroke="var(--panel)" stroke-width="2"><title>${p.symbol} — ${p.name}
RS ${p.rs}  Mom ${p.mom}  ${p.quadrant} for ${p.barsIn} bars</title></circle>`;
    s += `<text x="${(x(p.rs)+11).toFixed(1)}" y="${(y(p.mom)+4).toFixed(1)}" font-size="11.5"
           font-weight="700" fill="var(--ink)" paint-order="stroke"
           stroke="var(--panel)" stroke-width="3" stroke-linejoin="round"
           >${p.symbol}</text></g>`;
  });
  document.getElementById("rrg").innerHTML = s + "</svg>";
}

/* ---------- leaderboard ---------- */
function drawBoard(){
  const tb = document.querySelector("#board tbody");
  const max = Math.max(...DATA.points.map(p=>Math.abs(p.excess))) || 1;
  const col = {Leading:"--lead",Weakening:"--weak",Lagging:"--lag",Improving:"--improve"};
  tb.innerHTML = DATA.points.slice().sort((a,b)=>a.rank-b.rank).map(p=>{
    const c = p.excess>=0 ? "var(--lead)" : "var(--lag)";
    return `<tr data-sym="${p.symbol}">
      <td class="num" style="color:var(--ink-3)">${p.rank}</td>
      <td class="sector"><b>${p.symbol}</b><span>${p.name}</span></td>
      <td class="r num">${p.rs.toFixed(2)}</td>
      <td class="r num">${p.mom.toFixed(2)}</td>
      <td><span class="pill ${p.quadrant}">${p.quadrant}</span></td>
      <td class="r excess num" style="color:${c}">${(p.excess*100).toFixed(2)}%</td></tr>`;
  }).join("");
}

/* ---------- the test ---------- */
function esc(x){ return String(x==null?"":x).replace(/[&<>"]/g,
  c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c])); }
function money(x){ return (x<0?"-$":"$") + Math.abs(x).toLocaleString(undefined,
  {minimumFractionDigits:2, maximumFractionDigits:2}); }

function drawOptions(){
  const o = DATA.options, el = document.getElementById("opt");
  // The panel renders its own ABSENCE. Removing it would look identical to a
  // panel that had nothing to say, which is the opposite of what is true: the
  // variance edge is the one half of this engine with oracle-tested machinery
  // behind it, and if it is missing the reader should be told why.
  if(!o || !o.available){
    el.innerHTML = `<header><h2>The options decision</h2>
        <span class="eyebrow">not available</span></header>
      <div class="absent"><b>This page is showing only the context signals.</b><br>
        ${esc(o ? o.reason : "no options data reached this page")}.
        <div style="margin-top:10px">To put it here: <code>${esc(o ? o.how : "")}</code></div>
      </div>`;
    return;
  }
  const sideCls = o.volSide==="SELL VOL" ? "sell" : (o.volSide==="BUY VOL" ? "buy" : "no");
  const inputs = (o.inputs||[]).map(i=>`<li><span class="nm">${esc(i.name)}</span>
      <span class="rd">${esc(i.reading)}</span>
      <span class="st">${esc(i.status)}${i.effect<1 ? " &middot; x"+i.effect.toFixed(2) : ""}</span></li>`).join("");

  let order;
  if(o.placeable){
    const legs = (o.legs||[]).map(l=>`<tr>
        <td><span class="sideTag ${esc(l.side)}">${esc(l.side.toUpperCase())}</span></td>
        <td>${esc(l.kind.toUpperCase())}</td>
        <td class="r num">${l.strike.toFixed(2)}</td>
        <td class="r num">${l.price.toFixed(2)}</td></tr>`).join("");
    order = `<div class="order">
        <div class="hd"><b>SELL ${o.contracts} &times; ${esc(o.structure)}</b>
          <span class="eyebrow">${esc(o.symbol)} &middot; expires ${esc(o.expiryDate)}
          (${o.dte}d)</span></div>
        <table class="legs"><tbody>${legs}</tbody></table></div>
      <div class="nums">
        <div><div class="k">Limit, credit</div><div class="v" style="color:var(--lead)">
          ${money(o.limitCredit)}<span style="font-size:12px;color:var(--ink-3)">
          /contract</span></div></div>
        <div><div class="k">Credit total</div><div class="v">${money(o.creditTotal)}</div></div>
        <div><div class="k">Max loss, total</div><div class="v" style="color:var(--lag)">
          ${money(o.maxLossTotal)}</div></div>
        <div><div class="k">Risk budget</div><div class="v">${(o.maxRiskFrac*100).toFixed(1)}%
          <span style="font-size:12px;color:var(--ink-3)">of
          ${o.equity.toLocaleString()}</span></div></div>
        <div><div class="k">P(profit)</div><div class="v">${(o.probProfit*100).toFixed(1)}%</div></div>
        <div><div class="k">Model EV</div>
          <div class="v" style="color:${o.evPerContract>=0?"var(--lead)":"var(--lag)"}">
          ${o.evPerContract>=0?"+":""}${money(o.evPerContract)}</div></div>
      </div>
      ${o.exitRule ? `<p class="ev" style="color:var(--ink-2)"><b>Exit:</b>
        ${esc(o.exitRule)}${o.exitSplitNote ? `<br><span style="color:var(--ink-3)">${
          esc(o.exitSplitNote)}</span>` : ""}</p>` : ""}`;
  }else{
    order = `<div class="refuse">
        <div class="eyebrow" style="color:var(--lag);margin-bottom:8px">No order &mdash;
          refused, by name</div>
        ${(o.refusals||[]).map(r=>`<div class="code">${esc(r.code)}</div>
          <p>${esc(r.detail)}</p>`).join("")}
      </div>`;
  }

  const cut = o.sizeMultiplier < 1
    ? `size <b>&times;${o.sizeMultiplier.toFixed(2)}</b> after context`
    : `size <b>&times;1.00</b> &mdash; nothing cut it`;
  el.innerHTML = `<header><h2>The options decision</h2>
      <span class="eyebrow">${esc(o.symbol)} &middot; ${o.dte}d${
        o.dte!==o.requestedDte ? ` (asked ${o.requestedDte}d, snapped to a listed expiry)` : ""
      }${o.asof ? " &middot; " + esc(o.asof) : ""}</span></header>
    <div class="verdict">
      <span class="side ${sideCls}">${esc(o.volSide)}</span>
      <span class="mult">${cut}</span>
      <span class="mult">P ${(o.pVol*100).toFixed(1)}% vs Q ${(o.qVol*100).toFixed(1)}%
        &middot; VRP ${o.vrp>=0?"+":""}${(o.vrp*100).toFixed(2)} vol pts</span>
    </div>
    <p class="why">${esc(o.volReason)}</p>
    ${order}
    <ul class="inp">${inputs}</ul>
    <p class="ev"><b>Evidence:</b> ${esc(o.evidence)}</p>
    ${(o.warnings||[]).map(w=>`<p class="ev" style="color:var(--weak)">!! ${esc(w)}</p>`).join("")}`;
}

function drawForward(){
  const f = DATA.ledger, el = document.getElementById("fwd");
  if(!f || !f.available){
    el.innerHTML = `<header><h2>The forward test</h2>
        <span class="eyebrow">0 recorded &middot; 0 settled</span></header>
      <div class="absent"><b>Nothing on this page has been graded against an
        outcome that had not already happened.</b><br>
        ${esc(f ? f.reason : "no ledger reached this page")}.
        <div style="margin-top:10px">To start the clock:
          <code>${esc(f ? f.how : "")}</code></div>
        <div style="margin-top:10px;color:var(--ink-3)">The first score appears only
          after a full horizon has elapsed. That lag is the honest cost of an
          out-of-sample test, not a delay worth engineering away.</div>
      </div>`;
    return;
  }
  const S = f.series || [];
  // P vs Q over time. The gap between the lines IS the variance risk premium, so
  // a reader can see whether the edge persists or was one rich afternoon.
  const vals = S.flatMap(d=>[d.pVol, d.qVol]).filter(v=>typeof v === "number");
  const lo = Math.min(...vals, 0), hi = Math.max(...vals, 0.01);
  const W = 800, H = 150, PAD = 8;
  const x = i => PAD + (S.length<2 ? 0 : i*(W-2*PAD)/(S.length-1));
  const y = v => H-PAD - (v-lo)/((hi-lo)||1)*(H-2*PAD);
  const path = key => S.map((d,i)=> (typeof d[key]==="number"
      ? `${i&&typeof S[i-1][key]==="number"?"L":"M"}${x(i).toFixed(1)},${y(d[key]).toFixed(1)}`
      : "")).join("");
  const flags = S.map((d,i)=> d.coverage ? "" :
      `<circle cx="${x(i).toFixed(1)}" cy="${(H-PAD).toFixed(1)}" r="2.5"
        fill="var(--weak)"><title>${esc(d.date)}: ${d.coverageKnown
        ? "wings did not cover +/-10%, so this date's Q is biased LOW"
        : "this date did not record whether its wings covered +/-10%, so its Q "
          + "cannot be shown to be unbiased"}</title></circle>`).join("");
  const chart = `<svg class="vq" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none"
      role="img" aria-label="forecast vol versus implied vol over time">
      <path d="${path("qVol")}" fill="none" stroke="var(--lag)" stroke-width="2"/>
      <path d="${path("pVol")}" fill="none" stroke="var(--lead)" stroke-width="2"/>
      ${flags}</svg>
    <div class="key">
      <span><i style="background:var(--lag)"></i>Q &mdash; what the market implies</span>
      <span><i style="background:var(--lead)"></i>P &mdash; what we forecast</span>
      <span style="color:var(--ink-3)">Q above P = variance premium to sell.
        ${S.some(d=>!d.coverage) ? "Amber marks = thin wings, Q biased LOW." : ""}</span>
    </div>`;

  const lag = f.needMoreRecords > 0
    ? `<div class="clocks"><div><div class="k">First score in</div>
        <div class="v">${f.needMoreRecords}</div>
        <div class="k" style="text-transform:none;letter-spacing:0">more daily
          records</div></div></div>` : "";

  const score = f.settled > 0 ? `<div class="score">
      <div><div class="k">Mean NLL</div>P ${f.pNll.toFixed(4)} vs Q ${f.qNll.toFixed(4)}
        &mdash; <b style="color:${f.pNll<f.qNll?"var(--lead)":"var(--lag)"}">P wins
        ${(f.nllWinRate*100).toFixed(0)}% of dates</b></div>
      <div><div class="k">Left-tail loss</div>P ${f.pLeftTail.toFixed(4)} vs Q
        ${f.qLeftTail.toFixed(4)} &mdash; P wins
        ${(f.leftTailWinRate*100).toFixed(0)}%</div>
      <div><div class="k">Graded trades</div>${f.nTrades} &middot; total
        <b style="color:${f.totalPnl>=0?"var(--lead)":"var(--lag)"}">
        ${f.totalPnl>=0?"+":""}${money(f.totalPnl)}</b> &middot; hit
        ${(f.hitRate*100).toFixed(0)}%</div>
      <div><div class="k">~Annualised Sharpe</div>${f.annSharpe===f.annSharpe
        ? f.annSharpe.toFixed(2) : "n/a"}
        <span style="color:var(--ink-3)">read the interval, not the point</span></div>
    </div>` : "";

  // A chain carrying no asof gives entries with no date. Printing " - " would be
  // a broken range; saying the dates are missing is the actual state.
  const span = (f.firstDate && f.lastDate)
    ? `${esc(f.firstDate)} &ndash; ${esc(f.lastDate)}`
    : "dates not recorded by this source";
  el.innerHTML = `<header><h2>The forward test</h2>
      <span class="eyebrow">${span} &middot; ${f.dte}d horizon</span></header>
    <div class="clocks">
      <div><div class="k">Recorded</div><div class="v">${f.recorded}</div></div>
      <div><div class="k">Settled</div><div class="v">${f.settled}</div></div>
      <div><div class="k">Open</div><div class="v">${f.open}</div></div>
      <div><div class="k">Signal fired</div><div class="v">${
        S.filter(d=>d.traded).length}</div>
        <div class="k" style="text-transform:none;letter-spacing:0">${f.nTrades
        } graded so far</div></div>
    </div>
    ${lag}
    ${chart}
    ${score}
    <div class="verdict-line">${esc(f.verdict)}</div>`;
}

function drawTest(){
  const t = DATA.test, el = document.getElementById("testcard");
  if(!t){ el.remove(); return; }
  const good = /LEADING BEATS LAGGING/.test(t.panelVerdict);
  const rows = Q.filter(q=>t.quadrants[q]).map(q=>{
    const s = t.quadrants[q];
    const c = s.mean>=0 ? "var(--lead)" : "var(--lag)";
    return `<tr><td><span class="pill ${q}">${q}</span></td>
      <td class="r num">${s.n}</td>
      <td class="r num" style="color:${c}">${(s.mean*100).toFixed(2)}%</td>
      <td class="r num" style="color:var(--ink-2)">${s.t>=0?"+":""}${s.t.toFixed(2)}</td></tr>`;
  }).join("");
  el.innerHTML = `<header><h2>Does the chart predict anything?</h2>
      <span class="eyebrow">${t.dates} rebalance dates &middot; ${DATA.horizon}-bar horizon</span></header>
    <div class="verdict"><div class="eyebrow">Quadrant panel &mdash; forward excess return, before costs</div>
      <p><strong>${t.panelVerdict}</strong></p></div>
    <div class="tworow">
      <div class="scroll"><table>
        <thead><tr><th>Quadrant</th><th class="r">n</th>
          <th class="r">Mean forward</th><th class="r">t</th></tr></thead>
        <tbody>${rows}</tbody></table></div>
      <div>
        <div class="eyebrow">Tradeable version &mdash; long top 3, short bottom 3, costs charged</div>
        <p style="margin:8px 0 0;font-size:14px;color:var(--ink-2)">
          Total <b class="num" style="color:var(--ink)">${(t.bookReturn*100).toFixed(1)}%</b>
          over ${t.rebalances} rebalances &middot;
          Sharpe <b class="num" style="color:var(--ink)">${t.bookSharpe}</b>${
          t.turnover===undefined?"":` &middot; ${(t.turnover*100).toFixed(0)}% of the
          book traded per rebalance`}</p>
        ${t.permutationP===null||t.permutationP===undefined ? `
        <p style="margin:8px 0 0;font-size:14px;color:var(--ink-2)">
          No null distribution was run, so how often chance produces this is
          unmeasured.</p>` : `
        <p style="margin:8px 0 0;font-size:14px;color:var(--ink-2)">
          Against <b class="num" style="color:var(--ink)">${t.placeboDraws}</b>
          label-shuffled books,
          <b class="num" style="color:${t.permutationP<=0.10?'var(--lead)':'var(--lag)'}">
          ${(t.permutationP*100).toFixed(1)}%</b> did as well or better
          <span style="opacity:.75">(permutation p)</span>${
          t.placeboReturn===null?"":` &middot; one of them returned
          <b class="num" style="color:var(--ink)">${(t.placeboReturn*100).toFixed(1)}%</b>
          <span style="opacity:.75">&mdash; a single shuffle, shown for scale only;
          it is the distribution above that carries the claim</span>`}</p>`}
        <p class="hint">${t.bookVerdict}</p>
        ${!t.prereg ? "" : `
        <div style="margin-top:12px;padding:10px 12px;border:1px solid var(--edge);
                    border-radius:8px;background:var(--ground)">
          <div class="eyebrow" style="color:${t.prereg.passed?'var(--lead)':'var(--lag)'}">
            ${t.prereg.headline}</div>
          <p class="hint" style="margin:6px 0 0">
            <b>panel</b> ${t.prereg.panel_passed?"PASS":"FAIL"} &mdash;
            ${t.prereg.why_panel}<br>
            <b>book</b> ${t.prereg.book_passed?"PASS":"FAIL"} &mdash;
            ${t.prereg.why_book}</p>
          <p class="hint" style="margin:6px 0 0;opacity:.75">
            Criteria locked in PREREGISTRATION.md &sect;2.1 before any real data
            was seen, and evaluated by the engine rather than by a reader who has
            already seen the numbers.</p>
        </div>`}
      </div></div>
    <p class="hint">${good
      ? "The panel is the kindest possible test &mdash; no costs, no slippage. A signal that survives it still has to survive the long/short book beside it."
      : "The panel ignores costs entirely, so a signal that fails here cannot be rescued by better execution."}</p>`;
}

/* ---------- sparklines ---------- */
function drawSparks(){
  const col = {Leading:"--lead",Weakening:"--weak",Lagging:"--lag",Improving:"--improve"};
  document.getElementById("sparks").innerHTML = DATA.points.map(p=>{
    const v=p.spark, lo=Math.min(...v), hi=Math.max(...v), r=(hi-lo)||1;
    const W=150,H=38;
    const pt=i=>[ (i/(v.length-1)*W).toFixed(1), (H-(v[i]-lo)/r*H).toFixed(1) ];
    const d=v.map((_,i)=>{const[a,b]=pt(i);return `${i?"L":"M"}${a},${b}`}).join(" ");
    const [lx,ly]=pt(v.length-1);
    const c=`var(${col[p.quadrant]})`;
    const chg=((v[v.length-1]/v[0]-1)*100);
    return `<div class="spark" data-sym="${p.symbol}">
      <div class="top"><b>${p.symbol}</b>
        <span class="num" style="font-size:12px;color:${chg>=0?"var(--lead)":"var(--lag)"}">
          ${chg>=0?"+":""}${chg.toFixed(1)}%</span></div>
      <svg viewBox="0 0 ${W} ${H+4}" aria-label="${p.symbol} relative strength">
        <path d="${d} L${W},${H} L0,${H} Z" fill="${c}" opacity=".10"/>
        <path d="${d}" fill="none" stroke="${c}" stroke-width="1.5"
          stroke-linejoin="round" stroke-linecap="round"/>
        <circle cx="${lx}" cy="${ly}" r="2.6" fill="${c}"/></svg>
      <div style="font-size:11px;color:var(--ink-3);margin-top:2px">${p.name}</div></div>`;
  }).join("");
}

function link(){
  const set = (sym,on) => {
    document.querySelectorAll("[data-sym]").forEach(e=>{
      const me = e.getAttribute("data-sym")===sym;
      if(e.tagName==="TR") e.classList.toggle("on", on&&me);
      else e.style.opacity = (!on||me) ? "1" : ".22";
    });
  };
  document.addEventListener("mouseover", e=>{
    const h=e.target.closest("[data-sym]"); if(h) set(h.getAttribute("data-sym"), true);});
  document.addEventListener("mouseout", e=>{
    if(e.target.closest("[data-sym]")) set(null,false);});
  document.addEventListener("focusin", e=>{
    const h=e.target.closest("[data-sym]"); if(h) set(h.getAttribute("data-sym"), true);});
}

drawOptions(); drawForward(); drawRRG(); drawBoard(); drawTest(); drawSparks(); link();
</script>
"""


def render(payload: dict) -> str:
    """Fill the template by token replacement.

    Not %-formatting or str.format: the page is full of literal ``%`` (CSS widths,
    percentage labels) and ``{}`` (every CSS rule and JS block), and both of those
    schemes would try to interpret them.
    """
    subs = {
        "ASOF": html.escape(str(payload["asof"])),
        "BENCH": html.escape(payload["benchmark"]),
        "BENCHLAST": f"{payload['benchLast']:,.2f}",
        "BARS": str(payload["bars"]),
        "N": str(len(payload["points"])),
        "WINDOW": str(payload["window"]),
        "TAIL": str(payload["tail"]),
        "JSON": json.dumps(payload),
    }
    out = TEMPLATE
    for k, v in subs.items():
        out = out.replace("@@" + k + "@@", v)
    if "@@" in out:
        raise RuntimeError("unfilled template token remains")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", help="date + one price column per symbol (incl. SPY)")
    ap.add_argument("--demo", action="store_true", help="use a generated fixture")
    ap.add_argument("--out", default="rotation.html")
    ap.add_argument("--window", type=int, default=63)
    ap.add_argument("--mom-lag", dest="mom_lag", type=int, default=5)
    ap.add_argument("--tail", type=int, default=12)
    ap.add_argument("--horizon", type=int, default=21,
                    help="forward window the predictive test scores")
    ap.add_argument("--no-test", action="store_true",
                    help="skip the predictive test (renders the chart alone)")
    ap.add_argument("--chain-json", dest="chain_json", default="",
                    help="option chain (run_live --dump schema) — adds the OPTIONS "
                         "decision and order ticket, the validated half of the engine")
    ap.add_argument("--price-json", dest="price_json", default="",
                    help="price history for the chain's underlying (needed for P)")
    ap.add_argument("--symbol", default="", help="underlying inside --chain-json")
    ap.add_argument("--dte", type=int, default=30, help="tenor for the options panel")
    ap.add_argument("--equity", type=float, default=100_000.0,
                    help="account equity the order ticket sizes against")
    ap.add_argument("--max-risk-frac", dest="max_risk_frac", type=float, default=0.02,
                    help="fraction of equity the ticket may put at MAXIMUM LOSS")
    ap.add_argument("--ledger", default="",
                    help="paper-trade ledger; renders the FORWARD TEST panel (P vs Q "
                         "over time) and supplies the ticket's evidence label, read "
                         "from disk and never typed")
    a = ap.parse_args(argv)

    if a.demo or not a.csv:
        px, bench, dates = demo_world()
        if not a.demo:
            print("[demo] no --csv given, rendering a generated fixture")
    else:
        px, bench, dates = load_csv(a.csv)

    opts = options_panel(a.chain_json, a.price_json, symbol=a.symbol, dte=a.dte,
                         equity=a.equity, max_risk_frac=a.max_risk_frac,
                         ledger=a.ledger)
    payload = build_payload(px, bench, dates, window=a.window, mom_lag=a.mom_lag,
                            tail=a.tail, horizon=a.horizon, run_test=not a.no_test,
                            options=opts,
                            ledger=ledger_panel(a.ledger, dte_hint=a.dte))
    # utf-8 explicitly: the page carries em dashes and Python would otherwise
    # write them in the machine's own codepage, which the browser has no way to
    # know about. A dashboard that renders as mojibake on a non-English Windows
    # is a bug that only shows up on someone else's desk.
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write(render(payload))
    print(f"wrote {a.out}  ({len(payload['points'])} sectors, {payload['bars']} bars)")
    if payload["test"]:
        print("  panel: " + payload["test"]["panelVerdict"])
        print("  book : " + payload["test"]["bookVerdict"])
    o = payload["options"]
    if o.get("available"):
        print(f"  options: {o['action']} size x{o['sizeMultiplier']:.2f} -> "
              + (f"{o['contracts']} x {o['structure']} "
                 f"(risk {o['maxLossTotal']:,.0f})" if o["placeable"]
                 else "REFUSED: " + ", ".join(r["code"] for r in o["refusals"])))
    else:
        print("  options: not on the page — " + o.get("reason", ""))
    f = payload["ledger"]
    print(f"  forward: {f['recorded']} recorded, {f['settled']} settled"
          if f.get("available") else "  forward: EMPTY — " + f.get("reason", ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
