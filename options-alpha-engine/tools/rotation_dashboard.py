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


def build_payload(px: dict, bench: list, dates: list, *, window: int, mom_lag: int,
                  tail: int, horizon: int, spark: int = 120, run_test: bool = True):
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
          t.placeboReturn===null?"":` &middot; label-shuffled placebo
          <b class="num" style="color:var(--ink)">${(t.placeboReturn*100).toFixed(1)}%</b>`}</p>
        <p class="hint">${t.bookVerdict}</p>
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

drawRRG(); drawBoard(); drawTest(); drawSparks(); link();
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
    a = ap.parse_args(argv)

    if a.demo or not a.csv:
        px, bench, dates = demo_world()
        if not a.demo:
            print("[demo] no --csv given, rendering a generated fixture")
    else:
        px, bench, dates = load_csv(a.csv)

    payload = build_payload(px, bench, dates, window=a.window, mom_lag=a.mom_lag,
                            tail=a.tail, horizon=a.horizon, run_test=not a.no_test)
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
