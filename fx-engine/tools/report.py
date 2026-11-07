"""A report you can look at — built so that looking at it cannot mislead you.

`analyse.py` prints a table. A table is the right output for a refusal, and the
wrong one for seeing what a rule did over time. This renders the same analysis
as a self-contained HTML page with inline SVG.

NO JAVASCRIPT, AND THAT IS A DECISION. The options engine's dashboard builds its
SVG in a browser from a Python template, and that cost real time: a `\\"` inside
a non-raw triple-quoted template was eaten by Python, shipped a bare quote, and
blanked the ENTIRE page while every test still passed. It was found by running
the page's JavaScript in node, which is not a thing a test suite does by
accident. Here the SVG is generated in Python and the page has no script at all,
so the output can be checked by parsing it.

THE CHART PLOTS NET, AND GROSS ONLY AS A GHOST. This is the whole argument made
visible. A gross equity curve is the chart every losing system shows you: it
slopes up, it is arithmetically correct, and it describes money nobody received.
Here the solid line is after costs and the faint line is before, so the GAP
between them is what the broker took — drawn rather than described. If the two
lines diverge into a widening wedge, that wedge is the strategy's real problem
and no amount of signal work narrows it.

FOUR WAYS A CHART LIES, AND WHAT IS DONE ABOUT EACH.

1. A CONFIDENT LINE FROM A TINY SAMPLE. Twelve trades plotted smoothly look
   exactly like twelve hundred. Any pattern whose verdict carries
   TOO_FEW_TRADES is drawn, and then labelled over the top with its trade count
   and the fact that the shape means nothing — because the eye believes a line
   long after the reader has read the warning.

2. AUTOSCALING THAT FLATTERS. A y-axis fitted to the data makes any wiggle look
   like a trend. Zero is always on the axis, so a curve that never gets above
   water cannot be framed as if it did.

3. NaN IN THE PATH DATA. One non-finite coordinate makes an SVG polyline vanish
   with no error anywhere — a blank chart that reads as "no trades". Every point
   is checked before it is written, and a series with a bad point is refused by
   name rather than silently dropped.

4. ORDERING BY WHAT LOOKS BEST. Patterns are listed in the engine's own
   detector order, never sorted by return, because sorting by outcome is how a
   report turns six hypotheses into one cherry-picked winner.
"""

from __future__ import annotations

import html
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.verdict import TOO_FEW_TRADES

#: Chart geometry, in SVG user units. The page scales them to its width.
W, H = 640, 180
PAD_L, PAD_R, PAD_T, PAD_B = 48, 12, 14, 22

#: Below this many trades a curve's shape is noise, and the chart says so over
#: the top of the line rather than beside it.
SHAPE_MEANINGLESS_BELOW = 30


def _esc(s) -> str:
    """Everything that reaches the page goes through here.

    Pattern names come from this engine and are safe; file names come from the
    caller and are not. Escaping only the ones that look risky is how the other
    one gets through.
    """
    return html.escape(str(s), quote=True)


def _finite(*vals) -> bool:
    return all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals)


def _points(series: list, lo: float, hi: float) -> str:
    """Polyline points for a y-series indexed 0..n-1, scaled into the frame.

    Raises on a non-finite coordinate. An SVG polyline containing `nan`
    disappears with no error in any renderer, which reads as "no trades" — so
    this refuses by name instead of drawing nothing.
    """
    n = len(series)
    if n < 2:
        return ""
    span = hi - lo
    if not _finite(lo, hi) or span <= 0:
        raise ValueError(f"cannot scale a y-range of [{lo!r}, {hi!r}]")
    iw = W - PAD_L - PAD_R
    ih = H - PAD_T - PAD_B
    out = []
    for i, v in enumerate(series):
        if not _finite(v):
            raise ValueError(
                f"non-finite value at index {i}: {v!r}. An SVG polyline with a "
                f"non-finite point renders as nothing at all, which looks "
                f"identical to having no trades")
        x = PAD_L + iw * i / (n - 1)
        y = PAD_T + ih * (1.0 - (v - lo) / span)
        out.append(f"{x:.1f},{y:.1f}")
    return " ".join(out)


def equity_svg(trades: list, label: str = "") -> str:
    """Cumulative pips: NET solid, GROSS as a ghost. The gap is the cost."""
    if not trades:
        return ('<p class="none">no trades — nothing to draw, which is itself '
                'the result</p>')
    net, gross, a, b = [], [], 0.0, 0.0
    for t in trades:
        a += t.net_pips
        b += t.pips
        net.append(a)
        gross.append(b)

    # zero is always on the axis: a curve that never got above water must not be
    # framed as though it did
    lo = min(0.0, min(net), min(gross))
    hi = max(0.0, max(net), max(gross))
    if hi - lo <= 0:
        lo, hi = -1.0, 1.0
    pad = (hi - lo) * 0.08
    lo, hi = lo - pad, hi + pad

    iw = W - PAD_L - PAD_R
    zero_y = PAD_T + (H - PAD_T - PAD_B) * (1.0 - (0.0 - lo) / (hi - lo))
    parts = [f'<svg viewBox="0 0 {W} {H}" class="eq" role="img" '
             f'aria-label="{_esc(label)} cumulative pips, net of costs">']
    parts.append(f'<line class="axis" x1="{PAD_L}" y1="{zero_y:.1f}" '
                 f'x2="{PAD_L + iw}" y2="{zero_y:.1f}"/>')
    parts.append(f'<text class="tick" x="{PAD_L - 6}" y="{PAD_T + 8}">'
                 f'{hi:+,.0f}</text>')
    parts.append(f'<text class="tick" x="{PAD_L - 6}" y="{H - PAD_B + 4}">'
                 f'{lo:+,.0f}</text>')
    parts.append(f'<text class="tick" x="{PAD_L - 6}" y="{zero_y + 3:.1f}">0</text>')
    parts.append(f'<polyline class="gross" points="{_points(gross, lo, hi)}"/>')
    parts.append(f'<polyline class="net" points="{_points(net, lo, hi)}"/>')
    parts.append(f'<text class="xlab" x="{PAD_L}" y="{H - 4}">trade 1</text>')
    parts.append(f'<text class="xlab end" x="{PAD_L + iw}" y="{H - 4}">'
                 f'trade {len(trades):,}</text>')
    if len(trades) < SHAPE_MEANINGLESS_BELOW:
        parts.append(f'<text class="overlay" x="{PAD_L + iw / 2:.0f}" '
                     f'y="{H / 2:.0f}">{len(trades)} trades — the shape of this '
                     f'line means nothing</text>')
    parts.append('</svg>')
    return "".join(parts)


def winrate_svg(v) -> str:
    """Measured gross win rate against the rate this payoff and cost require.

    One bar each, same baseline. Where measured falls short of required, the
    refusal is the picture rather than a sentence underneath it.
    """
    got, need = v.gross.win_rate, v.win_rate_needed
    if not _finite(got, need) or need <= 0:
        return '<p class="none">no win rate to compare</p>'
    top = max(got, need, 1.0)
    bw = 260
    gw, nw = bw * got / top, bw * min(need, top) / top
    short = need > got
    return (
        f'<svg viewBox="0 0 400 54" class="wr" role="img" aria-label="'
        f'measured win rate {got:.1%} against {need:.1%} required">'
        f'<rect class="bar got" x="110" y="6" width="{gw:.1f}" height="16"/>'
        f'<rect class="bar {"need short" if short else "need"}" x="110" y="30" '
        f'width="{nw:.1f}" height="16"/>'
        f'<text class="blab" x="104" y="18">measured</text>'
        f'<text class="blab" x="104" y="42">needed</text>'
        f'<text class="bval" x="{110 + gw + 6:.1f}" y="18">{got:.1%}</text>'
        f'<text class="bval" x="{110 + nw + 6:.1f}" y="42">{need:.1%}'
        f'{f" ({need - got:+.1%})" if short else ""}</text>'
        f'</svg>')


CSS = """
:root{--bg:#fff;--fg:#1a1a1a;--dim:#6b7280;--line:#e5e7eb;
--net:#1d4ed8;--gross:#9ca3af;--ok:#059669;--bad:#dc2626;--warn:#b45309}
@media(prefers-color-scheme:dark){:root{--bg:#0f1115;--fg:#e8eaed;--dim:#9aa0a6;
--line:#2a2f3a;--net:#7aa2ff;--gross:#5b6270;--ok:#34d399;--bad:#f87171;
--warn:#fbbf24}}
*{box-sizing:border-box}
body{margin:0;padding:24px 16px 56px;background:var(--bg);color:var(--fg);
font:15px/1.55 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
max-width:860px;margin-inline:auto}
h1{font-size:21px;margin:0 0 2px}
h2{font-size:16px;margin:34px 0 10px;padding-bottom:6px;
border-bottom:1px solid var(--line)}
h3{font-size:14px;margin:22px 0 4px;font-family:ui-monospace,monospace}
.sub{color:var(--dim);font-size:13px;margin:0 0 4px}
table{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0}
th,td{text-align:right;padding:5px 7px;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left;font-family:ui-monospace,monospace}
th{color:var(--dim);font-weight:600}
td.v{font-family:ui-monospace,monospace}
.pass{color:var(--ok);font-weight:600}
.fail{color:var(--bad)}
.warn{color:var(--warn)}
svg{display:block;width:100%;height:auto;overflow:visible}
.eq{margin:2px 0 4px}
.axis{stroke:var(--line);stroke-width:1}
.net{fill:none;stroke:var(--net);stroke-width:2;stroke-linejoin:round}
.gross{fill:none;stroke:var(--gross);stroke-width:1.5;stroke-dasharray:3 3}
.tick,.xlab,.blab,.bval{fill:var(--dim);font:10px ui-monospace,monospace}
.tick,.blab{text-anchor:end}
.xlab.end{text-anchor:end}
.overlay{fill:var(--warn);font:bold 12px ui-sans-serif,system-ui,sans-serif;
text-anchor:middle}
.bar.got{fill:var(--net)}
.bar.need{fill:var(--gross)}
.bar.need.short{fill:var(--bad)}
.legend{color:var(--dim);font-size:12px;margin:0 0 2px}
.legend b{color:var(--net)}
.legend i{color:var(--gross);font-style:normal}
.none{color:var(--dim);font-size:13px;font-style:italic}
.ref{color:var(--bad);font-size:12.5px;margin:4px 0 0}
.note{color:var(--dim);font-size:12.5px;margin:4px 0 0}
footer{margin-top:40px;padding-top:12px;border-top:1px solid var(--line);
color:var(--dim);font-size:12px}
"""


def render(series, results: dict, judged: dict, costs, bars_per_night: float,
           n_hypotheses: int) -> str:
    """The whole page.

    ``results`` maps pattern name -> BacktestResult and ``judged`` maps the same
    names -> Verdict, both in the engine's detector order. They are passed
    separately and checked against each other because the first version read the
    trades off the Verdict with a defaulted `getattr` — and Verdict has no
    trades, so every chart rendered "no trades" with nothing raised anywhere. A
    default that papers over a missing field is indistinguishable from the field
    being legitimately empty, which is the one thing a chart must not confuse.
    """
    if set(results) != set(judged):
        raise ValueError(
            f"results and verdicts disagree about which patterns were run: "
            f"{sorted(set(results) ^ set(judged))}")
    out = [
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">",
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">",
        f"<title>{_esc(series.pair)} — falsification report</title>",
        f"<style>{CSS}</style></head><body>",
        f"<h1>{_esc(series.pair)} — falsification report</h1>",
        f"<p class=\"sub\">{series.n:,} bars at the {_esc(series.side.upper())}"
        + (f", {series.times[0]:%Y-%m-%d} to {series.times[-1]:%Y-%m-%d}"
           if series.times else "")
        + f" · {costs.round_turn_pips:g} pip round turn"
        f" · {costs.swap_markup_annual:.2%}/yr markup"
        f" · {bars_per_night:g} bars per rollover</p>",
        f"<p class=\"sub\">Corrected for {n_hypotheses} hypotheses — a FLOOR, "
        f"since the honest count includes every variant tried and dropped.</p>",
    ]

    if series.problems:
        out.append("<h2>What is wrong with the data</h2><table>"
                   "<tr><th>problem</th><th>count</th></tr>")
        for p in series.problems:
            out.append(f"<tr><td>{_esc(p.code)}</td><td class=\"v\">"
                       f"{p.count:,}</td></tr>")
        out.append("</table>")
        out.append(f"<p class=\"note\">{_esc(str(series.problems[0].detail))}</p>")

    out.append("<h2>Verdicts</h2><table><tr><th>pattern</th><th>trades</th>"
               "<th>win%</th><th>needed%</th><th>pips/trade</th>"
               "<th>verdict</th></tr>")
    for name, v in judged.items():
        cls = "pass" if v.tradeable else "fail"
        head = "TRADEABLE" if v.tradeable else (v.codes[0] if v.codes else "-")
        out.append(
            f"<tr><td>{_esc(name)}</td><td class=\"v\">{v.n_trades:,}</td>"
            f"<td class=\"v\">{v.gross.win_rate:.1%}</td>"
            f"<td class=\"v\">{v.win_rate_needed:.1%}</td>"
            f"<td class=\"v\">{v.net.net:+.2f}</td>"
            f"<td class=\"{cls}\">{_esc(head)}</td></tr>")
    out.append("</table>")

    out.append("<h2>What each one did, after costs</h2>")
    out.append("<p class=\"legend\"><b>— solid: net of costs</b> · "
               "<i>- - dashed: before costs, i.e. money nobody received</i>. "
               "The gap between them is what the broker took.</p>")
    for name, v in judged.items():
        out.append(f"<h3>{_esc(name)}</h3>")
        out.append(equity_svg(results[name].trades, name))
        out.append(winrate_svg(v))
        if v.refusals:
            out.append(f"<p class=\"ref\">{_esc(str(v.refusals[0]))}</p>")
        if TOO_FEW_TRADES in v.codes:
            out.append("<p class=\"note\">Every number above this line is "
                       "computed from that sample, so every number above this "
                       "line inherits its width.</p>")
    out.append(
        "<footer>Nothing here is a prediction, a recommendation, or a bound on "
        "your loss. A pass is a hypothesis that survived one in-sample test; "
        "run <code>--holdout</code> before believing it. A stop does not bound "
        "the loss: on 2015-01-15 EUR/CHF moved more than 30% with no quotes in "
        "between.</footer></body></html>")
    return "".join(out)
