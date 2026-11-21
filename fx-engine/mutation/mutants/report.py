"""Ways the chart could lie while still rendering perfectly."""

TARGET = "tools/report.py"
TESTS = "tests/test_report.py"

MUTANTS = [
 # --- the argument, drawn the right way round ---
 ("net and gross swapped",      '    parts.append(f\'<polyline class="gross" points="{_points(gross, lo, hi)}"/>\')\n    parts.append(f\'<polyline class="net" points="{_points(net, lo, hi)}"/>\')', '    parts.append(f\'<polyline class="gross" points="{_points(net, lo, hi)}"/>\')\n    parts.append(f\'<polyline class="net" points="{_points(gross, lo, hi)}"/>\')'),
 ("net plotted as gross",       '        net.append(a)\n        gross.append(b)', '        net.append(b)\n        gross.append(b)'),
 ("cost not subtracted",        "        a += t.net_pips", "        a += t.pips"),
 ("y axis inverted",            "        y = PAD_T + ih * (1.0 - (v - lo) / span)", "        y = PAD_T + ih * ((v - lo) / span)"),
 # --- zero on the axis ---
 ("zero dropped from scale",    "    lo = min(0.0, min(net), min(gross))\n    hi = max(0.0, max(net), max(gross))", "    lo = min(min(net), min(gross))\n    hi = max(max(net), max(gross))"),
 ("zero line not drawn",        '    parts.append(f\'<line class="axis" x1="{PAD_L}" y1="{zero_y:.1f}" \'', '    parts.append(f\'<!-- {PAD_L} {zero_y:.1f} \''),
 # --- a blank chart must not read as an empty one ---
 ("non-finite points allowed",  "        if not _finite(v):", "        if False:"),
 ("degenerate range allowed",   "    if not _finite(lo, hi) or span <= 0:", "    if False:"),
 ("no-trades draws a frame",    "    if not trades:", "    if False:"),
 ("one point makes a line",     "    if n < 2:", "    if n < 1:"),
 # --- a small sample must not look big ---
 ("small-sample overlay gone",  "    if len(trades) < SHAPE_MEANINGLESS_BELOW:", "    if False:"),
 ("overlay threshold lowered",  "SHAPE_MEANINGLESS_BELOW = 30", "SHAPE_MEANINGLESS_BELOW = 3"),
 ("overlay always shown",       "    if len(trades) < SHAPE_MEANINGLESS_BELOW:", "    if True:"),
 # --- the win-rate bars ---
 ("shortfall not marked",       "    short = need > got", "    short = False"),
 ("shortfall test inverted",    "    short = need > got", "    short = need < got"),
 ("bars compare net not gross", "    got, need = v.gross.win_rate, v.win_rate_needed", "    got, need = v.net.win_rate, v.win_rate_needed"),
 # --- the page ---
 ("mismatch not caught",        "    if set(results) != set(judged):", "    if False:"),
 ("patterns sorted by return",  "    for name, v in judged.items():\n        out.append(f\"<h3>{_esc(name)}</h3>\")", "    for name, v in sorted(judged.items(), key=lambda kv: -kv[1].net.net):\n        out.append(f\"<h3>{_esc(name)}</h3>\")"),
 ("escaping removed",           "    return html.escape(str(s), quote=True)", "    return str(s)"),
 ("correction count hidden",    "        f\"<p class=\\\"sub\\\">Corrected for {n_hypotheses} hypotheses — a FLOOR, \"", "        f\"<p class=\\\"sub\\\">Corrected. \" + \"\" * 0 + \""),
 ("problems section dropped",   "    if series.problems:", "    if False:"),
 ("footer caveats dropped",     '        "<footer>Nothing here is a prediction, a recommendation, or a bound on "', '        "<footer>"  # '),
 ("verdict class always pass",  '        cls = "pass" if v.tradeable else "fail"', '        cls = "pass"'),
]
