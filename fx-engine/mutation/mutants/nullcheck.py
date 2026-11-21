"""Ways the null check could flatter the file it is supposed to doubt.

The dangerous direction is one-sided. A null that is too EASY makes every
result look real, which is the failure this tool was written to fix: a
constant-sigma walk found 13 detections where the real file found 33, and the
gap read as signal when it was volatility clustering.

So most of these break the resampler's faithfulness — the moves, the wicks, the
block structure — and the rest break the report's refusals to quote a rate it
has not earned.
"""

TARGET = "tools/nullcheck.py"
TESTS = "tests/test_nullcheck.py"

MUTANTS = [
 # The resampler. Each of these makes the null a different market from the file.
 ("wicks dropped",               "    return [(bars[i].c - bars[i - 1].c,\n             bars[i].h - max(bars[i].o, bars[i].c),\n             min(bars[i].o, bars[i].c) - bars[i].l)\n            for i in range(1, len(bars))]", "    return [(bars[i].c - bars[i - 1].c, 0.0, 0.0)\n            for i in range(1, len(bars))]"),
 ("moves taken from opens",      "    return [(bars[i].c - bars[i - 1].c,", "    return [(bars[i].o - bars[i - 1].o,"),
 ("blocks become single bars",   "        seq.extend(moves[j:j + block])", "        seq.append(moves[j])"),
 ("block structure reversed",    "        j = rnd.randrange(0, max(len(moves) - block, 0) + 1)", "        j = max(len(moves) - block, 0)"),
 ("resample runs long",          "    for d, up, dn in seq[:len(moves)]:", "    for d, up, dn in seq:"),
 ("bars built without range",    "        out.append(Bar(o, max(o, c) + up, min(o, c) - dn, c))", "        out.append(Bar(o, max(o, c), min(o, c), c))"),
 ("high and low swapped",        "        out.append(Bar(o, max(o, c) + up, min(o, c) - dn, c))", "        out.append(Bar(o, max(o, c) + up, min(o, c) + dn, c))"),
 ("zero block allowed",          "    if block < 1:", "    if False:"),
 ("every null the same draw",    "            rnd = random.Random(seed + s)", "            rnd = random.Random(seed)"),
 # The sweep's bookkeeping.
 ("floor refusals not counted",  "            if TOO_FEW_TRADES in v.codes:\n                floored += 1", "            if False:\n                floored += 1"),
 ("passes counted as refusals",  "            if v.tradeable:\n                passed += 1", "            if not v.tradeable:\n                passed += 1"),
 ("nulls judged unconfirmed",    "                      costs_confirmed=True)", "                      costs_confirmed=False)"),
 ("median becomes the max",      "                     statistics.median(ns) if ns else 0.0,", "                     max(ns) if ns else 0.0,"),
 # The report's refusals. These are the ones that would let a reader quote a
 # rate the sweep never measured.
 ("incomparable nulls quoted",   "    if not comparable:", "    if False:"),
 ("comparability band widened",  "    comparable = [r for r in rows if real_n and 0.5 <= r[4] / real_n <= 2.0]", "    comparable = [r for r in rows if real_n and 0.01 <= r[4] / real_n <= 100.0]"),
 ("best block quoted not worst", "    worst = max(comparable, key=lambda r: r[1])", "    worst = min(comparable, key=lambda r: r[1])"),
 ("floor-only zero not flagged", "    if strict[3] == strict[2]:", "    if False:"),
 ("long block caveat dropped",   "        if worst[0] > MAX_SPAN + MAX_HOLD_BARS:", "        if False:"),
 ("low rate read as proof",      '        out.append(f"  Below one in ten, so this particular artifact does not "\n                   f"account for your result. It is not evidence the edge is "\n                   f"real: other artifacts exist, and the hold-out is still "\n                   f"the test that matters.")', '        out.append(f"  Below one in ten.")'),
 ("rate threshold raised",       "    if worst[1] / worst[2] >= 0.1:", "    if worst[1] / worst[2] >= 0.9:"),
 # The sweep's span. A sweep that never reaches a strict null cannot refute
 # anything, and one that never reaches a loose one cannot be read either.
 ("sweep loses its strict end",  "DEFAULT_BLOCKS = (30, 240, 1440)", "DEFAULT_BLOCKS = (240, 1440)"),
 ("sweep loses its loose end",   "DEFAULT_BLOCKS = (30, 240, 1440)", "DEFAULT_BLOCKS = (30, 240)"),
]
