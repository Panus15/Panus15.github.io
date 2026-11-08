"""Ways a rule you wrote could get through unchecked.

The look-ahead check is the one that matters. With it switched off, a one-bar
peek measured 4,312 trades at +3.49 pips each on a random walk and came back
TRADEABLE, so every mutant that weakens it is a mutant that ships that number.
"""

TARGET = "engine/rules.py"
TESTS = "tests/test_rules.py"

MUTANTS = [
 # --- the look-ahead check ---
 ("look-ahead never checked",   "        problems = check_look_ahead(rule.detect, bars)\n        if problems:", "        problems = []\n        if problems:"),
 ("look-ahead check skipped",   "    if check and bars:", "    if False:"),
 ("truncation span collapsed",  "LOOK_AHEAD_BARS = 40", "LOOK_AHEAD_BARS = 0"),
 ("only the full series tried", "    for cut in range(n - span, n + 1):", "    for cut in range(n, n + 1):"),
 ("missing detections ignored", "            if i not in visible:", "            if False:"),
 ("comparison inverted",        "            if i not in visible:", "            if i in visible:"),
 ("future detections compared", "        for i in sorted(d for d in full if d < cut):", "        for i in sorted(full):"),
 ("short series passes quietly","    if n < span + 2:", "    if False:"),
 ("complaint loses the bar",    "                    f\"a detection at bar {i} disappears when the series is \"", "                    f\"something is wrong \"  # "),
 # --- the declaration that cannot be guessed ---
 ("parameter count not needed", "    if not isinstance(pc, int) or isinstance(pc, bool) or pc < 1:", "    if False:"),
 ("zero parameters allowed",    "    if not isinstance(pc, int) or isinstance(pc, bool) or pc < 1:", "    if not isinstance(pc, int) or isinstance(pc, bool) or pc < 0:"),
 ("booleans count as integers", "    if not isinstance(pc, int) or isinstance(pc, bool) or pc < 1:", "    if not isinstance(pc, int) or pc < 1:"),
 ("floats count as integers",   "    if not isinstance(pc, int) or isinstance(pc, bool) or pc < 1:", "    if not isinstance(pc, (int, float)) or pc < 1:"),
 ("reason for the count hidden","            f\"it is the denominator of the multiple-testing correction, and this \"", "            f\"\"  # "),
 # --- the file itself ---
 ("detect not required",        "    if not hasattr(mod, \"detect\") or not callable(mod.detect):", "    if False:"),
 ("callable not required",      "    if not hasattr(mod, \"detect\") or not callable(mod.detect):", "    if not hasattr(mod, \"detect\"):"),
 ("missing file not caught",    "    if not os.path.exists(path):", "    if False:"),
 ("declared name ignored",      "    rule = Rule(name=name or getattr(mod, \"NAME\", None)", "    rule = Rule(name=name or None"),
 ("explicit name ignored",      "    rule = Rule(name=name or getattr(mod, \"NAME\", None)", "    rule = Rule(name=getattr(mod, \"NAME\", None)"),
 # --- the shape checks ---
 ("shape never checked",        "        check_shape(rule.detect(bars), len(bars))", "        pass"),
 ("direction not checked",      "        if d.direction not in (1, -1):", "        if False:"),
 ("index range not checked",    "        if not 0 <= d.index < n_bars:", "        if False:"),
 ("long levels not checked",    "        if d.direction > 0 and not (d.stop < d.entry < d.target):", "        if False:"),
 ("short levels not checked",   "        if d.direction < 0 and not (d.target < d.entry < d.stop):", "        if False:"),
 ("type not checked",           "        if not isinstance(d, Detection):", "        if False:"),
 ("return type not checked",    "    if not isinstance(detections, list):", "    if False:"),
 # --- what the engine claims to know ---
 ("your rule called untested",  "        return (YOUR_OWN_RULE,", "        return (\"never tested in FX\","),
 ("evidence drops the caveat",  "                f\"parameter(s). Nothing is known about it until it has been \"\n                f\"measured on data it was not built from\")", "                f\"parameter(s).\")"),
 ("source not recorded",        "                f\"loaded from {self.source}, declaring {self.parameter_count} \"", "                f\"loaded from somewhere, declaring {self.parameter_count} \""),
]
