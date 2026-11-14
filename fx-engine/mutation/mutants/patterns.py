"""Ways the detectors could fire on something that is not the pattern.

The look-ahead mutants matter most: a detector that reports a formation one bar
before it completed makes every downstream number optimistic, and the only
thing that catches it is truncating the series and checking the answer does not
change.
"""

TARGET = "models/patterns.py"
TESTS = "tests/test_patterns.py"

MUTANTS = [
 ("swing needs no left bars",   "        if all(h > bars[i - j].h for j in range(1, k + 1)) and \\", "        if all(True for j in range(1, k + 1)) and \\"),
 ("swing window ignores k",     "    for i in range(k, len(bars) - k):\n        h = bars[i].h", "    for i in range(0, len(bars)):\n        h = bars[i].h"),
 ("swing bars reduced to one",  "SWING_BARS = 3", "SWING_BARS = 1"),
 ("equality tolerance opened",  "EQUAL_TOL = 0.0015", "EQUAL_TOL = 0.5"),
 ("equality tolerance closed",  "EQUAL_TOL = 0.0015", "EQUAL_TOL = 0.0"),
 ("closeness not relative",     "    return abs(a - b) / ref <= tol", "    return abs(a - b) <= tol"),
 ("height floor removed",       "MIN_HEIGHT = 0.002", "MIN_HEIGHT = 0.0"),
 ("span floor removed",         "MIN_SPAN, MAX_SPAN = 10, 120", "MIN_SPAN, MAX_SPAN = 0, 120"),
 ("span ceiling removed",       "MIN_SPAN, MAX_SPAN = 10, 120", "MIN_SPAN, MAX_SPAN = 10, 100000"),
 ("parameter count understated","PARAMETER_COUNT = 4", "PARAMETER_COUNT = 1"),
 ("h&s completion looks ahead", "        for i in range(c + 1, min(c + MAX_SPAN, len(bars))):\n            if bars[i].c < neck:", "        for i in range(c, min(c + MAX_SPAN, len(bars))):\n            if bars[i].c < neck:"),
 ("h&s head need not be higher","        if not (head > ls and head > rs):", "        if False:"),
 ("h&s shoulders need not match","        if not _close_enough(ls, rs):", "        if False:"),
 ("flag cannot fire on the last","    for i in range(pole + 4, len(bars)):", "    for i in range(pole + 4, len(bars) - 1):"),
 ("engulfing cannot fire last", "    for i in range(1, len(bars)):        # see the note in flag() on the bound", "    for i in range(1, len(bars) - 1):        # see the note in flag() on the bound"),
 ("scan loses bar order",       "    return sorted(found, key=lambda d: d.index)", "    return found"),
]
