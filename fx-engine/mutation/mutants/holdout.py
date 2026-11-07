"""Ways the out-of-sample test could confirm something it has not tested.

The cut must be in time, only survivors may be re-tested, and the correction
denominator must change. Each of those can be broken while the report still
reads as a clean confirmation.
"""

TARGET = "engine/holdout.py"
TESTS = "tests/test_holdout.py"

MUTANTS = [
 ("split shuffles the series",  "    cut = int(len(bars) * frac)\n    return bars[:cut], bars[cut:]", "    import random as _r\n    b = list(bars); _r.Random(0).shuffle(b)\n    cut = int(len(bars) * frac)\n    return b[:cut], b[cut:]"),
 ("halves swapped",             "    return bars[:cut], bars[cut:]", "    return bars[cut:], bars[:cut]"),
 ("frac not validated",         "    if not 0.0 < frac < 1.0:", "    if False:"),
 ("split fraction ignored",     "    cut = int(len(bars) * frac)", "    cut = int(len(bars) * 0.5)"),
 ("everything re-tested",       "    survivors = out.survivors", "    survivors = list(results)"),
 ("nothing re-tested",          "    for r in survivors:\n        fn = chosen[r.name]", "    for r in []:\n        fn = chosen[r.name]"),
 ("confirmation always true",   "        r.confirmed = r.out_of_sample.tradeable", "        r.confirmed = True"),
 ("confirmation inverted",      "        r.confirmed = r.out_of_sample.tradeable", "        r.confirmed = not r.out_of_sample.tradeable"),
 ("oos reuses the full search", "                                n_hypotheses=len(survivors),", "                                n_hypotheses=len(chosen) * 4,"),
 ("oos drops the correction",   "                                n_hypotheses=len(survivors),", "                                n_hypotheses=1,"),
 ("in-sample width collapsed",  "        v = judge(res, costs, name=name, n_hypotheses=len(chosen) * 4,", "        v = judge(res, costs, name=name, n_hypotheses=1,"),
 ("tiny holdout still tested",  "    if not survivors or len(test) < MIN_HOLDOUT_BARS:", "    if not survivors:"),
 ("holdout floor lowered",      "MIN_HOLDOUT_BARS = 500", "MIN_HOLDOUT_BARS = 50"),
 ("tiny holdout not reported",  "        if self.n_test < MIN_HOLDOUT_BARS:", "        if False:"),
 ("trains on the later half",   "        res = run(train, fn(train), costs, bars_per_night=bars_per_night)", "        res = run(test, fn(test), costs, bars_per_night=bars_per_night)"),
 ("tests on the earlier half",  "        res = run(test, fn(test), costs, bars_per_night=bars_per_night)\n        r.out_of_sample", "        res = run(train, fn(train), costs, bars_per_night=bars_per_night)\n        r.out_of_sample"),
 ("single-use warning dropped", "        lines.append(\"YOU GET ONE. The held-back half stopped being \"", "        lines.append(\"\") or lines.append(\"x\") if False else lines.append(\"\")  # "),
 ("failure called a malfunction","        if not self.confirmed:\n            lines.append(\"A pattern that passes", "        if False:\n            lines.append(\"A pattern that passes"),
 ("untouched note dropped",     "        if not self.survivors:\n            lines.append(\"\")", "        if False:\n            lines.append(\"\")"),
 ("status ignores confirmation","        return \"CONFIRMED\" if self.confirmed else \"FAILED out-of-sample\"", "        return \"CONFIRMED\""),
 ("untested status wrong",      "        if self.out_of_sample is None:\n            return \"not tested out-of-sample\"", "        if self.out_of_sample is None:\n            return \"CONFIRMED\""),
 ("reported split size faked",  "    out = Holdout(frac=frac, n_train=len(train), n_test=len(test),", "    out = Holdout(frac=frac, n_train=int(len(bars) * 0.7), n_test=len(test),"),
 ("reported frac faked",        "    out = Holdout(frac=frac, n_train=len(train), n_test=len(test),", "    out = Holdout(frac=0.7, n_train=len(train), n_test=len(test),"),
]
