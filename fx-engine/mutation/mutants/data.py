"""Ways the loader could pass corrupt data through as clean.

Frozen quotes are the one worth the module: a dead feed written as a flat
market, which detectors read as consolidation and backtests fill inside for
nothing. None of these defects raises anything.
"""

TARGET = "engine/data.py"
TESTS = "tests/test_data.py"

MUTANTS = [
 ("frozen never detected",       "        if b.o == b.h == b.l == b.c:", "        if False:"),
 ("any flat bar counts",         "            if run >= FROZEN_RUN:\n                frozen.extend(range(start, i))", "            if run >= 1:\n                frozen.extend(range(start, i))"),
 ("run threshold raised",        "FROZEN_RUN = 3", "FROZEN_RUN = 6"),
 ("trailing run dropped",        "    if run >= FROZEN_RUN:\n        frozen.extend(range(start, n))", "    if False:\n        frozen.extend(range(start, n))"),
 ("frozen never disqualifies",   "        return self.count(FROZEN_QUOTES) / self.n <= FROZEN_UNUSABLE_FRAC", "        return True"),
 ("frozen limit loosened",       "FROZEN_UNUSABLE_FRAC = 0.05", "FROZEN_UNUSABLE_FRAC = 0.5"),
 ("high check dropped",          "           if not (b.h >= max(b.o, b.c) and b.l <= min(b.o, b.c) and b.h >= b.l)]", "           if not (b.l <= min(b.o, b.c) and b.h >= b.l)]"),
 ("low check dropped",           "           if not (b.h >= max(b.o, b.c) and b.l <= min(b.o, b.c) and b.h >= b.l)]", "           if not (b.h >= max(b.o, b.c) and b.h >= b.l)]"),
 ("violations not fatal",        "        fatal = (OHLC_VIOLATION, DUPLICATE_TIMESTAMP, OUT_OF_ORDER,\n                 NON_POSITIVE_PRICE)", "        fatal = (DUPLICATE_TIMESTAMP, OUT_OF_ORDER,\n                 NON_POSITIVE_PRICE)"),
 ("zero prices allowed",         "              if min(b.o, b.h, b.l, b.c) <= 0 or any(", "              if min(b.o, b.h, b.l, b.c) < 0 or any("),
 ("first index not recorded",    "            \"these produce fills that could not have happened\", bad[0]))", "            \"these produce fills that could not have happened\"))"),
 ("duplicates not detected",     "        dupes = [i for i in range(1, n) if times[i] == times[i - 1]]", "        dupes = []"),
 ("backwards not detected",      "        back = [i for i in range(1, n) if times[i] < times[i - 1]]", "        back = []"),
 ("weekend counted as missing",  "                if spans_saturday:", "                if False:"),
 ("everything called weekend",   "                if spans_saturday:", "                if True:"),
 ("sunday mistaken for saturday","                    (times[i - 1] + timedelta(days=k)).weekday() == 5", "                    (times[i - 1] + timedelta(days=k)).weekday() == 6"),
 ("gaps not detected at all",    "                if d <= interval:\n                    continue", "                if True:\n                    continue"),
 ("irregular never reported",    "            if odd > len(deltas) * 0.5:", "            if False:"),
 ("side not validated",          "    if side not in SIDES:", "    if False:"),
 ("ambiguity silently resolved", "    if disagreement:", "    if False:"),
 ("explicit format ignored",     "    if explicit:\n        return explicit", "    if False:\n        return explicit"),
 ("first row decides alone",     "    if len(live) == 1:\n        return live[0]\n    # several layouts fit the first row: let the rest of the file choose", "    return live[0]\n    # several layouts fit the first row: let the rest of the file choose"),
 ("later rows cannot rule out",  "        if len(survivors) < len(live):\n            live = survivors", "        if False:\n            live = survivors"),
 ("bad rows dropped silently",   "        except (IndexError, ValueError) as e:\n            raise ValueError(f\"row {i + 1} could not be read: {r} ({e})\") from e", "        except (IndexError, ValueError):\n            continue"),
 ("bar size invented",           "        if self.interval is None or self.interval.total_seconds() <= 0:", "        if False:"),
 ("nights per bar inverted",     "        return 86_400.0 / self.interval.total_seconds()", "        return self.interval.total_seconds() / 86_400.0"),
]
