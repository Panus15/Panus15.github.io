"""Generate a runnable example, so the engine can be tried before you have data.

`SETUP.md` section 4 asks you to export a CSV from your broker. That is the
right first step for a real answer and the wrong one for finding out what the
tool does — nobody should have to go and get data before they can see the shape
of the output. This writes a file you can run the whole walkthrough against in
the next minute.

    python3 tools/make_example.py
    # then follow example/README.md

WHAT IT WRITES, AND WHY THE NAME SHOUTS.

    example/EURUSD_H1_SYNTHETIC.csv    a seeded random walk, NOT market data
    example/sample_rule.py             a moving-average cross
    example/peeking_rule.py           the same rule with a one-bar look-ahead

The series is a random walk from a fixed seed. It is NOT EURUSD, it is not any
pair, and nothing the engine concludes about it is a finding about the market.
That is why SYNTHETIC is in the filename and in a comment on every row block:
a file called `EURUSD_H1.csv` sitting in a repository is exactly how a
synthetic fixture ends up quoted as a result six months later, and this engine
exists to stop that class of mistake rather than commit it.

The random walk is also the honest choice for a demonstration. Real sample data
would invite reading its verdicts as conclusions; a random walk cannot be
misread that way, because there is provably nothing in it. Watching the engine
refuse it is the demonstration.

WHY THE PEEKING RULE IS SHIPPED TOO. It is the one output in this repository
that cannot be argued with. Run it and the engine stops before printing a
number, naming the bar that gave it away; comment out the check and the same
rule reports thousands of trades at a healthy profit on data with nothing in it.
Seeing both is worth more than reading either.
"""

from __future__ import annotations

import io
import math
import os
import random
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HERE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "example")

#: Fixed, so the numbers in example/README.md stay true. Change it and the
#: README's figures stop matching, which is why it is a constant and not a flag.
SEED = 3

#: Hours of bars. About 18 months of a 24/5 week once weekends are dropped,
#: which is enough for the structural patterns to fire and still short enough
#: that most verdicts come back TOO_FEW_TRADES — which is the honest outcome
#: for a file this size and worth seeing.
HOURS = 12_000

#: Per-substep volatility. Four substeps per bar, giving hourly-sized bars at
#: roughly 8 pips of range.
SIGMA = 0.0004

BANNER = ("# SYNTHETIC — a seeded random walk, NOT market data. Nothing the "
          "engine concludes\n# about this file is a finding about EUR/USD or "
          "any other pair. Regenerate with\n# python3 tools/make_example.py\n")

SAMPLE_RULE = '''"""A moving-average cross — the rule almost everybody writes first.

Honest: it only ever reads bars at or before the one it fires on. Run it with
`--rule` and the loader's look-ahead check passes it silently.
"""

from models.patterns import Detection

#: Three knobs: the two windows and the stop distance. This is the denominator
#: of the multiple-testing correction, so it counts what COULD have been swept,
#: not what was left at its default.
PARAMETER_COUNT = 3

FAST, SLOW, STOP_PIPS = 10, 40, 30


def _sma(bars, i, n):
    return sum(b.c for b in bars[i - n + 1:i + 1]) / n


def detect(bars):
    out = []
    for i in range(SLOW, len(bars)):
        f_now, s_now = _sma(bars, i, FAST), _sma(bars, i, SLOW)
        f_prev, s_prev = _sma(bars, i - 1, FAST), _sma(bars, i - 1, SLOW)
        if f_prev <= s_prev and f_now > s_now:
            e = bars[i].c
            out.append(Detection("ma_cross", i, 1, e,
                                 e - STOP_PIPS * 0.0001,
                                 e + 2 * STOP_PIPS * 0.0001))
        elif f_prev >= s_prev and f_now < s_now:
            e = bars[i].c
            out.append(Detection("ma_cross", i, -1, e,
                                 e + STOP_PIPS * 0.0001,
                                 e - 2 * STOP_PIPS * 0.0001))
    return out
'''

PEEKING_RULE = '''"""The same shape of rule, reading ONE bar into the future.

This is what a look-ahead bug looks like in practice: four lines, no warning
signs, and an equity curve that slopes beautifully upward. The engine refuses
to run it and names the bar that gave it away.

To see what it would have reported, pass `check=False` to `load_rule` — on the
synthetic file it comes back TRADEABLE with thousands of trades at a healthy
profit, on data with provably nothing in it.
"""

from models.patterns import Detection

PARAMETER_COUNT = 1


def detect(bars):
    out = []
    for i in range(1, len(bars) - 1):
        # bars[i + 1] has not printed when bar i closes. That is the whole bug.
        if bars[i + 1].c > bars[i].c:
            e = bars[i].c
            out.append(Detection("peek", i, 1, e, e - 0.0030, e + 0.0060))
    return out
'''


def write_csv(path: str) -> int:
    """A seeded random walk, skipping the hours the FX market is shut.

    The weekend gap is left in deliberately: the loader separates holes that
    span a Saturday from holes that do not, and a file with no weekends would
    never exercise that.
    """
    rng = random.Random(SEED)
    s = SIGMA
    px, t, rows = 1.1000, datetime(2023, 1, 2), []
    for _ in range(HOURS):
        # FX runs Sunday evening to Friday evening; all Saturday and most of
        # Sunday are genuinely absent, not missing data
        if t.weekday() == 5 or (t.weekday() == 6 and t.hour < 22):
            t += timedelta(hours=1)
            continue
        o, path_px = px, [px]
        for _ in range(4):
            px *= math.exp(rng.gauss(0.0, s))
            path_px.append(px)
        rows.append((t, o, max(path_px), min(path_px), px))
        t += timedelta(hours=1)
    with io.open(path, "w", encoding="utf-8") as fh:
        fh.write(BANNER)
        fh.write("Date,Open,High,Low,Close\n")
        for r in rows:
            fh.write(f"{r[0]:%Y-%m-%d %H:%M:%S},"
                     + ",".join(f"{v:.5f}" for v in r[1:]) + "\n")
    return len(rows)


def main(argv=None) -> int:
    os.makedirs(HERE, exist_ok=True)
    csv_path = os.path.join(HERE, "EURUSD_H1_SYNTHETIC.csv")
    n = write_csv(csv_path)
    for name, body in (("sample_rule.py", SAMPLE_RULE),
                       ("peeking_rule.py", PEEKING_RULE)):
        io.open(os.path.join(HERE, name), "w", encoding="utf-8").write(body)
    print(f"wrote {n:,} bars to {csv_path}")
    print(f"      sample_rule.py and peeking_rule.py to {HERE}")
    print()
    print("The CSV is a seeded RANDOM WALK, not market data. Nothing the engine")
    print("concludes about it is a finding about EUR/USD. Next: example/README.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
