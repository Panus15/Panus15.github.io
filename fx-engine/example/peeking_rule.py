"""The same shape of rule, reading ONE bar into the future.

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
