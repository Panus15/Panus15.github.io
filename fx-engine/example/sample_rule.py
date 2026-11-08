"""A moving-average cross — the rule almost everybody writes first.

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
