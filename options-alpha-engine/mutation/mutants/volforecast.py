"""Ways the forecast could be wrong in the direction that manufactures an edge.

This module is the P side of the engine's only claim. Every signal is a
comparison of this forecast against the market's implied vol, so the edge is
`Q - P` and a forecast biased LOW makes every option look rich. That is the
asymmetry the table is built around: a defect that shrinks the forecast does
not produce a wrong answer in one place, it produces a sell signal everywhere.

Three families:

    the estimators      Parkinson, Garman-Klass and Rogers-Satchell each have a
                        constant that is not 1 and not guessable. Drop the
                        4*ln(2) and the range variance comes out 2.9x too big;
                        drop the (2*ln2 - 1) and Garman-Klass loses its drift
                        correction. These are the numbers nobody checks.
    the annualisation   252 is a multiplier on VARIANCE, so getting it wrong
                        scales vol by its square root -- a forecast off by
                        sqrt(252) is off by sixteen times.
    the term structure   term_vol exists because a flat forecast returned the
                        same annualised vol at 5 days and 334 days while the
                        market's own curve does not. Collapse the weight and
                        that regression is back.

`long_run_vol` used to carry a byte-identical copy of `close_to_close_vol`'s
body, which meant the sample-variance and mean-removal defects below had two
homes and no anchor could name either. It delegates now. Duplication is not a
style question when a mutation table has to point at exactly one line.

`_gap_var` deserves its own note. The overnight move is a real part of a
session's variance and the engine's GKYZ estimator includes it; a version that
drops it understates vol on every gapping market, which is every equity index.
"""

TARGET = "engine/volforecast.py"
TESTS = "tests/test_engine.py"

MUTANTS = [
 # Close-to-close and EWMA.
 ("variance divided by n",       "    var = sum((x - mean) ** 2 for x in rets) / (len(rets) - 1)", "    var = sum((x - mean) ** 2 for x in rets) / len(rets)"),
 ("mean not removed",            "    var = sum((x - mean) ** 2 for x in rets) / (len(rets) - 1)", "    var = sum(x * x for x in rets) / (len(rets) - 1)"),
 ("not annualised",              "    return math.sqrt(var * TRADING_DAYS)\n\n\ndef ewma_vol", "    return math.sqrt(var)\n\n\ndef ewma_vol"),
 ("trading days are calendar",   "TRADING_DAYS = 252", "TRADING_DAYS = 365"),
 ("ewma decay inverted",         "        var = lam * var + (1 - lam) * x * x", "        var = (1 - lam) * var + lam * x * x"),
 ("ewma has no memory",          "        var = lam * var + (1 - lam) * x * x", "        var = x * x"),
 ("log return inverted",         "    return [math.log(prices[i] / prices[i - 1]) for i in range(1, len(prices))]", "    return [math.log(prices[i - 1] / prices[i]) for i in range(1, len(prices))]"),
 ("window not applied",          "    \"\"\"Annualised close-to-close realised volatility over the last ``window`` days.\"\"\"\n    rets = log_returns(prices)[-window:]", "    \"\"\"Annualised close-to-close realised volatility over the last ``window`` days.\"\"\"\n    rets = log_returns(prices)"),

 # The range estimators. Each constant is the whole estimator.
 ("parkinson constant dropped",  "    return (math.log(h / l) ** 2) / (4.0 * math.log(2.0))", "    return math.log(h / l) ** 2"),
 ("parkinson range unsquared",   "    return (math.log(h / l) ** 2) / (4.0 * math.log(2.0))", "    return math.log(h / l) / (4.0 * math.log(2.0))"),
 ("garman klass drift term lost", "    return 0.5 * hl * hl - (2.0 * math.log(2.0) - 1.0) * co * co", "    return 0.5 * hl * hl"),
 ("garman klass sign flipped",   "    return 0.5 * hl * hl - (2.0 * math.log(2.0) - 1.0) * co * co", "    return 0.5 * hl * hl + (2.0 * math.log(2.0) - 1.0) * co * co"),
 ("garman klass half dropped",   "    return 0.5 * hl * hl - (2.0 * math.log(2.0) - 1.0) * co * co", "    return hl * hl - (2.0 * math.log(2.0) - 1.0) * co * co"),
 ("rogers satchell halved",      "    return (math.log(h / c) * math.log(h / o)\n            + math.log(l / c) * math.log(l / o))", "    return math.log(h / c) * math.log(h / o)"),
 ("rogers satchell sign flipped", "    return (math.log(h / c) * math.log(h / o)\n            + math.log(l / c) * math.log(l / o))", "    return (math.log(h / c) * math.log(h / o)\n            - math.log(l / c) * math.log(l / o))"),

 # The overnight gap, which is most of an index's variance on a gap day.
 ("overnight gap ignored",       "    return math.log(bar[0] / prev_close) ** 2", "    return 0.0"),
 ("gap unsquared",               "    return math.log(bar[0] / prev_close) ** 2", "    return math.log(bar[0] / prev_close)"),
 ("first bar invents a gap",     "    if prev_close is None or prev_close <= 0 or bar[0] <= 0:\n        return 0.0", "    if False:\n        return 0.0"),

 # Non-positive prices make every log undefined; returning 0.0 variance for
 # such a bar is the documented behaviour and it must not be skipped.
 ("parkinson takes bad prices",  "    if h <= 0 or l <= 0:\n        return 0.0", "    if False:\n        return 0.0"),
 ("garman klass takes bad prices", "    if min(o, h, l, c) <= 0:\n        return 0.0\n    hl = math.log(h / l)", "    if False:\n        return 0.0\n    hl = math.log(h / l)"),

 # The term structure.
 ("term weight collapsed",       "    weight = (1.0 - math.exp(-kT)) / kT          # -> 1 as T->0, -> 0 as T->inf", "    weight = 1.0          # -> 1 as T->0, -> 0 as T->inf"),
 ("term weight inverted",        "    weight = (1.0 - math.exp(-kT)) / kT          # -> 1 as T->0, -> 0 as T->inf", "    weight = math.exp(-kT) / kT          # -> 1 as T->0, -> 0 as T->inf"),
 ("term reversion has no speed", "    kT = max(kappa * T, 1e-9)", "    kT = max(T, 1e-9)"),
 ("term blends the wrong way",   "    var = vinf + (v0 - vinf) * weight", "    var = v0 + (vinf - v0) * weight"),
 ("term half life changed",      "def term_vol(prices: Sequence[float], T: float, *, kappa: float = 2.77,", "def term_vol(prices: Sequence[float], T: float, *, kappa: float = 0.1,"),
 ("negative variance rooted",    "    return math.sqrt(max(var, 1e-12))", "    return math.sqrt(abs(var))"),
]
