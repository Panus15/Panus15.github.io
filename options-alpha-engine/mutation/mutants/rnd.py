"""Ways the Q extraction could be biased low, which is the profitable direction.

This is the Q half of the engine's only claim, and it is the half whose errors
all point the same way. The tradeable premium is Q - P, so anything that
understates Q makes the premium look SMALLER and kills trades -- while anything
that understates P makes it look bigger. Both halves therefore need watching in
opposite directions, and this module's own docstring already names its three
traps:

    the forward         recovered from PUT-CALL PARITY, never assumed to be
                        spot. Assuming spot biases the variance by the whole
                        (F/K0 - 1)^2 correction term.
    the mean correction Var_Q = e^{rT}V - mu^2. Reporting sqrt(V/T) instead
                        drops the growth factor AND the -mu^2 term, which the
                        docstring says biases variance LOW.
    the coverage guard  truncated wings bias both extractors low "exactly in
                        stressed regimes" -- the regime the engine exists for.

ONE EQUIVALENT MUTANT IS DELIBERATELY ABSENT. Which strike the forward is
recovered from cannot change it: put-call parity holds at EVERY strike, so
`k0 = strikes[0]` and `k0 = nearest-to-spot` agree to about 1e-6, which is the
rounding in the quotes rather than a difference in the answer -- measured
across a 3-strike, an 81-strike and an 801-strike chain. The nearest-to-spot
choice is for numerical conditioning: there the call and the put are similar in
size, so the relative error in C - P is smallest. It is kept for that reason,
not because it changes a number.

COVERAGE_SIGMAS is the one constant here that was measured rather than chosen,
and its table is in the source: 2.5 sd keeps the truncation error at 0.10 vol
points against the 1-4 points of premium being harvested, where 2.0 would admit
a quarter of a one-point edge. Loosening it is therefore not a judgement call,
it is discarding the measurement.
"""

TARGET = "models/rnd.py"
TESTS = "tests/test_rnd_edge.py"

MUTANTS = [
 # The forward. Put-call parity is the whole point of recovering it.
 ("forward assumed to be spot",  "    return math.exp(chain.r * T) * (calls[k0] - puts[k0]) + k0", "    return chain.spot"),
 ("parity sign flipped",         "    return math.exp(chain.r * T) * (calls[k0] - puts[k0]) + k0", "    return math.exp(chain.r * T) * (puts[k0] - calls[k0]) + k0"),
 ("parity strike not added",     "    return math.exp(chain.r * T) * (calls[k0] - puts[k0]) + k0", "    return math.exp(chain.r * T) * (calls[k0] - puts[k0])"),
 ("parity not discounted",       "    return math.exp(chain.r * T) * (calls[k0] - puts[k0]) + k0", "    return (calls[k0] - puts[k0]) + k0"),

 # The model-free integral: the discretisation, the OTM selection, and the
 # (F/K0 - 1)^2 correction the forward exists to compute.
 ("strike spacing uniform",      "            out.append(0.5 * (strikes[i + 1] - strikes[i - 1]))", "            out.append(strikes[i] - strikes[i - 1])"),
 ("spacing not halved",          "            out.append(0.5 * (strikes[i + 1] - strikes[i - 1]))", "            out.append(strikes[i + 1] - strikes[i - 1])"),
 ("weight not inverse square",   "        total += (dk / (k * k)) * disc * o", "        total += (dk / k) * disc * o"),
 ("integral not discounted",     "        total += (dk / (k * k)) * disc * o", "        total += (dk / (k * k)) * o"),
 ("otm selection inverted",      "        if k < K0:\n            o = puts[k]\n        elif k > K0:\n            o = calls[k]", "        if k < K0:\n            o = calls[k]\n        elif k > K0:\n            o = puts[k]"),
 ("atm leg not averaged",        "            o = 0.5 * (calls[k] + puts[k])", "            o = calls[k] + puts[k]"),
 ("forward correction dropped",  "    var = (2.0 / T) * total - (1.0 / T) * (F / K0 - 1.0) ** 2", "    var = (2.0 / T) * total"),
 ("forward correction added",    "    var = (2.0 / T) * total - (1.0 / T) * (F / K0 - 1.0) ** 2", "    var = (2.0 / T) * total + (1.0 / T) * (F / K0 - 1.0) ** 2"),
 ("replication factor halved",   "    var = (2.0 / T) * total - (1.0 / T) * (F / K0 - 1.0) ** 2", "    var = (1.0 / T) * total - (1.0 / T) * (F / K0 - 1.0) ** 2"),
 ("variance not per year",       "    var = (2.0 / T) * total - (1.0 / T) * (F / K0 - 1.0) ** 2", "    var = 2.0 * total - (F / K0 - 1.0) ** 2"),
 ("K0 above the forward",        "    below = [k for k in strikes if k <= F]\n    K0 = max(below) if below else strikes[0]", "    above = [k for k in strikes if k >= F]\n    K0 = min(above) if above else strikes[0]"),

 # BKM. The contract weights are the paper; the mean correction is the trap
 # the module docstring calls out by name.
 ("bkm quad weight sign",        "        V += 2.0 * (1.0 - x) * base", "        V += 2.0 * (1.0 + x) * base"),
 ("bkm quad weight flat",        "        V += 2.0 * (1.0 - x) * base", "        V += 2.0 * base"),
 ("bkm cubic weight sign",       "        W += (6.0 * x - 3.0 * x * x) * base", "        W += (6.0 * x + 3.0 * x * x) * base"),
 ("bkm quartic weight sign",     "        X += (12.0 * x * x - 4.0 * x ** 3) * base", "        X += (12.0 * x * x + 4.0 * x ** 3) * base"),
 ("bkm mean correction dropped", "    var_q = er * V - mu * mu", "    var_q = er * V"),
 ("bkm growth factor dropped",   "    var_q = er * V - mu * mu", "    var_q = V - mu * mu"),
 ("bkm mean correction added",   "    var_q = er * V - mu * mu", "    var_q = er * V + mu * mu"),
 ("bkm mu drops its higher moments", "    mu = er - 1.0 - er / 2.0 * V - er / 6.0 * W - er / 24.0 * X", "    mu = er - 1.0 - er / 2.0 * V"),
 ("bkm mu sign flipped",         "    mu = er - 1.0 - er / 2.0 * V - er / 6.0 * W - er / 24.0 * X", "    mu = er - 1.0 + er / 2.0 * V + er / 6.0 * W + er / 24.0 * X"),
 ("bkm otm selection inverted",  "        o = calls[k] if k >= S else puts[k]", "        o = puts[k] if k >= S else calls[k]"),
 ("bkm anchored on the forward", "        x = math.log(k / S)", "        x = math.log(k / chain.spot) if False else math.log(k)"),

 # The coverage guard. Its threshold was measured, and loosening it discards
 # the measurement in exactly the regime the engine is built for.
 ("coverage requirement halved", "COVERAGE_SIGMAS = 2.5", "COVERAGE_SIGMAS = 1.0"),
 ("coverage requirement removed", "COVERAGE_SIGMAS = 2.5", "COVERAGE_SIGMAS = 0.0"),

 # Minimum strikes: three is the floor for a second difference at all.
 ("two strikes accepted",        "    if len(strikes) < 3:\n        raise ValueError(\"need >= 3 strikes with both call and put\")\n    F = _forward(chain, strikes, calls, puts, T)", "    if False:\n        raise ValueError(\"need >= 3 strikes with both call and put\")\n    F = _forward(chain, strikes, calls, puts, T)"),
 ("negative variance rooted",    "    return math.sqrt(max(var, 1e-10))", "    return math.sqrt(abs(var))"),
]
