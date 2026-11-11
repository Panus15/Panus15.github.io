"""Ways the solver could return a number instead of admitting it cannot.

Implied vol is the quantity the whole engine competes against: every "option
rich" or "option cheap" call is this number against a forecast. So the
dangerous failure here is not inaccuracy, it is a CONFIDENT answer on a quote
that has none. The module says so itself -- `None` means no Black-Scholes vol
reproduces the price, and "callers should skip such quotes rather than trust a
garbage number."

Three refusals carry that, and each is one line:

    a price below intrinsic value      no vol can reach it
    a price outside the [lo, hi] vol   the bracket does not contain a root
    a non-positive price or tenor      nothing to invert

Every mutant below either removes a refusal, so garbage comes back as a vol, or
breaks the bracket maintenance so the bisection fallback converges to the wrong
side. The fallback matters more than the Newton step: Newton is the fast path,
bisection is the one that has to be right when a quote is ugly, which is
exactly when a real chain disagrees with a model.
"""

TARGET = "engine/iv.py"
TESTS = "tests/test_engine.py"

MUTANTS = [
 # The three refusals.
 ("below the floor accepted",    "    if market_price < _lower_bound(S, K, T, r, q, kind) - 1e-8:\n        return None", "    if False:\n        return None"),
 ("floor tolerance inverted",    "    if market_price < _lower_bound(S, K, T, r, q, kind) - 1e-8:", "    if market_price < _lower_bound(S, K, T, r, q, kind) + 1e-8:"),
 ("floor sign flipped",          '    return max((fwd_s - fwd_k) if kind.lower() == "call" else (fwd_k - fwd_s),\n               0.0)', '    return max((fwd_k - fwd_s) if kind.lower() == "call" else (fwd_s - fwd_k),\n               0.0)'),
 ("floor can be negative",       '    return max((fwd_s - fwd_k) if kind.lower() == "call" else (fwd_k - fwd_s),\n               0.0)', '    return (fwd_s - fwd_k) if kind.lower() == "call" else (fwd_k - fwd_s)'),
 ("floor undiscounted again",    "    fwd_s = S * math.exp(-q * T)\n    fwd_k = K * math.exp(-r * T)", "    fwd_s = S\n    fwd_k = K"),
 ("floor discounts the wrong leg", "    fwd_s = S * math.exp(-q * T)\n    fwd_k = K * math.exp(-r * T)", "    fwd_s = S * math.exp(-r * T)\n    fwd_k = K * math.exp(-q * T)"),
 # The identifiability refusal: the defect that returned the Newton seed.
 ("unidentified vol returned",   "            return vol if identified(vol) else None\n        # Keep the bracket tight for the fallback.", "            return vol\n        # Keep the bracket tight for the fallback."),
 ("identifiability always true", "        return abs(price(S, K, T, r, q, v + step, kind)\n                   - price(S, K, T, r, q, max(v - step, 1e-12), kind)) > tol", "        return True"),
 ("unbracketed price solved",    "    if f_lo * f_hi > 0:\n        return None", "    if False:\n        return None"),
 ("bracket test inverted",       "    if f_lo * f_hi > 0:", "    if f_lo * f_hi < 0:"),
 ("zero price inverted",         "    if T <= 0 or market_price <= 0:\n        return None", "    if False:\n        return None"),
 ("expired quote inverted",      "    if T <= 0 or market_price <= 0:", "    if market_price <= 0:"),
 ("negative price accepted",     "    if T <= 0 or market_price <= 0:", "    if T <= 0:"),

 # The objective. Reversing it inverts every comparison that follows, which
 # moves the bracket the wrong way and makes "rich" read as "cheap".
 ("objective reversed",          "        return price(S, K, T, r, q, vol, kind) - market_price", "        return market_price - price(S, K, T, r, q, vol, kind)"),

 # Bracket maintenance. These are the ones that matter: they leave the solver
 # converging, to the wrong answer, with no sign that anything went wrong.
 ("newton bracket sides swapped", "        if diff > 0:\n            b = vol\n        else:\n            a = vol\n        # Vega (unnormalised) for the Newton step.", "        if diff > 0:\n            a = vol\n        else:\n            b = vol\n        # Vega (unnormalised) for the Newton step."),
 ("bisection sides swapped",     "        if diff > 0:\n            b = vol\n        else:\n            a = vol\n    return vol", "        if diff > 0:\n            a = vol\n        else:\n            b = vol\n    return vol"),
 ("bracket never narrowed",      "        if diff > 0:\n            b = vol\n        else:\n            a = vol\n        # Vega (unnormalised) for the Newton step.", "        if False:\n            b = vol\n        else:\n            a = vol\n        # Vega (unnormalised) for the Newton step."),
 ("bisection midpoint skewed",   "        vol = 0.5 * (a + b)\n        diff = f(vol)", "        vol = 0.9 * a + 0.1 * b\n        diff = f(vol)"),

 # The Newton step itself, and the guards that stop it running away.
 ("newton step added",           "        vol -= step", "        vol += step"),
 ("newton step doubled",         "        step = diff / vega", "        step = 2.0 * diff / vega"),
 ("newton escape unguarded",     "        if not (a < vol < b):  # Newton left the bracket -> bisect", "        if False:  # Newton left the bracket -> bisect"),
 ("flat gradient divided by",    "        if vega < 1e-8:\n            break  # flat gradient -> hand over to bisection", "        if False:\n            break  # flat gradient -> hand over to bisection"),
 ("vega missing sqrt T",         "        vega = S * math.exp(-q * T) * (math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi)) * math.sqrt(T)", "        vega = S * math.exp(-q * T) * (math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi))"),

 # Convergence. A tolerance that never trips, or a loop that never runs, turns
 # the solver into a function that returns its seed.
 ("tolerance never met",         "        if abs(diff) < tol:\n            return vol\n        # Keep the bracket tight for the fallback.", "        if False:\n            return vol\n        # Keep the bracket tight for the fallback."),
 ("tolerance loosened",          "    tol: float = 1e-6,", "    tol: float = 1e-1,"),
 ("newton never runs",           "    vol = 0.20\n    a, b = lo, hi", "    vol = 0.20\n    a, b = lo, hi\n    max_iter = 0"),
 ("seed returned unchanged",     "    for _ in range(max_iter):\n        diff = f(vol)", "    for _ in range(0):\n        diff = f(vol)"),
]
