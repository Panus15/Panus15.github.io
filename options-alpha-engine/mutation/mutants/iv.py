"""Ways the solver could return a number instead of admitting it cannot.

Implied vol is the quantity the whole engine competes against: every "option
rich" or "option cheap" call is this number against a forecast. So the
dangerous failure here is not inaccuracy, it is a CONFIDENT answer on a quote
that has none. The module says so itself -- `None` means no Black-Scholes vol
reproduces the price, and "callers should skip such quotes rather than trust a
garbage number."

SIX EQUIVALENT MUTANTS ARE DELIBERATELY ABSENT, and the reason is one worth
knowing about this module: the explicit refusals are REDUNDANT with the
bracket test. Remove the floor check, the zero-price check or the expired-tenor
check and the quote is still refused, because `f(lo) * f(hi) > 0` catches it a
few lines later -- verified by applying each mutation to a copy of the module
and comparing its answer against the real one over 286 quotes, where all six
agreed on every single case. They are kept as early exits that name the reason,
not as the only line of defence.

SEVEN MORE ARE ABSENT FOR A DIFFERENT REASON, AND IT TOOK THREE MEASUREMENTS
TO ESTABLISH IT. Breaking the Newton step, skewing the bisection midpoint,
disabling the flat-gradient break or skipping the Newton loop entirely DO
change the answer -- on 24 to 202 of those 286 quotes, so they are not
equivalent in the plain sense. But:

  * none is less accurate against the known vol in any way that matters (the
    largest extra error across the grid was 9.7e-5, and several were better
    than the real code on more quotes than they were worse);
  * every single answer still reprices its quote to within `tol` -- worst
    9.9e-7 against the real code's 9.901e-7.

So what they move is WHICH of the many vols inside the price tolerance comes
back. That is the bracket doing its job: Newton is the fast path and bisection
is the guarantee. `test_every_vol_the_solver_returns_reprices_the_quote_it_was
_given` holds the property that actually matters, and a Newton break severe
enough to stop convergence fails it.

One of the seven is worth singling out. "vega missing sqrt T" pointed at a
SECOND COPY of the vega formula living in this module, and a copy is a place
for two formulas to disagree. It now calls `pricing.greeks`, which a
finite-difference oracle covers, so there is one vega in this engine and it is
tested. The mutant is gone because the duplicated line is gone.

A last note on "tolerance never met": disabling that early return made the
answer BETTER on 96 quotes and worse on 2, by at most 1.5e-12. It is a speed
path, not a correctness one, and it should not be mistaken for the latter.

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
 ("floor sign flipped",          '    return max((fwd_s - fwd_k) if kind.lower() == "call" else (fwd_k - fwd_s),\n               0.0)', '    return max((fwd_k - fwd_s) if kind.lower() == "call" else (fwd_s - fwd_k),\n               0.0)'),
 ("floor undiscounted again",    "    fwd_s = S * math.exp(-q * T)\n    fwd_k = K * math.exp(-r * T)", "    fwd_s = S\n    fwd_k = K"),
 ("floor discounts the wrong leg", "    fwd_s = S * math.exp(-q * T)\n    fwd_k = K * math.exp(-r * T)", "    fwd_s = S * math.exp(-r * T)\n    fwd_k = K * math.exp(-q * T)"),
 # The identifiability refusal: the defect that returned the Newton seed.
 ("unidentified vol returned",   "            return vol if identified(vol) else None\n        # Keep the bracket tight for the fallback.", "            return vol\n        # Keep the bracket tight for the fallback."),
 ("identifiability always true", "        return abs(price(S, K, T, r, q, v + step, kind)\n                   - price(S, K, T, r, q, max(v - step, 1e-12), kind)) > tol", "        return True"),
 ("unbracketed price solved",    "    if f_lo * f_hi > 0:\n        return None", "    if False:\n        return None"),
 ("bracket test inverted",       "    if f_lo * f_hi > 0:", "    if f_lo * f_hi < 0:"),

 # The objective. Reversing it inverts every comparison that follows, which
 # moves the bracket the wrong way and makes "rich" read as "cheap".
 ("objective reversed",          "        return price(S, K, T, r, q, vol, kind) - market_price", "        return market_price - price(S, K, T, r, q, vol, kind)"),

 # Bracket maintenance. These are the ones that matter: they leave the solver
 # converging, to the wrong answer, with no sign that anything went wrong.
 ("newton bracket sides swapped", "        if diff > 0:\n            b = vol\n        else:\n            a = vol\n        # Vega for the Newton step, from the pricer rather than restated", "        if diff > 0:\n            a = vol\n        else:\n            b = vol\n        # Vega for the Newton step, from the pricer rather than restated"),
 ("bisection sides swapped",     "        if diff > 0:\n            b = vol\n        else:\n            a = vol\n    return vol", "        if diff > 0:\n            a = vol\n        else:\n            b = vol\n    return vol"),
 ("bracket never narrowed",      "        if diff > 0:\n            b = vol\n        else:\n            a = vol\n        # Vega for the Newton step, from the pricer rather than restated", "        if False:\n            b = vol\n        else:\n            a = vol\n        # Vega for the Newton step, from the pricer rather than restated"),

 # The Newton step itself, and the guards that stop it running away.
 ("newton escape unguarded",     "        if not (a < vol < b):  # Newton left the bracket -> bisect", "        if False:  # Newton left the bracket -> bisect"),

 # Convergence. A tolerance that never trips, or a loop that never runs, turns
 # the solver into a function that returns its seed.
 ("tolerance loosened",          "    tol: float = 1e-6,", "    tol: float = 1e-1,"),
 ("newton never runs",           "    vol = 0.20\n    a, b = lo, hi", "    vol = 0.20\n    a, b = lo, hi\n    max_iter = 0"),
]
