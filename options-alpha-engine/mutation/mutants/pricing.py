"""Ways the pricer could be wrong in a way the rest of the engine believes.

This module is the engine's own description of its foundation: "the
deterministic, verifiable core — everything else plugs into these functions."
Every mispricing signal is a comparison against a number from here, so a defect
here does not produce a wrong signal in one place. It moves every signal at
once, in the same direction, and nothing downstream can see it.

It is also the one module with real oracles. Put-call parity, delta bounds and
finite-difference Greeks are properties the right answer must satisfy and a
wrong one generally cannot, which is why this table leans on sign flips and
dropped terms rather than on magnitudes: those are the errors that survive a
smoke test and fail an identity.

ONE EQUIVALENT MUTANT IS DELIBERATELY ABSENT. Neutering the `else: raise` in
`greeks` changes nothing observable: execution falls through to
`Greeks(price=price(...))`, and `price` raises the identical ValueError with the
identical message. The two checks are redundant with each other, and the one in
`greeks` earns its place only by failing BEFORE the Greeks are computed rather
than after. No test can distinguish them.

The Greeks are the softer target. The engine sizes short vega against a book
cap, so a vega that is too small by `sqrt(T)` understates the whole book's
exposure — and `engine/portfolio.py` would still report it as inside the cap.
"""

TARGET = "engine/pricing.py"
TESTS = "tests/test_engine.py"

MUTANTS = [
 # d1 / d2. The drift term and the variance adjustment are where a sign error
 # is least visible: the price stays positive and plausible either way.
 ("drift sign flipped",         "    d1 = (math.log(S / K) + (r - q + 0.5 * vol * vol) * T) / v", "    d1 = (math.log(S / K) - (r - q + 0.5 * vol * vol) * T) / v"),
 ("carry net of the wrong leg", "    d1 = (math.log(S / K) + (r - q + 0.5 * vol * vol) * T) / v", "    d1 = (math.log(S / K) + (q - r + 0.5 * vol * vol) * T) / v"),
 ("ito term dropped",           "    d1 = (math.log(S / K) + (r - q + 0.5 * vol * vol) * T) / v", "    d1 = (math.log(S / K) + (r - q) * T) / v"),
 ("ito term undoubled",         "    d1 = (math.log(S / K) + (r - q + 0.5 * vol * vol) * T) / v", "    d1 = (math.log(S / K) + (r - q + vol * vol) * T) / v"),
 ("d2 above d1",                "    d2 = d1 - v", "    d2 = d1 + v"),
 ("d2 equals d1",               "    d2 = d1 - v", "    d2 = d1"),
 ("time not square rooted",     "    v = vol * math.sqrt(T)", "    v = vol * T"),
 ("moneyness inverted",         "    d1 = (math.log(S / K) + (r - q + 0.5 * vol * vol) * T) / v", "    d1 = (math.log(K / S) + (r - q + 0.5 * vol * vol) * T) / v"),

 # The two formulas. Swapping them, or discounting the wrong leg, breaks
 # put-call parity -- which is exactly what the oracle is there for.
 ("call and put swapped",       '    if kind == "call":\n        return S * df_q * _norm_cdf(d1) - K * df_r * _norm_cdf(d2)', '    if kind == "call":\n        return K * df_r * _norm_cdf(-d2) - S * df_q * _norm_cdf(-d1)'),
 ("call legs reversed",         "        return S * df_q * _norm_cdf(d1) - K * df_r * _norm_cdf(d2)", "        return K * df_r * _norm_cdf(d2) - S * df_q * _norm_cdf(d1)"),
 ("put legs reversed",          "        return K * df_r * _norm_cdf(-d2) - S * df_q * _norm_cdf(-d1)", "        return S * df_q * _norm_cdf(-d1) - K * df_r * _norm_cdf(-d2)"),
 ("strike not discounted",      "    df_r = math.exp(-r * T)\n    df_q = math.exp(-q * T)\n    if kind == \"call\":", "    df_r = 1.0\n    df_q = math.exp(-q * T)\n    if kind == \"call\":"),
 ("dividend yield ignored",     "    df_r = math.exp(-r * T)\n    df_q = math.exp(-q * T)\n    if kind == \"call\":", "    df_r = math.exp(-r * T)\n    df_q = 1.0\n    if kind == \"call\":"),
 ("discounts swapped",          "    df_r = math.exp(-r * T)\n    df_q = math.exp(-q * T)\n    if kind == \"call\":", "    df_r = math.exp(-q * T)\n    df_q = math.exp(-r * T)\n    if kind == \"call\":"),
 ("rate sign flipped",          "    df_r = math.exp(-r * T)\n    df_q = math.exp(-q * T)\n    if kind == \"call\":", "    df_r = math.exp(r * T)\n    df_q = math.exp(-q * T)\n    if kind == \"call\":"),

 # Expiry. An expired option is worth its intrinsic value and never less
 # than nothing -- dropping the floor prices a worthless option negative.
 ("expired intrinsic unsigned", "        intrinsic = (S - K) if kind == \"call\" else (K - S)", "        intrinsic = (K - S) if kind == \"call\" else (S - K)"),
 ("expired value can be negative", "        return max(intrinsic, 0.0)", "        return intrinsic"),
 ("expiry not special cased",   "    if T <= 0:  # expired -> intrinsic value", "    if False:  # expired -> intrinsic value"),

 # The normal distribution itself.
 ("cdf not scaled",             "    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))", "    return 0.5 * (1.0 + math.erf(x))"),
 ("pdf missing its constant",   "    return math.exp(-0.5 * x * x) / SQRT_2PI", "    return math.exp(-0.5 * x * x)"),
 ("pdf variance halved",        "    return math.exp(-0.5 * x * x) / SQRT_2PI", "    return math.exp(-x * x) / SQRT_2PI"),

 # Greeks. Finite differences are the oracle; these are the errors that pass
 # a glance. A vega short by sqrt(T) understates the whole book's exposure
 # against a cap that portfolio.py would still report as satisfied.
 ("put delta positive",         "        delta = -df_q * _norm_cdf(-d1)", "        delta = df_q * _norm_cdf(-d1)"),
 ("call delta uses d2",         "        delta = df_q * _norm_cdf(d1)", "        delta = df_q * _norm_cdf(d2)"),
 ("gamma not per unit spot",    "    gamma = df_q * nd1 / (S * vol * sqrtT)", "    gamma = df_q * nd1 / (vol * sqrtT)"),
 ("gamma missing its vol",      "    gamma = df_q * nd1 / (S * vol * sqrtT)", "    gamma = df_q * nd1 / (S * sqrtT)"),
 ("vega missing sqrt T",        "    vega = S * df_q * nd1 * sqrtT", "    vega = S * df_q * nd1"),
 ("vega per vol point",         "    vega = S * df_q * nd1 * sqrtT", "    vega = S * df_q * nd1 * sqrtT / 100.0"),
 ("call theta positive",        "        theta = (-(S * df_q * nd1 * vol) / (2 * sqrtT)\n                 - r * K * df_r * _norm_cdf(d2)", "        theta = ((S * df_q * nd1 * vol) / (2 * sqrtT)\n                 - r * K * df_r * _norm_cdf(d2)"),
 ("theta decay undoubled",      "        theta = (-(S * df_q * nd1 * vol) / (2 * sqrtT)\n                 - r * K * df_r * _norm_cdf(d2)\n                 + q * S * df_q * _norm_cdf(d1))", "        theta = (-(S * df_q * nd1 * vol) / sqrtT\n                 - r * K * df_r * _norm_cdf(d2)\n                 + q * S * df_q * _norm_cdf(d1))"),
 ("put rho positive",           "        rho = -K * T * df_r * _norm_cdf(-d2)", "        rho = K * T * df_r * _norm_cdf(-d2)"),
 ("rho missing its tenor",      "        rho = K * T * df_r * _norm_cdf(d2)", "        rho = K * df_r * _norm_cdf(d2)"),

 # Guards.
 ("bad kind priced",            "    raise ValueError(f\"kind must be 'call' or 'put', got {kind!r}\")\n\n\n@dataclass", "    return 0.0\n\n\n@dataclass"),
 ("non positive inputs allowed", "    if T <= 0 or vol <= 0 or S <= 0 or K <= 0:", "    if False:"),
 ("zero vol allowed",           "    if T <= 0 or vol <= 0 or S <= 0 or K <= 0:", "    if T <= 0 or S <= 0 or K <= 0:"),
 ("kind case sensitive",        "def price(S: float, K: float, T: float, r: float, q: float, vol: float, kind: str) -> float:\n    \"\"\"Black-Scholes-Merton fair value of a European option.\"\"\"\n    kind = kind.lower()", "def price(S: float, K: float, T: float, r: float, q: float, vol: float, kind: str) -> float:\n    \"\"\"Black-Scholes-Merton fair value of a European option.\"\"\"\n    kind = kind"),
]
