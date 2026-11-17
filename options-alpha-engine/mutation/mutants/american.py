"""Ways the de-Americanization could leave the bias it exists to remove.

engine/american.py is a bias correction with a KNOWN SIGN, and the sign is the
reason the module exists. Every Q extractor here integrates European prices;
US single-name and ETF options are American, and their early-exercise premium
looks to BKM like extra option value, i.e. extra implied variance. So reading a
raw American chain biases the recovered variance HIGH -- the market looks
richer than it is, which invites selling. A correction that silently does
nothing restores exactly that bias, and the output is still a chain of
plausible prices either way.

Three things carry the correction, and each gets mutants that undo it:

  1. THE TREE. CRR is u = e^{sigma*sqrt(dt)}, d = 1/u, with the risk-neutral
     p set so the tree's forward matches e^{(r-q)dt}. Get p wrong and every
     price is wrong by a drift error that no test of a single price catches,
     because the recovered IV absorbs it.
  2. THE EARLY-EXERCISE TEST at every node, which is the only line that makes
     the tree American at all. Replacing max(continuation, intrinsic) with the
     continuation value is a European tree wearing the American name, and the
     premium it reports is then exactly zero everywhere.
  3. RE-PRICING EUROPEAN AT THE AMERICAN VOL. This is the whole de-
     Americanization: solve the vol the market is paying under the American
     model, then ask what a European option costs at that vol. Repricing
     American instead returns the input unchanged -- a no-op correction that
     reports success.

THE SAME-TREE SUBTRACTION in `early_exercise_premium` is a numerical claim the
docstring makes explicitly: both legs run through the same binomial so the
discretization error cancels EXACTLY, where subtracting a closed-form
Black-Scholes price from a coarse tree would leak discretization noise and can
go negative. That is testable as an exact equality, not an inequality.

THE MID IS DE-AMERICANIZED, NOT THE BID AND ASK SEPARATELY, and the docstring
gives the reason: a routine zero-bid OTM wing has an unsolvable bid but a
perfectly solvable mid, so per-side handling drops the strike -- and rnd's
call/put intersection then drops the paired side too, truncating the wings and
biasing the recovered variance LOW. Having corrected a HIGH bias into a LOW
one, the module would still report a number.

FOUR EQUIVALENT MUTANTS ARE DELIBERATELY ABSENT. Two are provable and two were
measured.

The exercise boundary's `>=` versus `>` cannot change a value. The expressions
`cont if cont >= intrinsic else intrinsic` and `cont if cont > intrinsic else
intrinsic` differ only where cont == intrinsic, and there the first returns cont
and the second returns intrinsic -- the same number. This is max() either way.

The bisection's final `return 0.5 * (a + b)` is unreachable. The bracket [0.01,
5.0] halves 60 times to a width of 4.3e-18, and the binomial price is continuous
in vol (u = e^{sigma*sqrt(dt)} is continuous, the payoffs are, and max() of
continuous functions is), so |f(m)| falls under the 1e-5 tolerance and returns
from inside the loop long before the iterations run out. Measured, returning a
bracket endpoint instead differs on 0 of 50 round trips.

THE INTRINSIC GUARD IS SUBSUMED BY THE BRACKET CHECK, which is why both mutants
of it survive. An American option is worth at least its intrinsic value at EVERY
vol, so for a quote below intrinsic the price at the bracket's floor already
exceeds it: f(lo) > 0 and f(hi) > 0, the signs agree, and `fa * fb > 0` returns
None first. Measured over below-intrinsic calls and puts and a quote one cent
under intrinsic, every one returns None with the guard removed and with its
intrinsic set to zero. The guard is a cheap fast path and a statement of what a
broken quote is; it is not what refuses one.
"""

TARGET = "engine/american.py"
TESTS = "tests/test_american.py"

MUTANTS = [
 # 1. The CRR tree: the up move, the risk-neutral probability, the discount.
 ("up move not root of time",    "    u = math.exp(vol * math.sqrt(dt))", "    u = math.exp(vol * dt)"),
 ("up move ignores vol",         "    u = math.exp(vol * math.sqrt(dt))", "    u = math.exp(math.sqrt(dt))"),
 ("down move not reciprocal",    "    p = (math.exp((r - q) * dt) - 1.0 / u) / (u - 1.0 / u)", "    p = (math.exp((r - q) * dt) - u) / (u - 1.0 / u)"),
 ("probability denominator sum", "    p = (math.exp((r - q) * dt) - 1.0 / u) / (u - 1.0 / u)", "    p = (math.exp((r - q) * dt) - 1.0 / u) / (u + 1.0 / u)"),
 ("carry dropped from drift",    "    p = (math.exp((r - q) * dt) - 1.0 / u) / (u - 1.0 / u)", "    p = (math.exp(r * dt) - 1.0 / u) / (u - 1.0 / u)"),
 ("carry sign flipped",          "    p = (math.exp((r - q) * dt) - 1.0 / u) / (u - 1.0 / u)", "    p = (math.exp((r + q) * dt) - 1.0 / u) / (u - 1.0 / u)"),
 ("drift dropped entirely",      "    p = (math.exp((r - q) * dt) - 1.0 / u) / (u - 1.0 / u)", "    p = (1.0 - 1.0 / u) / (u - 1.0 / u)"),
 ("probability complemented",    "    p = (math.exp((r - q) * dt) - 1.0 / u) / (u - 1.0 / u)", "    p = 1.0 - (math.exp((r - q) * dt) - 1.0 / u) / (u - 1.0 / u)"),
 ("discount not applied",        "    disc = math.exp(-r * dt)", "    disc = 1.0"),
 ("discount compounded",         "    disc = math.exp(-r * dt)", "    disc = math.exp(r * dt)"),
 ("branches not discounted",     "    pu, pd = disc * p, disc * (1.0 - p)", "    pu, pd = p, 1.0 - p"),
 ("branches swapped",            "    pu, pd = disc * p, disc * (1.0 - p)", "    pu, pd = disc * (1.0 - p), disc * p"),
 ("probability clamp removed",   "    p = min(1.0, max(0.0, p))                    # clamp degenerate low-vol nodes", "    pass"),

 # The lattice geometry: node (i, j) sits at S * u^(2j - i).
 ("node index not doubled",      "        return S * pw[(2 * j - i) + steps]", "        return S * pw[(j - i) + steps]"),
 ("node index sign flipped",     "        return S * pw[(2 * j - i) + steps]", "        return S * pw[(2 * j + i) + steps]"),
 ("power table one sided",       "    pw = [u ** k for k in range(-steps, steps + 1)]", "    pw = [u ** k for k in range(0, 2 * steps + 1)]"),

 # Terminal payoffs.
 ("terminal payoffs swapped",    "    if kind == \"call\":\n        val = [max(spot(steps, j) - K, 0.0) for j in range(steps + 1)]\n    else:\n        val = [max(K - spot(steps, j), 0.0) for j in range(steps + 1)]", "    if kind == \"call\":\n        val = [max(K - spot(steps, j), 0.0) for j in range(steps + 1)]\n    else:\n        val = [max(spot(steps, j) - K, 0.0) for j in range(steps + 1)]"),
 ("terminal call not floored",   "        val = [max(spot(steps, j) - K, 0.0) for j in range(steps + 1)]", "        val = [spot(steps, j) - K for j in range(steps + 1)]"),
 ("terminal put not floored",    "        val = [max(K - spot(steps, j), 0.0) for j in range(steps + 1)]", "        val = [K - spot(steps, j) for j in range(steps + 1)]"),

 # 2. The early-exercise test: the only line that makes the tree American.
 ("early exercise never taken",  "                val[j] = cont if cont >= intrinsic else intrinsic", "                val[j] = cont"),
 ("early exercise always taken", "                val[j] = cont if cont >= intrinsic else intrinsic", "                val[j] = intrinsic"),
 ("early exercise test inverted", "                val[j] = cont if cont >= intrinsic else intrinsic", "                val[j] = cont if cont <= intrinsic else intrinsic"),
 ("node intrinsic sign flipped", "                intrinsic = (s - K) if kind == \"call\" else (K - s)", "                intrinsic = (K - s) if kind == \"call\" else (s - K)"),
 ("american flag ignored",       "            if american:", "            if False:"),

 # Expiry and the European leg of the premium.
 ("expired value not floored",   "        return max((S - K) if kind == \"call\" else (K - S), 0.0)", "        return (S - K) if kind == \"call\" else (K - S)"),
 ("premium sign flipped",        "    return am - eu", "    return eu - am"),
 ("premium european leg american", "    eu = _binomial(S, K, T, r, q, vol, kind, steps, american=False)", "    eu = _binomial(S, K, T, r, q, vol, kind, steps, american=True)"),
 ("premium european leg closed form", "    eu = _binomial(S, K, T, r, q, vol, kind, steps, american=False)", "    eu = pricing.price(S, K, T, r, q, vol, kind)"),

 # 3. De-Americanization proper: European at the AMERICAN vol.
 ("de americanization is a no op", "    return pricing.price(S, K, T, r, q, iv, kind), iv", "    return american_price(S, K, T, r, q, iv, kind, steps=steps), iv"),
 ("de americanized at a fixed vol", "    return pricing.price(S, K, T, r, q, iv, kind), iv", "    return pricing.price(S, K, T, r, q, 0.20, kind), iv"),

 # The American IV bisection.
 ("bracket check removed",       "    if fa * fb > 0:\n        return None", "    if False:\n        return None"),
 ("bisection side swapped",      "        if (fm > 0) == (fa > 0):\n            a, fa = m, fm\n        else:\n            b, fb = m, fm", "        if (fm > 0) == (fa > 0):\n            b, fb = m, fm\n        else:\n            a, fa = m, fm"),
 ("bisection tolerance inverted", "        if abs(fm) < tol:\n            return m", "        if abs(fm) > tol:\n            return m"),
 ("bisection midpoint skewed",   "        m = 0.5 * (a + b)\n        fm = f(m)", "        m = 0.25 * (a + b)\n        fm = f(m)"),
 ("vol bracket floor raised",    "                q: float, kind: str, *, steps: int = 128, lo: float = 0.01,", "                q: float, kind: str, *, steps: int = 128, lo: float = 0.26,"),

 # implied_carry: the market's own forward, from put-call parity.
 ("carry parity sign flipped",   "    fwd = kstar + math.exp(r * T) * (calls[kstar] - puts[kstar])", "    fwd = kstar + math.exp(r * T) * (puts[kstar] - calls[kstar])"),
 ("carry parity strike dropped", "    fwd = kstar + math.exp(r * T) * (calls[kstar] - puts[kstar])", "    fwd = math.exp(r * T) * (calls[kstar] - puts[kstar])"),
 ("carry parity not discounted", "    fwd = kstar + math.exp(r * T) * (calls[kstar] - puts[kstar])", "    fwd = kstar + (calls[kstar] - puts[kstar])"),
 ("carry solved with wrong sign", "    return r, r - math.log(fwd / chain.spot) / T", "    return r, r + math.log(fwd / chain.spot) / T"),
 ("carry ratio inverted",        "    return r, r - math.log(fwd / chain.spot) / T", "    return r, r - math.log(chain.spot / fwd) / T"),
 ("carry not annualised",        "    return r, r - math.log(fwd / chain.spot) / T", "    return r, r - math.log(fwd / chain.spot)"),
 ("carry anchored furthest out", "    kstar = min(common, key=lambda k: abs(k - chain.spot))", "    kstar = max(common, key=lambda k: abs(k - chain.spot))"),
 ("negative forward accepted",   "    if fwd <= 0:\n        return None", "    if False:\n        return None"),

 # de_americanize_chain: the spread reflow and the carry it prices against.
 ("implied forward off by default", "def de_americanize_chain(chain: OptionChain, *, steps: int = 128,\n                         use_implied_forward: bool = True) -> OptionChain:", "def de_americanize_chain(chain: OptionChain, *, steps: int = 128,\n                         use_implied_forward: bool = False) -> OptionChain:"),
 ("carry fallback ignores metadata", "        carry[dte] = c if c is not None else (chain.r, chain.q)", "        carry[dte] = c if c is not None else (chain.r, 0.0)"),
 ("half spread is the full spread", "        half = max(0.0, 0.5 * (qt.ask - qt.bid))", "        half = max(0.0, qt.ask - qt.bid)"),
 ("half spread dropped",         "        half = max(0.0, 0.5 * (qt.ask - qt.bid))", "        half = 0.0"),
 ("spread reflowed one sided",   "                               round(max(eur_mid - half, 0.0), 4),\n                               round(eur_mid + half, 4)))", "                               round(max(eur_mid - half, 0.0), 4),\n                               round(eur_mid, 4)))"),
 ("negative bid not floored",    "                               round(max(eur_mid - half, 0.0), 4),", "                               round(eur_mid - half, 4),"),
 ("unsolvable quotes kept raw",  "        if de is None:\n            continue", "        if de is None:\n            out.append(qt)\n            continue"),
]
