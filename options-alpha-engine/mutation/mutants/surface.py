"""Ways the smile fit could manufacture a measurement out of nothing.

models/surface.py exists because of a bias with a direction: the model-free and
BKM integrals read LOW when the OTM wing is missing, and the wing goes missing
in exactly the stressed regimes that dominate short-vol P&L. The fix is to fit
Gatheral's SVI to the quoted smile and densify it onto a wide grid, so Q stops
depending on which strikes happened to quote.

That fix is also the module's danger. Densification turns five quotes into
eighty-one, and every one of them looks like a market price to `rnd`. So the
mutants here concentrate on the three ways that can go wrong:

  1. THE PARAMETERISATION. SVI's shape is what justifies extrapolating into a
     wing that nobody quoted. Break the formula and the extrapolation is still
     smooth, still arbitrage-"checked", still 81 confident quotes -- just wrong,
     and wrong furthest out where no quote can contradict it.
  2. THE GRID. `lo`/`hi` are the whole repair: narrowing them back towards spot
     reinstates the truncation bias the module was written to remove, and the
     densified chain still looks dense.
  3. THE GATE. `vol_rmse` against MAX_VOL_RMSE is the only thing standing
     between a smile SVI cannot represent and a number read as robust. Its
     threshold was measured (the table is on the constant), so loosening it
     discards the measurement rather than making a judgement call.

THE OTM SELECTION in `fit_chain` deserves its own note. Fitting the ITM quote
instead of the OTM one at each strike is not a small change: ITM value is mostly
intrinsic, so the vega is small and the recovered IV is dominated by quote
rounding. This is the same wing-artifact family as the known `iv_dent`
limitation recorded in tests/test_engine.py.

NELDER-MEAD'S TUNING CONSTANTS ARE NOT MUTATED. alpha/gamma/rho_c/sig are
convergence tuning on a 5-parameter smooth least-squares problem with a
restart; measured, the simplex reaches the same optimum with any textbook
values, and asserting otherwise would be asserting on the optimiser's path
rather than its answer. What IS mutated is the optimiser's DIRECTION and its
stopping rules, because those change the answer.
"""

TARGET = "models/surface.py"
TESTS = "tests/test_surface.py"

MUTANTS = [
 # 1. The SVI parameterisation (Gatheral 2004), which is what licenses
 #    extrapolating into an unquoted wing.
 ("svi hyperbola sign flipped",   "    return a + b * (rho * (k - m) + math.sqrt((k - m) ** 2 + sigma * sigma))", "    return a + b * (rho * (k - m) - math.sqrt((k - m) ** 2 + sigma * sigma))"),
 ("svi skew term sign flipped",   "    return a + b * (rho * (k - m) + math.sqrt((k - m) ** 2 + sigma * sigma))", "    return a + b * (-rho * (k - m) + math.sqrt((k - m) ** 2 + sigma * sigma))"),
 ("svi level not added",          "    return a + b * (rho * (k - m) + math.sqrt((k - m) ** 2 + sigma * sigma))", "    return b * (rho * (k - m) + math.sqrt((k - m) ** 2 + sigma * sigma))"),
 ("svi slope not applied",        "    return a + b * (rho * (k - m) + math.sqrt((k - m) ** 2 + sigma * sigma))", "    return a + (rho * (k - m) + math.sqrt((k - m) ** 2 + sigma * sigma))"),
 ("svi shift dropped",            "    return a + b * (rho * (k - m) + math.sqrt((k - m) ** 2 + sigma * sigma))", "    return a + b * (rho * k + math.sqrt(k ** 2 + sigma * sigma))"),
 ("svi curvature not squared",    "    return a + b * (rho * (k - m) + math.sqrt((k - m) ** 2 + sigma * sigma))", "    return a + b * (rho * (k - m) + math.sqrt((k - m) ** 2 + sigma))"),
 ("svi hyperbola degenerates",    "    return a + b * (rho * (k - m) + math.sqrt((k - m) ** 2 + sigma * sigma))", "    return a + b * (rho * (k - m) + abs(k - m))"),

 # The log-moneyness the slice is a function of: it is measured from the
 # FORWARD, and mirroring it mirrors the skew.
 ("moneyness mirrored",           "        k = math.log(K / self.F)", "        k = math.log(self.F / K)"),
 ("moneyness not logged",         "        k = math.log(K / self.F)", "        k = K / self.F"),
 ("variance floor removed",       "        return max(svi_total_variance(k, self.a, self.b, self.rho, self.m, self.sigma), 1e-8)", "        return svi_total_variance(k, self.a, self.b, self.rho, self.m, self.sigma)"),

 # Total variance -> annualised vol. w = sigma^2 * T, so both operations matter.
 ("vol not annualised",           "        return math.sqrt(self.total_variance(K) / self.T)", "        return math.sqrt(self.total_variance(K))"),
 ("vol multiplied by tenor",      "        return math.sqrt(self.total_variance(K) / self.T)", "        return math.sqrt(self.total_variance(K) * self.T)"),
 ("variance returned as vol",     "        return math.sqrt(self.total_variance(K) / self.T)", "        return self.total_variance(K) / self.T"),

 # 3. The fit-quality gate. The threshold was measured; see MAX_VOL_RMSE.
 ("fit gate disabled",            "    if err > max_vol_rmse:", "    if False:"),
 ("fit gate inverted",            "    if err > max_vol_rmse:", "    if err < max_vol_rmse:"),
 ("fit tolerance loosened",       "MAX_VOL_RMSE = 0.02", "MAX_VOL_RMSE = 0.5"),
 ("fit tolerance removed",        "MAX_VOL_RMSE = 0.02", "MAX_VOL_RMSE = 1e9"),
 ("rmse not converted to vol",    "        return self.rmse / (2.0 * atm_vol * self.T)", "        return self.rmse"),
 ("rmse conversion unscaled",     "        return self.rmse / (2.0 * atm_vol * self.T)", "        return self.rmse / (2.0 * atm_vol)"),

 # The residual the gate reads. Not rooted / not averaged both shrink it.
 ("residual not rooted",          "    resid = math.sqrt(sum((svi_total_variance(k, a, b, rho, m, s) - wm) ** 2\n                          for k, wm, _ in pts) / len(pts))", "    resid = sum((svi_total_variance(k, a, b, rho, m, s) - wm) ** 2\n                          for k, wm, _ in pts) / len(pts)"),
 ("residual not averaged",        "                          for k, wm, _ in pts) / len(pts))", "                          for k, wm, _ in pts))"),
 ("residual always zero",         "    resid = math.sqrt(sum((svi_total_variance(k, a, b, rho, m, s) - wm) ** 2\n                          for k, wm, _ in pts) / len(pts))", "    resid = 0.0"),

 # 2. The densification grid: the repair itself.
 ("dense grid narrowed to spot",  "    def dense_chain(self, symbol=\"SVI\", *, dte=None, n=81, lo=0.6, hi=1.6) -> OptionChain:", "    def dense_chain(self, symbol=\"SVI\", *, dte=None, n=81, lo=0.9, hi=1.1) -> OptionChain:"),
 ("dense grid coarsened",         "    def dense_chain(self, symbol=\"SVI\", *, dte=None, n=81, lo=0.6, hi=1.6) -> OptionChain:", "    def dense_chain(self, symbol=\"SVI\", *, dte=None, n=5, lo=0.6, hi=1.6) -> OptionChain:"),
 ("dense grid one sided",         "    def dense_chain(self, symbol=\"SVI\", *, dte=None, n=81, lo=0.6, hi=1.6) -> OptionChain:", "    def dense_chain(self, symbol=\"SVI\", *, dte=None, n=81, lo=1.0, hi=1.6) -> OptionChain:"),
 ("dense chain flat vol",         "            vol = self.iv(K)", "            vol = self.iv(self.F)"),
 ("dense spot not discounted",    "        spot = self.F * math.exp(-(self.r - self.q) * self.T)", "        spot = self.F"),
 ("dense spot discount inverted", "        spot = self.F * math.exp(-(self.r - self.q) * self.T)", "        spot = self.F * math.exp((self.r - self.q) * self.T)"),
 ("dense grid endpoint dropped",  "            K = round(spot * (lo + (hi - lo) * i / (n - 1)), 4)", "            K = round(spot * (lo + (hi - lo) * i / n), 4)"),
 ("worthless quotes kept",        "                if px <= 0:\n                    continue", "                if False:\n                    continue"),

 # The forward, recovered from put-call parity exactly as in rnd.
 ("forward parity sign flipped",  "    F = math.exp(chain.r * T) * (calls[k0] - puts[k0]) + k0", "    F = math.exp(chain.r * T) * (puts[k0] - calls[k0]) + k0"),
 ("forward strike not added",     "    F = math.exp(chain.r * T) * (calls[k0] - puts[k0]) + k0", "    F = math.exp(chain.r * T) * (calls[k0] - puts[k0])"),
 ("forward not discounted",       "    F = math.exp(chain.r * T) * (calls[k0] - puts[k0]) + k0", "    F = (calls[k0] - puts[k0]) + k0"),
 ("forward fallback ignores cost of carry", "        return chain.spot * math.exp((chain.r - chain.q) * T), strikes, calls, puts", "        return chain.spot, strikes, calls, puts"),

 # The OTM selection, and the minimum strike count for a 5-parameter fit.
 ("itm quotes fitted",            "        px = calls[K] if K >= F else puts[K]\n        kind = \"call\" if K >= F else \"put\"", "        px = puts[K] if K >= F else calls[K]\n        kind = \"put\" if K >= F else \"call\""),
 ("fitted price and kind disagree", "        kind = \"call\" if K >= F else \"put\"", "        kind = \"put\" if K >= F else \"call\""),
 ("market variance not annualised", "        pts.append((math.log(K / F), iv * iv * T, 1.0))", "        pts.append((math.log(K / F), iv * iv, 1.0))"),
 ("market variance not squared",  "        pts.append((math.log(K / F), iv * iv * T, 1.0))", "        pts.append((math.log(K / F), iv * T, 1.0))"),
 ("unsolvable quotes fitted",     "        if iv is None or iv <= 0:\n            continue", "        if iv is None:\n            continue"),
 ("minimum strikes lowered",      "    if len(pts) < 5:", "    if len(pts) < 3:"),
 ("minimum strikes removed",      "    if len(pts) < 5:", "    if False:"),

 # The no-arbitrage penalties, which keep the fitted slice in the admissible
 # region rather than merely close to the quotes.
 ("slope penalty dropped",        "        if b < 0:\n            pen += 1e6 * (-b) + 1e3", "        if False:\n            pen += 1e6 * (-b) + 1e3"),
 ("width penalty dropped",        "        if s <= 1e-6:\n            pen += 1e6 * (1e-6 - s) + 1e3", "        if False:\n            pen += 1e6 * (1e-6 - s) + 1e3"),
 ("correlation penalty dropped",  "        if abs(rho) >= 1:\n            pen += 1e6 * (abs(rho) - 1) + 1e3", "        if False:\n            pen += 1e6 * (abs(rho) - 1) + 1e3"),
 ("min variance penalty dropped", "        if min_w < 0:\n            pen += 1e6 * (-min_w)", "        if False:\n            pen += 1e6 * (-min_w)"),
 ("min variance formula wrong",   "        min_w = a + b * s * math.sqrt(max(1 - rho * rho, 0.0))", "        min_w = a + b * s * math.sqrt(max(1 + rho * rho, 0.0))"),
 ("min variance ignores width",   "        min_w = a + b * s * math.sqrt(max(1 - rho * rho, 0.0))", "        min_w = a + b * math.sqrt(max(1 - rho * rho, 0.0))"),

 # The no-arb reporting, which is what a caller checks before trusting a wing.
 ("no arb check inverted",        "            if svi_total_variance(k, self.a, self.b, self.rho, self.m, self.sigma) < 0:\n                return False", "            if svi_total_variance(k, self.a, self.b, self.rho, self.m, self.sigma) > 0:\n                return False"),
 ("no arb scan skipped",          "        for i in range(n + 1):", "        for i in range(0):"),
 ("no arb params unchecked",      "        return self.b >= 0 and abs(self.rho) < 1 and self.sigma > 0", "        return True"),
 ("no arb allows unit correlation", "        return self.b >= 0 and abs(self.rho) < 1 and self.sigma > 0", "        return self.b >= 0 and abs(self.rho) <= 1 and self.sigma > 0"),
 ("no arb ignores slope sign",    "        return self.b >= 0 and abs(self.rho) < 1 and self.sigma > 0", "        return abs(self.rho) < 1 and self.sigma > 0"),
 ("no arb scan range narrowed",   "    def no_arb_ok(self, kmin=-1.5, kmax=1.5, n=60) -> bool:", "    def no_arb_ok(self, kmin=-0.01, kmax=0.01, n=60) -> bool:"),

 # The optimiser's direction and stopping rules -- not its tuning constants.
 ("simplex maximises",            "        res.sort(key=lambda r: r[1])", "        res.sort(key=lambda r: r[1], reverse=True)"),
 ("simplex stops at once",        "def _nelder_mead(f, x0, *, step=0.15, no_improv_break=40, max_iter=3000,", "def _nelder_mead(f, x0, *, step=0.15, no_improv_break=1, max_iter=3000,"),
 ("simplex iteration cap removed", "def _nelder_mead(f, x0, *, step=0.15, no_improv_break=40, max_iter=3000,", "def _nelder_mead(f, x0, *, step=0.15, no_improv_break=40, max_iter=1,"),
 ("simplex improvement test flipped", "        if best < prev_best - 1e-12:", "        if best > prev_best - 1e-12:"),
 ("simplex restart dropped",      "    best = _nelder_mead(obj, best, step=0.03)", "    best = best"),
 ("fitted width sign kept",       "    return SVISurface(a, b, rho, m, abs(s), F, T, chain.r, chain.q, rmse=resid)", "    return SVISurface(a, b, rho, m, s, F, T, chain.r, chain.q, rmse=resid)"),
]
