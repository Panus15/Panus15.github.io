"""Ways the verdict could say yes when it should say no.

This is the module that is allowed to refuse, so every mutant here is a way of
removing a refusal while leaving the output looking like a considered decision.
"""

TARGET = "engine/verdict.py"
TESTS = "tests/test_verdict.py"

MUTANTS = [
 ("refusals ignored",            "    v.tradeable = not v.refusals", "    v.tradeable = True"),
 ("losing rule passes",          "    if net.net <= 0:", "    if False:"),
 ("break-even passes",           "    if net.net <= 0:", "    if net.net < 0:"),
 ("default costs accepted",      "    if _using_default_costs(costs) and not costs_confirmed:", "    if False:"),
 ("confirmation ignored",        "    if _using_default_costs(costs) and not costs_confirmed:", "    if _using_default_costs(costs):"),
 ("either field counts as mine", '            and costs.swap_markup_annual == f["swap_markup_annual"].default)', '            or costs.swap_markup_annual == f["swap_markup_annual"].default)'),
 ("cheaper backtest accepted",   "    if cheapest < costs.round_turn_pips - 1e-9:", "    if False:"),
 ("dearest trade checked",       "    cheapest = min(t.cost_pips for t in res.trades)", "    cheapest = max(t.cost_pips for t in res.trades)"),
 ("sample floor removed",        "    if res.n < floor:", "    if False:"),
 ("floor lowered to 5",          "MIN_TRADES = 30", "MIN_TRADES = 5"),
 ("contested bar not raised",    "    floor = MIN_TRADES_CONTESTED if ev in CONTESTED else MIN_TRADES", "    floor = MIN_TRADES"),
 ("contested test inverted",     "    floor = MIN_TRADES_CONTESTED if ev in CONTESTED else MIN_TRADES", "    floor = MIN_TRADES_CONTESTED if ev not in CONTESTED else MIN_TRADES"),
 ("contested floor lowered",     "MIN_TRADES_CONTESTED = 100", "MIN_TRADES_CONTESTED = 30"),
 ("impossible p* accepted",      "    if v.win_rate_needed >= 1.0:", "    if False:"),
 ("p* boundary loosened",        "    if v.win_rate_needed >= 1.0:", "    if v.win_rate_needed > 1.0:"),
 ("significance not tested",     "    elif t_stat < t_req:", "    elif False:"),
 ("significance inverted",       "    elif t_stat < t_req:", "    elif t_stat > t_req:"),
 ("piled onto a losing rule",    "    elif t_stat < t_req:", "    if t_stat < t_req:"),
 ("t two-sided not one",         "    return NormalDist().inv_cdf(1.0 - alpha / n_hypotheses)", "    return NormalDist().inv_cdf(1.0 - alpha / (2.0 * n_hypotheses))"),
 ("no multiple-test correction", "    return NormalDist().inv_cdf(1.0 - alpha / n_hypotheses)", "    return NormalDist().inv_cdf(1.0 - alpha)"),
 ("alpha loosened to 50%",       "ALPHA = 0.05", "ALPHA = 0.5"),
 ("t ignores sample size",       "    return mean / (sd / math.sqrt(n))", "    return mean / sd"),
 ("zero variance always wins",   "        return math.inf if mean > 0 else 0.0", "        return math.inf"),
 ("one trade gets a t-stat",     "    if n < 2:", "    if n < 1:"),
 ("hypotheses default to one",   "        n_hypotheses = max(len(names), 1) * PARAMETER_COUNT", "        n_hypotheses = max(len(names), 1)"),
 ("empty name divides by zero",  "        n_hypotheses = max(len(names), 1) * PARAMETER_COUNT", "        n_hypotheses = len(names) * PARAMETER_COUNT"),
 ("ambiguity ignored",           "    if res.ambiguous_frac > AMBIGUITY_WARN_FRAC:", "    if False:"),
 ("ambiguity boundary tightened","    if res.ambiguous_frac > AMBIGUITY_WARN_FRAC:", "    if res.ambiguous_frac >= AMBIGUITY_WARN_FRAC:"),
 ("gross view uses net pips",    "    wins = [t.pips for t in res.trades if t.pips > 0]", "    wins = [t.net_pips for t in res.trades if t.net_pips > 0]"),
 ("gross view charges nothing",  "        cost_pips=statistics.fmean([t.cost_pips for t in res.trades]))", "        cost_pips=0.0)"),
 ("flipped counts every winner", "    flipped = sum(1 for t in res.trades if t.pips > 0 and t.net_pips <= 0)", "    flipped = sum(1 for t in res.trades if t.pips > 0)"),
 ("net expectancy re-charged",   "    net = res.expectancy()", "    net = gross"),
 ("risk cap removed",            "        v.suggested_risk_fraction = min(k, MAX_RISK_FRACTION)", "        v.suggested_risk_fraction = k"),
 ("full kelly not quartered",    "        k = kelly_fraction(net.win_rate, net.payoff_ratio) / KELLY_DIVISOR", "        k = kelly_fraction(net.win_rate, net.payoff_ratio)"),
 ("kelly halved not quartered",  "KELLY_DIVISOR = 4.0", "KELLY_DIVISOR = 2.0"),
 ("risk cap raised to 5%",       "MAX_RISK_FRACTION = 0.01", "MAX_RISK_FRACTION = 0.05"),
 ("refused rules still sized",   "    if v.tradeable:\n        k =", "    if True:\n        k ="),
 ("empty result not caught",     "    if res.n == 0:", "    if False:"),
 ("planned swaps risk and reward","    w = detection.reward / psize\n    l = detection.risk / psize", "    w = detection.risk / psize\n    l = detection.reward / psize"),
 ("planned charges no cost",     "    planned = Expectancy(win_rate=0.0, avg_win_pips=w, avg_loss_pips=l,\n                         cost_pips=c)", "    planned = Expectancy(win_rate=0.0, avg_win_pips=w, avg_loss_pips=l,\n                         cost_pips=0.0)"),
 ("planned ignores default costs","    if not costs_confirmed and _using_default_costs(costs):", "    if False:"),
 ("evidence not carried",        "    ev, ev_detail = evidence or EVIDENCE.get(\n        label, (NEVER_TESTED, \"no entry for this pattern\"))", "    ev, ev_detail = (NEVER_TESTED, \"\")"),
 ("evidence override ignored",   "    ev, ev_detail = evidence or EVIDENCE.get(\n        label, (NEVER_TESTED, \"no entry for this pattern\"))", "    ev, ev_detail = EVIDENCE.get(\n        label, (NEVER_TESTED, \"no entry for this pattern\"))"),
 ("summary hides refusals",      "        for r in self.refusals:\n            lines.append(f\"  REFUSED {r}\")", "        for r in []:\n            lines.append(f\"  REFUSED {r}\")"),
 ("contested note dropped",      "    if ev in CONTESTED and not v.refusals:", "    if False:"),
 ("win rate gap sign flipped",   "        return self.win_rate_needed - self.gross.win_rate", "        return self.gross.win_rate - self.win_rate_needed"),
]
