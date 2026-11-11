"""Ways the book-level risk layer could let the cap it advertises be breached.

This module exists because per-trade sizing gives false comfort: ten
"independent" 2%-risk short-vol trades are one correlated tail, not ten bets.
The engine's own demo measures the gap -- governed sizing peaked at 7,998 net
short vega against a stated 8,000 cap, while the ungoverned view reached 17,562
and never noticed, understating the book by 54-77%.

So the cap is the product. Every mutant here either lets exposure through the
cap or makes the aggregation report less exposure than the book carries:

    * correlation set to zero, which is the whole defect the module is about:
      it makes N correlated positions diversify away like sqrt(N)
    * same-underlying legs treated as merely correlated instead of identical
    * a comparison that admits the breaching trade rather than vetoing it
    * the drawdown kill-switch checked after the vega gates, or not at all
    * CVaR taking the mean of the tail's complement, or the best case

A cap that fails open is worse than no cap, because it is believed. That
sentence is in sizing.py's mutant table too, and it is the same lesson: both
modules exist to say no, and a rule that cannot say no is decoration.
"""

TARGET = "engine/portfolio.py"
TESTS = "tests/test_portfolio.py"

MUTANTS = [
 # The correlation-aware aggregation. This IS the module's thesis.
 ("correlation dropped",         "                c = 1.0 if (i == j or ui == uj) else rho", "                c = 1.0 if (i == j or ui == uj) else 0.0"),
 ("everything perfectly correlated", "                c = 1.0 if (i == j or ui == uj) else rho", "                c = 1.0"),
 ("same underlying merely correlated", "                c = 1.0 if (i == j or ui == uj) else rho", "                c = rho if (i == j or ui == uj) else rho"),
 ("own term dropped",            "                c = 1.0 if (i == j or ui == uj) else rho", "                c = 0.0 if i == j else rho"),
 ("default correlation zeroed",  "DEFAULT_VEGA_RHO = 0.8", "DEFAULT_VEGA_RHO = 0.0"),
 ("quadratic form not rooted",   "        return math.sqrt(max(total, 0.0))", "        return max(total, 0.0)"),
 ("negative quadratic form",     "        return math.sqrt(max(total, 0.0))", "        return math.sqrt(abs(total))"),

 # The governor's gates. Each `>` that becomes `>=` or `<` admits the trade
 # the gate exists to stop.
 ("short vega cap not enforced", "        if lim.max_net_short_vega is not None:", "        if False:"),
 ("short vega comparison flipped", "            if nsv > lim.max_net_short_vega:", "            if nsv < lim.max_net_short_vega:"),
 ("gross vega cap not enforced", "        if lim.max_gross_vega is not None:", "        if False:"),
 ("gross vega comparison flipped", "            if gv > lim.max_gross_vega:", "            if gv < lim.max_gross_vega:"),
 ("delta cap not enforced",      "        if lim.max_net_delta is not None:", "        if False:"),
 ("delta cap comparison flipped", "            if nd > lim.max_net_delta:", "            if nd < lim.max_net_delta:"),

 # The kill-switch. It is checked FIRST on purpose: no amount of vega headroom
 # makes new risk acceptable once the book is deep enough in the hole.
 ("kill switch disabled",        "        if current_drawdown >= lim.max_drawdown:", "        if False:"),
 ("kill switch boundary loosened", "        if current_drawdown >= lim.max_drawdown:", "        if current_drawdown > lim.max_drawdown:"),
 ("kill switch sign flipped",    "        if current_drawdown >= lim.max_drawdown:", "        if current_drawdown <= lim.max_drawdown:"),

 # CVaR sizing. The tail is the point; averaging the wrong slice of it, or
 # taking the best case, sizes a position against a loss that cannot happen.
 ("cvar takes the best case",    "        return max(-worst_pnl, 0.0)", "        return 0.0"),
 ("cvar loss unsigned",          "    return max(loss_acc / acc, 0.0)", "    return loss_acc / acc"),
 ("cvar denominator is the whole mass", "    return max(loss_acc / acc, 0.0)", "    return max(loss_acc / total_p, 0.0)"),
 ("empty scenarios sized anyway", "    if not scenarios:\n        return 0.0", "    if False:\n        return 0.0"),
 ("coarse tail not refused",     "    if 0 < tail < worst_mass - 1e-9 and len(rows) > 1:", "    if False:"),
 ("zero equity sized",           "    if equity <= 0 or cvar_limit <= 0:\n        return 0", "    if False:\n        return 0"),

 # Position-level Greek scaling: a short leg's vega must come back negative,
 # or the whole book's short exposure reads as long.
 ("short position not negated",  "    def is_short(self) -> bool:", "    def is_short(self) -> bool:\n        return False"),
]
