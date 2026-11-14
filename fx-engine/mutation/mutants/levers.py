"""Ways the advice could offer a fix that is not one.

The result worth the module is the refusal: when gross expectancy is zero or
negative, no change to cost, geometry, hold time or frequency helps. A mutant
that loses that check turns the module into the broker-shopping advice it exists
to contradict.
"""

TARGET = "engine/levers.py"
TESTS = "tests/test_levers.py"

MUTANTS = [
 ("negative gross still fixable","    gross = _gross_of(e)\n    if gross <= 0:\n        return Lever(\n            \"cost\"", "    gross = _gross_of(e)\n    if False:\n        return Lever(\n            \"cost\""),
 ("negative gross boundary",     "    gross = _gross_of(e)\n    if gross <= 0:\n        return Lever(\n            \"cost\"", "    gross = _gross_of(e)\n    if gross < 0:\n        return Lever(\n            \"cost\""),
 ("widening a negative edge ok", "    if gross <= 0:\n        return Lever(\n            \"geometry\"", "    if False:\n        return Lever(\n            \"geometry\""),
 ("hopeless verdict dropped",    "    if gross <= 0:\n        lines.append(", "    if False:\n        lines.append("),
 ("cost reachability inverted",  "    ok = gross > CHEAPEST_ROUND_TURN", "    ok = gross < CHEAPEST_ROUND_TURN"),
 ("cost always reachable",       "    ok = gross > CHEAPEST_ROUND_TURN", "    ok = True"),
 ("break-even cost is the cost", "        \"cost\", e.cost_pips, gross, ok,", "        \"cost\", e.cost_pips, e.cost_pips, ok,"),
 ("saving quoted as a gain",     "    at_best = gross - CHEAPEST_ROUND_TURN", "    at_best = e.cost_pips - gross"),
 ("cheapest account overstated", "CHEAPEST_ROUND_TURN = TYPICAL_ROUND_TURN_PIPS[0]", "CHEAPEST_ROUND_TURN = 0.05"),
 ("widening factor inverted",    "    k = e.cost_pips / gross if gross > 0 else math.inf", "    k = gross / e.cost_pips if gross > 0 else math.inf"),
 ("widening limit removed",      "    ok = k <= MAX_PLAUSIBLE_WIDENING", "    ok = True"),
 ("widening limit raised",       "MAX_PLAUSIBLE_WIDENING = 5.0", "MAX_PLAUSIBLE_WIDENING = 50.0"),
 ("already-clears branch gone",  "    if k <= 1.0:", "    if False:"),
 ("widening caveat dropped",     "f\"wider target reached less often, in opposite directions. \" + (", "f\"\" + ("),
 ("re-run advice dropped",       "            f\"Re-run the backtest at {e.avg_win_pips * k:.0f}/\"", "            f\"Fine at {e.avg_win_pips * k:.0f}/\""),
 ("win rate offered as a dial",  "        \"win rate\", e.win_rate, e.break_even_win_rate, False,", "        \"win rate\", e.win_rate, e.break_even_win_rate, True,"),
 ("win rate cost share dropped", "        f\"{e.cost_pips / (e.avg_win_pips + e.avg_loss_pips):.1%} of the \"", "        f\"some of the \""),
 ("payoff offered as a dial",    "        \"payoff\", e.payoff_ratio, need_b, False,", "        \"payoff\", e.payoff_ratio, need_b, True,"),
 ("payoff target wrong",         "    need_w = ((1.0 - e.win_rate) * e.avg_loss_pips + e.cost_pips) / e.win_rate", "    need_w = ((1.0 - e.win_rate) * e.avg_loss_pips) / e.win_rate"),
 ("no-winners case unhandled",   "    if e.win_rate <= 0:", "    if False:"),
 ("trade less offered as a fix", "    if e.net > 0:\n        return (f\"at {trades_per_year:,.0f} trades a year this is \"", "    if True:\n        return (f\"at {trades_per_year:,.0f} trades a year this is \""),
 ("annual total wrong",          "    annual = e.net * trades_per_year", "    annual = e.gross * trades_per_year"),
 ("profitable case not noticed", "    if e.net > 0:\n        lines.append(\"VERDICT: it already clears", "    if False:\n        lines.append(\"VERDICT: it already clears"),
 ("recommendation not named",    "        lines.append(f\"  -> {doable[0]}\")", "        lines.append(\"\")"),
 ("unreachable verdict dropped", "            \"VERDICT: there is a gross edge and nothing in your control is \"", "            \"\" \\\n            \""),
]
