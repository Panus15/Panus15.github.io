"""Ways the sizing cap could loosen exactly where the tail gets fatter.

This module's own docstring records the defect it was built around: a 2% cap
that divided the budget by the PREMIUM instead of the LOSS, so on a $100,000
account it sold 2,000 far-OTM 1-day puts for a maximum loss of 144,000% of the
account — and the cheaper the option, the more it sold. Cheap means far
out-of-the-money, which is the strike that is harmless until the day it is not.

So the mutants here are chosen for that direction. Every one of them makes the
engine size LARGER or admits a position whose loss it cannot bound:

    * dividing by premium rather than loss, the original bug
    * treating an unbounded short call as sizeable
    * a short put whose loss is capped at the premium kept
    * a wing that widens the loss instead of capping it
    * rounding contracts UP across the budget

A cap that fails open is worse than no cap, because it is believed.

THREE EQUIVALENT MUTANTS ARE DELIBERATELY ABSENT. `position_size` guards its
inputs at the top (`account_equity <= 0 or max_risk_frac <= 0: return 0`) and
ALSO clamps its result (`max(int(...), 0)`). Those two are redundant with each
other: remove the guard and a negative budget floor-divides to a negative that
the clamp catches; remove the clamp and the guard has already returned 0. So
removing either one alone changes no answer for any input, and no test can kill
either -- the redundancy is the point, not dead code. Removing BOTH would admit
a negative size, which is why neither is removed.
"""

TARGET = "engine/sizing.py"
TESTS = "tests/test_sizing.py"

MUTANTS = [
 # The original defect, restored. A long position risks what it paid; a SHORT
 # one does not, and this is the branch that decides which.
 ("short sized on its premium",  "    if quantity >= 0:                       # long: the premium is the whole risk\n        return max(entry_price, 0.0) * multiplier", "    if True:\n        return max(entry_price, 0.0) * multiplier"),
 ("long and short swapped",      "    if quantity >= 0:                       # long: the premium is the whole risk", "    if quantity <= 0:                       # long: the premium is the whole risk"),

 # A naked short call has no ceiling. Returning anything finite here invents a
 # worst case and calls it risk management.
 ("naked call given a bound",    "    return UNBOUNDED", "    return max(strike * multiplier - premium, 0.0)"),
 ("unbounded loss sized anyway", "    if max_loss_per_contract == UNBOUNDED or max_loss_per_contract == float(\"inf\"):", "    if False:"),

 # A short put's loss is the strike going to zero, less the premium kept --
 # not the premium, and not the strike with the premium added.
 ("short put loss is premium",   "    if kind == \"put\":\n        return max(strike * multiplier - premium, 0.0)", "    if kind == \"put\":\n        return premium"),
 ("premium added not subtracted", "        return max(strike * multiplier - premium, 0.0)", "        return max(strike * multiplier + premium, 0.0)"),

 # A long wing converts a naked short into a spread and CAPS the loss at the
 # width. Adding the premium instead of subtracting it overstates the cap, and
 # dropping the width makes the wing cosmetic.
 ("wing widens the loss",        "        return max(width - premium, 0.0)", "        return max(width + premium, 0.0)"),
 ("wing ignored",                "    if wing_strike is not None:", "    if False:"),
 ("wing width unsigned",         "        width = abs(wing_strike - strike) * multiplier", "        width = (wing_strike - strike) * multiplier"),

 # The budget itself.
 ("budget is the whole account", "    budget = max_risk_frac * account_equity", "    budget = account_equity"),
 ("contracts rounded up",        "    return max(int(budget // max_loss_per_contract), 0)", "    return max(-(-int(budget) // int(max_loss_per_contract or 1)), 0)"),

 # The guards. Each one admits an input that produces a meaningless size.
 ("zero loss accepted",          "    if max_loss_per_contract <= 0:", "    if False:"),
 ("bad kind accepted",           "    if kind not in (\"put\", \"call\"):", "    if False:"),
 ("kind case sensitive",         "    kind = kind.lower()", "    kind = kind"),

 # kelly_fraction is kept because it is correct for what it says it is. It must
 # not quietly return something usable on inputs it cannot represent.
 ("kelly returns a negative bet", "    return max(f, 0.0)", "    return f"),
 ("kelly divides by zero odds",  "    if odds <= 0:\n        return 0.0", "    if False:\n        return 0.0"),
]
