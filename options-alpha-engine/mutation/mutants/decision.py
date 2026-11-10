"""Ways the decision layer could let an unproven input size a position.

This module's whole design is two rules, and both of them are asymmetries that
a single flipped comparison undoes silently:

    1. an input with no evidence behind it may only REDUCE size. It can never
       flip a direction, create a trade, or raise conviction.
    2. an input that has been TESTED AND FAILED does not move size at all.

Rule 1 fails quietly in the expensive direction: an `effect` above 1.0 makes an
unmeasured signal manufacture position size, which is the exact mechanism by
which a research codebase turns into a loss. Rule 2 fails quietly in the
flattering direction: restoring the sector cut makes the printed decision look
like it reasons about the sector, when the number underneath was measured at
t=+0.71 and found empty.

NO TRADE is in here too. A decision layer that returns a nonzero multiplier on
a card that never fired is not a sizing bug, it is a trade nobody asked for.
"""

TARGET = "models/decision.py"
TESTS = "tests/test_decision.py"

MUTANTS = [
 # Rule 2: the sector quadrant was measured and predicts nothing, so it must
 # move size in NEITHER direction. Re-arming the cut is the regression this
 # module was rewritten to prevent.
 ("sector cut re-armed",         '    return Input("sector rotation", reading, TESTED_NO_EDGE, 1.0,', '    return Input("sector rotation", reading, TESTED_NO_EDGE, lagging_cut,'),
 ("disproven input relabelled",  '    return Input("sector rotation", reading, TESTED_NO_EDGE, 1.0,', '    return Input("sector rotation", reading, VALIDATED, 1.0,'),
 ("missing map treated as fine", '''    if rotation_point is None:
        return Input("sector rotation", "no sector map supplied", NO_DATA,''', '''    if rotation_point is None:
        return Input("sector rotation", "no sector map supplied", VALIDATED,'''),

 # Rule 1: the direction call has no backtest behind it, so it is carried for
 # context and sizes nothing. An effect above 1.0 would let it raise conviction.
 ("direction sizes the trade",   '''            "direction", f"{card.direction} (score {card.direction_score:+.2f})",
            FIXTURE_ONLY, 1.0,''', '''            "direction", f"{card.direction} (score {card.direction_score:+.2f})",
            FIXTURE_ONLY, 1.5,'''),
 ("direction claims validation", '''            "direction", f"{card.direction} (score {card.direction_score:+.2f})",
            FIXTURE_ONLY, 1.0,''', '''            "direction", f"{card.direction} (score {card.direction_score:+.2f})",
            VALIDATED, 1.0,'''),

 # The variance edge is the one input with tested machinery behind it, and it
 # is still only VALIDATED -- never proven on a live market.
 ("variance edge overclaimed",   '''        f"(P {getattr(card, 'p_vol', 0.0):.1%} vs Q {getattr(card, 'q_vol', 0.0):.1%})",
        VALIDATED,''', '''        f"(P {getattr(card, 'p_vol', 0.0):.1%} vs Q {getattr(card, 'q_vol', 0.0):.1%})",
        FIXTURE_ONLY,'''),

 # NO TRADE must stay NO TRADE.
 ("no trade still sized",        '''    if vol_side in ("NO TRADE", "", None):''', "    if False:"),
 ("no trade sized at full",      '''            expiry_days=getattr(card, "expiry_days", 0), action="NO TRADE",
            reason=getattr(card, "vol_reason", "the variance edge did not fire"),
            size_multiplier=0.0, inputs=inputs, warnings=warnings)''', '''            expiry_days=getattr(card, "expiry_days", 0), action="NO TRADE",
            reason=getattr(card, "vol_reason", "the variance edge did not fire"),
            size_multiplier=1.0, inputs=inputs, warnings=warnings)'''),
 ("empty vol_side not caught",   '''    if vol_side in ("NO TRADE", "", None):''', '''    if vol_side in ("NO TRADE",):'''),

 # The multiplier is the product of every input's effect. Dropping one, or
 # taking the best instead of compounding them, loosens every size.
 ("effects not compounded",      """    mult = 1.0
    for i in inputs:
        mult *= i.effect""", """    mult = 1.0
    for i in inputs:
        mult = min(mult, i.effect) if i.effect < 1.0 else mult"""),
 ("effects ignored entirely",    """    mult = 1.0
    for i in inputs:
        mult *= i.effect""", "    mult = 1.0"),
 ("worst input taken as size",   """    mult = 1.0
    for i in inputs:
        mult *= i.effect""", """    mult = max([i.effect for i in inputs] or [1.0])"""),

 # Warnings from the card must survive into the decision. A dropped warning is
 # a caveat the reader never sees next to a number they will act on.
 ("card warnings dropped",       '    warnings = list(getattr(card, "warnings", []) or [])', "    warnings = []"),

 # The size-down reason names which inputs cut it. Without it the reader sees a
 # 50% size and no way to find out why.
 ("size-down reason suppressed", "    if mult < 1.0:", "    if False:"),
 ("reason names the wrong ones", "                   + \", \".join(i.name for i in inputs if i.effect < 1.0))", "                   + \", \".join(i.name for i in inputs if i.effect >= 1.0))"),

 # The evidence ranking orders the inputs worst-first on the printed card.
 # Scrambling it buries the weakest evidence at the bottom.
 ("evidence ranking flattened",  "_RANK = {VALIDATED: 0, FIXTURE_ONLY: 1, NO_DATA: 2, TESTED_NO_EDGE: 3}", "_RANK = {VALIDATED: 0, FIXTURE_ONLY: 0, NO_DATA: 0, TESTED_NO_EDGE: 0}"),
 ("evidence ranking inverted",   "_RANK = {VALIDATED: 0, FIXTURE_ONLY: 1, NO_DATA: 2, TESTED_NO_EDGE: 3}", "_RANK = {VALIDATED: 3, FIXTURE_ONLY: 2, NO_DATA: 1, TESTED_NO_EDGE: 0}"),
]
