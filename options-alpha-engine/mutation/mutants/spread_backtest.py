"""Ways the defined-risk spread harness could make the wings look like free insurance.

engine/spread_backtest.py answers one question: do capped-loss structures earn their
premium against the delta-hedged naked straddle the rest of the repo trades? Its own
docstring records that the answer is not the one it was written expecting -- the
condor LOSES to the hedged straddle on a clean path and wins only where the hedge
fails -- so a harness that quietly flattered the wings would have reversed a finding.

  1. THE SETTLEMENT PAYOFF. Short legs owe intrinsic, long wings subtract it, and
     that subtraction IS the cap. Adding the wings instead of subtracting them, or
     leaving an intrinsic unfloored, turns a defined-risk trade into an unbounded one
     while still returning a number.
  2. THE DAILY MARK. Booking the whole result on expiry would penalise spreads for an
     accounting choice, so each leg is marked at its own frozen implied vol and the
     series telescopes to the settlement payoff. A sign slip here changes the Sharpe
     without changing the total.
  3. THE EXIT RULE. "Close at 50% of max profit or at 7 DTE" was RECOMMENDED on every
     ticket and implemented nowhere, so the numbers described a different strategy
     from the one on screen. The closing cost is the whole trade-off and is charged;
     omitting it is how a management rule measures as free money.
  4. THE VERDICT. Three branches, each protecting against a specific way of
     overselling the result: a calm sample is not evidence the wings worked, a worse
     tail is reported as worse, and a spread that gave up nothing is treated as
     suspicious rather than as a win.

MUTANTS THAT DELETE `t += dte` ARE NOT TRIED. They loop forever and the runner's only
defence is a ten-minute timeout per mutant, which costs far more than the information
is worth; the advance is exercised by every test that counts periods.

ONE EQUIVALENT MUTANT IS DELIBERATELY ABSENT. `rem <= 0` against `rem < 0` in the
daily mark decides whether the expiry bar is settled at intrinsic or priced by the
model at a zero tenor, and `pricing.price` at T = 0 returns exactly intrinsic --
checked on calls and puts in, at and out of the money. The two branches give the
same number on the only bar where they differ.

THE DEGENERATE-STRUCTURE GUARD IS REACHED BY SUBSTITUTING THE BUILDER. The spread
builders already refuse a zero-width wing, so `max_loss <= 0` cannot arrive through
them; the guard exists for a builder that hands one over, and the test makes one.
"""

TARGET = "engine/spread_backtest.py"
TESTS = "tests/test_spread_backtest.py"

MUTANTS = [
 # 1. The settlement payoff.
 ("wings added not subtracted",   "        total += intrinsic if leg.side == \"short\" else -intrinsic", "        total += intrinsic"),
 ("sides swapped",                "        total += intrinsic if leg.side == \"short\" else -intrinsic", "        total += -intrinsic if leg.side == \"short\" else intrinsic"),
 ("call intrinsic unfloored",     "        intrinsic = (max(terminal_spot - leg.strike, 0.0) if leg.kind == \"call\"", "        intrinsic = ((terminal_spot - leg.strike) if leg.kind == \"call\""),
 ("put intrinsic unfloored",      "                     else max(leg.strike - terminal_spot, 0.0))", "                     else (leg.strike - terminal_spot))"),
 ("call and put swapped",         "        intrinsic = (max(terminal_spot - leg.strike, 0.0) if leg.kind == \"call\"\n                     else max(leg.strike - terminal_spot, 0.0))", "        intrinsic = (max(leg.strike - terminal_spot, 0.0) if leg.kind == \"call\"\n                     else max(terminal_spot - leg.strike, 0.0))"),
 ("credit paid not received",     "    return mult * (spread.net_credit - spread_payoff(spread, terminal_spot)) \\", "    return mult * (-spread.net_credit - spread_payoff(spread, terminal_spot)) \\"),
 ("payoff received not paid",     "    return mult * (spread.net_credit - spread_payoff(spread, terminal_spot)) \\", "    return mult * (spread.net_credit + spread_payoff(spread, terminal_spot)) \\"),
 ("contract multiplier ignored",  "    return mult * (spread.net_credit - spread_payoff(spread, terminal_spot)) \\", "    return (spread.net_credit - spread_payoff(spread, terminal_spot)) \\"),
 ("settlement commission dropped", "        - commission * len(spread.legs)\n\n\ndef mark_spread_daily", "        - 0.0\n\n\ndef mark_spread_daily"),
 ("settlement commission on one leg", "        - commission * len(spread.legs)\n\n\ndef mark_spread_daily", "        - commission\n\n\ndef mark_spread_daily"),

 # 2. The daily mark.
 ("unmarkable leg accepted",      "    if any(v is None for _, v in legs):", "    if False:"),
 ("liability wings added",        "            total += px if leg.side == \"short\" else -px", "            total += px"),
 ("liability sides swapped",      "            total += px if leg.side == \"short\" else -px", "            total += -px if leg.side == \"short\" else px"),
 ("daily pnl sign flipped",       "        day[s] = mult * (prev - now)", "        day[s] = mult * (now - prev)"),
 ("daily pnl multiplier dropped", "        day[s] = mult * (prev - now)", "        day[s] = (prev - now)"),
 ("liability not carried forward", "        prev = now\n    return day", "        pass\n    return day"),
 ("entry commission not charged", "    day = {t: -commission * len(spread.legs)}", "    day = {t: 0.0}"),
 ("entry commission one leg",     "    day = {t: -commission * len(spread.legs)}", "    day = {t: -commission}"),
 ("time to expiry frozen",        "        rem = (t + dte - s) / 252.0\n        now = liability(prices[s], rem)", "        rem = dte / 252.0\n        now = liability(prices[s], rem)"),
 ("time to expiry in calendar days", "        rem = (t + dte - s) / 252.0\n        now = liability(prices[s], rem)", "        rem = (t + dte - s) / 365.0\n        now = liability(prices[s], rem)"),
 ("opening tenor in calendar days", "    prev = liability(prices[t], dte / 252.0)", "    prev = liability(prices[t], dte / 365.0)"),
 ("vols recovered at the wrong tenor", "        v = implied_vol(leg.price, prices[t], leg.strike, dte / 252.0, r, q, leg.kind)", "        v = implied_vol(leg.price, prices[t], leg.strike, dte / 365.0, r, q, leg.kind)"),

 # 3. The exit rule.
 ("profit target is full profit", "                      and cum >= take_profit_frac * max_gain)", "                      and cum >= max_gain)"),
 ("profit target ignored",        "                      and cum >= take_profit_frac * max_gain)", "                      and False)"),
 ("time stop disabled",           "        hit_time = min_dte_remaining is not None and remaining <= min_dte_remaining", "        hit_time = False"),
 ("time stop one day late",       "        hit_time = min_dte_remaining is not None and remaining <= min_dte_remaining", "        hit_time = min_dte_remaining is not None and remaining < min_dte_remaining"),
 ("closes on the open",           "        if bar == t or remaining <= 0:", "        if remaining <= 0:"),
 ("closes after settlement",      "        if bar == t or remaining <= 0:", "        if bar == t:"),
 ("max gain in the wrong units",  "    max_gain = mult * float(getattr(spread, \"max_gain\", 0.0) or 0.0)", "    max_gain = float(getattr(spread, \"max_gain\", 0.0) or 0.0)"),
 ("residual can go negative",     "        residual = max(max_gain - cum, 0.0)", "        residual = max_gain - cum"),
 ("closing spread not charged",   "        cost = close_spread_frac * residual + commission * len(spread.legs)", "        cost = commission * len(spread.legs)"),
 ("closing commission not charged", "        cost = close_spread_frac * residual + commission * len(spread.legs)", "        cost = close_spread_frac * residual"),
 ("closing commission one leg",   "        cost = close_spread_frac * residual + commission * len(spread.legs)", "        cost = close_spread_frac * residual + commission"),
 ("closing cost not deducted",    "        out[bar] = out[bar] - cost", "        out[bar] = out[bar]"),
 ("series not cut at the exit",   "        out = {b: marked[b] for b in bars if b <= bar}", "        out = dict(marked)"),
 ("exit reasons swapped",         "        return out, bar, (\"profit\" if hit_profit else \"time\")", "        return out, bar, (\"time\" if hit_profit else \"profit\")"),
 ("exit returns the wrong bar",   "        return out, bar, (\"profit\" if hit_profit else \"time\")", "        return out, bars[-1], (\"profit\" if hit_profit else \"time\")"),
 ("empty series not handled",     "    if not bars:\n        return dict(marked), None, \"expiry\"", "    if False:\n        return dict(marked), None, \"expiry\""),
 ("take profit constant",         "TAKE_PROFIT_FRAC = 0.50", "TAKE_PROFIT_FRAC = 0.90"),
 ("time stop constant",           "MIN_DTE_REMAINING = 7", "MIN_DTE_REMAINING = 3"),
 ("exit split profit and time swapped", "EXIT_SPLIT = {\"profit\": 0.60, \"time\": 0.40}", "EXIT_SPLIT = {\"profit\": 0.40, \"time\": 0.60}"),

 # 4. The verdict.
 ("calm sample not recognised",   "        if self.naked_worst >= 0:", "        if False:"),
 ("calm boundary",                "        if self.naked_worst >= 0:", "        if self.naked_worst > 0:"),
 ("tail saved sign flipped",      "        tail_saved = self.worst_trade - self.naked_worst", "        tail_saved = self.naked_worst - self.worst_trade"),
 ("worse tail not reported",      "        if tail_saved <= 0:", "        if False:"),
 ("worse tail boundary",          "        if tail_saved <= 0:", "        if tail_saved < 0:"),
 ("return given up sign flipped", "        ret_given_up = self.naked.total_return - self.metrics.total_return", "        ret_given_up = self.metrics.total_return - self.naked.total_return"),
 ("free lunch not flagged",       "        if ret_given_up <= 0:", "        if False:"),
 ("no comparison not handled",    "        if self.n_sold == 0 or self.naked is None:", "        if False:"),

 # run_spread_backtest: gates, realisation, bookkeeping.
 ("negative ev traded",           "        if not always_sell and sp.ev <= min_ev:", "        if False:"),
 ("always sell ignored",          "        if not always_sell and sp.ev <= min_ev:", "        if sp.ev <= min_ev:"),
 ("ev boundary",                  "        if not always_sell and sp.ev <= min_ev:", "        if not always_sell and sp.ev < min_ev:"),
 ("degenerate structure traded",  "        if sp.max_loss <= 0:", "        if False:"),
 ("event gate dropped",           "        if event_at is not None and event_at(t, trailing):", "        if False:"),
 ("event gate sees the future",   "        if event_at is not None and event_at(t, trailing):", "        if event_at is not None and event_at(t, prices):"),
 ("chain sees the future",        "        chain = chain_at(t, trailing)", "        chain = chain_at(t, prices)"),
 ("terminal on the wrong bar",    "        terminal = prices[t + dte]", "        terminal = prices[t + dte - 1]"),
 ("managed pnl not reported",     "            realised = sum(marked.values())", "            pass"),
 ("exits not counted",            "            exits[why] = exits.get(why, 0) + 1", "            pass"),
 ("daily pnl not booked",         "        for s, pv in marked.items():\n            pnl[s] += pv", "        for s, pv in marked.items():\n            pass"),
 ("worst is the best",            "        worst = realised if worst is None else min(worst, realised)", "        worst = realised if worst is None else max(worst, realised)"),
 ("best is the worst",            "        best = realised if best is None else max(best, realised)", "        best = realised if best is None else min(best, realised)"),
 ("cap budget drops commission",  "        budgeted = max(budgeted, contract_mult * sp.max_loss + commission * len(sp.legs))", "        budgeted = max(budgeted, contract_mult * sp.max_loss)"),
 ("cap budget is the smallest",   "        budgeted = max(budgeted, contract_mult * sp.max_loss + commission * len(sp.legs))", "        budgeted = min(budgeted, contract_mult * sp.max_loss + commission * len(sp.legs))"),
 ("capped call side inverted",    "                (l.kind == \"call\" and terminal > l.strike) or", "                (l.kind == \"call\" and terminal < l.strike) or"),
 ("capped put side inverted",     "                (l.kind == \"put\" and terminal < l.strike)) for l in sp.legs):", "                (l.kind == \"put\" and terminal > l.strike)) for l in sp.legs):"),
 ("naked comparison dropped",     "        if compare_naked:\n            from engine.signal_backtest", "        if False:\n            from engine.signal_backtest"),
 ("naked pnl not booked",         "                for s, pv in day.items():\n                    naked_pnl[s] += pv", "                for s, pv in day.items():\n                    pass"),
 ("naked worst is the best",      "                naked_worst = tot if naked_worst is None else min(naked_worst, tot)", "                naked_worst = tot if naked_worst is None else max(naked_worst, tot)"),
 ("naked count not kept",         "                n_naked += 1", "                pass"),
 ("warmup counted in the curve",  "    metrics = backtest.run_backtest(pnl[warmup:], starting_equity=starting_equity)", "    metrics = backtest.run_backtest(pnl, starting_equity=starting_equity)"),
 ("loop runs off the end",        "    while t + dte < n:", "    while t + dte <= n:"),
 ("structure map swapped",        "    build = {\"iron_condor\": spreads.iron_condor,\n             \"put_credit_spread\": spreads.put_credit_spread}[structure]", "    build = {\"iron_condor\": spreads.put_credit_spread,\n             \"put_credit_spread\": spreads.iron_condor}[structure]"),
]
