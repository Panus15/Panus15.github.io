"""Ways the multi-position book could stop being a book.

engine/book_backtest.py exists to answer one question the single-position
harnesses structurally cannot: does the correlation-aware vega cap bind when
positions OVERLAP? So the mutants concentrate on what makes the overlap real and
on what the governor is shown.

  1. WHAT THE GOVERNOR SEES. `govern=False` hands it an empty book and is the
     comparison the whole module is built around; `against = book if govern`
     is the line that makes the aggregation run inside the loop at all.
  2. THE OVERLAP. The stagger offsets, the retirement at expiry, and the
     one-position-per-symbol guard. Without the stagger the book is N
     synchronised trades that aggregate to one big position.
  3. WHEN EQUITY MOVES. Equity used to be credited with a trade's whole outcome
     at entry, so the NEXT entry -- one bar later, positions overlapping by
     design -- was sized and gated on P&L from bars that had not happened. It
     now accrues as the day-by-day P&L lands; see the test for the measurement.
  4. THE REALISED P&L: premium, hedge, and the costs charged at entry and on
     every re-hedge.

THE KILL-SWITCH PATH IS NOT EXERCISED BY ORDINARY CALM FIXTURES. Over 60 runs
at the default 25% drawdown limit it fired zero times under either accounting,
so its tests need a deliberately tight limit.
"""

TARGET = "engine/book_backtest.py"
TESTS = "tests/test_book_backtest.py"

MUTANTS = [
 # 1. What the governor sees.
 ("governor sees an empty book",  "            against = book if govern else portfolio.Portfolio([], rho=rho)", "            against = portfolio.Portfolio([], rho=rho)"),
 ("ungoverned sees the book",     "            against = book if govern else portfolio.Portfolio([], rho=rho)", "            against = book"),
 ("govern flag inverted",         "            against = book if govern else portfolio.Portfolio([], rho=rho)", "            against = portfolio.Portfolio([], rho=rho) if govern else book"),
 ("governor ignores drawdown",    "            ok, reason = gov.can_add(leg, against, equity, drawdown)", "            ok, reason = gov.can_add(leg, against, equity, 0.0)"),
 ("governor sees no equity",      "            ok, reason = gov.can_add(leg, against, equity, drawdown)", "            ok, reason = gov.can_add(leg, against, starting_equity, drawdown)"),
 ("refusal admitted anyway",      "            if not ok:\n                _skip(\"kill_switch\" if \"KILL\" in reason else \"vega_cap\")", "            if False:\n                _skip(\"kill_switch\" if \"KILL\" in reason else \"vega_cap\")"),
 ("kill switch counted as a cap", "                n_blocked += \"KILL\" not in reason", "                n_blocked += 1"),
 ("cap block not counted",        "                n_blocked += \"KILL\" not in reason", "                n_blocked += 0"),
 ("kill switch labelled a cap",   "                _skip(\"kill_switch\" if \"KILL\" in reason else \"vega_cap\")", "                _skip(\"vega_cap\")"),

 # 2. The overlap.
 ("stagger disabled",             "    offset = {s: (i * max(1, dte // len(syms)) if stagger else 0)", "    offset = {s: 0"),
 ("stagger ignores the flag",     "if stagger else 0)\n              for i, s in enumerate(syms)}", "if True else 0)\n              for i, s in enumerate(syms)}"),
 ("stagger offset not spread",    "    offset = {s: (i * max(1, dte // len(syms)) if stagger else 0)", "    offset = {s: (max(1, dte // len(syms)) if stagger else 0)"),
 ("entry cycle not the tenor",    "            if s in open_until or (t - warmup - offset[s]) % dte != 0:", "            if s in open_until or (t - warmup - offset[s]) % (dte + 1) != 0:"),
 ("entry ignores the open slot",  "            if s in open_until or (t - warmup - offset[s]) % dte != 0:", "            if (t - warmup - offset[s]) % dte != 0:"),
 ("entry ignores the cycle",      "            if s in open_until or (t - warmup - offset[s]) % dte != 0:", "            if s in open_until:"),
 ("positions never retired",      "            book.positions.remove(pos)", "            pass"),
 ("retired a bar late",           "        for s in [s for s, (exp, _) in open_until.items() if exp <= t]:", "        for s in [s for s, (exp, _) in open_until.items() if exp < t]:"),
 ("retired a bar early",          "        for s in [s for s, (exp, _) in open_until.items() if exp <= t]:", "        for s in [s for s, (exp, _) in open_until.items() if exp <= t + 1]:"),
 ("slot not freed at expiry",     "            _, pos = open_until.pop(s)", "            _, pos = open_until[s]"),
 ("expiry set to the wrong bar",  "            open_until[s] = (t + dte, leg)", "            open_until[s] = (t + dte + 5, leg)"),
 ("trade past the data allowed",  "            if t + dte >= n:\n                continue", "            if False:\n                continue"),
 ("peak open not tracked",        "            max_open = max(max_open, len(open_until))", "            max_open = 1"),

 # 3. When equity moves.
 ("equity never accrues",         "        if t > warmup:\n            equity += pnl[t - 1]\n            peak = max(peak, equity)", "        if False:\n            equity += pnl[t - 1]\n            peak = max(peak, equity)"),
 ("equity accrues a bar early",   "            equity += pnl[t - 1]", "            equity += pnl[t]"),
 ("equity accrues twice",         "            equity += pnl[t - 1]", "            equity += pnl[t - 1] + pnl[t - 1]"),
 ("peak not tracked",             "            equity += pnl[t - 1]\n            peak = max(peak, equity)", "            equity += pnl[t - 1]"),
 ("peak is the trough",           "            equity += pnl[t - 1]\n            peak = max(peak, equity)", "            equity += pnl[t - 1]\n            peak = min(peak, equity)"),
 ("drawdown inverted",            "        drawdown = max(0.0, 1.0 - equity / peak)", "        drawdown = max(0.0, equity / peak - 1.0)"),
 ("drawdown from the start",      "        drawdown = max(0.0, 1.0 - equity / peak)", "        drawdown = max(0.0, 1.0 - equity / starting_equity)"),
 ("drawdown unfloored",           "        drawdown = max(0.0, 1.0 - equity / peak)", "        drawdown = 1.0 - equity / peak"),

 # Sizing.
 ("regime gate dropped",          "            if regime_stressed(trailing):\n                _skip(\"regime\")\n                continue", "            if False:\n                _skip(\"regime\")\n                continue"),
 ("zero size traded",             "            if size <= 0:\n                _skip(\"size_zero\")\n                continue", "            if False:\n                _skip(\"size_zero\")\n                continue"),
 ("sized on starting equity",     "            size = portfolio.size_by_cvar(\n                equity, template,", "            size = portfolio.size_by_cvar(\n                starting_equity, template,"),
 ("cvar limit ignored",           "                cvar_limit=cvar_limit)", "                cvar_limit=1.0)"),
 ("shock reversed",               "portfolio.vol_spike_scenarios(spot_shock=-0.12,", "portfolio.vol_spike_scenarios(spot_shock=0.12,"),
 ("short leg held long",          "                underlying=s, strike=K, expiry_days=dte, kind=\"put\", quantity=-size,", "                underlying=s, strike=K, expiry_days=dte, kind=\"put\", quantity=size,"),
 ("vol premium dropped",          "            iv = max(fvol * (1.0 + premium), 0.05)", "            iv = max(fvol, 0.05)"),
 ("vol floor removed",            "            iv = max(fvol * (1.0 + premium), 0.05)", "            iv = fvol * (1.0 + premium)"),

 # 4. The realised P&L.
 ("entry cost not charged",       "            pnl[t] -= entry_cost", "            pass"),
 ("entry cost sign flipped",      "            pnl[t] -= entry_cost", "            pnl[t] += entry_cost"),
 ("entry cost spread dropped",    "            entry_cost = size * contract_mult * 0.5 * (spread_frac * v_prev + 0.04)", "            entry_cost = size * contract_mult * 0.5 * 0.04"),
 ("entry cost is the full spread", "            entry_cost = size * contract_mult * 0.5 * (spread_frac * v_prev + 0.04)", "            entry_cost = size * contract_mult * (spread_frac * v_prev + 0.04)"),
 ("premium sign flipped",         "                day = (size * contract_mult * (v_prev - v_now)", "                day = (size * contract_mult * (v_now - v_prev)"),
 ("hedge pnl dropped",            "                       + hedge_prev * (S - S_prev)", "                       + 0.0"),
 ("hedge pnl sign flipped",       "                       + hedge_prev * (S - S_prev)", "                       - hedge_prev * (S - S_prev)"),
 ("hedge cost dropped",           "                       - abs(hedge_now - hedge_prev) * S * hedge_bps)", "                       - 0.0)"),
 ("hedge cost unsigned",          "                       - abs(hedge_now - hedge_prev) * S * hedge_bps)", "                       - (hedge_now - hedge_prev) * S * hedge_bps)"),
 ("hedge uses today's delta",     "                       + hedge_prev * (S - S_prev)", "                       + hedge_now * (S - S_prev)"),
 ("hedge not carried forward",    "                v_prev, hedge_prev = v_now, hedge_now", "                v_prev = v_now"),
 ("value not carried forward",    "                v_prev, hedge_prev = v_now, hedge_now", "                hedge_prev = hedge_now"),
 ("day pnl not booked",           "                pnl[u] += day", "                pass"),
 ("remaining time not shrinking", "                rem = (t + dte - u) / 252.0", "                rem = dte / 252.0"),
 ("trade total not recorded",     "            trades.append(total)", "            pass"),

 # The result.
 ("vega peak not tracked",        "            peak_vega = max(peak_vega, book.net_short_vega())", "            peak_vega = 0.0"),
 ("naive view is the book",       "            naive = max((abs(portfolio.Portfolio([p], rho=rho).net_short_vega())\n                         for _, p in open_until.values()), default=0.0)", "            naive = book.net_short_vega()"),
 ("empty universe accepted",      "    if not syms:\n        raise ValueError(", "    if False:\n        raise ValueError("),
 ("shortest history ignored",     "    n = min(len(p) for p in prices_by_symbol.values())", "    n = max(len(p) for p in prices_by_symbol.values())"),
]
