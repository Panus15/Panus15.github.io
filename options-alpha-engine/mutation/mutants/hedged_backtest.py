"""Ways the delta-hedged short-straddle harness could report a harvest it did not earn.

engine/hedged_backtest.py is the harness every other result here is compared
against, so a flattering error in it flatters everything downstream. The mutants
are grouped by what each would silently make easier.

  1. THE STRADDLE AND THE HEDGE. Short a straddle, hold the opposite delta in
     shares, and the daily P&L is the option's mark-to-market gain, the shares'
     P&L on YESTERDAY's hedge, less the cost of rebalancing. Using today's delta
     for the shares P&L is a look-ahead; dropping the rebalancing cost removes the
     one friction a gamma-short book pays every single day.
  2. THE GATES. Regime, scheduled event, CVaR size, governor. Each one skipping
     the trade is the point; a gate that fails open turns the harness back into
     the naive sell-everything backtest the module exists to improve on.
  3. THE INTRA-TRADE KILL-SWITCH. A limit that is only checked when flat is not a
     limit. It marks equity daily and LIQUIDATES at the exit spread, rather than
     riding to expiry; the mutants break the trigger, the cost of leaving, and
     the bookkeeping that records it.
  4. THE VEGA-LOSS MODEL (opt-in). IV reprices off trailing realised vol with a
     deadband and a cap, using only past prices. Removing the deadband makes the
     calm path react to sampling noise; removing the cap lets a gap produce an
     unbounded IV.
"""

TARGET = "engine/hedged_backtest.py"
TESTS = "tests/test_hedged_backtest.py"

MUTANTS = [
 # 1a. The straddle itself.
 ("straddle uses the put twice",  "    return c.price + p.price, c.delta + p.delta", "    return c.price + c.price, c.delta + c.delta"),
 ("straddle value is a spread",   "    return c.price + p.price, c.delta + p.delta", "    return c.price - p.price, c.delta + p.delta"),
 ("straddle delta is a difference", "    return c.price + p.price, c.delta + p.delta", "    return c.price + p.price, c.delta - p.delta"),
 ("expired straddle not floored", "        val = max(S - K, 0.0) + max(K - S, 0.0)", "        val = (S - K) + (K - S)"),
 ("expired straddle keeps delta", "        return val, 0.0", "        return val, 1.0"),

 # 1b. The daily P&L of the hedged short.
 ("option mtm sign flipped",      "            option_mtm = size * contract_mult * (v_prev - v_now)      # short: gain if value falls", "            option_mtm = size * contract_mult * (v_now - v_prev)      # short: gain if value falls"),
 ("shares pnl uses today's delta", "            shares_pnl = hedge_prev * (S - S_prev)", "            shares_pnl = size * contract_mult * d_now * (S - S_prev)"),
 ("shares pnl dropped",           "            shares_pnl = hedge_prev * (S - S_prev)", "            shares_pnl = 0.0"),
 ("shares pnl sign flipped",      "            shares_pnl = hedge_prev * (S - S_prev)", "            shares_pnl = -hedge_prev * (S - S_prev)"),
 ("rebalance cost dropped",       "            rebal_cost = abs(hedge_now - hedge_prev) * S * hedge_bps", "            rebal_cost = 0.0"),
 ("rebalance cost unsigned",      "            rebal_cost = abs(hedge_now - hedge_prev) * S * hedge_bps", "            rebal_cost = (hedge_now - hedge_prev) * S * hedge_bps"),
 ("rebalance cost ignores price", "            rebal_cost = abs(hedge_now - hedge_prev) * S * hedge_bps", "            rebal_cost = abs(hedge_now - hedge_prev) * hedge_bps"),
 ("day pnl adds the cost",        "            day_pnl = option_mtm + shares_pnl - rebal_cost", "            day_pnl = option_mtm + shares_pnl + rebal_cost"),
 ("day pnl not booked",           "            pnl_by_day[s] += day_pnl", "            pass"),
 ("trade total not accumulated",  "            trade_total += day_pnl", "            pass"),
 ("state not carried forward",    "            v_prev, hedge_prev = v_now, hedge_now", "            v_prev = v_now"),
 ("hedge not carried forward",    "            v_prev, hedge_prev = v_now, hedge_now", "            hedge_prev = hedge_now"),
 ("time to expiry frozen",        "            rem = (i0 + dte - s) / 252.0", "            rem = dte / 252.0"),
 ("time to expiry in calendar days", "            rem = (i0 + dte - s) / 252.0", "            rem = (i0 + dte - s) / 365.0"),
 ("opening hedge not neutral",    "        hedge_prev = size * contract_mult * d_prev            # shares to hold (neutralise)", "        hedge_prev = 0.0"),
 ("opening hedge wrong size",     "        hedge_prev = size * contract_mult * d_prev            # shares to hold (neutralise)", "        hedge_prev = contract_mult * d_prev"),

 # Entry cost.
 ("entry cost not charged",       "        pnl_by_day[i0] -= trade_cost", "        pass"),
 ("entry cost not in the trade",  "        trade_total = -trade_cost", "        trade_total = 0.0"),
 ("entry crosses the full spread", "        trade_cost = size * contract_mult * 0.5 * entry_spread", "        trade_cost = size * contract_mult * entry_spread"),
 ("entry spread has no floor",    "        entry_spread = spread_frac * v_prev + 0.04", "        entry_spread = spread_frac * v_prev"),
 ("entry spread has no proportional part", "        entry_spread = spread_frac * v_prev + 0.04", "        entry_spread = 0.04"),
 ("entry cost sign flipped",      "        pnl_by_day[i0] -= trade_cost", "        pnl_by_day[i0] += trade_cost"),

 # 2. The gates.
 ("regime gate dropped",          "        if regime_stressed(trailing):\n            _skip(\"regime\")\n            i0 += dte\n            continue", "        if False:\n            _skip(\"regime\")\n            i0 += dte\n            continue"),
 ("event gate dropped",           "        if event_at is not None and event_at(i0, trailing):", "        if False:"),
 ("event gate sees the future",   "        if event_at is not None and event_at(i0, trailing):", "        if event_at is not None and event_at(i0, prices):"),
 ("event gate off by a bar",      "        if event_at is not None and event_at(i0, trailing):", "        if event_at is not None and event_at(i0 + 1, trailing):"),
 ("zero size traded",             "        if size <= 0:\n            _skip(\"size_zero\")", "        if False:\n            _skip(\"size_zero\")"),
 ("governor refusal ignored",     "        if not ok:\n            _skip(\"kill_switch\" if \"KILL\" in reason else \"vega_cap\")", "        if False:\n            _skip(\"kill_switch\" if \"KILL\" in reason else \"vega_cap\")"),
 ("kill switch mislabelled",      "            _skip(\"kill_switch\" if \"KILL\" in reason else \"vega_cap\")", "            _skip(\"vega_cap\")"),
 ("governor sees no drawdown",    "        ok, reason = gov.can_add(short_straddle_leg, portfolio.Portfolio(), equity, drawdown)", "        ok, reason = gov.can_add(short_straddle_leg, portfolio.Portfolio(), equity, 0.0)"),
 ("skip does not advance",        "            _skip(\"regime\")\n            i0 += dte\n            continue", "            _skip(\"regime\")\n            continue"),
 ("sizing ignores equity",        "        size = portfolio.size_by_cvar(\n            equity, template,", "        size = portfolio.size_by_cvar(\n            starting_equity, template,"),
 ("sizing shock reversed",        "portfolio.vol_spike_scenarios(spot_shock=-0.12, vol_bump=0.15)", "portfolio.vol_spike_scenarios(spot_shock=0.12, vol_bump=0.15)"),
 ("cvar limit ignored",           "            cvar_limit=cvar_limit,\n        )", "            cvar_limit=1.0,\n        )"),
 ("iv premium dropped",           "        iv_entry = max(fvol * (1.0 + premium), 0.05)", "        iv_entry = max(fvol, 0.05)"),
 ("iv floor removed",             "        iv_entry = max(fvol * (1.0 + premium), 0.05)", "        iv_entry = fvol * (1.0 + premium)"),
 ("strike not at the money",      "        K = round(spot, 0)", "        K = round(spot * 1.05, 0)"),
 ("forecast sees the future",     "        fvol = volforecast.har_rv_forecast(trailing)", "        fvol = volforecast.har_rv_forecast(prices)"),
 ("drawdown not a fraction of peak", "        drawdown = max(0.0, 1.0 - equity / peak)", "        drawdown = max(0.0, 1.0 - equity / starting_equity)"),
 ("short leg held long",          "            quantity=-size, entry_price=template.entry_price,", "            quantity=size, entry_price=template.entry_price,"),

 # 3. The intra-trade kill-switch.
 ("kill switch disabled",         "            if kill_midtrade and rem > 0:", "            if False:"),
 ("kill switch runs at expiry",   "            if kill_midtrade and rem > 0:", "            if kill_midtrade:"),
 ("kill switch ignores the flag", "            if kill_midtrade and rem > 0:", "            if rem > 0:"),
 ("equity not marked",            "                eq_now = equity + trade_total", "                eq_now = equity"),
 ("drawdown vs start not peak",   "                dd_now = max(0.0, 1.0 - eq_now / max(peak, eq_now))", "                dd_now = max(0.0, 1.0 - eq_now / starting_equity)"),
 ("peak not allowed to rise",     "                dd_now = max(0.0, 1.0 - eq_now / max(peak, eq_now))", "                dd_now = max(0.0, 1.0 - eq_now / peak)"),
 ("kill threshold inverted",      "                if dd_now > limits.max_drawdown:", "                if dd_now < limits.max_drawdown:"),
 ("kill at the boundary",         "                if dd_now > limits.max_drawdown:", "                if dd_now >= limits.max_drawdown:"),
 ("exit cost not charged",        "                    pnl_by_day[s] -= exit_cost", "                    pass"),
 ("exit cost not in the trade",   "                    trade_total -= exit_cost", "                    pass"),
 ("exit spread dropped",          "                    exit_cost = (size * contract_mult * 0.5\n                                 * (spread_frac * v_now + 0.04)\n                                 + abs(hedge_now) * S * hedge_bps)", "                    exit_cost = abs(hedge_now) * S * hedge_bps"),
 ("exit hedge unwind dropped",    "                                 + abs(hedge_now) * S * hedge_bps)", "                                 + 0.0)"),
 ("kill does not liquidate",      "                    killed_dd = max(killed_dd, dd_now)\n                    break", "                    killed_dd = max(killed_dd, dd_now)"),
 ("kill not counted",             "                    n_killed += 1", "                    pass"),
 ("worst drawdown not recorded",  "                    killed_dd = max(killed_dd, dd_now)", "                    pass"),
 ("worst drawdown is the last",   "                    killed_dd = max(killed_dd, dd_now)", "                    killed_dd = dd_now"),

 # Book-keeping at the end of a trade.
 ("equity not updated",           "        equity += trade_total\n        peak = max(peak, equity)", "        peak = max(peak, equity)"),
 ("peak not updated",             "        equity += trade_total\n        peak = max(peak, equity)", "        equity += trade_total"),
 ("trade not recorded",           "        trade_pnl.append(trade_total)", "        pass"),
 ("trade not counted",            "        n_trades += 1", "        pass"),
 ("holding period not advanced",  "        n_trades += 1\n        i0 += dte", "        n_trades += 1\n        i0 += 1"),
 ("loop runs off the end",        "    while i0 + dte < n:", "    while i0 + dte <= n:"),
 ("warmup ignored",               "    active = pnl_by_day[warmup:]", "    active = pnl_by_day"),

 # 4. The vega-loss model.
 ("vega loss always on",          "            if model_vega_loss:", "            if True:"),
 ("vega loss never on",           "            if model_vega_loss:", "            if False:"),
 ("deadband removed",             "                shock = max(0.0, recent_rv / fvol - (1.0 + iv_deadband))", "                shock = max(0.0, recent_rv / fvol - 1.0)"),
 ("shock not floored",            "                shock = max(0.0, recent_rv / fvol - (1.0 + iv_deadband))", "                shock = recent_rv / fvol - (1.0 + iv_deadband)"),
 ("shock ratio inverted",         "                shock = max(0.0, recent_rv / fvol - (1.0 + iv_deadband))", "                shock = max(0.0, fvol / recent_rv - (1.0 + iv_deadband))"),
 ("shock cap removed",            "                iv_t = iv_entry * (1.0 + iv_beta * min(shock, iv_shock_cap))", "                iv_t = iv_entry * (1.0 + iv_beta * shock)"),
 ("beta ignored",                 "                iv_t = iv_entry * (1.0 + iv_beta * min(shock, iv_shock_cap))", "                iv_t = iv_entry * (1.0 + min(shock, iv_shock_cap))"),
 ("iv falls in a shock",          "                iv_t = iv_entry * (1.0 + iv_beta * min(shock, iv_shock_cap))", "                iv_t = iv_entry * (1.0 - iv_beta * min(shock, iv_shock_cap))"),
 ("realised vol sees the future", "    return volforecast.close_to_close_vol(prices[lo:s + 1], window=window)", "    return volforecast.close_to_close_vol(prices[lo:s + 2], window=window)"),
 ("realised vol window off",      "    lo = max(0, s - window)", "    lo = max(0, s - 2 * window)"),
]
