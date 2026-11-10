"""The five ways a backtest lies, each introduced deliberately.

An inflated backtest does not error. The equity curve just bends upward, so
every one of these mutants produces a system that looks better and is wrong.
"""

TARGET = "engine/backtest.py"
TESTS = "tests/test_backtest.py"

MUTANTS = [
 # The measured-spread path, and the side table that is the only reader of
 # `side` in the engine. A swapped leg is invisible on a steady spread.
 ("bid legs swapped",            '    "bid": {1: (1.0, 0.0), -1: (0.0, 1.0)},', '    "bid": {1: (0.0, 1.0), -1: (1.0, 0.0)},'),
 ("ask legs swapped",            '    "ask": {1: (0.0, 1.0), -1: (1.0, 0.0)},', '    "ask": {1: (1.0, 0.0), -1: (0.0, 1.0)},'),
 ("mid pays a whole spread twice", '    "mid": {1: (0.5, 0.5), -1: (0.5, 0.5)},', '    "mid": {1: (1.0, 1.0), -1: (1.0, 1.0)},'),
 ("direction ignored",           "    w_in, w_out = SPREAD_LEGS[side][direction]", "    w_in, w_out = SPREAD_LEGS[side][1]"),
 ("entry and exit swapped",      "    return w_in * entry_spread + w_out * exit_spread", "    return w_in * exit_spread + w_out * entry_spread"),
 ("bad side allowed",            "    if side not in SPREAD_LEGS:\n        raise ValueError(f\"side must be one of {sorted(SPREAD_LEGS)}, got \"\n                         f\"{side!r} — it decides which bar's spread is paid\")", "    if False:\n        raise ValueError(f\"side must be one of {sorted(SPREAD_LEGS)}, got \"\n                         f\"{side!r} — it decides which bar's spread is paid\")"),
 ("bad direction allowed",       "    if direction not in (1, -1):", "    if False:"),
 ("spreads without side allowed", "    if (spreads is None) != (side is None):", "    if False:"),
 ("measured spread ignored",     "        if spreads is not None:", "        if False:"),
 ("missing spread invented",     "            if a is not None and b is not None:", "            if True:"),
 ("spread not converted to pips", "                sp = spread_paid(side, d.direction, a, b) / psize", "                sp = spread_paid(side, d.direction, a, b)"),
 ("source of costs not recorded", "    res = BacktestResult(costs_from_file=spreads is not None)", "    res = BacktestResult()"),
 ("ambiguous resolves to target", 'exit_px, exit_i, outcome = stop, j, "stop"\n                ambiguous = True', 'exit_px, exit_i, outcome = target, j, "target"\n                ambiguous = True'),
 ("ambiguity not counted",       "                res.ambiguous_bars += 1", "                pass"),
 ("gap fills at the stop",       'exit_px, exit_i, outcome = b.o, j, "stop"\n                slippage = abs(b.o - stop) / psize', 'exit_px, exit_i, outcome = stop, j, "stop"\n                slippage = abs(b.o - stop) / psize'),
 ("gap check removed",           "            if gapped:", "            if False:"),
 ("slippage not counted",        "                res.slipped_stops += 1", "                pass"),
 ("entry at detection bar",      "        i0 = d.index + 1                      # strictly after the completing bar", "        i0 = d.index"),
 ("entry at close not open",     "        entry_px = bars[i0].o", "        entry_px = bars[i0].c"),
 ("unresolved dropped",          "        if exit_px is None:", "        if False and exit_px is None:"),
 ("cost not charged",            "        cost = costs.total_cost_pips(nights, entry_px)", "        cost = 0.0"),
 ("nights always one",           "        cost = costs.total_cost_pips(nights, entry_px)", "        cost = costs.total_cost_pips(1, entry_px)"),
 ("bar size ignored",            "        nights = int(max(exit_i - i0, 0) / bars_per_night)", "        nights = int(max(exit_i - i0, 0))"),
 ("part of a night charged",     "        nights = int(max(exit_i - i0, 0) / bars_per_night)", "        nights = max(exit_i - i0, 0) / bars_per_night"),
 ("night rounded up",            "        nights = int(max(exit_i - i0, 0) / bars_per_night)", "        nights = int(max(exit_i - i0, 0) / bars_per_night) + 1"),
 ("bar size not validated",      "    if bars_per_night <= 0:", "    if False:"),
 ("trade nights default wrong",  "    def nights(self, bars_per_night: float = 1.0) -> int:", "    def nights(self, bars_per_night: float = 24.0) -> int:"),
 ("direction ignored",           "        gross = (exit_px - entry_px) * d.direction / psize", "        gross = (exit_px - entry_px) / psize"),
 ("wins counted on gross",       "        return [t for t in self.trades if t.net_pips > 0]", "        return [t for t in self.trades if t.pips > 0]"),
 ("expectancy double counts",    "                          avg_loss_pips=self.avg_loss, cost_pips=0.0)", "                          avg_loss_pips=self.avg_loss, cost_pips=1.5)"),
 ("no ambiguity warning",        "        if self.ambiguous_frac > AMBIGUITY_WARN_FRAC:", "        if False:"),
 ("no small sample warning",     "        if 0 < self.n < 30:", "        if False:"),
 ("no slippage warning",         "        if self.slipped_stops:", "        if False:"),
]
