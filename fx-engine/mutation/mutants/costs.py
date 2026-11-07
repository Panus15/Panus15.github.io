"""Ways the cost and expectancy arithmetic could understate what a trade pays.

Both files are swept together: expectancy is the formula and costs is the input,
and every failure mode in the research is one of them being wrong by a factor
nobody checked.
"""

TARGET = "engine/costs.py"
TESTS = "tests/test_expectancy.py"

MUTANTS = [
 ("JPY pip size wrong",         '    return 0.01 if pair.upper().endswith("JPY") else 0.0001', '    return 0.0001'),
 ("pip size case sensitive",    '    return 0.01 if pair.upper().endswith("JPY") else 0.0001', '    return 0.01 if pair.endswith("JPY") else 0.0001'),
 ("bp per pip inverted",        "    return pip_size(pair) / price * 10_000.0", "    return price / pip_size(pair) * 10_000.0"),
 ("bp per pip ignores price",   "    return pip_size(pair) / price * 10_000.0", "    return pip_size(pair) * 10_000.0"),
 ("negative price allowed",     "    if price <= 0:", "    if False:"),
 ("markup not charged",         "        net_annual = self.swap_markup_annual - self.carry_annual", "        net_annual = -self.carry_annual"),
 ("carry not netted",           "        net_annual = self.swap_markup_annual - self.carry_annual", "        net_annual = self.swap_markup_annual"),
 ("carry sign flipped",         "        net_annual = self.swap_markup_annual - self.carry_annual", "        net_annual = self.swap_markup_annual + self.carry_annual"),
 ("financing uses trading days","FINANCING_DAYS = 365.0", "FINANCING_DAYS = 252.0"),
 ("round turn not in the total","        return self.round_turn_pips + self.financing_pips(nights, price)", "        return self.financing_pips(nights, price)"),
 ("financing not in the total", "        return self.round_turn_pips + self.financing_pips(nights, price)", "        return self.round_turn_pips"),
 ("negative nights allowed",    "        if nights < 0:", "        if False:"),
 ("friction ignores leverage",  "        return per_trade_bp / 10_000.0 * trades_per_year * leverage", "        return per_trade_bp / 10_000.0 * trades_per_year"),
 ("friction ignores frequency", "        return per_trade_bp / 10_000.0 * trades_per_year * leverage", "        return per_trade_bp / 10_000.0 * leverage"),
 ("gross notional ignored",     "        return self.swap_markup_annual * gross_notional", "        return self.swap_markup_annual"),
 ("negative round turn allowed","        if self.round_turn_pips < 0:", "        if False:"),
 ("negative markup allowed",    "        if self.swap_markup_annual < 0:", "        if False:"),
 ("gap risk note dropped",      '        return ("a stop does NOT bound the loss: on 2015-01-15 EUR/CHF moved >30% "', '        return (""  # '),
]
