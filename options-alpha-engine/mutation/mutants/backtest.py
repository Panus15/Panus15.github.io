"""Ways the metrics a human reads could flatter a strategy that does not work.

engine/backtest.py had NO test file at all -- 678 tests passed around it, the
same hole engine/sizing.py was in -- while computing the Sharpe, Sortino,
drawdown, CAGR and hit rate that decide whether anything here gets traded.
engine/benchmark.py prints them.

Its own docstring states the stake: "Most options backtests lie. They fill at
mid, ignore commissions, and assume infinite liquidity." The mutants are
grouped by the lie each one restores.

  1. THE DIVISORS. A population standard deviation where a sample one belongs
     inflates the Sharpe by sqrt(n/(n-1)); normalising the downside deviation by
     the count of DOWN days instead of all days inflates the Sortino by the
     square root of the same ratio. Both flatter the strategy, both leave a
     plausible number, and on a long series neither is large enough to look
     wrong -- which is exactly why they need an arithmetic oracle rather than a
     sanity bound.
  2. ANNUALISATION. sqrt(252) on the Sharpe and the 1/years exponent on the
     CAGR are where a daily number becomes a headline. Multiplying by 252
     instead of its root, or raising to `years` instead of `1/years`, produces
     numbers of the right sign and the wrong magnitude.
  3. RUIN. The one trade this engine measures is SELLING options, whose loss is
     unbounded, so an equity curve reaching zero is the case that matters most
     and was the case that broke: equity passing exactly through zero raised
     ZeroDivisionError, and an account ending below zero gave a COMPLEX CAGR
     that made `summary()` raise. Letting post-ruin P&L through reports a
     survivor's Sharpe for an account that did not survive.
  4. THE FILL. Half the spread each way, plus slippage, plus commission per
     contract. Charging the full spread, or none, or adding it instead of
     subtracting, is the difference between a tradeable curve and a brochure.

THE SIDE WAS VALIDATED RATHER THAN DEFAULTED while writing these, and the reason
is worth keeping: `if side == "buy"` filled every other value -- "Buy", "BUY",
"sel", "" and None -- as a SELL, a full spread away and in the wrong direction.
`cost_vs_mid` takes an absolute value, so the FRICTION was right either way and
no test of the cost could have found it.

A CAGR OF -100% IS NOT A RUIN FLAG. One of the tests here originally assumed it
was: an 80% loss over three days annualises to 0.201^84 - 1, about 2.5e-59 - 1,
which IS -1.0 in floating point, on an account that survived with a fifth of its
capital. `ruined` has to be its own field because no metric can stand in for it,
and that is now asserted rather than assumed.
"""

TARGET = "engine/backtest.py"
TESTS = "tests/test_backtest.py"

MUTANTS = [
 # The equity curve and the returns taken off it.
 ("pnl subtracted from equity",  "        equity.append(equity[-1] + pnl)", "        equity.append(equity[-1] - pnl)"),
 ("equity not compounded",       "        equity.append(equity[-1] + pnl)", "        equity.append(equity[0] + pnl)"),
 ("returns inverted",            "    rets = [equity[i] / equity[i - 1] - 1.0 for i in range(1, len(equity))]", "    rets = [equity[i - 1] / equity[i] - 1.0 for i in range(1, len(equity))]"),
 ("returns are differences",     "    rets = [equity[i] / equity[i - 1] - 1.0 for i in range(1, len(equity))]", "    rets = [equity[i] - equity[i - 1] for i in range(1, len(equity))]"),
 ("returns not centred on one",  "    rets = [equity[i] / equity[i - 1] - 1.0 for i in range(1, len(equity))]", "    rets = [equity[i] / equity[i - 1] for i in range(1, len(equity))]"),
 ("first return dropped",        "for i in range(1, len(equity))]", "for i in range(2, len(equity))]"),

 # 1. The divisors.
 ("mean not averaged",           "    mean = sum(rets) / len(rets)", "    mean = sum(rets)"),
 ("variance uses population n",  "    var = sum((x - mean) ** 2 for x in rets) / max(len(rets) - 1, 1)", "    var = sum((x - mean) ** 2 for x in rets) / len(rets)"),
 ("variance not centred",        "    var = sum((x - mean) ** 2 for x in rets) / max(len(rets) - 1, 1)", "    var = sum(x ** 2 for x in rets) / max(len(rets) - 1, 1)"),
 ("variance not rooted",         "    std = math.sqrt(var)", "    std = var"),
 ("downside is upside",          "    downside = [min(x, 0.0) for x in rets]", "    downside = [max(x, 0.0) for x in rets]"),
 ("downside unclipped",          "    downside = [min(x, 0.0) for x in rets]", "    downside = list(rets)"),
 ("downside divided by down days", "    dvar = sum(x * x for x in downside) / max(len(downside), 1)", "    dvar = sum(x * x for x in downside) / max(sum(1 for x in downside if x < 0), 1)"),
 ("downside deviation not rooted", "    dstd = math.sqrt(dvar)", "    dstd = dvar"),

 # 2. Annualisation.
 ("sharpe not annualised",       "    sharpe = (mean / std * math.sqrt(TRADING_DAYS)) if std > 0 else 0.0", "    sharpe = (mean / std) if std > 0 else 0.0"),
 ("sharpe annualised by days",   "    sharpe = (mean / std * math.sqrt(TRADING_DAYS)) if std > 0 else 0.0", "    sharpe = (mean / std * TRADING_DAYS) if std > 0 else 0.0"),
 ("sharpe zero divisor unguarded", "    sharpe = (mean / std * math.sqrt(TRADING_DAYS)) if std > 0 else 0.0", "    sharpe = (mean / std * math.sqrt(TRADING_DAYS)) if std >= 0 else 0.0"),
 ("sortino not annualised",      "    sortino = (mean / dstd * math.sqrt(TRADING_DAYS)) if dstd > 0 else 0.0", "    sortino = (mean / dstd) if dstd > 0 else 0.0"),
 ("sortino zero divisor unguarded", "    sortino = (mean / dstd * math.sqrt(TRADING_DAYS)) if dstd > 0 else 0.0", "    sortino = (mean / dstd * math.sqrt(TRADING_DAYS)) if dstd >= 0 else 0.0"),
 ("trading year is a calendar year", "TRADING_DAYS = 252", "TRADING_DAYS = 365"),

 # Total return and CAGR.
 ("total return inverted",       "    total_return = equity[-1] / equity[0] - 1.0", "    total_return = equity[0] / equity[-1] - 1.0"),
 ("total return is a ratio",     "    total_return = equity[-1] / equity[0] - 1.0", "    total_return = equity[-1] / equity[0]"),
 ("years not converted",         "    years = len(rets) / TRADING_DAYS", "    years = len(rets)"),
 ("years multiplied not divided", "    years = len(rets) / TRADING_DAYS", "    years = len(rets) * TRADING_DAYS"),
 ("cagr exponent inverted",      "        cagr = (equity[-1] / equity[0]) ** (1 / years) - 1.0 if years > 0 else 0.0", "        cagr = (equity[-1] / equity[0]) ** years - 1.0 if years > 0 else 0.0"),
 ("cagr is the period return",   "        cagr = (equity[-1] / equity[0]) ** (1 / years) - 1.0 if years > 0 else 0.0", "        cagr = (equity[-1] / equity[0]) - 1.0 if years > 0 else 0.0"),

 # 3. Ruin.
 ("ruin not detected",           "        if equity[-1] <= 0.0:\n            ruin_day = day\n            break", "        if False:\n            ruin_day = day\n            break"),
 ("ruin at zero not counted",    "        if equity[-1] <= 0.0:", "        if equity[-1] < 0.0:"),
 ("ruin does not stop the curve", "        if equity[-1] <= 0.0:\n            ruin_day = day\n            break", "        if equity[-1] <= 0.0:\n            ruin_day = day"),
 ("ruin fires on any loss",      "        if equity[-1] <= 0.0:", "        if equity[-1] <= equity[0]:"),
 ("ruin not reported",           "        ruined=ruin_day is not None,", "        ruined=False,"),
 ("ruin day not reported",       "        ruin_day=ruin_day,", "        ruin_day=None,"),
 ("cagr of a dead account computed", "    if equity[-1] <= 0.0:\n        cagr = -1.0\n    else:", "    if False:\n        cagr = -1.0\n    else:"),
 ("cagr ruin floor inverted",    "    if equity[-1] <= 0.0:\n        cagr = -1.0", "    if equity[-1] > 0.0:\n        cagr = -1.0"),
 ("ruin line not printed",       "        if self.ruined:\n            lines.insert(0, f\"RUINED       : equity hit zero on day {self.ruin_day}\")", "        if False:\n            lines.insert(0, f\"RUINED       : equity hit zero on day {self.ruin_day}\")"),

 # Hit rate.
 ("flat days counted as wins",   "    hit_rate = sum(1 for x in rets if x > 0) / len(rets)", "    hit_rate = sum(1 for x in rets if x >= 0) / len(rets)"),
 ("hit rate counts losses",      "    hit_rate = sum(1 for x in rets if x > 0) / len(rets)", "    hit_rate = sum(1 for x in rets if x < 0) / len(rets)"),
 ("hit rate not a fraction",     "    hit_rate = sum(1 for x in rets if x > 0) / len(rets)", "    hit_rate = sum(1 for x in rets if x > 0)"),

 # Drawdown.
 ("drawdown tracks the trough",  "        peak = max(peak, v)", "        peak = min(peak, v)"),
 ("drawdown keeps the best",     "        mdd = min(mdd, v / peak - 1.0)", "        mdd = max(mdd, v / peak - 1.0)"),
 ("drawdown ratio inverted",     "        mdd = min(mdd, v / peak - 1.0)", "        mdd = min(mdd, peak / v - 1.0)"),
 ("drawdown measured from start", "        peak = max(peak, v)\n        mdd = min(mdd, v / peak - 1.0)", "        mdd = min(mdd, v / equity[0] - 1.0)"),
 ("drawdown capital guard removed", "    if equity[0] <= 0.0:\n        raise ValueError(f\"equity curve must start above zero, got {equity[0]!r}\")", "    if False:\n        raise ValueError(f\"equity curve must start above zero, got {equity[0]!r}\")"),
 ("drawdown empty guard removed", "    if not equity:\n        raise ValueError(\"no equity curve supplied\")", "    if False:\n        raise ValueError(\"no equity curve supplied\")"),

 # Input validation.
 ("empty pnl accepted",          "    if not daily_pnl:\n        raise ValueError(\"no P&L supplied\")", "    if False:\n        raise ValueError(\"no P&L supplied\")"),
 ("zero capital accepted",       "    if starting_equity <= 0:\n        raise ValueError(f\"starting_equity must be positive, got {starting_equity!r}\")", "    if False:\n        raise ValueError(f\"starting_equity must be positive, got {starting_equity!r}\")"),

 # 4. The fill.
 ("fill side unvalidated",       "        if side not in (\"buy\", \"sell\"):\n            raise ValueError(f\"side must be 'buy' or 'sell', got {self.side!r}\")", "        if False:\n            raise ValueError(f\"side must be 'buy' or 'sell', got {self.side!r}\")"),
 ("fill side not normalised",    "        side = self.side.lower() if isinstance(self.side, str) else self.side", "        side = self.side"),
 ("fill charges the full spread", "        half = 0.5 * self.spread", "        half = self.spread"),
 ("fill charges no spread",      "        half = 0.5 * self.spread", "        half = 0.0"),
 ("fill slippage ignored",       "        slip = self.slippage_frac * self.intended_price", "        slip = 0.0"),
 ("fill slippage not scaled",    "        slip = self.slippage_frac * self.intended_price", "        slip = self.slippage_frac"),
 ("fill sides swapped",          "        if side == \"buy\":\n            return self.intended_price + half + slip\n        return self.intended_price - half - slip", "        if side == \"buy\":\n            return self.intended_price - half - slip\n        return self.intended_price + half + slip"),
 ("fill buy pays no spread",     "            return self.intended_price + half + slip", "            return self.intended_price - half + slip"),
 ("fill cost unsigned",          "        per_contract = abs(self.executed_price - self.intended_price) * multiplier", "        per_contract = (self.executed_price - self.intended_price) * multiplier"),
 ("fill cost ignores multiplier", "        per_contract = abs(self.executed_price - self.intended_price) * multiplier", "        per_contract = abs(self.executed_price - self.intended_price)"),
 ("fill cost drops commission",  "        return contracts * (per_contract + self.commission)", "        return contracts * per_contract"),
 ("fill cost ignores size",      "        return contracts * (per_contract + self.commission)", "        return per_contract + self.commission"),
 ("fill commission free by default", "    commission: float = 0.65   # per contract, typical retail", "    commission: float = 0.0   # per contract, typical retail"),
]
