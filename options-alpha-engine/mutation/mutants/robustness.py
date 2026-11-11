"""Ways the interval machinery could report more confidence than it has.

This is the highest-leverage target in the tree. Every number this engine
prints is read through an interval from here, and the demo's own conclusion --
"mean trade P&L spans zero: the profit is not distinguishable from luck" -- is
one boolean on one dataclass. A defect here does not make a number wrong; it
makes a WRONG number look supported.

So every mutant below narrows an interval, hides a zero crossing, or breaks the
clustering the block bootstrap exists to preserve:

    * `spans_zero` off by one boundary, so a bound that touches zero reads clean
    * the level inverted, turning a 90% interval into a 10% one
    * blocks of one, which samples trades independently and quietly breaks up
      the very clustering that produces a drawdown -- the module docstring
      names this as the reason `block` exists
    * the sample standard deviation divided by n instead of n-1
    * the returned interval taking the point estimate as its own bound

The module's own history is the argument for this file: on the repo's fixture,
the same strategy with only the seed changed ran a Sharpe of -0.02 to +3.91. A
reader shown 1.85 concludes something a reader shown -0.02 does not.
"""

TARGET = "engine/robustness.py"
TESTS = "tests/test_robustness.py"

MUTANTS = [
 # spans_zero IS the refusal. One flipped boundary and a result whose interval
 # touches zero reads as supported.
 ("zero crossing missed",       "        return self.lo <= 0.0 <= self.hi", "        return self.lo < 0.0 < self.hi"),
 ("everything spans zero",      "        return self.lo <= 0.0 <= self.hi", "        return True"),
 ("nothing spans zero",         "        return self.lo <= 0.0 <= self.hi", "        return False"),
 ("only the low bound checked", "        return self.lo <= 0.0 <= self.hi", "        return self.lo <= 0.0"),
 ("spans-zero flag not printed", '        flag = "   <- spans zero" if self.spans_zero else ""', '        flag = ""'),
 # A return of 0.0049 printed at two decimals reads as +0.00, so a percentage
 # interval would look empty and the reader would conclude the opposite.
 ("percent intervals printed as cash", "        if pct:", "        if False:"),

 # The confidence level. Inverting the tails reports a 10% interval as a 90%
 # one, which is the narrowest possible lie.
 ("level inverted",             "    a = (1.0 - level) / 2.0", "    a = level / 2.0"),
 ("one tail only",              "    return Interval(mid, _quantile(vals, a), _quantile(vals, 1.0 - a),", "    return Interval(mid, _quantile(vals, a), _quantile(vals, 1.0),"),
 ("interval collapsed to point", "    return Interval(mid, _quantile(vals, a), _quantile(vals, 1.0 - a),", "    return Interval(mid, mid, mid,"),
 ("empty samples invent a zero", '        return Interval(float("nan"), float("nan"), float("nan"), 0, label)', "        return Interval(0.0, 0.0, 0.0, 0, label)"),
 ("nans counted as samples",    "    vals = sorted(x for x in samples if x is not None and not math.isnan(x))", "    vals = sorted(x for x in samples if x is not None)"),

 # The quantile itself: interpolation, and the ends.
 ("quantile not interpolated",  "    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac", "    return sorted_vals[lo]"),
 ("quantile interpolation inverted", "    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac", "    return sorted_vals[lo] * frac + sorted_vals[hi] * (1 - frac)"),
 ("quantile index off by one",  "    i = p * (len(sorted_vals) - 1)", "    i = p * len(sorted_vals)"),
 ("quantile runs off the end",  "    hi = min(lo + 1, len(sorted_vals) - 1)", "    hi = lo + 1"),

 # The block structure. `block=1` is independent resampling, which breaks the
 # clustering that produces a drawdown and reports a tighter interval than the
 # data supports -- the defect the parameter exists to prevent.
 ("blocks broken up",           "    blk = max(1, min(int(block), n))", "    blk = 1"),
 ("block longer than the data", "    blk = max(1, min(int(block), n))", "    blk = max(1, int(block))"),
 ("sample truncated away",      "        sample = sample[:n]", "        sample = sample"),
 ("resample not wrapped",       "                sample.append(pnl[(start + j) % n])", "                sample.append(pnl[min(start + j, n - 1)])"),
 ("every resample identical",   "            start = rng.randrange(n)", "            start = 0"),

 # The statistics computed on each resample.
 ("variance divided by n",      "        v = sum((x - m) ** 2 for x in sample) / max(n - 1, 1)", "        v = sum((x - m) ** 2 for x in sample) / n"),
 ("sharpe ignores dispersion",  "        sharpes.append(m / sd if sd > 1e-12 else 0.0)", "        sharpes.append(m)"),
 ("zero sd sharpe inflated",    "        sharpes.append(m / sd if sd > 1e-12 else 0.0)", "        sharpes.append(m / sd if sd > 0 else float('inf'))"),
 ("one trade bootstrapped",     '        raise ValueError("need at least 2 trades to bootstrap")', "        return {}"),
 ("total return not scaled",    "        totals.append(sum(sample) / starting_equity)", "        totals.append(sum(sample))"),
]
