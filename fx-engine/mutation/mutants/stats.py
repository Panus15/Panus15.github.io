"""Ways the sample-size arithmetic could flatter a claim.

The interval is the heart of it: a "Wilson" implementation that quietly
reproduces p +/- 1.96*se has reverted to the thing it was chosen to replace,
and at the sample sizes people actually have that one is wrong in the dangerous
direction.
"""

TARGET = "engine/stats.py"
TESTS = "tests/test_stats.py"

MUTANTS = [
 ("wilson reverts to textbook",  "    denom = 1.0 + z * z / n\n    centre = (p + z * z / (2.0 * n)) / denom\n    half = (z / denom) * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))",
                                 "    denom = 1.0\n    centre = p\n    half = z * math.sqrt(p * (1.0 - p) / n)"),
 ("centre loses its shrinkage",  "    centre = (p + z * z / (2.0 * n)) / denom", "    centre = p / denom"),
 ("denominator dropped",         "    denom = 1.0 + z * z / n", "    denom = 1.0"),
 ("half loses its second term",  "    half = (z / denom) * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))", "    half = (z / denom) * math.sqrt(p * (1.0 - p) / n)"),
 ("lower bound not clamped",     "    return (max(centre - half, 0.0), min(centre + half, 1.0))", "    return (centre - half, min(centre + half, 1.0))"),
 ("upper bound not clamped",     "    return (max(centre - half, 0.0), min(centre + half, 1.0))", "    return (max(centre - half, 0.0), centre + half)"),
 ("z is one-sided",              "    return NormalDist().inv_cdf(1.0 - (1.0 - confidence) / 2.0)", "    return NormalDist().inv_cdf(confidence)"),
 ("default confidence lowered",  "CONFIDENCE = 0.95", "CONFIDENCE = 0.90"),
 ("separation uses upper bound", "        if wilson_interval(round(win_rate * mid), mid, confidence)[0] > break_even:", "        if wilson_interval(round(win_rate * mid), mid, confidence)[1] > break_even:"),
 ("doubling uses upper bound",   "        if wilson_interval(round(win_rate * hi), hi, confidence)[0] > break_even:", "        if wilson_interval(round(win_rate * hi), hi, confidence)[1] > break_even:"),
 ("bisect overshoots by one",    "            hi = mid", "            hi = mid - 1"),
 ("search cap lowered",          "SEARCH_CAP = 10_000_000", "SEARCH_CAP = 100"),
 ("requirement is linear",       "    return (t_required * sd_pips / edge_pips) ** 2", "    return (t_required * sd_pips / edge_pips)"),
 ("edge multiplies not divides", "    return (t_required * sd_pips / edge_pips) ** 2", "    return (t_required * sd_pips * edge_pips) ** 2"),
 ("zero edge divides by zero",   "    if edge_pips <= 0:", "    if edge_pips < 0:"),
 ("detectable edge drops sqrt",  "    return t_required * sd_pips / math.sqrt(n)", "    return t_required * sd_pips / n"),
 ("years multiply not divide",   "    return n / trades_per_year", "    return n * trades_per_year"),
 ("impossible case not caught",  "    if n == math.inf:", "    if False:"),
 ("assumption hidden",           "            f\"at {trades_per_year:,.0f} trades a year, and that assumes the \"\n            f\"trades are independent, which overlapping FX positions are not\")", "            f\"at {trades_per_year:,.0f} trades a year\")"),
 ("report drops the years",      "    yrs = years_needed(n, trades_per_year)", "    yrs = n"),
]
