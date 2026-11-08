"""Ways the self-assessment could round in the engine's own favour.

A tool that measures its own accuracy fails by flattering itself, and two of
these were live defects found by reading the output rather than by a test.
"""

TARGET = "tools/calibrate.py"
TESTS = "tests/test_calibrate.py"

MUTANTS = [
 ("zero passes scores perfect",  "    if noise.in_sample_passes == 0:", "    if False:"),
 ("removal credit never shown",  "    if noise.in_sample_passes == 0:", "    if True:"),
 ("survival inverted",           "        return (self.confirmed / self.in_sample_passes\n                if self.in_sample_passes else 0.0)", "        return (1.0 - self.confirmed / self.in_sample_passes\n                if self.in_sample_passes else 0.0)"),
 ("noise/hutchinson link cut",   "    L.append(\"  The NOISE row is the one that belongs beside Hutchinson et al.\")", "    L.append(\"  Decay:\")"),
 ("trend comparison not warned", "    L.append(\"  The TREND row must NOT be compared to it. There the edge is \"", "    L.append(\"  \"  # "),
 ("published decay dropped",     "    L.append(\"  (2022), who measured 0.06/0.66 = 0.09 on currency technical \"", "    L.append(\"  (2022) also looked at currency technical \""),
 ("no-decay case not handled",   "                 + (\"no in-sample passes to measure\" if d != d else f\"{d:+.2f}\"))", "                 + f\"{d:+.2f}\")"),
 ("upper bound not stated",      "    L.append(f\"THE POWER FIGURE IS AN UPPER BOUND. The trend condition moves \"", "    L.append(f\"POWER. The trend condition moves \""),
 ("price move hidden",           "             f\"the price x{trend.move:.2f}\")", "             f\"the price\")"),
 ("cannot-say caveat dropped",   "    L.append(\"  how much lower, because FX does not supply one that strong to \"", "    L.append(\"  \"  # "),
 ("per-series rate is per-rule", "        if any(r.confirmed for r in h.results):\n            c.trials_with_confirm += 1", "        if False:\n            c.trials_with_confirm += 1"),
 ("found rate divides by rules", "        return self.trials_with_confirm / self.trials if self.trials else 0.0", "        return self.trials_with_confirm / self.rules if self.rules else 0.0"),
 ("confirm counted on refusal",  "            if r.confirmed:\n                c.confirmed += 1", "            if not r.confirmed:\n                c.confirmed += 1"),
 ("in-sample passes not counted","            c.in_sample_passes += 1", "            pass"),
 ("decay ratio inverted",        "                c.decays.append(b / a)", "                c.decays.append(a / b)"),
 ("trend drift removed",         "TREND_DRIFT = 0.00004", "TREND_DRIFT = 0.0"),
 ("drift not applied",           "            px *= math.exp(rng.gauss(drift / substeps, s))", "            px *= math.exp(rng.gauss(0.0, s))"),
 ("alpha not corrected",         "NOMINAL_ALPHA = 0.05 / (len(DETECTORS) * PARAMETER_COUNT)", "NOMINAL_ALPHA = 0.05"),
 ("benchmarks dropped",          "    for b in BENCHMARKS:\n        L.append(f\"  · {b.label}\")", "    for b in []:\n        L.append(f\"  · {b.label}\")"),
 ("benchmark sources dropped",   "        L.append(f\"      {b.source}\")", "        L.append(\"      (source withheld)\")"),
 ("sample-size cost dropped",    "    L.append(f\"AND THE COST OF FINDING OUT. Separating a 55% win rate from the \"", "    L.append(f\"\"  # "),
 ("tiny samples allowed",        "    if a.trials < 1 or a.bars < 1000:", "    if False:"),
]
