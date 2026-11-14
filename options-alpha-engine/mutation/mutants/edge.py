"""Ways the final comparison could turn a known trap back into a signal.

models/edge.py is where P and Q become the engine's claim, and it lists four
"honesty commitments baked in after adversarial review -- the difference
between a signal and a trap". Each one exists because the naive version loses
money in a specific, documented way, so each gets mutants that restore the
naive version:

  1. VRP IS POSITIVE IN EQUILIBRIUM. Selling whenever VRP > 0 harvests the
     non-arbitrageable premium as carry and calls it mispricing. So with a
     history the Z-SCORE is traded, not the level -- and a mutant that prefers
     the level when both are available undoes the whole commitment.
  2. HAR-RV LAGS VOL SPIKES, so the naive short-vol signal is loudest right
     into a developing crash. The regime gate suppresses SELL when realised
     vol is accelerating. Disabling it, or inverting the comparison so it
     suppresses in CALM markets, is the blow-up trap restored.
  3. SRP IS DIAGNOSTIC ONLY. Nothing may size on skew.
  4. AN EDGE THAT DIES CROSSING COSTS IS DROPPED. The last gate in `compare`
     demotes RICH and CHEAP to FAIR when the net edge is not positive.

The gates are ordered and the order is load-bearing: coverage first, then
regime, then the z-score, then cost. A coverage failure must not be overridden
by a strong z-score, which is what moving the checks around does.
"""

TARGET = "models/edge.py"
TESTS = "tests/test_rnd_edge.py"

MUTANTS = [
 # Commitment 1: trade the z-score, not the level.
 ("level preferred over z score", "    elif z is not None:\n        basis = \"vrp_zscore\"", "    elif False:\n        basis = \"vrp_zscore\""),
 ("z score threshold removed",   "    z_thresh: float = 1.0,", "    z_thresh: float = 0.0,"),
 ("z score sign flipped",        "        verdict = \"RICH\" if z >= z_thresh else \"CHEAP\" if z <= -z_thresh else \"FAIR\"", "        verdict = \"CHEAP\" if z >= z_thresh else \"RICH\" if z <= -z_thresh else \"FAIR\""),
 ("z computed on population sd", "        var = sum((x - m) ** 2 for x in vrp_history) / (len(vrp_history) - 1)", "        var = sum((x - m) ** 2 for x in vrp_history) / len(vrp_history)"),
 ("z not centred",               "        z = (vrp - m) / sd if sd > 1e-6 else None", "        z = vrp / sd if sd > 1e-6 else None"),
 ("z on a degenerate history",   "        z = (vrp - m) / sd if sd > 1e-6 else None", "        z = (vrp - m) / sd if sd > 0 else None"),
 ("short history accepted",      "    if vrp_history and len(vrp_history) >= 20:", "    if vrp_history:"),

 # Commitment 2: the regime gate, which exists because HAR-RV lags spikes.
 ("regime gate disabled",        "    elif stressed and vrp > 0:", "    elif False:"),
 ("regime gate suppresses calm", "    return l > 0 and (s / l) > thresh", "    return l > 0 and (s / l) < thresh"),
 ("regime gate blocks buying too", "    elif stressed and vrp > 0:", "    elif stressed:"),
 ("regime threshold removed",    "def regime_stressed(prices, short: int = 5, long: int = 21, thresh: float = 1.3) -> bool:", "def regime_stressed(prices, short: int = 5, long: int = 21, thresh: float = 99.0) -> bool:"),
 ("regime windows swapped",      "        s = volforecast.close_to_close_vol(prices, short)\n        l = volforecast.close_to_close_vol(prices, long)", "        s = volforecast.close_to_close_vol(prices, long)\n        l = volforecast.close_to_close_vol(prices, short)"),
 ("regime errors read as stress", "    except ValueError:\n        return False", "    except ValueError:\n        return True"),

 # Commitment 4: the cost gate, which must be able to veto a strong signal.
 ("cost gate disabled",          "    if verdict in (\"RICH\", \"CHEAP\") and edge_net <= 0:", "    if False:"),
 ("cost gate boundary loosened", "    if verdict in (\"RICH\", \"CHEAP\") and edge_net <= 0:", "    if verdict in (\"RICH\", \"CHEAP\") and edge_net < 0:"),
 ("cost gate inverted",          "    if verdict in (\"RICH\", \"CHEAP\") and edge_net <= 0:", "    if verdict in (\"RICH\", \"CHEAP\") and edge_net > 0:"),

 # The coverage gate, which must come first and must not be overridable.
 ("coverage gate disabled",      "    if not q.coverage_ok:", "    if False:"),
 ("coverage gate inverted",      "    if not q.coverage_ok:", "    if q.coverage_ok:"),

 # The premia themselves.
 ("vrp compares vols not variances", "    vrp = q.vol ** 2 - p_vol ** 2", "    vrp = q.vol - p_vol"),
 ("vrp sign flipped",            "    vrp = q.vol ** 2 - p_vol ** 2", "    vrp = p_vol ** 2 - q.vol ** 2"),
 ("srp sign flipped",            "    srp = q.skew - p_skew", "    srp = p_skew - q.skew"),
 ("vol premium sign flipped",    "        vol_premium=q.vol - p_vol, vrp_z=z,", "        vol_premium=p_vol - q.vol, vrp_z=z,"),
 ("carry threshold removed",     "    min_vrp: float = 1e-4,", "    min_vrp: float = 0.0,"),
 ("carry verdict sign flipped",  "        verdict = \"RICH\" if vrp > min_vrp else \"CHEAP\" if vrp < -min_vrp else \"FAIR\"", "        verdict = \"CHEAP\" if vrp > min_vrp else \"RICH\" if vrp < -min_vrp else \"FAIR\""),
]
