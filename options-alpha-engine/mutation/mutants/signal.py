"""Ways the scanner could call something tradeable that is not.

This is where P and Q become a trade, and its own docstring names the check
that matters: "we also report edge net of half the bid/ask spread, because you
never trade at mid. An edge that vanishes once you cross the spread is not an
edge -- that single check kills most naive options signals."

So the spread charge gets the most attention here. Dropping it, halving it
again, or adding it instead of subtracting all turn a scanner that refuses most
of a chain into one that finds opportunity everywhere, and the output still
looks like a sorted list of ranked trades either way.

THE TWO-SIDED CONSISTENCY CHECK is the subtler one. A verdict needs the vol
gap AND the dollar edge to agree in sign: implied above forecast must come with
a market price above fair. Drop either half and the scanner will call an option
rich on a vol gap while its own repricing says it is cheap -- a contradiction
that reads as a strong signal because two numbers point at it.

THREE EQUIVALENT MUTANTS ARE DELIBERATELY ABSENT, and measuring them was the
only way to tell.

The two-sided consistency check cannot fire at a sane band. Black-Scholes price
is monotone in vol, so a recovered IV above the forecast comes with a mid above
fair as a matter of arithmetic. Across 570 solvable quotes the vol gap and the
dollar edge disagreed on nine -- every one of them with the market vol EQUAL to
the forecast, a gap of at most 3.2e-4 and an edge of exactly zero. That is the
solver's price tolerance, not a disagreement, and it sits sixty times inside
the default 0.02 band, which calls all nine FAIR before the check is reached.
For the clause to decide anything, `iv_band` would have to be under 3e-4 --
i.e. the scanner would be trading on solver noise. It stays as a statement of
intent; it is not load-bearing.

Substituting the forecast for an unsolvable quote is equivalent for the same
structural reason: `iv = forecast_vol` makes the gap exactly zero, the verdict
FAIR, and the FAIR filter drops it from the output. The `continue` is the
honest spelling of what the filter would do anyway.

`abs(edge_mid)` in the net edge is deliberate and worth stating: the edge of a
SHORT is the premium you keep, so the magnitude is what the spread is charged
against, not the sign. Removing the abs makes every rich option's net edge
negative and silently empties the sell side of the book -- which, for an engine
whose only trade is selling, is the whole output.
"""

TARGET = "engine/signal.py"
TESTS = "tests/test_engine.py"

MUTANTS = [
 # The spread charge: the check the module says kills most naive signals.
 ("spread not charged",         "        edge_net = abs(edge_mid) - 0.5 * q.spread", "        edge_net = abs(edge_mid)"),
 ("spread charged in full",     "        edge_net = abs(edge_mid) - 0.5 * q.spread", "        edge_net = abs(edge_mid) - q.spread"),
 ("spread added to the edge",   "        edge_net = abs(edge_mid) - 0.5 * q.spread", "        edge_net = abs(edge_mid) + 0.5 * q.spread"),
 ("short edge sign kept",       "        edge_net = abs(edge_mid) - 0.5 * q.spread", "        edge_net = edge_mid - 0.5 * q.spread"),

 # The minimum edge and the fair band.
 ("minimum edge not applied",   "    actionable = [m for m in results if m.verdict != \"FAIR\" and m.edge_net >= min_edge_net]", "    actionable = [m for m in results if m.verdict != \"FAIR\"]"),
 ("fair verdicts traded",       "    actionable = [m for m in results if m.verdict != \"FAIR\" and m.edge_net >= min_edge_net]", "    actionable = [m for m in results if m.edge_net >= min_edge_net]"),
 ("minimum edge boundary",      "and m.edge_net >= min_edge_net]", "and m.edge_net > 0]"),
 ("fair band removed",          "    iv_band: float = 0.02,", "    iv_band: float = 0.0,"),
 ("fair band widened",          "    iv_band: float = 0.02,", "    iv_band: float = 1.0,"),
 ("minimum edge removed",       "    min_edge_net: float = 0.05,", "    min_edge_net: float = 0.0,"),

 # The two-sided consistency between the vol gap and the dollar edge.
 ("rich and cheap swapped",     "        if iv - forecast_vol > iv_band and edge_mid < 0:\n            verdict = \"RICH\"", "        if iv - forecast_vol > iv_band and edge_mid < 0:\n            verdict = \"CHEAP\""),
 ("rich dollar test inverted",  "        if iv - forecast_vol > iv_band and edge_mid < 0:", "        if iv - forecast_vol > iv_band and edge_mid > 0:"),
 ("cheap dollar test inverted", "        elif forecast_vol - iv > iv_band and edge_mid > 0:", "        elif forecast_vol - iv > iv_band and edge_mid < 0:"),
 ("vol gap direction flipped",  "        if iv - forecast_vol > iv_band and edge_mid < 0:", "        if forecast_vol - iv > iv_band and edge_mid < 0:"),

 # The edge itself, and the premium it is measured from.
 ("edge sign inverted",         "        edge_mid = fair - q.mid                      # +ve: market too cheap", "        edge_mid = q.mid - fair                      # +ve: market too cheap"),
 ("fair priced at market vol",  "        fair = price(chain.spot, q.strike, q.T, chain.r, chain.q, forecast_vol, q.kind)", "        fair = price(chain.spot, q.strike, q.T, chain.r, chain.q, iv, q.kind)"),
 ("iv premium sign flipped",    "        return self.market_iv - self.forecast_vol", "        return self.forecast_vol - self.market_iv"),

 # Unsolvable quotes must be skipped, not priced on a guessed vol.

 # The ranking. Best-first is what a caller acts on when it takes the top n.
 ("ranking inverted",           "    actionable.sort(key=lambda m: m.edge_net, reverse=True)", "    actionable.sort(key=lambda m: m.edge_net)"),
 ("ranked by gross edge",       "    actionable.sort(key=lambda m: m.edge_net, reverse=True)", "    actionable.sort(key=lambda m: m.edge_mid, reverse=True)"),
 ("not ranked at all",          "    actionable.sort(key=lambda m: m.edge_net, reverse=True)", "    pass"),
]
