"""News-risk gate — turn a sentiment feature into a vol-regime veto.

The engine already suppresses short-vol when realised vol is accelerating
(models.edge.regime_stressed), but that is BACKWARD-looking. This adds a
FORWARD-looking veto from news: elevated news volume plus strongly negative or
highly-dispersed sentiment flags impending event risk, so the engine can stop
selling vol BEFORE the realised spike the HAR gate only sees after the fact.

It composes with the existing gate WITHOUT changing any signature —
models.edge.compare already takes a `stressed` flag, so a caller passes:

    stressed = regime_stressed(prices) or news_risk_gate(feature)[0]

and everything downstream (verdict, sizing) is unchanged.
"""

from __future__ import annotations

from .sentiment import SentimentFeature


def news_risk_gate(
    feature: SentimentFeature,
    *,
    min_volume: float = 3.0,
    neg_threshold: float = -0.30,
    dispersion_threshold: float = 0.50,
    tail_threshold: float = -0.80,
) -> tuple[bool, str]:
    """Return (elevated_event_risk, reason).

    Fires only when there is ENOUGH news (min_volume) AND it is either broadly
    negative, highly dispersed (disagreement), or carries a severe single
    headline. Requiring volume avoids reacting to one stray story.
    """
    if feature.volume < min_volume:
        return False, f"insufficient news volume ({feature.volume:.1f} < {min_volume})"
    if feature.score <= neg_threshold:
        return True, f"negative news regime (score {feature.score:+.2f})"
    if feature.dispersion >= dispersion_threshold:
        return True, f"high news dispersion ({feature.dispersion:.2f}) — event risk"
    if feature.most_negative <= tail_threshold:
        return True, f"severe headline (most_negative {feature.most_negative:+.2f})"
    return False, "news benign"


def event_risk(price_stressed: bool, feature: SentimentFeature | None = None,
               macro=None, **gate_kwargs) -> tuple[bool, str]:
    """Combine the backward-looking price regime gate with the forward-looking
    news and macro gates. True if ANY fires. Pass this as `stressed` to
    edge.compare. ``macro`` is an optional models.macro.MacroSnapshot."""
    if price_stressed:
        return True, "realised vol accelerating (price regime)"
    if macro is not None:
        from .macro import macro_regime_gate
        fired, reason = macro_regime_gate(macro)
        if fired:
            return True, f"macro: {reason}"
    if feature is not None:
        return news_risk_gate(feature, **gate_kwargs)
    return False, "no elevated regime"
