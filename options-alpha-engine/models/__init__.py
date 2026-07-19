"""Core ML layer.

The engine's core technique is *probabilistic (distributional) forecasting*:
predict the full distribution of the terminal price S_T and compare it to the
option-implied (risk-neutral) distribution. The gap is the tradeable edge.

    density   MixtureLogNormal — the forecast OUTPUT TYPE (baseline AND MDN emit it)
    baseline  parametric distributional forecaster (HAR-RV vol -> mixture) [ships first]
    mdn       trainable Mixture Density Network forecaster — drop-in for baseline
    rnd       risk-neutral moments recovered from the option chain (BKM / model-free)
    edge      compares physical P forecast vs risk-neutral Q -> distributional signals
    sentiment Phase-2 NLP: news -> sentiment feature (lexicon core, FinBERT optional)
    news_signal  forward-looking vol-regime gate from sentiment (feeds edge.stressed)

Production target: swap `baseline`'s parameter function for a TFT-encoder + MDN
head that emits the SAME MixtureLogNormal. Everything else stays put.
"""

from . import (baseline, density, edge, mdn, news_signal, objective, rnd,
               sentiment)

__all__ = ["density", "baseline", "mdn", "rnd", "edge", "objective",
           "sentiment", "news_signal"]
