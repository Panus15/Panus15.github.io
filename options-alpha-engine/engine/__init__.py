"""Options alpha engine — MVP core.

Public surface:
    pricing      Black-Scholes-Merton price + Greeks
    iv           implied-volatility solver
    volforecast  realised-vol estimators (close-to-close, EWMA, HAR-RV)
    signal       mispricing scanner (implied vs forecast vol, net of spread)
    sizing       fractional-Kelly position sizing with a risk cap
    backtest     honesty-first backtest engine + risk metrics
    data         OptionQuote/OptionChain model + synthetic adapter
"""

from . import backtest, data, iv, pricing, signal, sizing, volforecast

__all__ = ["pricing", "iv", "volforecast", "signal", "sizing", "backtest", "data"]
__version__ = "0.1.0"
