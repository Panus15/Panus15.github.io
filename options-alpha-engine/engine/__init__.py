"""Options alpha engine — MVP core.

Public surface:
    pricing      Black-Scholes-Merton price + Greeks
    iv           implied-volatility solver
    volforecast  realised-vol estimators (close-to-close, EWMA, HAR-RV)
    signal       mispricing scanner (implied vs forecast vol, net of spread)
    sizing       fractional-Kelly position sizing with a risk cap
    backtest     honesty-first backtest engine + risk metrics
    data         OptionQuote/OptionChain model + synthetic adapter
    adapters     live (Polygon/ORATS) + offline (CSV/JSON) market-data adapters
    portfolio    correlation-aware short-vega caps, CVaR sizing, kill-switch
"""

from . import (adapters, backtest, data, iv, portfolio, pricing, signal,
               sizing, volforecast)

__all__ = ["pricing", "iv", "volforecast", "signal", "sizing", "backtest",
           "data", "adapters", "portfolio"]
__version__ = "0.1.0"
