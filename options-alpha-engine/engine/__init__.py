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
    deribit      keyless Deribit bridge — real EUROPEAN options, no API key
    portfolio    correlation-aware short-vega caps, CVaR sizing, kill-switch
    hedged_backtest  delta-hedged walk-forward short-vol backtest (governed)
"""

from . import (adapters, backtest, data, deribit, hedged_backtest, iv,
               portfolio, pricing, signal, sizing, volforecast)

__all__ = ["pricing", "iv", "volforecast", "signal", "sizing", "backtest",
           "data", "adapters", "deribit", "portfolio", "hedged_backtest"]
__version__ = "0.1.0"
