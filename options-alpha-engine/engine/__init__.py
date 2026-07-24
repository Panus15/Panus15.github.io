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
    deribit      keyless Deribit bridge — real EUROPEAN crypto options, no API key
    tradier      free-key Tradier bridge — SPX/XSP European chain + paper execution
    ibkr         Interactive Brokers (Client Portal) — real paper fills + latency
    portfolio    correlation-aware short-vega caps, CVaR sizing, kill-switch
    hedged_backtest  delta-hedged walk-forward short-vol backtest (governed)
    signal_backtest  multi-date signal-driven walk-forward (does the signal add value?)
    benchmark    same-path comparison: signal vs always-sell (income-ETF) vs buy-hold
    american     binomial American pricing + de-Americanization to a European chain
    stress       overnight-gap / jump stress — the short-gamma risk a smooth backtest hides
"""

from . import (adapters, american, backtest, benchmark, data, deribit,
               hedged_backtest, ibkr, iv, portfolio, pricing, signal,
               signal_backtest, sizing, stress, tradier, volforecast)

__all__ = ["pricing", "iv", "volforecast", "signal", "sizing", "backtest",
           "data", "adapters", "deribit", "tradier", "ibkr", "portfolio",
           "hedged_backtest", "signal_backtest", "benchmark", "american", "stress"]
__version__ = "0.1.0"
