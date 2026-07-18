"""End-to-end demo — runs with zero dependencies and zero data vendor.

    python3 demo.py

It walks the whole MVP loop:
  1. Pull price history + an option chain (synthetic, deterministic).
  2. Forecast realised vol three ways (close-to-close, EWMA, HAR-RV).
  3. Scan the chain for options that are rich/cheap vs the forecast,
     reporting edge AFTER crossing the spread.
  4. Size a trade with fractional Kelly + a 2% risk cap.
  5. Run a cost-aware backtest of a toy short-vol strategy and print
     the metrics that decide whether an edge is real.
"""

from engine import backtest, sizing, volforecast
from engine.data import SyntheticAdapter
from engine.signal import scan_chain


def main() -> None:
    md = SyntheticAdapter()
    symbol = "DEMO"

    # 1. Data ----------------------------------------------------------------
    prices = md.price_history(symbol, days=260)
    chain = md.option_chain(symbol, spot=prices[-1])
    print(f"Underlying {symbol}: spot={chain.spot:.2f}  "
          f"quotes={len(chain.quotes)}  r={chain.r:.2%}\n")

    # 2. Volatility forecast -------------------------------------------------
    ctc = volforecast.close_to_close_vol(prices)
    ewma = volforecast.ewma_vol(prices)
    har = volforecast.har_rv_forecast(prices)
    print("Realised-vol forecasts (annualised):")
    print(f"  close-to-close (21d) : {ctc:.1%}")
    print(f"  EWMA (lambda=0.94)   : {ewma:.1%}")
    print(f"  HAR-RV               : {har:.1%}   <- used as fair vol\n")

    # 3. Mispricing scan -----------------------------------------------------
    ideas = scan_chain(chain, forecast_vol=har, min_edge_net=0.05)
    print(f"Actionable mispricings (net of half-spread): {len(ideas)}")
    print(f"  {'dte':>3} {'strike':>7} {'kind':>4} {'IV':>6} {'fairVol':>7} "
          f"{'mid':>6} {'fair':>6} {'edge$':>6} {'verdict':>7}")
    for m in ideas[:8]:
        q = m.quote
        print(f"  {q.expiry_days:>3} {q.strike:>7.0f} {q.kind:>4} "
              f"{m.market_iv:>6.1%} {m.forecast_vol:>7.1%} "
              f"{q.mid:>6.2f} {m.fair_value:>6.2f} "
              f"{m.edge_net:>6.2f} {m.verdict:>7}")
    print()

    # 4. Position sizing -----------------------------------------------------
    if ideas:
        best = ideas[0]
        n = sizing.position_size(
            account_equity=100_000,
            contract_price=best.quote.mid,
            win_prob=0.58,          # placeholder; comes from your model in prod
            payoff_odds=1.0,
            kelly_scale=0.25,
            max_risk_frac=0.02,
        )
        print(f"Top idea: {best.verdict} {best.quote.kind} "
              f"{best.quote.strike:.0f} / {best.quote.expiry_days}d")
        print(f"Sized position (0.25 Kelly, 2% cap): {n} contracts\n")

    # 5. Cost-aware backtest of a toy short-vol strategy ---------------------
    # Illustrative daily P&L: collect the variance risk premium, minus the
    # occasional volatility spike. The point is the *engine + metrics*, not
    # this fixture's returns.
    daily_pnl = _toy_short_vol_pnl(prices)
    result = backtest.run_backtest(daily_pnl, starting_equity=100_000)
    print("Backtest (toy short-vol, costs included):")
    print("  " + result.summary().replace("\n", "\n  "))
    print("\nReminder: swap SyntheticAdapter for real data before trusting any "
          "of these numbers. This fixture only proves the plumbing works.")


def _toy_short_vol_pnl(prices):
    """Deterministic illustrative P&L stream derived from the price path.

    Short-vol harvests small daily premium (theta) but pays up on large moves.
    We inject periodic deterministic vol spikes so the series shows the true
    signature: a high hit rate punctuated by painful losing days and real
    drawdowns — "picking up pennies in front of a steamroller." A backtest
    that hides this is lying to you.
    """
    rets = volforecast.log_returns(prices)
    pnl = []
    premium_per_day = 40.0            # theta collected each day
    gamma_cost = 85_000.0            # penalty scaling for realised variance
    for i, x in enumerate(rets):
        move = abs(x)
        if i % 21 == 0:              # ~monthly vol spike / gap risk
            move *= 5.0
        pnl.append(premium_per_day - gamma_cost * move * move)
    return pnl


if __name__ == "__main__":
    main()
