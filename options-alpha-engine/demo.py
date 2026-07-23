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

from datetime import date

from engine import sizing, volforecast
from engine.data import SyntheticAdapter
from engine.hedged_backtest import price_path_with_crash, run_hedged_backtest
from engine.signal import scan_chain
from engine.signal_backtest import synthetic_chain_series
from tools.paper_trade import paper_trade_series
from models.news_signal import event_risk
from models.sentiment import (LexiconSentimentScorer, NewsItem,
                              aggregate_sentiment)
from models import objective
from models.baseline import BaselineDensityForecaster
from models.edge import regime_stressed, scan_distribution
from models.mdn import SequenceMDNForecaster


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

    # 3b. DISTRIBUTIONAL scan — the core ML technique --------------------------
    # Forecast the FULL distribution of S_T (physical P) and compare it to the
    # option-implied risk-neutral distribution (Q) recovered from the chain.
    # This generalises the scalar scan above into a per-expiry P-vs-Q comparison.
    forecaster = BaselineDensityForecaster()
    stressed = regime_stressed(prices)
    print(f"Distributional P-vs-Q scan (core technique)  [regime_stressed={stressed}]")
    print(f"  {'dte':>3} {'P_vol':>6} {'Q_vol':>6} {'VRP':>7} "
          f"{'P_skew':>7} {'Q_skew':>7} {'edge$':>7} {'verdict':>9}")
    for s in scan_distribution(chain, forecaster, prices, actionable_only=False):
        print(f"  {s.expiry_days:>3} {s.p_vol:>6.1%} {s.q_vol:>6.1%} "
              f"{s.variance_risk_premium:>+7.4f} {s.p_skew:>7.2f} {s.q_skew:>7.2f} "
              f"{s.edge_net:>7.1f} {s.verdict:>9}")
    print("  (VRP>0 = market implies more variance than we forecast -> sell vol;\n"
          "   SRP/skew shown but DIAGNOSTIC-only; verdict gated by regime + cost)\n")

    # 3c. News sentiment overlay (Phase 2) — FORWARD-looking vol-regime gate -----
    # The price regime gate is backward-looking (HAR lags a spike). A burst of
    # negative / dispersed news anticipates the vol spike, so it can veto selling
    # vol BEFORE the move. It composes via edge.compare's existing `stressed` flag.
    scorer = LexiconSentimentScorer()
    asof = date(2026, 7, 19)
    calm_news = [NewsItem("2026-07-19", "DEMO", "steady trading, guidance reaffirmed")]
    risk_news = [NewsItem("2026-07-19", "DEMO", "shares plunge on fraud probe"),
                 NewsItem("2026-07-19", "DEMO", "default fears and selloff deepen"),
                 NewsItem("2026-07-19", "DEMO", "crisis warning; volatility surges")]
    for label, news in (("calm news", calm_news), ("risk news", risk_news)):
        feat = aggregate_sentiment(news, scorer, asof=asof)
        fired, reason = event_risk(stressed, feat)
        print(f"News overlay [{label}]: score={feat.score:+.2f} disp={feat.dispersion:.2f} "
              f"vol={feat.volume:.1f} -> event_risk={fired} ({reason})")
    print("  -> when event_risk=True, pass it as edge.compare(stressed=True) to "
          "suppress short-vol early\n")

    # 3d. Promotion gate — would a trained MDN replace the HAR baseline? --------
    # The neural model (a drop-in emitting the SAME MixtureLogNormal) only ships
    # if it beats the baseline OUT-OF-SAMPLE on the S_T density. Train on the
    # first 75% of history, score both on the held-out tail. Lower NLL = better.
    cut = int(len(prices) * 0.75)
    mdn = SequenceMDNForecaster(context=63, seed=0)
    mdn.fit(prices[:cut], horizons=(30,), epochs=15, lr=0.05, max_windows=200)
    oos = objective.build_windows(prices[cut - 63:], context=63, horizons=(30,))
    nll_base = objective.dataset_nll(BaselineDensityForecaster(), oos, r=0.0, q=0.0)
    nll_mdn = objective.dataset_nll(mdn, oos, r=0.0, q=0.0)
    tail_base = sum(objective.left_tail_pinball(BaselineDensityForecaster()
                    .forecast(c, h / 365, spot=c[-1]), s) for c, h, s in oos) / len(oos)
    tail_mdn = sum(objective.left_tail_pinball(mdn.forecast(c, h / 365, spot=c[-1]), s)
                   for c, h, s in oos) / len(oos)
    winner = "MDN" if (nll_mdn < nll_base and tail_mdn < tail_base) else "HAR baseline"
    print(f"Promotion gate (out-of-sample, {len(oos)} windows, lower=better):")
    print(f"  {'':13}{'NLL':>9} {'left-tail':>11}")
    print(f"  HAR baseline {nll_base:>9.4f} {tail_base:>11.4f}")
    print(f"  trained MDN  {nll_mdn:>9.4f} {tail_mdn:>11.4f}")
    print(f"  -> ships: {winner}  (must win BOTH aggregate NLL and the tall tail)\n")

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

    # 5. Delta-hedged WALK-FORWARD backtest — governed + crash-aware ----------
    # The real proof: sell the straddle, delta-hedge daily, size via CVaR, and
    # let the risk governor veto trades. Run it on a calm sample AND one with a
    # crash — the gap between them is why a crash-free backtest lies.
    calm = SyntheticAdapter(seed=5).price_history("CALM", 620)
    crash = price_path_with_crash(756)
    rc = run_hedged_backtest(calm)
    rx = run_hedged_backtest(crash)
    print("Walk-forward delta-hedged short-vol (CVaR-sized, governor-gated):")
    print(f"  {'':16}{'Sharpe':>7} {'Sortino':>8} {'MaxDD':>7} {'trades':>7} {'skipped':>8}")
    print(f"  calm sample    {rc.metrics.sharpe:>7.2f} {rc.metrics.sortino:>8.2f} "
          f"{rc.metrics.max_drawdown:>7.1%} {rc.n_trades:>7} {rc.n_skipped:>8}")
    print(f"  WITH a crash   {rx.metrics.sharpe:>7.2f} {rx.metrics.sortino:>8.2f} "
          f"{rx.metrics.max_drawdown:>7.1%} {rx.n_trades:>7} {rx.n_skipped:>8}")
    print(f"  crash skips: {rx.skip_reasons}")
    print("  -> a crash-free sample looks like free money; the crash reveals the")
    print("     short-vol tail and the regime gate/kill-switch fire. THAT is honest.\n")

    # 6. Forward-test PAPER LEDGER — the accumulating out-of-sample scoreboard --
    # A backtest replays ONE strategy over history; this is the instrument you run
    # LIVE going forward: at each date freeze the P-forecast + the market Q, then
    # grade both against what actually happened. Two verdicts, kept separate:
    #   CALIBRATION — did our physical density beat the market-implied one OOS?
    #   PROFIT      — did the signal-fired trades make money, net of costs?
    # On synthetic data it (correctly) refuses to claim edge — that honesty is the
    # point. Feed it real recorded snapshots and it builds your true track record.
    ledger = paper_trade_series(price_path_with_crash(900),
                                synthetic_chain_series(dte=21),
                                BaselineDensityForecaster(), dte=21, warmup=63)
    print("Forward-test paper ledger (record -> settle -> proper-score):")
    print("  " + ledger.report().summary().replace("\n", "\n  ") + "\n")

    print("Reminder: swap SyntheticAdapter for real data before trusting any "
          "number. This fixture only proves the engine + risk discipline work.")


if __name__ == "__main__":
    main()
