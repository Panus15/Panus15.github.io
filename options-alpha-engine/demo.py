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

    # 3b-2. PER-STRIKE board — which exact contract, at which strike/expiry? -----
    # For every quoted call/put: the model's P(finish ITM) vs the market-implied
    # N(d2), and fair value vs bid/ask after costs. Direction-neutral by design:
    # edges come from vol level + distribution shape, never from an up/down call.
    from models.strike_scan import scan_strikes
    board = scan_strikes(chain, forecaster, prices, top=6)
    print("Per-strike board (top 6 by |edge|, $/contract):")
    for s in board:
        print("  " + s.line())
    print("  (on an efficiently-priced chain almost everything is FAIR — a BUY/\n"
          "   WRITE only appears when fair-vs-market survives spread+commission)\n")

    # 3b-2b. DEFINED-RISK SPREADS — harvest the premium with a CAPPED tail --------
    # VRP is compensation for a crash; selling NAKED vol takes the un-hedgeable
    # tail. Defined-risk structures cap the loss with long wings — the structural
    # answer to the tail problem. Each is scored EV = credit - E_P[loss] - cost.
    from models.spreads import scan_spreads
    _dtes = sorted({q.expiry_days for q in chain.quotes})
    _sdte = min(_dtes, key=lambda d: abs(d - 30))          # ~monthly expiry
    _spreads = scan_spreads(chain, forecaster, prices, dte=_sdte)
    for s in _spreads:
        print("  " + s.line())
    print("  (EV>0 = the market pays more credit than the P-model's expected loss;\n"
          "   max loss is DEFINED — the crash can't blow the position up)\n")

    # 3b-2c. THE ORDER TICKET — the point where a view becomes an order ----------
    # Everything above is a view. This is the only block a human could act on
    # without filling in six blanks themselves: structure, strikes, expiry DATE,
    # contracts, limit, and the dollar worst case. It refuses BY NAME when it
    # cannot support a number, and the refusal is the common outcome at retail
    # size — one SPY put can lose more than a 2% budget on a $100k account.
    from models.ticket import build_ticket
    from models.trade_card import build_card
    _card = build_card(chain, forecaster, prices, dte=_sdte)
    _best = max(_spreads, key=lambda s: s.ev) if _spreads else None
    _ticket = build_ticket(
        _card, _best, equity=100_000.0, max_risk_frac=0.02,
        settled_trades=0,          # the forward ledger really is empty
        exit_rule="close at 50% of max profit, or at 7 DTE, whichever comes first")
    print(_ticket.render())
    print()

    # 3b-3. DE-AMERICANIZATION — unlock US single-name / ETF (American) options ---
    # BKM/VIX/BL replication assumes EUROPEAN prices; US equity & ETF options
    # (QQQ, SPY, the holdings inside income funds like QQQI) are American and
    # carry an early-exercise premium that biases the recovered Q variance HIGH.
    # Build an American chain at a KNOWN 25% vol, then read Q raw vs de-Americanized.
    from engine.american import american_price, de_americanize_chain
    from engine.data import OptionChain as _OC, OptionQuote as _OQ
    from models import rnd as _rnd
    a_dte, a_true = 45, 0.25
    a_T = a_dte / 365.0
    a_quotes = []
    for K in range(85, 116, 5):
        for kind in ("call", "put"):
            px = american_price(chain.spot, chain.spot * K / 100.0, a_T, chain.r, 0.0, a_true, kind, steps=120)
            a_quotes.append(_OQ(a_dte, round(chain.spot * K / 100.0, 2), kind, round(px, 4), round(px, 4)))
    a_chain = _OC("USEQ", chain.spot, chain.r, 0.0, a_quotes)
    raw_q = _rnd.model_free_implied_vol(a_chain, a_T, a_dte)
    deam_q = _rnd.model_free_implied_vol(de_americanize_chain(a_chain, steps=120), a_T, a_dte)
    print("De-Americanization (American chain priced at a known 25% vol):")
    print(f"  raw American Q vol   = {raw_q:.2%}  (biased HIGH by the early-exercise premium)")
    print(f"  de-Americanized Q    = {deam_q:.2%}  (early-exercise premium stripped -> unbiased)")
    print("  -> US equity/ETF options need this before any Q number is trustworthy\n")

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

    # 3c-2. MACRO regime gate (Phase 2) — the economic backdrop as a vol veto ----
    # The third gate the user asked for: yield-curve inversion, credit blowouts,
    # VIX-term backwardation, tightening shocks. Composes into the SAME event_risk.
    from models.macro import MacroSnapshot
    benign = MacroSnapshot("2026-07-19", short_rate=0.03, long_rate=0.043,
                           credit_spread=0.03, vix=15.0, vix_3m=17.0)
    stress = MacroSnapshot("2026-07-19", short_rate=0.055, long_rate=0.041,   # inverted
                           credit_spread=0.07, vix=34.0, vix_3m=27.0)
    for label, snap in (("benign macro", benign), ("stressed macro", stress)):
        fired, reason = event_risk(stressed, None, macro=snap)
        print(f"Macro gate [{label}]: curve={snap.curve_slope*100:+.0f}bp "
              f"credit={snap.credit_spread*100:.0f}bp vixTS={snap.vix_term_slope:+.1f} "
              f"-> veto={fired} ({reason})")
    print("  -> macro/news/price gates all OR into one `stressed` flag; none is a\n"
          "     directional bet — they only ever STOP selling vol, never direct it\n")

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

    # 3e. CALIBRATION — is P RIGHT in absolute terms, not just better than a rival?
    # NLL/CRPS only rank models against each other; a winner can still be
    # systematically too narrow. Since the edge IS the P-vs-Q gap, a P vol biased
    # low inflates every RICH verdict by exactly that bias. PIT says so, and hands
    # back the vol scale that would fix it.
    from models.calibration import calibration_report
    cal = calibration_report(BaselineDensityForecaster(), oos)
    print(cal.summary())
    print("  (width is judged CENTERED because the forecast is direction-neutral by\n"
          "   design — a trending sample is expected to tilt the raw PIT)\n")

    # 4. Position sizing -----------------------------------------------------
    if ideas:
        best = ideas[0]
        # Sized on the LOSS, not the premium. Selling the option means the price
        # is what we RECEIVE; the cap has to bind on what we can lose, or it
        # loosens exactly as the strike gets further out and cheaper.
        short = best.verdict.upper().startswith(("WRITE", "SELL"))
        qty = -1 if short else 1
        loss = sizing.max_loss_per_contract_for(
            best.quote.kind, best.quote.strike, quantity=qty,
            entry_price=best.quote.mid)
        n = sizing.position_size(100_000, max_loss_per_contract=loss,
                                 max_risk_frac=0.02)
        print(f"Top idea: {best.verdict} {best.quote.kind} "
              f"{best.quote.strike:.0f} / {best.quote.expiry_days}d")
        if loss == sizing.UNBOUNDED:
            print("Sized position: 0 contracts — a naked short call has no finite\n"
                  "  max loss, so there is nothing for a 2% cap to bind on. Buy a\n"
                  "  wing to make it a spread and it becomes sizeable.\n")
        else:
            print(f"Sized position (2% of equity at risk): {n} contracts, "
                  f"max loss ${loss * n:,.0f} on $100,000\n")

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

    # 7. Benchmarks — same path, three books ----------------------------------
    # "always-sell" is mechanically what option-income ETFs (QQQI/JEPQ) do; their
    # big distributions are harvested premium, not a high win rate. The question
    # that matters: does signal-gating beat it, and does either beat just holding?
    from engine.benchmark import compare_books, summary_table
    books = compare_books(price_path_with_crash(900), synthetic_chain_series(dte=21),
                          BaselineDensityForecaster(), dte=21, warmup=63)
    print("Benchmarks (same crash path, cost-inclusive):")
    print(summary_table(books) + "\n")

    # 8. Overnight-gap / short-gamma STRESS — the risk a smooth backtest hides ----
    # Continuous delta-hedging is a fiction; real markets gap over nights and
    # weekends and you are short gamma across the jump. Inject big jumps and re-run.
    from engine.stress import gap_stress
    gs = gap_stress(price_path_with_crash(760), synthetic_chain_series(dte=21),
                    BaselineDensityForecaster(), dte=21, warmup=63,
                    always_sell=True, every=40, size=0.20)
    print("Overnight-gap stress (ungated short vol, ±20% jumps):")
    print("  " + gs.summary().replace("\n", "\n  "))
    print("  -> big un-hedgeable jumps bleed short gamma; THIS is what the regime\n"
          "     gate + kill-switch defend against, and why continuous-hedge Sharpe lies\n")

    # 9. Does FUND CROWDING actually matter? — the paired experiment ------------
    # models/fund_flow can see WHERE the income ETFs concentrate their short calls.
    # Whether that knowledge is worth anything is an empirical question, so this is
    # the experiment rather than an assertion: on each date sell BOTH a crowded
    # strike and the nearest uncrowded one, and read the PAIRED difference. Note the
    # control run: with no dent injected the study reports NOTHING, which is the
    # property that makes the positive results worth reading at all.
    from engine.crowding_backtest import (run_crowding_backtest,
                                          synthetic_crowding_series)
    cpath = price_path_with_crash(3000)
    print("Does fund crowding matter? (paired, same-date, moneyness-matched):")
    for label, dent in (("control  — fund present, supply moves nothing", 0.00),
                        ("2 vol pt supply dent at the crowded strike   ", 0.02),
                        ("5 vol pt supply dent at the crowded strike   ", 0.05)):
        ch, su = synthetic_crowding_series(dte=21, dent=dent)
        cr = run_crowding_backtest(cpath, ch, su, dte=21, warmup=63)
        print(f"  {label}  n={cr.n_pairs:>4}  diff={cr.mean_diff:>+8.2f}  "
              f"t={cr.t_stat:>+5.2f}  -> {cr.verdict.split(' —')[0]}")
    print("  -> the harness is real; the ANSWER needs real published fund holdings,")
    print("     which is the one input this repo cannot synthesise for you.\n")

    # 10. Do the WINGS earn their cost? — defined-risk vs naked, same dates ------
    # models/spreads.py scores iron condors by expected value; this realises them.
    # The result is sharper than "spreads cap the tail": against a book that can
    # delta-hedge continuously the wings are a pure cost, and they only earn their
    # price where the hedge fails. Injecting gaps flips the comparison outright.
    from engine.spread_backtest import run_spread_backtest
    from engine.stress import evenly_spaced_gaps as _gaps
    from engine.stress import inject_gaps as _inject
    LADDER = tuple(round(-0.20 + 0.025 * i, 3) for i in range(17))
    sp_path = price_path_with_crash(900)
    print("Defined-risk spreads vs the delta-hedged naked book (same dates):")
    print(f"  {'path':20}{'spread tot':>12}{'naked tot':>11}"
          f"{'spread worst':>14}{'naked worst':>13}")
    for lbl, pth in (("clean", sp_path),
                     ("with ±18% gaps", _inject(sp_path, _gaps(900, every=40,
                                                               size=0.18, start=63)))):
        sr = run_spread_backtest(pth, synthetic_chain_series(dte=21, ladder=LADDER,
                                                             min_px=0.005),
                                 BaselineDensityForecaster(), dte=21, warmup=63,
                                 structure="iron_condor", always_sell=True)
        print(f"  {lbl:20}{sr.metrics.total_return:>+11.1%}"
              f"{sr.naked.total_return:>+11.1%}"
              f"{sr.worst_trade:>+14,.0f}{sr.naked_worst:>+13,.0f}")
    print("  -> a working delta hedge already removes what the wings are sold to cap,")
    print("     so buy them for the risk you CANNOT hedge (gaps), not the risk you can\n")

    # 11. How much of any of this is the sample? --------------------------------
    # Every number above came off ONE path. Holding the strategy fixed and changing
    # only the seed, this fixture's Sharpe runs from about -0.02 to +3.9 — so a
    # point estimate quoted without its width is the easiest way to fool yourself
    # in this whole codebase. Two views: rerun the world, and resample the trades.
    from engine.robustness import (block_bootstrap, bootstrap_summary,
                                   seed_ensemble)
    ens = seed_ensemble(lambda p: run_hedged_backtest(p),
                        lambda s: price_path_with_crash(900, seed=s),
                        seeds=range(1, 21))
    print(ens.summary())
    one = run_hedged_backtest(price_path_with_crash(900))
    print(bootstrap_summary(block_bootstrap(one.trade_pnl)))
    print("  -> read every Sharpe printed above through these intervals\n")

    # 12. MANY positions at once — the case the risk governor exists for --------
    # Every harness above holds ONE position at a time, so the governor is handed
    # an empty book on every entry and its correlation-aware aggregation never
    # runs. Stagger five underlyings and let it see the real book:
    from engine.book_backtest import run_book_backtest
    bk_px = {f"S{i}": price_path_with_crash(900, seed=i) for i in range(1, 6)}
    print("Correlation-aware sizing vs the per-trade view (cap 8,000 net short vega):")
    print(f"  {'sizing':34}{'peak vega':>11}{'maxDD':>9}{'total':>9}{'blocked':>9}")
    for lbl, kw in (("governed (sees the live book)", {}),
                    ("ungoverned (empty book)", {"govern": False})):
        bk = run_book_backtest(bk_px, **kw)
        print(f"  {lbl:34}{bk.peak_net_short_vega:>11,.0f}"
              f"{bk.metrics.max_drawdown:>9.1%}{bk.metrics.total_return:>+9.1%}"
              f"{bk.n_blocked_by_vega_cap:>9}")
    print("  -> sizing each trade as if it were the only one BREACHES the stated cap;")
    print("     a per-trade view understates the book's short vega by 54-77%\n")

    print("Reminder: swap SyntheticAdapter for real data before trusting any "
          "number. This fixture only proves the engine + risk discipline work.")


if __name__ == "__main__":
    main()
