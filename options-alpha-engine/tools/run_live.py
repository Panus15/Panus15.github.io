"""Unified live runner — point the engine at REAL options data and get a report.

    python3 -m tools.run_live deribit --currency BTC --dte 30      # keyless, European crypto
    python3 -m tools.run_live tradier --symbol SPX  --dte 30       # needs TRADIER_TOKEN
    python3 -m tools.run_live replay  --chain-json btc.json --price-json btc_px.json

It runs the whole pipeline on whatever chain + price history the source returns:
  1. Q side   — model-free (VIX-style) vol + BKM risk-neutral moments (rnd).
  2. P-vs-Q   — per-expiry distributional scan: variance/skew risk premium (edge).
  3. Backtest — delta-hedged walk-forward short-vol on the underlying's history.
  4. Gate     — (optional, --gate) does a trained MDN beat the HAR baseline OOS?

Use --dump to freeze the fetched chain to the JsonFileAdapter schema so you can
replay it deterministically offline forever (no network, reproducible research).

This session's sandbox blocks market-data hosts; run it where egress is open (your
machine, or an environment whose network policy allows the vendor). The `analyze`
core is pure and unit-tested offline; only `fetch` touches the network.
"""

from __future__ import annotations

import argparse

from engine import hedged_backtest
from engine.data import OptionChain
from models import edge, rnd
from models.baseline import BaselineDensityForecaster


def analyze(chain: OptionChain, prices: list, *, dte: int = 30,
            label: str = "", run_backtest: bool = True, run_gate: bool = False,
            american: bool = False, contract_mult: float = 100.0) -> dict:
    """Run the full pipeline on a chain + price history. PURE (no network).

    ``american=True`` de-Americanizes the chain first (binomial American IV ->
    European-equivalent prices) so the Q-extractors, which assume European
    options, get an unbiased chain. REQUIRED for US single-name / ETF options
    (QQQ, SPY, ...); a no-op-ish pass for already-European chains (SPX/XSP, BTC)."""
    T = dte / 365.0
    if american:
        from engine.american import de_americanize_chain
        n_before = len(chain.quotes)
        chain = de_americanize_chain(chain)
        print(f"[de-Americanized] {n_before} American quotes -> "
              f"{len(chain.quotes)} European-equivalent (early-exercise premium stripped)")
    report: dict = {"symbol": chain.symbol, "spot": chain.spot, "asof": chain.asof,
                    "n_quotes": len(chain.quotes), "dte": dte, "american": american}
    print(f"\n=== {label or chain.symbol} @ {chain.asof} "
          f"spot={chain.spot:,.2f} quotes={len(chain.quotes)} ===")

    # 1. Q-side moments -----------------------------------------------------
    try:
        q_vol = rnd.model_free_implied_vol(chain, T, dte)
        m = rnd.bkm_moments(chain, T, dte)
        report["q_vol"] = q_vol
        report["q_skew"] = m.skew
        report["coverage_ok"] = m.coverage_ok
        print(f"Q (risk-neutral, {dte}d): model-free vol={q_vol:.1%}  "
              f"BKM skew={m.skew:+.2f} exkurt={m.kurtosis:+.2f}  "
              f"coverage_ok={m.coverage_ok} ({m.n_strikes} strikes)")
    except ValueError as e:
        print(f"Q extraction failed: {e} (need >=3 strikes with both call+put at {dte}d)")

    # 2. P-vs-Q distributional scan ----------------------------------------
    forecaster = BaselineDensityForecaster()
    if len(prices) >= 30:
        signals = edge.scan_distribution(chain, forecaster, prices, actionable_only=False)
        report["signals"] = [
            {"dte": s.expiry_days, "p_vol": s.p_vol, "q_vol": s.q_vol,
             "vrp": s.variance_risk_premium, "verdict": s.verdict} for s in signals]
        print(f"P-vs-Q scan  {'dte':>5} {'P_vol':>7} {'Q_vol':>7} {'VRP':>9} {'verdict':>9}")
        for s in signals[:8]:
            print(f"             {s.expiry_days:>5} {s.p_vol:>7.1%} {s.q_vol:>7.1%} "
                  f"{s.variance_risk_premium:>+9.4f} {s.verdict:>9}")
    else:
        print(f"P-vs-Q scan skipped: only {len(prices)} price points (need >=30 for HAR)")

    # 2b. Per-strike board — which exact contract looks mispriced ------------
    if len(prices) >= 30:
        from models.strike_scan import scan_strikes
        board = scan_strikes(chain, forecaster, prices, top=8, contract_mult=contract_mult)
        report["strike_board"] = [
            {"dte": s.expiry_days, "strike": s.strike, "kind": s.kind,
             "edge_buy": s.edge_buy, "edge_write": s.edge_write,
             "verdict": s.verdict} for s in board]
        print("Per-strike board (top 8 by |edge|):")
        for s in board:
            print("  " + s.line())

    # 3. Delta-hedged walk-forward backtest --------------------------------
    if run_backtest and len(prices) >= 63 + dte + 5:
        res = hedged_backtest.run_hedged_backtest(prices, dte=dte)
        report["backtest"] = {"sharpe": res.metrics.sharpe,
                              "max_drawdown": res.metrics.max_drawdown,
                              "total_return": res.metrics.total_return,
                              "trades": res.n_trades, "skipped": res.n_skipped}
        print(f"Backtest (delta-hedged, {len(prices)}d history): "
              f"Sharpe={res.metrics.sharpe:.2f}  MaxDD={res.metrics.max_drawdown:.1%}  "
              f"trades={res.n_trades} skipped={res.n_skipped}")
    elif run_backtest:
        print(f"Backtest skipped: need >= {63 + dte + 5} price points, have {len(prices)}")

    # 4. Promotion gate (optional, slow) -----------------------------------
    if run_gate and len(prices) >= 160:
        report["gate"] = _promotion_gate(prices)

    print("\nReminder: a single snapshot is a spot check. A real edge claim needs an\n"
          "honest OUT-OF-SAMPLE, cost-inclusive equity curve over many dates + a crash.")
    return report


def _promotion_gate(prices: list) -> dict:
    from models import objective
    from models.mdn import SequenceMDNForecaster
    cut = int(len(prices) * 0.75)
    mdn = SequenceMDNForecaster(context=63, seed=0)
    mdn.fit(prices[:cut], horizons=(30,), epochs=20, lr=0.05, max_windows=200)
    oos = objective.build_windows(prices[cut - 63:], context=63, horizons=(30,))
    base = objective.dataset_nll(BaselineDensityForecaster(), oos)
    mdn_nll = objective.dataset_nll(mdn, oos)
    winner = "MDN" if mdn_nll < base else "HAR baseline"
    print(f"Promotion gate (OOS NLL, {len(oos)} windows): HAR={base:.4f} MDN={mdn_nll:.4f} "
          f"-> ships: {winner}")
    return {"nll_baseline": base, "nll_mdn": mdn_nll, "winner": winner}


def fetch(source: str, args) -> tuple:
    """Return (chain, prices) from a live vendor or a saved replay file."""
    if source == "deribit":
        from engine.deribit import DeribitAdapter
        ad = DeribitAdapter(args.currency)
        return ad.option_chain(), ad.price_history(days=args.days)
    if source == "tradier":
        from engine.tradier import TradierAdapter
        ad = TradierAdapter()
        return ad.option_chain(args.symbol), ad.price_history(args.symbol, args.days)
    if source == "replay":
        from engine.adapters import JsonFileAdapter
        ad = JsonFileAdapter(price_json=args.price_json, chain_json=args.chain_json)
        prices = ad.price_history(args.symbol) if args.price_json else []
        return ad.option_chain(args.symbol), prices
    raise ValueError(f"unknown source {source!r}")


def main(argv=None):
    p = argparse.ArgumentParser(description="Run the options engine on real data")
    p.add_argument("source", choices=["deribit", "tradier", "replay"])
    p.add_argument("--currency", default="BTC")
    p.add_argument("--symbol", default="SPX")
    p.add_argument("--dte", type=int, default=30)
    p.add_argument("--days", type=int, default=400)
    p.add_argument("--chain-json", dest="chain_json")
    p.add_argument("--price-json", dest="price_json")
    p.add_argument("--dump", help="save the fetched chain to this JSON path")
    p.add_argument("--gate", action="store_true", help="also run the MDN promotion gate")
    p.add_argument("--no-backtest", dest="backtest", action="store_false")
    p.add_argument("--american", action="store_true",
                   help="de-Americanize the chain before Q extraction (US equity/ETF options)")
    args = p.parse_args(argv)

    chain, prices = fetch(args.source, args)
    if args.dump:
        from engine.deribit import save_chain_json
        save_chain_json(chain, args.dump)
        print(f"saved chain -> {args.dump} (replay offline with: "
              f"run_live replay --chain-json {args.dump})")
    # Crypto options are 1 coin/contract; US equity & ETF options are 100 shares.
    mult = 1.0 if args.source == "deribit" else 100.0
    analyze(chain, prices, dte=args.dte, label=args.source,
            run_backtest=args.backtest, run_gate=args.gate, american=args.american,
            contract_mult=mult)


if __name__ == "__main__":
    main()
