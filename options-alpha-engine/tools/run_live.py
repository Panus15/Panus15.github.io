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


#: Underlyings whose options are ONE unit per contract (crypto), not 100 shares.
_UNIT_CONTRACT_SYMBOLS = ("BTC", "ETH", "SOL", "XRP", "DOGE", "AVAX", "MATIC")


def default_contract_mult(source: str, chain: OptionChain) -> float:
    """Contract size to assume: 1 unit for crypto, 100 shares for equity/ETF.

    Keyed off the CHAIN, not just the CLI source — the documented workflow is
    `--dump` from a live vendor and then `replay` the file offline, and a replayed
    crypto chain must keep its 1-coin contract size. Deriving it from the source
    alone silently reverts a dumped BTC chain to 100x on replay, overstating every
    dollar figure by two orders of magnitude. Override with --contract-mult.
    """
    if source == "deribit":
        return 1.0
    symbol = (chain.symbol or "").upper()
    if any(tok in symbol for tok in _UNIT_CONTRACT_SYMBOLS):
        return 1.0
    return 100.0


def analyze(chain: OptionChain, prices: list, *, dte: int = 30,
            label: str = "", run_backtest: bool = True, run_gate: bool = False,
            american: bool = False, contract_mult: float = 100.0) -> dict:
    """Run the full pipeline on a chain + price history. PURE (no network).

    ``american=True`` de-Americanizes the chain first (binomial American IV ->
    European-equivalent prices) so the Q-extractors, which assume European
    options, get an unbiased chain. REQUIRED for US single-name / ETF options
    (QQQ, SPY, ...); a no-op-ish pass for already-European chains (SPX/XSP, BTC)."""
    # Vendors list their OWN expiries; a requested --dte 30 usually does not exist
    # (Deribit quotes 5/12/19/33/61/152/243/334, say). Snap to the nearest listed
    # expiry instead of silently failing Q extraction on an empty slice.
    available = sorted({q.expiry_days for q in chain.quotes})
    if available and dte not in available:
        snapped = min(available, key=lambda d: abs(d - dte))
        print(f"[expiry] no {dte}d expiry listed; snapping to the nearest: {snapped}d "
              f"(available: {', '.join(str(d) for d in available[:10])}"
              f"{'...' if len(available) > 10 else ''})")
        dte = snapped
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
        res = hedged_backtest.run_hedged_backtest(
            prices, dte=dte, contract_mult=int(contract_mult))
        report["backtest"] = {"sharpe": res.metrics.sharpe,
                              "max_drawdown": res.metrics.max_drawdown,
                              "total_return": res.metrics.total_return,
                              "trades": res.n_trades, "skipped": res.n_skipped}
        reasons = "; ".join(f"{k}:{v}" for k, v in res.skip_reasons.items()) or "none"
        print(f"Backtest (delta-hedged, {len(prices)}d history): "
              f"Sharpe={res.metrics.sharpe:.2f}  MaxDD={res.metrics.max_drawdown:.1%}  "
              f"trades={res.n_trades} skipped={res.n_skipped} ({reasons})")
        if res.n_trades == 0 and res.skip_reasons:
            top = max(res.skip_reasons, key=res.skip_reasons.get)
            why = {
                "size_zero": ("CVaR sizing returned 0 contracts — one contract's tail "
                              "loss exceeds the risk budget. On a high-notional "
                              "underlying (BTC ~$64k/contract) a small account "
                              "genuinely cannot carry one; raise starting_equity or "
                              "cvar_limit, or trade a defined-risk spread instead."),
                "vega_cap": ("the short-vega cap bound every trade — the position's "
                             "vega is large relative to the equity limit."),
                "regime": "the regime gate suppressed selling (realised vol accelerating).",
                "kill_switch": "the drawdown kill-switch was active.",
            }.get(top, "see skip_reasons above.")
            print(f"  -> no trades because {top} dominated: {why}")
    elif run_backtest:
        print(f"Backtest skipped: need >= {63 + dte + 5} price points, have {len(prices)}")

    # 3b. Is the P density even CALIBRATED on this underlying? ---------------
    # The VRP is P-vs-Q, so a P vol biased low inflates every RICH verdict by
    # exactly that bias. PIT answers it in absolute terms, and hands back the vol
    # scale that would fix it. Width is judged on a CENTERED PIT because this
    # forecast is direction-neutral by design.
    if len(prices) >= 150:
        from models import calibration
        windows = objective_windows(prices, dte)
        if len(windows) >= 40:
            cal = calibration.calibration_report(forecaster, windows)
            report["calibration"] = {"vol_scale": cal.vol_scale,
                                     "verdict": cal.verdict.split(" — ")[0],
                                     "coverage_90": cal.centered_coverage_90}
            print("\n" + cal.summary())
            if cal.vol_scale > 1.15:
                print(f"  !! the P vol looks ~{(cal.vol_scale - 1) * 100:.0f}% too LOW here, so the "
                      f"VRP above is OVERSTATED — treat RICH verdicts with suspicion")
            elif cal.vol_scale < 0.87:
                print(f"  !! the P vol looks ~{(1 - cal.vol_scale) * 100:.0f}% too HIGH here, so the "
                      f"VRP above is UNDERSTATED")

            # Re-scan with the bias corrected, using a scale fitted ONLY on the
            # earlier part of the history (walk-forward — a scale fitted on the
            # same data it corrects is in-sample fitting and proves nothing).
            wf = calibration.walk_forward_scale(forecaster, prices, dte=dte)
            report["wf_vol_scale"] = wf
            if abs(wf - 1.0) > 0.05:
                fixed = calibration.CalibratedForecaster(forecaster, vol_scale=wf)
                adj = edge.scan_distribution(chain, fixed, prices, actionable_only=False)
                report["signals_calibrated"] = [
                    {"dte": s.expiry_days, "p_vol": s.p_vol, "q_vol": s.q_vol,
                     "vrp": s.variance_risk_premium, "verdict": s.verdict} for s in adj]
                print(f"\nSENSITIVITY: P-vs-Q re-scanned with a walk-forward vol scale "
                      f"x{wf:.2f} (fitted on the earlier history only):")
                print(f"  {'dte':>5} {'P_vol':>7} {'Q_vol':>7} {'VRP':>9} {'verdict':>9}")
                for s in adj[:8]:
                    print(f"  {s.expiry_days:>5} {s.p_vol:>7.1%} {s.q_vol:>7.1%} "
                          f"{s.variance_risk_premium:>+9.4f} {s.verdict:>9}")
                print("  -> read this as a STRESS TEST of the signal, not a better "
                      "estimate: the scale\n     conflates genuine model bias with a "
                      "regime shift between the fitting window\n     and now, so on a "
                      "regime-changing sample it OVERCORRECTS. What matters is\n"
                      "     whether a verdict SURVIVES it — one that flips was never robust.")
                # Pair by EXPIRY, not position: both scans are sorted by |VRP|, so
                # zipping them compares different contracts.
                raw_by_dte = {s.expiry_days: s.verdict for s in signals}
                flips = sum(1 for s in adj
                            if raw_by_dte.get(s.expiry_days, s.verdict) != s.verdict)
                print(f"  {flips}/{len(adj)} verdicts flip under this correction.")
                if wf >= 1.9 or wf <= 0.55:
                    print("  !! the scale hit its clamp — the estimate is extreme "
                          "(usually a regime change,\n     not a model bias). Treat both "
                          "scans as wide error bars, not a decision.")

    # 3c. THE CARD — collapse everything above into one decision object -----
    if len(prices) >= 30:
        from models.trade_card import build_card
        card = build_card(chain, forecaster, prices, dte=dte,
                          calibration=locals().get("cal"))
        report["card"] = {"vol_side": card.vol_side, "direction": card.direction,
                          "entry": card.entry, "target": card.target,
                          "stop": card.stop, "expected_value": card.expected_value}
        print("\n" + card.render())

    # 4. Promotion gate (optional, slow) -----------------------------------
    if run_gate and len(prices) >= 160:
        report["gate"] = _promotion_gate(prices)

    print("\nReminder: a single snapshot is a spot check. A real edge claim needs an\n"
          "honest OUT-OF-SAMPLE, cost-inclusive equity curve over many dates + a crash.")
    return report


def objective_windows(prices: list, dte: int, *, context: int = 63):
    """Leak-free (context, horizon, outcome) windows at this horizon, capped so a
    long history does not make the calibration scan crawl."""
    from models import objective
    w = objective.build_windows(prices, context=context, horizons=(dte,))
    return w[::max(1, len(w) // 250)]


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
    p.add_argument("--contract-mult", dest="contract_mult", type=float, default=None,
                   help="units per contract (1 for crypto, 100 for US equity/ETF); "
                        "default is inferred from the source and the chain symbol")
    args = p.parse_args(argv)

    chain, prices = fetch(args.source, args)
    if args.dump:
        from engine.deribit import save_chain_json
        save_chain_json(chain, args.dump)
        print(f"saved chain -> {args.dump} (replay offline with: "
              f"run_live replay --chain-json {args.dump})")
    mult = args.contract_mult if args.contract_mult else default_contract_mult(
        args.source, chain)
    analyze(chain, prices, dte=args.dte, label=args.source,
            run_backtest=args.backtest, run_gate=args.gate, american=args.american,
            contract_mult=mult)


if __name__ == "__main__":
    main()
