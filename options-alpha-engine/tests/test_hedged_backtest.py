"""Tests for the delta-hedged walk-forward backtest.

The point is not a specific Sharpe — it's that the engine behaves HONESTLY:
a crash-free sample looks like free money, a crash reveals the tail, and the
risk governor (regime gate, vega cap, drawdown kill-switch) actually fires.
Run: python3 tests/test_hedged_backtest.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import portfolio
from engine.data import SyntheticAdapter
from engine.hedged_backtest import price_path_with_crash, run_hedged_backtest


def test_calm_path_harvests_premium():
    calm = SyntheticAdapter(seed=5).price_history("X", 620)
    r = run_hedged_backtest(calm)
    assert r.n_trades > 5
    assert r.metrics.total_return > 0          # premium harvested when no crash
    assert r.metrics.sharpe > 0
    assert r.metrics.max_drawdown > -0.05       # shallow drawdown in a calm sample


def test_crash_reveals_the_tail():
    calm = SyntheticAdapter(seed=5).price_history("X", 620)
    crash = price_path_with_crash(756)
    rc = run_hedged_backtest(calm)
    rx = run_hedged_backtest(crash)
    # A crash makes the drawdown deeper and the Sharpe worse than the calm sample.
    assert rx.metrics.max_drawdown < rc.metrics.max_drawdown
    assert rx.metrics.sharpe < rc.metrics.sharpe


def test_regime_gate_skips_during_spike():
    crash = price_path_with_crash(756)
    r = run_hedged_backtest(crash)
    assert r.skip_reasons.get("regime", 0) > 0   # short-vol suppressed as vol accelerates


def test_drawdown_kill_switch_fires():
    crash = price_path_with_crash(756)
    r = run_hedged_backtest(
        crash, limits=portfolio.RiskLimits(max_net_short_vega=8000, max_drawdown=0.03)
    )
    assert r.skip_reasons.get("kill_switch", 0) > 0   # blocks new risk after breach


def test_vega_cap_blocks_oversized_book():
    crash = price_path_with_crash(756)
    r = run_hedged_backtest(
        crash, limits=portfolio.RiskLimits(max_net_short_vega=50, max_drawdown=0.9)
    )
    assert r.n_trades == 0
    assert r.skip_reasons.get("vega_cap", 0) > 0


def test_walk_forward_non_overlapping_coverage():
    # trades + skips must tile the timeline in non-overlapping dte blocks.
    calm = SyntheticAdapter(seed=5).price_history("X", 620)
    dte, warmup = 21, 63
    r = run_hedged_backtest(calm, dte=dte, warmup=warmup)
    blocks = (len(calm) - warmup) // dte
    assert r.n_trades + r.n_skipped == blocks


def test_vega_loss_deepens_crash_drawdown():
    # Same crash path, same params — only the opt-in vega-loss model is toggled.
    # Modelling the IV spike must make the SHORT vega book's crash mark-to-market
    # DEEPER (more negative max_drawdown) and the Sharpe WORSE than the constant-IV
    # (gamma-only) run. Entry sizing/gates are IV-entry based, so the SAME trades
    # are selected either way and the effect is purely the vega mark-to-market.
    crash = price_path_with_crash(756)
    off = run_hedged_backtest(crash, model_vega_loss=False)
    on = run_hedged_backtest(crash, model_vega_loss=True)
    assert on.n_trades == off.n_trades                          # identical trade set
    assert on.metrics.max_drawdown < off.metrics.max_drawdown   # strictly deeper crash DD
    assert on.metrics.sharpe < off.metrics.sharpe               # strictly worse Sharpe


def test_vega_loss_negligible_on_calm_path():
    # With no vol spike, trailing realised vol never runs hot enough to clear the
    # deadband, so the vega-loss model leaves a calm run essentially unchanged —
    # the vega loss is negligible when vol does not spike.
    calm = SyntheticAdapter(seed=5).price_history("X", 620)
    off = run_hedged_backtest(calm, model_vega_loss=False)
    on = run_hedged_backtest(calm, model_vega_loss=True)
    assert on.n_trades == off.n_trades
    assert abs(on.metrics.max_drawdown - off.metrics.max_drawdown) < 1e-4
    assert abs(on.metrics.sharpe - off.metrics.sharpe) < 0.05


def _run_all():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
