"""Tests for the multi-date signal-driven walk-forward (engine/signal_backtest.py).

These assert the FRAMEWORK's mechanics — walk-forward (no look-ahead), that the
signal is selective and regime-gated vs the always-sell baseline — NOT that the
signal makes money (that is the empirical question you answer by feeding it real
chain snapshots; on a synthetic fixture the signal is deliberately ~neutral).
Run: python3 tests/test_signal_backtest.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.hedged_backtest import price_path_with_crash
from engine.signal_backtest import run_signal_backtest, synthetic_chain_series
from models.baseline import BaselineDensityForecaster

PRICES = price_path_with_crash(900)
F = BaselineDensityForecaster()


def test_runs_and_metrics_finite():
    r = run_signal_backtest(PRICES, synthetic_chain_series(dte=21), F, dte=21)
    assert r.n_periods > 10
    assert r.metrics.sharpe == r.metrics.sharpe          # not NaN
    assert -1.0 < r.metrics.max_drawdown <= 0.0


def test_signal_is_selective_and_regime_gated():
    ch = synthetic_chain_series(dte=21)
    sig = run_signal_backtest(PRICES, ch, F, dte=21, always_sell=False)
    alw = run_signal_backtest(PRICES, ch, F, dte=21, always_sell=True)
    assert sig.n_sold < alw.n_sold                       # signal trades fewer
    assert sig.skip_reasons.get("regime", 0) > 0         # regime gate fired in the crash
    # always-sell trades every period whose chain is usable
    assert alw.n_sold == alw.n_periods - alw.skip_reasons.get("bad_chain", 0)


def test_high_min_vrp_suppresses_selling():
    r = run_signal_backtest(PRICES, synthetic_chain_series(dte=21), F, dte=21, min_vrp=1.0)
    assert r.skip_reasons.get("not_rich", 0) > 0 and r.n_sold <= 2


def test_walk_forward_no_lookahead():
    # The chain builder must only ever see prices up to the decision date t.
    seen = []
    base = synthetic_chain_series(dte=21)

    def spy(t, trailing):
        seen.append((t, len(trailing)))
        return base(t, trailing)

    run_signal_backtest(PRICES, spy, F, dte=21, warmup=63)
    assert seen
    for t, length in seen:
        assert length == t + 1                           # exactly prices[:t+1], no future


def test_none_chain_is_skipped():
    r = run_signal_backtest(PRICES, lambda t, tr: None, F, dte=21)
    assert r.n_sold == 0 and r.skip_reasons.get("no_chain", 0) == r.n_periods


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
