"""Tests for the benchmark harness (engine/benchmark.py).
Run: python3 tests/test_benchmark.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.benchmark import compare_books, summary_table
from engine.hedged_backtest import price_path_with_crash
from engine.signal_backtest import synthetic_chain_series
from models.baseline import BaselineDensityForecaster

PRICES = price_path_with_crash(700)
CHAIN = synthetic_chain_series(dte=21)
F = BaselineDensityForecaster()
WARMUP = 63


def test_three_books_with_finite_metrics():
    books = compare_books(PRICES, CHAIN, F, dte=21, warmup=WARMUP)
    assert [b.name.split()[0] for b in books] == ["signal-gated", "always-sell", "buy-and-hold"]
    for b in books:
        m = b.metrics
        assert m.sharpe == m.sharpe and m.max_drawdown == m.max_drawdown
        assert -1.0 < m.max_drawdown <= 0.0


def test_signal_book_is_selective():
    sig, alw, _ = compare_books(PRICES, CHAIN, F, dte=21, warmup=WARMUP)
    assert 0 < sig.n_trades < alw.n_trades


def test_buy_and_hold_total_return_is_exact():
    _, _, bh = compare_books(PRICES, CHAIN, F, dte=21, warmup=WARMUP)
    expected = PRICES[-1] / PRICES[WARMUP] - 1.0
    assert abs(bh.metrics.total_return - expected) < 1e-9


def test_summary_table_lists_all_books():
    table = summary_table(compare_books(PRICES, CHAIN, F, dte=21, warmup=WARMUP))
    for name in ("signal-gated", "income-ETF", "buy-and-hold"):
        assert name in table
    assert "Sharpe" in table and "MaxDD" in table


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
