"""Tests for confidence intervals (engine/robustness.py).

An interval that does not actually cover what it claims is worse than no interval
at all — it launders a guess into a statistic. So the load-bearing test here is a
COVERAGE ORACLE: draw many datasets from a known distribution, build the 90%
interval on each, and check the true value really does land inside about 90% of
the time. The rest checks the two failure modes that matter in practice: an
interval that spans zero must SAY so, and blocking must actually preserve the
clustering that makes a losing streak a losing streak.

Run: python3 tests/test_robustness.py
"""

import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.hedged_backtest import price_path_with_crash, run_hedged_backtest
from engine.robustness import (Interval, block_bootstrap, bootstrap_summary,
                               interval_from, seed_ensemble)


def test_interval_mechanics():
    i = interval_from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], level=0.90, label="x")
    assert i.n == 10 and i.lo < i.point < i.hi
    assert not i.spans_zero
    assert Interval(0.5, -1.0, 2.0, 10).spans_zero
    assert Interval(-0.5, -2.0, 1.0, 10).spans_zero
    assert not Interval(2.0, 1.0, 3.0, 10).spans_zero
    assert "spans zero" in Interval(0.5, -1.0, 2.0, 10, "s").line()
    # an empty sample degrades to nan rather than inventing a number
    empty = interval_from([])
    assert math.isnan(empty.point) and empty.n == 0


def test_the_bootstrap_interval_actually_covers_90_percent():
    """The oracle. A 90% interval that covers 50% of the time is a lie."""
    rng = random.Random(11)
    true_mean, sd, n, reps = 5.0, 40.0, 60, 250
    hits = 0
    for k in range(reps):
        pnl = [rng.gauss(true_mean, sd) for _ in range(n)]
        b = block_bootstrap(pnl, n_boot=300, block=1, seed=k, level=0.90)
        iv = b["mean_trade"]
        if iv.lo <= true_mean <= iv.hi:
            hits += 1
    rate = hits / reps
    assert 0.83 <= rate <= 0.97, f"90% interval covered {rate:.1%} over {reps} draws"


def test_blocks_preserve_clustering_and_widen_the_interval():
    """Resampling single trades breaks up losing streaks and understates the risk."""
    # strongly autocorrelated: long runs of wins then long runs of losses
    pnl = ([200.0] * 12 + [-200.0] * 12) * 4
    single = block_bootstrap(pnl, n_boot=1500, block=1, seed=1)
    blocked = block_bootstrap(pnl, n_boot=1500, block=12, seed=1)
    w_single = single["mean_trade"].hi - single["mean_trade"].lo
    w_block = blocked["mean_trade"].hi - blocked["mean_trade"].lo
    assert w_block > w_single * 1.5, (
        f"blocking must widen the interval on clustered P&L: "
        f"block=1 width {w_single:.1f} vs block=12 width {w_block:.1f}")


def test_a_book_with_no_edge_is_reported_as_having_no_edge():
    rng = random.Random(3)
    noise = [rng.gauss(0.0, 100.0) for _ in range(40)]
    b = block_bootstrap(noise, n_boot=1000, seed=2)
    assert b["mean_trade"].spans_zero
    assert "not distinguishable from luck" in bootstrap_summary(b)

    # ...and a book with a real, large edge is NOT dismissed
    strong = [rng.gauss(500.0, 50.0) for _ in range(40)]
    b2 = block_bootstrap(strong, n_boot=1000, seed=2)
    assert not b2["mean_trade"].spans_zero
    assert "not distinguishable from luck" not in bootstrap_summary(b2)


def test_the_bootstrap_is_deterministic():
    pnl = [1.0, -2.0, 3.0, -4.0, 5.0, 6.0, -1.0, 2.0]
    a = block_bootstrap(pnl, n_boot=200, seed=7)
    c = block_bootstrap(pnl, n_boot=200, seed=7)
    assert a["mean_trade"].lo == c["mean_trade"].lo
    assert a["mean_trade"].hi == c["mean_trade"].hi
    d = block_bootstrap(pnl, n_boot=200, seed=8)
    assert (d["mean_trade"].lo, d["mean_trade"].hi) != (a["mean_trade"].lo,
                                                        a["mean_trade"].hi)


def test_too_few_trades_is_refused_not_guessed():
    for bad in ([], [42.0]):
        try:
            block_bootstrap(bad)
        except ValueError:
            continue
        raise AssertionError(f"bootstrapping {bad} should refuse, not guess")


def test_the_seed_ensemble_exposes_how_wide_the_headline_sharpe_really_is():
    """The finding that motivated this module: same strategy, 60x range of Sharpe."""
    ens = seed_ensemble(lambda p: run_hedged_backtest(p),
                        lambda s: price_path_with_crash(900, seed=s),
                        seeds=range(1, 13))
    assert len(ens.seeds) == 12
    assert len(ens.per_seed) == 12
    spread = ens.sharpe.hi - ens.sharpe.lo
    assert spread > 1.0, (
        f"the fixture is known to swing from about -0.02 to +3.9 across seeds; "
        f"an interval of width {spread:.2f} is not reporting that")
    assert ens.sharpe.lo < ens.sharpe.point < ens.sharpe.hi
    assert "Seed ensemble" in ens.summary()
    # each seed really is a different world
    assert len({round(r["sharpe"], 6) for r in ens.per_seed}) > 8


def test_the_ensemble_survives_a_harness_that_throws():
    def flaky(prices):
        if len(prices) % 2 == 0:
            raise ValueError("no")
        return run_hedged_backtest(prices)

    ens = seed_ensemble(flaky, lambda s: price_path_with_crash(900 + s % 2, seed=s),
                        seeds=range(1, 9))
    assert 0 < len(ens.seeds) < 8, "failed runs must be dropped, not fatal"


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
