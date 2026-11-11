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
                               interval_from, seed_ensemble, _quantile)


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


# ---------------------------------------------------------------------------
# The sixteen survivors of this module's first mutation sweep.
#
# Nine of twenty-five defects were caught. The rest were in the places that
# decide how wide an interval is -- the quantile, the resampling, and the
# dispersion -- which is to say: in everything that makes "spans zero" mean
# something. A narrowed interval does not make a number wrong. It makes a
# wrong number look supported, which is worse, because it is believed.
# ---------------------------------------------------------------------------

def test_the_quantile_interpolates_between_its_two_neighbours():
    # [0, 1, 2, 3, 4] at p=0.1 sits at index 0.4, four tenths of the way from
    # 0 to 1. Literal values, because every wrong version here lands on a
    # plausible number: no interpolation gives 0.0, inverted gives 0.6, and an
    # index of p*n rather than p*(n-1) gives 0.5.
    v = [0.0, 1.0, 2.0, 3.0, 4.0]
    assert abs(_quantile(v, 0.1) - 0.4) < 1e-12, _quantile(v, 0.1)
    assert abs(_quantile(v, 0.5) - 2.0) < 1e-12, _quantile(v, 0.5)
    assert abs(_quantile(v, 0.75) - 3.0) < 1e-12, _quantile(v, 0.75)


def test_the_quantile_does_not_run_off_either_end():
    # p=1.0 indexes the last element; without the min() it reaches past it.
    v = [0.0, 1.0, 2.0, 3.0, 4.0]
    assert _quantile(v, 1.0) == 4.0, _quantile(v, 1.0)
    assert _quantile(v, 0.0) == 0.0, _quantile(v, 0.0)
    assert math.isnan(_quantile([], 0.5)), "an empty list has no quantile"
    assert _quantile([7.0], 0.9) == 7.0, "one sample is its own quantile"


def test_an_interval_entirely_below_zero_does_not_span_zero():
    # `lo <= 0 <= hi` with the upper half dropped reports every losing
    # interval as spanning zero -- which reads as "not distinguishable from
    # luck" when the honest reading is "distinguishably bad".
    losing = Interval(-3.0, -5.0, -1.0, 100, "all negative")
    assert not losing.spans_zero, "a wholly negative interval spanned zero"
    winning = Interval(3.0, 1.0, 5.0, 100, "all positive")
    assert not winning.spans_zero, "a wholly positive interval spanned zero"
    straddling = Interval(0.5, -1.0, 2.0, 100, "straddling")
    assert straddling.spans_zero
    touching = Interval(0.0, 0.0, 2.0, 100, "touching")
    assert touching.spans_zero, "a bound AT zero must count as spanning it"


def test_both_tails_are_trimmed_not_just_the_lower_one():
    # A 90% interval takes 5% off each end. Taking the 100th percentile as the
    # upper bound reports the maximum resample as if it were a bound.
    samples = [float(i) for i in range(101)]        # 0..100
    iv = interval_from(samples, level=0.90)
    assert abs(iv.lo - 5.0) < 1e-9, f"lo={iv.lo}, expected the 5th percentile"
    assert abs(iv.hi - 95.0) < 1e-9, (
        f"hi={iv.hi}: 100.0 is the maximum, 95.0 is the 95th percentile")


def test_nans_are_dropped_from_the_samples_not_counted_as_values():
    iv = interval_from([1.0, 2.0, float("nan"), 3.0, None], level=0.90)
    assert iv.n == 3, f"n={iv.n}: a nan or None was counted as a sample"


def test_a_percentage_interval_is_printed_as_a_percentage():
    # A return of 0.0049 in the cash format reads as +0.00, so the interval
    # looks empty and the reader concludes the opposite of the truth.
    line = Interval(0.0049, -0.0012, 0.0103, 50, "total return").line(pct=True)
    assert "%" in line, f"pct=True printed no percent sign: {line}"
    assert "+0.49%" in line, f"the point estimate is not readable: {line}"


def test_the_resample_is_the_same_length_as_the_data():
    # With block=3 over 10 trades the loop builds 4 blocks, so 12 values. The
    # truncation to n is what keeps the mean a mean: without it every resample
    # sums 12 values and divides by 10.
    pnl = [10.0] * 10
    boot = block_bootstrap(pnl, n_boot=50, block=3, seed=1)
    assert abs(boot["mean_trade"].point - 10.0) < 1e-9, boot["mean_trade"].point
    assert abs(boot["mean_trade"].lo - 10.0) < 1e-9, (
        f"lo={boot['mean_trade'].lo}: every trade is 10.0, so every resample "
        f"must mean 10.0 — a longer sample divided by n inflates it")
    assert abs(boot["mean_trade"].hi - 10.0) < 1e-9, boot["mean_trade"].hi


def test_blocks_wrap_around_rather_than_clamping_to_the_last_trade():
    # `(start + j) % n` wraps; `min(start + j, n - 1)` repeats the final trade,
    # which over-weights it in every resample that starts near the end. One
    # huge last trade makes the difference visible.
    pnl = [0.0] * 9 + [1000.0]
    boot = block_bootstrap(pnl, n_boot=400, block=3, seed=2)
    assert abs(boot["mean_trade"].point - 100.0) < 1e-9, boot["mean_trade"].point
    assert boot["mean_trade"].hi < 400.0, (
        f"hi={boot['mean_trade'].hi}: the last trade is being repeated inside "
        f"blocks instead of wrapping to the first")


def test_the_RESAMPLED_statistics_not_just_the_point_estimate():
    """The four in-loop statistics, asserted on the INTERVAL.

    My first attempt at these asserted on `.point` and killed nothing, because
    the point estimate is computed once outside the resampling loop from the
    observed data -- the mutants change the per-resample lists that become
    `lo` and `hi`. Testing the point tells you nothing about the resampling.

    The fixture makes every outcome enumerable. With [0, 10] and block=1 each
    resample draws two values from {0, 10}, so there are exactly three
    distinct results:

        (0,0) and (10,10)  mean 0 or 10, sd 0      -> Sharpe 0
        (0,10) either way  mean 5, sd sqrt(50)     -> Sharpe 0.7071

    Dividing the variance by n instead of n-1 gives sd 5.0 and a Sharpe of
    1.0; appending the mean instead of mean/sd gives 5.0; and an infinity on
    the zero-sd draws puts inf in the list. All three move the upper bound,
    and none of them moves the point.
    """
    boot = block_bootstrap([0.0, 10.0], n_boot=600, block=1, seed=11,
                           starting_equity=100_000.0)
    hi = boot["trade_sharpe"].hi
    assert math.isfinite(hi), (
        f"the Sharpe bound is {hi}: a zero-dispersion resample produced an "
        f"infinity, which formats as a record Sharpe")
    assert abs(hi - 0.7071) < 0.01, (
        f"Sharpe hi={hi:.4f}. 1.0000 means the variance was divided by n "
        f"rather than n-1; 5.0 means the mean was stored in place of "
        f"mean/sd; 0.7071 is the only value a sample sd can give here")

    # Total return: the largest resample is 20.0 of P&L on 100,000 of equity.
    rhi = boot["total_return"].hi
    assert abs(rhi - 0.0002) < 1e-9, (
        f"total return hi={rhi}: 20.0 means the sum was never divided by the "
        f"starting equity")


def test_the_observed_point_estimate_uses_a_sample_sd_too():
    # The other half: `obs_sd` is its own line, and this is what pins it.
    pnl = [1.0, 2.0, 3.0, 10.0]
    boot = block_bootstrap(pnl, n_boot=1, block=1, seed=3)
    mean = sum(pnl) / 4
    var = sum((x - mean) ** 2 for x in pnl) / 3          # n-1
    want = mean / math.sqrt(var)
    assert abs(boot["trade_sharpe"].point - want) < 1e-9, (
        f"{boot['trade_sharpe'].point} against {want}")


def test_the_total_return_point_is_scaled_by_the_starting_equity():
    pnl = [1000.0] * 10          # 10,000 on 100,000 = 10%
    boot = block_bootstrap(pnl, n_boot=20, block=1, seed=6,
                           starting_equity=100_000.0)
    assert abs(boot["total_return"].point - 0.10) < 1e-9, (
        f"{boot['total_return'].point}: 10,000 of P&L on 100,000 of equity is "
        f"0.10, and 10000.0 means it was never divided")


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
