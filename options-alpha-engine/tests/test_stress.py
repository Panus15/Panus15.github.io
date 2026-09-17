"""Tests for overnight-gap / jump stress (engine/stress.py).
Run: python3 tests/test_stress.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.hedged_backtest import price_path_with_crash
from engine.signal_backtest import synthetic_chain_series
from engine.stress import evenly_spaced_gaps, gap_stress, inject_gaps
from models.baseline import BaselineDensityForecaster


def test_inject_gaps_shifts_the_level_forward():
    p = [100.0] * 10
    g = inject_gaps(p, [(5, math.log(1.10))])
    assert all(abs(x - 100.0) < 1e-9 for x in g[:5])
    assert all(abs(x - 110.0) < 1e-9 for x in g[5:])      # level jumps and stays
    assert p[5] == 100.0                                  # original untouched (copy)


def test_evenly_spaced_gaps_alternate_sign():
    gs = evenly_spaced_gaps(300, every=42, size=0.10, start=63)
    assert gs and gs[0][0] == 105
    assert gs[0][1] > 0 and gs[1][1] < 0                  # alternating ±
    assert all(0 <= i < 300 for i, _ in gs)


def test_large_gaps_bleed_the_ungated_short_vol_book():
    # Short gamma: a LARGE jump adds realised variance faster than the book can
    # re-price IV to offset, so the ungated always-sell book does clearly worse on
    # the gapped path — the exact tail the regime gate exists to defend against.
    # (Small gaps can even help, because the book then sells the elevated IV; the
    # danger is the big, un-hedgeable move, so the stress uses one.)
    P = price_path_with_crash(760)
    CH = synthetic_chain_series(dte=21)
    F = BaselineDensityForecaster()
    res = gap_stress(P, CH, F, dte=21, warmup=63, always_sell=True, every=40, size=0.20)
    assert res.n_gaps > 0
    assert res.gapped.metrics.total_return < res.clean.metrics.total_return
    assert res.gapped.metrics.max_drawdown < res.clean.metrics.max_drawdown
    assert "gap stress" in res.summary()


def test_gaps_do_not_all_land_on_trade_boundaries():
    """A jump only hurts a short-gamma book if the book is holding gamma into it.

    The default every=42 against the default dte=21 put 100% of the gaps exactly
    on entry/expiry boundaries, where remaining time is zero and gamma is dead —
    the stress ran, reported a number, and measured almost nothing. Found by
    audit: with the resonance removed the same 16 jumps cost 10 points more.
    """
    from engine.stress import evenly_spaced_gaps

    raw = evenly_spaced_gaps(760, every=42, size=0.20, start=63)
    assert {(i - 63) % 21 for i, _ in raw} == {0}, "fixture premise changed"

    P = price_path_with_crash(760)
    CH = synthetic_chain_series(dte=21)
    F = BaselineDensityForecaster()
    res = gap_stress(P, CH, F, dte=21, warmup=63, always_sell=True,
                     every=42, size=0.20)
    # the de-resonated schedule must actually bite
    bleed = res.clean.metrics.total_return - res.gapped.metrics.total_return
    assert bleed > 0.05, (
        f"gaps at every=42/dte=21 cost only {bleed:.1%} — they are probably "
        f"landing on expiry boundaries again")

    # a non-resonant schedule is left exactly where the caller put it
    off = evenly_spaced_gaps(760, every=40, size=0.20, start=63)
    assert len({(i - 63) % 21 for i, _ in off}) > 1


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
