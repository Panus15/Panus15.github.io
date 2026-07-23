"""Tests for the paper-trading forward-test ledger (tools/paper_trade.py).

These assert the LEDGER's mechanics — no look-ahead in record, correct maturity
settlement, an unbiased P-vs-Q scoreboard, idempotent settle, and persistence —
NOT that the strategy makes money (that is the empirical question you answer by
feeding it real recorded snapshots).
Run: python3 tests/test_paper_trade.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.hedged_backtest import price_path_with_crash
from engine.signal_backtest import synthetic_chain_series
from models.baseline import BaselineDensityForecaster
from models.density import single_lognormal_riskneutral
from tools import paper_trade
from tools.paper_trade import PaperLedger, paper_trade_series

PRICES = price_path_with_crash(900)
CHAIN = synthetic_chain_series(dte=21)
F = BaselineDensityForecaster()


def test_record_then_settle_scores_finite():
    led = paper_trade_series(PRICES, CHAIN, F, dte=21, warmup=63)
    rep = led.report()
    assert rep.n_recorded > 10
    assert rep.n_settled > 0
    # every settled entry has a realised price and finite proper scores
    for e in led.entries:
        if e["status"] == "settled":
            assert e["s_realized"] == PRICES[e["entry_index"] + e["dte"]]
            assert e["p_nll"] == e["p_nll"] and e["q_nll"] == e["q_nll"]   # not NaN
    assert 0.0 <= rep.nll_win_rate <= 1.0


def test_no_lookahead_in_record():
    # record() must only ever forecast from prices[:entry_index+1].
    seen = []

    class Spy:
        def forecast(self, ctx, T, **kw):
            seen.append(len(ctx))
            return F.forecast(ctx, T, **kw)

    led = PaperLedger()
    for t in (100, 200, 300):
        led.record(CHAIN(t, PRICES[:t + 1]), PRICES, t, Spy(), dte=21)
    assert seen == [101, 201, 301]


def test_matching_density_ties_the_scoreboard():
    # Oracle: if P is set to EXACTLY the market Q log-normal, every proper score
    # must tie. Proves the scoreboard has no structural handicap either way.
    led = PaperLedger()
    e = led.record(CHAIN(300, PRICES[:301]), PRICES, 300, F, dte=21)
    assert e is not None
    T = e["dte"] / paper_trade.CAL_DAYS
    q_dist = single_lognormal_riskneutral(e["entry_spot"], T, e["r"], e["q"], e["q_vol"])
    e["p_density"] = paper_trade._density_to_list(q_dist)     # force P == Q
    led.settle(PRICES)
    assert abs(e["p_nll"] - e["q_nll"]) < 1e-9
    assert abs(e["p_crps"] - e["q_crps"]) < 1e-6
    assert abs(e["p_left_tail"] - e["q_left_tail"]) < 1e-9


def test_settle_is_idempotent_and_skips_immature():
    led = PaperLedger()
    # an entry whose horizon runs past the series end has no outcome -> stays open
    late = len(PRICES) - 20            # 880: matures at 901 > 900, never settles here
    e_late = led.record(CHAIN(late, PRICES[:late + 1]), PRICES, late, F, dte=21)
    e_mat = led.record(CHAIN(200, PRICES[:201]), PRICES, 200, F, dte=21)
    assert e_late is not None and e_mat is not None
    first = led.settle(PRICES)
    second = led.settle(PRICES)                    # idempotent: nothing new
    assert first == 1 and second == 0
    statuses = {e["entry_index"]: e["status"] for e in led.entries}
    assert statuses[200] == "settled" and statuses[late] == "open"


def test_persistence_roundtrip():
    led = paper_trade_series(PRICES, CHAIN, F, dte=21, warmup=63)
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as fh:
        path = fh.name
    try:
        led.save(path)
        back = PaperLedger.load(path)
        assert len(back.entries) == len(led.entries)
        assert back.entries[0]["p_density"] == led.entries[0]["p_density"]
        assert back.report().n_settled == led.report().n_settled
    finally:
        os.unlink(path)


def test_report_separates_calibration_from_profit():
    led = paper_trade_series(PRICES, CHAIN, F, dte=21, warmup=63)
    rep = led.report()
    # calibration is scored on ALL settled dates; trades only on fired ones
    n_traded = sum(1 for e in led.entries if e["status"] == "settled" and e["traded"])
    assert rep.n_trades == n_traded
    assert rep.n_trades <= rep.n_settled
    if rep.n_trades > 0:
        assert 0.0 <= rep.trade_hit_rate <= 1.0
        assert rep.total_pnl == rep.total_pnl            # finite
    assert "VERDICT" in rep.summary()


def test_none_chain_records_nothing():
    led = paper_trade_series(PRICES, lambda t, tr: None, F, dte=21, warmup=63)
    assert led.report().n_recorded == 0


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
