"""Tests for the paper-trading forward-test ledger (tools/paper_trade.py).

These assert the LEDGER's mechanics — no look-ahead in record, correct maturity
settlement, an unbiased P-vs-Q scoreboard, idempotent settle, and persistence —
NOT that the strategy makes money (that is the empirical question you answer by
feeding it real recorded snapshots).
Run: python3 tests/test_paper_trade.py
"""

import contextlib
import io
import json
import math
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import pricing
from engine.data import OptionChain, OptionQuote
from engine.hedged_backtest import price_path_with_crash
from engine.signal_backtest import _realize_short_straddle, synthetic_chain_series
from models.baseline import BaselineDensityForecaster
from models.density import single_lognormal_riskneutral
from tools import paper_trade
from tools.paper_trade import CAL_DAYS, PaperLedger, paper_trade_series

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


def test_p_better_density_wins_the_scoreboard():
    # Guards the report's win DIRECTION. Force P to a tight density spiked at the
    # realised price (strictly better than the broad market Q); it MUST register as
    # a P win. A flipped comparison (p_nll > q_nll) drives nll_win_rate to 0 -> fail.
    led = PaperLedger()
    e = led.record(CHAIN(300, PRICES[:301]), PRICES, 300, F, dte=21)
    assert e is not None
    mat = e["entry_index"] + e["dte"]
    e["p_density"] = [[1.0, math.log(PRICES[mat]), 0.01]]     # near-Dirac at the outcome
    led.settle(PRICES)
    assert e["p_nll"] < e["q_nll"]
    assert led.report().nll_win_rate == 1.0


def test_trade_pnl_matches_direct_realization():
    # Pins the profit path's ARITHMETIC and ARG WIRING: a mis-wired strike/iv or a
    # zeroed pnl would diverge from an independent realisation with the same args.
    led = paper_trade_series(PRICES, CHAIN, F, dte=21, warmup=63)
    traded = [e for e in led.entries if e["status"] == "settled" and e["traded"]]
    assert traded, "expected at least one traded+settled entry"
    e = traded[0]
    direct = sum(_realize_short_straddle(
        PRICES, e["entry_index"], e["dte"], e["strike"], e["entry_iv"],
        e["r"], e["q"], e["hedge_bps"], e["spread_frac"]).values())
    assert abs(e["trade_pnl"] - direct) < 1e-9


def test_profit_path_actually_fires():
    # The synthetic series must exercise the delta-hedged path, else a broken gate
    # that silently records zero trades would pass every other (guarded) trade test.
    led = paper_trade_series(PRICES, CHAIN, F, dte=21, warmup=63)
    rep = led.report()
    assert rep.n_trades > 0
    traded = [e for e in led.entries if e["status"] == "settled" and e["traded"]]
    assert traded and all(e["trade_pnl"] is not None for e in traded)


def test_costs_persist_across_save_load():
    # Record (do NOT settle) with non-default costs, save, load WITHOUT kwargs, then
    # settle both. Per-entry cost persistence must make the trade P&Ls identical; if
    # costs fell back to the loaded ledger's defaults, the P&Ls would diverge.
    led = PaperLedger(hedge_bps=2e-3, spread_frac=0.05)
    for idx in range(63, 400, 21):
        led.record(CHAIN(idx, PRICES[:idx + 1]), PRICES, idx, F, dte=21)
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as fh:
        path = fh.name
    try:
        led.save(path)
        back = PaperLedger.load(path)                    # loads with default 5e-4 / 0.015
        assert back.hedge_bps != led.hedge_bps           # ledger-level defaults DO differ
        led.settle(PRICES)
        back.settle(PRICES)
        by_idx = {e["entry_index"]: e for e in back.entries}
        traded = [e for e in led.entries if e["traded"] and e["status"] == "settled"]
        assert traded
        for e in traded:
            assert abs(e["trade_pnl"] - by_idx[e["entry_index"]]["trade_pnl"]) < 1e-9
    finally:
        os.unlink(path)


def test_multi_expiry_reads_requested_dte():
    # Two expiries at distinct vols; record(dte=21) must read strike/iv/q_vol from
    # the 21d slice only -- a dropped expiry filter would grab the 42d 0.40 vol.
    spot, r, q = 100.0, 0.03, 0.0
    quotes = []
    for dte_, vol in ((21, 0.20), (42, 0.40)):
        T = dte_ / 365.0
        for mny in (-0.10, -0.05, 0.0, 0.05, 0.10):
            K = round(spot * (1 + mny), 2)
            for kind in ("call", "put"):
                px = pricing.price(spot, K, T, r, q, vol, kind)
                if px < 0.02:
                    continue
                quotes.append(OptionQuote(dte_, K, kind, round(max(px - 0.02, 0.01), 2),
                                          round(px + 0.02, 2)))
    chain = OptionChain("MX", spot, r, q, quotes)
    prices = [100.0 * (1 + 0.01 * math.sin(i / 5.0)) for i in range(80)]
    led = PaperLedger()
    e = led.record(chain, prices, len(prices) - 1, F, dte=21)
    assert e is not None
    assert e["strike"] == 100.0
    assert abs(e["entry_iv"] - 0.20) < 0.03          # 21d vol, not the 42d 0.40
    assert abs(e["q_vol"] - 0.20) < 0.05


def test_news_gate_vetoes_an_otherwise_traded_entry():
    # Find an index the strategy trades on, then re-record the SAME decision with
    # a burst of negative news -> the trade must be vetoed and the reason kept.
    from models.sentiment import SentimentFeature
    led = paper_trade_series(PRICES, CHAIN, F, dte=21, warmup=63)
    traded_idx = next(e["entry_index"] for e in led.entries if e["traded"])

    risk_news = SentimentFeature(score=-0.8, volume=5.0, dispersion=0.1,
                                 most_negative=-0.9, n_items=5)
    calm_news = SentimentFeature(score=0.4, volume=5.0, dispersion=0.1,
                                 most_negative=0.0, n_items=5)
    led2 = PaperLedger()
    chain = CHAIN(traded_idx, PRICES[:traded_idx + 1])
    vetoed = led2.record(chain, PRICES, traded_idx, F, dte=21, news_feature=risk_news)
    allowed = led2.record(chain, PRICES, traded_idx, F, dte=21, news_feature=calm_news)
    assert vetoed["traded"] is False and "negative news" in vetoed["gate_reason"]
    assert allowed["traded"] is True and "benign" in allowed["gate_reason"]


def _dump_chain_json(chain, path, asof="2026-01-01"):
    obj = {"symbol": chain.symbol, "spot": chain.spot, "r": chain.r, "q": chain.q,
           "asof": asof, "quotes": [{"expiry_days": q.expiry_days, "strike": q.strike,
                                     "kind": q.kind, "bid": q.bid, "ask": q.ask}
                                    for q in chain.quotes]}
    with open(path, "w") as fh:
        json.dump(obj, fh)


def test_cli_record_settle_report_offline():
    # End-to-end CLI over the offline `replay` source: record seeds a stable price
    # journal and appends an entry; growing the journal with the real future path
    # (simulating days passing) lets settle/report grade it out of sample.
    from tools.paper_trade import main

    full = price_path_with_crash(300)
    d = tempfile.mkdtemp()
    try:
        cp, pp, lp = (os.path.join(d, "chain.json"),
                      os.path.join(d, "px.json"), os.path.join(d, "led.jsonl"))
        _dump_chain_json(CHAIN(99, list(full[:100])), cp)
        with open(pp, "w") as fh:
            json.dump(list(full[:100]), fh)

        with contextlib.redirect_stdout(io.StringIO()):
            main(["record", "--source", "replay", "--chain-json", cp, "--price-json", pp,
                  "--ledger", lp, "--dte", "21", "--symbol", "SYN"])
        assert os.path.exists(lp) and os.path.exists(lp + ".prices.json")
        led = PaperLedger.load(lp)
        assert len(led.entries) == 1 and led.entries[0]["entry_index"] == 99
        assert led.entries[0]["status"] == "open"            # journal only 100 bars -> immature
        assert json.load(open(lp + ".prices.json")) == list(full[:100])

        # ~a month of daily bars arrive: extend the journal with the REAL future path
        with open(lp + ".prices.json", "w") as fh:
            json.dump(list(full[:130]), fh)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main(["settle", "--ledger", lp])
        assert "newly matured" in buf.getvalue()
        assert PaperLedger.load(lp).entries[0]["status"] == "settled"   # matured at 120 < 130

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main(["report", "--ledger", lp])
        assert "VERDICT" in buf.getvalue() and "CALIBRATION" in buf.getvalue()
    finally:
        shutil.rmtree(d)



# --------------------------------------------------------------------------
# The deadlock: reachable tenors and tradeable tenors had no overlap
# --------------------------------------------------------------------------

def _prices():
    """A price path scaled to the ETF spot the ladder is built around."""
    return [680.0 * p / PRICES[0] for p in PRICES[:400]]


def _ladder(spot=680.0, dtes=(1, 2, 3, 23, 30, 35), iv=0.16):
    """A realistic ETF book: near-daily expiries plus a monthly, penny quotes.

    The wings matter. At 1-3 DTE a strike 10% out is worth fractions of a cent
    and the zero-bid filter drops it, so rnd._coverage_ok (needs +/-10% cover)
    fails there and only there.
    """
    from engine import pricing
    from engine.data import OptionChain, OptionQuote
    quotes = []
    for dte in dtes:
        t = dte / 365.0
        for k in [round(spot * m, 1) for m in
                  (0.85, 0.88, 0.90, 0.95, 1.0, 1.05, 1.10, 1.12, 1.15)]:
            for kind in ("call", "put"):
                px = pricing.price(spot, k, t, 0.04, 0.0, iv, kind)
                if px < 0.01:                    # the vendor's zero-bid wings
                    continue
                quotes.append(OptionQuote(dte, k, kind,
                                          round(px * 0.995, 2), round(px * 1.005, 2)))
    return OptionChain("SPY", spot, 0.04, 0.0, quotes, asof="2026-08-25")


def test_a_requested_tenor_that_is_not_listed_still_records():
    """THE blocker. rnd._slice matches expiry_days exactly, so `--dte 30`
    against a chain listing 1/2/3/23/35 yielded an empty slice, every Q
    extractor raised, record() returned None and the ledger stayed empty
    forever. The forward test — the only thing that can answer whether this
    engine predicts anything — could never start."""
    chain = _ladder(dtes=(1, 2, 3, 23, 35))
    led = PaperLedger()
    e = led.record(chain, _prices(), 200, BaselineDensityForecaster(), dte=30)
    assert e is not None, "an unlisted tenor still records nothing"
    assert e["dte"] == 35 or e["dte"] == 23, e["dte"]
    assert e["requested_dte"] == 30


def test_the_snap_is_recorded_not_silent():
    """A silent move from 30d to 3d would fill the ledger with entries at a
    tenor nobody chose, every VRP measured against a forecast horizon it does
    not match, and nothing in the file to reveal it afterwards."""
    chain = _ladder(dtes=(1, 2, 3, 23, 35))
    e = PaperLedger().record(chain, _prices(), 200, BaselineDensityForecaster(),
                             dte=30)
    assert e["expiry_snapped"], "the snap left no trace in the entry"
    assert "30d" in e["expiry_snapped"] and str(e["dte"]) in e["expiry_snapped"]

    exact = PaperLedger().record(_ladder(dtes=(30,)), _prices(), 200,
                                 BaselineDensityForecaster(), dte=30)
    assert exact["expiry_snapped"] is None, "it announced a snap that never happened"
    assert exact["requested_dte"] == exact["dte"] == 30


def test_the_ledger_records_whether_the_chain_was_converted():
    """A ledger mixing de-Americanized and raw chains is unanalysable later:
    they are not the same measurement. PREREGISTRATION.md §4.5."""
    chain = _ladder(dtes=(30,))
    plain = PaperLedger().record(chain, _prices(), 200,
                                 BaselineDensityForecaster(), dte=30)
    conv = PaperLedger().record(chain, _prices(), 200,
                                BaselineDensityForecaster(), dte=30, american=True)
    assert plain["de_americanized"] is False
    assert conv["de_americanized"] is True


def test_record_accepts_the_american_flag_that_the_registration_requires():
    """run_live had --american; record did not, so every SPY/QQQ forecast this
    ledger froze violated the repo's own stated rule while looking correct.

    Asserted through the CLI's own help rather than by introspecting a parser
    object: --american lives on the `record` SUBparser, and a check that reads
    the top-level parser passes whether or not the flag was ever wired up.
    """
    import contextlib
    import io
    from tools.paper_trade import main as pt_main

    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            pt_main(["record", "--help"])
    except SystemExit:
        pass
    out = buf.getvalue()
    assert "--american" in out, f"record still has no --american flag:\n{out}"
    assert "--max-expirations" in out, "record cannot widen the expiry window"
    assert "SPY" in out or "ETF" in out, "the flag does not say when it is required"


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
