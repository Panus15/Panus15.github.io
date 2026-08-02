"""Tests for the scheduled-event gate (models/events.py).
Run: python3 tests/test_events.py
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.events import (EventCalendar, MarketEvent, earnings_iv_note,
                           event_gate)

CAL = EventCalendar([
    MarketEvent("2026-02-10", "AAPL", "earnings"),
    MarketEvent("2026-03-18", "", "fomc"),
    MarketEvent("2026-02-12", "MSFT", "earnings"),
    MarketEvent("2026-02-11", "AAPL", "conference"),      # low impact
])


def test_event_inside_the_option_life_fires():
    fired, why = event_gate(CAL, "AAPL", "2026-02-01", 30)
    assert fired and "earnings" in why and "9d" in why


def test_event_after_expiry_does_not_fire():
    fired, why = event_gate(CAL, "AAPL", "2026-02-01", 5)   # window ends 02-06
    assert not fired and "no scheduled events" in why


def test_other_symbols_earnings_do_not_fire():
    fired, _ = event_gate(CAL, "TSLA", "2026-02-01", 30)
    assert not fired                                        # MSFT/AAPL are not TSLA


def test_market_wide_events_apply_to_every_symbol():
    # FOMC has no symbol -> it counts for anyone, but only when high_impact_only=False
    # (fomc is not in HIGH_IMPACT, which is deliberately single-name focused).
    assert not event_gate(CAL, "TSLA", "2026-03-10", 20)[0]
    fired, why = event_gate(CAL, "TSLA", "2026-03-10", 20, high_impact_only=False)
    assert fired and "fomc" in why


def test_low_impact_event_is_filtered_by_default():
    # AAPL has a 'conference' on 02-11 and earnings on 02-10; a window that catches
    # only the conference must not fire under the default high-impact filter.
    fired, _ = event_gate(CAL, "AAPL", "2026-02-11", 0)
    assert not fired
    assert event_gate(CAL, "AAPL", "2026-02-11", 0, high_impact_only=False)[0]


def test_no_calendar_is_honest_about_it():
    fired, why = event_gate(None, "AAPL", "2026-02-01", 30)
    assert not fired
    assert "absence of data" in why                         # not "no events"


def test_undated_snapshot_degrades_safely():
    assert event_gate(CAL, "AAPL", None, 30)[0] is False
    assert event_gate(CAL, "AAPL", "not-a-date", 30)[0] is False


def test_calendar_from_file_and_between():
    rows = [{"date": "2026-05-01", "symbol": "NVDA", "kind": "earnings"},
            {"date": "2026-06-01", "symbol": "NVDA", "kind": "earnings"}]
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(rows, fh)
        path = fh.name
    try:
        cal = EventCalendar.from_file(path)
        assert len(cal.events) == 2
        hits = cal.between("2026-04-01", "2026-05-15", symbol="NVDA")
        assert len(hits) == 1 and hits[0].date == "2026-05-01"
        assert cal.between("2026-04-01", "2026-05-15", symbol="AMD") == []
    finally:
        os.unlink(path)


def test_iv_crush_note_changes_with_timing():
    assert "crush" in earnings_iv_note(0)
    assert "peak" in earnings_iv_note(3)
    assert "build" in earnings_iv_note(20)
    assert earnings_iv_note(None) == ""


# --------------------------------------------------------------------------
# the gate is only worth anything if it reaches a P&L harness
# --------------------------------------------------------------------------

def _dated(n, start="2024-01-02"):
    """ISO dates aligned 1:1 with a price index."""
    import datetime
    d0 = datetime.date.fromisoformat(start)
    return [(d0 + datetime.timedelta(days=i)).isoformat() for i in range(n)]


def test_event_stress_at_builds_a_backtest_veto():
    from models.events import EventCalendar, MarketEvent, event_stress_at

    dates = _dated(400)
    cal = EventCalendar([MarketEvent(dates[100], "X", "earnings")])
    gate = event_stress_at(cal, "X", dates, 21)

    assert gate(90, None) is True, "an earnings date 10d out must veto"
    assert gate(100, None) is True, "the event day itself is inside the window"
    assert gate(50, None) is False, "an event 50d out is beyond a 21d option"
    assert gate(120, None) is False, "the event has passed"
    assert gate(9_999, None) is False, "an undatable bar must not invent a veto"
    # no calendar -> no callback, so callers get the ungated behaviour unchanged
    assert event_stress_at(None, "X", dates, 21) is None


def test_the_event_gate_actually_stops_trades_in_the_backtest():
    """Plumbing tests are cheap; this asserts the gate changes the OUTCOME."""
    from engine.hedged_backtest import price_path_with_crash
    from engine.signal_backtest import run_signal_backtest, synthetic_chain_series
    from models.baseline import BaselineDensityForecaster
    from models.events import EventCalendar, MarketEvent, event_stress_at

    prices = price_path_with_crash(900)
    chains = synthetic_chain_series(dte=21)
    fc = BaselineDensityForecaster()
    dates = _dated(len(prices))

    ungated = run_signal_backtest(prices, chains, fc, dte=21, warmup=63,
                                  always_sell=True)
    # an earnings print every ~6 weeks, the single-name reality
    cal = EventCalendar([MarketEvent(dates[i], "X", "earnings")
                         for i in range(80, len(prices), 42)])
    gate = event_stress_at(cal, "X", dates, 21)
    gated = run_signal_backtest(prices, chains, fc, dte=21, warmup=63,
                                always_sell=True, event_at=gate)

    assert gated.skip_reasons.get("event", 0) > 5, gated.skip_reasons
    assert gated.n_sold < ungated.n_sold, (
        f"the gate skipped {gated.skip_reasons.get('event')} dates but sold "
        f"{gated.n_sold} vs {ungated.n_sold} — it is not reaching entry")
    assert ungated.skip_reasons.get("event", 0) == 0

    # and it binds the ALWAYS-SELL baseline too: exempting the baseline would
    # hand it a free pass and make every benchmark comparison dishonest
    sig = run_signal_backtest(prices, chains, fc, dte=21, warmup=63, event_at=gate)
    assert sig.skip_reasons.get("event", 0) > 0


def test_the_event_gate_reaches_the_hedged_backtest_too():
    from engine.hedged_backtest import price_path_with_crash, run_hedged_backtest
    from models.events import EventCalendar, MarketEvent, event_stress_at

    prices = price_path_with_crash(900)
    dates = _dated(len(prices))
    base = run_hedged_backtest(prices, dte=21, warmup=63)
    cal = EventCalendar([MarketEvent(dates[i], "", "earnings")
                         for i in range(80, len(prices), 42)])
    gated = run_hedged_backtest(prices, dte=21, warmup=63,
                                event_at=event_stress_at(cal, "", dates, 21))
    assert gated.skip_reasons.get("event", 0) > 5, gated.skip_reasons
    assert gated.n_trades < base.n_trades


def test_the_paper_ledger_records_the_event_veto():
    from engine.hedged_backtest import price_path_with_crash
    from engine.signal_backtest import synthetic_chain_series
    from models.baseline import BaselineDensityForecaster
    from models.events import EventCalendar, MarketEvent
    from tools.paper_trade import PaperLedger

    prices = price_path_with_crash(300)
    chains = synthetic_chain_series(dte=21)
    chain = chains(200, prices[:201])
    assert chain is not None
    asof = "2024-06-03"

    led = PaperLedger()
    free = led.record(chain, prices, 200, BaselineDensityForecaster(),
                      dte=21, asof=asof)
    led2 = PaperLedger()
    cal = EventCalendar([MarketEvent("2024-06-12", chain.symbol, "earnings")])
    held = led2.record(chain, prices, 200, BaselineDensityForecaster(),
                       dte=21, asof=asof, calendar=cal)

    assert free is not None and held is not None
    assert held["regime_stressed"] is True
    assert held["traded"] is False, "a dated earnings print must veto the entry"
    assert "earnings" in (held["gate_reason"] or "")
    # an event OUTSIDE the option's life must not veto anything
    led3 = PaperLedger()
    far = led3.record(chain, prices, 200, BaselineDensityForecaster(), dte=21,
                      asof=asof, calendar=EventCalendar(
                          [MarketEvent("2025-01-01", chain.symbol, "earnings")]))
    assert far["traded"] == free["traded"]


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
