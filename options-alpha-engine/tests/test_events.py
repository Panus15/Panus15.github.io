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
