"""Tests for the macro regime gate (models/macro.py) + its composition.
Run: python3 tests/test_macro.py
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.macro import FileMacroAdapter, MacroSnapshot, macro_regime_gate
from models.news_signal import event_risk


def _calm():
    return MacroSnapshot("2026-01-02", short_rate=0.03, long_rate=0.042,
                         credit_spread=0.03, rate_change_1m=0.0, vix=15.0, vix_3m=17.0)


def test_calm_macro_does_not_fire():
    fired, reason = macro_regime_gate(_calm())
    assert not fired and "benign" in reason


def test_inverted_curve_fires():
    snap = MacroSnapshot("2026-01-02", short_rate=0.05, long_rate=0.04)  # -100bp
    fired, reason = macro_regime_gate(snap)
    assert fired and "inverted" in reason


def test_wide_credit_fires():
    snap = MacroSnapshot("2026-01-02", short_rate=0.03, long_rate=0.05, credit_spread=0.08)
    fired, reason = macro_regime_gate(snap)
    assert fired and "credit" in reason


def test_vix_backwardation_fires():
    # near VIX well above 3m VIX = acute near-term stress.
    snap = MacroSnapshot("2026-01-02", short_rate=0.03, long_rate=0.05, vix=35.0, vix_3m=28.0)
    assert snap.vix_term_slope == -7.0
    fired, reason = macro_regime_gate(snap)
    assert fired and "backwardation" in reason


def test_rapid_tightening_fires():
    snap = MacroSnapshot("2026-01-02", short_rate=0.05, long_rate=0.06, rate_change_1m=0.0075)
    fired, reason = macro_regime_gate(snap)
    assert fired and "tightening" in reason


def test_curve_and_term_slope_math():
    snap = MacroSnapshot("x", short_rate=0.05, long_rate=0.04, vix=20.0, vix_3m=22.0)
    assert abs(snap.curve_slope - (-0.01)) < 1e-12
    assert snap.vix_term_slope == 2.0
    assert MacroSnapshot("x").vix_term_slope is None       # missing VIX -> None


def test_event_risk_composes_macro():
    # price calm, no news, but inverted curve -> event_risk fires with a macro reason.
    inverted = MacroSnapshot("2026-01-02", short_rate=0.05, long_rate=0.04)
    fired, reason = event_risk(False, None, macro=inverted)
    assert fired and reason.startswith("macro:")
    # calm macro + calm price + no news -> no veto
    assert event_risk(False, None, macro=_calm())[0] is False
    # price regime dominates (checked first)
    assert event_risk(True, None, macro=_calm()) == (True, "realised vol accelerating (price regime)")


def test_file_macro_adapter_is_point_in_time():
    rows = [
        {"asof": "2026-01-01", "short_rate": 0.03, "long_rate": 0.045},
        {"asof": "2026-02-01", "short_rate": 0.05, "long_rate": 0.041},   # inverting
        {"asof": "2026-03-01", "short_rate": 0.05, "long_rate": 0.038},
    ]
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(rows, fh)
        path = fh.name
    try:
        ad = FileMacroAdapter(path)
        assert ad.at("2026-01-15").asof == "2026-01-01"    # most recent on/before
        assert ad.at("2026-02-15").asof == "2026-02-01"
        assert ad.at("2025-12-01") is None                 # nothing yet
        assert ad.latest().asof == "2026-03-01"
    finally:
        os.unlink(path)


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
