"""Tests for the buyer's pre-trade check (tools/option_check.py).

The person this is for can only BUY options, in one app, whose screen shows an
implied vol, a strike and a premium. These tests pin that what the tool prints
from those three numbers is arithmetic anyone can check by hand -- and, where it
can be, against the app's own figures: the break-even it prints for IREN's
16-strike call at 2.41 is the 18.41 Dime shows as "ราคาคืนทุน".

Run: python3 tests/test_option_check.py
"""

import io
import math
import os
import random
import sys
import tempfile
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import pricing, volforecast
from tools import option_check as oc


def _gbm(vol, n=600, seed=3, s0=16.0):
    rng = random.Random(seed)
    dv = vol / math.sqrt(252)
    px = [s0]
    for _ in range(n - 1):
        px.append(px[-1] * math.exp(-0.5 * dv * dv + dv * rng.gauss(0, 1)))
    return px


CLOSES = _gbm(0.50)


def test_the_forecast_is_the_engine_forecast_not_a_new_one():
    """One forecaster in the repo. This tool must not grow its own."""
    a = oc.assess(CLOSES, iv=1.0, days=24)
    assert a["forecast_vol"] == volforecast.term_vol(CLOSES, 24 / 365.0)
    assert a["realized_21d"] == volforecast.close_to_close_vol(CLOSES, 21)


def test_a_fifty_percent_stock_reads_its_own_vol_back():
    """A path generated at 50% vol must forecast near 50%, or every ratio printed
    after it is wrong by the same factor."""
    a = oc.assess(CLOSES, iv=0.50, days=24)
    assert 0.40 < a["forecast_vol"] < 0.60, a["forecast_vol"]


def test_the_band_follows_the_ratio():
    """Twice the vol the stock shows is overpriced; half is cheap; equal is fair."""
    f = oc.assess(CLOSES, iv=0.5, days=24)["forecast_vol"]
    assert oc.assess(CLOSES, iv=2.0 * f, days=24)["band"] == "OVERPRICED"
    assert oc.assess(CLOSES, iv=1.0 * f, days=24)["band"] == "FAIR"
    assert oc.assess(CLOSES, iv=0.5 * f, days=24)["band"] == "CHEAP"
    a = oc.assess(CLOSES, iv=1.2, days=24)
    assert abs(a["ratio"] - 1.2 / a["forecast_vol"]) < 1e-12
    assert abs(a["gap_points"] - (1.2 - a["forecast_vol"]) * 100) < 1e-9


def test_the_band_boundaries_are_the_published_ones():
    f = oc.assess(CLOSES, iv=0.5, days=24)["forecast_vol"]
    assert oc.assess(CLOSES, iv=oc.RICH_RATIO * f * 1.0001, days=24)["band"] == "OVERPRICED"
    assert oc.assess(CLOSES, iv=oc.RICH_RATIO * f * 0.9999, days=24)["band"] == "FAIR"
    assert oc.assess(CLOSES, iv=oc.CHEAP_RATIO * f * 0.9999, days=24)["band"] == "CHEAP"
    assert oc.assess(CLOSES, iv=oc.CHEAP_RATIO * f * 1.0001, days=24)["band"] == "FAIR"
    assert (oc.RICH_RATIO, oc.CHEAP_RATIO) == (1.25, 0.85)


def test_the_one_sd_move_is_spot_times_vol_times_root_time():
    a = oc.assess(CLOSES, iv=1.2142, days=24, spot=16.39)
    want = 16.39 * 1.2142 * math.sqrt(24 / 365.0)
    assert abs(a["market_move"] - want) < 1e-12
    assert abs(a["forecast_move"] - 16.39 * a["forecast_vol"] * math.sqrt(24 / 365.0)) < 1e-12


def test_the_break_even_matches_the_app():
    """Dime prints "ราคาคืนทุน 18.41 USD" for IREN's 16-strike call at 2.41, and
    1.81 for EOSE's 1.50 call at 0.31. Strike plus premium, both times."""
    a = oc.assess(CLOSES, iv=1.2142, days=24, spot=16.39, strike=16, premium=2.41)
    assert abs(a["breakeven"] - 18.41) < 1e-9
    assert abs(a["move_needed"] - (18.41 / 16.39 - 1)) < 1e-12
    e = oc.assess(CLOSES, iv=1.1078, days=10, spot=1.71, strike=1.5, premium=0.31)
    assert abs(e["breakeven"] - 1.81) < 1e-9
    p = oc.assess(CLOSES, iv=1.0, days=24, spot=16.39, strike=15, premium=0.80, kind="put")
    assert abs(p["breakeven"] - 14.20) < 1e-9
    assert p["move_needed"] < 0


def test_fair_value_and_overpay_are_the_option_priced_at_the_forecast():
    a = oc.assess(CLOSES, iv=1.2142, days=24, spot=16.39, strike=16, premium=2.41)
    fair = pricing.price(16.39, 16.0, 24 / 365.0, 0.04, 0.0, a["forecast_vol"], "call")
    assert abs(a["fair_at_forecast"] - fair) < 1e-12
    assert abs(a["overpay_per_contract"] - (2.41 - fair) * 100) < 1e-9
    # a forecast well under the quoted IV means the buyer is paying over
    assert a["overpay_per_contract"] > 0


def test_the_probability_is_a_probability_and_points_the_right_way():
    """Beyond the break-even, under a driftless lognormal at the forecast vol."""
    c = oc.assess(CLOSES, iv=1.0, days=24, spot=16.39, strike=16, premium=2.41)
    sig = c["forecast_vol"] * math.sqrt(24 / 365.0)
    d = (math.log(18.41 / 16.39) + 0.5 * sig * sig) / sig
    want = 1 - 0.5 * (1 + math.erf(d / math.sqrt(2)))
    assert abs(c["p_beyond_breakeven"] - want) < 1e-12
    assert 0 < c["p_beyond_breakeven"] < 0.5          # break-even is above spot
    far = oc.assess(CLOSES, iv=1.0, days=24, spot=16.39, strike=30, premium=0.05)
    assert far["p_beyond_breakeven"] < c["p_beyond_breakeven"]
    put = oc.assess(CLOSES, iv=1.0, days=24, spot=16.39, strike=15, premium=0.8, kind="put")
    assert 0 < put["p_beyond_breakeven"] < 0.5


def test_spot_defaults_to_the_last_close():
    assert oc.assess(CLOSES, iv=1.0, days=24)["spot"] == CLOSES[-1]


def test_bad_inputs_are_refused_by_name():
    for kw, needle in ((dict(iv=0.0, days=24), "iv"), (dict(iv=1.0, days=0), "days"),
                       (dict(iv=1.0, days=24, kind="straddle"), "kind"),
                       (dict(iv=1.0, days=24, strike=16), "--price")):
        try:
            oc.assess(CLOSES, **kw)
            assert False, kw
        except ValueError as e:
            assert needle in str(e), (kw, e)
    try:
        oc.assess(CLOSES[:50], iv=1.0, days=24)
        assert False, "a 50-close history must be refused"
    except ValueError as e:
        assert str(oc.MIN_CLOSES) in str(e)


def _write_csv(closes):
    """One distinct date per close. (A first version built dates arithmetically and
    repeated some, so the reader silently kept fewer closes than were written.)"""
    import datetime
    fd, path = tempfile.mkstemp(suffix=".csv")
    d0 = datetime.date(2023, 1, 2)
    with os.fdopen(fd, "w") as fh:
        fh.write("date,close\n")
        for i, c in enumerate(closes):
            fh.write(f"{d0 + datetime.timedelta(days=i)},{c:.6f}\n")
    return path


def test_the_csv_round_trip_keeps_every_close():
    path = _write_csv(CLOSES[-300:])
    try:
        assert oc._closes_from_csv(path) == [round(c, 6) for c in CLOSES[-300:]]
    finally:
        os.remove(path)


def test_the_cli_runs_offline_and_says_what_it_is():
    path = _write_csv(CLOSES[-300:])
    try:
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = oc.main(["IREN", "--iv", "121.42", "--days", "24", "--spot", "16.39",
                          "--strike", "16", "--price", "2.41", "--prices-csv", path])
        out = buf.getvalue()
    finally:
        os.remove(path)
    assert rc == 0, out
    for needle in ("IREN", "18.41", "121.4%", "OVERPRICED", "ไม่ใช่สัญญาณ", "ประกาศงบ"):
        assert needle in out, (needle, out)


def test_the_cli_refuses_a_short_history_instead_of_guessing():
    path = _write_csv(CLOSES[:60])
    try:
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = oc.main(["X", "--iv", "100", "--days", "24", "--prices-csv", path])
    finally:
        os.remove(path)
    assert rc == 1 and "refused" in buf.getvalue()


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
