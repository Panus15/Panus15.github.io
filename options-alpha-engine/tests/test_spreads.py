"""Tests for defined-risk spread construction (models/spreads.py).

Oracle: price a chain from the forecaster's OWN density (r=0, zero spread) -> no
structure can be +EV beyond costs (no free lunch against yourself). Make the
market richer (implied > model) and the condor turns positive. Risk is always
capped. Run: python3 tests/test_spreads.py
"""

import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.data import OptionChain, OptionQuote
from models.baseline import BaselineDensityForecaster
from models.spreads import iron_condor, put_credit_spread, scan_spreads

R, Q, DTE = 0.0, 0.0, 30
T = DTE / 365.0
F = BaselineDensityForecaster()
# ~24% annualised vol so the OTM wings actually carry value (a flat path prices
# them to ~0 and there is no condor to build).
_rng = random.Random(3)
_walk = [100.0]
for _ in range(90):
    _walk.append(_walk[-1] * math.exp(_rng.gauss(0.0, 0.015)))
_scale = 100.0 / _walk[-1]                      # pin the last price to 100 (vol unchanged)
PRICES = [round(p * _scale, 4) for p in _walk]
S = PRICES[-1]


def _chain(mult=1.0, spread=0.0):
    """Chain priced from the model density, scaled by ``mult`` (market richness)."""
    p = F.forecast(PRICES, T, r=R, q=Q, spot=S)
    quotes = []
    for k in range(60, 141, 5):
        K = float(k)
        for kind in ("call", "put"):
            fair = p.price(K, R, T, kind) * mult
            if fair < 0.02:
                continue
            quotes.append(OptionQuote(DTE, K, kind, round(max(fair - spread / 2, 0.01), 4),
                                      round(fair + spread / 2, 4)))
    return OptionChain("X", S, R, Q, quotes)


def test_iron_condor_structure_and_capped_risk():
    ic = iron_condor(_chain(), F, PRICES, dte=DTE, body=0.05, wing=0.05, r=R, q=Q)
    assert ic is not None and len(ic.legs) == 4
    shorts = [l for l in ic.legs if l.side == "short"]
    longs = [l for l in ic.legs if l.side == "long"]
    assert len(shorts) == 2 and len(longs) == 2
    # long wings are further OTM than the shorts they protect
    sp = next(l for l in shorts if l.kind == "put"); lp = next(l for l in longs if l.kind == "put")
    sc = next(l for l in shorts if l.kind == "call"); lc = next(l for l in longs if l.kind == "call")
    assert lp.strike < sp.strike and lc.strike > sc.strike
    # risk is DEFINED: max loss = widest wing - credit + commissions, and finite/positive
    wing = max(sp.strike - lp.strike, lc.strike - sc.strike)
    assert abs(ic.max_loss - (wing - ic.net_credit + 0.65 / 100 * 4)) < 1e-6
    assert 0 < ic.max_loss < wing
    assert 0.0 <= ic.prob_profit <= 1.0
    assert ic.break_evens[0] < ic.break_evens[1]


def test_no_free_lunch_when_priced_from_the_model():
    ic = iron_condor(_chain(mult=1.0), F, PRICES, dte=DTE, r=R, q=Q)
    # self-priced (r=0) -> credit == expected loss under P, so EV is just -commissions
    assert -0.1 < ic.ev < 0.0


def test_rich_market_makes_the_condor_positive_ev():
    fair = iron_condor(_chain(mult=1.0), F, PRICES, dte=DTE, r=R, q=Q)
    rich = iron_condor(_chain(mult=1.15), F, PRICES, dte=DTE, r=R, q=Q)
    assert rich.ev > fair.ev                       # richer implied vol -> more edge
    assert rich.ev > 0.0                           # and now genuinely +EV vs the P model


def test_put_credit_spread_two_legs_and_capped():
    pcs = put_credit_spread(_chain(mult=1.12), F, PRICES, dte=DTE, r=R, q=Q)
    assert pcs is not None and len(pcs.legs) == 2
    assert all(l.kind == "put" for l in pcs.legs)
    assert 0 < pcs.max_loss < 1e9 and pcs.max_gain == round(pcs.net_credit - 0.65 / 100 * 2, 4)
    assert 0.0 <= pcs.prob_profit <= 1.0


def test_scan_ranks_by_ev_and_line_renders():
    board = scan_spreads(_chain(mult=1.12), F, PRICES, dte=DTE)
    assert len(board) == 2
    assert board[0].ev >= board[1].ev
    assert "EV=" in board[0].line() and "maxL=" in board[0].line()


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
