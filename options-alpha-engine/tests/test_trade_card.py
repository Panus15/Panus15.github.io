"""Tests for the trade card (models/trade_card.py).

The card must DERIVE every number (levels are density quantiles, probabilities
come from that same density) and must never quietly present the unvalidated
directional leg as if it were the validated variance one.
Run: python3 tests/test_trade_card.py
"""

import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import pricing
from engine.data import OptionChain, OptionQuote
from models.baseline import BaselineDensityForecaster
from models.trade_card import build_card

DTE, R, Q = 30, 0.03, 0.0
T = DTE / 365.0
F = BaselineDensityForecaster()

_rng = random.Random(4)
_walk = [100.0]
for _ in range(200):
    _walk.append(_walk[-1] * math.exp(_rng.gauss(0.0, 0.013)))
PRICES = [round(x * 100.0 / _walk[-1], 4) for x in _walk]      # end at 100
SPOT = PRICES[-1]


def _chain(iv=0.26, spot=SPOT, strikes=range(70, 131, 5), min_px=0.02):
    quotes = []
    for k in strikes:
        for kind in ("call", "put"):
            px = pricing.price(spot, float(k), T, R, Q, iv, kind)
            if px < min_px:
                continue
            quotes.append(OptionQuote(DTE, float(k), kind, round(px * 0.99, 4),
                                      round(px * 1.01, 4)))
    return OptionChain("TEST", spot, R, Q, quotes, asof="2026-07-26")


def test_card_reports_both_legs_and_renders():
    card = build_card(_chain(), F, PRICES, dte=DTE)
    txt = card.render()
    assert "VOLATILITY" in txt and "DIRECTION" in txt
    assert "[validated by this repo]" in txt
    assert card.vol_side in ("SELL VOL", "BUY VOL", "NO TRADE")
    assert card.direction in ("LONG", "SHORT", "NEUTRAL")
    assert card.direction_is_validated is False        # never claim otherwise


def test_rich_market_says_sell_vol_and_cheap_says_buy():
    # P vol here is ~0.21, so 0.40 is a rich market and 0.14 a cheap one. (Going
    # much below that prices the wings to ~0, the chain thins out and Q extraction
    # correctly refuses — covered by test_sparse_chain_refuses_rather_than_guesses.)
    rich = build_card(_chain(iv=0.40), F, PRICES, dte=DTE)
    # A cheap market's +-10% wings really are worth ~a cent, so the fixture must
    # quote them (min_px) or the coverage guard correctly refuses the chain --
    # which is what test_sparse_chain_refuses_rather_than_guesses covers.
    cheap = build_card(_chain(iv=0.14, strikes=range(85, 116, 2), min_px=1e-6),
                       F, PRICES, dte=DTE)
    assert rich.vol_side == "SELL VOL" and rich.vrp > 0
    assert cheap.vol_side == "BUY VOL" and cheap.vrp < 0
    assert rich.q_vol > rich.p_vol > cheap.q_vol


def test_levels_are_density_quantiles_not_invented():
    card = build_card(_chain(), F, PRICES, dte=DTE, direction_threshold=-1.0)
    assert card.direction in ("LONG", "SHORT")                 # forced non-neutral
    dist = F.forecast(PRICES, T, r=R, q=Q, spot=SPOT)
    if card.direction == "LONG":
        assert abs(card.target - dist.quantile(0.75)) < 1e-6
        assert abs(card.stop - dist.quantile(0.25)) < 1e-6
        assert card.target > card.entry > card.stop
        assert abs(card.p_target - dist.prob_above(card.target)) < 1e-9
    else:
        assert abs(card.target - dist.quantile(0.25)) < 1e-6
        assert abs(card.stop - dist.quantile(0.75)) < 1e-6
        assert card.target < card.entry < card.stop
        assert abs(card.p_target - dist.cdf(card.target)) < 1e-9
    # EV is exactly P(win)*reward - P(lose)*risk, and R:R is derived
    assert abs(card.expected_value
               - (card.p_target * card.reward - card.p_stop * card.risk)) < 1e-9
    assert abs(card.reward_risk - card.reward / card.risk) < 1e-9


def test_stressed_regime_blocks_the_vol_leg():
    wild = PRICES[:-6] + [PRICES[-6] * m for m in (0.93, 1.08, 0.94, 1.07, 0.92, 1.09)]
    card = build_card(_chain(iv=0.40), F, wild, dte=DTE)
    assert card.vol_side == "NO TRADE" and "regime" in card.vol_reason


def test_sparse_chain_refuses_rather_than_guesses():
    thin = OptionChain("THIN", SPOT, R, Q, [
        OptionQuote(DTE, 100.0, "call", 1.0, 1.1),
        OptionQuote(DTE, 100.0, "put", 1.0, 1.1)], asof="2026-07-26")
    card = build_card(thin, F, PRICES, dte=DTE)
    assert card.vol_side == "NO TRADE"
    assert card.warnings                                       # says why


def test_calibration_bias_becomes_a_warning():
    class Rep:
        verdict = "TOO NARROW — outcomes land in the tails too often"
        vol_scale = 1.6
    card = build_card(_chain(iv=0.40), F, PRICES, dte=DTE, calibration=Rep())
    assert "TOO NARROW" in card.calibration_note
    assert any("OVERSTATED" in w for w in card.warnings)
    assert "TRUST" in card.render()


def test_directional_leg_always_carries_its_health_warning():
    card = build_card(_chain(), F, PRICES, dte=DTE, direction_threshold=-1.0)
    assert card.direction != "NEUTRAL"
    assert any("unvalidated" in w.lower() for w in card.warnings)
    assert "UNVALIDATED" in card.render()


def test_scheduled_event_blocks_the_vol_leg():
    # THE single-stock trap: before earnings, implied vol is high for a REASON.
    # The engine would otherwise see a huge VRP and shout RICH.
    from models.events import EventCalendar, MarketEvent
    rich = _chain(iv=0.40)                                   # market >> forecast
    assert build_card(rich, F, PRICES, dte=DTE).vol_side == "SELL VOL"

    cal = EventCalendar([MarketEvent("2026-08-05", "TEST", "earnings")])
    gated = build_card(rich, F, PRICES, dte=DTE, calendar=cal)
    assert gated.vol_side == "NO TRADE"
    assert "earnings" in gated.vol_reason
    assert any("scheduled jump" in w for w in gated.warnings)

    # an event AFTER expiry must not block it
    late = EventCalendar([MarketEvent("2027-01-01", "TEST", "earnings")])
    assert build_card(rich, F, PRICES, dte=DTE, calendar=late).vol_side == "SELL VOL"


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
