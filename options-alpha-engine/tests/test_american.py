"""Tests for American pricing + de-Americanization (engine/american.py).

Oracles that pin the numerics:
  * American call, NO dividend == European call (early exercise never optimal).
  * American put > European put (early-exercise premium is strictly positive).
  * American IV round-trips the vol it was priced at.
  * The payoff test: an American chain priced at a KNOWN vol has its model-free
    Q vol biased HIGH when read raw, and de-Americanization recovers the truth.
Run: python3 tests/test_american.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import american, pricing
from engine.data import OptionChain, OptionQuote
from models import rnd

S, R, VOL = 100.0, 0.05, 0.25


def test_american_call_no_dividend_equals_european():
    # With q=0 an American call is never exercised early -> equals BSM European.
    for K in (80, 100, 120):
        am = american.american_price(S, K, 0.5, R, 0.0, VOL, "call", steps=300)
        eu = pricing.price(S, K, 0.5, R, 0.0, VOL, "call")
        assert abs(am - eu) < 0.02, (K, am, eu)


def test_american_put_exceeds_european():
    # A put with positive rates carries a real early-exercise premium.
    am = american.american_price(S, 110, 0.5, R, 0.0, VOL, "put", steps=300)
    eu = pricing.price(S, 110, 0.5, R, 0.0, VOL, "put")
    assert am > eu + 0.02, (am, eu)
    prem = american.early_exercise_premium(S, 110, 0.5, R, 0.0, VOL, "put", steps=300)
    assert prem > 0.0


def test_american_iv_roundtrips():
    for K, kind in ((95, "put"), (105, "call"), (100, "put")):
        px = american.american_price(S, K, 0.4, R, 0.01, 0.30, kind, steps=200)
        iv = american.american_iv(px, S, K, 0.4, R, 0.01, kind, steps=200)
        assert iv is not None and abs(iv - 0.30) < 0.01, (K, kind, iv)


def test_de_americanize_call_no_div_is_noop():
    # No early-exercise premium to strip -> European-equivalent ~ the input price.
    px = american.american_price(S, 100, 0.5, R, 0.0, VOL, "call", steps=200)
    eur, iv = american.de_americanize_price(px, S, 100, 0.5, R, 0.0, "call", steps=200)
    assert abs(eur - px) < 0.02 and abs(iv - VOL) < 0.01


def test_de_americanize_strips_put_premium():
    # European-equivalent put price < the American market price, same vol recovered.
    px = american.american_price(S, 110, 0.5, R, 0.0, VOL, "put", steps=200)
    eur, iv = american.de_americanize_price(px, S, 110, 0.5, R, 0.0, "put", steps=200)
    assert eur < px - 0.02 and abs(iv - VOL) < 0.01


def _american_chain(true_vol, dte, steps):
    """A chain whose every quote is the American fair value at ``true_vol``."""
    T = dte / 365.0
    quotes = []
    for K in range(85, 116, 5):
        for kind in ("call", "put"):
            px = american.american_price(S, float(K), T, R, 0.0, true_vol, kind, steps=steps)
            quotes.append(OptionQuote(dte, float(K), kind, round(px, 4), round(px, 4)))
    return OptionChain("AMER", S, R, 0.0, quotes)


def _european_chain(true_vol, dte):
    """The genuine European chain at ``true_vol`` — the de-Am target."""
    T = dte / 365.0
    quotes = []
    for K in range(85, 116, 5):
        for kind in ("call", "put"):
            px = pricing.price(S, float(K), T, R, 0.0, true_vol, kind)
            quotes.append(OptionQuote(dte, float(K), kind, round(px, 4), round(px, 4)))
    return OptionChain("EURO", S, R, 0.0, quotes)


def test_de_americanization_recovers_true_q_vol():
    # The payoff oracle: read Q vol off a KNOWN-vol American chain raw (biased
    # high by the early-exercise premium) vs after de-Americanization. The strong
    # oracle: de-Am must match what a GENUINE European chain at the same vol reads
    # (that model-free number differs from true_vol only by strike discretization,
    # which is identical for both chains) — i.e. de-Am is exact, not merely closer.
    dte, true_vol = 45, 0.25
    T = dte / 365.0
    chain = _american_chain(true_vol, dte, steps=120)
    euro_floor = rnd.model_free_implied_vol(_european_chain(true_vol, dte), T, dte)
    raw_q = rnd.model_free_implied_vol(chain, T, dte)
    deam_q = rnd.model_free_implied_vol(american.de_americanize_chain(chain, steps=120), T, dte)
    assert raw_q > euro_floor                                # raw is biased HIGH
    assert abs(deam_q - euro_floor) < 1e-3                   # de-Am == the European truth
    assert abs(deam_q - euro_floor) < abs(raw_q - euro_floor)


def test_dividend_call_early_exercise_premium():
    # q > r -> an American CALL carries a real early-exercise premium. Guards the
    # call-side max(continuation, intrinsic): a wrong impl returns the European value.
    am = american.american_price(S, 80, 1.0, 0.02, 0.08, 0.25, "call", steps=300)
    eu = pricing.price(S, 80, 1.0, 0.02, 0.08, 0.25, "call")
    assert am > eu + 0.1, (am, eu)
    assert american.early_exercise_premium(S, 80, 1.0, 0.02, 0.08, 0.25, "call", steps=300) > 0.1


def test_early_exercise_premium_never_negative_no_div_call():
    # Same-tree European cancels discretization -> premium is exactly >= 0, not noise.
    for K in (80, 100, 120):
        prem = american.early_exercise_premium(S, K, 1.0, 0.05, 0.0, 0.25, "call", steps=128)
        assert prem >= 0.0, (K, prem)
    assert american.early_exercise_premium(S, 100, 1.0, 0.05, 0.0, 0.25, "call", steps=128) == 0.0


def test_de_americanize_dividend_chain_recovers_european_floor():
    # The module's stated purpose is dividend ETFs. With q > r the CALL wing carries
    # a real early-exercise premium; de-Am must still recover the European-floor Q vol.
    # A q-passthrough regression (chain.q -> 0) biases the recovered vol and fails here.
    dte, true_vol, r, qy = 45, 0.25, 0.02, 0.06
    T = dte / 365.0
    quotes_a, quotes_e = [], []
    for K in range(85, 116, 5):
        for kind in ("call", "put"):
            pa = american.american_price(S, float(K), T, r, qy, true_vol, kind, steps=120)
            pe = pricing.price(S, float(K), T, r, qy, true_vol, kind)
            quotes_a.append(OptionQuote(dte, float(K), kind, round(pa, 4), round(pa, 4)))
            quotes_e.append(OptionQuote(dte, float(K), kind, round(pe, 4), round(pe, 4)))
    chain_a = OptionChain("USEQ", S, r, qy, quotes_a)
    euro_floor = rnd.model_free_implied_vol(OptionChain("E", S, r, qy, quotes_e), T, dte)
    deam_q = rnd.model_free_implied_vol(american.de_americanize_chain(chain_a, steps=120), T, dte)
    assert abs(deam_q - euro_floor) < 2e-3, (deam_q, euro_floor)


def test_de_americanize_keeps_zero_bid_wing_and_preserves_spread():
    # A zero-bid OTM wing (routine on equity chains) must NOT drop the strike, and
    # the reflowed spread must bracket the European mid.
    dte, T = 30, 30 / 365.0
    px = american.american_price(S, 130.0, T, 0.05, 0.0, 0.25, "call", steps=120)
    quotes = [
        OptionQuote(dte, 100.0, "call", round(px * 0 + 2.0, 4), 2.2),   # normal quote
        OptionQuote(dte, 130.0, "call", 0.0, round(px + 0.05, 4)),      # zero-bid OTM wing
    ]
    deam = american.de_americanize_chain(OptionChain("W", S, 0.05, 0.0, quotes), steps=120)
    strikes = {(q.strike, q.kind) for q in deam.quotes}
    assert (130.0, "call") in strikes                      # wing survives (mid was solvable)
    wing = next(q for q in deam.quotes if q.strike == 130.0)
    assert wing.bid <= wing.mid <= wing.ask and wing.bid >= 0.0
    # original half-spread preserved
    assert abs((wing.ask - wing.bid) - (px + 0.05 - 0.0)) < 1e-3


def test_implied_forward_recovers_dividend_and_borrow():
    # An ETF chain priced at a TRUE carry (div+borrow) but whose static metadata q
    # is WRONG (0). implied_carry must recover the true carry from put-call parity,
    # and de-Am using it must land closer to the true European floor than trusting q.
    dte, r, q_true, vol = 45, 0.05, 0.03, 0.25
    T = dte / 365.0
    a_quotes, e_quotes = [], []
    for K in range(85, 116, 5):
        for kind in ("call", "put"):
            pa = american.american_price(S, float(K), T, r, q_true, vol, kind, steps=120)
            pe = pricing.price(S, float(K), T, r, q_true, vol, kind)
            a_quotes.append(OptionQuote(dte, float(K), kind, round(pa, 4), round(pa, 4)))
            e_quotes.append(OptionQuote(dte, float(K), kind, round(pe, 4), round(pe, 4)))
    chain = OptionChain("ETF", S, r, 0.0, a_quotes)              # metadata q = 0.0 (WRONG)

    _, q_eff = american.implied_carry(chain, dte, T)
    assert abs(q_eff - q_true) < 0.015                          # recovered from parity

    floor = rnd.model_free_implied_vol(OptionChain("E", S, r, q_true, e_quotes), T, dte)
    q_impl = rnd.model_free_implied_vol(
        american.de_americanize_chain(chain, steps=120, use_implied_forward=True), T, dte)
    q_wrong = rnd.model_free_implied_vol(
        american.de_americanize_chain(chain, steps=120, use_implied_forward=False), T, dte)
    assert abs(q_impl - floor) < abs(q_wrong - floor)          # implied forward is better


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
