"""Tests for American pricing + de-Americanization (engine/american.py).

Oracles that pin the numerics:
  * American call, NO dividend == European call (early exercise never optimal).
  * American put > European put (early-exercise premium is strictly positive).
  * American IV round-trips the vol it was priced at.
  * The payoff test: an American chain priced at a KNOWN vol has its model-free
    Q vol biased HIGH when read raw, and de-Americanization recovers the truth.
Run: python3 tests/test_american.py
"""

import math
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


def test_the_probability_clamp_keeps_a_low_vol_dividend_tree_sane():
    """CRR's risk-neutral p leaves [0, 1] when the carry outruns the vol over one
    step -- e^{(r-q)dt} outside [1/u, u], i.e. vol < |r-q|*sqrt(dt).

    That is not exotic: a high-dividend ETF quoted at a low vol. At q = 0.30 and
    vol = 0.01 the raw p is -0.281, and a negative probability compounded through
    128 backward-induction layers prices this put at 659089 instead of 7.50. The
    clamp is the only thing between that and a chain of quotes.
    """
    dt = 0.5 / 128
    u = math.exp(0.01 * math.sqrt(dt))
    p_raw = (math.exp((R - 0.30) * dt) - 1.0 / u) / (u - 1.0 / u)
    assert p_raw < 0.0, p_raw                              # measured -0.281
    px = american.american_price(S, 100.0, 0.5, R, 0.30, 0.01, "put", steps=128)
    assert 7.0 < px < 8.0, px                              # measured 7.49854
    # and it stays a price, not a number that merely happens to be finite
    assert px >= max(100.0 - S, 0.0)
    assert px <= 100.0


def test_expired_option_is_worth_intrinsic_never_a_negative_number():
    """T <= 0 short-circuits to the payoff, and the payoff has a floor. Without it
    an expired out-of-the-money option is worth MINUS its distance from the
    strike, which flows straight into a chain as a negative price."""
    assert american.american_price(S, 120.0, 0.0, R, 0.0, 0.25, "call") == 0.0
    assert american.american_price(S, 80.0, 0.0, R, 0.0, 0.25, "put") == 0.0
    assert american.american_price(S, 80.0, 0.0, R, 0.0, 0.25, "call") == 20.0
    assert american.american_price(S, 120.0, 0.0, R, 0.0, 0.25, "put") == 20.0


def test_a_quote_too_cheap_to_solve_is_refused_not_guessed():
    """The vol bracket is [0.01, 5.0], so a quote below the price at 0.01 is not
    solvable -- a stale or crossed market, routine enough. Without the bracket
    check the bisection walks to the top of the range and returns 5.0, a 500% vol
    reported as a measurement.

    Measured: the cheapest solvable ATM call here is 2.469, so 0.001 is three
    orders of magnitude outside the bracket.
    """
    assert american.american_price(S, 100.0, 0.5, R, 0.0, 0.01, "call", steps=100) > 2.0
    assert american.american_iv(0.001, S, 100.0, 0.5, R, 0.0, "call", steps=100) is None
    # a solvable quote at the same strike still solves, so this is not a blanket refusal
    px = american.american_price(S, 100.0, 0.5, R, 0.0, 0.25, "call", steps=100)
    iv = american.american_iv(px, S, 100.0, 0.5, R, 0.0, "call", steps=100)
    assert iv is not None and abs(iv - 0.25) < 0.01


def test_implied_carry_is_discounted_and_anchored_near_spot():
    """Two details of `implied_carry` that only show up on chains chosen for them.

    F = K + e^{rT}(C - P): dropping the discount factor costs little at 45 days
    and 1.55x the error at a year (0.01294 against 0.00833 at r = 8%).

    The anchor strike matters in the other direction. Parity is exact for
    European options and only approximately right for American ones, because the
    put's early-exercise premium is not the call's -- a bias that grows away from
    the money. On a chain with q > r, where BOTH wings carry premium, anchoring on
    the furthest strike instead of the nearest takes the error from -0.00323 to
    -0.03369, an order of magnitude.
    """
    for dte, r, q_true, tol in ((365, 0.08, 0.01, 0.010), (45, 0.02, 0.06, 0.010)):
        T = dte / 365.0
        quotes = []
        for K in range(85, 116, 5):
            for kind in ("call", "put"):
                px = american.american_price(S, float(K), T, r, q_true, 0.25, kind, steps=120)
                quotes.append(OptionQuote(dte, float(K), kind, round(px, 4), round(px, 4)))
        _, q_eff = american.implied_carry(OptionChain("E", S, r, 0.0, quotes), dte, T)
        assert abs(q_eff - q_true) < tol, (dte, r, q_true, q_eff)


def test_a_forward_that_cannot_be_positive_is_refused():
    """Parity on a broken quote can imply a negative forward -- here a 50 strike
    with the put at 100 against a 0.10 call gives F = -50.52. q_eff would then be
    the log of a negative number; this has to refuse before that."""
    broken = OptionChain("B", S, 0.05, 0.0,
                         [OptionQuote(45, 50.0, "call", 0.1, 0.1),
                          OptionQuote(45, 50.0, "put", 100.0, 100.0)])
    assert american.implied_carry(broken, 45, 45 / 365.0) is None


def test_the_implied_forward_is_on_by_default():
    """Every existing caller passed `use_implied_forward` explicitly, so the
    shipped default was never exercised and could have been either value."""
    dte, r, q_true, vol = 45, 0.05, 0.03, 0.25
    T = dte / 365.0
    a_quotes, e_quotes = [], []
    for K in range(85, 116, 5):
        for kind in ("call", "put"):
            pa = american.american_price(S, float(K), T, r, q_true, vol, kind, steps=120)
            pe = pricing.price(S, float(K), T, r, q_true, vol, kind)
            a_quotes.append(OptionQuote(dte, float(K), kind, round(pa, 4), round(pa, 4)))
            e_quotes.append(OptionQuote(dte, float(K), kind, round(pe, 4), round(pe, 4)))
    chain = OptionChain("ETF", S, r, 0.0, a_quotes)         # metadata q = 0.0 (WRONG)
    floor = rnd.model_free_implied_vol(OptionChain("E", S, r, q_true, e_quotes), T, dte)
    default = rnd.model_free_implied_vol(
        american.de_americanize_chain(chain, steps=120), T, dte)        # no keyword
    static = rnd.model_free_implied_vol(
        american.de_americanize_chain(chain, steps=120, use_implied_forward=False), T, dte)
    assert abs(default - floor) < abs(static - floor), (default, static, floor)


def test_carry_falls_back_to_the_chains_own_q_not_to_zero():
    """With no put at an expiry, parity has nothing to anchor on and the carry
    falls back to the chain's metadata. Falling back to q = 0 instead silently
    prices a dividend ETF as though it paid nothing -- and with q > r that is
    exactly where the CALL wing's early-exercise premium lives, so the premium
    this module exists to strip would be left in place.

    Measured on a calls-only q = 8% chain: the real fallback takes K=90 from
    10.1333 to 9.7901, which is the true European value 9.79003 to 7e-05.
    """
    T = 45 / 365.0
    quotes = []
    for K in (90.0, 100.0, 110.0):
        px = american.american_price(S, K, T, 0.02, 0.08, 0.25, "call", steps=120)
        quotes.append(OptionQuote(45, K, "call", round(px, 4), round(px, 4)))
    chain = OptionChain("C", S, 0.02, 0.08, quotes)
    assert american.implied_carry(chain, 45, T) is None     # no put to anchor on
    deam = american.de_americanize_chain(chain, steps=120)
    by_strike = {q.strike: q.mid for q in deam.quotes}
    for K in (90.0, 100.0, 110.0):
        euro = pricing.price(S, K, T, 0.02, 0.08, 0.25, "call")
        assert abs(by_strike[K] - euro) < 1e-3, (K, by_strike[K], euro)
    # the premium really was there to strip, so this is not a no-op either way
    assert by_strike[90.0] < 10.1333 - 0.3


def test_a_zero_bid_put_does_not_reflow_to_a_negative_bid():
    """The half-spread is reflowed around the EUROPEAN mid, which is lower than
    the American one -- so for a wide zero-bid quote the lower edge goes through
    zero. Measured: a deep ITM put quoted 0.00 / 60.00 has a mid of 30, a half
    spread of 30, and a European mid of 26.90, so the reflowed bid is -3.10
    without the floor. A negative bid is not a price."""
    T = 0.5
    chain = OptionChain("Z", S, R, 0.0, [OptionQuote(182, 130.0, "put", 0.0, 60.0)])
    deam = american.de_americanize_chain(chain, steps=120)
    assert len(deam.quotes) == 1
    wing = deam.quotes[0]
    assert wing.bid == 0.0, wing.bid                        # floored, not -3.10
    assert wing.bid <= wing.mid <= wing.ask
    eur, _ = american.de_americanize_price(30.0, S, 130.0, T, R, 0.0, "put", steps=120)
    assert eur - 30.0 < -1.0                                # the unfloored edge was negative


def test_a_broken_quote_is_dropped_not_passed_through_raw():
    """A mid below intrinsic is a broken quote, and `de_americanize_price` returns
    None for it. Passing the original American quote through instead puts an
    un-corrected price into a chain labelled European -- the one input `rnd` is
    entitled to assume has been corrected."""
    T = 0.5
    assert american.de_americanize_price(20.0, S, 130.0, T, R, 0.0, "put", steps=120) is None
    good = american.american_price(S, 95.0, T, R, 0.0, 0.25, "put", steps=120)
    chain = OptionChain("X", S, R, 0.0, [
        OptionQuote(182, 130.0, "put", 20.0, 20.0),         # below intrinsic: broken
        OptionQuote(182, 95.0, "put", round(good, 4), round(good, 4)),
    ])
    deam = american.de_americanize_chain(chain, steps=120)
    strikes = {q.strike for q in deam.quotes}
    assert 130.0 not in strikes, strikes
    assert 95.0 in strikes                                  # the sound quote survives


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
