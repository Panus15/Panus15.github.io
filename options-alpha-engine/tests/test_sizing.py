"""Tests for position sizing (engine/sizing.py).

This file exists because the module did not have one, and that is precisely how a
2% risk cap came to authorise a position with a 144,000% maximum loss. 391 tests
passed around it; none of them asked the sizer a question.

The property under test is not "does it return a plausible number". It is that the
cap is a THEOREM: for every input, the worst case the sizer authorises is at or
under the budget it was given. A cap that is merely usually right is a label.

Run: python3 tests/test_sizing.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.sizing import (UNBOUNDED, kelly_fraction, max_loss_per_contract_for,
                           position_size)

EQ = 100_000.0
MULT = 100


def test_the_cap_holds_for_every_short_put_it_will_size():
    """The headline property, swept rather than spot-checked.

    The old sizer divided the budget by the PREMIUM, so the further out of the
    money the strike, the cheaper the option, and the larger the position it
    authorised — a cap that loosened exactly as the tail got fatter.
    """
    for strike in (100.0, 300.0, 650.0, 688.0, 700.0, 720.0, 1000.0):
        for premium in (0.01, 0.05, 0.40, 1.10, 5.00, 25.0):
            loss = max_loss_per_contract_for("put", strike, quantity=-1,
                                             entry_price=premium)
            n = position_size(EQ, max_loss_per_contract=loss, max_risk_frac=0.02)
            worst = n * loss
            assert worst <= 0.02 * EQ + 1e-9, (
                f"strike {strike} premium {premium}: {n} contracts risks "
                f"{worst:,.0f} against a budget of {0.02 * EQ:,.0f}")


def test_a_cheaper_option_never_buys_a_bigger_position():
    """The perverse incentive, pinned directly. Monotonicity in the LOSS is the
    invariant; the premium must not be able to invert it."""
    strike = 700.0
    sizes = [position_size(EQ, max_loss_per_contract=max_loss_per_contract_for(
        "put", strike, quantity=-1, entry_price=p)) for p in (0.01, 0.05, 0.40, 1.10)]
    assert sizes == sorted(sizes), (
        f"a cheaper option bought a bigger position: {sizes}")


def test_a_hundred_thousand_dollar_account_cannot_sell_a_naked_index_put():
    """Uncomfortable, correct, and previously hidden.

    One SPY put at 688 can lose 688 x 100 = $68,800. Against a $2,000 budget the
    honest answer is zero contracts, not eighteen. This test exists so that answer
    cannot quietly become non-zero again.
    """
    loss = max_loss_per_contract_for("put", 688.0, quantity=-1, entry_price=1.10)
    assert loss > 60_000
    assert position_size(EQ, max_loss_per_contract=loss, max_risk_frac=0.02) == 0


def test_a_naked_short_call_is_refused_rather_than_guessed():
    """An unbounded loss has nothing for a cap to bind on. Inventing a worst case
    and sizing against it is how a risk system becomes decoration."""
    loss = max_loss_per_contract_for("call", 800.0, quantity=-1, entry_price=1.5)
    assert loss == UNBOUNDED
    assert position_size(EQ, max_loss_per_contract=loss) == 0


def test_a_wing_makes_a_short_call_sizeable_and_the_width_sets_the_size():
    """The only honest route to selling calls here: cap the loss structurally."""
    narrow = max_loss_per_contract_for("call", 800.0, quantity=-1,
                                       entry_price=1.5, wing_strike=810.0)
    wide = max_loss_per_contract_for("call", 800.0, quantity=-1,
                                     entry_price=1.5, wing_strike=850.0)
    assert narrow == (10.0 * MULT - 1.5 * MULT)
    assert wide > narrow
    n_narrow = position_size(EQ, max_loss_per_contract=narrow)
    n_wide = position_size(EQ, max_loss_per_contract=wide)
    assert n_narrow > 0 and n_narrow * narrow <= 0.02 * EQ
    assert n_wide < n_narrow, (n_narrow, n_wide)
    # a 50-wide spread risks $4,850 a contract against a $2,000 budget, so zero
    # is the correct answer and not a bug - the width IS the risk
    assert n_wide == 0 and wide > 0.02 * EQ


def test_a_long_option_risks_only_what_it_paid():
    """The one case the old formula got right, kept right."""
    for kind in ("put", "call"):
        loss = max_loss_per_contract_for(kind, 700.0, quantity=+1, entry_price=3.25)
        assert loss == 3.25 * MULT
    n = position_size(EQ, max_loss_per_contract=3.25 * MULT)
    assert n == int(2_000 // 325)


def test_the_premium_kept_reduces_the_loss_but_never_below_zero():
    deep = max_loss_per_contract_for("put", 10.0, quantity=-1, entry_price=99.0)
    assert deep == 0.0, "a premium larger than the strike must floor at zero, not go negative"
    normal = max_loss_per_contract_for("put", 700.0, quantity=-1, entry_price=2.0)
    assert normal == 700.0 * MULT - 2.0 * MULT


def test_a_loss_cannot_be_passed_positionally_where_a_premium_was():
    """The substitution that caused the defect is now a TypeError, not a number.

    The old signature was position_size(equity, contract_price, ...). Anyone
    porting that call by hand must be stopped by the interpreter rather than
    handed a 144,000% position.
    """
    try:
        position_size(EQ, 1.10)                      # noqa: the old call shape
    except TypeError:
        pass
    else:
        raise AssertionError("the old premium-positional call still works")


def test_degenerate_inputs_return_zero_not_an_exception_or_a_number():
    assert position_size(0.0, max_loss_per_contract=100.0) == 0
    assert position_size(-5.0, max_loss_per_contract=100.0) == 0
    assert position_size(EQ, max_loss_per_contract=0.0) == 0
    assert position_size(EQ, max_loss_per_contract=-100.0) == 0
    assert position_size(EQ, max_loss_per_contract=100.0, max_risk_frac=0.0) == 0


def test_kelly_is_still_correct_about_its_own_question():
    """Kept, unused by the sizer, and still pinned: it is right about binary bets
    and the reason it was removed is that a short option is not one."""
    assert kelly_fraction(0.6, 1.0) == 0.19999999999999996 or abs(
        kelly_fraction(0.6, 1.0) - 0.2) < 1e-12
    assert kelly_fraction(0.4, 1.0) == 0.0, "a negative edge must not bet"
    assert kelly_fraction(0.9, 0.0) == 0.0, "zero odds is not a bet"
    assert kelly_fraction(0.5, 2.0) == 0.25


def test_the_sizer_no_longer_consults_kelly():
    """Not a style point. min(kelly, cap) let a two-outcome formula move the size
    of a payoff it cannot represent; the cap must be the only thing that binds."""
    import inspect

    from engine import sizing
    src = inspect.getsource(sizing.position_size)
    assert "kelly" not in src.lower(), src



def test_the_stated_risk_fraction_is_honoured_exactly_not_clamped():
    """If the caller says 5%, the sizer must use 5%.

    A hidden clamp is defensible-looking and dishonest: the operator reads the
    number they passed, the sizer uses a different one, and the gap only shows up
    as an unexplained difference between the cap they set and the loss they took.
    Being silently MORE conservative is still lying about the budget.
    """
    loss = 250.0
    for frac in (0.005, 0.01, 0.02, 0.05, 0.10, 0.25, 0.60):
        n = position_size(EQ, max_loss_per_contract=loss, max_risk_frac=frac)
        assert n == int((frac * EQ) // loss), (
            f"asked for {frac:.1%} of equity, sized as if it were something else")


def test_the_unbounded_refusal_is_explicit_not_an_accident_of_float_division():
    """int(budget // inf) happens to be 0, so the guard looks redundant. It is
    not: the refusal is a stated rule, and anything later added between the
    budget and the division - a floor, a minimum lot - would silently turn an
    unsizeable naked short into a position. Pinned as intent."""
    import inspect

    from engine import sizing
    src = inspect.getsource(sizing.position_size)
    assert "UNBOUNDED" in src, "the explicit refusal was removed"
    assert position_size(EQ, max_loss_per_contract=UNBOUNDED) == 0


# ---------------------------------------------------------------------------
# Gaps the first mutation sweep of this module found.
# ---------------------------------------------------------------------------

def test_a_wing_caps_a_short_PUT_spread_as_well_as_a_short_call():
    # Every wing test here used a short CALL, whose wing sits ABOVE the strike,
    # so `abs(wing - strike)` was never doing anything. On a short PUT the wing
    # sits BELOW: without the abs the width goes negative, the loss collapses
    # to zero, and position_size then refuses a perfectly ordinary spread.
    # Sell the 100 put, buy the 95: five points of width, $500 a contract.
    loss = max_loss_per_contract_for("put", 100.0, quantity=-1,
                                     entry_price=1.00, wing_strike=95.0)
    assert abs(loss - 400.0) < 1e-9, (
        f"{loss}: expected 5 points x 100 = 500 less the 100 premium kept")
    n = position_size(100_000.0, max_loss_per_contract=loss,
                      max_risk_frac=0.02)
    assert n == 5, f"{n} contracts: 2% of 100k is 2,000, and 2,000/400 is 5"


def test_a_kind_that_is_not_put_or_call_is_refused_not_guessed():
    # Without the check a typo takes the long branch and returns the premium --
    # a plausible-looking number for a structure the function does not model.
    for bad in ("banana", "PUTT", "", "p"):
        try:
            max_loss_per_contract_for(bad, 100.0, quantity=-1, entry_price=1.0)
        except ValueError as e:
            assert "put" in str(e) and "call" in str(e), e
        else:
            assert False, f"kind {bad!r} was accepted"


def test_the_kind_is_read_case_insensitively():
    # Upper case is what a chain feed hands you. Dropping the `.lower()` turns
    # every one of those into the refusal above.
    for ok in ("PUT", "Put", "pUt"):
        loss = max_loss_per_contract_for(ok, 100.0, quantity=-1,
                                         entry_price=1.00)
        assert abs(loss - 9900.0) < 1e-9, f"{ok}: got {loss}"
    assert max_loss_per_contract_for("CALL", 100.0, quantity=-1,
                                     entry_price=1.0) == UNBOUNDED


def test_kelly_never_returns_a_negative_bet_or_divides_by_zero_odds():
    # A negative Kelly means "do not bet", not "bet backwards". Returning the
    # raw number would hand a caller a negative position fraction.
    assert kelly_fraction(0.10, 1.0) == 0.0, "a losing edge must size nothing"
    assert kelly_fraction(0.60, 0.0) == 0.0, "zero odds must not divide"
    assert kelly_fraction(0.60, -1.0) == 0.0, "negative odds must not size"
    assert kelly_fraction(0.60, 1.0) > 0.0, "a real edge must still size"


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
