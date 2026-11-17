"""Tests for the order ticket (models/ticket.py).

The property that matters is not that a ticket looks complete. It is that a
PLACEABLE ticket cannot lie about its risk: the loss it prints, times the
contracts it prints, must sit inside the budget it was given — and a ticket that
cannot satisfy that must refuse by name rather than by silence.

Every refusal code is tested independently, because a code that cannot fire is a
claim of safety nobody has checked.

Run: python3 tests/test_ticket.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.spreads import Spread, SpreadLeg
from models.ticket import (DECISION_SIZED_TO_ZERO, DECISION_STANDS_DOWN,
                           MIN_SETTLED_FOR_EVIDENCE, NO_STRUCTURE, NO_VOL_EDGE,
                           NON_POSITIVE_EV, SIZE_ZERO, STALE_QUOTES,
                           UNBOUNDED_RISK, build_ticket)

EQ = 100_000.0
ASOF = "2026-09-11"


class _Card:
    def __init__(self, side="SELL VOL"):
        self.symbol = "SPY"
        self.asof = ASOF
        self.expiry_days = 30
        self.vol_side = side
        self.vol_reason = "IV rich against the forecast"


def _spread(*, max_loss=8.5, credit=1.5, ev=0.35, dte=30, name="put credit 5-wide"):
    """Per-share numbers, as models.spreads produces them."""
    return Spread(name=name, expiry_days=dte,
                  legs=[SpreadLeg(kind="put", strike=690.0, side="short", price=2.10),
                        SpreadLeg(kind="put", strike=680.0, side="long", price=0.60)],
                  net_credit=credit, max_loss=max_loss, max_gain=credit,
                  break_evens=(688.5,), prob_profit=0.78, ev=ev)


def _codes(t):
    return [c for c, _ in t.refusals]


class _Input:
    """Stands in for models.decision.Input — only .name and .effect are read."""
    def __init__(self, name, effect=1.0):
        self.name, self.effect = name, effect


class _Decision:
    """Stands in for models.decision.Decision. Built by hand so these tests pin
    the CONTRACT between the two modules rather than re-testing decision.py."""
    def __init__(self, action="SELL VOL", size_multiplier=1.0, cut=()):
        self.action = action
        self.size_multiplier = size_multiplier
        self.inputs = [_Input(n, e) for n, e in cut]


# --------------------------------------------------------------------------
# The fused decision must actually reach the order
# --------------------------------------------------------------------------

def test_the_decision_size_cut_reaches_the_contract_count():
    """The defect this closes: build_ticket read only the card, so a decision of
    "SELL VOL, size x0.60" shipped exactly the same order as x1.00 — 67-100% more
    risk than the fused view authorised. A cut the last step discards is a
    comment, not a risk control."""
    for equity, mult in ((100_000.0, 0.6), (250_000.0, 0.6), (1_000_000.0, 0.6),
                         (1_000_000.0, 0.5), (1_000_000.0, 0.85)):
        full = build_ticket(_Card(), _spread(), equity=equity)
        cut = build_ticket(_Card(), _spread(), equity=equity,
                           decision=_Decision(size_multiplier=mult,
                                              cut=(("fund crowding", mult),)))
        assert cut.contracts == int(full.contracts * mult), (
            f"equity {equity:,.0f} x{mult}: shipped {cut.contracts}, "
            f"authorised {int(full.contracts * mult)}")
        assert cut.max_loss_total <= full.max_loss_total


def test_the_cut_never_rounds_up_into_more_risk_than_authorised():
    """Swept, because the rounding direction is the whole property: at a x0.6 on
    2 contracts, rounding gives 1 and rounding-to-nearest gives 1, but at x0.9 on
    5 they differ (4 vs 5) and only the floor is safe."""
    for mult in (0.05, 0.25, 0.5, 0.6, 0.75, 0.85, 0.9, 0.99):
        for equity in (100_000.0, 250_000.0, 500_000.0, 1_000_000.0):
            full = build_ticket(_Card(), _spread(), equity=equity).contracts
            cut = build_ticket(_Card(), _spread(), equity=equity,
                               decision=_Decision(size_multiplier=mult)).contracts
            assert cut <= full * mult + 1e-9, (mult, equity, cut, full)


def test_the_decision_can_only_cut_never_raise():
    """A multiplier above 1.0 is a bug upstream, and the ticket must not act on
    it — rule 1 of decision.py is that an unproven input may only reduce size."""
    full = build_ticket(_Card(), _spread(), equity=1_000_000.0).contracts
    for bad in (1.5, 3.0, 100.0):
        t = build_ticket(_Card(), _spread(), equity=1_000_000.0,
                         decision=_Decision(size_multiplier=bad))
        assert t.contracts == full, (bad, t.contracts, full)


def test_a_garbage_multiplier_refuses_instead_of_defaulting_to_full_size():
    """NaN survives min/max clamping untouched and inf clamps UP to 1.0, so the
    obvious guard turns a corrupt risk input into a maximum-size order. Unusable
    means zero here."""
    full = build_ticket(_Card(), _spread(), equity=1_000_000.0).contracts
    assert full > 0
    for bad in (None, "big", float("nan"), float("inf"), float("-inf")):
        t = build_ticket(_Card(), _spread(), equity=1_000_000.0,
                         decision=_Decision(size_multiplier=bad))
        assert t.contracts == 0, (bad, t.contracts)
        assert DECISION_SIZED_TO_ZERO in _codes(t), (bad, _codes(t))


def test_the_per_contract_dollar_numbers_are_not_scaled_by_the_cut():
    """The multiplier sizes the ORDER, not the instrument. This caught a real
    name collision: the new size multiplier shadowed the contract multiplier, so
    a x0.60 decision would have printed a $1.50 limit instead of $150."""
    full = build_ticket(_Card(), _spread(), equity=1_000_000.0)
    cut = build_ticket(_Card(), _spread(), equity=1_000_000.0,
                       decision=_Decision(size_multiplier=0.6))
    assert cut.limit_credit == full.limit_credit == 1.5 * 100
    assert cut.max_loss_per_contract == full.max_loss_per_contract == 8.5 * 100
    assert cut.ev_per_contract == full.ev_per_contract


def test_a_negative_multiplier_never_reaches_the_screen_as_a_negative_size():
    """The refusal path already stops a negative from sizing anything, so the
    ONLY place it is still observable is the printed multiplier — and a ticket
    reading "size x-0.50" is a worse thing to show a human than "x0.00"."""
    t = build_ticket(_Card(), _spread(), equity=1_000_000.0,
                     decision=_Decision(size_multiplier=-0.5))
    assert t.size_multiplier == 0.0, t.size_multiplier
    assert t.contracts == 0
    assert "x-" not in t.render(), t.render()


def test_a_decision_that_disagrees_with_the_card_stands_the_ticket_down():
    for action in ("NO TRADE", "BUY VOL"):
        t = build_ticket(_Card(), _spread(), equity=EQ,
                         decision=_Decision(action=action, size_multiplier=1.0))
        assert DECISION_STANDS_DOWN in _codes(t), (action, _codes(t))
        assert not t.placeable and t.contracts == 0


def test_a_cut_below_one_contract_refuses_by_its_own_name():
    """Distinct from SIZE_ZERO: the account is big enough, the CONTEXT said no.
    The remedy differs, so the code must too."""
    t = build_ticket(_Card(), _spread(), equity=EQ,
                     decision=_Decision(size_multiplier=0.1,
                                        cut=(("fund crowding", 0.1),)))
    codes = _codes(t)
    assert DECISION_SIZED_TO_ZERO in codes, codes
    assert SIZE_ZERO not in codes, "the account was never the problem"
    detail = dict(t.refusals)[DECISION_SIZED_TO_ZERO]
    assert "fund crowding" in detail, detail


def test_a_no_trade_decision_does_not_list_the_same_cause_twice():
    """A decision whose action is NO TRADE also reports x0.00, so the naive check
    raised DECISION_SIZED_TO_ZERO alongside NO_VOL_EDGE — one cause printed twice,
    the second naming no input, which reads as an extra problem to go and fix."""
    card = _Card(side="NO TRADE")
    t = build_ticket(card, _spread(), equity=EQ,
                     decision=_Decision(action="NO TRADE", size_multiplier=0.0))
    codes = _codes(t)
    assert NO_VOL_EDGE in codes, codes
    assert DECISION_SIZED_TO_ZERO not in codes, codes
    assert DECISION_STANDS_DOWN not in codes, "they agree; there is no disagreement"


def test_an_absent_decision_is_visible_on_the_ticket_rather_than_silent():
    """An optional safety check that can be skipped silently is the same defect
    wearing a keyword argument."""
    t = build_ticket(_Card(), _spread(), equity=EQ)
    assert not t.decision_applied
    assert "NOT APPLIED" in t.render()
    t2 = build_ticket(_Card(), _spread(), equity=EQ, decision=_Decision())
    assert t2.decision_applied
    assert "NOT APPLIED" not in t2.render()
    assert "x1.00" in t2.render()


def test_a_refused_ticket_still_shows_what_the_context_did():
    """Otherwise the only screens naming the cut are the ones already placing
    orders, which is backwards."""
    t = build_ticket(_Card(), _spread(), equity=EQ,
                     decision=_Decision(size_multiplier=0.1,
                                        cut=(("fund crowding", 0.1),)))
    assert not t.placeable
    assert "fused context" in t.render()


def test_the_budget_theorem_still_holds_once_the_cut_is_applied():
    """The headline property of test #1, re-swept with a decision in the path —
    a second cap must not be able to break the first."""
    for mult in (0.1, 0.5, 0.85, 1.0):
        for loss in (0.5, 8.5, 45.0):
            for frac in (0.005, 0.02, 0.05):
                t = build_ticket(_Card(), _spread(max_loss=loss), equity=EQ,
                                 max_risk_frac=frac,
                                 decision=_Decision(size_multiplier=mult))
                if not t.placeable:
                    continue
                assert t.max_loss_total <= frac * EQ + 1e-9


def test_a_placeable_ticket_cannot_breach_its_own_budget():
    """The headline property, swept rather than spot-checked. If this ever fails,
    the ticket is printing a risk number it does not honour."""
    for loss in (0.5, 2.0, 8.5, 19.99, 45.0, 120.0):
        for frac in (0.005, 0.01, 0.02, 0.05):
            t = build_ticket(_Card(), _spread(max_loss=loss), equity=EQ,
                             max_risk_frac=frac)
            if not t.placeable:
                continue
            assert t.max_loss_total <= frac * EQ + 1e-9, (
                f"loss {loss}/share at {frac:.1%}: {t.contracts} contracts risk "
                f"{t.max_loss_total:,.2f} against {frac * EQ:,.2f}")


def test_the_printed_numbers_are_consistent_with_each_other():
    t = build_ticket(_Card(), _spread(), equity=EQ)
    assert t.placeable
    assert t.max_loss_total == t.max_loss_per_contract * t.contracts
    assert t.credit_total == t.limit_credit * t.contracts
    assert t.max_loss_per_contract == 8.5 * 100
    assert t.limit_credit == 1.5 * 100
    body = t.render()
    for want in ("SELL", "max loss", "limit", "P(profit)", "evidence"):
        assert want in body, body


def test_every_leg_reaches_the_ticket_with_its_side_and_price():
    """A ticket missing a leg is an order that cannot be placed, and the missing
    leg is always the long wing — the one that makes the risk finite."""
    t = build_ticket(_Card(), _spread(), equity=EQ)
    assert len(t.legs) == 2
    sides = {(l.side, l.kind, l.strike) for l in t.legs}
    assert ("short", "put", 690.0) in sides
    assert ("long", "put", 680.0) in sides
    assert all(l.price > 0 for l in t.legs)


def test_the_expiry_becomes_a_real_date_not_a_day_count():
    """"30d" is not something a human can select in a broker."""
    t = build_ticket(_Card(), _spread(dte=30), equity=EQ, asof="2026-09-11")
    assert t.expiry_date == "2026-10-11", t.expiry_date


def test_a_bad_asof_degrades_to_a_relative_date_rather_than_crashing():
    t = build_ticket(_Card(), _spread(), equity=EQ, asof="not-a-date")
    assert t.expiry_date.startswith("+")


# --------------------------------------------------------------------------
# Each refusal code fires on its own
# --------------------------------------------------------------------------

def test_no_vol_edge_refuses():
    t = build_ticket(_Card(side="NO TRADE"), _spread(), equity=EQ)
    assert NO_VOL_EDGE in _codes(t) and not t.placeable
    assert t.contracts == 0, "a refused ticket must not still carry a size"


def test_no_structure_refuses_without_crashing():
    t = build_ticket(_Card(), None, equity=EQ)
    assert NO_STRUCTURE in _codes(t) and not t.placeable
    assert "REFUSED" in t.render()


def test_non_positive_ev_refuses():
    t = build_ticket(_Card(), _spread(ev=0.0), equity=EQ)
    assert NON_POSITIVE_EV in _codes(t)
    t2 = build_ticket(_Card(), _spread(ev=-0.2), equity=EQ)
    assert NON_POSITIVE_EV in _codes(t2)


def test_an_unbounded_structure_refuses_rather_than_being_sized():
    t = build_ticket(_Card(), _spread(max_loss=0.0), equity=EQ)
    assert UNBOUNDED_RISK in _codes(t) and not t.placeable
    assert t.contracts == 0


def test_size_zero_refuses_and_says_what_did_not_fit():
    """The honest common case for a small account: the structure is fine and the
    account is too small for one of it."""
    t = build_ticket(_Card(), _spread(max_loss=500.0), equity=10_000.0,
                     max_risk_frac=0.02)
    assert SIZE_ZERO in _codes(t)
    detail = dict(t.refusals)[SIZE_ZERO]
    assert "50,000" in detail and "200" in detail, detail


def test_a_chain_with_no_asof_is_flagged_as_stale():
    card = _Card()
    card.asof = ""
    t = build_ticket(card, _spread(), equity=EQ, asof="")
    assert STALE_QUOTES in _codes(t)


def test_refusals_accumulate_rather_than_masking_each_other():
    """A reader fixing one problem should see the others waiting, not discover
    them one at a time across four runs."""
    t = build_ticket(_Card(side="NO TRADE"), _spread(ev=-1.0, max_loss=500.0),
                     equity=10_000.0)
    codes = set(_codes(t))
    assert {NO_VOL_EDGE, NON_POSITIVE_EV, SIZE_ZERO} <= codes, codes


# --------------------------------------------------------------------------
# Evidence, which must come from a counter and not from confidence
# --------------------------------------------------------------------------

def test_with_an_empty_ledger_the_ticket_says_it_has_no_live_evidence():
    t = build_ticket(_Card(), _spread(), equity=EQ, settled_trades=0)
    assert "no live evidence" in t.evidence
    assert "0 trades" in t.evidence or "0 trades" in t.evidence.replace("settled ", "")
    assert "no live evidence" in t.render()


def test_the_evidence_label_tracks_the_counter_and_nothing_else():
    for n, want in ((0, "no live evidence"), (1, "below the"),
                    (MIN_SETTLED_FOR_EVIDENCE - 1, "below the"),
                    (MIN_SETTLED_FOR_EVIDENCE, "enough to ask")):
        t = build_ticket(_Card(), _spread(), equity=EQ, settled_trades=n)
        assert want in t.evidence, (n, t.evidence)


def test_a_refused_ticket_still_reports_its_evidence():
    """Otherwise the only screens carrying the caveat are the ones already
    showing numbers, which is backwards."""
    t = build_ticket(_Card(side="NO TRADE"), _spread(), equity=EQ)
    assert "evidence" in t.render()


def test_the_exit_rule_is_shown_when_given():
    """A recommendation without a management rule is not the strategy that was
    tested, so its backtested numbers do not describe it."""
    t = build_ticket(_Card(), _spread(), equity=EQ,
                     exit_rule="close at 50% of max profit or 7 DTE, whichever first")
    assert "close at 50%" in t.render()


def test_the_probability_says_which_event_it_is_the_probability_of():
    """P(profit) is a TERMINAL statistic — the chance the position finishes
    profitable at expiry. Printed bare next to "close at 50% of max profit" it
    invites the reader to attach it to an exit it does not describe."""
    body = build_ticket(_Card(), _spread(), equity=EQ).render()
    i = body.index("P(profit)")
    assert "AT EXPIRY" in body[i:i + 80], body[i:i + 80]


def test_the_measured_exit_split_is_printed_beside_the_rule_it_describes():
    """The probability of the RECOMMENDED exit is not computed anywhere in this
    repo — it is a first-passage problem, not a terminal-density one — so what is
    shown is the frequency that was actually observed, with its sample."""
    note = ("of 600 trades measured under this rule, 60% ended at the profit "
            "target and 40% at the time stop (10 generated paths, not market data)")
    t = build_ticket(_Card(), _spread(), equity=EQ,
                     exit_rule="close at 50% of max profit, or at 7 DTE",
                     exit_split_note=note)
    body = t.render()
    assert note in body
    # and it sits on the line DIRECTLY BELOW the rule, not merely nearby: a
    # character-distance check passed with forty blank lines wedged between them.
    lines = body.splitlines()
    i = next(n for n, l in enumerate(lines) if "close at 50%" in l)
    assert note in lines[i + 1], (
        f"the measured split is not on the line under the rule it describes:\n"
        + "\n".join(repr(l) for l in lines[i:i + 3]))
    # absent, nothing is invented
    assert "measured under this rule" not in build_ticket(
        _Card(), _spread(), equity=EQ, exit_rule="hold to expiry").render()


def test_a_smaller_account_gets_fewer_contracts_never_more():
    sizes = [build_ticket(_Card(), _spread(), equity=e).contracts
             for e in (25_000.0, 50_000.0, 100_000.0, 250_000.0)]
    assert sizes == sorted(sizes), sizes


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
