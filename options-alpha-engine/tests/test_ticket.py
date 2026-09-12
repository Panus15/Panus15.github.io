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
from models.ticket import (MIN_SETTLED_FOR_EVIDENCE, NO_STRUCTURE, NO_VOL_EDGE,
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
