"""Position sizing — risk of ruin is the enemy, not missed upside.

THE DEFECT THIS MODULE WAS BUILT AROUND, stated first because it shaped the API.

``position_size`` used to divide the risk budget by ``contract_price * multiplier``
and call the result a 2% cap. That is correct for BUYING an option, where the price
paid is the most you can lose. It is catastrophically wrong for SELLING one, which
is the only thing this engine does: there the price is the premium RECEIVED, and the
loss is unbounded above it. Measured on a $100,000 account at the advertised 2% cap:

    sell SPY 30d put ~10% OTM  ($1.10)     18 contracts     max loss    1,238% of the account
    sell SPY 30d put ~15% OTM  ($0.40)     50 contracts     max loss    3,250%
    sell SPY  7d put far OTM   ($0.05)    400 contracts     max loss   28,000%
    sell SPY  1d put far OTM   ($0.01)  2,000 contracts     max loss  144,000%

Note the direction: the CHEAPER the option, the MORE it sold. Cheap means far
out-of-the-money, which is exactly the strike that is harmless until the day it is
not. A cap that loosens as the tail gets fatter is worse than no cap, because it is
believed.

So the risk budget is now denominated in LOSS, and the caller must say what a loss
is. There is no default, and no way to pass a premium by accident: ``max_loss_per_contract``
is keyword-only and has no fallback.

ON KELLY, AND WHY IT IS GONE. Kelly on ``(win_prob, payoff_odds)`` is a two-outcome
formula. A short option is not two outcomes; it is a continuous payoff with a
loss-to-premium ratio that runs past 40x. Feeding a binary formula a payoff it cannot
represent produces a number with a respectable name and no meaning, and the previous
code then took ``min(kelly, cap)`` so the cap silently bound in every realistic case
anyway. ``kelly_fraction`` is kept — it is correct for what it says it is, and the
tests pin it — but nothing in the sizing path calls it any more.
"""

from __future__ import annotations


def kelly_fraction(edge: float, odds: float) -> float:
    """Kelly bet fraction for a simple win/lose payoff.

    edge : probability of winning (0..1)
    odds : net payoff multiple on a win (e.g. 1.0 = even money)

    Correct for a BINARY bet. Not applicable to an option short — see the module
    docstring. Retained because it is right about its own question.
    """
    if odds <= 0:
        return 0.0
    p = edge
    q = 1 - p
    f = (odds * p - q) / odds
    return max(f, 0.0)


#: Returned by ``max_loss_per_contract_for`` when the structure has no finite loss.
UNBOUNDED = float("inf")


def max_loss_per_contract_for(kind: str, strike: float, *, quantity: int,
                              entry_price: float, multiplier: int = 100,
                              wing_strike: float | None = None) -> float:
    """Worst-case loss of ONE contract at expiry, as a positive number.

    Long anything: you lose what you paid. Short put: the strike can go to zero,
    so the loss is bounded by ``strike`` minus the premium kept. Short call: the
    underlying has no ceiling, so the loss is UNBOUNDED unless a long wing caps it.

    ``wing_strike`` converts a naked short into a vertical spread and makes the
    loss finite — that is the ONLY way to size a short call here, and deliberately
    so: the alternative is inventing a worst case and calling it risk management.
    """
    kind = kind.lower()
    if kind not in ("put", "call"):
        raise ValueError(f"kind must be 'put' or 'call', got {kind!r}")
    if quantity >= 0:                       # long: the premium is the whole risk
        return max(entry_price, 0.0) * multiplier

    premium = max(entry_price, 0.0) * multiplier
    if wing_strike is not None:
        width = abs(wing_strike - strike) * multiplier
        return max(width - premium, 0.0)
    if kind == "put":
        return max(strike * multiplier - premium, 0.0)
    return UNBOUNDED


def position_size(
    account_equity: float,
    *,
    max_loss_per_contract: float,
    max_risk_frac: float = 0.02,
) -> int:
    """Contracts such that ``n * max_loss_per_contract <= max_risk_frac * equity``.

    This is a floor division on the LOSS, which makes the cap a theorem rather
    than a label: the returned size cannot breach it, and a structure whose loss
    is unbounded returns 0 rather than a number.

    ``max_loss_per_contract`` is keyword-only with no default so that a premium
    cannot be passed positionally where a loss is expected — the exact substitution
    that produced a 144,000% position.
    """
    if account_equity <= 0 or max_risk_frac <= 0:
        return 0
    if max_loss_per_contract == UNBOUNDED or max_loss_per_contract == float("inf"):
        return 0                            # refuse: no finite loss to cap
    if max_loss_per_contract <= 0:
        return 0                            # a riskless trade is a mispriced input
    budget = max_risk_frac * account_equity
    return max(int(budget // max_loss_per_contract), 0)
