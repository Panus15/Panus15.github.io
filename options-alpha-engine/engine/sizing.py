"""Position sizing — risk of ruin is the enemy, not missed upside.

Even a genuine edge blows up if you bet too big. Use *fractional* Kelly and a
hard per-trade risk cap. Full Kelly is theoretically growth-optimal but so
volatile that real desks trade a fraction of it (1/4 to 1/2).
"""

from __future__ import annotations


def kelly_fraction(edge: float, odds: float) -> float:
    """Kelly bet fraction for a simple win/lose payoff.

    edge : probability of winning (0..1)
    odds : net payoff multiple on a win (e.g. 1.0 = even money)
    """
    if odds <= 0:
        return 0.0
    p = edge
    q = 1 - p
    f = (odds * p - q) / odds
    return max(f, 0.0)


def position_size(
    account_equity: float,
    contract_price: float,
    *,
    win_prob: float,
    payoff_odds: float,
    kelly_scale: float = 0.25,
    max_risk_frac: float = 0.02,
    multiplier: int = 100,
) -> int:
    """Number of contracts to trade.

    Sized by fractional Kelly, then hard-capped so no single trade risks more
    than ``max_risk_frac`` of the account (default 2%). Returns an integer
    contract count (>= 0).
    """
    f = kelly_scale * kelly_fraction(win_prob, payoff_odds)
    capital_at_risk = min(f, max_risk_frac) * account_equity
    cost_per_contract = contract_price * multiplier
    if cost_per_contract <= 0:
        return 0
    return max(int(capital_at_risk // cost_per_contract), 0)
