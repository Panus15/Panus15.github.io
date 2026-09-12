"""Portfolio-level risk layer for a short-volatility options book.

WHY THIS MODULE EXISTS
----------------------
Per-trade sizing (``engine.sizing``: fractional Kelly + a 2% per-trade cap) is
necessary but NOT sufficient for a short-vol book, and trusting it alone gives
false comfort. The reason is *factor concentration*:

    Every short-vol position (short strangles, naked puts, iron condors, ...) is
    a bet on the SAME underlying risk factor -- implied volatility staying put
    or falling. Ten "independent" 2%-risk trades on SPX/NDX/RUT are not ten
    independent bets; in a vol spike they ALL lose at once. N x 2% is a single
    correlated tail, not diversification.

A naive per-trade view implicitly treats positions as independent, so it reports
a "diversified" risk that scales like sqrt(N). The real, correlated exposure
scales like N. This module makes that gap explicit and adds the controls the
per-trade layer cannot provide:

1. Correlation-aware Greek aggregation -- net delta/gamma/vega/theta plus an
   *effective* short-vega that treats correlated underlyings as ONE factor, so
   exposure does not falsely diversify away (``Portfolio``).
2. A ``RiskGovernor`` that vetoes a proposed trade if it would breach the
   correlation-aware vega cap, and a drawdown KILL-SWITCH that blocks ALL new
   risk once the book is deep enough in the hole (``RiskLimits``).
3. CVaR / expected-shortfall sizing that replaces the binary Kelly bet: size a
   short-vol trade off its fat LEFT tail under a vol-spike scenario, so a bigger
   modeled shock always yields FEWER contracts (``size_by_cvar``).
4. A defined-risk converter that turns a naked short option (unbounded tail)
   into a vertical spread with a hard, bounded max loss (``to_defined_risk``).

Everything is pure standard library and plugs into ``engine.pricing`` for
repricing under stress. Nothing here mutates the per-trade sizing module; this
supersedes it at the *portfolio* level.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import pricing

# Default cross-underlying correlation for an equity-index short-vol book.
# Index vols (SPX/NDX/RUT/VIX complex) co-move violently in a risk-off spike;
# 0.8 is a deliberately HIGH prior so the model refuses to hand out a
# diversification benefit that will not be there when it matters.
DEFAULT_VEGA_RHO = 0.8

# Standard US equity-option contract multiplier.
DEFAULT_MULTIPLIER = 100


# --------------------------------------------------------------------------- #
# Position
# --------------------------------------------------------------------------- #
@dataclass
class Position:
    """A held (or proposed) single-leg option position.

    ``quantity`` is SIGNED: positive = long, negative = short. All per-contract
    Greeks come from ``engine.pricing`` using the market context carried on the
    position (``spot``, ``r``, ``q``, ``vol``), so the position can also be
    repriced under stress scenarios. Position-level (a.k.a. "dollar") Greeks are
    the per-contract Greek x quantity x multiplier -- shorts therefore carry
    NEGATIVE vega/gamma, which is exactly the risk this module polices.
    """

    underlying: str
    strike: float
    expiry_days: int
    kind: str                 # "call" or "put"
    quantity: int             # SIGNED: negative == short
    entry_price: float        # premium paid (long) or received (short), per contract
    spot: float               # underlying spot used for Greeks / repricing
    r: float = 0.0            # risk-free rate
    q: float = 0.0            # dividend yield
    vol: float = 0.20         # implied vol at entry (annualised)
    multiplier: int = DEFAULT_MULTIPLIER
    #: Days per year for this position's tenor. 365 is right for a CALENDAR expiry
    #: ("30 days to expiry"); 252 is right when the caller counts TRADING bars.
    #: Both conventions exist in this repo and mixing them silently was a real bug —
    #: see the note on `T` below.
    days_per_year: float = 365.0

    # --- time --------------------------------------------------------------- #
    @property
    def T(self) -> float:
        """Time to expiry in YEARS.

        This used to be hardcoded to ``expiry_days / 365``, while the simulation
        loops in hedged_backtest / signal_backtest price the SAME position at
        ``dte / 252`` because their ``dte`` counts trading bars. At a 21-bar tenor
        that is a 1.45x difference in T, so the risk governor sized on an option
        with 16.8% less vega than the one actually held — it under-measured the
        risk of every position it approved. Callers that count bars must pass
        ``days_per_year=252`` so the sizing sees the tenor being traded.
        """
        return self.expiry_days / self.days_per_year

    @property
    def is_short(self) -> bool:
        return self.quantity < 0

    # --- Greeks ------------------------------------------------------------- #
    def contract_greeks(self) -> pricing.Greeks:
        """Per-single-long-contract Greeks (before sign/quantity/multiplier)."""
        return pricing.greeks(
            self.spot, self.strike, self.T, self.r, self.q, self.vol, self.kind
        )

    def _scaled(self, greek: float) -> float:
        return greek * self.quantity * self.multiplier

    def net_delta(self) -> float:
        return self._scaled(self.contract_greeks().delta)

    def net_gamma(self) -> float:
        return self._scaled(self.contract_greeks().gamma)

    def net_vega(self) -> float:
        """Signed dollar vega: negative for short-vol positions."""
        return self._scaled(self.contract_greeks().vega)

    def net_theta(self) -> float:
        return self._scaled(self.contract_greeks().theta)

    # --- expiry payoff (for defined-risk / worst-case checks) --------------- #
    def intrinsic(self, spot: float) -> float:
        if self.kind.lower() == "call":
            return max(spot - self.strike, 0.0)
        return max(self.strike - spot, 0.0)

    def pnl_at_expiry(self, spot: float) -> float:
        """Realised P&L of this leg if held to expiry at terminal ``spot``.

        = quantity x multiplier x (intrinsic - entry_price). For a short leg
        (quantity < 0) a large intrinsic value is a large LOSS.
        """
        return self.quantity * self.multiplier * (self.intrinsic(spot) - self.entry_price)


# --------------------------------------------------------------------------- #
# Portfolio -- correlation-aware aggregation
# --------------------------------------------------------------------------- #
@dataclass
class Portfolio:
    """A book of positions with correlation-aware risk aggregation.

    CORRELATION MODEL (vega)
    ------------------------
    Each position i carries a signed dollar vega v_i. We form a position-level
    correlation matrix C:

        C_ij = 1.0    if i == j                       (self)
        C_ij = 1.0    if underlying(i) == underlying(j)  (same name -> perfectly correlated)
        C_ij = rho    otherwise                        (cross-underlying, default 0.8)

    Three views of the book's vega then fall out:

      * ``net_vega``          -- signed LINEAR sum  (directional net vega).
      * ``independent_vega``  -- sqrt(sum v_i^2), i.e. C = identity. This is the
                                 "diversified" number a correlation-BLIND, per-
                                 trade view implies. For N identical shorts it is
                                 sqrt(N) x single -> the FALSE comfort.
      * ``effective_short_vega`` -- sqrt(v^T C v), the correlation-AWARE factor
                                 magnitude. For N identical shorts on one name
                                 (all C_ij = 1) it is N x single -> the exposure
                                 that actually shows up in a vol spike.

    The whole point: ``effective_short_vega`` does not diversify a single-factor
    book down to sqrt(N); it keeps the full ~N exposure the per-trade view hides.
    """

    positions: list[Position] = field(default_factory=list)
    rho: float = DEFAULT_VEGA_RHO

    def add(self, position: Position) -> None:
        self.positions.append(position)

    def with_position(self, position: Position) -> "Portfolio":
        """Return a NEW portfolio = this book + ``position`` (non-mutating)."""
        return Portfolio(list(self.positions) + [position], rho=self.rho)

    # --- linear net Greeks -------------------------------------------------- #
    def net_delta(self) -> float:
        return sum(p.net_delta() for p in self.positions)

    def net_gamma(self) -> float:
        return sum(p.net_gamma() for p in self.positions)

    def net_vega(self) -> float:
        """Signed linear net vega (short-vol book is negative)."""
        return sum(p.net_vega() for p in self.positions)

    def net_theta(self) -> float:
        return sum(p.net_theta() for p in self.positions)

    def gross_vega(self) -> float:
        """Sum of |vega| across legs -- total vega notional at work."""
        return sum(abs(p.net_vega()) for p in self.positions)

    # --- correlation-aware vega -------------------------------------------- #
    def independent_vega(self) -> float:
        """RSS of per-position vega assuming ZERO correlation (the false-comfort,
        'diversified' number). Understates a single-factor book by ~sqrt(N)."""
        return math.sqrt(sum(p.net_vega() ** 2 for p in self.positions))

    def effective_short_vega(self, rho: float | None = None) -> float:
        """Correlation-aware factor-vega magnitude = sqrt(v^T C v).

        Same-underlying legs are perfectly correlated (net linearly); different
        underlyings are correlated at ``rho``. For a single-factor short-vol book
        this equals ~N x single -- it does NOT diversify away.
        """
        rho = self.rho if rho is None else rho
        vals = [(p.underlying, p.net_vega()) for p in self.positions]
        total = 0.0
        for i, (ui, vi) in enumerate(vals):
            for j, (uj, vj) in enumerate(vals):
                c = 1.0 if (i == j or ui == uj) else rho
                total += c * vi * vj
        return math.sqrt(max(total, 0.0))

    def net_short_vega(self, rho: float | None = None) -> float:
        """Correlation-aware NET SHORT vega exposure (a positive number).

        Returns the effective factor-vega magnitude when the book is net short
        vol (the dangerous direction), else 0.0. This is the number the
        ``RiskGovernor`` caps -- the real single-factor short-vega a naive
        per-trade view understates.
        """
        if self.net_vega() < 0:
            return self.effective_short_vega(rho)
        return 0.0

    # --- expiry payoff ------------------------------------------------------ #
    def pnl_at_expiry(self, spot: float) -> float:
        return sum(p.pnl_at_expiry(spot) for p in self.positions)


# --------------------------------------------------------------------------- #
# Risk limits + governor (caps + drawdown kill-switch)
# --------------------------------------------------------------------------- #
@dataclass
class RiskLimits:
    """Hard portfolio-level limits. ``None`` disables an individual check.

    ``max_net_short_vega`` and ``max_gross_vega`` are in DOLLAR vega (per 1.00
    change in vol, i.e. +100 vol points -- the pricing convention). Set them
    from the account's tolerance for a vol-spike loss. ``max_drawdown`` is a
    fraction of peak equity; breaching it trips the kill-switch.
    """

    max_net_short_vega: float | None = None   # cap on correlation-aware short vega
    max_gross_vega: float | None = None       # cap on total vega notional
    max_net_delta: float | None = None        # cap on |net delta| (directional drift)
    max_drawdown: float = 0.20                 # kill-switch threshold (fraction of peak)


class RiskGovernor:
    """Vetoes proposed trades against portfolio-level limits.

    Two independent gates, checked in order:

      1. DRAWDOWN KILL-SWITCH -- if the book's current drawdown has breached
         ``max_drawdown``, NO new risk is allowed, regardless of vega headroom.
         Stop digging.
      2. VEGA / DELTA CAPS -- otherwise the trade is simulated into the book and
         rejected if it would push the *correlation-aware* net short vega (or
         gross vega, or net delta) past its cap.
    """

    def __init__(self, limits: RiskLimits):
        self.limits = limits

    def can_add(
        self,
        position: Position,
        portfolio: Portfolio,
        equity: float,
        current_drawdown: float,
    ) -> tuple[bool, str]:
        """Return ``(allowed, reason)`` for adding ``position`` to ``portfolio``.

        ``current_drawdown`` is a POSITIVE fraction (e.g. 0.15 == 15% below peak).
        """
        lim = self.limits

        # --- gate 1: drawdown kill-switch (overrides everything) ------------ #
        if current_drawdown >= lim.max_drawdown:
            return (
                False,
                f"KILL-SWITCH: drawdown {current_drawdown:.1%} >= max "
                f"{lim.max_drawdown:.1%}; all new risk blocked",
            )

        # --- gate 2: caps on the prospective book -------------------------- #
        prospective = portfolio.with_position(position)

        if lim.max_net_short_vega is not None:
            nsv = prospective.net_short_vega()
            if nsv > lim.max_net_short_vega:
                return (
                    False,
                    f"VEGA CAP: correlated net short vega {nsv:,.0f} would exceed "
                    f"cap {lim.max_net_short_vega:,.0f} (single-factor concentration)",
                )

        if lim.max_gross_vega is not None:
            gv = prospective.gross_vega()
            if gv > lim.max_gross_vega:
                return (
                    False,
                    f"GROSS VEGA CAP: {gv:,.0f} would exceed cap "
                    f"{lim.max_gross_vega:,.0f}",
                )

        if lim.max_net_delta is not None:
            nd = abs(prospective.net_delta())
            if nd > lim.max_net_delta:
                return (
                    False,
                    f"NET DELTA CAP: |net delta| {nd:,.0f} would exceed cap "
                    f"{lim.max_net_delta:,.0f}",
                )

        return (True, "OK: within all portfolio risk limits")


# --------------------------------------------------------------------------- #
# CVaR / tail-based sizing (replaces binary Kelly for short vol)
# --------------------------------------------------------------------------- #
@dataclass
class Scenario:
    """A single stress state for a UNIT position.

    ``spot_mult``   multiplies the underlying spot (0.90 == a 10% drop).
    ``vol_shift``   is added to implied vol (ABSOLUTE, e.g. +0.10 == +10 vol pts).
    ``prob``        is the state's probability weight (weights need not sum to 1;
                    they are normalised internally).
    ``days_forward`` optionally decays time to expiry by this many calendar days.
    """

    spot_mult: float = 1.0
    vol_shift: float = 0.0
    prob: float = 1.0
    days_forward: int = 0


def vol_spike_scenarios(
    *,
    spot_shock: float = -0.10,
    vol_bump: float = 0.10,
    tail_prob: float = 0.10,
) -> list[Scenario]:
    """Default two-state scenario set: a calm state + a vol-SPIKE tail.

    SCENARIO MODEL
    --------------
    Short-vol P&L is asymmetric: benign in the calm state, brutal in a spike
    where spot GAPS DOWN and implied vol JUMPS UP simultaneously (the classic
    risk-off move). We model that as a two-point mixture:

        calm  : spot x 1,              vol + 0,         prob = 1 - tail_prob
        spike : spot x (1 + spot_shock), vol + vol_bump, prob = tail_prob

    A bigger ``spot_shock`` / ``vol_bump`` => a fatter left tail => a larger
    per-contract CVaR => ``size_by_cvar`` returns FEWER contracts.

    WHAT THIS SET CANNOT DO, stated because the name "CVaR at 95% confidence"
    implies otherwise. With two points there is exactly ONE loss state carrying
    ``tail_prob`` of the mass, so every confidence level whose tail (1 - alpha)
    falls at or below ``tail_prob`` averages over that same single state and
    returns the SAME number. At the shipped default of tail_prob=0.10, alpha=0.90,
    0.95, 0.99 and 0.999 are all identical - alpha is inert. The figure is not a
    quantile of a distribution; it is the loss in one hand-written scenario, and
    ``spot_shock``/``vol_bump`` are the knobs that actually move it.

    ``cvar_per_contract`` refuses a level it cannot resolve rather than returning
    a number that looks like a 99% figure and is not. Pass a richer multi-severity
    list when you have one and alpha starts to mean what it says.
    """
    return [
        Scenario(spot_mult=1.0, vol_shift=0.0, prob=1.0 - tail_prob),
        Scenario(spot_mult=1.0 + spot_shock, vol_shift=vol_bump, prob=tail_prob),
    ]


def scenario_pnl(template: Position, scenario: Scenario) -> float:
    """P&L of ONE contract of ``template`` (in its signed direction) in ``scenario``.

    Reprices the option via ``engine.pricing`` at the shocked spot / vol / time
    and marks it against ``entry_price``. Sign follows the template's direction
    (short => a price increase is a loss).
    """
    S2 = template.spot * scenario.spot_mult
    vol2 = max(template.vol + scenario.vol_shift, 1e-6)
    days2 = max(template.expiry_days - scenario.days_forward, 0)
    T2 = max(days2 / 365.0, 1e-6)
    new_price = pricing.price(S2, template.strike, T2, template.r, template.q, vol2, template.kind)
    direction = -1.0 if template.quantity < 0 else 1.0
    return direction * (new_price - template.entry_price) * template.multiplier


def cvar_per_contract(
    template: Position,
    scenarios: list[Scenario],
    alpha: float = 0.95,
) -> float:
    """Expected shortfall (CVaR) at level ``alpha`` of ONE contract, as a
    POSITIVE loss number (0.0 if the modeled tail shows no loss).

    Discrete CVaR: sort the scenario P&Ls worst-first, walk down accumulating
    probability mass until (1 - alpha) is covered, and average the loss over
    exactly that worst tail mass.
    """
    if not scenarios:
        return 0.0
    total_p = sum(s.prob for s in scenarios)
    if total_p <= 0:
        return 0.0

    # (pnl, normalised prob), worst pnl (largest loss) first.
    rows = sorted(
        ((scenario_pnl(template, s), s.prob / total_p) for s in scenarios),
        key=lambda r: r[0],
    )
    tail = 1.0 - alpha
    # The level must be RESOLVABLE by this scenario set. If the worst state alone
    # carries more mass than the tail we were asked about, every alpha in that
    # range returns the same number, and calling it "CVaR at 99%" is a claim the
    # data cannot support. Say so instead.
    worst_mass = rows[0][1]
    # 1.0 - 0.9 is 0.09999999999999998, so the exactly-resolvable boundary needs
    # a tolerance or the shipped default refuses itself
    if 0 < tail < worst_mass - 1e-9 and len(rows) > 1:
        raise ValueError(
            f"alpha={alpha} asks for the worst {tail:.1%} of outcomes, but the "
            f"worst scenario alone carries {worst_mass:.1%} of the probability "
            f"mass. Every alpha above {1 - worst_mass:.2f} returns the same "
            f"number here, so this set cannot resolve that level - supply a "
            f"multi-severity scenario list, or ask for alpha <= {1 - worst_mass:.2f}.")
    if tail <= 0:
        # Degenerate: alpha == 1 -> pure worst-case loss.
        worst_pnl = rows[0][0]
        return max(-worst_pnl, 0.0)

    acc = 0.0
    loss_acc = 0.0
    for pnl, p in rows:
        take = min(p, tail - acc)
        if take <= 0:
            break
        loss_acc += take * (-pnl)   # loss = -pnl
        acc += take
        if acc >= tail:
            break
    if acc <= 0:
        return 0.0
    return max(loss_acc / acc, 0.0)


def size_by_cvar(
    equity: float,
    position_template: Position,
    *,
    shock_scenarios: list[Scenario],
    cvar_limit: float,
    alpha: float = 0.90,
) -> int:
    """Contracts to trade so the position's CVaR stays within a dollar budget.

    Replaces the binary Kelly bet for short vol: instead of "edge => bet f% of
    bankroll", we size directly off the fat LEFT tail. Steps:

      1. Build per-contract P&Ls across ``shock_scenarios`` (a vol-spike set).
      2. Compute per-contract CVaR (expected shortfall) at ``alpha``.
      3. n = floor( (cvar_limit x equity) / cvar_per_contract ).

    A bigger modeled shock raises the per-contract CVaR and therefore returns
    FEWER contracts -- the desired monotonicity. Returns a non-negative int.
    Returns 0 when the scenarios show no tail loss (nothing for CVaR to size).
    """
    if equity <= 0 or cvar_limit <= 0:
        return 0
    es = cvar_per_contract(position_template, shock_scenarios, alpha)
    if es <= 0:
        return 0
    budget = cvar_limit * equity
    return max(int(budget // es), 0)


# --------------------------------------------------------------------------- #
# Defined-risk conversion (bound the naked-short tail)
# --------------------------------------------------------------------------- #
def to_defined_risk(short_leg: Position, wing_offset: float) -> list[Position]:
    """Convert a naked short option into a vertical spread with BOUNDED loss.

    WHY: a naked short option has an unbounded (or, for a short put, strike-sized)
    left tail -- exactly the single-factor short-vol catastrophe this module
    exists to contain. Buying a long "wing" further out-of-the-money caps the
    max loss at the strike width (minus net credit), converting an open-ended
    tail into a defined-risk vertical.

        short put  Ks  ->  buy a long put  at Ks - wing_offset  (protects downside)
        short call Ks  ->  buy a long call at Ks + wing_offset  (protects upside)

    Returns ``[short_leg, long_wing]``. The long wing is priced from the short
    leg's market context (spot/vol/r/q) via ``engine.pricing`` and carries the
    mirrored (long) quantity.
    """
    if short_leg.quantity >= 0:
        raise ValueError("to_defined_risk expects a SHORT leg (quantity < 0)")

    kind = short_leg.kind.lower()
    if kind == "put":
        wing_strike = short_leg.strike - wing_offset
    elif kind == "call":
        wing_strike = short_leg.strike + wing_offset
    else:
        raise ValueError(f"kind must be 'call' or 'put', got {short_leg.kind!r}")
    if wing_strike <= 0:
        raise ValueError("wing_offset too large: non-positive wing strike")

    wing_price = pricing.price(
        short_leg.spot, wing_strike, short_leg.T,
        short_leg.r, short_leg.q, short_leg.vol, kind,
    )
    long_wing = Position(
        underlying=short_leg.underlying,
        strike=wing_strike,
        expiry_days=short_leg.expiry_days,
        kind=kind,
        quantity=-short_leg.quantity,   # mirror to LONG, same magnitude
        entry_price=wing_price,
        spot=short_leg.spot,
        r=short_leg.r,
        q=short_leg.q,
        vol=short_leg.vol,
        multiplier=short_leg.multiplier,
    )
    return [short_leg, long_wing]
