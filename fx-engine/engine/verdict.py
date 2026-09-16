"""The verdict — the module that is allowed to say no, and has to say why.

Everything upstream produces numbers. This is the piece that turns them into a
decision, and the decision it is most often correct to reach is DO NOT TRADE
THIS. That refusal is worth more than any signal in the engine, because the
measured base rate says most of what reaches this point does not survive costs:

    FXCM, 43 million real trades      61% win rate on EUR/USD and still losing
    Ben-David/Birru/Prokopenya        62.8% of trades won, traders still lost
    ESMA / FCA / AMF                  74-89% of retail accounts lose money

Those are not people who picked bad patterns. They are people who won more
often than they lost and paid more when they lost than they made when they won.
Nothing in a chart prevents that. Arithmetic does, and only if it is applied
before the trade rather than explained after it.

SO EVERY REFUSAL HERE HAS A NAME. "This does not look good" is useless; you
cannot argue with it, fix it, or check whether it was right. A refusal in this
module is a code you can look up, with the number that triggered it and the
number it would have needed. There are seven:

    NOTHING_MEASURED         a planned payoff is not an edge
    COST_NOT_MEASURED        you accepted my default costs instead of yours
    COSTS_DISAGREE           the backtest was run on cheaper costs than these
    TOO_FEW_TRADES           the sample cannot support the claim
    NO_WIN_RATE_SAVES_IT     cost >= the whole target; p* >= 100%
    NEGATIVE_NET_EXPECTANCY  measured, and it loses
    NOT_SIGNIFICANT          positive, but inside the noise once corrected
    GOVERNED_BY_ASSUMPTION   the ambiguous-bar rule decided it, not the data

COSTS_DISAGREE is the quiet one. Costs are charged inside the backtest, so this
function cannot re-charge them without double counting — which means a result
measured at 0.5 pips can be handed to a verdict configured for 2.5 and nothing
would notice. The trades carry what they were charged; comparing it against the
model being judged is a two-line check that closes a hole nothing else can see.

THE THREE THAT PEOPLE ARGUE WITH, ANSWERED IN ADVANCE.

"I'll fix it with position sizing." No. Expectation is linear in size, so
scaling a negative number by any positive size leaves it negative — martingale,
grid, averaging down and "recovery" all vary size and none changes the sign.
What they change is the SHAPE: they raise the win rate and move the loss into a
rarer, larger event. That is why 95%-win-rate robots exist and then stop
existing. `expectancy.sizing_cannot_fix` states it in code; NEGATIVE_NET_
EXPECTANCY quotes the resulting number so the claim is concrete.

"The win rate is high." A 90% win rate breaks even at a payoff of 0.111. The
required win rate p* = (L+c)/(W+L) is reported on every verdict, refused or
not, next to the measured one, because the gap between those two is the entire
question and the win rate alone is half an equation.

"It was profitable in the backtest." Then the question is whether it was
profitable by more than its own standard error, after correcting for how many
specifications were tried. `NOT_SIGNIFICANT` applies that test. It is a FLOOR,
not a true correction — see `n_hypotheses` — because the honest denominator
includes every variant you looked at and discarded, and only you know that
number.

WHAT A PASS HERE IS NOT. It is not a prediction, and it is not a bound on your
loss. A stop is an instruction, not a guarantee: on 2015-01-15 EUR/CHF moved
more than 30% with no quotes in between and clients owed money beyond their
equity. `suggested_risk_fraction` is capped hard for that reason and is still
not a floor under the outcome.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from statistics import NormalDist

from engine.backtest import AMBIGUITY_WARN_FRAC, BacktestResult
from engine.costs import CostModel, pip_size
from engine.expectancy import (Expectancy, kelly_fraction, required_win_rate,
                               sizing_cannot_fix)
from models.patterns import (EVIDENCE, FAILS_RANDOM_NULL, NEVER_TESTED,
                             PARAMETER_COUNT, TESTED_AND_FAILED)

# ---------------------------------------------------------------------------
# Refusal codes. Named so they can be looked up, argued with, and checked later.
# ---------------------------------------------------------------------------

NOTHING_MEASURED = "NOTHING_MEASURED"
COST_NOT_MEASURED = "COST_NOT_MEASURED"
TOO_FEW_TRADES = "TOO_FEW_TRADES"
NO_WIN_RATE_SAVES_IT = "NO_WIN_RATE_SAVES_IT"
NEGATIVE_NET_EXPECTANCY = "NEGATIVE_NET_EXPECTANCY"
NOT_SIGNIFICANT = "NOT_SIGNIFICANT"
GOVERNED_BY_ASSUMPTION = "GOVERNED_BY_ASSUMPTION"
COSTS_DISAGREE = "COSTS_DISAGREE"

#: One-sided significance level before the multiple-testing correction.
ALPHA = 0.05

#: Trades required before a win rate means anything. At n=30 the standard error
#: of a 50% win rate is 9.1 points, so the 95% interval spans roughly +/-18
#: points — wide enough to contain both "good system" and "coin flip". 30 is
#: therefore a FLOOR below which the question cannot be asked, not a level at
#: which it is answered.
MIN_TRADES = 30

#: Patterns the literature has already tested and rejected need a bigger sample
#: than the study that rejected them, not a nominally positive one. At n=100 the
#: standard error is 5.0 points, which is the first point at which a 10-point
#: claim is distinguishable from noise at all.
MIN_TRADES_CONTESTED = 100

#: Evidence labels that trigger the higher bar.
CONTESTED = (TESTED_AND_FAILED, FAILS_RANDOM_NULL)

#: Hard ceiling on risk per trade as a fraction of equity, whatever the maths
#: says. It binds in practice and is meant to: the edge estimate behind any
#: larger number is itself estimated on a few hundred trades.
MAX_RISK_FRACTION = 0.01

#: Kelly is divided by this before the cap applies. Full Kelly assumes the edge
#: estimate is exact; on an FX sample it routinely over-levers several times.
KELLY_DIVISOR = 4.0


def required_t(n_hypotheses: int, alpha: float = ALPHA) -> float:
    """The t-statistic a positive result must clear, Bonferroni-corrected.

    One-sided, because a strategy that loses significantly is not a discovery.
    Normal rather than Student because `MIN_TRADES` already keeps n >= 30, where
    the difference is smaller than the error in the correction itself.
    """
    if n_hypotheses < 1:
        raise ValueError(f"n_hypotheses must be >= 1, got {n_hypotheses!r}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
    return NormalDist().inv_cdf(1.0 - alpha / n_hypotheses)


def edge_t_stat(net_pips: list) -> float:
    """mean / (sd / sqrt(n)) over the per-trade NET results.

    Optimistic by construction, and worth saying so: it treats trades as
    independent, which overlapping positions in one pair are not, and it uses
    the sample's own standard deviation, which a short sample understates. A
    result that fails this test has failed a test that was rigged in its favour.
    """
    n = len(net_pips)
    if n < 2:
        return 0.0
    mean = statistics.fmean(net_pips)
    sd = statistics.stdev(net_pips)
    if sd == 0.0:
        # every trade returned exactly the same amount: degenerate, not skill
        return math.inf if mean > 0 else 0.0
    return mean / (sd / math.sqrt(n))


def _using_default_costs(costs: CostModel) -> bool:
    """True when the caller never entered their own numbers.

    Read from the dataclass fields rather than repeated literals, so this cannot
    drift away from the defaults it is testing for.
    """
    f = CostModel.__dataclass_fields__
    return (costs.round_turn_pips == f["round_turn_pips"].default
            and costs.swap_markup_annual == f["swap_markup_annual"].default)


@dataclass(frozen=True)
class Refusal:
    """One named reason, with the number that caused it."""
    code: str
    detail: str

    def __str__(self) -> str:
        return f"{self.code}: {self.detail}"


@dataclass
class Verdict:
    """The decision, everything behind it, and everything against it."""

    name: str
    tradeable: bool
    net: Expectancy                 # measured, costs already inside the trades
    gross: Expectancy               # before costs, with the cost charged visibly
    refusals: list = field(default_factory=list)
    n_trades: int = 0
    t_stat: float = 0.0
    t_required: float = 0.0
    n_hypotheses: int = 1
    flipped_by_cost: int = 0
    evidence: str = NEVER_TESTED
    evidence_detail: str = ""
    suggested_risk_fraction: float = 0.0
    notes: list = field(default_factory=list)

    @property
    def codes(self) -> list:
        return [r.code for r in self.refusals]

    @property
    def win_rate_needed(self) -> float:
        """p* = (L + c) / (W + L), on the GROSS averages with the cost explicit.

        The number the whole module exists to put next to the measured win rate.
        """
        return self.gross.break_even_win_rate

    @property
    def win_rate_gap(self) -> float:
        """Points of win rate missing. Negative means there is room to spare."""
        return self.win_rate_needed - self.gross.win_rate

    def summary(self) -> str:
        head = "TRADEABLE" if self.tradeable else "STAND DOWN"
        lines = [f"{self.name or 'verdict'}: {head}  ({self.n_trades} trades)",
                 self.net.summary()]
        lines.append(
            f"  required win rate {self.win_rate_needed:>7.1%} vs measured "
            f"{self.gross.win_rate:.1%} gross  ({self.win_rate_gap:+.1%} short)")
        lines.append(
            f"  of which cost alone{self.gross.cost_penalty_points:>7.1%}"
            f"   ({self.flipped_by_cost} winning trade(s) lost money after cost)")
        lines.append(
            f"  t-stat {self.t_stat:>8.2f} vs {self.t_required:.2f} required "
            f"across {self.n_hypotheses} hypotheses")
        lines.append(f"  evidence: {self.evidence} — {self.evidence_detail}")
        for r in self.refusals:
            lines.append(f"  REFUSED {r}")
        for n in self.notes:
            lines.append(f"  note: {n}")
        if self.tradeable:
            lines.append(
                f"  risk at most {self.suggested_risk_fraction:.2%} of equity per "
                f"trade — and a stop does not bound the loss, so that is a plan, "
                f"not a floor")
        return "\n".join(lines)


def _gross_view(res: BacktestResult) -> Expectancy:
    """What the trades looked like BEFORE cost, with the mean cost charged.

    Wins and losses are re-classified on GROSS pips, so a trade that won and
    then lost money to cost counts as a win here and a loss in `net`. That
    difference is `flipped_by_cost`, and it is the FXCM result in miniature.

    THE TWO VIEWS AGREE ON EXPECTANCY AND DISAGREE ON WIN RATE. That sounds
    like a contradiction and is not: mean(gross) - mean(cost) is mean(net) by
    construction, so the pips-per-trade figure is identical either way and it
    does not matter which one the verdict tests. What the cost moves is the
    PARTITION — which trades count as wins — and therefore the win rate, the
    payoff ratio and the bar you have to clear. Cost does not subtract from
    your edge twice; it raises the requirement, which is the harder thing to
    see and the reason both views are reported side by side.
    """
    if not res.trades:
        return Expectancy(win_rate=0.0, avg_win_pips=0.0, avg_loss_pips=0.0)
    wins = [t.pips for t in res.trades if t.pips > 0]
    losses = [t.pips for t in res.trades if t.pips <= 0]
    return Expectancy(
        win_rate=len(wins) / len(res.trades),
        avg_win_pips=statistics.fmean(wins) if wins else 0.0,
        avg_loss_pips=-statistics.fmean(losses) if losses else 0.0,
        cost_pips=statistics.fmean([t.cost_pips for t in res.trades]))


def judge(res: BacktestResult, costs: CostModel, *, name: str = "",
          n_hypotheses: int = None, costs_confirmed: bool = False,
          alpha: float = ALPHA) -> Verdict:
    """Decide whether a measured result is worth trading, and name every no.

    ``n_hypotheses`` is the multiple-testing denominator. Left unset it is
    derived as (distinct pattern names) x `PARAMETER_COUNT`, which is a FLOOR:
    the honest number also counts every pair, timeframe, stop placement and
    parameter value you tried and set aside, and only you know that. Passing a
    larger number makes the test harder, which is the direction honesty runs in.

    ``costs_confirmed`` is the one thing this function cannot check for you. The
    broker's swap markup is not published, it is charged in both directions, and
    it decides more than any indicator here — so accepting the default is a
    refusal until you say you measured it.
    """
    names = {t.name for t in res.trades} or ({name} if name else set())
    if n_hypotheses is None:
        n_hypotheses = max(len(names), 1) * PARAMETER_COUNT

    label = name or (sorted(names)[0] if len(names) == 1 else "mixed")
    ev, ev_detail = EVIDENCE.get(label, (NEVER_TESTED, "no entry for this pattern"))

    net = res.expectancy()
    gross = _gross_view(res)
    t_stat = edge_t_stat([t.net_pips for t in res.trades])
    t_req = required_t(n_hypotheses, alpha)
    flipped = sum(1 for t in res.trades if t.pips > 0 and t.net_pips <= 0)

    v = Verdict(name=label, tradeable=False, net=net, gross=gross,
                n_trades=res.n, t_stat=t_stat, t_required=t_req,
                n_hypotheses=n_hypotheses, flipped_by_cost=flipped,
                evidence=ev, evidence_detail=ev_detail)

    if res.n == 0:
        v.refusals.append(Refusal(
            NOTHING_MEASURED,
            "no trades. A planned target/stop ratio is an intention, not an "
            "edge — it assumes the target is reached first and that both levels "
            "fill where they are written"))
        return v

    if _using_default_costs(costs) and not costs_confirmed:
        v.refusals.append(Refusal(
            COST_NOT_MEASURED,
            f"the cost model is still at its defaults "
            f"({costs.round_turn_pips} pips round turn, "
            f"{costs.swap_markup_annual:.2%}/yr markup), so this is my guess at "
            f"your broker rather than your broker. {costs.measured_markup_hint}. "
            f"Pass costs_confirmed=True once you have"))

    cheapest = min(t.cost_pips for t in res.trades)
    if cheapest < costs.round_turn_pips - 1e-9:
        v.refusals.append(Refusal(
            COSTS_DISAGREE,
            f"the backtest charged as little as {cheapest:.2f} pips per trade, "
            f"but this cost model puts the round turn ALONE at "
            f"{costs.round_turn_pips:.2f} before any financing. The result was "
            f"measured on cheaper costs than the ones being judged — re-run the "
            f"backtest with this CostModel rather than adjusting afterwards"))

    floor = MIN_TRADES_CONTESTED if ev in CONTESTED else MIN_TRADES
    if res.n < floor:
        why = (f"the literature already tested this pattern and rejected it, so "
               f"the sample has to beat that study rather than merely be positive"
               if ev in CONTESTED else
               f"below this the standard error of the win rate is wider than any "
               f"edge being claimed")
        v.refusals.append(Refusal(
            TOO_FEW_TRADES,
            f"{res.n} trades, {floor} required — {why}"))

    if v.win_rate_needed >= 1.0:
        v.refusals.append(Refusal(
            NO_WIN_RATE_SAVES_IT,
            f"cost of {gross.cost_pips:.2f} pips equals or exceeds the whole "
            f"{gross.avg_win_pips:.1f}+{gross.avg_loss_pips:.1f} pip range, so "
            f"the required win rate is {v.win_rate_needed:.0%}. No win rate, "
            f"entry filter or sizing rule fixes this; the target is too small "
            f"for what it costs to reach"))

    if net.net <= 0:
        per_100 = sizing_cannot_fix(net.net, 100.0)
        v.refusals.append(Refusal(
            NEGATIVE_NET_EXPECTANCY,
            f"{net.net:+.2f} pips per trade after cost. Needs "
            f"{v.win_rate_needed:.1%} and measured {gross.win_rate:.1%} gross "
            f"({v.win_rate_gap:+.1%}). Scaling does not help: 100 units of this "
            f"returns {per_100:+.1f} pips, because expectation is linear in size"))
    elif t_stat < t_req:
        v.refusals.append(Refusal(
            NOT_SIGNIFICANT,
            f"positive at {net.net:+.2f} pips/trade but t={t_stat:.2f} against "
            f"{t_req:.2f} required once corrected for {n_hypotheses} hypotheses. "
            f"This test already favours the strategy — it treats overlapping "
            f"trades as independent — so failing it is decisive"))

    if res.ambiguous_frac > AMBIGUITY_WARN_FRAC:
        v.refusals.append(Refusal(
            GOVERNED_BY_ASSUMPTION,
            f"{res.ambiguous_frac:.0%} of trades resolved on a bar holding BOTH "
            f"the stop and the target, where OHLC cannot say which came first. "
            f"The stop-first assumption decided this result, not the data — "
            f"re-run on higher-resolution bars before believing either sign"))

    v.notes.extend(res.warnings)
    if ev in CONTESTED and not v.refusals:
        v.notes.append(
            f"this pattern's published evidence is negative ({ev_detail}) — a "
            f"positive result here contradicts it and should be treated as a "
            f"claim to re-test out of sample, not as a confirmation")

    v.tradeable = not v.refusals
    if v.tradeable:
        k = kelly_fraction(net.win_rate, net.payoff_ratio) / KELLY_DIVISOR
        v.suggested_risk_fraction = min(k, MAX_RISK_FRACTION)
    return v


def judge_planned(detection, costs: CostModel, *, costs_confirmed: bool = False):
    """What an UN-BACKTESTED pattern would need to be worth taking.

    Returns a Verdict that always refuses with NOTHING_MEASURED — nothing has
    been measured — but fills in the number that makes the refusal useful: the
    win rate this target and stop would require at your costs. It is the fastest
    way to find out that a 5-pip scalp needs 65%, before writing any code.
    """
    psize = pip_size(costs.pair)
    w = detection.reward / psize
    l = detection.risk / psize
    c = costs.round_turn_pips
    ev, ev_detail = EVIDENCE.get(detection.name,
                                 (NEVER_TESTED, "no entry for this pattern"))
    planned = Expectancy(win_rate=0.0, avg_win_pips=w, avg_loss_pips=l,
                         cost_pips=c)
    v = Verdict(name=detection.name, tradeable=False, net=planned, gross=planned,
                n_trades=0, t_stat=0.0, t_required=required_t(PARAMETER_COUNT),
                n_hypotheses=PARAMETER_COUNT, evidence=ev,
                evidence_detail=ev_detail)
    v.refusals.append(Refusal(
        NOTHING_MEASURED,
        f"nothing has been backtested. This {w:.0f}/{l:.0f} pip plan would need "
        f"a {required_win_rate(w, l, c):.1%} win rate at {c} pips of cost — "
        f"{planned.cost_penalty_points:.1%} of that is the cost alone. A planned "
        f"payoff of {detection.planned_payoff:.2f} assumes the target is reached "
        f"before the stop, which is the assumption the backtest exists to test"))
    if not costs_confirmed and _using_default_costs(costs):
        v.refusals.append(Refusal(
            COST_NOT_MEASURED,
            f"costs are at defaults; the figure above is my guess at your broker. "
            f"{costs.measured_markup_hint}"))
    return v
