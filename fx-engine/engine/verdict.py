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
number it would have needed. There are ten:

    NOTHING_MEASURED         a planned payoff is not an edge
    RESTS_ON_ONE_PERIOD      drop its best stretch and the edge is gone
    COST_NOT_MEASURED        you accepted my default costs instead of yours
    COSTS_DISAGREE           the backtest was run on cheaper costs than these
    COSTS_BELOW_RECORDED     your own data recorded a wider spread than this
    TOO_FEW_TRADES           the sample cannot support the claim
    NO_WIN_RATE_SAVES_IT     cost >= the whole target; p* >= 100%
    NEGATIVE_NET_EXPECTANCY  measured, and it loses
    NOT_SIGNIFICANT          positive, but inside the noise once corrected
    GOVERNED_BY_ASSUMPTION   the ambiguous-bar rule decided it, not the data

RESTS_ON_ONE_PERIOD is the one that a significance test cannot reach. The
t-stat treats trades as independent draws from one distribution, so a sample
whose edge lives entirely in a single stretch of the file passes it on the
strength of that stretch: 33 trades with 95% of the profit in the final month
of three scored t=4.02 against 2.87 required. Nothing else in this module looks
at WHEN a trade happened.

The test is deliberately threshold-free, because any cutoff on "how
concentrated is too concentrated" would be invented. The file is cut into equal
stretches, the one contributing the most profit is removed, and the SIGN of
what remains is what decides. An edge that needs its best stretch to stay
positive is a description of that stretch. Note what this does NOT do: it does
not ask whether the remainder is significant, because dropping a quarter of any
sample costs it a quarter of its power, and a rule that fires on every sample
near the trade floor would say nothing about concentration at all.

COSTS_BELOW_RECORDED is the only refusal here that argues from the DATA rather
than from arithmetic. Most price files cannot contradict a cost assumption --
they hold no cost -- so `costs_confirmed` has to be taken on trust. A
MetaTrader export carries a per-bar spread column, and that turns the trust
into a measurement: whatever the broker advertises, a round turn declared below
the spread the file itself recorded on those very bars cannot have been
available on them. Note the direction. This can only ever call a cost
assumption too CHEAP; it can never confirm one, because commission and slippage
are not in the file either.

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
from engine.stats import trades_needed, trades_to_separate
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
COSTS_BELOW_RECORDED = "COSTS_BELOW_RECORDED"
RESTS_ON_ONE_PERIOD = "RESTS_ON_ONE_PERIOD"

#: Equal stretches of the file the trades are split across for that test. Four
#: is a choice, not a derivation: enough slices that one regime does not span
#: them all, few enough that each is long enough to hold a run of trades.
SEGMENTS = 4

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

    HOW OPTIMISTIC, MEASURED. `demo.py` runs this pipeline over random walks,
    where nothing can be true. At zero cost, 28% of pattern-tests come back
    tradeable judged as a single hypothesis, and 13% still do after the
    Bonferroni correction — against a nominal 5% and 0.2%. The overlap is why.
    So a PASS here is a floor on doubt, never proof; only out-of-sample data
    settles it. A FAIL, by contrast, is decisive, because the test was already
    tilted the other way.
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
    #: Per-stretch (n, win_rate, net_pips_per_trade, total_net_pips), in time
    #: order. Printed on every verdict: a concentration this does not refuse is
    #: still something you should be able to see.
    segments: list = field(default_factory=list)

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
        # the GROSS view is printed, not the net one. Both give the same net
        # expectancy -- that identity is proved in `_gross_view` -- but the net
        # view carries cost_pips=0, so printing it shows "cost 0.00 pips" on a
        # result that was costed, which reads as if the cost had been forgotten.
        lines = [f"{self.name or 'verdict'}: {head}  ({self.n_trades} trades)",
                 self.gross.summary()]
        lines.append(
            f"  short by {self.win_rate_gap:+.1%} of win rate; "
            f"{self.flipped_by_cost} winning trade(s) lost money after cost")
        lines.append(
            f"  t-stat {self.t_stat:>8.2f} vs {self.t_required:.2f} required "
            f"across {self.n_hypotheses} hypotheses")
        if self.segments:
            shown = "  ".join(
                f"{n}@{net:+.1f}" if n else "--" for n, _w, net, _t in self.segments)
            lines.append(f"  by period        {shown}   "
                         f"(trades@net pips each, earliest first)")
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


def segment_trades(res: BacktestResult, k: int = SEGMENTS) -> list:
    """Split the trades into `k` equal stretches of the file, in time order.

    Equal stretches of the INDEX SPAN, not equal numbers of trades. The
    question is whether the edge is confined to a period, and a split by trade
    count would hide exactly that by construction -- it would give every
    stretch the same number of trades however they were bunched.

    The `min(..., k - 1)` below cannot fire and is kept as a guard rather than
    a branch: with `width = (hi - lo + 1) / k`, the largest index gives
    `k(hi - lo) / (hi - lo + 1)`, which is strictly below `k`. It earns its
    place only against a later edit to `width` -- `(hi - lo) / k` would put the
    last trade one past the end -- so its mutant is recorded as equivalent in
    mutation/README.md instead of sitting in the table as a known survivor.
    """
    if k < 2:
        raise ValueError(f"need at least 2 segments to compare, got {k}")
    if not res.trades:
        return []
    lo = min(t.entry_index for t in res.trades)
    hi = max(t.entry_index for t in res.trades)
    width = (hi - lo + 1) / k
    buckets = [[] for _ in range(k)]
    for t in res.trades:
        j = min(int((t.entry_index - lo) / width), k - 1)
        buckets[j].append(t)
    out = []
    for b in buckets:
        if not b:
            out.append((0, 0.0, 0.0, 0.0))
            continue
        total = sum(t.net_pips for t in b)
        wins = sum(1 for t in b if t.net_pips > 0)
        out.append((len(b), wins / len(b), total / len(b), total))
    return out


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
          alpha: float = ALPHA, evidence: tuple = None,
          recorded_spread_pips: float = None) -> Verdict:
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

    ``recorded_spread_pips`` is the one part of that which a file CAN check,
    when it carried a spread column: pass `Series.spread_pips(q)` and a round
    turn declared below it is refused. It only ever catches a cost assumption
    that is too cheap — commission and slippage are not in the file, so no
    value here can confirm a cost model, only contradict one.
    """
    names = {t.name for t in res.trades} or ({name} if name else set())
    if n_hypotheses is None:
        n_hypotheses = max(len(names), 1) * PARAMETER_COUNT

    label = name or (sorted(names)[0] if len(names) == 1 else "mixed")
    # `evidence` is for a rule this engine has never heard of, where NEVER_TESTED
    # would be the wrong thing to say: that means the literature did not test a
    # KNOWN pattern, not that there is no literature to have tested anything.
    ev, ev_detail = evidence or EVIDENCE.get(
        label, (NEVER_TESTED, "no entry for this pattern"))

    net = res.expectancy()
    gross = _gross_view(res)
    net_pips = [t.net_pips for t in res.trades]
    t_stat = edge_t_stat(net_pips)
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

    if recorded_spread_pips is not None \
            and costs.round_turn_pips < recorded_spread_pips - 1e-9:
        v.refusals.append(Refusal(
            COSTS_BELOW_RECORDED,
            f"the round turn is set to {costs.round_turn_pips:.2f} pips, but "
            f"your own data recorded a spread of {recorded_spread_pips:.2f} "
            f"pips on these bars — and the spread is only part of a round turn, "
            f"before any commission or slippage. A cost this low was not "
            f"available on the bars the edge was measured on"))

    segments = segment_trades(res)
    v.segments = segments
    total = sum(t.net_pips for t in res.trades)
    # Only asked of a result that CLEARED significance. That is the whole point
    # of the rule -- it catches what the t-stat misses -- and asking it of a
    # thin edge would make it fire on insignificance instead: when the total is
    # small against the per-trade spread, dropping any stretch flips the sign,
    # and the "best" one is not special. NOT_SIGNIFICANT says that, better.
    if total > 0 and len(segments) >= 2 and t_stat >= t_req:
        best = max(range(len(segments)), key=lambda j: segments[j][3])
        rest_n = res.n - segments[best][0]
        rest = total - segments[best][3]
        share = segments[best][3] / total
        if rest <= 0 and rest_n > 0:
            v.refusals.append(Refusal(
                RESTS_ON_ONE_PERIOD,
                f"one of {len(segments)} equal stretches of the file holds "
                f"{share:.0%} of the net profit, and without it the remaining "
                f"{rest_n} trade(s) come to {rest:+.1f} pips. The edge is a "
                f"description of that stretch, not of the pair — and the t-stat "
                f"cannot see this, because it treats every trade as the same "
                f"draw whenever it happened"))
        elif share > 0.5:
            v.notes.append(
                f"{share:.0%} of the net profit came from one of "
                f"{len(segments)} equal stretches of the file "
                f"({segments[best][0]} of {res.n} trades). The rest still "
                f"clears {rest / rest_n:+.2f} pips/trade, so this is not "
                f"refused — but the sample is not the independent draws the "
                f"t-stat assumes it is")

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
        # the floor is generic; this is what THIS claim would actually take
        sep = trades_to_separate(gross.win_rate, v.win_rate_needed)
        need = ("more than this module will search for — the claimed win rate is "
                "not above the one it has to beat" if sep == math.inf else
                f"about {max(sep, float(floor)):,.0f} before "
                f"{gross.win_rate:.0%} is distinguishable from the "
                f"{v.win_rate_needed:.0%} it must beat")
        v.refusals.append(Refusal(
            TOO_FEW_TRADES,
            f"{res.n} trades, {floor} required — {why}. On these numbers: {need}"))

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
        sd = statistics.stdev(net_pips) if len(net_pips) > 1 else 0.0
        want = trades_needed(net.net, sd, t_req)
        v.refusals.append(Refusal(
            NOT_SIGNIFICANT,
            f"positive at {net.net:+.2f} pips/trade but t={t_stat:.2f} against "
            f"{t_req:.2f} required once corrected for {n_hypotheses} hypotheses. "
            f"A {net.net:+.2f} pip edge against a {sd:.1f} pip standard deviation "
            f"needs about {want:,.0f} trades to establish, and you have {res.n}. "
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
