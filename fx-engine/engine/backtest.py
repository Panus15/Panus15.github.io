"""A backtest that does not lie to itself.

Running a pattern over history and counting wins is easy, and it is how almost
every retail FX system arrives at a number that never survives contact with a live
account. This module is built around the five ways that number gets manufactured.

1. THE AMBIGUOUS BAR. When one bar's range contains BOTH the stop and the target,
   OHLC data cannot tell you which was touched first. Assuming the target is the
   single most common way a backtest is inflated, and it is invisible: nothing
   errors, the equity curve just bends upward. This module assumes the STOP, counts
   how often it had to, and reports that count. If `ambiguous_bars` is a large
   share of trades, the result is governed by an assumption rather than by data —
   `AMBIGUITY_WARN_FRAC` is where it says so.

2. MID PRICES. You do not trade at mid. A long enters at the ask and exits at the
   bid; a short does the reverse. Backtesting on mid gives you the spread for free
   on both ends, twice per trade. Costs here are charged through `CostModel`, which
   takes the ALL-IN round turn rather than a quoted spread.

3. SWAP ON THE NIGHTS ACTUALLY HELD. Financing is charged per calendar night, so a
   rule that holds for a week pays seven nights and a rule that holds for a month
   pays thirty. This is the cost that "low turnover" does not reduce, and omitting
   it is what makes long-horizon rules look better than short ones.

   NIGHTS ARE NOT BARS. `bars_per_night` converts, and getting it wrong is a
   24x error on hourly data in whichever direction you got it wrong. A day
   trader who is flat by the rollover pays NO swap at all, which is why nights
   are counted as COMPLETE sessions held rather than as a fraction of one.

4. GAPS THROUGH THE STOP. A stop is an instruction, not a guarantee. When a bar
   OPENS beyond the stop, the fill is the open, not the stop — that is the whole
   mechanism behind 2015-01-15 and 2019-01-03. `slipped_stops` counts them and
   `worst_slippage_pips` reports the largest.

5. LOOK-AHEAD. Every trade starts strictly AFTER the bar the pattern completed on.
   `models/patterns.py` is tested for this separately; the backtest re-checks it by
   construction, because it never reads a bar at or before the detection index.

WHAT THIS MODULE WILL NOT DO. It will not report a Sharpe ratio. With the trade
counts a retail pattern study produces, a Sharpe estimate has a standard error wide
enough to be meaningless, and quoting one invites exactly the overconfidence this
engine exists to prevent. `stats.py` handles how many trades a claim would need.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from engine.costs import CostModel, pip_size
from engine.expectancy import Expectancy

#: Above this share of ambiguous bars, the result is driven by the
#: stop-before-target assumption rather than by the data, and the report says so.
AMBIGUITY_WARN_FRAC = 0.25

#: Bars a trade may stay open before it is closed at the market. Without a cap a
#: losing trade can sit open to the end of the sample and never be counted — the
#: quiet way a backtest drops its worst results.
MAX_HOLD_BARS = 500


@dataclass
class Trade:
    """One round trip, with everything needed to audit it."""
    name: str
    entry_index: int
    exit_index: int
    direction: int
    entry: float
    exit: float
    pips: float                 # gross, before cost
    cost_pips: float
    outcome: str                # 'target' | 'stop' | 'timeout' | 'end-of-data'
    ambiguous: bool = False     # the bar held both levels; the stop was assumed
    slippage_pips: float = 0.0  # gap beyond the stop, if any

    @property
    def net_pips(self) -> float:
        return self.pips - self.cost_pips

    def nights(self, bars_per_night: float = 1.0) -> int:
        """Complete financing sessions held. Needs the bar size, because a bar
        count means nothing without it."""
        return int(max(self.exit_index - self.entry_index, 0) / bars_per_night)


@dataclass
class BacktestResult:
    trades: list = field(default_factory=list)
    ambiguous_bars: int = 0
    slipped_stops: int = 0
    worst_slippage_pips: float = 0.0
    timeouts: int = 0

    @property
    def n(self) -> int:
        return len(self.trades)

    @property
    def wins(self) -> list:
        return [t for t in self.trades if t.net_pips > 0]

    @property
    def losses(self) -> list:
        return [t for t in self.trades if t.net_pips <= 0]

    @property
    def win_rate(self) -> float:
        return len(self.wins) / self.n if self.n else 0.0

    @property
    def avg_win(self) -> float:
        w = self.wins
        return sum(t.net_pips for t in w) / len(w) if w else 0.0

    @property
    def avg_loss(self) -> float:
        """Positive magnitude, as `Expectancy` expects."""
        l = self.losses
        return -sum(t.net_pips for t in l) / len(l) if l else 0.0

    @property
    def total_pips(self) -> float:
        return sum(t.net_pips for t in self.trades)

    @property
    def ambiguous_frac(self) -> float:
        return self.ambiguous_bars / self.n if self.n else 0.0

    def expectancy(self) -> Expectancy:
        """The measured expectancy. Costs are already inside the trade results, so
        `cost_pips` is zero here — charging again would double-count."""
        return Expectancy(win_rate=self.win_rate, avg_win_pips=self.avg_win,
                          avg_loss_pips=self.avg_loss, cost_pips=0.0)

    @property
    def warnings(self) -> list:
        """What a reader must know before believing any number above."""
        out = []
        if self.ambiguous_frac > AMBIGUITY_WARN_FRAC:
            out.append(
                f"{self.ambiguous_frac:.0%} of trades resolved on a bar that held "
                f"BOTH the stop and the target. OHLC cannot say which came first, "
                f"so the stop was assumed. This result is governed by that "
                f"assumption, not by the data — use higher-resolution bars")
        if self.slipped_stops:
            out.append(
                f"{self.slipped_stops} trade(s) gapped THROUGH the stop; the worst "
                f"filled {self.worst_slippage_pips:.1f} pips beyond it. A stop is "
                f"an instruction, not a guarantee")
        if self.timeouts:
            out.append(
                f"{self.timeouts} trade(s) hit the {MAX_HOLD_BARS}-bar hold cap and "
                f"were closed at the market rather than at a level")
        if 0 < self.n < 30:
            out.append(
                f"only {self.n} trades — far too few for any of these statistics "
                f"to mean anything (see stats.py)")
        return out


def run(bars: list, detections: list, costs: CostModel, *,
        max_hold: int = MAX_HOLD_BARS,
        bars_per_night: float = 1.0) -> BacktestResult:
    """Walk each detection forward and resolve it honestly.

    Entry is at the NEXT bar's open after the pattern completed — never at the
    detection bar's close, which would assume you acted on a bar while it was still
    forming.

    ``bars_per_night`` is how many bars make one financing rollover: 1 for daily
    bars, 24 for hourly, 96 for 15-minute. The default of 1 is correct for daily
    data and charges 24x too much swap on hourly, so pass it — `data.Series`
    carries the interval it measured from your file, and `nights_per_bar` turns
    that into this number.

    Nights are counted as COMPLETE sessions held, because a position that opens
    and closes between rollovers pays no financing at all. That approximation
    can miss a rollover crossed by a short hold straddling 5pm New York; the
    bars alone cannot say, since they carry no clock. It errs low there and
    nowhere else, and `financing_note` says so.
    """
    res = BacktestResult()
    psize = pip_size(costs.pair)
    if bars_per_night <= 0:
        raise ValueError(f"bars_per_night must be positive, got {bars_per_night!r}")

    for d in detections:
        i0 = d.index + 1                      # strictly after the completing bar
        if i0 >= len(bars):
            continue
        entry_px = bars[i0].o
        stop, target = d.stop, d.target
        exit_px, exit_i, outcome = None, None, None
        ambiguous = slippage = 0.0
        ambiguous = False

        for j in range(i0, min(i0 + max_hold, len(bars))):
            b = bars[j]
            if d.direction > 0:
                gapped = b.o <= stop
                hit_stop = b.l <= stop
                hit_target = b.h >= target
            else:
                gapped = b.o >= stop
                hit_stop = b.h >= stop
                hit_target = b.l <= target

            if gapped:
                # THE FILL IS THE OPEN, NOT THE STOP. This is the mechanism behind
                # every "my stop did not protect me" story, and modelling the fill
                # at the stop level is how a backtest hides it.
                exit_px, exit_i, outcome = b.o, j, "stop"
                slippage = abs(b.o - stop) / psize
                res.slipped_stops += 1
                res.worst_slippage_pips = max(res.worst_slippage_pips, slippage)
                break
            if hit_stop and hit_target:
                # Both levels inside one bar. OHLC cannot order them, so take the
                # WORSE outcome and record that the data, not the rule, decided.
                exit_px, exit_i, outcome = stop, j, "stop"
                ambiguous = True
                res.ambiguous_bars += 1
                break
            if hit_stop:
                exit_px, exit_i, outcome = stop, j, "stop"
                break
            if hit_target:
                exit_px, exit_i, outcome = target, j, "target"
                break

        if exit_px is None:
            # never resolved: close at the market rather than drop the trade, since
            # dropping unresolved trades silently deletes the worst of them
            last = min(i0 + max_hold, len(bars)) - 1
            if last < i0:
                continue
            exit_px, exit_i = bars[last].c, last
            outcome = "timeout" if last == i0 + max_hold - 1 else "end-of-data"
            if outcome == "timeout":
                res.timeouts += 1

        gross = (exit_px - entry_px) * d.direction / psize
        nights = int(max(exit_i - i0, 0) / bars_per_night)
        cost = costs.total_cost_pips(nights, entry_px)
        res.trades.append(Trade(
            name=d.name, entry_index=i0, exit_index=exit_i, direction=d.direction,
            entry=entry_px, exit=exit_px, pips=gross, cost_pips=cost,
            outcome=outcome, ambiguous=ambiguous, slippage_pips=slippage))

    return res


def summary(res: BacktestResult, label: str = "") -> str:
    """The report, with the verdict and the caveats attached to it."""
    if res.n == 0:
        return f"{label or 'backtest'}: no trades"
    e = res.expectancy()
    head = f"{label or 'backtest'}: {res.n} trades"
    lines = [head, e.summary(),
             f"  total          {res.total_pips:>8.1f} pips",
             f"  outcomes       target {sum(1 for t in res.trades if t.outcome == 'target')}"
             f" / stop {sum(1 for t in res.trades if t.outcome == 'stop')}"
             f" / unresolved {sum(1 for t in res.trades if t.outcome in ('timeout', 'end-of-data'))}"]
    for w in res.warnings:
        lines.append(f"  !! {w}")
    return "\n".join(lines)
