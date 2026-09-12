# Pre-registration — what we will run on real data, decided before we see it

**Status: §2.1 (sector rotation) has been RUN on real data and ANSWERED NO — see
§7. Everything else is still open.**

## Why this file exists

Everything in this repo has been developed against fixtures the author wrote. Over
that development, roughly two dozen constants were chosen — window lengths,
thresholds, horizons, cost assumptions — and each was chosen *while looking at
results*. That is normal and unavoidable during construction. It becomes fatal the
moment real data arrives, because a parameter tuned on the data it is then tested
on produces a number that means nothing, and the tuning does not feel like
cheating while it is happening. It feels like fixing a bug.

So the parameters and the decision rules are written down **here, first**. When the
real data lands, the run uses these values. If a result disappoints and a different
window would have helped, that is a finding about the strategy, not a reason to
edit this file.

This is the difference between a test and a story fitted after the fact. It costs
one afternoon and it is the cheapest credibility available.

## The rules

1. **Locked before data.** Every value in §2 is fixed as of this commit. Changing
   one after seeing a real result requires a new dated section in §6 stating what
   changed, why, and what the result was *before* the change. The old result stays
   in the file.
2. **One primary question per study.** Secondary readings are exploratory and are
   labelled as such. A study with five outcomes and one significant result has
   found nothing.
3. **The verdict is whatever the harness prints.** Each harness already refuses to
   conclude below its own thresholds. Those refusals are the result, not a problem
   to work around by widening a window until something appears.
4. **Placebos and nulls run alongside, always.** Where a harness has one
   (`rotation_backtest`'s shuffled labels, `crowding_backtest`'s paired design),
   it runs on every real-data pass and is reported next to the headline.
5. **Intervals, not points.** Every reported Sharpe or return carries the
   `engine/robustness.py` interval. A point estimate alone is not a result.

## §2 — Locked parameters

### 2.1 Sector rotation (`models/rotation.py`, `engine/rotation_backtest.py`)

| Parameter | Value | Why this one |
|---|---|---|
| `window` | 63 | one quarter of trading days — the standard RRG lookback |
| `mom_lag` | 5 | one week; shorter is twitchy, longer is late |
| `tail` | 12 | ~2.5 weeks of visible path |
| `horizon` | 21 | one month forward; matches the option holding period elsewhere |
| `k` (long/short legs) | 3 | top and bottom quartile-ish of 11 sectors |
| `cost_bps` | 10 | generous for liquid sector ETFs; the point is that it is charged |
| `rank_by` | `both` | the RRG's own definition, i.e. the claim being tested |
| `t_thresh` | 2.0 | ~5% two-sided |
| `min_obs` | 30 per quadrant | below this the panel refuses |

**Primary question.** Over the full available history of the 11 SPDR sector ETFs
vs SPY: does the quadrant panel report `LEADING BEATS LAGGING` with |t| ≥ 2, AND
does the long/short book beat its label-shuffled placebo by more than 2×?

**Pre-committed prediction.** The synthetic work says the momentum axis carries no
edge of its own (Sharpe −1.17, 90% interval [−3.24, +1.22] over 15 worlds) and that
ranking on relative strength alone was at least as good. If that replicates on real
sector history, `rank_by='rs'` should match or beat `rank_by='both'`. **Recorded now
so it cannot be claimed afterwards either way.**

**Secondary, exploratory (labelled as such in any write-up):** per-quadrant means,
the `rs` vs `both` vs `momentum` comparison, quadrant dwell times, counter-clockwise
transition counts.

### 2.2 Fund crowding (`engine/crowding_backtest.py`)

| Parameter | Value | Why this one |
|---|---|---|
| `dte` | 21 | one monthly cycle, matching the funds' own roll |
| `warmup` | 63 | one quarter before the first decision |
| `crowded_min` | 0.30 | a strike holding ≥30% of mapped fund notional |
| `uncrowded_max` | 0.05 | effectively untouched by the mapped funds |
| `max_moneyness_gap` | 0.05 | the control leg must be comparable, not merely available |
| `hedge_bps` / `spread_frac` | 5e-4 / 0.015 | as used throughout the repo |
| `min_pairs` | 30 | below this it prints NOT ENOUGH DATA |
| `t_thresh` | 2.0 | ~5% two-sided |
| `lag_days` (archive) | 1 | holdings publish T+1; same-day is look-ahead |

**Primary question.** Selling a fund-crowded strike vs the nearest comparable
uncrowded one, paired on the same date: is the mean paired difference significant
at |t| ≥ 2 over ≥ 30 pairs, and in which direction?

**Pre-committed prediction.** None. This is the question the repo does not have an
opinion about — both stories (the vol there is already crushed, versus the flow is
mechanical and persistent) are plausible. Stating no prediction is itself the
honest position and is recorded so that whichever way it lands cannot be described
afterwards as "what we expected".

**Data requirement, and the reason it gates everything:** ≥ 30 usable pair dates at
a 21-day spacing means **≥ ~18 months of daily books**, or a shorter archive with
overlapping expiries. `tools/archive_holdings.py` must be running daily from now;
point-in-time holdings cannot be bought retroactively at any price.

**§2.2a — the necessary condition, checkable in an afternoon.** Eighteen months is
a long time to wait to find out the premise was never arithmetically possible. The
mechanism requires that a fund hold a meaningful share of the OPEN INTEREST at its
own strike; a fund holding 1% of the contracts outstanding there is one participant
among a hundred and cannot be the marginal seller setting the price.
`models/oi_share.py` measures that share and `tools/oi_share.py` runs it against one
archived book plus one live chain. **The bands were fixed before any data was seen:**

| max share at any strike | reading |
|---|---|
| < 5% | **REFUTED** — close §2.2 rather than wait on it |
| 5–20% | **INCONCLUSIVE** — starting the archive is defensible, and this is the reason |
| ≥ 20% | **PLAUSIBLE** — the mechanism is not ruled out; the study is worth running |

Passing is not evidence of an effect. It only says the arithmetic does not forbid
one. Open interest of 0 means UNKNOWN in this codebase, and a chain without it
returns NO OPEN INTEREST DATA rather than a refutation — closing a study on an
absence of evidence would be the same error in the opposite direction.

**Known before running it, and recorded here so the result is not a surprise:**
JEPI and JEPQ implement their overwriting through OTC equity-linked notes and hold
no listed options at all, so for those two there is nothing to hold a share OF. The
screen reports unmatched fund lines rather than discarding them, because a fund
absent from the listed market must not read as a small participant in it.

### 2.3 Options edge (`models/edge.py`, `engine/signal_backtest.py`, `tools/paper_trade.py`)

| Parameter | Value | Why this one |
|---|---|---|
| `dte` | 30 | the standard vol tenor; matches VIX construction |
| `warmup` | 63 | one quarter |
| `min_vrp` | 0.0 | sell only when Q variance exceeds P |
| `hedge_bps` / `spread_frac` | 5e-4 / 0.015 | round-trip cost model |
| `cvar_limit` | 0.03 | 3% of equity at the CVaR shock |
| `max_drawdown` | 0.25 | kill-switch, checked intra-trade — but it has NEVER FIRED at this level on any path measured, so it is an untested control rather than a demonstrated one (§4.7) |
| `max_net_short_vega` | 8,000 | governor cap |
| gates enabled | regime + macro + **events** | events was previously off in every run |

**Primary question.** On the forward paper-trade ledger only — not on any
retrospective fit — is the cost-inclusive, gate-respecting book's mean trade P&L
interval strictly above zero after ≥ 30 settled trades?

**Pre-committed prediction.** No. The repo's own benchmark on synthetic data shows
the signal-gated book losing to buy-and-hold, and the honest prior for a
single-name short-vol book net of costs is roughly zero. Anything above that needs
the interval, not the point.

### 2.4 Calibration (`models/calibration.py`)

`stride` = the horizon (non-overlapping); `MIN_EFFECTIVE_WINDOWS` = 25;
`tol_sd` = 0.02. The verdict is withheld below 25 independent windows and that
refusal is reported as-is.

## §3 — What counts as a negative result

A negative result is a result and gets written up the same way. Specifically:

- Rotation: panel |t| < 2, or the book failing to clear its placebo by 2×.
- Crowding: fewer than 30 pairs, or |t| < 2.
- Options: the mean trade P&L interval spanning zero after 30 settled trades.

None of these are grounds for re-running with different parameters. They are
grounds for either a longer sample under the *same* parameters, or for stopping.

## §4 — Known weaknesses these tests will NOT fix

Stated up front so they are not later presented as surprises.

1. **The crash tail is a hand-set prior — now measured, and it holds.**
   `models/baseline.py` fixes six shape constants and
   `objective.left_tail_pinball` scores challengers *against that guess*.
   `models/tail_fit.py` fits all six by coordinate descent on a walk-forward
   split and finds that fitting improves TRAIN loss every time and TEST loss
   never: **−0.8% out of sample at 5,000 bars, −1.6% at 9,000**. The shape is not
   identifiable from a few hundred non-overlapping monthly windows even with
   ~36 years of daily data. So the defaults act as a regulariser rather than as
   an unexamined guess — but "beats a prior that cannot be beaten by fitting" is
   still a weaker claim than "models crashes", and the gate should be read that
   way.
2. ~~The correlation-aware vega cap has never been exercised.~~ **CLOSED.**
   `engine/book_backtest.py` now runs staggered positions across several
   underlyings so the governor sees a live book. Sizing against an empty book —
   how every single-position harness does it — peaks at **17,562 of net short
   vega against a stated cap of 8,000**, a 2.2× breach nothing noticed, because
   no single position is near the cap alone.
3. ~~`portfolio.Position.T` uses a 365 clock while the simulations run 252.~~
   **FIXED.** `Position` takes `days_per_year` and the bar-counting callers pass
   252. The defect had understated vega by 16.8% at a 21-bar tenor.
4. **No parameter is walk-forward selected** except the calibration vol scale.
   Everything in §2 is a judgement call, which is exactly why it is locked here.
5. **Single-name option chains are American** and must be de-Americanized
   (`--american`); index and crypto chains are not affected. ~~`paper_trade
   record` had no such flag, so the ledger this rule exists to protect was the
   one place it could not be applied.~~ **FIXED** — `record --american`, and the
   entry records `de_americanized` so a ledger mixing both is detectable rather
   than merely wrong.
6. **The forward ledger could not record at all, and that was invisible.**
   Tradier fetched the 3 soonest expiries; SPY/QQQ list them near-daily, so the
   reachable tenors were 1-4 DTE, where ±10% wings are worth ~0, are dropped by
   the zero-bid filter, and fail `rnd._coverage_ok`. The tenors that pass
   coverage (30-45d) were never fetched, and `rnd._slice` matched `expiry_days`
   EXACTLY, so `--dte 30` also found nothing even when it was listed. **FIXED**
   — expiries are now chosen by distance from the requested tenor rather than by
   soonest, and the tenor snaps to the nearest listed expiry with the move
   recorded in the entry (`requested_dte`, `expiry_snapped`). Found while
   verifying an operator runbook, not by a test; §5.2 had been "start the
   ledger" for weeks and starting it would have banked nothing.
7. **The intra-trade kill-switch has never fired, so it is an untested control.**
   At the locked `max_drawdown=0.25` the CVaR-sized book tops out near 8%
   drawdown, so the switch never reaches its own trigger: `kill_midtrade` True
   and False produce **byte-identical** totals on every path measured, calm and
   crash alike. It is inert, not broken — lowering the threshold until it is
   reachable does fire it, and that experiment is the finding: at 0.06 it turns
   −0.56% into **−6.90%** while cutting max drawdown only 8.4% → 7.6%. **6.3
   points of return for 0.8 points of drawdown.** The default is deliberately
   NOT being lowered: 0.25 is locked in §2.3, and lowering a locked parameter to
   make a feature fire — toward a worse outcome — is the exact move this file
   exists to prevent. The honest statement is that the book is protected by the
   CVaR limit and the vega cap, and that the kill-switch behind them is
   unexercised. Measured in `tests/test_hedged_backtest.py`, tabulated in
   PIPELINE.md §2.7.

## §5 — Order of execution

1. `tools/archive_holdings.py fetch` — daily, starting immediately. Nothing else
   is time-critical; this one loses data permanently if deferred.
2. `tools/paper_trade.py record` — daily. The first graded score arrives `dte`
   records later, so the clock starts when it starts.
3. Sector rotation on real ETF history — the cheapest real answer available, and
   the only one whose data is free and already complete.
4. Crowding — blocked on (1) reaching ~18 months.
5. Options edge — blocked on (2) reaching 30 settled trades.

## §6 — Amendments

### 6.1 The placebo becomes a distribution, not one shuffle (2026-08-05)

**What changed.** §2.1's rule "beat the label-shuffled placebo by more than 2×"
compares the book to **one** shuffle. That control was measured and it does not
work. On a fixture world built with **no signal at all**, the single-draw
placebo's own total swung from **−16.6% to +7.9%** purely on its seed, and the
harness declared an edge in **6 of 20 seeds** — a 30% false-positive rate on a
null world.

**The rule now also requires a permutation p ≤ 0.10** over 200 shuffles: the
fraction of shuffled books that matched or beat the real one. Measured after the
change, null worlds fire 2/10 (the nominal rate) and signal worlds 10/10.

**Why this is a tightening and not a rescue.** The registered 2× rule is kept
in full and the new requirement is added on top with `and`, so every book that
failed before still fails. `test_the_amendment_can_only_make_passing_harder`
asserts exactly that across a grid of returns and p-values — if any combination
could pass under the amendment but not under the original, the test fails.

**Effect on §7.1: none.** That book returned −27.0%, and a book that lost money
fails on the first clause regardless of any placebo.

**A second defect found in the same pass.** Charging each shuffled book its own
turnover made the real book systematically cheaper than its control — ranked
picks are stickier (66% churn) than random ones (77%) — which is a cost
advantage, not a predictive one. It dragged the null world's median p to 0.079.
The control is now **matched on turnover**, paying the real book's fee, so the
only thing the shuffle changes is the labels. The p-value is now invariant to
`cost_bps`, which is the check that it differences out.

*(Amendment discipline: 6.1 changes a control that was demonstrably broken. It
does not change any parameter in §2, and no §7 verdict moves because of it.)*

## §7 — Results

### 7.1 Sector rotation — **ANSWERED NO** (first real-data run, 2026-08-05)

Run by the operator on real SPDR sector history via `tools/quickstart.py`, at the
parameters locked in §2.1. Nothing was tuned; this was the first pass.

| | |
|---|---|
| Rebalance dates | 91 (11 sectors × 91 = 1,001 observations) |
| Horizon | 21 bars, non-overlapping |
| Panel (before costs) | Leading − Lagging **+0.25%/period, t = +0.71** |
| Leading | n=228, mean **+0.04%**, t = +0.14 |
| Weakening | n=218, mean −0.44%, t = −1.43 |
| Lagging | n=274, mean −0.21%, t = −0.88 |
| Improving | n=281, mean −0.25%, t = −0.90 |
| Long/short book | **−27.0%** net of costs, Sharpe **−1.9** |
| Label-shuffled placebo | −18.4% (**one** shuffle — see the correction below) |
| Permutation p | not run; the harness had no null distribution until §6.1 |

**Both halves of the primary question fail.** The panel needed |t| ≥ 2 and got
0.71. The book returned −27.0% — a book that lost money fails whatever the
placebo did. Per §3 this is a negative result and it is recorded as one; per
rule 1 the parameters are not being changed to look for a better one.

> **Correction (2026-08-05, same day).** This section first read "the book lost
> more than its placebo" as though that comparison carried weight. It does not.
> That placebo was a **single shuffle**, and §6.1 then measured the single-draw
> control firing on 6 of 20 null worlds. **−27.0% vs −18.4% is one draw against
> one draw and supports no claim about chance.** The failure stands entirely on
> the book losing money and the panel's t=+0.71. The `−18.4%` figure below is
> retained only because it is almost exactly the cost toll, which is a fact
> about the fee schedule rather than about the placebo.

**Two readings that are NOT rationalisations, because both are checkable:**

1. **Most of the book's loss is the toll, not the call.** The harness charges
   `2 × cost_bps` per rebalance (`cost = 4k·bps/10⁴/2k` = 20bp) and books a fixed
   notional each period, so returns add rather than compound: **91 × 20bp =
   −18.20% from costs alone**, verified by running the harness on a
   cost-only return stream. The placebo's **−18.4%** is that number and
   essentially nothing else — gross ≈ **−0.2%** — which is exactly what a
   label-shuffled control should do: earn nothing, pay the toll. Backing the same
   toll out of the real book leaves gross ≈ **−8.8%**. So the rotation call was
   mildly negative on this sample; the rest of the −27.0% is the rebalancing
   bill, and `cost_bps` = 10 on liquid sector ETFs is a deliberately generous
   assumption charged 12 times a period.

   *Since amended:* the fee was billed at 100% turnover every rebalance whether
   or not the leaderboard moved. Measured churn is ~63%, so the toll on a re-run
   will be nearer **−11%** and the book nearer **−20%**. Still a failure, which
   is why fixing it is a correction rather than a rescue — a misspecification
   that flatters the result would not have been touched before publishing this.
2. **Every quadrant mean is negative, and that is arithmetic, not a bug.** The
   panel scores excess return vs **cap-weighted** SPY while the quadrants hold
   equally-weighted sectors. In a concentration regime — and XLK's +18.0%
   relative against every other sector negative says this sample is one — the
   average sector loses to SPY by construction. The Leading − Lagging *spread*
   differences this out, which is why the spread is the registered statistic and
   the levels are not.

**The pre-committed prediction is untested.** §2.1 predicted `rank_by='rs'` would
match or beat `rank_by='both'`. The dashboard runs `both` only, so this has not
been checked and is not being claimed either way.

**What is NOT concluded.** That sector rotation cannot work; that the RRG is
worthless; that a cheaper rebalance would rescue it. One sample, one parameter
set, one cost assumption. What *is* concluded is the registered question:
on this history, at these parameters, the chart did not predict and the book did
not pay.

**Amendment discipline.** The obvious next moves — a longer sample, a lower
`cost_bps`, `rank_by='rs'`, a smaller `k` — are each a *new* question. Any of them
run against this data is exploratory and gets labelled as such; none of them
retroactively changes the answer above.

**Acted on.** `models/decision.py` used to cut position size when the underlying's
sector sat in Lagging (×0.5) or Weakening (×0.75). That cut rested on a story
about the chart, not a measurement, and the measurement has now been taken. The
quadrant no longer moves size in either direction; it is printed as context and
carries the status `tested: no edge`, which ranks below `no data` — an untested
input might yet be true, a disproven one has had its turn. Cutting size on a
signal shown to be empty is not caution: it spends real position size on noise
and prints a sentence that looks like reasoning. **A study whose result does not
change the system is a study that was not run.**
