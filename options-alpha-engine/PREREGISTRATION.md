# Pre-registration — what we will run on real data, decided before we see it

**Status: OPEN. Nothing in this document has been run on real market data yet.**

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

### 2.3 Options edge (`models/edge.py`, `engine/signal_backtest.py`, `tools/paper_trade.py`)

| Parameter | Value | Why this one |
|---|---|---|
| `dte` | 30 | the standard vol tenor; matches VIX construction |
| `warmup` | 63 | one quarter |
| `min_vrp` | 0.0 | sell only when Q variance exceeds P |
| `hedge_bps` / `spread_frac` | 5e-4 / 0.015 | round-trip cost model |
| `cvar_limit` | 0.03 | 3% of equity at the CVaR shock |
| `max_drawdown` | 0.25 | kill-switch, now checked intra-trade |
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

1. **The crash tail is a hand-set prior.** `models/baseline.py` fixes
   `stress_weight=0.15`, `stress_sd_mult=2.30`, `base_drop_mult=1.10`,
   `leverage_exp=0.50`, `horizon_exp=0.25`, `realized_gain=1.5`, and
   `objective.left_tail_pinball` scores challenger models *against that guess*.
   Passing the tail gate means "beats a hand-tuned prior", not "models crashes".
2. **The correlation-aware vega cap has never been exercised.** Every harness holds
   one position at a time, so `RiskGovernor.can_add` always sees an empty book and
   the N-vs-√N aggregation that justifies the module has never run.
3. **`portfolio.Position.T` uses a 365 clock while the simulations run 252.**
   Measured: vega understated 16.8% at a 21-bar tenor. At current contract
   granularity the rounding absorbs it (3 contracts either way), so it is a
   correctness defect rather than a live risk, but it is unfixed as of this writing.
4. **No parameter is walk-forward selected** except the calibration vol scale.
   Everything in §2 is a judgement call, which is exactly why it is locked here.
5. **Single-name option chains are American** and must be de-Americanized
   (`--american`); index and crypto chains are not affected.

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

*(none — this document has not yet been run against real data)*
