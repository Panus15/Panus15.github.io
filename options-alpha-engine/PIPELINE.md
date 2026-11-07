# The pipeline — what runs, in what order, and what it has actually shown

One page covering the whole system: the flow, every measured number, and how to
operate it. `ARCHITECTURE.md` explains *why* each module is built the way it is;
this explains *how the parts run together* and *what they have proven*.

**Scale:** 47 test files · **562 tests, all green** · pure stdlib, no
numpy/scipy/pandas · every load-bearing change mutation-verified. **10 more** require
numpy (`test_gru.py`, `test_neural.py`): those two files print SKIP rather than a row
of PASS lines for work that did not happen, so a stdlib machine runs 562 and CI —
which installs numpy on purpose — runs **572**. `tests/test_wiring.py` fails if this
sentence stops matching the tree, and `tools/check_offline.py` proves every file
passes with the network denied.

**Read this first.** One study has been run on real market data and it ANSWERED NO
(sector rotation, PREREGISTRATION.md §7.1). The forward options ledger has settled
**0 trades** and the crowding archive holds **0 days**. Everything else in this
document is either an ORACLE result — maths checked against independent maths,
which proves the code is right and says nothing about profit — or a measurement on
fixtures the author generated. Nothing here is evidence that this engine makes
money, and the two clocks that could produce such evidence have not started.

---

## 1. The flow

```
  DAILY, FOREVER                      ON DEMAND                       THE VERDICT
  ─────────────                       ─────────                       ───────────

  archive_holdings ─┐            ┌─ adapters / deribit / tradier
   fund option books│            │  → OptionChain
   (expires if not  │            ▼
    captured today) │        american.py  de-Americanize (US single names)
                    │            │
                    │            ├──────────────┬──────────────┐
                    │            ▼              ▼              ▼
                    │        rnd + surface   volforecast    rotation
                    │        Q: model-free   P: HAR-RV on   RS-Ratio /
                    │        vol, BKM skew   RANGE bars +   RS-Momentum
                    │                        term structure  (TESTED: NO EDGE)
                    │            │              │              │
                    │            └──────┬───────┘              │
                    │                   ▼                      │
                    │              edge.py  P vs Q             │
                    │              VRP z-score, per strike     │
                    │                   │                      │
                    │        GATES ─────┤                      │
                    │        regime · news · macro · events    │
                    │                   ▼                      │
                    └──── fund_flow ─→ trade_card              │
                          crowding      one decision screen    │
                             │               │                 │
                             └───────────────┼─────────────────┘
                                             ▼
                                    decision.py  FUSE
                                    unproven inputs may only CUT size
                                    DISPROVEN inputs do not move it at all
                                        │
                                        ▼
                              portfolio.py  govern
                              CVaR size · correlation-aware
                              vega cap · kill-switch (NEVER FIRED)
                                        │
                                        ▼
                              sizing.py  cap on the LOSS
                              (naked shorts size to 0 and REFUSE)
                                        │
                                        ▼
                              ticket.py  a placeable order,
                              or a named refusal
                              (takes the DECISION, so the
                               context's size cut reaches it)
                                        │
        ┌───────────────────────────────┼───────────────────────────────┐
        ▼                ▼              ▼              ▼                ▼
   hedged/signal    book_backtest   spread_backtest  crowding_bt   rotation_bt
   one position     MANY at once    wings vs naked   fund crowding  does RRG work
                    (governor sees                                  tail_fit: is the
                     a real book)                                   tail prior any good?
        └────────────────┴──────┬───────┴──────────────┴────────────────┘
                                ▼
                    robustness.py  seed ensemble + block bootstrap
                    every number gets its interval
                                ▼
                    paper_trade.py  the forward ledger
                    record → settle → proper-score
```

Data enters on the left, an interval-bearing verdict leaves at the bottom. The two
boxes at the top-left run **every day** and are the only parts where delay costs
something that cannot be bought back.

---

## 2. What it has measured

Everything below is reproducible with `python3 demo.py`. **All of it is on
synthetic fixtures** — see §4 for what that does and does not license.

### 2.1 The honest bar: does the signal beat doing nothing?

| book | Sharpe | total | max DD |
|---|---|---|---|
| signal-gated | −0.04 | −0.4% | −4.5% |
| always-sell (the QQQI/JEPQ mechanic) | −0.01 | −0.1% | −4.5% |
| **buy-and-hold** | **+0.50** | **+30.4%** | −27.1% |

The signal **loses to holding the index** on the shipped crash path. This is the
system's own benchmark reporting against itself, and it is printed by `demo.py`
every run rather than buried.

### 2.2 How wide is any of that?

Same strategy, same parameters, only the RNG seed changed:

```
seed ensemble Sharpe   +1.48   90% interval [+0.07, +4.22]     (20 paths)
bootstrap mean trade   -17.1   90% interval [-460, +305]       <- SPANS ZERO
```

33 trades cannot distinguish this book's profit from luck, and the report says so
in those words. A point estimate without its interval is not a result here.

### 2.3 Risk control: the finding that justified building `book_backtest`

Five underlyings, staggered entries, a stated cap of 8,000 net short vega:

| sizing | peak net short vega | vs cap | max DD | total |
|---|---|---|---|---|
| **governed** (sees the live book) | 7,998 | at the cap | −16.2% | +4.5% |
| **ungoverned** (empty book — how every other harness sizes) | **17,562** | **2.2× breach** | −25.1% | +19.4% |

A per-trade view understates the book's real short vega by **54–77%**, because no
single position is near the cap alone. The cap costs 15 points of return and buys
9 points of drawdown — that is the trade it exists to make.

### 2.4 Defined-risk spreads: buy the wings for what you *cannot* hedge

| path | spread total | naked total | spread worst | naked worst |
|---|---|---|---|---|
| clean | −0.9% | **+4.5%** | −559 | **−87** |
| ±18% gaps | **−2.7%** | −3.9% | **−694** | −2,901 |

A working delta hedge already removes the directional damage the wings are sold to
cap, so on a smooth path they are a pure cost. They earn their price exactly where
the hedge fails.

### 2.5 The two experiment harnesses discriminate

**Fund crowding** — paired, same-date, moneyness-matched (3,000 bars):

| injected | pairs | mean paired diff | t | verdict |
|---|---|---|---|---|
| nothing | 138 | −10.98 | −0.98 | NO SIGNIFICANT EFFECT |
| 2 vol pt dent | 133 | −36.13 | −3.07 | CROWDED IS WORSE |
| 5 vol pt dent | 97 | −84.17 | −5.21 | CROWDED IS WORSE |

**Sector rotation** — quadrant panel + long/short with a label-shuffled placebo:

| world | panel verdict | book | placebo |
|---|---|---|---|
| alpha is fresh noise daily | NO SIGNIFICANT EFFECT | −21.7% | −14.6% |
| alpha persists | LEADING BEATS LAGGING, t=+5.21 | **+73.4%** | −13.0% |

> **The placebo column above is superseded.** A SINGLE label shuffle was measured
> swinging from −16.6% to +7.9% on its seed alone, and declaring an edge in **6 of
> 20 null worlds** — a 30% false-positive rate. The control is now a permutation
> test over 200 shuffles, matched on turnover, and the null fires **2/10** (its
> nominal rate) while signal worlds fire **10/10**. The p-value is invariant to
> `cost_bps`, which is the check that the fee differences out. See
> PREREGISTRATION.md §6.1.

Both report **nothing** when there is nothing. That property is what makes the
positive readings worth looking at at all.

### 2.6 Calibration: the correction that changed a conclusion

The PIT test was counting 308 overlapping windows as 308 independent draws:

```
KS distance D = 0.097   (unchanged)
  as 308 "independent" windows  →  p = 0.0054   "miscalibrated"
  on the 10 genuinely independent →  p = 0.9999   "no evidence"
```

Same data, same statistic, opposite conclusion. The verdict is now withheld
entirely below 25 independent windows.

---

### 2.7 The risk layer, after it was audited

Two defects here were arithmetic, not modelling, and both were ruin-shaped.

**The 2% cap was computed on the premium, not the loss.** Correct for buying an
option; catastrophic for selling one, which is all this engine does. On $100,000:

| sold | contracts the old cap allowed | max loss |
|---|---|---|
| SPY 30d put ~10% OTM ($1.10) | 18 | **1,238%** of equity |
| SPY 30d put ~15% OTM ($0.40) | 50 | **3,250%** |
| SPY 7d put far OTM ($0.05) | 400 | **28,000%** |
| SPY 1d put far OTM ($0.01) | 2,000 | **144,000%** |

The cheaper the option, the more it sold — a cap that loosened exactly as the tail
got fatter. `engine/sizing.py` now caps on the LOSS, and the honest consequence is
pinned by a test: **a $100,000 account cannot sell even one naked index put** under
a real 2% cap. One SPY 688 put risks $68,800. Defined-risk structures are the only
thing it can size, which is why `models/ticket.py` builds only those.

**`cvar_per_contract`'s `alpha` did nothing.** With a two-point scenario set, one
loss state carries `tail_prob` of the mass, so 0.90 / 0.95 / 0.99 / 0.999 all
returned 2,443.31. "CVaR at 95% confidence" was the loss in one hand-written
scenario. It now refuses a level its scenarios cannot resolve; on a four-point set
alpha rises properly (1,944 / 3,363 / 8,286).

`engine/sizing.py` had **no test file at all** — 391 tests passed around it.

**The intra-trade kill-switch has never protected anything.** At the shipped
`max_drawdown=0.25` the CVaR-sized book tops out near 8% drawdown, so the switch is
INERT: `kill_midtrade` True and False give byte-identical results on every path
tried, calm and crash alike. The mechanism is not broken — lowering the threshold
until it is reachable shows it firing — but what it buys is the finding:

| `max_drawdown` | kills | total | max DD |
|---|---|---|---|
| 0.25 (shipped) | 0 | −0.56% | −8.4% |
| 0.06 | 1 | **−6.90%** | −7.6% |

**6.3 points of return for 0.8 points of drawdown.** On this fixture it is a bad
trade when it acts, which is why the default is not being lowered to make it fire —
that would be tuning a locked parameter toward a worse outcome to justify a
feature. It is listed here as untested protection, not as protection.


**The size cut never reached the order.** `decision.py` exists to let unproven
inputs cut size and never raise it. `build_ticket` read only the variance card, so
a decision of "SELL VOL, **size x0.60**" built the *same* order as x1.00 — the cut
was discarded at the last step, where it becomes money. Measured on one put credit
spread with fund crowding firing:

| equity | shipped | authorised | risk shipped | risk authorised | excess |
|---|---|---|---|---|---|
| $100,000 | 2 | 1 | $1,700 | $850 | **+100%** |
| $250,000 | 5 | 3 | $4,250 | $2,550 | **+67%** |
| $1,000,000 | 23 | 13 | $19,550 | $11,050 | **+77%** |

The ticket now takes `decision=`, floors (never rounds) the product, and refuses by
its own code when the cut lands below one contract — separate from `SIZE_ZERO`,
because "the context said no" and "your account is too small" have different
remedies. Two traps found while fixing it, both by tests rather than by reading:
the new multiplier **shadowed the contract multiplier**, which would have printed a
$1.50 limit instead of $150; and a NaN multiplier **survives `min(max(x,0),1)`
untouched** (every comparison against NaN is False) while `inf` clamps *up* to 1.0,
so the obvious guard turned a corrupt risk input into a full-size order. Unusable
now means zero, not maximum. **13/13 mutants killed.**

And an absent decision is no longer silent: the ticket prints `fused context NOT
APPLIED` on its face, because an optional safety check that can be skipped quietly
is the same defect wearing a keyword argument. `tests/test_wiring.py` pins the call
shape at every entry point — reachability was never the problem here, the ticket was
reachable and *uninformed*.

### 2.7b The page a human opens was leading with the disproven half

The one study this repo has run on real market data **answered NO**: the sector
quadrant predicted nothing (§7.1 of PREREGISTRATION.md, t=+0.71 on a test that
charges no costs). That verdict is rendered honestly — and it was still the *first*
thing on the page, while the half backed by oracle-tested machinery, the variance
edge and the order it implies, printed only into a terminal. Ordering is not
decoration: a reader acts on what is at the top.

The dashboard now leads with **the options decision** — vol side, P vs Q, VRP, the
fused size multiplier, and the order ticket itself with every leg, the limit, the
dollar worst case and the evidence caveat — or, when no chain was supplied, with a
panel that **says so and gives the command**. Rotation moved below it and keeps its
own verdict. `tests/test_dashboard.py` pins the ordering, because the next person
to add a section will not know it was a decision.

The dashboard had **no tests at all** — 476 of them passed around the one file a
human actually looks at — and it is the one module where a defect is invisible to
Python. `TEMPLATE` is a non-raw triple-quoted string, so this line:

```
c=>({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;"}[c])
```

shipped a bare quote to the browser, a syntax error that blanked **the entire
page**, and every Python test still passed. Found by executing the page's own
script in node, not by reading it. Two guards now: a portable test that no
backslash survives in the template region (the root cause — Python eats it), and a
`node --check` on the script the page actually ships, which runs where node exists
and says plainly when it does not.

Also found while testing it: the panel rounded `maxLossPerContract` and
`maxLossTotal` **independently**, so the two numbers on screen did not multiply
out. The payload now carries money unrounded and the page formats at display time.
**12/13 mutants killed here; the 13th — dropping `decision=` — is killed by
`tests/test_wiring.py`, which names the file and line.**

### 2.7c The engine was recommending a rule it had never measured

Every order ticket printed **"close at 50% of max profit, or at 7 DTE, whichever
comes first"**. Nothing in the repo implemented it — `engine/spread_backtest.py`
held every spread to expiry. So the Sharpe, the worst trade and the return being
reported described a *different strategy* from the one on the screen, and the gap
was invisible because each half was internally consistent.

The rule now exists (`take_profit_frac` / `min_dte_remaining`, default **off**, so
every previously published number is reproduced exactly) and has been measured
paired per trade with the round trip to close charged — a management rule measured
without its closing cost reads as free money:

| cost to close | mean diff / trade | 90% interval | sd held → managed |
|---|---|---|---|
| 1.5% | +6.45 | [−5.20, +18.99] | 213.7 → 133.5 |
| 5.0% | +2.87 | [−8.88, +15.57] | 213.7 → 138.0 |
| 12.0% | **−4.29** | [−16.35, +8.36] | 213.7 → 146.9 |

**The P&L advantage is noise at every closing cost** — the interval spans zero
throughout and the mean changes sign around 8%. Through the full harness over 10
paths:

| metric | held | managed | managed better on |
|---|---|---|---|
| total return | −1.92% | −1.58% | 6/10 |
| **worst trade** | −634 | **−505** | **10/10** |
| max drawdown | −2.87% | −2.05% | 7/10 |
| **Sharpe** | −0.61 | **−0.87** | **1/10** |

So the rule **reliably cuts the tail** (unanimous on the worst trade, 20% better)
and usually cuts drawdown, **does not change returns**, and **makes Sharpe worse
almost always**. That last row is the one worth understanding rather than
explaining away: Sharpe is mean ÷ sd, the fixture's mean is negative, so cutting
dispersion makes the ratio *more* negative. **Variance reduction flatters nothing
on a strategy that loses money**, and anyone citing Sharpe to justify managing a
trade here would be reading the arithmetic backwards.

One constant now carries the rule (`EXIT_RULE_TEXT`), and `tests/test_wiring.py`
fails if any entry point spells it out again — that is how the recommendation and
the measurement drifted apart in the first place. **13/13 mutants killed**, after a
first sweep where **7 of 13 survived**: the tests had exercised the rule through
one real trade, which happened to exit on time and never once touched the profit
target. Deterministic constructed series replaced it.

### 2.7d The forward test existed and was invisible

Every other panel on the page is one snapshot — today's decision, today's quadrant
— and no snapshot can answer the only question that matters, which is whether the
edge **persists**. `tools/paper_trade.py` has been recording a dated `(P, Q)` pair
on every run since it was written. The trend existed. Nothing rendered it.

That matters more than it sounds, because the largest risk to this project is not a
modelling error — it is that the daily job stops being run, and **an invisible clock
is one nobody winds**. The dashboard now carries a forward-test panel: recorded /
settled / open, P against Q over time with the gap between the lines being the
variance premium itself, the calibration and profit scoreboards, and the ledger's
own verdict. Verified on a 112-record offline ledger (109 settled, 73 graded
trades), and the verdict it printed was `P does NOT beat the market's Q density OOS`
**while the trades showed +$2,934 at a 1.84 Sharpe** — the two questions kept
separate, which is the whole point of that split.

An empty ledger renders as **0 recorded, 0 settled** with the command that starts
the clock, never as a blank panel, and while nothing has settled it prints how many
more daily records the first score needs. That lag is the honest cost of an
out-of-sample test, not a delay to engineer away.

Four display defects found by testing it, each one a number that would have misled:

- **"Traded 74" beside "Trades 73"** read as an inconsistency. They are different
  events — the signal fired on 74 dates, 73 of those have been graded — now labelled
  *Signal fired* and *Graded trades*.
- A chain carrying no `asof` gave every entry an empty date, so the header rendered a
  bare dash as its range. It now says the dates were not recorded.
- A ledger entry **missing** `coverage_ok` was read as coverage *confirmed*. An
  unmarked point on the chart reads as "checked and good", so a date that never
  recorded the flag is marked too, with its own reason.
- The countdown to a first score was not gated on `settled == 0`, so a ledger with 3
  graded results at a 21-day horizon would still have demanded **18 more records**
  from someone already holding the answer.

**16/16 mutants killed**, after a first sweep where 5 of 16 survived — including
`abs()` on the total P&L, which the positive fixture could not catch and which is
the cheapest possible way to turn a failing forward test into a passing-looking one.

### 2.7e The job that advances the clocks never drew the page that shows them

`tools/daily.py` captures fund books, records the forward ledger and refreshes
prices. Its own docstring argues that *progress you can see is progress that keeps
happening* — and it printed its progress as text, from a command nobody had to run
twice, while the dashboard that now renders both clocks had to be invoked
separately. `--dashboard` closes that: the page is rendered last, from whatever the
steps actually produced. Verified end to end with the ledger record step **failing**
(no vendor token in this sandbox) — the page still rendered and reported the
ledger's real `112 recorded, 109 settled`.

Two defects found while wiring it, both in code that already existed:

**The flag-coverage test could not catch the thing it promised.** Its docstring said
"a seventh flag added tomorrow and forgotten fails here" while the body asserted on
a hardcoded list of eight — so adding four dashboard flags, exactly its stated case,
would have passed. The expectation is now derived from `_cfg` and `_EMIT`, plus a
second test that spies on the parser for the other half of the same leak: a flag
accepted on the command line that never reaches the job config and is therefore
silently ignored.

**`_step` leaked the one exception it exists to contain.** Its handler computed
`int(e.code)`, and `sys.exit("some message")` sets `code` to a *string* — so a step
exiting with a message raised `ValueError` out of the very function whose docstring
reads "a failing step must not stop the rest", taking every later step with it.
Found because a test fixture built a CSV without its benchmark column and the real
error never surfaced. A non-integer code is now a message and a failure.

**12/12 mutants killed.** One of them — "always uses the generated fixture" —
survived the first sweep: no test had supplied a real price basket, so a job that
quietly drew invented prices over real ones would have passed.

### 2.7f Two numbers on the ticket that described different things

**The probability and the exit rule did not refer to the same event.** The ticket
printed `P(profit) 85.0%` directly above `close at 50% of max profit, or at 7 DTE`.
The 85% is the chance the position finishes profitable **at expiry** — not the
chance of the exit being recommended, which this repo does not compute anywhere,
because it is a first-passage problem and the engine produces a terminal density.
Two numbers side by side invite multiplication. The label now says `AT EXPIRY`, and
under the rule sits the frequency that *was* measured:

> of 600 trades measured under this rule, **60%** ended at the profit target and
> **40%** at the time stop (10 generated paths, not market data)

A published constant like that rots silently into a claim about behaviour the code
no longer has, so a test re-measures it against the harness on three of the same
paths and fails if the split has moved.

### 2.7g The advertised scale was maintained by hand

`PIPELINE.md` opens with a test count. It was hand-edited **six times in one
session**, which is the reliable signal that it should not be. A drifting count is
worse than none: a reader checking "all green" against a smaller tree cannot tell
whether tests were deleted or the sentence was never updated, and the document loses
its claim to being measured rather than asserted. `tests/test_wiring.py` now derives
the count by parsing the tree and fails on any documented figure that disagrees.

Three things surfaced the moment it ran:

- **10 test functions across `test_gru.py` and `test_neural.py` run zero times**
  here. Not a defect — both files skip cleanly without numpy and deliberately print
  SKIP rather than a row of PASS lines for work that did not happen — but nothing
  recorded it, so "all green" quietly meant 10 fewer results than the tree held.
  Those files now declare `OPTIONAL_DEPENDENCY` and the header states the 10.
- The first regex flagged **two legitimate numbers**: a historical note (`391 tests
  passed around it`, recording how many existed when `sizing.py` had none) and a
  per-file count (`7 correctness tests`). Only the canonical whole-suite phrasings
  are checked now.
- The marker was detected as a **substring**, and `test_wiring.py` mentions the name
  in its own source — so it excluded its own seven tests from the count it was
  computing. It is read as a module-level assignment via the AST.

### 2.7h Three tests that passed because the network was broken

CI caught what this sandbox could not. `tools/daily.py`'s dashboard step was a
closure inside `run`, so every test of it had to call `run` — which also **fetches
prices and records a live ledger entry**. Outbound access is blocked here, so those
two steps failed, and three tests passed *because of it*:

- one asserted `not prices["ok"]` — it was testing that the vendor was unreachable
- one wrote a fixture CSV and read the page back; in CI the fetch **overwrote that
  CSV** with freshly downloaded prices before the page was drawn
- one compared the ledger's counts before and after; in CI the recorder **appended
  live vendor entries** in between, so the two never matched

Worse than three flaky tests: the suite was making real vendor API calls, and only
the absence of a network was hiding it.

The dashboard step is now a module-level `render_dashboard(cfg)` — testable alone,
which is why the closure was the actual defect — and the tests call it directly. A
`_no_network()` context manager replaces `socket.socket` and `urllib.request.urlopen`
with a raising stub, so a test that reaches a vendor **fails by name** instead of
depending on whether the machine it runs on happens to have access. Verified both
ways: the guard blocks and restores, and running the old style of call under it turns
the hidden access into `AssertionError: this test reached the network`.

Then the obvious question: **is anything else doing this?** `tools/check_offline.py`
answers it by re-running every file with external connections denied. It is a CI step
now, because the runner is where a network actually exists and therefore where the
check means anything — and on its first CI run it immediately found one more:

> `tests/test_daily.py   exit 0, 24 external attempt(s)`
> `EXTERNAL NETWORK: create_connection to query1.finance.yahoo.com`

A test named *"the day's capture must survive one vendor being down"* was **really
downloading all twelve symbols from Yahoo and Stooq**, and passing either way, because
it only asserts that both steps were attempted. It had been doing that since it was
written, in the suite and on every CI run. The vendor is now taken down *in* the test
rather than hoped to be down, which makes it mean what its name says.

That run also exposed a blind spot in the checker itself, and in the `_no_network()`
guard: both refused `socket.socket` and `urlopen` but not **`socket.create_connection`**,
which is what `http.client` actually calls — so the fetch walked straight past a
context manager whose entire job was to stop it. And refusing by socket *address* is
useless on a machine routing through a local HTTP proxy, because the connect target is
then loopback and the real host never appears — which is exactly why the dev sandbox
called this file clean while CI did not. The shim refuses at the **URL** layer too,
where the intent is visible, and with that it reproduces CI's finding locally. `file://`
stays allowed: `test_archive_holdings.py` reads fixtures that way and touches no network.

Loopback stays open deliberately. A blunt socket block also fails
`test_fetch_prices.py`, `test_archive_holdings.py` and `test_quickstart.py`, which
stand up local HTTP servers — the correct way to test a fetcher offline — so blocking
them would punish the right pattern.

Two more things the checker got wrong before it was right, both found by testing the
checker itself:

- Its own test **probes an external host on purpose** to prove the shim refuses, and
  `main()` relays that child's output, which the outer run then counted as the file's
  own offence. One named exemption, carrying a reason, waives the *counter* — never
  the exit status.
- It claimed to catch a test that reaches a vendor, **swallows the error and passes
  anyway**. It could not: the exception carries the message, so catching it erased the
  only evidence, and the probe came back with 0 attempts and a clean exit. The shim now
  writes each refusal to stderr **before** raising. That is the subtler offence of the
  two, and it is the exact shape of what CI then found — `exit 0, 24 external attempts`.

**All 47 files pass.**

### 2.7i The one study that could finish in an afternoon could not run at all

`tools/oi_share.py` exists to answer the crowding premise in a day instead of
eighteen months: does a fund hold a meaningful share of the open interest at its own
strikes? Run it and it reported **NO OPEN INTEREST DATA**, every time, whatever it
was given. Three defects were stacked and each one presented as the same message —
blaming the chain:

1. **The dump threw the field away.** `TradierAdapter` parses `open_interest` into
   every quote. `save_chain_json` — the `--dump` command the docs recommend — did
   not write it, and `JsonFileAdapter` did not read it. The documented capture route
   lost the one field the screen needs. `tools/oi_share.py` even printed *"a
   replayed chain only does if it was dumped with it"*, naming a precondition the
   repo's own dump command could not satisfy. Deribit's adapter dropped it too.
2. **A contract in a column called `symbol` lost its root.** `pick` reads "symbol"
   as a ticker column, and issuer files routinely put the CONTRACT there.
   `"QQQ   261016C00570000"` is truthy, so `underlying or root` kept the whole
   symbol — and the screen then filtered every line out against `"QQQ"` *before* the
   unmatched counter, giving **zero rows and zero unmatched**. The comment directly
   above that line explains the OSI fallback exists because "guessing the name is
   how a real file parses to zero option lines"; the code then did exactly that to
   the underlying.
3. **The CLI passed no as-of date**, so days-to-expiry was None for every line. The
   archiver names its files `FUND/YYYY-MM-DD.csv`, so the date is its own convention
   rather than a guess — and it is reported when used, never silently replaced with
   today, because a book read a week late shifts every line by seven days and
   matches nothing.

It now reaches a verdict. On a fixture, `MECHANISM PLAUSIBLE — max share 28.8% at
one strike, 27.2% pooled` — which is what the output looks like, not a result about
any real fund.

**The path the operator runs was untested.** Every existing test built a `FundBook`
in memory while the CLI reads a CSV through `from_file` and a chain through
`JsonFileAdapter`; all three defects lived in that gap. And the model itself printed
one sentence for two different failures — "nothing matched" now says so, because
blaming a chain's open interest for an expiry mismatch sends the operator to re-dump
a chain that was fine. **10/10 mutants killed.**

### 2.7j The coverage gate was a fixed ±10% and should have scaled with σ√T

`rnd._coverage_ok` decided whether a recovered Q was trustworthy at all — and
`models/edge.py` refuses to trade when it says False. The rule was a **fixed ±10%
of spot**, with no tenor in it. The width a strike-integral needs scales with
σ√T, so that one number meant:

| chain | ±10% is worth | recovered vol error |
|---|---|---|
| 30d, 15% vol | 2.33 σ | fine |
| 90d, 22% vol | 0.92 σ | ~−1.2 vol points |
| 180d, 35% vol | **0.41 σ** | **~−2.7 vol points**, flag still True |

Too low is the dangerous direction: it makes the market look **cheap**, which
suppresses selling and can invite buying.

The threshold is measured, not chosen. On **European** chains — where early
exercise plays no part at all, so any bias is the integral's own truncation — the
error depends on coverage in σ√T units and barely on tenor or vol:

| coverage | error in the recovered vol |
|---|---|
| 1.0 σ | −1.16% to −2.68% |
| 1.5 σ | −0.33% to −0.80% |
| 2.0 σ | −0.07% to −0.27% |
| **2.5 σ** | **−0.001% to −0.10%** |
| 3.0 σ | ~0 |

2.5 is the first level whose worst case is small against the 1–4 vol points of
premium this engine harvests; 2.0 would admit up to a quarter of a one-point edge.
The width is sized from an **ATM** implied vol, deliberately not from the
model-free estimate — that is the quantity being tested, and it is biased low
exactly when coverage is poor, so using it would shrink the requirement on the
chains that need it widened.

This began as a check on something else. US single-name and ETF options are
American and the docs call de-Americanizing mandatory before Q extraction — and
the dashboard never did it. Measured, that bias is **+0.2% relative** at 30d, and
de-Americanizing overshoots about as far the other way; it only flips a decision
when |VRP| is under ~0.04 vol points, which is a no-edge zone anyway. Chasing it
to 90d and 180d is what exposed the real error, which was an order of magnitude
larger and had nothing to do with early exercise.

**It refuses more chains now, and the refusals say why**: `coverage_report` prints
"puts stop at −15.4%; 2.5 sd at an ATM vol of 25.5% over 30d needs ±18.3%" instead
of a bare False. Five fixtures had to be widened, and every one of them was
truncated by a **price floor**, not by a narrow strike grid: a flat-vol synthetic
chain prices deep puts below a cent, so the wings vanish. Real markets keep them
quotable because skew prices them far above flat-vol value. The Deribit control
test is the evidence this is real and not pedantry — widening its ladder from ±20%
to ±45% at a 60% vol took its recovered vol from needing a 0.03 tolerance to an
error of **+0.0036**.
### 2.8 The volatility input, measured against a known answer

The fetcher discarded 5 of the 6 fields Yahoo already returns. The range carries
information about the path between closes, and a squared close-to-close return is a
one-observation variance estimate.

Efficiency is an ORACLE — true σ is fixed and known:

| estimator | measured | published |
|---|---|---|
| Parkinson | 5.27× | ~5.2 |
| Garman-Klass | 8.33× | ~7.4 |
| Rogers-Satchell | 6.51× | ~6–8 |

End to end on `term_vol`, the function `baseline.py` actually calls, 21-day horizon
against a known latent variance:

| input | RMSE (vol pts) | bias |
|---|---|---|
| close-to-close | 8.523 | 1.041 |
| range GKYZ | **6.542** | **1.005** |

Paired over 40 worlds the squared-error difference is +2.05e-03, 90% interval
[+1.16e-03, +2.95e-03] — it excludes zero. **This is an oracle measurement, not a
backtest**: it says the same model forecasts better on a better input and says
nothing about profit.

Two traps are guarded because both point the dangerous way — a p_vol biased LOW
makes every expiry look rich and turns the book permanently short volatility:

- Yahoo adjusts only the CLOSE. Pairing a raw open with an adjusted previous close
  makes a 0.7% dividend read as a 0.7% gap. O/H/L are rescaled by `adjclose/close`.
- Session-only estimators miss the overnight gap and read ~0.87 of the truth, so
  the gap term is explicit and `calibration_scale` catches the residual on TRAIN
  data only.

Measured while choosing between them, which changed the conclusion:

| drift/σ | Parkinson | Garman-Klass | Rogers-Satchell |
|---|---|---|---|
| 0.0 | 0.913 | 0.881 | 0.883 |
| 2.0 | **2.226** | 1.286 | 0.810 |
| 4.0 | **6.375** | 2.531 | 0.691 |

Parkinson reads a TREND as volatility. Yang-Zhang is deliberately absent: its
selling point is drift-independence, which Rogers-Satchell already has, and the
"14× more efficient" figure is a theoretical bound no market meets.

### 2.9 Two screens that can close a question early

**`tools/oi_share.py`** answers the crowding premise's necessary condition in an
afternoon rather than in 18 months: does a fund hold a meaningful share of the open
interest at its own strike? Bands were fixed before any data was seen
(PREREGISTRATION.md §2.2a): **<5% refutes and closes §2.2**, 5–20% is inconclusive,
≥20% says the mechanism is not ruled out. Open interest of 0 means UNKNOWN, and a
chain without it returns NO OPEN INTEREST DATA rather than a refutation.

**`models/ticket.py`** turns a view into an order or a named refusal — **eight**
codes, each tested firing alone, accumulating rather than masking each other. A
placeable ticket's contracts × max-loss is inside its budget for every input,
swept — and now swept again with a decision in the path, because a second cap must
not be able to break the first. It takes the fused `decision`, so the size cut that
sector rotation and fund crowding bought actually reaches the contract count (§2.7).
It is reached from `run_live` and not only from the demo, so the real-data path ends
at an order rather than at a view.

---

## 3. How to run it

### 3.1 Every trading day — one command, scheduled once

```bash
python3 -m tools.daily install --holdings holdings --ledger paper.jsonl \
        --source tradier --symbol SPY --dte 30 --dashboard rotation.html
# writes daily-job.bat/.sh and PRINTS the scheduler line; add --apply to register it

python3 -m tools.daily run       # what the scheduler will run
python3 -m tools.daily status    # how far along the clocks are
```

`--dashboard` renders the page at the end of the capture, from whatever the steps
actually produced. That is not decoration. The clocks this job exists to advance
were visible only as text from a command nobody had to run twice, and **a clock
nobody looks at is one nobody winds** — which is this project's largest risk,
stated in the module's own docstring. It renders on a partial capture too: if the
price fetch fails it draws the generated fixture and says so in the step log, and a
page reading `0 settled` is the honest state and more use than no page.

`install` writes a wrapper script and hands the scheduler ONE quoted path. It used
to interpolate `sys.executable` unquoted into a `schtasks /tr` value that was
already double-quoted, so `C:\Program Files\Python313\python.exe` split at the
space and the task failed every night while schtasks reported it registered — a
clock the operator believed was running. It also dropped `--symbol`, so a job
configured for SPY scheduled itself without it and recorded SPX instead, forever.
Flags now come from one table and a test asserts the whole table reaches the
command, not just the flag that broke.

That test, however, **asserted on a hardcoded list of eight flags while its own
docstring promised that "a seventh flag added tomorrow and forgotten fails here"**
— so it could not have caught the thing it claimed to guard. Adding four flags for
the dashboard is exactly the case it was supposed to cover. The expectation is now
derived: every key `_cfg` produces must have an `_EMIT` rule, every rule that fires
must put its flag on the line, and a second test spies on the parser to catch the
other half of the same leak — a flag accepted on the command line that never
reaches the job config, and is therefore silently ignored.

Note the defaults: `--ledger` and `--prices` are EMPTY, so a bare `run` archives
holdings only and says nothing about skipping the rest. Pass them explicitly.

The largest risk to this project is not a modelling error, it is that nobody runs
the daily job. A fund publishes today's option book and overwrites it, so a day
missed is gone at any price, and `crowding_backtest` needs ~630 trading days of
them. A routine that depends on remembering a command every morning does not
happen, so this collapses it to one scheduled invocation that keeps going when a
step fails and reports the clock in the only units that matter:

```
  fund holdings   [####................] 22%   144/630 trading days
                  first 2026-01-05  latest 2026-07-23  missing 0
                  the crowding study becomes possible around 2028-06-15
  paper ledger    NOT STARTED — forward-test time only accrues once it does
```

The underlying tools still run standalone if you prefer:

```bash
python3 -m tools.archive_holdings fetch --dir holdings
python3 -m tools.paper_trade record --source tradier --symbol SPY --american \
        --dte 30 --ledger paper.jsonl --events-json earnings.json
```

`fetch` exits 1 when NOTHING was captured for today — it used to return 0 while
printing "0/4 newly archived", so a scheduled job looked green on a day it banked
nothing irrecoverable. "already have it" counts as success; a same-day re-run and a
weekend are not failures.

`record` needs `--american` on SPY/QQQ (PREREGISTRATION.md §4.5) and snaps `--dte`
to the nearest listed expiry, recording both tenors in the entry. Without the snap
it could not record at all: Tradier returned the 3 soonest expiries, which on a
near-daily-expiry ETF are 1–4 DTE, where the ±10% wings fail `coverage_ok` — the
reachable tenors and the tradeable ones did not intersect.

### 3.2 Any time — read the market

```bash
python3 -m tools.run_live deribit --currency BTC --dte 30 --dump btc.json
python3 -m tools.run_live tradier --symbol SPX --dte 30            # needs a free key
python3 -m tools.run_live replay  --chain-json btc.json --price-json p.json
python3 -m tools.run_live tradier --symbol QQQ --dte 30 --american --gate
```

Prints: model-free Q vol + BKM moments, the P-vs-Q scan, the per-strike board, the
PIT calibration with its effective sample size, the trade card, the fused decision,
**the order ticket** (structure, strikes, expiry date, contracts, limit, dollar worst
case — or a named refusal) — and a **simulation** clearly labelled as not being a
backtest of the chain above.

The ticket sizes against `--equity` (default $100,000) at `--max-risk-frac` (default
2% of equity *at maximum loss*, not at premium). Its evidence label is read from
`--ledger` on disk rather than from a flag, because the settled-trade count is the
one number an operator has an incentive to inflate, so it is not made typeable:

```bash
python3 -m tools.run_live tradier --symbol SPX --dte 30 \
        --equity 250000 --max-risk-frac 0.02 --ledger spx.jsonl
```

### 3.3 Getting price history in

One command, no key, no signup:

```bash
python3 -m tools.fetch_prices --out sectors.csv     # 11 sector ETFs + SPY
python3 -m tools.fetch_prices --source stooq --out sectors.csv    # pin one source
```

Two sources, tried in order — Yahoo's chart JSON first, Stooq's daily CSV second.
Both are keyless and quota-free, which is what lets this sit in a script somebody
actually re-runs. Either is a convenience feed, not a survivorship-safe research
database — neither will tell you about delistings and each adjusts prices its own
way — so they are fine for deciding whether a rotation chart predicts anything on
eleven large liquid ETFs, and not for a claim that money was made.

**Why two.** Stooq put a JavaScript bot-check in front of its CSV endpoint and
began answering every request with an HTML page saying *"This site requires
JavaScript"* — **with HTTP 200**. Nothing raised. A fetcher that trusted the
status code would have written twelve HTML files into the cache, reported
success, and left a cache that looks real forever. So every response is validated
by **parsing it**, never by its status code: a file that yields no date column and
no close column is not cached, whatever the server said about it. A page where
data was expected is reported as `BLOCKED: the vendor served a web page`, which
names the cause instead of leaving a mystery, and the basket falls through to the
next source. One dead ticker never aborts the other eleven.

When every source is blocked the run says so and points at the chart-export path
below; when every source fails with a network error it says *that* instead —
different problems, different advice, and the tool does not dump the vendor's HTML
into your terminal in either case.

If you would rather use chart exports: there is no public data API for TradingView
(the Charting Library is a renderer you feed, Pine runs on their servers and
cannot hand data back, and scraping the feeds is against their terms), but
**Export chart data...** gives one CSV per symbol and this merges them:

```bash
python3 -m tools.merge_csv --dir tradingview_exports --out sectors.csv --require SPY
```

Rows are matched on the DATE, never on row number: symbols have different
holidays and listing dates (XLRE lists 2015, XLC 2018), and a positional merge
would pair one sector's Tuesday with another's Wednesday and corrupt every
relative-strength number with nothing downstream able to notice. The merge prints
each symbol's own span and names whichever one truncated the join.

Real time is not the constraint here. The RRG needs 63 bars of history per point
and the backtest wants 1,300+ bars for ~55 rebalance dates; one live tick changes
nothing. TradingView also cannot supply an option chain at all, which is the core
input for everything on the options side.

### 3.4 The dashboards

```bash
python3 -m tools.rotation_dashboard --csv sectors.csv --out rotation.html
python3 -m tools.rotation_dashboard --demo --out rotation.html      # no data needed
python3 demo.py                                                     # the whole loop
```

**Or all of §3.3–3.4 at once** — update, fetch, render, open, and save the two
verdict lines to `RESULT.txt`:

```bash
python3 -m tools.quickstart          # START.bat on Windows
python3 -m tools.quickstart --demo   # generated world, no network
```

Worth saying why a launcher earns its place in a research repo. Every failure it
removes was a shell mistake, not a modelling one: a placeholder path pasted
literally, two commands merged onto one line, `cd options-alpha-engine` run from
inside `options-alpha-engine`, and — the expensive one — a blocked vendor that
wrote no CSV, so the *render* step failed with `FileNotFoundError: sectors.csv`
and the actual cause had already scrolled off the screen. So the launcher runs
every step from its own directory rather than the shell's, runs them in one
process so they cannot be merged or reordered, and refuses to start a step until
the previous one produced the file it promised. A run that cannot finish names
the step that stopped it.

It also pins the encoding, which is not cosmetic: the page and the verdicts are
full of em dashes, and both `print()` and `open()` otherwise use whatever
codepage the machine has. Run under an interpreter whose default encoding is
ASCII, the old path raised `UnicodeEncodeError` from inside `print` at the moment
the answer was ready — demonstrated in a subprocess, not observed in the wild;
the codepages actually reported so far happen to carry an em dash. The page also
declared no charset at all, so a browser opening it locally guessed. It now
declares `<meta charset="utf-8">` and is written as utf-8, and the console is
asked for utf-8 with `errors="replace"`, so a punctuation mark cannot end a run
that has already computed its result.

The launcher runs the **pre-registered** parameters (`window` 63, `mom_lag` 5,
`tail` 12, `horizon` 21), and a test asserts they still equal both the CLI's
defaults and the rows in `PREREGISTRATION.md` §2.1. A one-click path that
quietly ran a different experiment would produce numbers that look official and
cite a registration that does not describe them.

### 3.5 In Python

```python
from engine.book_backtest import run_book_backtest
from engine.robustness import seed_ensemble, block_bootstrap, bootstrap_summary
from engine.rotation_backtest import quadrant_panel, run_rotation_backtest
from engine.crowding_backtest import run_crowding_backtest
from tools.archive_holdings import supply_at_from_archive

# the crowding study, fed straight from the archive (point-in-time enforced)
supply_at = supply_at_from_archive("holdings", "QQQ", dates, prices, lag_days=1)
print(run_crowding_backtest(prices, chain_at, supply_at, dte=21).summary())

# never quote a Sharpe without this
print(bootstrap_summary(block_bootstrap(result.trade_pnl)))
```

### 3.6 Tests

```bash
for t in tests/test_*.py; do python3 "$t"; done      # 562 tests, 47 files
```

Run with `PYTHONDONTWRITEBYTECODE=1`. A same-length constant edit inside one second
matched a `.pyc`'s mtime and size once, and a restored file went on testing as
though it were still mutated — which silently inverted a mutation sweep.

`tests/test_wiring.py` is worth knowing about: it fails if any module is built,
tested, and reachable from nothing a person runs. That is not hypothetical —
`models/events.py` passed its own tests for weeks while being consumed by nothing
but a display card, so every backtest number in the repo was produced with the
scheduled-event gate silently off, and no unit test could see it.

---

## 4. What this has NOT shown

Stated here rather than left to be discovered.

1. **No edge on any real market.** Except where §7 says otherwise, every number in
   §2 comes from fixtures written by the author or from an oracle. The engine is
   verified; the strategy is not.
2. **The crowding question is unanswered** and stays unanswered until the archive
   reaches ~18 months — or until `tools/oi_share.py` refutes its premise in an
   afternoon, which is the cheaper order to try them in.
3. ~~The rotation result is synthetic.~~ **RUN, AND IT ANSWERED NO.** Real SPDR
   sector history, 91 rebalances: the panel needed |t| ≥ 2 and got **+0.71**; the
   book returned **−27.0%** net of costs. PREREGISTRATION.md §7.1. The measuring
   instrument was also found broken and has since been rebuilt — the placebo
   comparison recorded there is retracted, see §6.1.
3b. **The forward ledger has settled 0 trades and the archive holds 0 days.** No
   amount of engineering changes this; only calendar time does.
4. **The crash tail is a hand-set prior, but fitting cannot beat it.**
   `models/tail_fit.py` fits all six shape constants walk-forward: in-sample loss
   improves every time, out-of-sample **never** (−0.8% at 5,000 bars, −1.6% at
   9,000). The shape is not identifiable from a few hundred non-overlapping
   monthly windows, so the defaults act as a regulariser — but the promotion gate
   still means "beat a prior nobody can beat by fitting", not "models crashes".
5. **No parameter is walk-forward selected** except the calibration vol scale.
   That is why `PREREGISTRATION.md` locks them before real data arrives.
6. **No margin, financing, borrow, market impact beyond a flat spread, latency, or
   partial fills.** Sizing output is a contract count, not an executable order.

---

## 5. Next, in order

| # | step | blocked on | why this order |
|---|---|---|---|
| 1 | `oi_share` on one archived book + one live chain | one day of holdings, a free Tradier token | **can close §2.2 in an afternoon instead of in 18 months** |
| 2 | `archive_holdings fetch` daily | nothing | the only irreversible clock; a day missed is gone |
| 3 | `paper_trade record` daily | a free Tradier token | forward time only accrues once started |
| 4 | crowding study | step 2 reaching ~18 months, and step 1 not refuting it | needs dated books that do not exist yet |
| 5 | options edge verdict | step 3 reaching 30 settled trades | the only honest test of the core claim |

**Step 1 first, and it is new.** Every other step spends calendar time. This one
spends an afternoon and can make step 4 unnecessary: if the funds hold under 5% of
the open interest at their own strikes, the mechanism is arithmetically ruled out
and the 18 months are better spent on step 3. Two things are already known and
recorded in §2.2a — JEPI and JEPQ hold no listed options at all (OTC equity-linked
notes), and Cboe has measured the aggregate version of the thesis and found call
skew going UP as these funds grew sixfold.

**Steps 2 and 3 are the only irreversible ones.** Everything else in this document
can be done later at the same cost. These cannot.
