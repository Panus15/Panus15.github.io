# The pipeline — what runs, in what order, and what it has actually shown

One page covering the whole system: the flow, every measured number, and how to
operate it. `ARCHITECTURE.md` explains *why* each module is built the way it is;
this explains *how the parts run together* and *what they have proven*.

**Scale:** 45 test files · **463 tests, all green** · pure stdlib, no
numpy/scipy/pandas · every load-bearing change mutation-verified.

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

**`models/ticket.py`** turns a view into an order or a named refusal — six codes,
each tested firing alone, accumulating rather than masking each other. A placeable
ticket's contracts × max-loss is inside its budget for every input, swept.

---

## 3. How to run it

### 3.1 Every trading day — one command, scheduled once

```bash
python3 -m tools.daily install --holdings holdings --ledger paper.jsonl \
        --source tradier --symbol SPY --dte 30
# writes daily-job.bat/.sh and PRINTS the scheduler line; add --apply to register it

python3 -m tools.daily run       # what the scheduler will run
python3 -m tools.daily status    # how far along the clocks are
```

`install` writes a wrapper script and hands the scheduler ONE quoted path. It used
to interpolate `sys.executable` unquoted into a `schtasks /tr` value that was
already double-quoted, so `C:\Program Files\Python313\python.exe` split at the
space and the task failed every night while schtasks reported it registered — a
clock the operator believed was running. It also dropped `--symbol`, so a job
configured for SPY scheduled itself without it and recorded SPX instead, forever.
Flags now come from one table and a test asserts the whole table reaches the
command, not just the flag that broke.

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
PIT calibration with its effective sample size, the trade card — and a **simulation**
clearly labelled as not being a backtest of the chain above.

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
for t in tests/test_*.py; do python3 "$t"; done      # 463 tests, 45 files
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
