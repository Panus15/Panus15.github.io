# The pipeline — what runs, in what order, and what it has actually shown

One page covering the whole system: the flow, every measured number, and how to
operate it. `ARCHITECTURE.md` explains *why* each module is built the way it is;
this explains *how the parts run together* and *what they have proven*.

**Scale:** 39 test files · **326 tests, all green** · ~11,500 lines · pure stdlib,
no numpy/scipy/pandas · every load-bearing change mutation-verified.

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
                    │        Q: model-free   P: HAR-RV +    RS-Ratio /
                    │        vol, BKM skew   term structure RS-Momentum
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
                                        │
                                        ▼
                              portfolio.py  govern
                              CVaR size · correlation-aware
                              vega cap · intra-trade kill-switch
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

## 3. How to run it

### 3.1 Every trading day — one command, scheduled once

```bash
python3 -m tools.daily install --holdings holdings --ledger btc.jsonl
# prints the cron / schtasks line for this machine; paste it and forget it

python3 -m tools.daily run       # what the scheduler will run
python3 -m tools.daily status    # how far along the clocks are
```

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
python3 -m tools.paper_trade record --source deribit --currency BTC \
        --dte 30 --ledger btc.jsonl --events-json earnings.json
```

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
python3 -m tools.fetch_prices --out sectors.csv     # 11 sector ETFs + SPY, from stooq
```

Stooq serves plain daily CSV over a stable URL with no key and no quota, which
makes it the one source that can sit in a script somebody actually re-runs. It is
a convenience feed, not a survivorship-safe research database — it will not tell
you about delistings and its adjustments are its own — so it is fine for deciding
whether a rotation chart predicts anything on eleven large liquid ETFs, and not
for a claim that money was made.

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
for t in tests/test_*.py; do python3 "$t"; done     # 278 tests, 34 files
```

---

## 4. What this has NOT shown

Stated here rather than left to be discovered.

1. **No edge on any real market.** Every number in §2 comes from fixtures written
   by the author. The engine is verified; the strategy is not.
2. **The crowding question is unanswered** and stays unanswered until the archive
   reaches ~18 months. The harness is ready; the data is not.
3. **The rotation result is synthetic.** Real sector history is free and complete —
   this is the cheapest real answer available and it has not been run.
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
| 1 | run `archive_holdings fetch` daily | nothing | the only irreversible clock |
| 2 | run `paper_trade record` daily | nothing | forward time only accrues once started |
| 3 | sector rotation on real ETF history | **a CSV of daily closes** | free data, harness ready, first real answer |
| 4 | crowding study | step 1 reaching ~18 months | needs dated books that do not exist yet |
| 5 | options edge verdict | step 2 reaching 30 settled trades | the only honest test of the core claim |

**Step 3 is the bottleneck and the cheapest win.** Eleven sector ETFs plus SPY,
daily closes, free from any vendor — and it converts a synthetic finding into the
first real out-of-sample result this repository has ever produced.
