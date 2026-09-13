# Quickstart — test the engine on real options data

## 0. The one-command path (sector rotation, real prices)

**Windows:** open the `options-alpha-engine` folder and **double-click `START.bat`**.

**macOS / Linux:**

```bash
python3 -m tools.quickstart
```

It updates the checkout, downloads prices, renders the chart, opens it, and
writes the verdict lines to `RESULT.txt`. It runs every step **from its own
folder**, so it does not matter where your shell is — and if the data step
produces nothing it stops there and names the reason, instead of letting the
next step die on a missing file.

No network, or the vendor is blocking? `--demo` renders a generated world and
downloads nothing:

```bash
python3 -m tools.quickstart --demo
```

**Put the OPTIONS decision on the page too.** Without a chain the dashboard can only
show the context signals — and the one study this repo has run on real data answered
NO for the sector quadrant, so a page with only that half is a page leading with its
weakest number. It says so plainly rather than looking complete. Dump a chain once,
then pass it:

```bash
python3 -m tools.run_live tradier --symbol SPX --dte 30 --dump spx.json
python3 -m tools.quickstart --chain-json spx.json --price-json spx_px.json \
        --symbol SPX --equity 250000 --ledger spx.jsonl
```

The page then leads with the vol side, P vs Q, the VRP, the fused size multiplier,
and **the order itself** — every leg, the limit, the dollar worst case — or a named
refusal.

`--ledger` adds the second panel, and it is the one that can answer *"does the edge
persist?"* — recorded / settled / open, **P against Q over time** (the gap between
the lines is the variance premium), the calibration and profit scoreboards, and the
verdict. It also supplies the ticket's evidence label from the settled count on
disk: read, never typed. With no ledger it prints **0 recorded, 0 settled** and the
command that starts the clock, because an invisible clock is one nobody winds.

## 0b. Everything else, no network, no installs

```bash
cd options-alpha-engine
python3 demo.py                     # full loop on synthetic data
python3 -m tools.check_offline      # prove no test needs a network
for t in tests/test_*.py; do python3 "$t"; done   # 547 tests, 47 files
```

## 1. Test on REAL options (run where outbound network is open)

> ⚠️ A hosted web session's sandbox usually blocks market-data hosts. Run these
> on **your own machine**, or in an environment whose **network policy allows the
> vendor host** (e.g. `www.deribit.com`). Nothing here needs a paid account.

### Deribit — keyless, European crypto options (the zero-friction path)

BTC/ETH options are **European + cash-settled**, so the BKM / VIX / Breeden-
Litzenberger extraction is valid with no de-Americanizing — the cleanest first
real-data test.

```bash
python3 -m tools.run_live deribit --currency BTC --dte 30 --dump btc.json
```

You get: model-free (VIX-style) Q vol + BKM skew/kurtosis, a per-expiry P-vs-Q
variance-risk-premium scan, and a delta-hedged walk-forward backtest on BTC-
PERPETUAL history. **Cross-check** the printed model-free vol against Deribit's
**DVOL** index for the same time — agreement within a vol point or two validates
the whole Q path on real options.

### Tradier — free-key, SPX/XSP European equity options

```bash
export TRADIER_TOKEN=...            # free sandbox token: developer.tradier.com
python3 -m tools.run_live tradier --symbol SPX --dte 30
```

SPX/XSP are European cash-settled index options (clean BKM, USD quotes). Tradier
Sandbox also does **paper orders** (`engine.tradier.TradierAdapter.place_paper_order`).

### Replay offline — freeze once, iterate forever

`--dump` writes the chain to the JSON schema the offline adapter reads, so you can
re-run the exact snapshot with no network (great for CI / reproducible research):

```bash
python3 -m tools.run_live replay --chain-json btc.json --price-json prices.json
```

## 1b. Build a REAL track record — the forward-test paper ledger

`run_live` is a one-shot spot check. To actually answer *"is the edge real?"* you
need to accumulate dated forecasts and grade them once their horizon elapses. That
is `tools/paper_trade.py` — record → settle → proper-score — with a one-command CLI:

```bash
# --- run ONCE PER TRADING DAY: freeze today's forecast + the market Q ---
python3 -m tools.paper_trade record --source deribit --currency BTC \
        --dte 30 --ledger btc.jsonl

# --- any day: grade everything matured and print the scoreboard ---
python3 -m tools.paper_trade report --ledger btc.jsonl
```

A companion append-only price journal (`btc.jsonl.prices.json`) is seeded from
history on the first `record` and grows one close per day, so every entry keeps a
**stable index** and `no look-ahead` holds even as the vendor's window slides. The
first graded score therefore appears only after `dte` more daily records — that lag
is the honest cost of a real out-of-sample test.

You get two verdicts, kept honest and separate: **CALIBRATION** (did our physical
`P` density beat the market-implied `Q` one out-of-sample, by NLL and left-tail?)
and **PROFIT** (did the signal-fired, delta-hedged, cost-inclusive trades make
money?). The same thing runs fully offline on a replayed snapshot —
`--source replay --chain-json c.json --price-json p.json` — or in Python via
`PaperLedger` / `paper_trade_series(...)` (that is what `demo.py` section 6 prints).

## 2. What to look at (and not fool yourself)

- **`coverage_ok`** must be `True` — sparse/illiquid wings bias the Q variance LOW,
  exactly in stressed regimes. Prefer liquid index / BTC-ETH chains.
- **American options (US single-name / ETF: QQQ, SPY, ...)** must be de-Americanized
  before Q extraction — pass `--american` to `run_live` (binomial American IV ->
  European-equivalent prices; raw American prices bias the recovered variance HIGH).
  European instruments (SPX/XSP index, BTC/ETH) need no flag.
- A single snapshot is a **spot check**, not evidence of edge. The milestone is an
  honest **out-of-sample, cost-inclusive equity curve over many dates including a
  crash** (`--gate` adds the MDN-vs-HAR promotion check; run the backtest over long
  history). Gross single-name VRP is only ~1–4 vol points — costs eat most of it.

## 3. Data-source tiers (see `models/MODEL_SPEC.md` for the full map)

| Tier | Sources | Note |
|---|---|---|
| 0 · start today | Deribit (keyless), OptionsDX free SPX CSV | European, zero/near-zero auth |
| 1 · free-key | Tradier Sandbox, Deribit testnet, marketdata.app | adds paper execution |
| 2 · American (US equity/ETF) | Polygon/Finnhub free | defensible Q via `--american` de-Am |
| 3 · paid, defensible | ORATS, CBOE DataShop, OptionMetrics/WRDS | survivorship-safe |
