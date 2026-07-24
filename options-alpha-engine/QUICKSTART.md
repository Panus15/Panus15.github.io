# Quickstart — test the engine on real options data

## 0. Right now, no network, no installs

```bash
cd options-alpha-engine
python3 demo.py                     # full loop on synthetic data
for t in tests/test_*.py; do python3 "$t"; done   # 136 tests
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
