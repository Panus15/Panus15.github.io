# Quickstart — test the engine on real options data

## 0. Right now, no network, no installs

```bash
cd options-alpha-engine
python3 demo.py                     # full loop on synthetic data
for t in tests/test_*.py; do python3 "$t"; done   # 108 tests
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
is `tools/paper_trade.py` — record → settle → proper-score, persisted to an
append-only JSON-lines file so you record live and settle offline later:

```python
from tools.run_live import fetch                      # the network boundary
from tools.paper_trade import PaperLedger
from models.baseline import BaselineDensityForecaster

# --- each trading day: freeze today's forecast + the market Q (no look-ahead) ---
led = PaperLedger.load("btc.jsonl") if __import__("os").path.exists("btc.jsonl") else PaperLedger()
chain, prices = fetch("deribit", type("A", (), {"currency": "BTC", "days": 400})())
led.record(chain, prices, len(prices) - 1, BaselineDensityForecaster(), dte=30)
led.save("btc.jsonl")

# --- any time later: settle everything matured, print the scoreboard ------------
led.settle(prices)                                     # prices = the realised path
print(led.report().summary())
```

You get two verdicts, kept honest and separate: **CALIBRATION** (did our physical
`P` density beat the market-implied `Q` one out-of-sample, by NLL and left-tail?)
and **PROFIT** (did the signal-fired, delta-hedged, cost-inclusive trades make
money?). Offline, `tools.paper_trade.paper_trade_series(...)` replays the whole
loop deterministically (that is what `demo.py` section 6 prints).

## 2. What to look at (and not fool yourself)

- **`coverage_ok`** must be `True` — sparse/illiquid wings bias the Q variance LOW,
  exactly in stressed regimes. Prefer liquid index / BTC-ETH chains.
- **European instruments only** for a real Q number. US single-name equity options
  are American (Polygon/Finnhub/Alpaca free tiers) — plumbing smoke tests only.
- A single snapshot is a **spot check**, not evidence of edge. The milestone is an
  honest **out-of-sample, cost-inclusive equity curve over many dates including a
  crash** (`--gate` adds the MDN-vs-HAR promotion check; run the backtest over long
  history). Gross single-name VRP is only ~1–4 vol points — costs eat most of it.

## 3. Data-source tiers (see `models/MODEL_SPEC.md` for the full map)

| Tier | Sources | Note |
|---|---|---|
| 0 · start today | Deribit (keyless), OptionsDX free SPX CSV | European, zero/near-zero auth |
| 1 · free-key | Tradier Sandbox, Deribit testnet, marketdata.app | adds paper execution |
| 2 · American (plumbing only) | Polygon/Finnhub free | NOT a defensible Q number |
| 3 · paid, defensible | ORATS, CBOE DataShop, OptionMetrics/WRDS | survivorship-safe |
