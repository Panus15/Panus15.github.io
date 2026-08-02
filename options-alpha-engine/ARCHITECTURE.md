# Architecture — how the engine thinks

A guided tour of the whole system, from the one idea it is built on to the last
line of the honesty checklist. If you read only one file to understand this
codebase, read this one.

---

## 1. The one idea

Most option "signals" try to predict **direction** — will the underlying go up or
down? Almost nobody has a real, durable edge there, and backtests that claim one
are usually fooling themselves.

This engine predicts something else entirely: the **whole probability
distribution of the terminal price** `S_T`, and compares it to the distribution
the option market is *already pricing in*. Two distributions, two measures:

| | symbol | what it is | who builds it |
|---|---|---|---|
| **Physical** | **P** | what we forecast will *actually* happen | `models/baseline.py` (HAR-RV) |
| **Risk-neutral** | **Q** | what the option chain *implies* | `models/rnd.py` (BKM) |

The tradeable quantity lives in the **gap** between them:

```
variance risk premium   VRP = Var_Q − Var_P      (implied variance − forecast variance)
skew   risk premium      SRP = skew_Q − skew_P
```

In equilibrium `Var_Q > Var_P` almost always — implied vol trades above realised
vol. Selling that gap (short vol, delta-hedged) harvests the **variance risk
premium**. But — and this is the whole discipline — **VRP is a risk premium, not
free money.** You are paid it because you lose badly in a crash. So the engine
never just "sells because VRP > 0"; it trades the *deviation* of VRP from its own
history and refuses to sell into a developing storm.

Everything else in the repo is machinery around that single comparison.

---

## 2. The data flow

```
   ADAPTERS            DE-AM              Q-SIDE                P-SIDE
 Deribit / Tradier   american.py     rnd.py + surface.py   baseline / mdn / neural
 IBKR / Polygon  ─▶  (US options ─▶  BKM / VIX / SVI  ┐    HAR-RV → MixtureLogNormal
 CSV/JSON replay      → European)                     │              │
                                                      ▼              ▼
                                              ┌───────────────────────────┐
                                              │   edge.py   P  vs  Q       │
                                              │  VRP z-score, per-strike   │
                                              └───────────┬───────────────┘
                                     GATES ────────────────┤
                          price (edge.regime_stressed)      │  strike_scan.py
                          news  (news_signal + sentiment)   │  per-contract board
                          macro (macro.py)                  ▼
                                              ┌───────────────────────────┐
                                              │  portfolio.py  govern      │
                                              │  vega cap · CVaR · kill    │
                                              └───────────┬───────────────┘
                                                          ▼
                          PROVE:  hedged_backtest · signal_backtest · benchmark
                                  objective (promotion gate) · paper_trade (live-forward)
```

Every box is a small, separately-tested module. Data enters on the left; an
honest, cost-inclusive verdict leaves on the right.

---

## 3. Module by module

### Read the market (the Q side)
- **`engine/adapters.py`, `deribit.py`, `tradier.py`, `ibkr.py`** — turn any
  vendor into one `OptionChain`. Deribit is keyless (European crypto — the
  zero-friction real-data path). Offline `CsvFileAdapter` / `JsonFileAdapter`
  replay a frozen snapshot forever.
- **`engine/american.py`** — US single-name / ETF options (QQQ, SPY, the holdings
  inside income funds like QQQI) are *American*; their price carries an
  early-exercise premium that biases model-free variance **high**. This solves the
  American implied vol on a binomial tree and re-prices a European equivalent, so
  the Q-extractors stay valid. Without it, US equity Q numbers are simply wrong.
- **`models/rnd.py`** — the Q engine: model-free (VIX / Britten-Jones-Neuberger)
  variance, and Bakshi-Kapadia-Madan skew & kurtosis. Forward from put-call
  parity, spot-anchored, `Var_Q = e^{rT}V − μ²`. A coverage guard flags
  sparse/truncated chains where the number can't be trusted.
- **`models/surface.py`** — an SVI fit densifies a sparse or wing-truncated chain
  before extraction, so the Q number survives thin real-world chains.

### Forecast (the P side)
- **`models/density.py`** — the shared output type: a **mixture of log-normals**
  over `S_T`. Keeps price positive, has closed-form CDF and option payoffs, and is
  exactly what a Mixture Density Network head emits. Its oracle test: a
  single-component risk-neutral mixture reprices Black-Scholes *exactly*.
- **`models/baseline.py`** — ships first. A HAR-RV vol forecast → a calm+crash
  two-component mixture with **data-driven** skew (leverage effect + horizon decay
  + realised skew), so the shape genuinely varies with regime instead of being a
  scale-invariant constant.
- **`models/mdn.py`, `neural.py`** — trainable density heads that emit the *same*
  `MixtureLogNormal`, so they are drop-in replacements for the baseline.

### Decide
- **`models/edge.py`** — the P-vs-Q comparison. Trades the VRP **z-score** (rich
  vs its own past), gated by regime, charged the round-trip spread + commission +
  a hedge estimate. An edge that dies crossing costs is dropped.
- **`models/strike_scan.py`** — zooms from "sell this expiry's vol" to "*which
  contract*": for every call/put, the model's P(finish ITM) vs the market-implied
  `N(d2)`, and fair value vs bid/ask after costs → BUY / WRITE / FAIR. Still
  direction-neutral: edges come from vol level and distribution shape. Each signal
  carries `break_even_slip_frac` — how many spreads of slippage the edge survives.
- **`models/spreads.py`** — the *structural* answer to the tail problem. VRP is
  paid because you lose in a crash; rather than forecast the tail better, stop
  selling naked vol and sell DEFINED-RISK structures (iron condor, put credit
  spread) whose max loss is capped by long wings. Each is scored `EV = credit −
  E_P[loss] − cost` against the physical density, with break-evens and P(profit).
  You give up some premium to buy back the un-hedgeable tail — the honest way to
  actually harvest the premium the whole engine is built around.
- **Gates** — three forward-looking vetoes that all OR into one `stressed` flag
  passed to `edge.compare`. Never directional; they only ever *stop* selling vol:
  - `edge.regime_stressed` — realised short-vol accelerating (backward-looking).
  - `models/news_signal.py` + `sentiment.py` — a burst of negative/dispersed news.
  - `models/macro.py` — yield-curve inversion, credit blowout, VIX-term
    backwardation, or a tightening shock.

### Control
- **`engine/portfolio.py`** — correlation-aware short-vega cap (N identical shorts
  are N× the risk, not √N), CVaR sizing (not binary Kelly), a drawdown
  kill-switch, and defined-risk conversion.

### Prove (the part that matters most)
- **`engine/hedged_backtest.py` / `signal_backtest.py`** — delta-hedged,
  walk-forward, non-overlapping, cost-inclusive, governor-gated. Non-overlapping
  matters: overlapping trades inflate Sharpe.
- **`engine/benchmark.py`** — the same path run three ways: the signal book vs
  *always-sell* (mechanically what a covered-call income ETF does — its yield is
  harvested premium, not a win rate) vs buy-and-hold. The honest bar.
- **`models/objective.py`** — the promotion gate: NLL / CRPS / pinball + a
  decomposed **left-tail** score. A fancier model must beat the HAR baseline
  out-of-sample on *both* the aggregate and the tail before it ships. (On real
  GOOG-2008 data the MDN won aggregate NLL but lost the crash tail — so the gate
  correctly refused it.)
- **`tools/paper_trade.py`** — the forward-test ledger: record today's forecast +
  the market's Q, settle when the horizon elapses, and score both with proper
  scoring rules. This is the only instrument that can turn "the math works" into
  "the edge is real," and it deliberately refuses to claim edge on modelled data.
- **`engine/stress.py`** — overnight-gap / jump stress. Continuous delta-hedging
  is a fiction; markets gap over nights and weekends and you are short gamma
  across the jump. Inject jumps and re-run the book so the gamma bleed a smooth
  backtest omits becomes a number. (A finding worth internalising: *small* gaps
  can help — you then sell elevated IV — but a *big* un-hedgeable move bleeds; the
  gate defends the tail, not the average.)
- **`engine/crowding_backtest.py`** — the experiment behind `models/fund_flow.py`.
  Knowing *where* the income ETFs concentrate their short calls is only worth
  something if selling there actually pays differently, so this measures it: on
  each date sell BOTH a crowded strike and the nearest uncrowded one, and read the
  **paired** difference (same date, same expiry, so market beta cancels). Two
  design choices carry the whole result — the control leg is **matched on
  moneyness** (pairing a 6%-OTM crowded call against whatever uncrowded strike
  came first puts the high-gamma near-the-money contract on the control side and
  manufactures an effect: it produced a spurious *t* = −2.27 before the fix), and
  the verdict **refuses to conclude** below 30 pairs or |*t*| < 2. The harness
  ships tested in both directions; the answer needs real published holdings.

---

## 4. The honesty checklist (baked into the code, not the README)

1. **No look-ahead.** Forecasts see only prices up to the decision date; the paper
   ledger keys entries to a stable, append-only price journal.
2. **Costs are mandatory.** Every backtest fills through the spread and pays
   commission; a vol-point edge is converted to dollars via vega and charged.
3. **VRP is a premium, not alpha.** Trade its z-score, gated by regime — selling
   the raw level just re-discovers the carry everyone already earns.
4. **The tail is scored separately.** Aggregate NLL is dominated by getting the
   vol level right; a model can pass it while emitting garbage crash tails, so the
   left tail has its own gate.
5. **Benchmarked against what you could just buy.** Beating always-sell (the
   income-ETF mechanic) and buy-and-hold is the bar, not an absolute Sharpe.
6. **Proven ≠ profitable.** The math is proven by oracle tests and four adversarial
   multi-agent reviews; the machinery runs on real *underlying* data (GOOG through
   2008). What has **not** happened is pricing a live option chain — so the alpha
   claim stays at zero until the engine is run live on a real market.

---

## 5. Where to start reading

`demo.py` runs the entire loop on synthetic data with zero dependencies — every
section above prints a few real lines. Then `tools/run_live.py` points the same
pipeline at a real, keyless chain, and `tools/paper_trade.py` accumulates the
out-of-sample track record that is the real milestone. See `QUICKSTART.md`.
