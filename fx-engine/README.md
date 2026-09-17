# fx-engine

A falsification engine for retail FX. It does not generate signals. It takes a
trading idea, your broker's costs and your price data, and tries to establish
that the idea cannot make money — reporting, by name, exactly which arithmetic
kills it and what would have to be different.

Most ideas do not survive. That is the product.

```bash
python3 demo.py                       # the whole loop, on data with nothing in it
python3 tools/analyse.py YOUR.csv --pair EURUSD --side bid \
    --round-turn 1.3 --swap-markup 0.012 --costs-measured
```

Pure standard library. No dependencies, no data vendor, no account required.

---

## Why it is built this way

The research behind it (`research/BRIEF.md`, with sources and per-claim quality
labels in `EVIDENCE.md` and `VERIFICATION.md`) found the same shape behind every
failure: **an edge that exists gross and dies net.**

| | gross | net |
|---|---|---|
| intraday predictability | real | dead after 1.01bp |
| carry, retail | real | dead after a 0.8%/yr markup on 200% notional |
| dollar factor | real | 59% of it is a carry leg retail never receives |
| real retail accounts | 61% win rate | still losing: 48 pip wins, 83 pip losses |

That last row is 43 million real FXCM trades. Those traders were not picking bad
patterns — they won more often than they lost and still lost. Nothing on a chart
prevents that; arithmetic does, and only if it is applied *before* the trade.

**On the patterns specifically.** Only head-and-shoulders has a substantial
peer-reviewed FX literature and it is negative (Lucke 2003; Chang & Osler 1999,
where it was profitable on 2 of 5 rates and dominated by a simple filter rule).
Candlestick strategies were indistinguishable from bootstrapped random OHLC
(Marshall, Young & Rose 2006). Double tops, triangles and flags have **never been
tested in FX** — the win-rate tables quoted for them are US equities, with no
costs, no stop and no exit rule. Every detection this engine produces carries its
own evidence label, and they cannot be switched off.

## What it measured about itself

`demo.py` runs the entire pipeline over seeded random walks, where nothing can be
true by construction, and counts how often a pattern still comes back tradeable:

```
                        corrected   judged alone   of 150
  at realistic costs            3             12
  at zero cost                 20             42
```

A costless backtest calls **28% of tests on pure noise tradeable**. That is what
"my backtest was profitable" is worth before costs — and it is why `costs.py` is
module one rather than a correction at the end.

Three still survive both costs and the Bonferroni correction, against a nominal
0.2%. That is not a bug in the correction: `edge_t_stat` treats trades as
independent and they overlap heavily, since the same bars generate many of them.
**A passing t-stat here is a floor on doubt, never proof.** A failing one is
decisive, because the test was already tilted the other way.

The demo also runs a positive control on a series that genuinely trends, because
a filter that refuses everything is indistinguishable from one that is stuck.

## The eight refusals

Every no has a name, the number that triggered it, and the number it needed.

| code | means |
|---|---|
| `NOTHING_MEASURED` | a planned payoff is not an edge |
| `COST_NOT_MEASURED` | you accepted the default costs instead of yours |
| `COSTS_DISAGREE` | the backtest ran on cheaper costs than the verdict uses |
| `TOO_FEW_TRADES` | the sample cannot support the claim |
| `NO_WIN_RATE_SAVES_IT` | cost ≥ the whole target; required win rate ≥ 100% |
| `NEGATIVE_NET_EXPECTANCY` | measured, and it loses |
| `NOT_SIGNIFICANT` | positive, but inside the noise once corrected |
| `GOVERNED_BY_ASSUMPTION` | the ambiguous-bar rule decided it, not the data |

Three arguments answered in code rather than in prose:

- *"I'll fix it with position sizing."* Expectation is linear in size, so scaling
  a negative number leaves it negative. The refusal quotes the scaled figure.
- *"The win rate is high."* A 90% win rate breaks even at a payoff of 0.111.
  `p* = (L+c)/(W+L)` is printed next to your measured rate on every verdict.
- *"It was profitable in the backtest."* Then it must be profitable by more than
  its own standard error, corrected across the hypotheses tried.

## The modules

| | |
|---|---|
| `engine/costs.py` | all-in round turn, swap markup charged in **both** directions, gross-notional drag |
| `engine/expectancy.py` | `E = p·W − (1−p)·L`, required win rate, why sizing cannot fix a sign |
| `engine/stats.py` | how many trades a claim needs; Wilson intervals, not the textbook one |
| `models/patterns.py` | six patterns, four declared parameters, an evidence label each |
| `engine/backtest.py` | built around the five ways a backtest lies |
| `engine/verdict.py` | the decision, and the eight named refusals |
| `engine/data.py` | loads your CSV and looks for seven kinds of silent corruption |
| `engine/holdout.py` | the out-of-sample test — the only one that settles anything |
| `engine/levers.py` | what you can actually change, and whether it is enough |
| `tools/analyse.py` | all of the above, on your file |

**The five ways a backtest lies**, since `backtest.py` is organised around them:
the ambiguous bar (one bar holding both stop and target — OHLC cannot order them,
so the stop is assumed and counted); mid prices; swap on the nights actually held;
gaps *through* the stop (the fill is the open, not the stop); and look-ahead.

**The seven kinds of corruption** `data.py` looks for, of which the first is worth
the module alone: frozen quotes, where a vendor fills a dead feed by repeating the
last price. Those bars are not ticks, they are the *absence* of ticks written as
if they were data — detectors read them as consolidation and backtests fill inside
them for free.

## Things it deliberately will not do

- **Report a Sharpe ratio.** At retail trade counts the standard error makes it
  meaningless, and quoting one invites the overconfidence this engine prevents.
- **Repair your data.** A loader that drops bad rows hands you a clean series with
  an unknown relationship to the market.
- **Guess an ambiguous date order.** `03/04/2024` is April in London and March in
  New York; choosing one shifts every bar by up to eleven months with no error and
  a chart that still looks like EURUSD. It refuses and asks.
- **Assume which side of the book you are on.** `--side` is required, because
  backtesting on mid hands you half the spread twice per trade.
- **Pretend a stop bounds your loss.** On 2015-01-15 EUR/CHF moved >30% with no
  quotes in between and clients owed money beyond their equity.

## The one measurement worth more than any indicator

Compare your broker's quoted swap against the published tom-next/rate
differential, **on both the long and the short side**, for a full week including
the Wednesday triple charge. It is charged in both directions and it is not
advertised. Until you have done it, every verdict here carries
`COST_NOT_MEASURED` — pass `--costs-measured` once you have.

## The honest ceiling

The realistic best case for a retail systematic FX system is a low single-digit
Sharpe contribution, most likely indistinguishable from zero, in a population
where 74–89% of accounts lose money and which is negative-sum by construction
once the spread is deducted.

So the defensible use of this is **not to find a way to win, but to find out —
cheaply, before funding an account — whether an idea can clear its own costs.**
On the evidence, most cannot, and knowing which is worth more than another
indicator.

Nothing here is trading advice, and a pass is a hypothesis that survived one
test, not a prediction.

## Tests

```bash
for t in tests/test_*.py; do python3 "$t"; done
```

196 tests. Every module is also mutation-tested: 217 deliberate defects
introduced, 217 caught. The sweeps live outside the repo because they rewrite
source files; each runs under `PYTHONDONTWRITEBYTECODE=1`, after a stale `.pyc`
once inverted one and reported mutants killed that were not.
