# Setup

From nothing to a verdict on your own data. Every command below is run from the
`fx-engine/` directory and every one of them was executed while writing this
page.

---

## 1. What you need

**Python 3.8 or newer. Nothing else.**

No `pip install`, no virtualenv, no account, no API key. The engine imports only
`argparse`, `collections`, `csv`, `dataclasses`, `datetime`, `html`,
`importlib`, `io`, `math`, `os`, `random`, `statistics` and `sys` — all standard
library. 3.8 is the floor because `statistics.NormalDist` arrived there; it is
tested on 3.11.

```bash
python3 --version          # 3.8.0 or higher
```

If that prints 3.7 or lower, or `python3: command not found`, install Python
from python.org and come back. Nothing else on this page will work first.

## 2. Get it and check it

```bash
git clone https://github.com/Panus15/Panus15.github.io.git
cd Panus15.github.io/fx-engine
```

**Run the tests before trusting anything.** They take about twenty seconds and
they are the only evidence that the copy you have works:

```bash
for t in tests/test_*.py; do python3 "$t" | tail -1; done
```

You should see fourteen lines, each `N/N passed`, totalling 330. Any `FAIL` line
means stop — do not use a copy whose tests fail, and open an issue instead.

Optionally, check the engine against deliberate sabotage. This rewrites source
files in place and puts them back, so it refuses to run if you have uncommitted
changes:

```bash
python3 mutation/run.py            # 366 defects introduced, 366 should be caught
```

## 3. See it work before you give it your data

```bash
python3 demo.py
```

This runs the whole pipeline on **seeded random walks** — data with no structure
in it by construction — and reports what it concludes. The point is to watch it
refuse: if it passed data with nothing in it, nothing it says about your data
would mean anything. It also runs a positive control on a genuinely trending
series, because a filter that refuses everything is indistinguishable from one
that is stuck.

Then ask it how often it is wrong:

```bash
python3 tools/calibrate.py --trials 25
```

Six seconds. It runs the complete pipeline over 25 random series and 25 trending
series and prints its own false-positive rate and its power, next to the
published research. **Read the upper-bound caveat on the power figure** — the
trend it is tested against is about three times stronger than any EURUSD has
ever had.

## 4. Get your price data

The engine reads a CSV of OHLC bars. Minimum viable file:

```
Date,Open,High,Low,Close
2023-01-02 00:00:00,1.07020,1.07065,1.06995,1.07044
2023-01-02 01:00:00,1.07044,1.07090,1.07031,1.07077
```

**Where to get it.** Your broker's platform almost certainly exports CSV —
MetaTrader does it from the History Center, cTrader and TradingView both have an
export. That is the best source, because it is the data from the venue you would
actually trade on. Free historical FX bars are also published by several
providers; the loader handles the common vendor layouts (headerless
`time,O,H,L,C`, semicolon or tab separators, split date and time columns).

**If you exported from MetaTrader 5, it works as-is and it gives you more than
the minimum.** The export looks like this — tab-separated, angle-bracketed
column names, a dotted year-first date, and the date and the clock in two
separate columns:

```
<DATE>	<TIME>	<OPEN>	<HIGH>	<LOW>	<CLOSE>	<TICKVOL>	<VOL>	<SPREAD>
2025.11.11	04:07:00	154.312	154.339	154.311	154.336	32	0	3
```

Pass it directly; no `columns=` and no `--time-format` are needed. Two things
about it are worth knowing:

- **MetaTrader bars are BID prices**, so `--side bid`. The chart is drawn on the
  bid and the `<SPREAD>` column is how you recover the ask.
- **`<SPREAD>` is recorded per bar, in POINTS** — not pips. On a 3-decimal JPY
  quote a point is 0.001 and a pip is 0.01, so a value of `3` is 0.3 pips. The
  loader converts it for you; see 5c, which is the reason this column matters.

**Three things about the file matter more than its length.**

- **Which side of the book.** Bid, ask, or mid. The numbers cannot tell you and
  the engine will not guess, so `--side` is required. If you do not know, ask
  your data provider; if you cannot find out, use `--side mid` and read the
  warning, because a study on mid collects half the spread twice per trade.
- **How much of it.** No amount of careful arithmetic rescues a short file.
  A few thousand bars will mostly produce `TOO_FEW_TRADES`. Aim for at least a
  year of hourly data (~6,000 bars) and preferably several.
- **Whether the timestamps are unambiguous.** `03/04/2024` is April in London
  and March in New York. If your file is slash-dated and never has a day past
  the twelfth, the loader refuses and asks for `--time-format` rather than
  silently shifting every bar by up to eleven months.

**Check the file before analysing it.** This loads it, reports what is wrong,
and stops:

```bash
python3 -c "
import sys; sys.path.insert(0, '.')
from engine.data import load_csv
print(load_csv('YOUR.csv', 'EURUSD', side='bid').report())"
```

Read the output. `FROZEN_QUOTES` past 5% of the series means the file is largely
a dead feed written as a flat market and the engine will refuse to analyse it.

## 5. Measure your broker's costs

**This is the step that decides more than any indicator, and it is the one
nobody does.** Until you have done it every verdict carries
`COST_NOT_MEASURED`, by design.

Two numbers are needed.

### 5a. The all-in round turn, in pips

Not the advertised spread. Spread **plus** commission **plus** the slippage you
actually get.

1. Open your platform's order history. For each of your last 20 trades, record
   the price you asked for and the price you got. The difference is slippage.
2. Average it. Add the typical spread at the hours you trade — not the headline
   figure, which is quoted for the London open.
3. Add commission, converted to pips. On a raw/ECN account this is most of your
   cost: a $7 per lot round turn on EURUSD is about 0.7 pips.

Published all-in round turns on EURUSD run from about **0.9 pips** (raw account,
near-zero spread plus commission) to about **2.5 pips** (standard account once
slippage is counted). If your figure is below 0.9 you have forgotten something.

### 5b. The swap markup, per year, as a fraction

This is the broker's margin on overnight financing. It is charged in **both**
directions — it never pays you, on either side — and it is not advertised.

1. Find your broker's swap table for your pair (usually "Contract
   Specifications" or "Symbol Information"). Record the long and short figures.
2. Find the published overnight interest-rate differential, or tom-next, for the
   same pair on the same date. Central bank rates are published; your platform
   may also show the raw tom-next.
3. **The markup is what your broker charges beyond that differential — on both
   sides.** Do it for a full week, including **Wednesday**, which is charged
   triple to cover the weekend value date. A week of readings is the minimum:
   one day tells you nothing about the average.
4. Convert to a fraction per year. If the combined long+short markup works out
   to 0.004% of notional per night, that is `0.004% × 365 ≈ 1.5%/yr`, so
   `--swap-markup 0.015`.

Published figures run from **0.5%/yr** at the cheapest end to **3%/yr** at
market-maker shops. If the markup exceeds the yield spread you intend to
harvest, a carry-like strategy is arithmetically dead and no signal work changes
that.

**When the swap does not matter at all.** Financing is charged per NIGHT, and
the engine converts your bar size into nights from the measured interval. On
M1 data that is 1,440 bars to a night, and the hold cap is 500 bars — so no
trade in a minute-bar study is ever financed, and `--swap-markup` cannot change
its verdict. Check the `financing:` line the run prints before spending a week
on 5b: if your holds never cross a rollover, skip it.

### 5c. Or let the file tell you, if it recorded the spread

A price file usually cannot contradict a cost assumption, because it holds no
costs — which is why `--costs-measured` has to be taken on trust. A MetaTrader
export is the exception: its `<SPREAD>` column is the spread at every bar, on
the very bars your edge was measured on.

The run prints it:

```
recorded:  0.30 pip spread at the 50% percentile of your own bars
           (median 0.30, 99th 4.60) — spread ALONE, before commission
```

Three things follow, and the third is the one that costs money.

- **Declare a round turn below that and the run refuses**, with
  `COSTS_BELOW_RECORDED`. A cost cheaper than the spread your own data recorded
  was not available on those bars, whatever the marketing says.
- **It can only ever refute, never confirm.** Commission and slippage are not in
  the file, so no value here clears `COST_NOT_MEASURED`. The spread is a FLOOR
  on a round turn, not a round turn.
- **Read the tail, not the median.** On the file above the median is 0.30 pips
  and the 99th percentile is 4.60 — fifteen times wider. A strategy that trades
  the open, the news minute or the rollover pays the tail. `--spread-quantile
  0.95` holds your cost model against that end instead.

## 6. Your first real run

```bash
python3 tools/analyse.py YOUR.csv \
    --pair EURUSD --side bid \
    --round-turn 1.3 --swap-markup 0.015 \
    --costs-measured \
    --report report.html
```

Replace `1.3` and `0.015` with **your** measured numbers. `--costs-measured` is
your signature that you did section 5; without it every verdict is refused with
`COST_NOT_MEASURED`.

Open `report.html` in a browser. The solid line is after costs, the dashed line
before — the gap between them is what the broker took.

**Expect to be refused.** That is the normal outcome and it is the product.

## 7. Test your own rule

A rule file is a Python module with two things in it:

```python
# my_rule.py
from models.patterns import Detection

PARAMETER_COUNT = 3        # fast window, slow window, stop distance

FAST, SLOW, STOP_PIPS = 10, 40, 30


def _sma(bars, i, n):
    return sum(b.c for b in bars[i - n + 1:i + 1]) / n


def detect(bars):
    out = []
    for i in range(SLOW, len(bars)):
        f_now, s_now = _sma(bars, i, FAST), _sma(bars, i, SLOW)
        f_prev, s_prev = _sma(bars, i - 1, FAST), _sma(bars, i - 1, SLOW)
        if f_prev <= s_prev and f_now > s_now:
            e = bars[i].c
            out.append(Detection("ma_cross", i, 1, e,
                                 e - STOP_PIPS * 0.0001,
                                 e + 2 * STOP_PIPS * 0.0001))
    return out
```

```bash
python3 tools/analyse.py YOUR.csv --pair EURUSD --side bid \
    --round-turn 1.3 --swap-markup 0.015 --costs-measured \
    --rule my_rule.py
```

**`PARAMETER_COUNT` is required and must be honest.** It is the denominator of
the multiple-testing correction. Count every knob you *could* have swept, not
the ones you left at their defaults. If you tried twenty variants and kept the
best, the number is twenty — declaring 1 is what makes `NOT_SIGNIFICANT` stop
working.

**Before it runs anything, your rule is checked for look-ahead.** It is re-run
on the series truncated at successive bars; a detection that disappears when the
data stops at its own index needed future bars to be found. This is the
commonest defect in a homemade rule and it is invisible — with the check off, a
one-bar peek measured 4,312 trades at +3.49 pips each on a random walk and came
back TRADEABLE.

## 8. The hold-out — you get one

```bash
python3 tools/analyse.py YOUR.csv --pair EURUSD --side bid \
    --round-turn 1.3 --swap-markup 0.015 --costs-measured \
    --rule my_rule.py --holdout
```

Finds on the first 70% of your file, tests the survivors on the last 30%, which
nothing has looked at. **This is the only result here worth much**, and you can
use it once: the held-back half stops being held back the moment you read the
result and adjust something. Nothing in the code can enforce that.

A pattern that passes in-sample and fails out-of-sample is the **ordinary**
outcome. On currency technical rules the published decay is Sharpe 0.66 → 0.06,
with returns not surviving modest costs (Hutchinson et al., 2022).

## 8b. Could your result be your own file's noise?

The hold-out needs enough trades on each side of the split to say anything. When
it refuses for `TOO_FEW_TRADES` you are not out of tests — you are out of the
one that needed a bigger sample. This one does not:

```bash
python3 -m tools.nullcheck YOUR.csv --pair USDJPY --side bid \
    --pattern double_top --round-turn 1.0 --swap-markup 0.006
```

It resamples **your own file's bar-to-bar moves** in contiguous blocks, runs the
same pattern over each resample, and counts how often a result as good as yours
turns up. Your file's fat tails, volatility clustering and session pattern all
survive; the relation between one block and the next does not.

**A plain random walk is not good enough and the tool says so.** Tried first
here: on a real USDJPY M1 file a constant-volatility Gaussian walk produced a
median of 13 `double_top` detections against the file's 33, and all 40 seeds
were refused by the 30-trade floor. That reads as a 0% false-positive rate and
it is worth nothing — the gap was volatility clustering, and the floor refused
every seed before merit was ever assessed.

**Read the block length, because it is the whole assumption.** Patterns span
10–120 bars and are held up to 500, so:

| block | what it keeps | what the pass rate means |
|---|---|---|
| shorter than a pattern | the return distribution | a strict null: this is a false-positive rate |
| around a pattern | the pattern, not its outcome | in between |
| longer than pattern + hold | whole patterns AND their outcomes | **not a null** — it reshuffles your sessions |

Worked example, the file section 4 shows:

```
   block        passed  refused: floor  trades (med)   best net
      30     0/30    0%         27/30              23     +10.39
     240     3/30   10%          8/30              34     +15.07
    1440     8/30   27%          6/30              36     +21.98
```

That file's `double_top` was TRADEABLE in-sample: 33 trades, 78.8% wins, +17.5
pips a trade, t = 3.86 against 2.87 required. At `block=1440` the nulls find 36
trades against its 33 — the same size of sample — and **27% of them pass too**.
One of them scored +21.98 pips a trade at t = 5.46 on reshuffled data.

The honest reading is bracketed, not a single number: at the strict end (block
30) nothing passes, at the day end 27% does, and a day-long block keeps whole
patterns and their outcomes. So what that file shows is that its result does not
depend on the ORDER of its sessions, only on which sessions were in it — which
is a claim about a handful of days, not about USDJPY. Set against
`RESTS_ON_ONE_PERIOD` reporting 95% of the profit in one month of three, those
two findings are the same finding.

**It cannot confirm an edge.** A low rate means your result is not THIS
artifact. There are others.

## 9. Reading a refusal

| code | what it means | what to do |
|---|---|---|
| `COST_NOT_MEASURED` | you accepted the default costs | do section 5, then pass `--costs-measured` |
| `NOTHING_MEASURED` | no trades at all | longer file, or the rule never fires |
| `TOO_FEW_TRADES` | the sample cannot support the claim | **more data** — not a cheaper broker. The refusal says how many trades |
| `NEGATIVE_NET_EXPECTANCY` | measured, and it loses | run the lever section it prints. If gross is also negative, the rule is the problem, not the spread |
| `NO_WIN_RATE_SAVES_IT` | cost ≥ the whole target | widen the target, or stop. No win rate fixes it |
| `NOT_SIGNIFICANT` | positive but inside the noise | the refusal says how many trades it would take. Often the honest answer is "more than you will ever have" |
| `GOVERNED_BY_ASSUMPTION` | ambiguous bars decided it | re-run on higher-resolution bars |
| `COSTS_DISAGREE` | the backtest was cheaper than the verdict | re-run the backtest with the same `CostModel` |
| `COSTS_BELOW_RECORDED` | your file recorded a wider spread than you declared | use a round turn at or above the spread in your own data — see 5c |
| `RESTS_ON_ONE_PERIOD` | drop its best stretch and the edge is gone | nothing to fix. The result describes that stretch; it is not a property of the pair |

When a rule is refused on arithmetic, the tool prints what you could change —
cost, target/stop geometry, hold time, frequency — and whether any of it is
enough. **If gross expectancy is zero or negative, none of it is**, and it says
so: a rule that loses before costs still loses at a free broker.

## 10. What it will not do

- **Predict anything.** A pass is a hypothesis that survived one test.
- **Report a Sharpe ratio.** At retail trade counts the standard error makes it
  meaningless.
- **Repair your data.** Problems are counted and located; what to do about them
  is a decision with consequences.
- **Bound your loss.** A stop is an instruction, not a guarantee. On 2015-01-15
  EUR/CHF moved more than 30% with no quotes in between and clients ended up
  owing money beyond their account equity.
- **Tell you a trade is good.** It can only tell you whether an idea clears its
  own costs on the data you have. On the evidence, most do not — and finding out
  cheaply is worth more than another indicator.
