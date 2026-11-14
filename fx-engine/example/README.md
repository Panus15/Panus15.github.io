# A walkthrough you can run right now

`SETUP.md` section 4 asks you to export a CSV from your broker. That is the
right first step for a real answer and the wrong one for finding out what this
tool does. Everything below runs in about a minute on a file you generate here.

```bash
python3 tools/make_example.py
```

> **The CSV is a seeded random walk. It is NOT market data.** Nothing on this
> page is a finding about EUR/USD or any other pair — the file is named
> `EURUSD_H1_SYNTHETIC.csv` and carries the same warning in its own header so
> that it cannot be quoted as a result later. A random walk is the honest choice
> for a demonstration precisely because there is provably nothing in it: real
> sample data would invite reading its verdicts as conclusions.
>
> Every number below was produced by running the command above it.

---

## 1. Check the file before analysing it

```bash
python3 -c "
import sys; sys.path.insert(0, '.')
from engine.data import load_csv
print(load_csv('example/EURUSD_H1_SYNTHETIC.csv', 'EURUSD', side='bid').report())"
```

```
EURUSD_H1_SYNTHETIC.csv: 8,734 bars of EURUSD at the BID, 2023-01-02 to 2024-05-15
  interval       1:00:00
  skipped        3 comment line(s) beginning with #
  no defects found — which is not the same as none
```

It found the hourly interval on its own, skipped the header comment and **said
how many lines it skipped**. The weekend gaps are in the file and are *not*
reported as missing data: the loader separates holes that span a Saturday, when
the market is shut, from holes that do not.

## 2. The six shipped patterns

```bash
python3 tools/analyse.py example/EURUSD_H1_SYNTHETIC.csv \
    --pair EURUSD --side bid \
    --round-turn 1.3 --swap-markup 0.015 --costs-measured
```

```
pattern                   n   win%  need%  pips/trade  verdict
head_and_shoulders       63  60.3%  48.5%        8.75  TOO_FEW_TRADES
double_top              116  56.0%  57.6%       -1.16  NEGATIVE_NET_EXPECTANCY
double_bottom           112  57.1%  58.5%       -1.04  NEGATIVE_NET_EXPECTANCY
triangle                  9  44.4%  36.2%        7.47  TOO_FEW_TRADES
flag                    564   9.8%   9.2%        0.37  NOT_SIGNIFICANT
engulfing             2,206  34.2%  37.4%       -1.01  NEGATIVE_NET_EXPECTANCY
```

**All six refused, each for its own named reason.** Read the second and third
columns together: `double_top` won 56.0% of the time and needed 57.6%. The
failure is not a bad win rate — it is a win rate that loses anyway, which is
the FXCM result in miniature.

`head_and_shoulders` shows +8.75 pips a trade and is still refused, because 63
trades cannot support the claim and that pattern has a published FX rejection
to overturn, so its floor is 100 rather than 30.

Then it tells you what you could change:

```
CLOSEST TO WORKING — engulfing
net -1.01 pips/trade (gross +0.29, cost 1.31)
  [cannot] cost: break-even needs a round turn under 0.29 pips; you pay 1.31.
           The cheapest published anywhere is about 0.9, so no broker is cheap
           enough. The gross edge is smaller than the cost of trading at all.
  [CAN]    geometry: 4.44x wider would clear the cost IF the win rate held —
           and it will not. Re-run at 94/47 pips and let it answer.
```

A gross edge of **+0.29 pips** against a **1.31 pip** cost. No broker on earth
closes that gap, and the tool says so rather than leaving you to shop.

## 3. Your own rule

```bash
python3 tools/analyse.py example/EURUSD_H1_SYNTHETIC.csv \
    --pair EURUSD --side bid \
    --round-turn 1.3 --swap-markup 0.015 --costs-measured \
    --rule example/sample_rule.py
```

```
rule:      sample_rule from example/sample_rule.py, 3 declared parameter(s), no look-ahead found

search:    3 hypotheses across 1 rule(s), summing what each declares

pattern                   n   win%  need%  pips/trade  verdict
sample_rule             278  35.3%  35.1%        0.10  NOT_SIGNIFICANT
```

A plain moving-average cross. **+0.10 pips a trade** — nominally positive, and
inside the noise. The correction uses the 3 parameters the rule declares, not a
number this engine guessed.

## 4. The look-ahead check, which is the one output worth seeing

`example/peeking_rule.py` is the same shape of rule with four characters
changed: it reads `bars[i + 1]`.

```bash
python3 tools/analyse.py example/EURUSD_H1_SYNTHETIC.csv \
    --pair EURUSD --side bid \
    --round-turn 1.3 --swap-markup 0.015 --costs-measured \
    --rule example/peeking_rule.py
```

```
could not analyse: example/peeking_rule.py looks ahead: a detection at bar 8693
disappears when the series is truncated at bar 8694, so finding it needed bars
8694..8733 — which had not printed at bar 8693
```

Exit code 1, before a single number is printed, naming the bar that gave it
away.

**Now what it would have reported** with that check switched off:

```
4,312 trades · +3.32 pips/trade · win 38.8% · TRADEABLE
total +14,308 pips over 8,734 bars of a random walk
```

Fourteen thousand pips, on data with **provably nothing in it**. Nothing about
those four lines of code looks wrong, and no part of the backtest errors. This
is the defect that check exists for, and it is the commonest one in a homemade
rule.

## 5. The hold-out

```bash
python3 tools/analyse.py example/EURUSD_H1_SYNTHETIC.csv \
    --pair EURUSD --side bid \
    --round-turn 1.3 --swap-markup 0.015 --costs-measured \
    --rule example/sample_rule.py --holdout
```

```
HOLD-OUT: found on 6,113 bars, tested on 2,621 never seen (30% held back)
  sample_rule: refused in-sample
    in-sample         183 trades     +2.33 pips/trade  NOT_SIGNIFICANT

Nothing passed in-sample, so there was nothing to test. The held-back half is
still untouched — which is worth more than a result would have been.
```

Note what it did **not** say. It did not warn that you have spent your one
out-of-sample test, because nothing reached it. Telling you that you had used
something you still have would be a lie.

## 6. The engine's own accuracy

```bash
python3 tools/calibrate.py --trials 25
```

Six seconds, and the only honest answer to "how far can I trust this":

```
condition  rule-trials  in-sample  CONFIRMED  any rule
noise              150      0.7%       0.0%        0%
trend              150     17.3%      16.7%      100%
```

On noise, no series produced a confirmation. On a real trend, every one did.
Median out-of-sample ÷ in-sample net pips: **noise −0.08, trend +1.31** — and
Hutchinson et al. (2022) measured 0.09 on real currency technical rules, so the
noise row says this hold-out is at least as hard as the published test.

**The power figure is an upper bound.** The trend it is tested against moves the
price about twice over the sample, roughly three times faster than the strongest
trend EURUSD has ever had.

---

## What this walkthrough demonstrated

Six patterns refused, a hand-written rule refused, and a rule that cheats caught
before it could print a number — on data where the correct answer to everything
is no. That is the whole product: **it costs you an afternoon instead of an
account.**

When you bring your own file, the one thing that changes everything is section 5
of `SETUP.md`. The `--round-turn 1.3` and `--swap-markup 0.015` above are
placeholders. Yours decide the answer.
