# What the evidence actually says about trading FX — and what to build because of it

This is the research brief the system design rests on. It was produced **before**
any code, because the request was to find the references first and design second.

**How it was made.** Seven independent research agents, one per angle, each told to
report contradicting evidence with the same energy as supporting evidence, to label
source quality honestly, and to answer whether each effect survives *retail* costs.
95 claims. Then the **positive** claims — the ones a design would rest on — were
handed to separate agents whose only instruction was to refute them.

**Method limitation, stated up front.** This session's egress proxy blocked
`WebFetch` for every domain. Agents could search and read titles, abstracts and
snippets, and in some cases full Federal Reserve working-paper versions, but mostly
could not open published articles. Every claim in `EVIDENCE.md` and every verdict in
`VERIFICATION.md` carries its own verification label. Abstract-level claims are
second-hand until checked against the article. **Four of the six planned
verifications did not run** (order flow, 12-month trend, round-number clustering,
PPP value) because the session hit its usage limit; those claims are marked
UNVERIFIED below and must not be treated as established.

---

## 1. The premise

The request was to find *"the chart pattern or trading plan with the highest win
rate"* and build from there. The research answers that directly, and the answer is
that the target does not exist and would not help if it did.

**There is no academic basis for ranking FX chart patterns by win rate.** The
peer-reviewed FX technical-trading literature does not report win rates, payoff
ratios, or expectancy at all — it reports mean returns, standard deviations, Sharpe
ratios, skewness and kurtosis. The win-rate tables retail traders cite come
overwhelmingly from Bulkowski's *Encyclopedia of Chart Patterns*, which is **US
stocks only**, applies **no transaction costs**, specifies **no stop**, specifies
**no exit rule**, and measures success as a move to an "ultimate high/low" that is
unknowable at trade time.

**Win rate and profitability are empirically decoupled in FX, and the evidence is
unusually clean.** Expectancy is `E = p·W − (1−p)·L`. The break-even payoff ratio is
`W/L = (1−p)/p`, so a 90% win rate still loses money at any payoff below 0.111.

| Source | Win rate | What happened |
|---|---|---|
| FXCM / DailyFX, 43 million real trades | **61%** on EUR/USD | Still lost money: average winner 48 pips, average loser 83 pips |
| Ben-David, Birru & Prokopenya, retail FX accounts | **62.8%** of trades | Traders still lost; losers held >2× as long as winners |
| ESMA / FCA / AMF regulator disclosures | — | **74–89% of retail CFD/FX accounts lose money**; AMF found 89% lost an average of €10,900 |

Those regulator figures are net of spread, commission, swap and slippage as actually
paid by real clients. They are the correct base rate for the population, and they
have not improved in the eight years since disclosure became mandatory.

The mechanism that manufactures a high win rate in retail FX is the **disposition
effect** — closing winners fast and holding losers — which raises the win rate while
lowering expectancy. The usual "fix" (tighter targets) makes it worse, because
**cost raises the required win rate by exactly `c/(W+L)`**, which bites hardest on
precisely the small-target, high-win-rate systems being sought.

---

## 2. What did not survive

### Chart patterns

**Head-and-shoulders is the only classical chart pattern with a substantial
peer-reviewed literature testing it on FX.** Flags, pennants, triangles, wedges,
rectangles and double tops/bottoms have **no peer-reviewed FX profitability test at
all**. A system built on them is not "supported by research" in any sense — the
research does not exist.

And head-and-shoulders itself fails:

- **Lucke (2003, *Applied Economics*)** — a dedicated FX test. Clean negative:
  returns not significantly positive.
- **Chang & Osler (1999, *Economic Journal*)** — the paper usually cited *for* H&S.
  Profitable on only **two of five** dollar rates, and the rule is **dominated by a
  simple filter rule**. A complex rule beaten by a simple one on gross terms is
  beaten by more on net terms.
- **Osler (1998, NY Fed)** — H&S traders are **noise traders**: the pattern spikes
  volume but does not profitably predict direction, and its price impact mean-reverts
  within about two weeks. The move is the pattern-traders' own temporary impact.
- **Lo, Mamaysky & Wang (2000, *Journal of Finance*)** — the paper most often waved
  at as proof that patterns work — is **US equities**, makes **no profitability
  claim**, and its authors warn that detecting a pattern and trading it profitably
  are different problems.

**Candlestick patterns fail against a properly randomised null.** Marshall, Young &
Rose (2006) bootstrapped random open/high/low/close series and found candlestick
strategies indistinguishable from them. The single FX-specific positive result is in
a low-tier journal with plain z-tests, no data-snooping correction and no costs.

### Technical rules generally

They worked, and then they stopped.

- **1973–1990:** filter and moving-average rules earned genuine risk-adjusted excess
  returns on majors — confirmed by a true out-of-sample test run decades later
  (Neely, Weller & Ulrich, *JFQA* 2009). This was not data mining.
- **By the early-to-mid 1990s:** excess return on majors went to approximately zero,
  in several cases negative. This is the most robust finding in the whole field.
- **Hsu, Taylor & Wang (*JIE* 2016)** — the largest, most careful, most
  pro-technical-analysis modern study — finds **zero statistically significant
  predictive rules in developed-market currencies** since 1992–96.
- **Most recent large out-of-sample test (2022):** mean Sharpe **0.66 in-sample →
  0.06 out-of-sample** over 1999–2020, not surviving modest costs.
- Averaged across rules and currencies **without ex-post selection**, FX technical
  trading gives a Sharpe of about **0.17**. Every headline number in this literature
  is selection-conditioned.

**Intraday — the horizon most retail systems actually trade — is the cleanest
result of all:** large *gross* predictability exists and is **completely destroyed by
costs**. Break-even is ~1.01bp one-way against a realistic institutional cost of
2–2.5bp. Retail is worse.

Directly on point: among **actual retail currency traders**, using technical
indicators is **negatively associated with performance** (Abbey & Doukas, *JPM*
2012).

### Carry — REFUTED for retail, high confidence

Carry is where the textbooks say FX returns live, and it is also the textbook
high-win-rate/catastrophic-payoff trade. It fails at retail for a specific,
arithmetic reason.

A long-3 / short-3 G10 book is **200% gross notional**. The broker's financing
markup is a **debit on every leg** — it never pays you on either side — so the drag
is markup × 200% of equity, while the carry income is earned only on the **net**
yield spread.

| Broker markup | Annual drag on equity | Share of a 3%/yr gross carry spread it consumes |
|---|---|---|
| 0.8%/yr (cheapest published: tastyfx/IG) | 1.60% | **53%** |
| 1.5%/yr | 3.00% | **100%** |
| 2.0%/yr | 4.00% | **133%** |

And the gross spread has not been 3%. **DBV** — the Invesco fund running *literally
this book* — compounded 2010–2022 to **+0.80%/yr total return**, roughly **+0.1%/yr
over cash**, Sharpe ≈ 0.06. It took a ~37% drawdown from its 2007 peak, **never
recovered it**, and was **liquidated in March 2023**.

Best-case honest arithmetic: +0.85%/yr index return − 1.60%/yr markup at the
*cheapest* published fee = **−0.75%/yr**, before spread, before slippage, before the
tail. The ">5% p.a. survives bid-ask" figure is true of nothing a retail account can
hold. Hsu/Taylor/Wang (*JIMF* 2024) additionally find carry not profitable
out-of-sample after data-snooping correction, and a companion paper finds
post-publication decay explicitly attributed in part to **retail entry**.

### The dollar factor — REFUTED, high confidence

This was the most promising-looking claim ("most accessible to retail, low
turnover, survives execution costs"). All three parts fail.

**The famous "two factors explain 20–90% of exchange rate movements" is a variance
decomposition, not a prediction.** It is a contemporaneous R², and the dollar factor
*is the cross-sectional average of the same exchange-rate changes*, including the
dependent variable. Regressing a component on an average containing it mechanically
produces a high R². It is a co-movement statement with **zero** information about
forecastability. This is the identical category error as citing Evans–Lyons
contemporaneous order-flow R² as a trading signal.

**The tradable version is weak.** Buy-and-hold equal-weighted currency basket:
Sharpe **0.15 gross** over ~26 years. The timed "dollar carry" version: Sharpe 0.66,
but **in-sample, gross, on a sample ending 2010, never published out-of-sample**.
Hutchinson et al. (2022) find average out-of-sample Sharpe across currency factors
falls from **+0.39 to −0.32** post-publication, and state that "currencies no longer
respond to interest rate and real exchange rate differentials."

**And the cost argument is the wrong cost.** "Low turnover so it survives execution
costs" confuses turnover cost with *holding* cost. ~59% of the cumulative gain is the
interest differential — precisely the leg retail does not receive at par, charged
daily on full notional, which low turnover does nothing to reduce.

---

## 3. What is unverified

These were scheduled for adversarial attack and the session ran out. They are
**not** established:

- **Disaggregated customer order flow** (cited Sharpe 1.26–1.45). The researching
  agent itself flagged it UNRESOLVED — it could not determine whether the figures are
  gross or net of spreads. It also requires proprietary dealer data retail cannot buy.
- **12-month time-series trend** (SG Trend Index net Sharpe ~0.4–0.6 since 2000). The
  open question is whether the **FX sleeve alone** carries that, or whether it is a
  diversification result across 50–100 markets that collapses to 10–15 currency pairs.
  Bhardwaj/Gorton/Rouwenhorst found the average CTA's net excess return statistically
  indistinguishable from zero. Short-horizon trend has specifically decayed since
  2008, concentrated in small-tick contracts — which is what FX majors are.
- **Round-number order clustering** (Osler 2003, *JF*) — the strongest *technical*
  finding in FX, but from proprietary 1999–2000 dealer order data, never turned into a
  tested cost-inclusive strategy in 25 years. Note it may argue **against** placing
  stops at round numbers rather than for a strategy.
- **PPP value** — lowest turnover of any FX factor, but weakest in raw form, and
  possibly just short-carry in disguise.

---

## 4. The costs that decide everything

| Item | Realistic retail figure |
|---|---|
| All-in round turn, EUR/USD | **0.9–2.5 pips** (raw+commission ≈ standard once slippage is counted) |
| Swap/rollover markup | **0.5–3%/yr on notional, charged in both directions** |
| Annual friction at 250 trades/yr, 1.5 pips | **3.4% of equity at 1:1 leverage, 10.2% at 3:1** |
| Cost effect on required win rate | `+c/(W+L)` — a 5/5-pip scalp needs **65%** instead of 50% at 1.5 pips cost |

Two structural facts no backtest contains:

- **A pip is not a pip across pairs.** 1 pip = 0.909bp of notional on EURUSD at 1.10,
  0.787bp on GBPUSD at 1.27, 0.667bp on USDJPY at 150. Comparing spreads in pips is
  misleading.
- **A stop does not bound your loss.** 15 Jan 2015 (SNB de-peg): EUR/CHF moved >30%
  with no quotes in between; clients ended up owing money beyond account equity and
  institutions failed. 3 Jan 2019: yen +3% in ~30 seconds with no material news.
  EU/UK negative-balance protection now caps the loss at the account balance — it does
  not cap it at your stop.

And a theorem worth stating plainly: **no position-sizing scheme — martingale, grid,
averaging down, "recovery" — converts negative expectancy into positive.** This is
why 95%-win-rate robots exist, and then stop existing.

---

## 5. What this implies for a design

The evidence does not support building a pattern scanner. It does not support
building a retail carry book or a dollar-factor timer either — both were attacked
and both failed on **retail cost arithmetic**, not on statistics.

What the evidence *does* support is uncomfortable but actionable: **the quantities
that decide whether any FX system makes money are the ones nobody measures, and they
are measurable before risking a cent.**

Every failure mode above reduces to the same shape — an edge that exists gross and
dies net:

- Intraday predictability: real gross, dead after 1.01bp.
- Carry: real gross, dead after a 0.8%/yr markup on 200% notional.
- Dollar factor: 59% of the gain is the carry leg retail does not receive at par.
- Retail outcomes: 61% win rate, still negative, because 48 < 83.

So the first thing to build is not a signal. It is the thing that tells you whether
**your** broker and **your** rule clear **your** costs — a falsification engine that
makes a strategy prove positive expectancy net of measured costs before it is allowed
to place an order, and that refuses by name when it cannot.

Concretely, the first falsifiable test is the one every source above points at and no
retail platform performs:

> **Measure your own broker's actual swap markup.** Compare its quoted swap against
> the published tom-next/rate differential, on both the long and short side, for a
> full week including the Wednesday triple charge. If the markup exceeds the yield
> spread you intend to harvest, the strategy is arithmetically dead and no amount of
> signal work changes that.

That is five minutes of measurement that decides more than any pattern ever will.

---

## 6. The honest ceiling

The realistic best case for a retail systematic FX system is **a low single-digit
Sharpe contribution, most likely indistinguishable from zero**, in a population where
74–89% of accounts lose money and where the aggregate is negative-sum by construction
once the spread is deducted.

The most defensible use of this work is therefore **not** to find a way to win, but
to find out — cheaply, before funding an account — whether a given idea can clear its
own costs. On the evidence above, most cannot, and knowing which is worth more than
another indicator.

Sources for every claim are in `EVIDENCE.md` and `VERIFICATION.md`, with per-claim
quality labels and verification levels.
