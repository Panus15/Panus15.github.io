# Adversarial verification of the positive claims

The negative findings in `EVIDENCE.md` are consistent and well sourced. What needed attacking was the **positive** side — the claims a system design would actually rest on, where being wrong costs money. Each agent below was told to REFUTE, to default to refuted when evidence is thin or gross-of-costs, and to answer specifically whether a retail account can capture the effect.

**Same method limitation:** WebFetch blocked, so abstracts and search snippets only. Each verdict states its own verification level.

---

## carry — **REFUTED** (confidence: high)

### The case against

VERIFICATION LEVEL: I read abstracts, publisher pages and search snippets only. WebFetch is blocked for all domains and the egress proxy refused every curl CONNECT (403), so I could NOT open the full text of MSSS 2012, Burnside et al. 2008, Daniel-Hodrick-Lu, or the Hsu/Taylor/Wang papers. Table-level figures below marked [snippet] come from search-result summaries, not from tables I read. The DBV annual returns and all arithmetic built on them I computed myself.

The claim is a period-specific, gross-of-retail-cost, partly-EM result presented as a general retail-tradeable fact. Four independent attacks each land.

(a) RETAIL SWAP MARKUP — THE FATAL ONE. A long-3/short-3 G10 book is 100% NAV long + 100% NAV short = 200% gross notional. The broker markup is a DEBIT on every leg — it never pays you, on either side — so drag = markup x 200% of NAV, not x 100%. Computed:
  0.5%/yr markup -> 1.00%/yr drag on NAV
  0.8%/yr        -> 1.60%/yr
  2.0%/yr        -> 4.00%/yr
  3.0%/yr        -> 6.00%/yr
tastyfx/IG publish tom-next + 0.8%/yr admin fee per position; another broker quotes ~0.5%/yr; the low end of your 0.5-3% range is roughly where the cheapest regulated brokers sit, the top end is normal for market-maker shops. Now the killer: what fraction of the GROSS CARRY INCOME that eats. Carry income = (avg top-3 yield - avg bottom-3 yield) x 100% NAV. At a 3% G10 carry spread (generous for the last 15 years), a 0.8% markup takes 53% of it; a 1.5% markup takes 100%; a 2% markup takes 133%. At a 2% spread, even the 0.8% markup takes 80%. The broker is the senior claimant on the carry. This is categorically worse than the interbank bid-ask the papers model — MSSS's transaction-cost haircut is ~1.4%/yr [snippet], a one-off spread cost; the retail markup is a continuous 1.6-6%/yr accrual on notional. Note the ESMA 30:1 cap is NOT the binding constraint (200% gross needs only 6.67% margin); the markup scales with notional regardless of equity, so you cannot lever your way out of it.

(b) POST-2008 IS FLAT. DBV (Invesco DB G10 Currency Harvest) is literally the claim's book — long top-3 yield, short bottom-3, 200% gross, quarterly rebalance. Its annual total returns 2010-2022: +0.85, +0.08, +10.06, -2.79, +0.47, -8.61, +6.30, -4.36, +0.37, +4.78, -1.21, +3.08, +2.73. I compound these to +10.90% cumulative over 13 years = +0.80%/yr CAGR. That is TOTAL return including T-bill collateral income (~0.7%/yr average over that window), so EXCESS return over cash is roughly +0.1%/yr. Annual stdev 4.77%, naive Sharpe ~0.06. The academic literature agrees: the 2022 JIFMIM paper finds the efficient carry Sharpe fell from 1.08 pre-GFC to 0.25 post-GFC [snippet]. Not 0.4-0.9. The two papers cited in the claim have samples ending 2007 (Burnside) and ~2009 (MSSS) — they both stop at the crisis, which is precisely the regime break.

(c) CRASH RISK — THE DRAWDOWN NEVER RECOVERED. The DB G10 Harvest index peaked July 2007 and the strategy took a ~37% drawdown with persistent negative skew [snippet, Alpha in Academia "Carry's Zero", Aug 2026, citing a 20-year Sharpe of 0.07]. Recovery time is not measured in years — it never happened: Invesco liquidated DBV in March 2023, ~15.7 years after the peak, still below it. The flagship retail-accessible vehicle for this exact trade was closed for want of returns and assets. Daniel-Hodrick-Lu (CFR 2017) report the dollar-NEUTRAL carry trade is highly negatively skewed with considerable downside risk and insignificant abnormal returns once you strip out the dollar leg [snippet] — i.e. the part of the trade that matches a retail G10 long/short book is the part with no alpha and all the tail. August 2024 is the live rehearsal: a BOJ hike plus a weak US payroll print unwound yen-funded carry in days, Nikkei -12.4% in one session, and JPMorgan estimated three-quarters of global carry positions were removed.

(d) IT FAILS DATA-SNOOPING TESTS OUT OF SAMPLE. Hsu, Taylor & Wang, "The Out-of-Sample Performance of Carry Trades", JIMF 143 (2024): 48 countries, 1983-2015, White reality-check and stepwise (Romano-Wolf) corrections. Conclusion per the abstract: carry strategies profitable in one period are generally NOT profitable out of sample, especially after correcting for data snooping, and even allowing for learning and stop-losses; what consistency exists is concentrated in a brief 2001-2005 window. The companion paper (Hsu, Li, Taylor & Wang, J. Empirical Finance 83, 2025) tests 13 influential published carry strategies and finds profitability declined significantly AFTER publication — and explicitly attributes part of the decay to rising retail participation following public disclosure. You would be the marginal retail entrant into a strategy whose own authors document that retail entry is what killed it.

BOTTOM LINE ARITHMETIC. Best case honest inputs: post-2008 G10 carry index excess return ~+0.85%/yr (before DBV's 0.75% fund fee), retail markup drag at the CHEAPEST published admin fee (0.8%/yr) = 1.60%/yr. Net = -0.75%/yr, before spread, before slippage, before the 37% tail. At a 2%/yr markup it is -3.15%/yr. The claim's ">5% p.a. survives bid-ask" is true of nothing a retail account can hold.

### Can retail actually capture it?

No. Blunt version: you would be paying the broker for the privilege of holding the left tail.

The specific failure is that retail financing is charged on NOTIONAL and is a debit on BOTH legs, while carry income is earned on the NET yield spread. A long-3/short-3 G10 book runs 200% gross notional, so the markup hits at 2x its headline rate: 1.6%/yr at the cheapest published retail admin fee (0.8%/yr, tastyfx/IG), 4-6%/yr at typical market-maker shops. Against a post-2008 realised G10 carry excess return of roughly 0.1-0.9%/yr, that is negative before you place a single trade. Even at a fat 3%/yr carry spread, a 0.8% markup takes 53% of the gross income and a 1.5% markup takes 100% of it.

What retail additionally cannot replicate: (1) MSSS's headline numbers come from a 48-currency cross-section where the high-yield quintile is EM (TRY, BRL, IDR, INR, HUF) — retail EM pairs cost 20-60 pips round turn and many are simply unquotable, so you are forced into the G10-only subsample which is the weakest version; (2) no interbank forward access means you cannot lock the forward points, you rent them nightly at the broker's marked-up tom-next, and the broker re-prices that daily against you with no obligation to show the interbank rate; (3) many retail swap tables already show NEGATIVE swap on BOTH sides of low-differential G10 pairs, which is the markup exceeding the entire differential — check your own broker's table before believing any of this, it is the single most informative five minutes available to you.

Leverage is NOT the constraint here and I want to be fair about that: 200% gross at ESMA 30:1 needs only 6.67% margin, so the cap is not what stops you. The markup is.

The honest retail-accessible version of "FX carry" is: buy a short-duration foreign sovereign bond ETF, or hold a hedged/unhedged bond fund, and stop paying 1.6-6%/yr of notional to rent forward points. If you want the trade in FX form, you are structurally the wrong counterparty.

### If it holds at all, under what conditions

The claim survives ONLY under all of the following simultaneously, and it is not a retail trade under any of them:

1. SAMPLE ENDING BEFORE OR AT 2008. Burnside et al. (JEEA 2008) runs to ~2007; MSSS (JF 2012) to ~2009. Within those windows a ~5%/yr net-of-quoted-spread return and Sharpe 0.4-0.9 is defensible. Extend to 2010-2022 and DBV — the exact long-3/short-3 200%-gross book — compounds to +0.80%/yr total return, ~+0.1%/yr over cash, Sharpe ~0.06.

2. THE CROSS-SECTION INCLUDES EMERGING MARKETS. MSSS's quintile sort spans 48 currencies; the return is concentrated in the high-yield EM quintile. G10-only is materially weaker. Retail EM execution costs (20-60 pips round turn, frequent unquotability, gap risk) destroy what is left.

3. INTERBANK FORWARD EXECUTION AT QUOTED BID-ASK ONLY, WITH NO FINANCING MARKUP. The papers charge you a one-off spread on rebalance (~1.4%/yr per MSSS [snippet]). They do not charge you 1.6-6%/yr of continuous markup on 200% notional, because an institution rolling its own forwards does not pay one. This is the single assumption that cannot be transported to a retail account.

4. A HOLDING PERIOD LONG ENOUGH TO ABSORB A 15+ YEAR UNRECOVERED DRAWDOWN. The strategy's ~37% 2007-08 drawdown was never recovered before DBV was liquidated in March 2023. If your answer to crash risk is "I will hold through it," the historical record says that meant holding an underwater position until the fund closed.

5. IGNORING THE OUT-OF-SAMPLE / SNOOPING LITERATURE. Hsu-Taylor-Wang (JIMF 2024) find no persistent out-of-sample profitability after reality-check and stepwise correction, with consistency confined to 2001-2005; Hsu-Li-Taylor-Wang (JEmpFin 2025) find significant post-publication decay across 13 influential carry strategies.

ONE COMPONENT THAT MAY GENUINELY SURVIVE, WITH A CAVEAT: Daniel-Hodrick-Lu report that the DOLLAR-carry component (timing the aggregate dollar on the average forward discount) has a higher Sharpe, minimal skewness and no downside risk, while the dollar-NEUTRAL cross-sectional carry — the thing the claim describes and the thing retail would build — has insignificant abnormal returns and all the tail risk [snippet; I did not read the tables]. If anything here is real, it is the dollar-timing leg, not the long-3/short-3 book. That leg is one position rather than six, so it is far less exposed to the 200%-gross markup problem — but it is a directional macro bet on the dollar, not a carry harvest, and nothing in my sources establishes it holds post-2013 or net of retail costs. Do not build the six-leg book on the strength of it.

MINIMUM THRESHOLD FOR THE RETAIL BOOK TO BREAK EVEN: gross G10 carry spread must exceed 2x your all-in broker markup. At a 0.8%/yr markup you need >1.6%/yr spread just to reach zero; at 2%/yr you need >4%/yr. Measure your broker's ACTUAL markup by comparing its quoted swap on a pair against the published tom-next/rate differential, on both the long and short side, for a full week including the Wednesday triple charge. If you will not do that measurement, do not run the strategy.

### Sources

- Menkhoff, Sarno, Schmeling, Schrimpf (2012), 'Carry Trades and Global Foreign Exchange Volatility', Journal of Finance 67(2):681-718 — https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.2012.01728.x (abstract/snippet only; full text unreachable)
- Burnside, Eichenbaum, Rebelo (2008), 'Carry Trade: The Gains of Diversification', JEEA 6(2-3):581-588 — https://academic.oup.com/jeea/article-abstract/6/2-3/581/2295921 (abstract only)
- Hsu, Taylor, Wang (2024), 'The out-of-sample performance of carry trades', Journal of International Money and Finance 143 — https://www.sciencedirect.com/science/article/abs/pii/S0261560624000299 (abstract only)
- Hsu, Li, Taylor, Wang (2025), 'On the profitability of influential carry-trade strategies: Data-snooping bias and post-publication performance', Journal of Empirical Finance 83 — https://www.sciencedirect.com/science/article/abs/pii/S0927539825000623 (abstract only)
- 'Currency carry trade: The decline in performance after the 2008 Global Financial Crisis', J. Intl Financial Markets Institutions & Money 76 (2022) — https://www.sciencedirect.com/science/article/abs/pii/S1042443121001670 (abstract/snippet: Sharpe 1.08 pre-GFC -> 0.25 post-GFC)
- Daniel, Hodrick, Lu (2017), 'The Carry Trade: Risks and Drawdowns', Critical Finance Review 6:211-262 — https://business.columbia.edu/sites/default/files-efs/pubfiles/6378/Daniel.Hodrick.Lu.Carry%20Trade.Critical%20Finance%20Review.2017.pdf (abstract/snippet only)
- Invesco DB G10 Currency Harvest Fund (DBV) annual returns 2010-2022 and March 2023 liquidation — https://finance.yahoo.com/quote/DBV/ and SEC 10-K filings https://www.sec.gov/Archives/edgar/data/1354730/000095017023010813/dbv-20221231.htm
- 'Carry's Zero', Alpha in Academia (Aug 2026) — https://www.alphainacademia.com/p/carrys-zero (19.5 years of nothing, ~37% drawdown, 20-year Sharpe 0.07)
- tastyfx overnight funding methodology: tom-next + 0.8%/yr admin fee — https://www.tastyfx.com/markets/overnight-funding-rates/
- Reuters, 'Carry on trading: rate-based G10 currency bets make a comeback' (May 2026) — https://www.investing.com/news/economy-news/analysiscarry-on-trading-ratebased-g10-currency-bets-make-a-comeback-4691116
- JPMorgan estimate that three-quarters of global carry trades were unwound in Aug 2024 — https://www.business-standard.com/world-news/three-quarters-of-global-carry-trades-removed-jpmorgan-chase-co-124080801944_1.html

## dollar-factor — **REFUTED** (confidence: high)

### The case against

VERIFICATION LEVEL: search snippets and published abstracts only. WebFetch is blocked for every domain in this session, so I read no full paper texts. Every number below is from an abstract, a listing page, or a search snippet quoting a paper. Treat table-level figures as second-hand.

The claim has three parts and all three fail.

(a) THERE IS NO NET-OF-COST, OUT-OF-SAMPLE SHARPE FOR A TRADABLE DOLLAR-FACTOR STRATEGY. I could not find one. What exists:
- Unconditional dollar factor (DOL = long an equal-weighted basket of foreign currencies, short USD; the genuinely "low turnover" version): Ackermann, Pohl & Schmedders report an equally-weighted currency portfolio Sharpe of 0.15 over ~26 years, vs 0.91 for the mean-variance-efficient portfolio. 0.15 is GROSS. That is the honest number for "the dollar factor" as a buy-and-hold. It is not a strategy; it is noise with a coupon.
- Timed version (LRV "dollar carry", JFE 2014): annualized Sharpe 0.66 (0.56 for a variant). Sample 12/1983-6/2010. In-sample, gross of costs. Quantpedia's implementation of the same rule reports 0.44 unconditional and "as large as 1.37" only when a forecasting model is layered on — again in-sample and conditional.
- So the load-bearing positive claim rests on one paper, one 26.5-year sample ending 16 years ago, in-sample, gross. No OOS number was ever published. Default to refuted.

(b) THE "20-90%" IS A VARIANCE DECOMPOSITION. YES, THIS IS THE EVANS-LYONS ERROR AGAIN. Verdelhan (2018, JF 73(1):375-418) abstract: two factors "account for 20% to 90% of the daily, monthly, quarterly, and annual exchange rate MOVEMENTS." That is a contemporaneous R-squared from regressing each bilateral rate change on the same-period factors. And the dollar factor IS the cross-sectional average of those same exchange-rate changes, including the dependent variable. Regressing a component on an average that contains it mechanically produces a high R-squared — identical to regressing a stock's return on the market return and announcing 80% explanatory power. It is a co-movement statement. It carries exactly zero information about forecastability, and therefore zero tradability. The ECB working-paper version reports the same as 18-83% of monthly bilateral USD moves across 13 developed countries; the number changes, the category error does not.
Worse, Verdelhan concedes the dollar factor is not priced in the standard cross-section: "portfolios of countries sorted by interest rates do not allow for a significant estimation of the dollar risk because all portfolios load in the same way on this factor." He had to invent a bespoke cross-section sorted on time-varying rolling dollar betas to get any pricing at all — a construction with its own estimation-noise and look-ahead problems.

(c) THE TIMING SIGNAL'S OOS RECORD IS ABSENT OR NEGATIVE.
- Econometrics: LRV's headline is that average forward discount plus US IP growth "forecast up to 25% of dollar return variation at the ONE-YEAR horizon." Persistent regressor + overlapping annual returns = the textbook Stambaugh (1999) bias and Valkanov (2003) spurious-long-horizon setup, where the OLS t-stat over-rejects and diverges with horizon. The average forward discount (US rates vs a G10 basket) flips sign a handful of times in 26 years, so the effective number of independent observations is single digits, not 318 months.
- Hsu, Taylor & Wang (JIMF 143, 2024), 48 countries 1983-2015, 1,323 strategies expanded to 17,199 with learning and stop-loss rules: carry strategies profitable in one period are "generally not profitable in an ensuing out-of-sample period, especially after correcting for data-snooping."
- Hutchinson, Kyziropoulos, O'Brien, O'Reilly & Sharma (Int. Rev. Fin. Analysis, 2022), "Are carry, momentum and value still there in currencies?": average out-of-sample Sharpe across currency factor strategies falls from +0.39 to MINUS 0.32 post-publication. And specifically: "currencies no longer respond to interest rate and real exchange rate differentials." The interest-rate differential is the dollar carry signal.
- Hassan & Mano (NBER w20294 / QJE): the "dollar trade" anomaly and the forward premium puzzle are the SAME object, "driven almost exclusively by the cross-time component," while the cross-sectional carry is a separate, static thing. They "never reject the hypothesis that currencies with high interest rates are expected to depreciate rather than appreciate," i.e. none of their estimates require any systematic link between risk premia and predictable exchange-rate moves. So the dollar factor's timing premium is the forward premium puzzle wearing a hat — and the FPP is the single least stable regularity in FX, with the UIP beta widely documented as flipping sign after 2000/2008.

(d) THE COST ARGUMENT IN THE CLAIM IS THE WRONG COST. "Low turnover therefore survives execution costs" conflates turnover cost with HOLDING cost. The dollar factor's return is mostly carry, not spot. LRV's own decomposition: $100 in 12/1983 grows to $1,467 by 6/2010, of which $860 is the interest-rate component and only $607 the predictable dollar move. Roughly 59% of the cumulative gain is the interest differential — which is precisely the leg retail does not receive at par.

### Can retail actually capture it?

Mostly no, and for a reason the claim never addresses.

The turnover point is actually correct and I will not pretend otherwise. A monthly-rebalanced dollar basket whose signal flips roughly once a year pays maybe 10-25 bps/yr in retail spreads. At retail size there is no price impact at all (Filippou, Maurer, Pezzo & Taylor, JFE 159, 2024, find proportional bid-ask costs "relatively small" and price impact the binding cost — price impact is irrelevant to a 30k account). So spread is genuinely not the killer.

The killer is the swap markup, which is charged DAILY on FULL notional and which low turnover does nothing to reduce:
- The studied portfolio is ~10 legs (EUR, GBP, JPY, CHF, CAD, AUD, NZD, SEK, NOK, DKK) held continuously against USD, all on the same side. Because the legs are directionally aligned versus USD, the broker markup does NOT net out — you pay it on essentially the full gross notional, every day, forever.
- At a 0.5-3%/yr markup that is 50-300 bps/yr of drag.
- Against that: the unconditional dollar factor's gross mean is roughly 1.5-2%/yr (consistent with Sharpe ~0.15 on ~8-10% vol). A 1%/yr markup removes half to two-thirds of it. A 2%/yr markup makes it negative. The buy-and-hold dollar factor is not merely unprofitable at retail, it is structurally negative-expectancy at a typical CFD broker.
- For the timed dollar carry, gross mean is roughly 4-6%/yr on ~9-10% vol. A 1-2%/yr markup cuts the in-sample Sharpe from ~0.5-0.66 to roughly 0.25-0.40 — before any out-of-sample decay, and the cited OOS evidence says the decay is large and possibly total.
- Leverage does not rescue this. The markup scales with notional exactly as the return does, so it is a fixed haircut on the Sharpe ratio, not a sizing problem. 30:1 caps are beside the point.

Instrument availability also breaks the replication: retail brokers do not offer DKK at all in practice, and quote SEK and NOK at 15-40 pip spreads with punitive swap. The retail version is a 6-7 leg approximation, i.e. a different portfolio from the one studied.

Separately, "the most liquid instruments on earth" is only true for the EUR/JPY/GBP legs. The Scandi legs that the equal-weighted basket needs are 3-10x wider at retail.

Finally, the dollar factor at retail is not a diversifier. It is one concentrated bet on the dollar with ~8-10% annualized vol and negative skew in risk-off episodes, and it is near-perfectly collinear with the USD leg of every other position the trader holds.

THE ONE GENUINELY RETAIL-ACCESSIBLE ROUTE: CME FX futures (including micros — M6E, MJY, M6A, M6B). There the interest differential is embedded in the futures basis at near-institutional rates rather than delivered through a marked-up swap, and the round-turn cost is roughly $5-10 plus one tick. A quarterly roll on 5-6 contracts is genuinely cheap. This is the version of the trade worth building. It is NOT what "a normal retail broker account" means, and the claim as written points the reader at the wrong venue.

### If it holds at all, under what conditions

Two narrow forms survive.

1. AS A RISK STATEMENT, NOT AN ALPHA STATEMENT. Verdelhan's 20-90% co-movement result is real and useful, but its only correct implication is risk management: a retail trader holding EURUSD long, GBPUSD long and AUDUSD long does not hold three positions, they hold roughly one dollar position at 1.5-2.5x the intended size. Size for the common factor; do not double-count. This is a position-sizing correction, not a source of return, and nothing in it is tradable as alpha.

2. AS A TIMING STRATEGY, ONLY UNDER ALL FIVE OF THESE, SIMULTANEOUSLY:
   (i) executed in CME FX futures (micros are fine), NOT spot CFDs — so the carry arrives via the basis rather than a marked-up swap;
   (ii) accepted as a 5-6 currency approximation (EUR, JPY, GBP, AUD, CAD, CHF), dropping the Scandi/DKK legs the paper uses, which changes the portfolio;
   (iii) all-in holding cost demonstrably under ~50 bps/yr on notional — the trader must actually measure this on their own statements, not assume it;
   (iv) expectations set at an in-sample gross Sharpe of at most ~0.5, with an honest OOS prior near zero given Hutchinson et al.'s +0.39 to -0.32 finding and Hsu-Taylor-Wang's data-snooping result;
   (v) sized as a satellite, never a core engine, and with a pre-committed kill rule — because the signal flips so rarely that you will not accumulate enough independent observations to ever statistically reject a dead strategy within a trading lifetime. That last point is decisive: a signal with ~10 independent switches in 26 years is unfalsifiable in real time, which is the opposite of what you want under a system you intend to trust.

WHAT DOES NOT SURVIVE UNDER ANY CONDITION: the framing that the dollar factor is "the FX factor most accessible to retail" because it is low-turnover and liquid. Low turnover is irrelevant when the expected return is a carry stream and the cost is a daily holding charge. And the 20-90% figure must be removed from any document that uses it to support tradability — it is a variance decomposition, it is mechanically inflated by the factor containing the dependent variable, and citing it as evidence of an exploitable edge is the identical error to citing Evans-Lyons contemporaneous order flow R-squared.

### Sources

- https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12587 - Verdelhan (2018), 'The Share of Systematic Variation in Bilateral Exchange Rates', J. Finance 73(1):375-418. Abstract: two factors account for 20-90% of daily/monthly/quarterly/annual exchange rate MOVEMENTS. Contemporaneous variance decomposition, not predictability. Abstract only.
- https://www.ecb.europa.eu/events/pdf/conferences/130627/2.1a_A.Verdelhan_Paper.pdf - Working paper version: 18-83% of monthly bilateral USD moves across 13 developed countries; also the admission that interest-rate-sorted portfolios 'all load in the same way' on the dollar factor so it cannot be priced in the standard cross-section. Snippet only.
- http://web.mit.edu/adrienv/www/LRV_JFE_2014.pdf - Lustig, Roussanov & Verdelhan, 'Countercyclical Currency Risk Premia', JFE 111 (2014). Dollar carry Sharpe 0.66 (0.56 variant); sample 12/1983-6/2010; $100 -> $1,467 of which $860 is the interest component and $607 the predictable dollar move; AFD + US IP forecast 'up to 25%' of dollar return variation at the 1-year horizon. In-sample, gross. Snippet only.
- https://www.sciencedirect.com/science/article/abs/pii/S0261560624000299 - Hsu, Taylor & Wang, 'The out-of-sample performance of carry trades', J. Int. Money & Finance 143 (2024). 48 countries 1983-2015, 1,323 strategies (17,199 with learning/stop-loss): in-sample-profitable carry strategies 'generally not profitable' out of sample after data-snooping correction. Abstract only.
- https://www.sciencedirect.com/science/article/pii/S1057521922002058 - Hutchinson, Kyziropoulos, O'Brien, O'Reilly & Sharma, 'Are carry, momentum and value still there in currencies?', Int. Rev. Financial Analysis (2022). Average out-of-sample Sharpe falls from +0.39 to -0.32; 'currencies no longer respond to interest rate and real exchange rate differentials'. Abstract only.
- https://www.nber.org/system/files/working_papers/w20294/w20294.pdf - Hassan & Mano, 'Forward and Spot Exchange Rates in a Multi-currency World'. The 'dollar trade' anomaly and the forward premium puzzle are driven almost exclusively by the cross-time component; they never reject that high-interest currencies are expected to depreciate. Snippet only.
- https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2184336 - Ackermann, Pohl & Schmedders, 'Optimal and Naive Diversification in Currency Markets'. Equally-weighted currency portfolio Sharpe 0.15 vs 0.91 mean-variance efficient, over ~26 years. Gross. Snippet only.
- https://www.sciencedirect.com/science/article/abs/pii/S0304405X24001090 - Filippou, Maurer, Pezzo & Taylor, 'Importance of transaction costs for asset allocation in FX markets', JFE 159 (2024). Proportional bid-ask costs relatively small; volume price impact is what turns popular strategies unprofitable at size. Abstract only.
- https://quantpedia.com/strategies/dollar-carry-trade - Quantpedia's implementation spec: equal-weighted AFD over EUR, AUD, CAD, DKK, JPY, NZD, NOK, SEK, CHF, GBP vs 3m US T-bill, monthly rebalance. Reports Sharpe 0.44 unconditional, up to 1.37 with a forecasting model. In-sample. Listing snippet only.
- https://www.sciencedirect.com/science/article/abs/pii/S1042443121001670 - 'Currency carry trade: the decline in performance after the 2008 Global Financial Crisis'. Lower post-GFC returns; reverse carry profitable 2012-2016. Abstract only.
- https://www.nber.org/system/files/working_papers/w14082/w14082.pdf - Lustig, Roussanov & Verdelhan (2011), 'Common Risk Factors in Currency Markets', RFS 24(11). DOL definition; HML_FX price of risk 546bp/yr; carry spread 4.8%/yr after transaction costs. Snippet only.
- https://www.forex.com/en-uk/about-us/financial-transparency/trading-costs-charges/ and https://spreadwisefx.com/guides/how-forex-brokers-make-money - retail swap mechanics: brokers fund at tom-next, add a markup plus admin fee, book to client daily; swap income is a material broker revenue line on positions held overnight; Wednesday triple-swap. Confirms the markup is a daily charge on notional that low turnover cannot avoid.
