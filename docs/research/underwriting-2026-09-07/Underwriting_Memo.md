# Investment underwriting update — 7 September 2026
## Six-company, price-aware research: NSSC, MOD, POWL, LMAT, APPF and WINA

**Decision:** No full purchase gate is cleared. NAPCO and Modine remain the strongest conditional investigations. The other names need a better price, stronger operating evidence or substantially greater growth durability than the base assumptions. This is an assessment of research candidates, not a recommendation about the user's actual holdings.

This package supplements the prior Research Atlas without silently rewriting it. It contains source-linked facts, explicit analyst assumptions, normalized cash-flow models, public estimate comparisons, reverse valuations, transaction/unit-economics tests, a two-year payoff model, and a decision log. No automated research or trading has been activated.

## 1. Prices and conditional values

| Company | Quote | Bear DCF | Base DCF | Bull DCF | Base at 12% enterprise rate |
| --- | --- | --- | --- | --- | --- |
| NSSC | $37.01 | $18.10 | $32.24 | $44.83 | $25.51 |
| MOD | $194.66 | $47.40 | $152.33 | $256.02 | $124.52 |
| POWL | $181.17 | $59.99 | $113.27 | $179.81 | $94.36 |
| LMAT | $80.67 | $37.58 | $57.24 | $73.40 | $45.75 |
| APPF | $214.28 | $80.56 | $136.80 | $206.99 | $105.47 |
| WINA | $318.17 | $122.00 | $166.98 | $206.64 | $128.48 |

Prices are dedicated-finance-tool quote snapshots reflecting the 4 September US trading session; some trade timestamps are after the regular close. They are not asserted to be certified official closing prices. Financial balances generally come from June-quarter statements. Known claims and distributions are adjusted explicitly; subsequent operating cash, repurchases and complete September capitalization are not asserted to be known.

The DCF uses ten forward economic periods. Annual anchors differ: some use current annual guidance, some current annual results, and Modine uses an annualized quarter for retained segment starting scale. These are not precise quarterly/fiscal-stub forecasts. A final trade decision requires dated cash flows, matched fiscal periods and a capital-structure roll-forward.

The base enterprise discount rate is 10% for LMAT, WINA, NSSC and APPF; 10.5% for MOD and POWL. Terminal growth is 2.5% for WINA and 3% otherwise. Within a company those assumptions are held constant across business scenarios, so a bull outcome is not manufactured by also lowering the discount rate. Different discount rates and terminal growth are tested separately.

**These are scenarios, not confidence intervals or broker targets.** No calibrated probabilities or expected returns are claimed. A 12% enterprise discount rate is not automatically a 12% shareholder IRR or an automatic purchase limit. Longer growth duration or a lower required return can support higher values than these base cases.

## 2. Owner-economic cash-flow convention

The operating model is:
**FCFF = operating income after recurring share compensation × (1 − normalized tax) + depreciation/amortization − capital expenditure − incremental operating working capital − specified cash expenses.**

Capitalized software is investment. Historical CFO diagnostics subtract stock compensation as a replacement-compensation convention, not because it was historically paid in cash. Forward operating margins already include recurring compensation, so the forecast does not subtract it twice. Operating lease expense remains in operating costs; it is not also treated as funded debt.

Acquisition growth is not free: no future acquisition benefit is assumed without its purchase cost. No buyback yield is added after separately assuming fewer shares.

Terminal growth also requires investment:
**Terminal FCFF = terminal NOPAT × (1 − terminal growth / terminal incremental return on capital).**

| Company | Y1 revenue | Y1 EBIT | Y1 FCFF | Y5 FCFF | Cash/claims bridge | Terminal share |
| --- | --- | --- | --- | --- | --- | --- |
| NSSC | 217.1 | 69.4 | 54.5 | 86.1 | 101.2 | 53.2% |
| MOD | 3,145.0 | 423.5 | 101.9 | 595.9 | 553.5 | 60.5% |
| POWL | 1,500.0 | 315.0 | 183.9 | 321.8 | 230.8 | 53.1% |
| LMAT | 303.9 | 85.1 | 53.8 | 90.8 | 178.0 | 54.2% |
| APPF | 1,301.5 | 266.8 | 189.0 | 372.3 | 171.7 | 55.8% |
| WINA | 94.3 | 57.5 | 41.0 | 55.4 | -42.8 | 50.6% |

All values in this table are USD millions. WINA revenue is the economic franchise/royalty core, excluding offsetting fee programs. MOD includes retained data-center/HVAC operations, not Performance Technologies. Between approximately 50% and 61% of the base operating valuation depends on terminal value. That sensitivity is why the output should not be treated as accurate to cents.

## 3. What published estimates already expect

| Company | First fiscal EPS | Next fiscal EPS | Third fiscal EPS | Qualification |
| --- | --- | --- | --- | --- |
| LMAT | FY2026 Dec: $2.87 | FY2027: $3.18 | FY2028: $3.46 | Quote subpanel stale; actual model quote from finance tool. Alternate provider FY26 $2.88. |
| WINA | FY2026 Dec: $11.37 | FY2027: $12.49 | Not used | One analyst: reference estimate, not a broad consensus. New pass-through fees distort revenue comparisons. |
| NSSC | FY2027 June: $1.65 | FY2028: $1.83 | Not used | Alternate source FY27 EPS $1.62 and revenue $219.1m. Normalize litigation, tariff refunds and taxes. |
| APPF | FY2026 Dec: $6.90 | FY2027: $8.47 | FY2028: $10.90 | Adjusted EPS is not after-SBC owner cash. Reconcile fiscal period, adjustments and shares. |
| MOD | FY2027 Mar: $7.51 | FY2028: $10.93 | FY2029: $15.51 | Separation perimeter unresolved; retained-only DCF cannot be compared directly with consolidated EPS. |
| POWL | FY2026 Sep: $5.38 | FY2027: $6.88 | FY2028: $8.25 | Split-adjusted estimate; June-to-June backlog is not the Oct-to-Sep fiscal year. Alternate FY26 estimate $5.37. |

Sources: the six company-specific MarketWatch estimate pages, retrieved 7 September. Their quote subpanels were not used. Public estimates are reference observations, not a licensed point-in-time consensus tape; accounting adjustments and update conventions vary. WINA has only one analyst, which cannot be treated as a broad observable market belief.

A second public provider in the prior research showed NSSC FY2027 EPS $1.62, LMAT FY2026 $2.88 and POWL FY2026 $5.37. These differences are preserved rather than averaged into artificial precision.

Powell's FY2027 reference increased from $6.58 a month earlier to $6.88. Modine's FY2028 reference fell from $11.18 to $10.93 but still implies substantial growth. Therefore “orders and earnings will rise” is not, by itself, a contrarian thesis. [POWL_MW; MOD_MW]

The workbook tests two annual EPS equivalents. NSSC's base Year 1 EBIT, assumed interest income of $3.486m, an 18% tax rate and 35.891m claims produce approximately $1.66—near the $1.65 reference. Powell's base $315m EBIT, assumed $10m interest income, 24% tax and 36.604m claims produce approximately $6.75—below $6.88. These are reasonableness checks, not fiscal-matched earnings-surprise forecasts. The observation is that no large positive earnings gap has yet been demonstrated.

## 4. NAPCO: the closest case, not yet an obvious mispricing

### What was resolved
The filing records $45.638m of operating income, a $16m litigation charge and the associated unpaid liability. Applying the issuer's approximate 50-basis-point annual net tariff effect to $202.316m of revenue gives about $1.012m of non-recurring benefit. Normalized historical EBIT is therefore approximately **$60.6m**, not simply the headline amount. The quarterly 600-basis-point benefit must not be extrapolated across the full year. [NSSC_K; NSSC_R]

Removing a non-recurring expense from future earnings does not erase its cash obligation. The model reserves the $16m settlement from reported liquidity. It also reserves the June dividend payable that was paid July 3; it does not subtract the future October dividend. [NSSC_K; NSSC_Q3]

### What the model requires
Exact reported revenue anchors are $97.528m recurring services and $104.788m equipment. The base path grows services 13%, 12%, 11%, 10% and 9%; equipment 2–3%. Service gross margin stays 90.3%, equipment margins normalize around 29–30%, and operating costs rise 5.5% annually. This yields approximately $54.5m Year 1 owner-economic FCFF, reaching $86.1m in Year 5.

The price-equivalent reverse test requires roughly **13.2% service growth every year for five years**, with other base inputs fixed. That is possible, but the price already requires persistence rather than merely one more good quarter.

### Still unresolved / decision
Gross activations, disconnects, account pricing, carrier costs and dealer incentives are not adequately disclosed in the retrieved materials. Growth does not reveal gross churn. Base DCF is about $32 versus the $37 quote. Continue conditional diligence; do not claim the market has ignored the recurring model when annual EPS is near published estimates.

## 5. Modine: value the correct company and charge for its growth

### What was resolved
Current data-center revenue is $348.6m at a 20.2% gross margin, versus a prior comparison containing a favorable warranty settlement. Restoring the entire historical margin gap is not a clean forecast of recoverable costs. Every 100 basis points on that quarter's revenue is $3.486m of gross profit before operating cost, investment and tax. [MOD_R]

The announced transaction changes the valuation perimeter. The retained model contains data centers and HVAC only. It adds the proposed $210m corporate proceeds and the indicated approximately 21m THRM shares once, while excluding PT operating cash flow. At the verified $40.70 THRM quote the stock distribution has an indicative value of $854.7m—not guaranteed cash. [MOD_DEAL]

Reported cash/debt, a $50m operating reserve, NCI proxy and $20m closing/stub reserve produce a base net equity bridge of $553.5m. Those reserves and closing assumptions are explicit research choices. The customer advance is not added again on top of reported cash.

### What the model requires
The base DC revenue growth path is 43%, 28%, 23%, 18%, 15%; DC EBIT margin expands to 22%. HVAC grows more slowly. Capital spending starts at 6.5% of revenue and fades to 4.5%, versus 3% D&A, and working-capital investment remains positive. Year 1 FCFF is only about $102m despite $423m EBIT; Year 5 FCFF reaches about $596m.

That is already a demanding operating outcome. Base value is about $152 including conditional transferred value; the bullish operating case is about $256. All three operating cases assume closing. The separate no-close SOTP is a rough alternative, not a complete transaction-failure forecast or a probability estimate.

### Still unresolved / decision
The decisive question is not whether cooling demand is strong. It is whether retained margins and cash returns exceed what current valuation and comparable estimates already require. Obtain a quarterly recovery-cost bridge and a reconciled post-deal fiscal forecast before claiming an EPS edge. Track customer bargaining, competing capacity and financing-backed order conversion.

## 6. Powell: large backlog is real; the expectations gap remains unproven

### What was resolved
About $1.3bn of June backlog is expected to become revenue over the next twelve months. The model makes the additional revenue assumption explicit rather than applying one multiple to total backlog. [POWL_Q]

Reported cash and investments are about $633.6m, but customer advances finance obligations. Reserving $350.36m of net contract liabilities, $50m operating cash and the current $2.428m contingent acquisition claim leaves approximately $230.8m of equity-bridge cash. The reserve is conservative and editable, not a claim that contract liabilities are debt. Future FCFF does not charge a second full unwind of that same reserve. The $2.344m comparison-period contingent amount was rejected in favor of the current column. [POWL_Q]

### What the model requires
Year 1 revenue is $1.3bn scheduled backlog plus an assumed $200m service/new-order component: $1.5bn at 21% operating margin. Cash capital expenditure is increased above the recent light historical run rate. The resulting owner FCFF is about $184m in Year 1 and $322m in Year 5.

Public FY2027 EPS is already about $6.88. Our annual earnings equivalent is not above that reference, and the fiscal calendar is not identical to the backlog period. Base DCF is approximately $113 versus $181 quoted.

### Still unresolved / decision
Continue as a high-quality watch, not a demonstrated expectations-arbitrage investment. Upgrade only if the fiscal delivery/margin bridge supports materially better earnings and cash than the already-rising reference, or the price changes enough. Project slippage, fixed-price costs and a reversal of advance funding are the major cash risks.

## 7. LeMaitre: the best original quality score did not survive every gate

### What was resolved
The convertible must be treated as alternative debt/equity claims. The model deducts $172.5m face debt and uses a 23.088m diluted-claim proxy excluding 1.443m if-converted shares, then checks the alternative fully converted endpoint. It does not subtract debt while simultaneously using the full conversion denominator. This is not option pricing of every outstanding award. [LMAT_Q]

More importantly, the FDA issued an Artegraft-related warning letter in August 2025. The latest 10-Q describes continued remediation and additional observations after the June 2026 inspection. This is not proof of a future shutdown; it is a material unresolved risk omitted by a simple margin/growth screen. [LMAT_FDA; LMAT_Q]

### Model and decision
The base model still gives substantial credit: growth begins at 10%, margin rises toward 31%, and only a stated limited additional cash reserve is charged. The base value is roughly $57 versus $81; a stronger growth case at the same discount assumptions is about $73.

The price could be justified by longer growth duration or more favorable assumptions, but that is not a margin of safety while regulatory economics remain unsettled. Regulatory evidence and cash consequences are now prerequisites to upgrading the investment, not a footnote after valuation.

## 8. Winmark: new revenue is not automatically new royalty profit

### What was resolved
The new advertising-fund and software programs create revenue but corresponding expenses. The model excludes those flows from the core royalty economics and gives the base case no automatic incremental software profit. The national fund also changes required store marketing; the $295 monthly software fee began in September. [WINA_FEES]

The official Plato's Closet summary reports 2025 average sales of $1,307,875 and merchandise gross profit of $833,084. These are sample averages, not net owner earnings. [WINA_STORE]

### A measurable franchisee test
Assume 5% royalties, 6% required marketing, a $90k owner replacement wage, $450k invested capital and a 15% pretax investment return. These assumptions leave a ceiling of **about $528k for all other operating costs** after the $3,540 annual software fee. The incremental required marketing plus software burden is about **$16.6k annually at the cited sales**, before benefits.

This establishes a useful threshold, not observed store ROIC. Payroll, rent, shrink, insurance, utilities and other actual costs remain necessary.

### Model and decision
Core growth starts around 6.5%, operating margin eventually reaches 63%, and software reinvestment does not disappear. Base DCF is around $167 versus $318. The reverse growth test requires a materially higher path under the same assumptions. Longer durability or lower required return may support more value, but historical renewals alone do not prove that path. WINA moves to a price-and-evidence watch rather than being purchased on the royalty label.

## 9. AppFolio: analyze gross profit and after-compensation cash

### What was resolved
First-half CFO was $121.902m. Subtracting physical investment, capitalized software and the recurring SBC economic adjustment leaves **$80.660m**, before a fuller tax/working-capital normalization. The $221.685m liquid balance is separate from the $87.668m private investment portfolio, to which the base bridge assigns zero credit. [APPF_Q]

Subscription and value-added service growth are not economically interchangeable. Managed units, payments, other services and fee monetization need separate gross-profit and retention analysis.

### Model and decision
The base model grows revenue 16%, 15%, 14%, 12% and 10%, after-SBC EBIT margin reaching 24.5%. Owner FCFF rises from $189m to $372m over five years. Even with that expansion, base DCF is approximately $137 versus $214.

The price-equivalent growth test requires about 21.3% annual revenue growth for five years with other assumptions fixed. Better margins or longer growth duration could substitute, but neither is established by adjusted EPS alone. This remains a price watch until the service-level economics justify a stronger case.

## 10. Reverse valuation is a test, not a mind-reading exercise

| Company | Driver varied | Constant growth needed | Other assumptions |
| --- | --- | --- | --- |
| NSSC | Recurring-service revenue | 13.2% | Base margins, costs, reinvestment, discount rate and cash bridge held fixed |
| MOD | Data-center revenue | 29.0% | Base margins, costs, reinvestment, discount rate and cash bridge held fixed |
| POWL | Total revenue after Year 1 | 21.7% | Base margins, costs, reinvestment, discount rate and cash bridge held fixed |
| LMAT | Total revenue | 15.8% | Base margins, costs, reinvestment, discount rate and cash bridge held fixed |
| APPF | Total revenue | 21.3% | Base margins, costs, reinvestment, discount rate and cash bridge held fixed |
| WINA | Core royalty/franchise revenue | 17.4% | Base margins, costs, reinvestment, discount rate and cash bridge held fixed |

Growth is varied over Years 1–5 except POWL, where Year 1 is fixed and Years 2–5 change. Thereafter growth fades toward the terminal assumption. Other combinations of margins, reinvestment, durability and discount rate can produce the same price. The workbook implements a formula-driven growth grid, not a claim that investors literally forecast these percentages.

## 11. Two-year payoff is a different underwriting question

| Company | Bear path at 18x exit | Base at 14x | Base at 16x | Base at 18x | Bull at 18x | Exit EV/EBIT for 12% hurdle |
| --- | --- | --- | --- | --- | --- | --- |
| NSSC | -2.1% | 5.3% | 11.6% | 17.6% | 29.9% | 16.1x |
| MOD | -21.2% | 4.6% | 11.4% | 17.7% | 39.3% | 16.2x |
| POWL | -13.3% | -0.2% | 6.1% | 12.0% | 33.0% | 18.0x |
| LMAT | -8.2% | -1.1% | 4.7% | 10.2% | 18.8% | 18.7x |
| APPF | -17.0% | -11.3% | -5.7% | -0.4% | 11.2% | 23.1x |
| WINA | -6.8% | -9.8% | -3.7% | 2.0% | 8.4% | 21.7x |

The percentages are annualized conditional returns, not probabilities or expected returns. This illustration distributes all owner cash after assumed interest annually, holds share count/cash-claim bridge constant, ignores personal taxes/costs and sells at the stated multiple of Year 3 EBIT. MOD assumes transaction closing; distribution timing is simplified.

NSSC and MOD can produce approximately 17.6–17.7% annual returns on the base path at 18x exit EBIT. But an exit at 14x reduces those base-path returns to about 5%; the MOD bear operating path at the same 18x produces approximately -21% annually. The market can price the next quarter favorably without making a long-duration DCF cheap. Both the operating forecast and the future buyer's price require evidence.

A 12% two-year hurdle requires an exit multiple around 16.1x for NSSC, 16.2x for MOD and 18.0x for POWL under their base paths. Those are useful price/earnings milestones, not promised resale multiples.

## 12. Process changes now applied

**Match expectations before claiming an edge.** Preserve fiscal calendars, perimeter, adjusted definitions, taxes and shares. A strong year-over-year result is not a forecast beat.

**Classify cash.** Operating liquidity, advance-funded working capital, settlement claims, contingent acquisition liabilities, paid dividends and private assets are not fungible surplus.

**Count claims and transactions once.** Convertibles, stock distributions, proceeds and disposed earnings need a coherent bridge.

**Value economic growth.** Fee pass-through and capital-funded expansion must be separated from incremental owner profit. SBC remains a cost.

**Bound unknowns.** Franchise expenses and churn stay unknown; use break-even costs and reverse valuations rather than inventing customer economics.

**Separate investment and trading cases.** Terminal value and two-year exit pricing carry different risks. Show both without quietly converting one thesis into the other.

**Retain source discrepancies and chronology.** No mechanical averaging of different vendors. The dated cash/claims and normalized fiscal forecasts must be refreshed before action.

### Research sequence after this update
The highest-value tasks are the NSSC account-economics bridge and MOD retained margin/capital bridge. Powell needs fiscal earnings above, not merely consistent with, already-higher public estimates. LMAT needs regulatory progress. WINA needs actual owner-adjusted store returns. APPF needs recurring gross-profit and retention by economic service.

This is a finite research snapshot. It does not imply continued unscheduled background work.

## Sources
- **LMAT_Q:** https://www.sec.gov/Archives/edgar/data/1158895/000119312526333762/ck0001158895-20260630.htm
- **LMAT_R:** https://ir.lemaitre.com/news-releases/news-release-details/lemaitre-q2-2026-financial-results
- **LMAT_FDA:** https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/lemaitre-vascular-inc-713502-08112025
- **WINA_Q:** https://www.sec.gov/Archives/edgar/data/908315/000090831526000028/wina-20260627x10q.htm
- **WINA_K:** https://www.sec.gov/Archives/edgar/data/908315/000090831526000007/wina-20251227x10k.htm
- **WINA_FEES:** https://www.sec.gov/Archives/edgar/data/908315/000090831526000012/wina-20260316x8k.htm
- **WINA_STORE:** https://www.winmarkfranchises.com/platos-closet/the-investment/how-much-can-i-make-/
- **NSSC_K:** https://www.sec.gov/Archives/edgar/data/69633/000110465926100081/nssc-20260630x10k.htm
- **NSSC_R:** https://investor.napcosecurity.com/2026-08-24-NAPCO-Security-Technologies,-Inc-Reports-Fiscal-Q4-and-Full-Year-2026-Results
- **NSSC_Q3:** https://investor.napcosecurity.com/2026-05-04-NAPCO-Security-Technologies%2C-Inc-Reports-Fiscal-2026-Q3-Results
- **APPF_Q:** https://www.sec.gov/Archives/edgar/data/1433195/000143319526000058/appf-20260630.htm
- **APPF_R:** https://ir.appfolioinc.com/news-releases/news-release-details/appfolio-inc-announces-second-quarter-2026-financial-results
- **MOD_Q:** https://www.sec.gov/Archives/edgar/data/67347/000110465926088569/mod-20260630x10q.htm
- **MOD_R:** https://investors.modine.com/news/news-details/2026/Modine-Reports-First-Quarter-Fiscal-2027-Results/default.aspx
- **MOD_DEAL:** https://ir.gentherm.com/news-releases/news-release-details/gentherm-and-modines-performance-technologies-business-combine
- **MOD_LTA:** https://investors.modine.com/news/news-details/2026/Modine-Announces-Landmark-4-Billion-Long-Term-Capacity-Agreement-through-2029-with-Strategic-Data-Center-Customer-for-Airedale-by-Modine-Cooling-Solutions/default.aspx
- **POWL_Q:** https://www.sec.gov/Archives/edgar/data/80420/000008042026000107/powl-20260630.htm
- **POWL_R:** https://powellindustriesinc.gcs-web.com/news-releases/news-release-details/powell-industries-announces-third-quarter-fiscal-2026-results
- **LMAT_MW:** https://www.marketwatch.com/investing/stock/lmat/analystestimates
- **WINA_MW:** https://www.marketwatch.com/investing/stock/wina/analystestimates
- **NSSC_MW:** https://www.marketwatch.com/investing/stock/nssc/analystestimates
- **APPF_MW:** https://www.marketwatch.com/investing/stock/appf/analystestimates
- **MOD_MW:** https://www.marketwatch.com/investing/stock/mod/analystestimates
- **POWL_MW:** https://www.marketwatch.com/investing/stock/powl/analystestimates
