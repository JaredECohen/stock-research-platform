# Research framework implementation — existing data stack, no FactSet

**Date:** 7 September 2026. **Scope:** implementing the data/provenance portion of the validation plan, plus selected primary-source checks. This is not a completed six-company model audit or a demonstrated investment edge.

## Decision

FactSet is not part of the required stack. The user already reported subscriptions to Alpha Vantage, FMP, Massive/Polygon and ThetaData. The working priority is public primary sources, then existing licensed feeds where authenticated access is demonstrated. No subscription purchase, account connection or repository change was made.

## 1. Provider roles and limits

| Role | Source | Current evidence and limitation |
|---|---|---|
| Primary financial and contractual facts | SEC and issuer investor relations | Public document retrieval used. Standard XBRL facts do not replace custom segment disclosures. [S05] |
| Current estimates and revision fields | Alpha Vantage | Documentation covers annual/quarterly EPS, revenue, analyst counts and revision history. User account access not tested. [S01] |
| Estimate/fundamental cross-check | FMP | Official analyst-estimates endpoint verified. Coverage and historical-vintage properties remain untested. [S02] |
| Prices and corporate actions | Massive/Polygon | Split-adjusted bars are not dividend-adjusted total return. [S03] |
| Detailed market data where necessary | ThetaData | Local Terminal and product entitlements required. Its generated EOD report must not be silently called the exchange official close. [S04] |

The Alpha Vantage public demo request returned an Information response rather than data. This proves only that the public demo did not supply estimates; it does not establish that the user's plan can or cannot access the endpoint. [S08]

## 2. Implemented controls

A manual single-request collector supports documented Alpha Vantage and FMP endpoints. It reads credentials from environment variables, does not print or persist known secrets, checks for HTTP-200 application errors, and writes unique timestamped captures with hashes. Successful JSON retrieval remains `schema_not_yet_validated`, not approved consensus.

The comparison gate rejects mismatched or unknown fiscal periods, metrics, currency, units, accounting definitions, share basis and business perimeters. Historical eligibility rejects a current snapshot used before collection. A separate verified historical record needs provenance evidence, not merely an old period-end date. Price-return controls reject split-adjusted-only series as total return.

The collector is code delivered for a configured environment; authenticated requests were not executed here. It is not an agent, scheduler, integration install, canonical vendor parser or complete economic model. The 22 passing unit tests check specific control behaviors, not coverage, forecast skill or stock returns.

## 3. Primary research continued without FactSet

### NSSC: contractual versus observed recurrence

The 10-K says communication services are month-to-month and customers can cancel at any time, without a refund. That makes observed retention more important than any assumption of long-term contractual lock-in. Contract terms were verified; churn and lifetime account economics were not. [S06]

**Framework change:** record contract term, cancellation rights and observed retention separately for every recurring-revenue thesis. High gross margins do not fill a retention-data gap.

### MOD: reported earnings growth needs a tax bridge

For the quarter ended June 30, the issuer's reported values reconcile as follows, USD millions: [S07]

| Component | 2026 | 2025 | Contribution to net-income change |
|---|---:|---:|---:|
| Pretax earnings | 68.6 | 65.7 | +2.9 |
| Tax benefit/(provision) | +5.7 | -14.0 | +19.7 |
| Noncontrolling deduction | -0.4 | -0.5 | +0.1 |
| Net earnings attributable to Modine | 73.9 | 51.2 | +22.7 |

Our arithmetic attributes roughly 87% of the reported net-income increase to the tax-line change. This is an accounting bridge, not a causal tax analysis or sustainable earnings forecast. The release identifies stock-compensation tax effects and anticipated offsets within the fiscal year. No quarterly benefit should be automatically annualized. [S07]

**Framework change:** before labeling EPS acceleration an operating inflection, reconcile pretax income, taxes, noncontrolling interests and share count. The retained-business margin/capital model is still outstanding.

## 4. What proceeds, and what stays gated

Primary-source economics, quarterly forecasting, normalized cash-flow models and reverse valuation do not require FactSet. A claim that our EPS exceeds consensus requires a same-basis estimate comparison. A retrospective consensus test requires authenticated historic vintages. When the latter are unavailable, use prospective snapshots and appropriately labeled price-implied analysis; never manufacture a historical dataset.

Snapshots create a prospective evidence trail only when actually captured. No future captures occur automatically. Provider-reported recent revision fields remain useful context, but are not a substitute for arbitrary-date historical snapshots unless their temporal semantics are independently verified.

## 5. Integration status

`Updated_Project_Instructions.md`, `Updated_Validation_Plan.md`, and `Updated_Task_Backlog.json` incorporate these changes. Original files are preserved. The source/provider registers, evidence records, accounting bridge and unit-test output make this update auditable. Existing price targets and scenarios were not refreshed or re-certified.

NSSC/MOD independent rebuilds, six-company capital reconciliations, industry comparison studies, provider account tests and historical/prospective outcome evaluations remain open. No autonomous research was enabled.

## Sources

- **S01 — Alpha Vantage official API documentation**. https://www.alphavantage.co/documentation/
- **S02 — FMP Financial Estimates API documentation**. https://site.financialmodelingprep.com/developer/docs/stable/financial-estimates
- **S03 — Massive price adjustment methodology**. https://massive.com/knowledge-base/article/is-massives-stock-data-adjusted-for-splits-or-dividends
- **S04 — ThetaData v3 stock EOD documentation**. https://docs.thetadata.us/operations/stock_history_eod.html
- **S05 — SEC EDGAR API documentation**. https://www.sec.gov/search-filings/edgar-application-programming-interfaces
- **S06 — NAPCO FY2026 Form 10-K**. https://www.sec.gov/Archives/edgar/data/69633/000110465926100081/nssc-20260630x10k.htm
- **S07 — Modine Q1 FY2027 earnings release**. https://investors.modine.com/news/news-details/2026/Modine-Reports-First-Quarter-Fiscal-2027-Results/default.aspx
- **S08 — Alpha Vantage public demo probe**. https://www.alphavantage.co/query?function=EARNINGS_ESTIMATES&symbol=IBM&apikey=demo
