# Market data backfill evidence — 2026-09-13

Data-import status: **final**. Analysis reads saved API artifacts only. The latest captured import ran on `82b2a2f52c441b8084a4e778d54321973adca9f8`; this is not a claim about the final application runtime after subsequent deployments.

Fundamentals coverage and fetch freshness pass for **172/173 companies**. Incomplete company count: **1**; awaiting captured results: **0**.

The six primary statement/cadence buckets themselves are complete and fresh for **172/173 companies**. The stricter success count also requires no unresolved blocking data issues.

Captured 376 unique sync reports for 197 tickers. Financial writes: 396,736, including 116,741 timestamp-only refreshes. Source upgrades: 43,819.

These totals count captured writes, not distinct database rows. All row-level conflicts, provider attempts and old/new source identities are retained in the companion JSON.

| Incomplete company | Missing or stale buckets | Reason categories |
| --- | --- | --- |
| ANSS | income/annual, income/quarterly, balance/annual, balance/quarterly, cash/annual, cash/quarterly | freshness_or_refresh, missing_history, stored_or_legacy_conflict |

The latest coverage API counts **177,092 stored price rows** and **277,469 stored fundamental rows across its returned targets** (companies, benchmark and legacy memo symbols). These sums include preexisting and excluded observations within those targets, but exclude reserved quarantine namespaces. They are **not verified full physical-table totals**; usable coverage is assessed separately.

Applied BK repair receipts separately record **2,041 restored rows** and **518 quarantined financial rows**. Reserved namespaces: `~Q686e51a5eebc6a`: 518 rows. Full original identities and actual after-images remain in the companion JSON. Quarantined rows are excluded from normal returned-target row totals; repair actions are separate from the sync-write totals above.

The database-only coverage read at **2026-09-13T06:37:39.005947+00:00** finds complete price coverage for **172/197 targets**, including **170/173 companies**. Company exceptions: ANSS, AVB, EA. Complete noncompany targets: SPY, AUDC.

The capture verified runtime `1f7f44751721a871579614160bb53f52bf1728db` before the coverage read and `1f7f44751721a871579614160bb53f52bf1728db` afterward. The returned Company universe contains **173 unique tickers** across 173 Company rows; duplicate identities: none.

The **22 incomplete legacy memo symbols** are: ASOFT1, AUDA, AUDB, AUDN0, AUDN1, AUDN2, AUDN3, AUDN4, CHAIN, RTRIP, TSTAA, TSTADM, TSTBB, TSTCAP, TSTHALF, TSTNEU, TSTNOMA, TSTONE, TSTONE2, TSTPATCH, TSTRFL, TSTSHORT. Their absent or incomplete series remain explicit; provider data is not invented for them.

| Company price exception | Observed provider series end | Coverage limitation |
| --- | --- | --- |
| ANSS | alpha_vantage: 2025-07-16; fmp: 2025-07-16 | Current requested range remains incomplete; see listing-event evidence and full missing-session identities in JSON. |
| AVB | alpha_vantage: 2026-08-17; fmp: 2026-08-24 | Current requested range remains incomplete; see listing-event evidence and full missing-session identities in JSON. |
| EA | fmp: 2026-08-04; alpha_vantage: 2026-08-04 | Current requested range remains incomplete; see listing-event evidence and full missing-session identities in JSON. |

AVB provider bars extend through August 17 (Alpha Vantage) and August 24 (FMP), despite the issuer-reported halt before the August 17 open. These are observed provider records, not a certification that trading continued after the halt.

Compared with the preserved **2026-09-13T05:16:07.046191+00:00** coverage capture, returned-target stored row counts changed by **+1,288 prices** and **+0 fundamental facts**. Company tickers added: none; removed: none. These are observed net storage changes between reads, not additional import-write counts or attribution to a particular background job.

Returned targets with changed stored row counts or kind: **MSFT** (stored_price_row_count: 802 → 2090). Full before/after provider series bounds and close basis remain in the comparison JSON.

Latest targeted retries succeeded for **BK, LULU, ARM, PLTR, SNOW**, completed by **2026-09-13T05:15:18.977093+00:00**. Their newest receipts and the later global read supersede the original failure status; all earlier issues remain in the audit arrays.

A separately authorized benchmark extension completed at **2026-09-13T05:04:48.527888+00:00** on `a330420c0472487203d1b9ee28b371c92a9534a2`: SPY returned and stored-readback confirmed **1,900 bars from 2019-02-21 through 2026-09-11**. The subsequent database-only LLY read returned the same 1,900-date span with SPY-observed continuity verified. SPY selected FMP provider EOD close; LLY selected Alpha Vantage raw-as-traded prices. These are separate security series and are not spliced. No new memo request or outcome evaluation was performed for this extension. Full bars, source/basis selection, timestamps and hashes are preserved in [the benchmark-extension evidence](2026-09-13-benchmark-history-extension.json). The continuity check uses observed SPY sessions, not an independently verified exchange calendar.

## Corrections and evidence

The original run and subsequent correction captures are retained separately. Counts above use the newest available per-ticker result or global coverage capture; audit arrays retain every captured unique run, so a failed original attempt is not erased by a successful correction.

| Observed cause | Correction and preserved evidence |
| --- | --- |
| Reported zero debt became NULL; reported total debt was replaced by a component sum | FMP mapping now preserves zero and prefers reported total debt, deriving only with both components known. The saved ADBE annual balance response reports FY2025 short-term debt 0 and total debt 6.648 billion versus 6.210 billion from the two components. Missing values are not assumed zero. |
| Old ingestion assigned USD without persisting the provider currency | Named-provider refresh can correct legacy currency at the same period end, with old/new source, value and currency audit. ASML raw annual statements report EUR; TSM corrections use reported TWD. Known-provider disagreements remain blocking, and no FX conversion occurs. |
| Calendar-year labels contradicted explicit fiscal-year labels for HD, LOW, JNJ, LULU and TGT | Exact provider-confirmed period-end/statement/line/cadence identities authorize transactional relabel chains. Full relabel audits preserve row IDs and availability; ambiguous destinations and known facts remain protected. Normal FMP ingestion now uses valid explicit fiscalYear to prevent recurrence. |
| MSFT/NVDA retained bare-year annual aliases without period-end dates | The final read retains 128 excluded aliases for each company, unchanged and unusable. Separate dated canonical provider facts supply coverage. Excluded alias identities remain warnings only when the corresponding valid named-provider fact exists; missing facts and known-provider contradictions stay blocking. |
| LULU's populated legacy FY2026 debt duplicates collided with confirmed FY2025 facts at the same period end | The final retry succeeds. Legacy IDs 35596 and 35598 remain stored; usable reads select the unique named-provider replacements 35607 and 35609. The full old/replacement values and identities remain in exclusion and relabel-warning audits. |
| BNY historical financial responses use inconsistent fiscal labels relative to canonical BK | Fundamentals route to BK. Exact bad-run rows are reconciled only through a durable reviewed plan, canonical-provider matches, before-image fences and full quarantine/restoration audits. The revenue disagreement does not itself prove a different issuer; economic comparability remains a separate open issue. |
| Widening legacy history requests triggered unnecessary provider fallback for ARM, PLTR and SNOW | Required coverage remains the user's requested history floor; older requested rows support reconciliation without falsely requiring prelisting history. The original cross-provider disagreements remain in the audit. |
| Legitimate optional historic NULLs were treated as whole-company failure | NULL optional lines remain stored and fully identified as warnings. Missing primary values, nonfinite values, invalid identities and missing currency on numeric facts remain blocking. |

Captured fiscal relabels: **1,116**. These overlap financial writes; do not add them to writes as distinct rows. The companion JSON records every relabel and source upgrade, including the original and resulting identities.

Listing changes are recorded in [the primary-source listing-event evidence](2026-09-13-market-data-listing-events.json). ANSS's last trading date was July 16, 2025; stale current coverage after its acquisition is a real listing-history limitation. AVB and EA likewise stopped trading in August 2026. Successor-company prices are not spliced into acquired securities. BK→BNY is separately supported as the same security by the issuer's unchanged-CUSIP notice. Acquisition exceptions do not make missing historical sessions acceptable.


Unsuccessful backfills are tracked separately from primary-line coverage:

| Company | Six primary buckets complete and fresh | Unresolved issue kinds |
| --- | --- | --- |
| ANSS | False | coverage_gap: 6, provider_partial_coverage: 2, refresh_incomplete: 1, stored_value_conflict: 200 |
| MDT | True | stored_period_end_conflict: 32 |

MDT's **32 FY2018 rows** remain an explicitly retained older-history exception: legacy period end **2018-04-30**, incoming provider period end **2018-04-27**. Its requested two-year primary coverage passes. No date was guessed or overwritten. Row IDs: 20036, 20037, 20038, 20039, 20040, 20041, 20042, 20043, 20044, 20045, 20046, 20047, 20048, 20049, 20127, 20128, 20129, 20130, 20131, 20132, 20133, 20134, 20135, 20136, 20137, 20187, 20188, 20189, 20190, 20191, 20192, 20193.

Coverage means the primary lines—revenue, total assets and operating cash flow—span the requested annual and quarterly periods and were fetched within seven days. It does not assert every optional line is present. Unavailable provider data is not treated as a confirmed entitlement denial.

**Raw storage completeness is distinct from PIT admissibility.** The final LULU read exposes canonical rows **35607 and 35609** with period end **2026-02-01** but `available_at` **2025-04-18**, source `lag_rule`. Their stored fields remain preserved, but the original readers admitted these future-period facts in an offline reproduction. The [PIT eligibility repair](../ops/scorecard-pit-eligibility.md), commit `0213b97` integrated as `2f54b59`, rejects future period ends, availability before period end, and unknown dates in both read paths; every excluded identity and reason remains visible. **It is deployed in runtime `1f7f44751721a871579614160bb53f52bf1728db`**, verified live on the worker at 05:38:44 UTC and web at 05:39:32 UTC. Both full backend suites recorded 3,583 passes and the sole known configuration-artifact failure; see the [runtime validation receipt](2026-09-13-final-runtime-validation.json). The completed import build remains unchanged; the current database-only coverage vintage and its preserved prior comparison are stated above. No live score rerun or historical result rewrite was requested. Passing the raw history coverage check does not certify PIT inputs or turn latest-restated provider values into an as-originally-reported ledger.

**BK provider comparability remains open after fiscal repair.** Canonical Q1 2026 revenue is $9.863bn, versus issuer revenue $5.409bn; subtracting $4.454bn interest expense reconciles them. Q2 canonical revenue already equals issuer revenue $5.698bn, so the series mixes definitions. Q2 provider EPS $2.43 and diluted shares 698.164m differ from issuer $2.45 and 692.223m. These comparisons support no automatic economic normalization. [Issuer release](https://www.sec.gov/Archives/edgar/data/1390777/000139077726000071/ex991_earningsreleasex2q26.htm); [exact provider/issuer evidence](2026-09-13-bk-provider-comparability.json).

Global coverage artifact: `/private/tmp/mm-handover/data-backfill-runtime-closeout/global-coverage.json`.

No provider calls, database writes, memo generation, or outcome changes are performed by this analysis.

The [post-import outcome audit](2026-09-13-market-data-after-final-import-audit.json) finds **all 710 previously captured outcomes unchanged across 14 stored fields**, with no missing or new IDs relative to the 710-row follow-up snapshot. The earlier 709-row baseline differs only by the already-observed automatic outcome 800. This verifies preservation; it does not validate the investment accuracy of those outcomes.
