# Stored outcome audit

`GET /api/admin/outcome-audit` requires the configured admin bearer token and
returns all persisted `memo_outcomes`, ordered by ID. It is read-only: one
outer-joined SELECT, no provider calls, no table creation, no reevaluation, and
no outcome, reflection, or KPI changes. It has no date filter or row cap.
Missing snapshots remain in the report instead of disappearing through an
inner join. The query selects snapshot metadata without loading memo JSON.

The response includes `total_rows`, `returned_rows`, `excluded_rows` (zero),
`truncated` (false), `counts`, `reason_counts`, `by_horizon`, `predicate`,
`limitations`, and `rows`. Each row retains the persisted outcome fields and
adds snapshot metadata, elapsed age, threshold, baseline-date evidence,
`late_evaluation_candidate`, `status`, and `reasons`.

The late-evaluation predicate is a strict comparison:

```
evaluated_at - generated_at > (horizon_days + 30) * 7.0 / 5.0 days
```

The standard thresholds are 84, 168, 294, and 553 days for horizons 30, 90,
180, and 365. Naive stored timestamps are interpreted as UTC; aware values
are converted to UTC. Comparisons retain exact elapsed microseconds. A row
exactly at the threshold is not flagged; one microsecond beyond it is.

`candidate` means the age threshold was crossed or a recorded baseline lies
more than seven calendar days from the memo. `indeterminate` identifies missing
snapshots, invalid chronology/metadata, ticker mismatches, and backtest rows
when no candidate evidence exists. `not_flagged` means neither rule fired;
**it does not mean verified or clean**. Candidate rows may also contain
indeterminate metadata, so inspect the complete `reasons` list.

The 7/5 approximation assumes a complete weekday price series. A deterministic
weekday replay checks that it catches baseline drift beyond seven days in that
model and exhibits false positives. A separate truncated-response example
demonstrates a false negative: an old 30-day evaluation performed at age 45
days can use a baseline over a week late while remaining below the 84-day
threshold. Holidays, stale caches, provider behavior, and missing bars also
limit inference. Historical responses and provider identity were not stored,
so this audit cannot reconstruct or certify legacy returns from age alone.

The scorer requests memo age plus seven calendar days, rounded up to one of
120/260/400/800. The cache key includes that rung, so all due horizons of one
snapshot share a key, while differently aged snapshots can still occupy all
four keys for a ticker. The 800-day maximum is an operational request budget,
not proof that older market data does not exist. Skipped pair identifiers are
reported in the loop note and logs. New outcomes record ticker baseline and
target dates, requested window, and (when alpha exists) benchmark dates and
symbol. These identify the chosen observations, not their economic validity.
Benchmark returns require valid closes on those exact ticker observation
dates. Missing either close preserves the directional outcome but leaves
alpha/benchmark return unavailable, with the required dates recorded in the
note. Non-finite and nonpositive prices cannot become scored observations.

Provider verification found that the actual price chain is FMP → Tiingo →
Polygon. FMP sends `limit=days`; Polygon requests a date interval and retains
the last `days` bars. The Tiingo adapter previously omitted dates and therefore
requested the latest-price endpoint; it now sends explicit `startDate` and
`endDate`, with calendar slack before retaining the requested bars. This
distinction is documented by [Tiingo's EOD API](https://www.tiingo.com/documentation/end-of-day).
The public [FMP endpoint documentation](https://site.financialmodelingprep.com/developer/docs/stable/historical-price-eod-full)
confirms the EOD endpoint but did not expose a usable `limit` contract during
review; actual response dates remain the scorer's coverage check. All adapter
regressions use mocked HTTP under the normal network guard.

No automatic repair follows this report. Preserve the report and investigate
candidate rows against trustworthy historical prices before proposing any
owner-approved repair. Headline accuracy still uses the existing absolute
directional-return criterion; changing that product metric is a separate
decision.
