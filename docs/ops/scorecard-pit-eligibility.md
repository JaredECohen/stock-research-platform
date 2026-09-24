# Strict financial period eligibility

Both `scorecard_pit.snapshot_as_of` and the scoring execution path through
`scorecard_features.pit_snapshot` use the shared pure date predicate in
`finance/pit_eligibility.py`. A row needs a known period end and availability
date, both on or before the requested date, and availability cannot precede
the period end. Missing dates are never inferred from fiscal labels by readers.

The September 13 final coverage read exposed LULU rows 35607 and 35609 with
period end February 1, 2026 and recorded availability April 18, 2025. Those
stored fields remain untouched. The reader excludes them as
`available_before_period_end`, including after their period has ended.
`period_end_after_as_of` and `available_after_as_of` identify ordinary future
observations separately; one row can have multiple reasons.

The FMP-primary re-pull (FIX-006, 2026-09-24) preserves this exclusion: FMP's
own rows whose availability precedes their period end are flagged
(`primary_availability_precedes_period_end`), never moved, re-dated or
re-inserted under a new availability, so 35607/35609 stay excluded. The
upsert that let an INSERT-only availability fall before a moved period end now
refuses any `period_end` change on a row that already has `available_at`.
When FMP and another provider both hold a fiscal year's line, the FMP row wins
the duplicate before the latest-availability rule; that changed `inputs_hash`
once for tickers where a secondary row used to win.

The public snapshot returns counts and uncapped `excluded_rows`, retaining
row ID, ticker, fiscal label, statement, line, value, currency, source and
date provenance. The execution stream appends optional row metadata to the
existing eight-field feature tuple. The engine retains a sorted audit in
`PitSnapshot.excluded_rows`, persisted through existing
`feature_raw._context.pit_exclusions`. Run notes include aggregate counts.
Rows supplied without metadata retain explicit null IDs/source fields.

Exclusion evidence participates in the normal input fingerprint so a changed
excluded row set cannot silently reuse an older audit. The patch neither
schedules score runs nor rewrites historical scores, memos, outcomes or
financial rows. Formulas and the scorecard specification are unchanged.
Eligibility prevents the demonstrated temporal leak; it does not turn
latest-restated provider values into an as-originally-reported revision ledger.
