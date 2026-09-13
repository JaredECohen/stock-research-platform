# Industry job ownership and rolling deployment

The industry report queue uses immutable per-attempt ownership. A claim writes a
random owner token and a 120-second lease to `industry_report_jobs`; a keeper
renews that exact unexpired token every 20 seconds while work runs. A transient
renewal error is logged and retried. An expired lease cannot be resurrected.
Startup and the existing idle polling cadence recover only expired owned jobs,
retaining the existing attempt limit, retry backoff and final deterministic
attempt. No scheduler or report-generation budget changes are involved.

The captured token follows synchronous report execution through a ContextVar.
LLM and embedding dispatch, progress, analytics persistence, report publication,
cross-industry snapshot publication and job completion check ownership. A lost
claim raises the existing cancellation signal, which bypasses ordinary provider
fallback. A provider request dispatched before ownership was lost cannot be
recalled; its response cannot authorize subsequent work or publication.

Publication locks the owned job and writes the output plus its exact receipt
(`report_id` or `snapshot_id`) in one database transaction. Retrying a live
receipt returns that committed output; recovery after expiry completes its job
without recomputing. Failed commits roll back both the receipt and the output,
including changes to latest-good flags and existing cross-snapshot payloads.
Analytics writes are also fenced by the captured job identity. Original memo-job
lease behavior is unchanged.

Both web and worker synchronously reconcile their database schema and execute
zero-row mapped queries for both job models before accepting requests or starting
queues. Missing ownership or snapshot-receipt columns fail startup, including
when best-effort reconciliation logged a failure and returned normally.

## First deployment with legacy jobs still queued

Legacy `running` rows with missing ownership are **deferred**, never assumed dead.
Startup logs every identity; subsequent heartbeat recovery also names any legacy
claims made by a predecessor during rolling overlap. Ordinary queue/status reads
include `ownership_tracked`, `lease_expires_at` and `snapshot_id`, without exposing
the owner token. Queued jobs can be claimed normally by the new worker.

When natural drain is impractical, perform the following bounded cutover:

1. Deploy the ownership-aware runtime. Its default recovery leaves unknown
   legacy running rows untouched while predecessors may still be executing.
2. Verify through deployment/process records that **all** predecessor workers
   have retired. Preserve instance IDs and shutdown timestamps as the operator's
   retirement evidence. A deployment becoming healthy alone is insufficient.
3. Read the remaining legacy job metadata. The authenticated operator operation
   accepts 1–50 explicit expected rows, each with `id`, `run_id`, `attempts`,
   `started_at` and `heartbeat_at`, plus a nonempty retirement attestation.
4. `industry_legacy_recovery.recover_legacy_jobs` checks every expected value,
   `status=running`, and both ownership fields being NULL under a conditional
   database lock. Changed, owned or absent rows are rejected individually. It
   never claims or executes a job, contacts a provider, or resets an attempt count.
5. A group job can be completed only when exactly one consistent report linked
   by `IndustryReport.job_id` proves publication. Multiple or inconsistent
   reports reject recovery. A same-period cross snapshot does not establish
   which legacy job wrote it, so an existing same-period snapshot rejects
   legacy recovery with every candidate snapshot ID. Otherwise the unchanged attempt cap/backoff determines requeue or failure.

Every successful disposition appends the exact expected before-image, action,
verified report ID when present, timestamp and redacted retirement evidence to
job progress in the same transaction. The whole bounded batch commits once;
an unexpected failure rolls back all dispositions. All requested IDs and all ambiguous report
IDs are returned; dispositions are logged. Repeating the request cannot mutate
an already recovered job. This operator attestation is explicit authorization
based on external retirement evidence, not automated proof of process death.
Existing reports, snapshots, memo versions and outcome rows are not repaired or
deleted by cutover.

Tests use fresh SQLite databases and mocked clients with the outbound netguard,
including a second-process overlap check, expiry takeover, stale write attempts,
publication/receipt rollback, live receipt resumption, unchanged retry policy,
legacy exact-field rejection and synchronous schema failure. No live report,
memo, outcome evaluation or provider test is needed for these checks.
