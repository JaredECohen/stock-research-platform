# Regeneration ownership during worker overlap

A claimed regeneration job has a unique per-attempt owner token and a 120-second
lease. Its active execution renews the lease every 20 seconds. The existing
worker poll recovers expired owned attempts, so recovery still happens when a
new worker initially sees a healthy predecessor and then waits for its shutdown.
There is no additional scheduled monitoring loop.

The captured claim travels through the synchronous graph using a ContextVar.
Nested LLM contexts preserve it; normal Python context copying preserves it for
context-aware child tasks. The current regeneration graph has no independent
thread executor. A future executor must copy the context explicitly.

Ownership is checked before checkpoint work and paid model dispatch. Checkpoint
saves, progress and completion acquire a conditional database write lock using
that original token and an unexpired lease. Cancellation bypasses ordinary
agent/provider fallbacks. A provider request already in flight cannot be undone;
after lease loss, later guarded dispatch and publication are refused.

Snapshot publication validates and locks ownership in the same transaction as
inserting the snapshot and recording its exact version on the job. This is the
publication receipt. If the process dies before normal completion, recovery
finalizes that published job without running the graph again. Stale success,
failure and charge-release paths cannot modify a replacement attempt. Caller-
owned sessions retain commit and rollback authority.

## First rollout from an older worker

Both web and worker synchronously initialize the database and execute the mapped
job projection before starting loops or accepting requests. Missing ownership
columns fail startup even if generic reconciliation only logged its failure.
Provider seeding remains separate from this DB-only readiness check.

An old binary cannot honor a new fence. Let old-code running memo jobs drain
before initiating the rolling deployment. After replacement, verify the
predecessor's shutdown and check for remaining legacy unowned rows. Render
initiates predecessor shutdown during replacement; no stop-service step is
required. Legacy running rows without
ownership metadata are **not** assumed dead, adopted, requeued or failed.
Startup logs name every such job, ticker, run ID and start time; the recovery
result also returns the full `legacy_deferred` list. Resolve any remaining row
only after explicit evidence of its prior process's death and a reviewed action.
No broad recovery endpoint is introduced.

The observed incident was automatic filing-event META job 9, run
`e118eef6-01e7-49c3-861e-e65e18d6f9ff`: old worker `gk46d` began at
2026-09-13 05:06:40 UTC; replacement `pkbcm` requeued/claimed it at 05:07:27
while the predecessor still completed a provider call at 05:07:48. The predecessor
stopped at 05:08:22. The replacement eventually published version 2 at 05:13:58;
no duplicate final publication was established. This automatic event was separate
from the single authorized manual META job 7.

No historical memo or outcome is rewritten, no memo is triggered by deployment,
and no model, prompt, rating, valuation formula or provider budget is changed.
