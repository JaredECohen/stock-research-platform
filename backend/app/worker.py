"""Background worker entrypoint — memo regen queue + monitoring loops.

Run with `python -m app.worker`.

Why this exists (2026-08-12): the web service was doing three jobs in one
container — serve HTTP, drain the `regen_jobs` queue, and run the 15
APScheduler monitoring loops. Their memory profiles are nothing alike. A
request is a few MB; a memo regen is the process's largest allocator
(26+ LLM round-trips, filing bodies, chunk embeddings); the nightly
sweeps walk the whole S&P 500 universe and overlap each other around
03:00–06:00 UTC. Sizing one instance for the union means the web service
pays for the worker's peak all day, and a regen spike takes user-facing
traffic down with it — which is exactly what the Render OOM-kill did.

Split apart:
  - **web** (`ENABLE_REGEN_WORKER=false`, `ENABLE_MONITORING=false`)
    serves requests and enqueues jobs. Its peak becomes predictable, so
    it can be sized small.
  - **worker** (this module, both flags true) owns execution and gets
    the RAM the sweeps actually need.

The `regen_jobs` queue is already durable and DB-backed, and
`enable_regen_worker` was already documented as the flag for "an
API-only replica that enqueues but never executes", so no coordination
work is needed: the two processes communicate entirely through Postgres.

Safe to run as a single instance only — `claim_next_job` uses a
row-level claim, but the monitoring loops are not written to be
multi-instance safe (they'd duplicate provider spend). Keep the worker
service at one replica.
"""
from __future__ import annotations

import logging
import signal
import sys
import threading

from .log_setup import configure_logging

# Quiets httpx's per-request logging and scrubs query-string credentials
# out of every record — provider keys were reaching Render's logs in
# plaintext. See `app/log_setup.py`.
configure_logging()
log = logging.getLogger("app.worker")

_shutdown = threading.Event()


def _handle_signal(signum, _frame) -> None:
    log.info("worker received signal %s — shutting down", signum)
    _shutdown.set()


def _heartbeat() -> None:
    """Persist a liveness marker readable by the web service.

    Uses the same `record_run` path as the monitoring loops, so it shows
    up in `/api/admin/cron-health` alongside them and inherits the same
    never-raises guarantee. `rss` in the note makes the worker's memory
    curve visible through the API too, not only in Render's log viewer.

    FEAT-003: the Industry Analysis drainer gets its own row on the same
    tick. It is scheduled by nothing — a thread that sits idle six days a
    week and drains on Sundays — so without a heartbeat of its own the
    only evidence it is alive arrives once a week, and a drainer that died
    on Monday would look healthy until the reports silently failed to
    appear. Its row carries queue depth, not just liveness.
    """
    try:
        import os

        from .config import settings
        from .monitoring import record_run
        from .services import memory_probe
        rss = memory_probe.rss_mb()
        ready = settings.llm_enabled
        mode = "live" if ready else "demo"
        note = (
            f"llm={settings.active_llm_provider}; generation_mode={mode}; "
            f"embeddings={'openai' if settings.has_openai else 'hash'}; "
            f"build={os.environ.get('RENDER_GIT_COMMIT', 'unknown')}"
        )
        if rss is not None:
            note = f"rss={rss:.0f}MB; " + note
        record_run(
            "worker_heartbeat",
            success=settings.app_env.lower() != "production" or ready,
            note=note,
        )
    except Exception:  # pragma: no cover — liveness must not kill the worker
        log.warning("worker heartbeat failed", exc_info=True)
    try:
        from .config import settings
        if settings.enable_industry_reports:
            from .monitoring import record_run
            from .services import industry_report_worker
            # The drainer writes its own row on its own tick, so this
            # process speaks for it only when the thread is NOT there —
            # a dead drainer must not be masked by a healthy main loop.
            if not industry_report_worker.is_running():
                record_run(
                    industry_report_worker.HEARTBEAT_NAME, success=False,
                    note="drainer thread not running (ENABLE_INDUSTRY_REPORTS=true) — "
                         "no weekly Industry Analysis report will be generated",
                )
    except Exception:  # pragma: no cover — liveness must not kill the worker
        log.warning("industry report drainer heartbeat failed", exc_info=True)


def main() -> int:
    from .config import settings
    from .database import bootstrap_runtime_schema
    from .services import memory_probe

    # Render sends SIGTERM on deploy/scale-down; handling it lets an
    # in-flight memo finish instead of being killed mid-write. Python only
    # allows handler registration on the main thread, and `main()` is also
    # called from a thread in tests — degrade to "no graceful shutdown"
    # rather than dying on import-order trivia.
    try:
        signal.signal(signal.SIGTERM, _handle_signal)
        signal.signal(signal.SIGINT, _handle_signal)
    except ValueError:
        log.warning(
            "worker not on the main thread — signal handlers not installed; "
            "shutdown will not be graceful"
        )

    memory_probe.log_rss("worker_boot")

    # Schema migration must finish before the asynchronous seed, scheduler or
    # queue can query mapped columns. Do not continue with a missing job fence.
    # This is DB-only; provider seeding remains on the daemon thread below.
    bootstrap_runtime_schema()

    # Idempotent — the web service runs the same seed on its boot. Doing
    # it here too means the worker doesn't depend on web having started
    # first (Render gives no ordering guarantee between services), and
    # the monitoring loops need a populated `companies` table.
    #
    # Run on a daemon thread, NOT inline. The seed does a live-provider
    # profile lookup per universe member, so it takes minutes against the
    # S&P 500 and is uninterruptible. Inline, it owned the main thread
    # through the whole of boot: a deploy's SIGTERM was recorded but not
    # acted on until the seed returned, and Render escalates to SIGKILL
    # about 30s later — so every deploy killed the worker mid-seed.
    # Daemon-threading it lets main() reach the shutdown-aware loop
    # immediately, and the process can exit without waiting for it.
    #
    # Schema readiness is synchronous above. Universe enrichment remains
    # asynchronous; jobs and loops may see the existing stored universe while
    # enrichment is in progress.
    def _seed() -> None:
        try:
            from .seed_universe import run_full_seed
            log.info("worker seed: %s", run_full_seed())
        except Exception as exc:
            log.warning("worker seed failed (continuing): %s", exc)

        # Catch the pgvector column up with the JSON embeddings, then build
        # the HNSW index. This has to happen *somewhere* that can block for
        # minutes and has production credentials, and this thread is the
        # only such place: init_db() would fail the deploy's health check,
        # and the inline sync in upsert_source is capped at ~1000 rows so it
        # can't stall a memo run — which means the historical corpus would
        # otherwise never converge without someone remembering to run
        # scripts/backfill_pgvector by hand.
        #
        # Safe on every boot: idempotent, resumable, and a no-op costing one
        # COUNT once the corpus is populated. No-ops entirely off Postgres.
        try:
            from .services import vector_store
            result = vector_store.backfill_and_index()
            if not result["skipped"]:
                log.info("worker pgvector backfill: %s", result)
        except Exception as exc:
            log.warning("worker pgvector backfill failed (continuing): %s", exc)

        # Phase 6: register the in-code scorecard methodology so the web
        # process can serve `/api/scorecard/spec` from the registry (it
        # falls back to the in-code spec until then). Queue recovery is
        # deliberately NOT here — `scorecard_loop.run_once` recovers at
        # the start of every tick, so a stale `running` row can never be
        # observed before recovery has had its turn.
        try:
            from .services import scorecard_service
            log.info("worker scorecard registry: %s", scorecard_service.ensure_version_registered())
        except Exception as exc:
            log.warning("worker scorecard registry failed (continuing): %s", type(exc).__name__)

    threading.Thread(target=_seed, name="worker-seed", daemon=True).start()

    scheduler = None
    if settings.enable_monitoring:
        try:
            from apscheduler.schedulers.background import BackgroundScheduler  # type: ignore

            from .monitoring import register_all
            scheduler = BackgroundScheduler(daemon=True)
            register_all(scheduler)
            scheduler.start()
            log.info("worker monitoring scheduler started")
        except Exception as exc:
            log.warning("worker monitoring failed to start: %s", exc)
    else:
        log.info("worker monitoring disabled (ENABLE_MONITORING=false)")

    # `start_worker` no-ops when ENABLE_REGEN_WORKER=false. If someone
    # deploys the worker service with the flag off, say so loudly rather
    # than idling silently while jobs pile up in the queue.
    from .services.regen_worker import start_worker, stop_worker
    if _shutdown.is_set():
        log.info("shutdown requested during scheduler startup — exiting")
        if scheduler is not None:
            try:
                scheduler.shutdown(wait=False)
            except Exception:  # pragma: no cover
                pass
        return 0
    if not start_worker():
        log.error(
            "regen worker did not start (ENABLE_REGEN_WORKER=%s) — this "
            "process has nothing to do; queued memos will not be executed",
            settings.enable_regen_worker,
        )

    # FEAT-003 — the Industry Analysis drainer. A second durable queue with
    # its own thread rather than work inside the weekly loop: report
    # generation is minutes of stats + LLM per group, and running it on the
    # scheduler's thread would block every other loop behind it. Gated on
    # ENABLE_INDUSTRY_REPORTS, which render.yaml sets true on this service
    # only — a page view on web must never generate a report.
    from .services.industry_report_worker import start_worker as start_industry_worker
    from .services.industry_report_worker import stop_worker as stop_industry_worker
    if settings.enable_industry_reports:
        if start_industry_worker():
            log.info("industry report drainer started")
        else:
            log.error(
                "industry report drainer did not start with "
                "ENABLE_INDUSTRY_REPORTS=true — weekly reports will be "
                "enqueued and never drained",
            )
    else:
        log.info("industry reports disabled (ENABLE_INDUSTRY_REPORTS=false)")

    # FIX-006 — the one-time FMP-primary re-pull. Not a monitoring loop: a
    # daemon thread that dry-runs every company, records the exact plan in a
    # `financial_data_repairs` ledger, and executes only after the owner
    # authorizes that reviewed plan (see `services/fmp_repull_ledger.py`).
    # It no-ops without a configured FMP history provider.
    if settings.enable_monitoring:
        try:
            from .services.fmp_repull_ledger import start_thread as start_repull_thread
            start_repull_thread(_shutdown)
        except Exception as exc:
            log.warning("fundamentals repull thread failed to start: %s", type(exc).__name__)

    # Heartbeat immediately, then every 5 minutes. Without this, nothing
    # outside the container can tell a healthy worker from a crash-looping
    # one until a loop happens to fire — and the earliest, edgar_poller, is
    # 30 minutes out. The row is what makes `/api/admin/cron-health`,
    # served by the *web* service, able to answer "is the worker alive
    # right now?" without anyone opening the Render dashboard.
    _heartbeat()

    log.info("worker ready; waiting for shutdown signal")
    while not _shutdown.is_set():
        # The regen worker and APScheduler both run on their own threads;
        # this loop only keeps the process alive and periodically reports
        # RSS so the worker's own memory curve is visible in Render logs.
        if _shutdown.wait(timeout=300):
            break
        memory_probe.log_rss("worker_heartbeat")
        _heartbeat()

    stop_worker()
    stop_industry_worker()
    if scheduler is not None:
        try:
            scheduler.shutdown(wait=False)
        except Exception:  # pragma: no cover
            pass
    log.info("worker stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
