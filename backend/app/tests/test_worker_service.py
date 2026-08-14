"""Tests for the `app.worker` background-service entrypoint.

Nothing else imports this module, so without these tests a syntax or
wiring error in it ships as a silently dead worker: the Render service
would boot, crash, and the only symptom would be memos never
regenerating and nightly loops never running.
"""
from __future__ import annotations


def test_worker_module_imports_without_starting_anything():
    import app.worker as worker
    assert callable(worker.main)
    assert not worker._shutdown.is_set()


def test_monitoring_registers_every_loop():
    """The worker's whole job is running these. `register_all` must not
    need apscheduler at import time, or the worker can't even wire them."""
    from app.monitoring import register_all

    class FakeScheduler:
        def __init__(self):
            self.jobs = []

        def add_job(self, fn, trigger, **kw):
            self.jobs.append(kw.get("id") or getattr(fn, "__module__", "?"))

    sched = FakeScheduler()
    register_all(sched)
    assert len(sched.jobs) == 15
    assert "edgar_poller" in sched.jobs
    assert "history_backfill" in sched.jobs


def test_worker_main_exits_promptly_when_shutdown_is_already_set(monkeypatch):
    """Render escalates SIGTERM to SIGKILL in ~30s.

    The first version of this worker ran the universe seed inline, which
    takes minutes against live providers, so a deploy's SIGTERM was not
    acted on until the seed returned and the process got killed instead of
    exiting. The seed now runs on a daemon thread; this pins that `main()`
    reaches its shutdown check without waiting for it.
    """
    import time
    import app.worker as worker

    def _slow_seed(*a, **k):
        time.sleep(30)  # stands in for a live-provider universe sweep
        return {}

    monkeypatch.setattr("app.seed_universe.run_full_seed", _slow_seed)
    monkeypatch.setattr("app.services.regen_worker.start_worker", lambda: True)
    monkeypatch.setattr("app.services.regen_worker.stop_worker", lambda *a, **k: None)

    worker._shutdown.set()
    try:
        started = time.monotonic()
        assert worker.main() == 0
        elapsed = time.monotonic() - started
    finally:
        worker._shutdown.clear()

    assert elapsed < 10, f"main() blocked on the seed for {elapsed:.1f}s"
