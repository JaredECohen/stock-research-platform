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
    from app.monitoring import KNOWN_LOOPS, register_all

    class FakeScheduler:
        def __init__(self):
            self.jobs = []

        def add_job(self, fn, trigger, **kw):
            self.jobs.append(kw.get("id") or getattr(fn, "__module__", "?"))

    sched = FakeScheduler()
    register_all(sched)
    # `KNOWN_LOOPS` is what cron-health expects to see reporting, so the
    # two lists must be the same length — a literal count here went stale
    # the first time a loop was added (sample_build_loop, FEAT-002).
    assert len(sched.jobs) == len(KNOWN_LOOPS)
    assert sorted(sched.jobs) == sorted(KNOWN_LOOPS)
    assert "edgar_poller" in sched.jobs
    assert "sample_build_loop" in sched.jobs
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


def test_production_heartbeat_exposes_missing_llm_configuration(monkeypatch):
    from app import monitoring, worker
    from app.config import settings
    from app.services import memory_probe
    calls = []
    monkeypatch.setattr(settings, "app_env", "production")
    monkeypatch.setattr(settings, "openai_api_key", "")
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    monkeypatch.setattr(settings, "enable_industry_reports", False)
    monkeypatch.setattr(memory_probe, "rss_mb", lambda: 165)
    monkeypatch.setattr(monitoring, "record_run", lambda *a, **kw: calls.append(kw))
    worker._heartbeat()
    assert calls[0]["success"] is False
    assert "generation_mode=demo" in calls[0]["note"]
    assert "llm=none" in calls[0]["note"]
    assert "embeddings=hash" in calls[0]["note"]


def test_heartbeat_reports_the_routing_value_this_process_loaded(monkeypatch):
    """The worker has no HTTP port; its cron-health note is where a
    post-deploy check sees whether ENABLE_INDUSTRY_ANALYST_ROUTING reached
    this process (a Blueprint sync may not apply a new render.yaml key)."""
    from app import monitoring, worker
    from app.config import settings
    calls = []
    monkeypatch.setattr(settings, "enable_industry_reports", False)
    monkeypatch.setattr(monitoring, "record_run", lambda *a, **kw: calls.append(kw))
    for value, shown in ((True, "industry_routing=on"), (False, "industry_routing=off")):
        monkeypatch.setattr(settings, "enable_industry_analyst_routing", value)
        worker._heartbeat()
        assert shown in calls[-1]["note"]
