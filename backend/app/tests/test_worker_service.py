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


def test_worker_heartbeat_reports_gemini_backend(monkeypatch):
    """The worker is the only process that calls Gemini; its heartbeat is
    where a post-deploy check sees which backend it loaded. Vertex wins when
    both are set. The key itself never appears."""
    from app import monitoring, worker
    from app.config import settings
    calls = []
    secret = "AIza-not-a-real-key-0123456789"
    monkeypatch.setattr(settings, "enable_industry_reports", False)
    monkeypatch.setattr(monitoring, "record_run", lambda *a, **kw: calls.append(kw))
    for key, project, shown in ((secret, "", "gemini=api"), ("", "", "gemini=off"),
                                (secret, "some-project", "gemini=vertex")):
        monkeypatch.setattr(settings, "gemini_api_key", key)
        monkeypatch.setattr(settings, "vertex_project_id", project)
        worker._heartbeat()
        assert shown in calls[-1]["note"]
        assert secret not in calls[-1]["note"]


def test_worker_logs_routing_line_and_model_access(monkeypatch, caplog):
    """The worker runs the loops, the regen queue and every Gemini call, but
    only the web service printed a routing line (M1 handoff, slice A2a). It
    now logs the same `LLM routing:` summary — which also runs the
    unpriced-configured-model check on this process — and starts the
    `models.list` access check (`llm.model_access_report()`)."""
    import logging
    import threading

    import app.worker as worker
    from app.agents import llm

    reported = threading.Event()
    monkeypatch.setattr(llm, "model_access_report", lambda: reported.set() or {})
    priced: list[bool] = []
    real_unpriced = llm.unpriced_configured_models
    monkeypatch.setattr(llm, "unpriced_configured_models", lambda: priced.append(True) or real_unpriced())
    # The seed thread runs outside the scheduler proxy, so it names its own
    # origin (attribution critique #16); recorded from inside the thread
    # `worker.main()` really starts.
    seeded: list[str | None] = []
    seed_done = threading.Event()

    def _seed(*_a, **_k):
        seeded.append(llm.current_call_context()["origin"])
        seed_done.set()
        return {}

    monkeypatch.setattr("app.seed_universe.run_full_seed", _seed)
    monkeypatch.setattr("app.services.regen_worker.start_worker", lambda: True)
    monkeypatch.setattr("app.services.regen_worker.stop_worker", lambda *a, **k: None)
    caplog.set_level(logging.INFO, logger="app.worker")

    worker._shutdown.set()
    try:
        assert worker.main() == 0
    finally:
        worker._shutdown.clear()

    lines = [r.getMessage() for r in caplog.records
             if r.name == "app.worker" and r.getMessage().startswith("LLM routing: ")]
    assert len(lines) == 1
    assert "tier.research=" in lines[0] and "attribution=" in lines[0] and "gemini.news=" in lines[0]
    assert priced, "the price-row check did not run on the worker"
    assert reported.wait(5), "model_access_report was not started"
    assert seed_done.wait(10), "the seed thread did not reach run_full_seed"
    assert seeded == ["worker:seed"]
