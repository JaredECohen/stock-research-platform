"""Who started the work: `origin`, `job_id` and `ticker` on every LLM call
(slice B8-A2a; attribution design §4.8, critique #1, #6, #11, #16).

Every loop job, request, regen job, script and worker thread that can reach
the LLM layer opens an UMBRELLA context naming the origin (and job / run /
ticker where it has one) — never an agent, because the agent named there
would be credited with every specialist call nested under it. The rows and
`llm_call` lines then answer "which loop / endpoint / job paid for this
call", which today reads "-" for every loop and route.
"""
from __future__ import annotations

import ast
import contextvars
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from app.agents import llm, llm_attribution
from app.tests import llm_fakes
from app.tests.llm_fakes import FakeClient, anthropic_response

BACKEND = Path(__file__).resolve().parents[2]


def _context() -> dict[str, Any]:
    return llm.current_call_context()


# ---------------------------------------------------------------------------
# Loops: the scheduler proxy
# ---------------------------------------------------------------------------

def test_scheduler_proxy_origin_per_job_and_known_loops(monkeypatch):
    """Every job `register_all` registers runs under `origin=loop:<id>` in a
    FRESH context (critique #11): a context variable one job sets — here the
    failover-event list — is gone when the next job starts, instead of
    accumulating on a reused scheduler thread for the life of the worker."""
    import app.monitoring as monitoring

    seen: dict[str, dict[str, Any]] = {}
    leaked: list[Any] = []

    def _probe(loop: str):
        def run(*_a: Any, **_k: Any) -> None:
            leaked.append(llm._FAILOVER_EVENTS.get())
            seen[loop] = _context()
            llm._FAILOVER_EVENTS.set([{"from": "openai", "to": "anthropic", "reason": "probe"}])
        return run

    for name in monitoring.KNOWN_LOOPS:
        module = getattr(monitoring, name)
        for attr in ("run_once", "tick"):
            if callable(getattr(module, attr, None)):
                monkeypatch.setattr(module, attr, _probe(name))

    class FakeScheduler:
        def __init__(self) -> None:
            self.jobs: list[tuple[Any, dict[str, Any]]] = []

        def add_job(self, fn, trigger, **kw):
            self.jobs.append((fn, kw))

    sched = FakeScheduler()
    monitoring.register_all(sched)
    ids = [kw["id"] for _fn, kw in sched.jobs]
    assert set(ids) == set(monitoring.KNOWN_LOOPS) and len(ids) == len(monitoring.KNOWN_LOOPS)

    # One thread context, jobs back to back: the way a reused pool thread
    # runs them.
    thread_ctx = contextvars.Context()
    for fn, kw in sched.jobs:
        # The job keeps its loop's identity (KNOWN_LOOPS pins, job reprs).
        assert fn.__wrapped__.__qualname__.endswith("run")
        thread_ctx.run(fn, **kw.get("kwargs", {}))

    assert set(seen) == set(monitoring.KNOWN_LOOPS)
    for loop, ctx in seen.items():
        assert ctx["origin"] == f"loop:{loop}", loop
        # An umbrella names the origin only.
        assert ctx["agent_name"] in (None, "unknown"), loop
    assert leaked == [None] * len(leaked), "a job saw the previous job's context"
    # And the caller's own context is untouched by the jobs it ran.
    assert llm._FAILOVER_EVENTS.get() is None


def test_scheduler_proxy_passes_everything_else_through():
    import app.monitoring as monitoring

    class Sched:
        running = True

        def add_job(self, fn, *args, **kw):
            return ("added", fn, args, kw)

    proxy = monitoring._OriginScheduler(Sched())
    assert proxy.running is True
    tag, fn, args, kw = proxy.add_job(lambda: None, "interval", minutes=5, id="x_loop")
    assert tag == "added" and args == ("interval",) and kw == {"minutes": 5, "id": "x_loop"}
    assert fn() is None


# ---------------------------------------------------------------------------
# API: the route-template origin
# ---------------------------------------------------------------------------

def test_middleware_origin_is_route_template_on_sync_endpoint(monkeypatch):
    """The route TEMPLATE reaches a SYNC endpoint's thread (FastAPI runs it
    in a pool thread with a copied context — asserted, not assumed). Never
    the raw path: ids overflow `llm_call_logs.origin` on Postgres and could
    carry user-supplied text into logs (critique #6). Wired as an app-level
    dependency so it runs after routing (see `main.llm_request_origin`)."""
    from fastapi.testclient import TestClient

    from app.api import routes_stocks
    from app.main import app, llm_request_origin

    assert not inspect.iscoroutinefunction(routes_stocks.get_stock)
    assert any(d.dependency is llm_request_origin for d in app.router.dependencies)
    seen: list[dict[str, Any]] = []

    def probe(ticker: str) -> dict[str, Any]:
        seen.append(_context())
        return {}

    monkeypatch.setattr(routes_stocks, "get_full_financials", probe)
    client = TestClient(app)
    client.get("/api/stocks/zzorigin1")
    assert seen, "the probe endpoint was not reached"
    assert seen[0]["origin"] == "api:/api/stocks/{ticker}"
    assert "zzorigin1" not in str(seen[0]["origin"]).lower()
    assert seen[0]["agent_name"] in (None, "unknown")
    # Nothing leaks back into the test's context.
    assert _context()["origin"] is None


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------

def test_regen_worker_context_fields(monkeypatch):
    """A regen job's context is open from the claim onward (so the ticker
    introduction and the fundamentals pull are covered) and carries origin,
    job, ticker and run — never an agent."""
    from app.database import SessionLocal, init_db
    from app.models import RegenJob
    from app.services import regen_worker

    init_db()
    seen: dict[str, dict[str, Any]] = {}

    def fake_run(ticker, *, scenario="soft_landing", force_refresh=False, run_id=None, **kw):
        seen["memo"] = _context()
        from app.services.regen_lease import current_claim
        claim = current_claim()
        with SessionLocal() as db:
            db.get(RegenJob, claim.job_id).memo_version = 3
            db.commit()
        return SimpleNamespace(ticker=ticker, rating_label="Neutral")

    monkeypatch.setattr(regen_worker, "run_stock_memo", fake_run)
    monkeypatch.setattr(regen_worker, "_introduce_ticker",
                        lambda job_id, ticker: seen.setdefault("introduce", _context()))
    monkeypatch.setattr(regen_worker, "_pull_fundamentals_through",
                        lambda job_id, ticker: seen.setdefault("pull", _context()))
    with SessionLocal() as db:
        db.query(RegenJob).delete()
        db.commit()
    try:
        job, _ = regen_worker.enqueue("ZZORG")
        done = regen_worker.process_next_job()
        assert done and done["status"] == "succeeded"
        for step in ("introduce", "pull", "memo"):
            ctx = seen[step]
            assert ctx["origin"] == "worker:regen", step
            assert ctx["job_id"] == f"regen:{job['id']}", step
            assert ctx["ticker"] == "ZZORG" and ctx["run_id"] == job["run_id"], step
            assert ctx["feature"] == "research_run", step
            assert ctx["agent_name"] in (None, "unknown"), step
    finally:
        with SessionLocal() as db:
            db.query(RegenJob).delete()
            db.commit()


def test_worker_umbrella_keeps_an_outer_origin(monkeypatch):
    """A script draining the queue started the work: its origin is kept."""
    from app.database import SessionLocal, init_db
    from app.models import RegenJob
    from app.services import regen_worker

    init_db()
    seen: list[dict[str, Any]] = []

    def fake_run(ticker, *, run_id=None, **kw):
        seen.append(_context())
        from app.services.regen_lease import current_claim
        with SessionLocal() as db:
            db.get(RegenJob, current_claim().job_id).memo_version = 4
            db.commit()
        return SimpleNamespace(ticker=ticker, rating_label="Neutral")

    monkeypatch.setattr(regen_worker, "run_stock_memo", fake_run)
    monkeypatch.setattr(regen_worker, "_introduce_ticker", lambda *a: None)
    monkeypatch.setattr(regen_worker, "_pull_fundamentals_through", lambda *a: None)
    with SessionLocal() as db:
        db.query(RegenJob).delete()
        db.commit()
    try:
        regen_worker.enqueue("ZZORH")
        with llm.llm_call_context(origin="script:audit_unit_costs"):
            regen_worker.process_next_job()
        assert seen[0]["origin"] == "script:audit_unit_costs"
        assert seen[0]["job_id"].startswith("regen:")
    finally:
        with SessionLocal() as db:
            db.query(RegenJob).delete()
            db.commit()


def test_worker_threads_name_their_origin(monkeypatch):
    """The seed and fmp-repull threads run outside the scheduler proxy
    (critique #16)."""
    import threading

    from app.services import fmp_repull_ledger

    seen: list[str | None] = []
    monkeypatch.setattr(fmp_repull_ledger, "_loop", lambda shutdown: seen.append(_context()["origin"]))
    t = threading.Thread(target=fmp_repull_ledger._loop_with_origin, args=(threading.Event(),))
    t.start()
    t.join(5)
    assert seen == ["worker:fmp_repull"]

    src = (BACKEND / "app" / "worker.py").read_text()
    assert 'llm_call_context(origin="worker:seed")' in src
    assert "target=_seed_with_origin" in src


# ---------------------------------------------------------------------------
# Umbrella contexts never name an agent
# ---------------------------------------------------------------------------

# The umbrella sites this slice owns; the repo-wide AGENT_CONTEXT_SITES
# sweep is slice A2z's.
_UMBRELLA_FILES = (
    "app/monitoring/__init__.py",
    "app/main.py",
    "app/worker.py",
    "app/services/regen_worker.py",
    "app/services/industry_report_worker.py",
    "app/services/postmortem_service.py",
    "app/services/fmp_repull_ledger.py",
    "app/scripts/capture_industry_ui_fixture.py",
    "scripts/audit_unit_costs.py",
    "scripts/index_research_notes.py",
    "scripts/postmortem_backfill.py",
    "scripts/corpus_repair.py",
    "scripts/validate_model_access.py",
)


def _agent_context_sites(rel: str) -> list[tuple[str, int]]:
    tree = ast.parse((BACKEND / rel).read_text())
    out: list[tuple[str, int]] = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
            if name == "llm_call_context" and any(k.arg == "agent_name" for k in node.keywords):
                out.append((fn.name, node.lineno))
    return out


def test_umbrella_contexts_never_set_agent():
    """Critique #1: an umbrella that names an agent credits it with every
    specialist call beneath it (audit_unit_costs' "unit_cost_audit" swallowed
    whole memo runs). Only AGENT_CONTEXT_SITES may name one."""
    bad = []
    for rel in _UMBRELLA_FILES:
        for fn_name, line in _agent_context_sites(rel):
            if (rel, fn_name) not in llm_attribution.AGENT_CONTEXT_SITES:
                bad.append(f"{rel}:{line} {fn_name}")
    assert not bad, f"umbrella contexts naming an agent: {bad}"


def test_audit_unit_costs_origin():
    import scripts.audit_unit_costs as audit

    seen: list[dict[str, Any]] = []
    sample = audit._Sample("research_run", 0, "ZZAUD")
    row = sample.run(lambda run_id: seen.append(_context()) or {"status": "ok"})
    assert row["status"] == "ok"
    assert seen[0]["origin"] == "script:audit_unit_costs"
    assert seen[0]["run_id"] == row["run_id"]
    assert seen[0]["agent_name"] in (None, "unknown")


@pytest.mark.parametrize("module,origin", [
    ("scripts.index_research_notes", "script:index_research_notes"),
    ("scripts.postmortem_backfill", "script:postmortem_backfill"),
    ("scripts.corpus_repair", "script:corpus_repair"),
    ("scripts.validate_model_access", "script:validate_model_access"),
    ("app.scripts.capture_industry_ui_fixture", "script:capture_industry_ui_fixture"),
])
def test_scripts_declare_their_origin(module, origin):
    import importlib
    mod = importlib.import_module(module)
    assert mod.ORIGIN == origin
    assert "llm_call_context(origin=ORIGIN)" in inspect.getsource(mod)


def test_validate_model_access_is_allow_listed_with_a_reason():
    import scripts.validate_model_access as vma
    assert "deliberate" in (vma.__doc__ or "") + inspect.getsource(vma)
    assert "no llm_call_logs rows" in vma.DIRECT_PROVIDER_CALLS_REASON


# ---------------------------------------------------------------------------
# Loop-driven agents: postmortem and news impact rows
# ---------------------------------------------------------------------------

def test_postmortem_news_impact_rows_attributed(monkeypatch, tmp_path):
    """Through the real LLM layer with a fake client: the news-impact and
    postmortem rows name their action, registry agent, ticker and (for the
    postmortem) the snapshot being judged — FIX-019's monitoring-loop rows
    read agent "unknown" with no ticker."""
    from datetime import datetime

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app.agents import news_impact_agent
    from app.config import settings
    from app.models import MemoOutcome, MemoOutcomeEligibility, MemoPostmortem, MemoSnapshot
    from app.services import postmortem_service as pm
    from app.tests.test_update_orchestrator import _stub_alert, _stub_memo

    impact = anthropic_response('{"material": false, "patch": {}, "rationales": {}, "delta_summary": ""}')
    lesson = anthropic_response('{"lesson": "We were early.", "agent_attribution": {}, '
                                '"regime_at_memo": "", "sector_lesson": ""}')
    client = FakeClient(impact, lesson)
    llm_fakes.live(monkeypatch, anthropic=client, active="anthropic")

    run_id = "origins-news-pm"
    with llm.llm_call_context(run_id=run_id, origin="loop:news_loop"):
        news_impact_agent.assess(_stub_memo("ZZNEWS"), _stub_alert())

    # A postmortem through run_postmortems, on an isolated snapshot table.
    engine = create_engine(f"sqlite:///{tmp_path / 'pm.db'}")
    for model in (MemoSnapshot, MemoOutcome, MemoPostmortem, MemoOutcomeEligibility):
        model.__table__.create(engine)
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(pm, "SessionLocal", sessions)
    monkeypatch.setattr(settings, "enable_long_term_memory", False)
    with sessions() as db:
        snap = MemoSnapshot(ticker="ZZPM", version=1, generated_at=datetime(2026, 1, 1),
                            memo_json={"ticker": "ZZPM", "rating_label": "Bullish",
                                       "generation_mode": "live", "one_sentence_thesis": "t"})
        db.add(snap)
        db.flush()
        db.add(MemoOutcome(memo_snapshot_id=snap.id, ticker="ZZPM", horizon_days=90,
                           forward_return=.2, benchmark_return=.05, alpha=.15))
        db.commit()
        snap_id = snap.id
    with llm.llm_call_context(run_id=run_id, origin="loop:postmortem_loop"):
        report = pm.run_postmortems(horizon_days=90, limit=1)
    assert report["written"] == 1

    rows = {r.action: r for r in llm_fakes.rows_for(run_id)}
    news = rows["news.impact"]
    assert (news.agent_name, news.role, news.ticker, news.origin) == (
        "News Impact", "news_impact", "ZZNEWS", "loop:news_loop")
    post = rows["postmortem.review"]
    assert (post.agent_name, post.role, post.ticker, post.origin) == (
        "Postmortem", "postmortem", "ZZPM", "loop:postmortem_loop")
    assert post.job_id == f"snapshot:{snap_id}"
    engine.dispose()
