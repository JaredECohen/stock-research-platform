"""The debate's thread pool writes both advocates' rows (slice B8-D3; plan
P11, design §12.6).

The app's first thread pool: two advocates call the model at once, each in
a copied context, and each call writes an `llm_call_logs` row from its own
thread. On sqlite, concurrent writers can hit lock errors that the LLM
layer swallows, which would silently lose a cost row. So the validation
memos run sequentially, and this ONE test proves the parallel path on its
own sqlite file, isolated from the suite's shared database."""
from __future__ import annotations

import threading
import uuid

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app import database
from app.agents import debate, llm
from app.config import settings
from app.models import LLMCallLog
from app.services import llm_metrics
from app.tests import llm_fakes


class _ThreadRecordingClient(llm_fakes.FakeClient):
    def __init__(self, *responses):
        super().__init__(*responses)
        self.threads: set[str] = set()
        self._barrier = threading.Barrier(2, timeout=10)

    def _next(self, **kwargs):
        self.threads.add(threading.current_thread().name)
        # Both advocates are inside the provider call at the same time.
        self._barrier.wait()
        return super()._next(**kwargs)


def test_parallel_pair_writes_both_rows(monkeypatch, tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'debate-concurrency.db'}",
                           connect_args={"check_same_thread": False}, future=True)
    LLMCallLog.__table__.create(bind=engine, checkfirst=True)
    own = sessionmaker(bind=engine, autocommit=False, autoflush=False, future=True)
    monkeypatch.setattr(database, "SessionLocal", own)
    monkeypatch.setattr(llm_metrics, "SessionLocal", own)

    client = _ThreadRecordingClient(llm_fakes.anthropic_response('{"queries": []}'))
    llm_fakes.live(monkeypatch, anthropic=client, openai=None, active="anthropic")
    monkeypatch.setattr(settings, "debate_provider", "anthropic")
    monkeypatch.setattr(settings, "debate_model", "claude-opus-5-5")
    route = debate.resolve_route()
    assert route is not None

    run_id = f"debate-par-{uuid.uuid4().hex[:8]}"
    reqs = debate._requests("research", route, "SHARED\n", "ACME", "12 months", 8000)
    with llm.llm_call_context(run_id=run_id):
        results = debate.run_pair(reqs, debate.llm_call, parallel=True)

    assert {s: r.out for s, r in results.items()} == {"bull": {"queries": []}, "bear": {"queries": []}}
    assert len(client.threads) == 2 and all(t.startswith("debate") for t in client.threads)
    with own() as db:
        rows = list(db.execute(select(LLMCallLog).where(LLMCallLog.run_id == run_id)).scalars())
    assert sorted((r.agent_name, r.action, r.success) for r in rows) == [
        ("Bear Advocate", "debate.bear_research", True), ("Bull Advocate", "debate.bull_research", True)]
    # The budget reads them back (own DB): both advocates' spend is there.
    spend = llm_metrics.cost_per_run(run_id, agents=debate.ADVOCATE_AGENTS)
    assert spend["n_calls"] == 2 and spend["cost_usd_total"] > 0
