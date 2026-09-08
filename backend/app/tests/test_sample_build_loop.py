"""`monitoring/sample_build_loop` (FEAT-002, S3).

The only writer of `public_samples`. Runs on the worker: weekly and on
an admin trigger that arrives through a control row (the two processes
share nothing but the database). Bounded to five tickers, sequential;
a failing kind keeps its last good row; no LLM configured → no
commentary call; `record_run` carries `built=` / `degraded=` counts.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from app.config import settings
from app.database import SessionLocal
from app.finance.dcf import run_dcf
from app.models import CronLoopRun, MemoSnapshot, PublicSample
from app.monitoring import sample_build_loop
from app.schemas import DCFAssumptions, DCFResult, MispricingThesis
from app.services import dcf_store, memo_store, public_samples
from app.tests.auth_helpers import ClerkStub, enable_auth
from app.tests.factories import make_memo

LISTED = ["ZZL1", "ZZL2", "ZZL3"]


@pytest.fixture()
def clerk():
    return ClerkStub()


@pytest.fixture(params=["wall_off", "wall_on"])
def wall(request, monkeypatch, clerk):
    """The loop is auth-agnostic; running both ways pins that it stays so."""
    if request.param == "wall_on":
        yield from enable_auth(monkeypatch, clerk)
    else:
        monkeypatch.setattr(settings, "auth_enabled", False)
        yield None


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.setattr(settings, "sample_tickers", ",".join(LISTED))
    import app.monitoring as monitoring
    monitoring._LAST_RUNS.pop(sample_build_loop.LOOP_NAME, None)
    _purge()
    yield
    monitoring._LAST_RUNS.pop(sample_build_loop.LOOP_NAME, None)
    _purge()


def _purge() -> None:
    with SessionLocal() as db:
        db.query(PublicSample).filter(
            PublicSample.ticker.in_(LISTED + [public_samples.CONTROL_TICKER])
        ).delete(synchronize_session=False)
        db.query(MemoSnapshot).filter(MemoSnapshot.ticker.in_(LISTED)).delete(synchronize_session=False)
        db.query(CronLoopRun).filter(CronLoopRun.loop_name == sample_build_loop.LOOP_NAME).delete(synchronize_session=False)
        db.commit()
    from app.models import DCFModel
    with SessionLocal() as db:
        db.query(DCFModel).filter(DCFModel.ticker.in_(LISTED)).delete(synchronize_session=False)
        db.commit()


@pytest.fixture()
def stub_providers(monkeypatch):
    """Comps and prices go through the provider chain; stub them so the
    loop tests are about the loop. Commentary is exercised separately."""
    monkeypatch.setattr(public_samples, "_build_comps", lambda t: ({"target": {"ticker": t}, "peers": []}, "stub", []))
    monkeypatch.setattr(public_samples, "_build_prices", lambda t: ({"points": [{"date": "2026-09-05", "close": 10.0}]}, "stub", []))


@pytest.fixture()
def recorded(monkeypatch):
    calls: list[tuple[tuple, dict]] = []
    real = sample_build_loop.record_run

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        real(*args, **kwargs)

    monkeypatch.setattr(sample_build_loop, "record_run", spy)
    return calls


def _seed_memo(ticker: str):
    return memo_store.save_memo(make_memo(
        ticker=ticker, company_name=f"{ticker} Corp",
        mispricing_thesis=MispricingThesis(consensus_view="c", our_view="o", gap="g"),
        price_at_memo=42.0,
    ))


def _seed_dcf(ticker: str):
    a = DCFAssumptions(base_revenue=1000.0, net_debt=0.0, diluted_shares=100.0, current_price=42.0)
    result = DCFResult(
        ticker=ticker, current_price=42.0,
        base=run_dcf(a), bull=run_dcf(a, scenario_name="bull", label="Bull"),
        bear=run_dcf(a, scenario_name="bear", label="Bear"),
    )
    return dcf_store.save_version(ticker, assumptions=a, dcf_result=result, trigger="initial")


def _rows(ticker: str) -> dict[str, PublicSample]:
    with SessionLocal() as db:
        rows = db.query(PublicSample).filter(PublicSample.ticker == ticker).all()
        db.expunge_all()
        return {r.kind: r for r in rows}


# ---------------------------------------------------------------------------
# run_once
# ---------------------------------------------------------------------------

def test_run_once_builds_from_stored_memo_and_dcf_and_records_counts(wall, stub_providers, recorded):
    _seed_memo(LISTED[0])
    _seed_dcf(LISTED[0])
    assert not settings.has_llm, "CI has no LLM keys; the commentary call must be skipped"

    out = sample_build_loop.run_once()
    assert out["success"] is True
    assert out["tickers"] == LISTED and out["ok"] == LISTED and out["failed"] == []

    rows = _rows(LISTED[0])
    assert {"memo", "dcf", "comps", "prices"} <= set(rows)
    assert "commentary" not in rows, "no LLM configured → no commentary row"
    assert rows["memo"].payload["company_name"] == f"{LISTED[0]} Corp"
    assert rows["memo"].source_ref.startswith("memo_snapshot:")
    assert rows["dcf"].payload["base"]["implied_share_price"] is not None
    assert rows["dcf"].source_ref.startswith("dcf_model:")
    assert rows["memo"].built_by == "worker" and rows["memo"].etag
    # Tickers with nothing stored get no rows and a degraded note each.
    assert _rows(LISTED[1]) == {} or set(_rows(LISTED[1])) <= {"comps", "prices"}
    assert any(n.startswith(f"{LISTED[1]} memo: no stored memo") for n in out["degraded"])
    assert any("commentary: skipped (no LLM configured)" in n for n in out["degraded"])

    (args, kwargs), = recorded
    assert args == (sample_build_loop.LOOP_NAME,)
    assert kwargs["success"] is True
    assert f"built={out['built']}" in kwargs["note"] and f"degraded={len(out['degraded'])}" in kwargs["note"]
    assert out["built"] >= 4
    with SessionLocal() as db:
        row = db.query(CronLoopRun).filter(CronLoopRun.loop_name == sample_build_loop.LOOP_NAME).one()
    assert "built=" in row.note and "degraded=" in row.note


def test_run_once_keeps_the_last_good_row_when_a_kind_fails(stub_providers, monkeypatch, recorded):
    _seed_memo(LISTED[0])
    old = datetime.utcnow() - timedelta(days=2)
    sample_build_loop.run_once([LISTED[0]], now=old)
    before = _rows(LISTED[0])["memo"]

    def boom(ticker, db):
        raise RuntimeError("memo store exploded")

    monkeypatch.setattr(public_samples, "_build_memo", boom)
    out = sample_build_loop.run_once([LISTED[0]])
    assert out["success"] is True, "one failing kind is degradation, not a failed run"
    after = _rows(LISTED[0])["memo"]
    assert after.etag == before.etag and after.built_at == before.built_at
    assert any("memo: build failed (RuntimeError); previous row kept" in n for n in out["degraded"])
    # The kinds that did not fail were refreshed.
    assert _rows(LISTED[0])["prices"].built_at > before.built_at


def test_run_once_survives_a_ticker_that_fails_outright(monkeypatch, recorded):
    def explode(ticker, **kw):
        if ticker == LISTED[1]:
            raise RuntimeError("worker hiccup")
        return {"ticker": ticker, "built": ["memo"], "degraded": []}

    monkeypatch.setattr(public_samples, "build_for_ticker", explode)
    out = sample_build_loop.run_once()
    assert out["ok"] == [LISTED[0], LISTED[2]] and out["failed"] == [LISTED[1]]
    assert out["success"] is True and out["built"] == 2
    (_args, kwargs), = recorded
    assert f"tickers_failed={LISTED[1]}" in kwargs["note"]
    assert "RuntimeError" in " ".join(out["degraded"])


def test_run_once_is_a_failure_when_nothing_could_be_built(monkeypatch, recorded):
    def explode(ticker, **kw):
        raise RuntimeError("everything is down")

    monkeypatch.setattr(public_samples, "build_for_ticker", explode)
    out = sample_build_loop.run_once()
    assert out["success"] is False and out["ok"] == []
    (_args, kwargs), = recorded
    assert kwargs["success"] is False


def test_run_once_is_bounded_to_five_tickers_sequentially(monkeypatch, recorded):
    many = [f"ZZM{i}" for i in range(8)]
    monkeypatch.setattr(settings, "sample_tickers", ",".join(many))
    seen: list[str] = []

    def record(ticker, **kw):
        seen.append(ticker)
        return {"ticker": ticker, "built": [], "degraded": []}

    monkeypatch.setattr(public_samples, "build_for_ticker", record)
    out = sample_build_loop.run_once()
    assert seen == many[:sample_build_loop.MAX_TICKERS] == out["tickers"]
    assert out["success"] is True


def test_run_once_ignores_tickers_outside_the_allowlist(monkeypatch, recorded):
    seen: list[str] = []
    monkeypatch.setattr(public_samples, "build_for_ticker", lambda t, **kw: seen.append(t) or {"ticker": t, "built": [], "degraded": []})
    out = sample_build_loop.run_once(["zzl2", "NVDA"])
    assert seen == [LISTED[1]] and out["not_listed"] == ["NVDA"]
    (_args, kwargs), = recorded
    assert "not_listed=NVDA" in kwargs["note"]


# ---------------------------------------------------------------------------
# Commentary — optional LLM call
# ---------------------------------------------------------------------------

def test_commentary_is_written_when_an_llm_is_configured(stub_providers, monkeypatch):
    _seed_memo(LISTED[0])
    monkeypatch.setattr(settings, "anthropic_api_key", "stub-key-for-routing-only")
    assert settings.has_llm
    from app.agents import llm
    prompts: list[str] = []

    def fake_chat_text(prompt, **kw):
        prompts.append(prompt)
        return "  A neutral paragraph about the research.  "

    monkeypatch.setattr(llm, "chat_text", fake_chat_text)
    out = public_samples.build_for_ticker(LISTED[0])
    assert "commentary" in out["built"]
    row = _rows(LISTED[0])["commentary"]
    assert row.payload["text"] == "A neutral paragraph about the research."
    assert row.payload["model"] == "anthropic:cheap" and row.payload["generated_at"]
    assert LISTED[0] in prompts[0] and "No recommendations" in prompts[0]


def test_commentary_failure_is_a_degraded_note_not_an_exception(stub_providers, monkeypatch):
    _seed_memo(LISTED[0])
    monkeypatch.setattr(settings, "anthropic_api_key", "stub-key-for-routing-only")
    from app.agents import llm

    def boom(*a, **k):
        raise TimeoutError("provider slow")

    monkeypatch.setattr(llm, "chat_text", boom)
    out = public_samples.build_for_ticker(LISTED[0])
    assert "commentary" not in out["built"]
    assert "commentary: LLM call failed (TimeoutError)" in out["degraded"]
    assert "memo" in out["built"]


def test_no_llm_means_no_llm_call_at_all(stub_providers, monkeypatch):
    _seed_memo(LISTED[0])
    from app.agents import llm

    def boom(*a, **k):
        raise AssertionError("chat_text must not be called without a configured LLM")

    monkeypatch.setattr(llm, "chat_text", boom)
    monkeypatch.setattr(llm, "chat_json", boom)
    out = public_samples.build_for_ticker(LISTED[0])
    assert "commentary: skipped (no LLM configured)" in out["degraded"]


# ---------------------------------------------------------------------------
# tick — admin trigger + weekly cadence
# ---------------------------------------------------------------------------

def test_tick_serves_a_pending_admin_request_then_clears_it(monkeypatch, recorded):
    with SessionLocal() as db:
        req = public_samples.request_rebuild(db, [LISTED[2]])
    seen: list[list[str]] = []
    monkeypatch.setattr(sample_build_loop, "run_once", lambda tickers=None, now=None: seen.append(tickers) or {"ok": True})

    assert sample_build_loop.tick() == {"ok": True}
    assert seen == [[LISTED[2]]]
    with SessionLocal() as db:
        assert public_samples.pending_request(db) is None
    assert req["queued"] is True


def test_tick_keeps_a_request_that_arrived_mid_build(monkeypatch):
    with SessionLocal() as db:
        public_samples.request_rebuild(db, [LISTED[0]], now=datetime(2026, 9, 1))

    def run(tickers=None, now=None):
        # An operator clicks rebuild while this build is running.
        with SessionLocal() as db:
            public_samples.request_rebuild(db, [LISTED[1]], now=datetime(2026, 9, 2))
        return {}

    monkeypatch.setattr(sample_build_loop, "run_once", run)
    sample_build_loop.tick()
    with SessionLocal() as db:
        pending = public_samples.pending_request(db)
    assert pending is not None and pending["tickers"] == [LISTED[0], LISTED[1]]


def test_tick_builds_when_the_weekly_run_is_due_and_idles_otherwise(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(sample_build_loop, "run_once", lambda tickers=None, now=None: calls.append("ran") or {})

    # Never recorded → due.
    assert sample_build_loop.weekly_due()
    sample_build_loop.tick()
    assert calls == ["ran"]

    import app.monitoring as monitoring
    monitoring.record_run(sample_build_loop.LOOP_NAME, note="built=3 degraded=0")
    assert not sample_build_loop.weekly_due()
    assert sample_build_loop.tick() is None and calls == ["ran"]

    # Six days later: still not due; eight days later: due (cross-process:
    # the anchor is the DB row, so drop the in-memory copy first).
    monitoring._LAST_RUNS.pop(sample_build_loop.LOOP_NAME, None)
    with SessionLocal() as db:
        row = db.query(CronLoopRun).filter(CronLoopRun.loop_name == sample_build_loop.LOOP_NAME).one()
        row.last_run_at = datetime.utcnow() - timedelta(days=6)
        db.commit()
    assert sample_build_loop.tick() is None
    with SessionLocal() as db:
        row = db.query(CronLoopRun).filter(CronLoopRun.loop_name == sample_build_loop.LOOP_NAME).one()
        row.last_run_at = datetime.utcnow() - timedelta(days=8)
        db.commit()
    sample_build_loop.tick()
    assert calls == ["ran", "ran"]


def test_tick_records_a_crash_before_reraising(monkeypatch, recorded):
    def crash(tickers=None, now=None):
        raise RuntimeError("synthetic sample build failure")

    monkeypatch.setattr(sample_build_loop, "run_once", crash)
    with pytest.raises(RuntimeError, match="synthetic sample build failure"):
        sample_build_loop.tick()
    (args, kwargs), = recorded
    assert args == (sample_build_loop.LOOP_NAME,) and kwargs["success"] is False
    assert "RuntimeError" in kwargs["note"]


# ---------------------------------------------------------------------------
# Registration + cron-health
# ---------------------------------------------------------------------------

def test_loop_is_registered_under_its_known_name():
    from app.monitoring import KNOWN_LOOPS, register_all

    assert sample_build_loop.LOOP_NAME in KNOWN_LOOPS

    class FakeScheduler:
        def __init__(self):
            self.jobs = []

        def add_job(self, fn, trigger, **kw):
            self.jobs.append((fn, trigger, kw))

    sched = FakeScheduler()
    register_all(sched)
    mine = [j for j in sched.jobs if j[2].get("id") == sample_build_loop.LOOP_NAME]
    assert len(mine) == 1, "exactly one job id, or cron-health expects a loop that never reports"
    fn, trigger, kw = mine[0]
    assert fn is sample_build_loop.tick and trigger == "interval"
    assert kw["minutes"] == sample_build_loop.POLL_MINUTES


def test_cron_health_treats_the_loop_as_weekly():
    """A three-day-old build is fine; a nine-day-old one is stale."""
    import app.monitoring as monitoring
    from app.api.routes_admin import cron_health_endpoint

    monitoring._LAST_RUNS.pop(sample_build_loop.LOOP_NAME, None)
    for age_days, stale in ((3, False), (9, True)):
        with SessionLocal() as db:
            db.query(CronLoopRun).filter(CronLoopRun.loop_name == sample_build_loop.LOOP_NAME).delete(synchronize_session=False)
            db.add(CronLoopRun(
                loop_name=sample_build_loop.LOOP_NAME, last_run_at=datetime.utcnow() - timedelta(days=age_days),
                success=True, note="built=3 degraded=0", reported_by="worker",
            ))
            db.commit()
        row = next(r for r in cron_health_endpoint()["loops"] if r["loop"] == sample_build_loop.LOOP_NAME)
        assert row["stale"] is stale, (age_days, row)
