"""Retention GC for `research_snapshots`.

`cache_put` is insert-only, and until 2026-09-10 nothing ever deleted from
this table, so it grew ~8k rows/day and eventually made the lineage scan
large enough to OOM-kill the worker. The scan is bounded now; this keeps
the table bounded so it cannot grow into the next unbounded thing.

The property that matters most here is the conservative one: deleting a row
that is still serving reads turns a cache hit into a recompute, and for the
LLM-backed kinds that costs real money. So the newest row per
`(subject, kind)` is never deleted, whatever its age.

These tests backdate only their own rows past the retention cutoff, so a GC
pass leaves every other test's (freshly written) rows alone.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta

from sqlalchemy import event

from app.cache import cache_put, gc_snapshots
from app.cache.snapshots import SNAPSHOT_RETENTION_DAYS, ResearchSnapshot
from app.database import SessionLocal, engine


def _unique(label: str) -> str:
    return f"{label}::{datetime.utcnow().isoformat()}::{time.perf_counter_ns()}"


def _write(subject: str, kind: str, *, age_days: float, payload=None) -> int:
    """Write a snapshot and backdate it to `age_days` ago."""
    snap = cache_put(subject, kind, payload=payload or {"v": 1}, generated_by="test")
    with SessionLocal() as db:
        row = db.get(ResearchSnapshot, snap.id)
        row.generated_at = datetime.utcnow() - timedelta(days=age_days)
        db.commit()
    return snap.id


def _alive(ids: list[int]) -> set:
    with SessionLocal() as db:
        return {
            row_id for row_id in ids
            if db.get(ResearchSnapshot, row_id) is not None
        }


OLD = SNAPSHOT_RETENTION_DAYS + 10
RECENT = 1.0


def test_newest_row_per_key_survives_however_old_it_is():
    """A key written once and never again keeps its row forever — deleting
    it would turn the next read into a recompute."""
    subject = _unique("solo")
    only = _write(subject, "company_cold", age_days=OLD * 10)
    gc_snapshots()
    assert _alive([only]) == {only}


def test_superseded_rows_past_retention_are_deleted():
    subject = _unique("superseded")
    old_ids = [_write(subject, "news_hot", age_days=OLD + i) for i in range(3)]
    newest = _write(subject, "news_hot", age_days=OLD - 1 if OLD > 1 else 0)
    stats = gc_snapshots()
    assert stats["deleted"] >= 3
    assert _alive(old_ids) == set(), "superseded and past retention"
    assert _alive([newest]) == {newest}, "newest per key is kept"


def test_superseded_rows_inside_retention_are_kept():
    """Recent history is still useful; retention is not 'keep exactly one'."""
    subject = _unique("recent")
    older = _write(subject, "sector_warm", age_days=RECENT + 1)
    newest = _write(subject, "sector_warm", age_days=RECENT)
    gc_snapshots()
    assert _alive([older, newest]) == {older, newest}


def test_kinds_are_independent_keys():
    """Same subject, two kinds: each keeps its own newest row."""
    subject = _unique("twokinds")
    cold_old = _write(subject, "company_cold", age_days=OLD + 5)
    cold_new = _write(subject, "company_cold", age_days=OLD + 1)
    warm_only = _write(subject, "sector_warm", age_days=OLD + 5)
    gc_snapshots()
    assert _alive([cold_old]) == set()
    assert _alive([cold_new, warm_only]) == {cold_new, warm_only}


def test_keep_per_key_retains_more_than_one_when_asked():
    subject = _unique("keep2")
    ids = [_write(subject, "warm", age_days=OLD + (5 - i)) for i in range(5)]
    gc_snapshots(keep_per_key=2)
    survivors = _alive(ids)
    assert len(survivors) == 2
    assert survivors == set(ids[-2:]), "the two newest, by generated_at"


def test_max_delete_caps_the_pass_and_reports_it():
    """A first run against a huge table must be bounded work, not a stall."""
    subject = _unique("capped")
    ids = [_write(subject, "warm", age_days=OLD + (10 - i)) for i in range(6)]
    stats = gc_snapshots(max_delete=2)
    assert stats["capped"] == 1
    assert stats["deleted"] == 2
    assert len(_alive(ids)) == 4, "the rest wait for the next pass"


def test_gc_is_idempotent_and_reports_nothing_on_a_clean_table():
    subject = _unique("idem")
    ids = [_write(subject, "warm", age_days=OLD + i) for i in range(3)]
    first = gc_snapshots()
    assert first["deleted"] >= 2
    before = _alive(ids)
    second = gc_snapshots()
    assert _alive(ids) == before, "second pass changes nothing"
    assert second["capped"] == 0


def test_gc_never_reads_the_payload_column():
    """The reaper that exists to prevent an OOM must not cause one."""
    subject = _unique("nopayload")
    _write(subject, "warm", age_days=OLD, payload={"blob": "x" * 5000})
    _write(subject, "warm", age_days=RECENT, payload={"blob": "y" * 5000})

    seen: list[str] = []

    def before(conn, cursor, statement, params, context, executemany):
        if "research_snapshots" in statement and statement.upper().lstrip().startswith("SELECT"):
            seen.append(" ".join(statement.split()))

    event.listen(engine, "before_cursor_execute", before)
    try:
        gc_snapshots()
    finally:
        event.remove(engine, "before_cursor_execute", before)

    assert seen, "expected the GC to scan"
    assert all("payload" not in s.lower() for s in seen), seen


# ---------------------------------------------------------------------------
# Scheduler wiring
# ---------------------------------------------------------------------------

def test_loop_is_registered_and_known():
    """A GC that silently stops being scheduled is how the table grew the
    first time, so the registration is pinned like every other loop."""
    from app.monitoring import KNOWN_LOOPS, snapshot_gc
    assert "snapshot_gc" in KNOWN_LOOPS

    class _Sched:
        def __init__(self): self.jobs = []
        def add_job(self, fn, trigger, **kw): self.jobs.append((trigger, kw))

    sched = _Sched()
    snapshot_gc.register(sched)
    (trigger, kw), = sched.jobs
    assert trigger == "cron" and kw["id"] == "snapshot_gc"
    # 04:15 sits between checkpoint_gc (04:00) and mispricing_audit (04:30).
    assert (kw["hour"], kw["minute"]) == (4, 15)


def test_loop_records_a_run_with_its_counts(monkeypatch):
    from app.monitoring import snapshot_gc
    recorded = {}
    monkeypatch.setattr(
        snapshot_gc, "record_run",
        lambda name, **kw: recorded.update({"name": name, **kw}),
    )
    monkeypatch.setattr(
        snapshot_gc, "gc_snapshots",
        lambda **kw: {"scanned": 12, "deleted": 3, "capped": 0},
    )
    assert snapshot_gc.run_once()["deleted"] == 3
    assert recorded["name"] == "snapshot_gc" and recorded["success"] is True
    assert "scanned=12" in recorded["note"] and "deleted=3" in recorded["note"]


def test_loop_reports_failure_without_raising(monkeypatch):
    """Never raise into the scheduler; never claim success either."""
    from app.monitoring import snapshot_gc
    recorded = {}
    monkeypatch.setattr(
        snapshot_gc, "record_run",
        lambda name, **kw: recorded.update({"name": name, **kw}),
    )
    def boom(**kw): raise RuntimeError("db gone")
    monkeypatch.setattr(snapshot_gc, "gc_snapshots", boom)
    monkeypatch.setattr(snapshot_gc, "_prune_learning_renders", lambda: {"deleted": 0, "capped": 0})
    assert snapshot_gc.run_once() == {"scanned": 0, "deleted": 0, "capped": 0, "ledger_deleted": 0,
                                      "renders_deleted": 0}
    assert recorded["success"] is False and "RuntimeError" in recorded["note"]


# ---------------------------------------------------------------------------
# W7 (S19): shadow learning_renders retention
# ---------------------------------------------------------------------------

def test_shadow_render_retention(tmp_path, monkeypatch):
    """One audit row per consumer per memo run adds up. Shadow rows past 180
    days are reaped in bounded batches; inject rows (what a memo was actually
    shown) and recent shadow rows are kept. A prune failure is reported, not
    raised, and never stops the snapshot GC."""
    from app.learning import context
    from app.models import LearningRender
    from app.monitoring import snapshot_gc
    from app.tests.learning_helpers import learning_db

    sessions, engine = learning_db(tmp_path, monkeypatch, context)
    now = datetime.utcnow()
    old = now - timedelta(days=context.SHADOW_RENDER_RETENTION_DAYS + 5)
    recent = now - timedelta(days=context.SHADOW_RENDER_RETENTION_DAYS - 5)

    def add(mode: str, at: datetime) -> int:
        with sessions() as s:
            row = LearningRender(run_id="r", consumer="pm_memo", ticker="GCR", mode=mode, chars=10,
                                 items=[], dropped=[], created_at=at)
            s.add(row)
            s.commit()
            return int(row.id)

    def alive() -> set[int]:
        with sessions() as s:
            return {r.id for r in s.query(LearningRender).all()}

    try:
        old_shadow = [add("shadow", old) for _ in range(5)]
        kept = {add("shadow", recent), add("inject", old), add("inject", recent)}

        # Bounded: two batches of two stop with one left, and say so.
        assert context.prune_shadow_renders(batch_size=2, max_batches=2) == {"deleted": 4, "capped": 1}
        assert alive() == kept | {old_shadow[-1]}

        recorded: dict = {}
        monkeypatch.setattr(snapshot_gc, "record_run", lambda name, **kw: recorded.update({"name": name, **kw}))
        monkeypatch.setattr(snapshot_gc, "gc_snapshots",
                            lambda **kw: {"scanned": 0, "deleted": 0, "capped": 0, "ledger_deleted": 0})
        stats = snapshot_gc.run_once()
        assert stats["renders_deleted"] == 1 and alive() == kept
        assert recorded["success"] is True and "renders_deleted=1" in recorded["note"]
        assert snapshot_gc.run_once()["renders_deleted"] == 0          # idempotent

        def broken():
            raise RuntimeError("renders table gone")

        monkeypatch.setattr(snapshot_gc, "_prune_learning_renders", broken)
        assert snapshot_gc.run_once()["renders_deleted"] == 0
        assert recorded["success"] is False and "renders_error=RuntimeError" in recorded["note"]
    finally:
        engine.dispose()


# ---------------------------------------------------------------------------
# Cost ledger
# ---------------------------------------------------------------------------

def _write_cost(subject: str, kind: str, *, age_days: float, tokens: int = 10) -> int:
    from app.cache.snapshots import CacheCostLog
    with SessionLocal() as db:
        row = CacheCostLog(subject=subject, kind=kind, cost_tokens=tokens, note="test")
        db.add(row)
        db.commit()
        row.generated_at = datetime.utcnow() - timedelta(days=age_days)
        db.commit()
        return row.id


def _cost_alive(ids: list[int]) -> set:
    from app.cache.snapshots import CacheCostLog
    with SessionLocal() as db:
        return {i for i in ids if db.get(CacheCostLog, i) is not None}


def test_cost_ledger_rows_past_their_retention_are_deleted():
    """The ledger grows faster than the snapshots — a cache hit appends a
    row too — so it needs reaping as well, just on a longer horizon."""
    from app.cache.snapshots import LEDGER_RETENTION_DAYS
    subject = _unique("ledger")
    ancient = _write_cost(subject, "company_cold", age_days=LEDGER_RETENTION_DAYS + 5)
    recent = _write_cost(subject, "company_cold", age_days=1)
    stats = gc_snapshots()
    assert stats["ledger_deleted"] >= 1
    assert _cost_alive([ancient]) == set()
    assert _cost_alive([recent]) == {recent}, "inside retention, kept"


def test_cost_ledger_retention_is_far_longer_than_snapshot_retention():
    """A snapshot is re-derivable; a cost record is not, so it is kept
    much longer. Guards against someone equalising the two by accident."""
    from app.cache.snapshots import LEDGER_RETENTION_DAYS, SNAPSHOT_RETENTION_DAYS
    assert LEDGER_RETENTION_DAYS >= SNAPSHOT_RETENTION_DAYS * 4


def test_total_token_cost_sums_in_sql_without_loading_rows():
    """Regression: this used to select every ledger row and sum in Python,
    the same unbounded shape that killed the worker elsewhere."""
    from app.cache import total_token_cost
    subject = _unique("sum")
    _write_cost(subject, "company_cold", age_days=1, tokens=100)
    _write_cost(subject, "company_cold:hit", age_days=1, tokens=7)

    seen: list[str] = []

    def before(conn, cursor, statement, params, context, executemany):
        if "cache_cost_logs" in statement and statement.upper().lstrip().startswith("SELECT"):
            seen.append(" ".join(statement.split()))

    event.listen(engine, "before_cursor_execute", before)
    try:
        total = total_token_cost(since=datetime.utcnow() - timedelta(days=2))
    finally:
        event.remove(engine, "before_cursor_execute", before)

    assert total >= 100
    assert seen, "expected a query"
    assert any("sum(" in s.lower() for s in seen), seen
    assert all("note" not in s.lower() for s in seen), "row bodies must not be selected"


def test_total_token_cost_still_excludes_hits_and_honours_since():
    from app.cache import total_token_cost
    subject = _unique("sum2")
    _write_cost(subject, "k", age_days=0.5, tokens=50)
    _write_cost(subject, "k:hit", age_days=0.5, tokens=999)
    _write_cost(subject, "k", age_days=30, tokens=777)
    since = datetime.utcnow() - timedelta(days=2)
    assert total_token_cost(since=since) >= 50
    with_hits = total_token_cost(since=since, exclude_hits=False)
    assert with_hits >= total_token_cost(since=since) + 999
