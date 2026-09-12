"""Regression: the lineage cascade must not scan the table per node.

The production worker (512 MB) was SIGKILLed once an hour on 2026-09-10
inside `mark_stale_descendants`. The function ran `select(ResearchSnapshot)`
— every column, `payload` JSON blob included, no LIMIT — *inside* its BFS
loop, and `invalidate` re-entered it once per invalidated row. Since
`research_snapshots` is insert-only and was never reaped, the scan grew
with the table until it no longer fit in memory.

Correctness alone would not have caught this: the old code produced exactly
the right answer, just by reading the whole table to get it. So these tests
assert the *cost* — how many times the table is scanned, and which columns
come back — not only the result.
"""
from __future__ import annotations

import time
from datetime import datetime

import pytest
from sqlalchemy import event

from app.cache import cache_put, invalidate, mark_stale_descendants
from app.cache.snapshots import ResearchSnapshot
from app.database import SessionLocal, engine


def _unique(label: str) -> str:
    return f"{label}::{datetime.utcnow().isoformat()}::{time.perf_counter_ns()}"


class _Capture:
    """Record every SQL statement issued against `research_snapshots`."""

    def __init__(self) -> None:
        self.statements: list[str] = []

    def __enter__(self):
        def before(conn, cursor, statement, params, context, executemany):
            if "research_snapshots" in statement:
                self.statements.append(" ".join(statement.split()))
        self._before = before
        event.listen(engine, "before_cursor_execute", before)
        return self

    def __exit__(self, *exc):
        event.remove(engine, "before_cursor_execute", self._before)
        return False

    @property
    def lineage_scans(self) -> list[str]:
        """SELECTs that read the whole live table (the cascade's own scan)."""
        return [
            s for s in self.statements
            if s.upper().startswith("SELECT") and "parent_snapshot_ids" in s
        ]


def _chain(depth: int) -> list[int]:
    """A parent with `depth` descendants, each the child of the previous."""
    subject = _unique("chain")
    root = cache_put(subject, "company_cold", payload={"n": 0}, generated_by="test")
    ids = [root.id]
    for i in range(depth):
        child = cache_put(
            _unique(f"chain-{i}"), "sector_warm",
            payload={"blob": "x" * 2000, "n": i + 1},
            generated_by="test",
            parent_snapshots=[ids[-1]],
        )
        ids.append(child.id)
    return ids


def _stale(snapshot_id: int) -> bool:
    with SessionLocal() as db:
        return bool(db.get(ResearchSnapshot, snapshot_id).stale)


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("depth", [1, 5, 12])
def test_cascade_scans_the_table_once_whatever_the_depth(depth):
    """One scan per call. The old code ran one per frontier node, so this
    count grew with the lineage and, on a big table, so did the memory."""
    ids = _chain(depth)
    with _Capture() as cap:
        marked = mark_stale_descendants(ids[0])
    assert marked == depth, "every descendant still gets marked"
    assert len(cap.lineage_scans) == 1, (
        f"expected 1 lineage scan for a depth-{depth} chain, "
        f"got {len(cap.lineage_scans)}"
    )


def test_cascade_never_selects_the_payload_column():
    """The blob is what made the scan fatal; the walk never reads it."""
    ids = _chain(3)
    with _Capture() as cap:
        mark_stale_descendants(ids[0])
    scan = cap.lineage_scans[0]
    assert "payload" not in scan.lower(), scan
    assert "sources_used" not in scan.lower(), scan
    assert "parent_snapshot_ids" in scan


def test_invalidate_cascades_once_for_all_matched_rows():
    """`invalidate` used to re-enter the cascade per row, so K live rows for
    a subject meant K full scans. Now the seed set is walked together."""
    subject = _unique("multi")
    parents = [
        cache_put(subject, "sector_warm", payload={"i": i}, generated_by="test")
        for i in range(6)
    ]
    for p in parents:
        cache_put(
            _unique("child"), "company_warm:dcf",
            payload={"blob": "y" * 500}, generated_by="test",
            parent_snapshots=[p.id],
        )
    with _Capture() as cap:
        count = invalidate(subject, kind="sector_warm")
    assert count == 6
    assert len(cap.lineage_scans) == 1, (
        f"6 invalidated rows should share one cascade, got {len(cap.lineage_scans)}"
    )


def test_invalidate_does_not_load_payloads_of_the_rows_it_invalidates():
    subject = _unique("nopayload")
    cache_put(subject, "news_hot", payload={"blob": "z" * 4000}, generated_by="test")
    with _Capture() as cap:
        invalidate(subject, kind="news_hot")
    selects = [s for s in cap.statements if s.upper().startswith("SELECT")]
    assert selects, "expected at least one select"
    assert all("payload" not in s.lower() for s in selects), selects


# ---------------------------------------------------------------------------
# Correctness (unchanged behaviour)
# ---------------------------------------------------------------------------

def test_deep_chain_is_fully_marked_stale():
    ids = _chain(8)
    assert mark_stale_descendants(ids[0]) == 8
    assert not _stale(ids[0]), "the seed itself is not a descendant of itself"
    assert all(_stale(i) for i in ids[1:])


def test_already_stale_rows_are_not_recounted():
    ids = _chain(4)
    assert mark_stale_descendants(ids[0]) == 4
    assert mark_stale_descendants(ids[0]) == 0, "idempotent"


def test_a_cycle_in_lineage_terminates():
    """Lineage is written by callers, so it is not guaranteed acyclic."""
    a = cache_put(_unique("cyc-a"), "warm", payload={}, generated_by="test")
    b = cache_put(_unique("cyc-b"), "warm", payload={}, generated_by="test",
                  parent_snapshots=[a.id])
    with SessionLocal() as db:
        row = db.get(ResearchSnapshot, a.id)
        row.parent_snapshot_ids = [b.id]      # a <- b <- a
        db.commit()
    assert mark_stale_descendants(a.id) == 1  # b only; the walk stops
    assert _stale(b.id)


def test_many_seeds_are_accepted_and_deduplicated():
    ids = _chain(3)
    # Passing the same seed twice must not double-count its descendants.
    assert mark_stale_descendants([ids[0], ids[0]]) == 3


def test_empty_seed_set_is_a_noop():
    with _Capture() as cap:
        assert mark_stale_descendants([]) == 0
    assert cap.lineage_scans == []


def test_non_integer_lineage_entries_are_skipped_not_fatal():
    """A hand-written or migrated payload can hold junk; it must not raise."""
    parent = cache_put(_unique("junk-p"), "warm", payload={}, generated_by="test")
    child = cache_put(_unique("junk-c"), "warm", payload={}, generated_by="test")
    with SessionLocal() as db:
        row = db.get(ResearchSnapshot, child.id)
        row.parent_snapshot_ids = ["not-an-id", None, parent.id]
        db.commit()
    assert mark_stale_descendants(parent.id) == 1
    assert _stale(child.id)
